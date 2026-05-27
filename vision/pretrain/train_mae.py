#!/usr/bin/env python3
import copy
import importlib
import json
import logging
import os
import time
from collections import OrderedDict
from contextlib import suppress
from datetime import datetime
from functools import partial

import torch
import yaml

import train as train_base
from timm import utils
from timm.data import create_dataset, create_loader, resolve_data_config
from timm.models import safe_model_name, resume_checkpoint, model_parameters
from timm.optim import create_optimizer_v2, optimizer_kwargs
from timm.scheduler import create_scheduler_v2, scheduler_kwargs
from timm.utils import NativeScaler

from model import mae_vit_base_patch16_224


_logger = logging.getLogger('train_mae')


group = train_base.parser.add_argument_group('MAE parameters')
group.add_argument('--mask-ratio', type=float, default=0.75,
                   help='Patch masking ratio for MAE pretraining.')
group.add_argument('--decoder-embed-dim', type=int, default=512,
                   help='MAE decoder embedding dimension.')
group.add_argument('--decoder-depth', type=int, default=8,
                   help='Number of MAE decoder transformer blocks.')
group.add_argument('--decoder-num-heads', type=int, default=16,
                   help='Number of MAE decoder attention heads.')
group.add_argument('--norm-pix-loss', action='store_true', default=True,
                   help='Normalize pixels within each patch before reconstruction loss.')
group.add_argument('--no-norm-pix-loss', action='store_false', dest='norm_pix_loss',
                   help='Disable per-patch pixel normalization for MAE loss.')
group.add_argument('--max-train-steps', type=int, default=None,
                   help='Stop after this many optimizer updates. Useful for smoke tests.')
group.add_argument('--save-encoder-only', action='store_true', default=True,
                   help='Write an additional checkpoint containing only encoder weights.')
group.add_argument('--no-save-encoder-only', action='store_false', dest='save_encoder_only',
                   help='Disable encoder-only checkpoint export.')


def _parse_args():
    args_config, remaining = train_base.config_parser.parse_known_args()
    if args_config.config:
        with open(args_config.config, 'r') as f:
            cfg = yaml.safe_load(f)
            train_base.parser.set_defaults(**cfg)

    args = train_base.parser.parse_args(remaining)
    args_text = yaml.safe_dump(args.__dict__, default_flow_style=False)
    return args, args_text


def main():
    utils.setup_default_logging()
    args, args_text = _parse_args()

    if args.device_modules:
        for module in args.device_modules:
            importlib.import_module(module)

    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

    args.prefetcher = not args.no_prefetcher
    args.grad_accum_steps = max(1, args.grad_accum_steps)
    device = utils.init_distributed_device(args)
    if args.distributed:
        _logger.info(
            'Training MAE in distributed mode. '
            f'Process {args.rank}, total {args.world_size}, device {args.device}.')
    else:
        _logger.info(f'Training MAE with a single process on 1 device ({args.device}).')

    model_dtype = None
    if args.model_dtype:
        assert args.model_dtype in ('float32', 'float16', 'bfloat16')
        model_dtype = getattr(torch, args.model_dtype)

    amp_dtype = torch.float16
    if args.amp:
        assert model_dtype is None or model_dtype == torch.float32
        assert args.amp_dtype in ('float16', 'bfloat16')
        if args.amp_dtype == 'bfloat16':
            amp_dtype = torch.bfloat16

    utils.random_seed(args.seed, args.rank)
    if args.fuser:
        utils.set_jit_fuser(args.fuser)
    if args.fast_norm:
        train_base.set_fast_norm()

    in_chans = args.in_chans
    if in_chans is None:
        in_chans = args.input_size[0] if args.input_size is not None else 3
    img_size = args.img_size
    if img_size is None:
        img_size = args.input_size[-1] if args.input_size is not None else 224

    model = mae_vit_base_patch16_224(
        img_size=img_size,
        in_chans=in_chans,
        decoder_embed_dim=args.decoder_embed_dim,
        decoder_depth=args.decoder_depth,
        decoder_num_heads=args.decoder_num_heads,
        norm_pix_loss=args.norm_pix_loss,
    )
    if args.grad_checkpointing:
        for block in model.blocks:
            block.grad_checkpointing = True

    if utils.is_primary(args):
        _logger.info(
            f'Model mae_vit_base_patch16_224 created, param count:'
            f'{sum(m.numel() for m in model.parameters())}')

    data_config = resolve_data_config(vars(args), model=None, verbose=utils.is_primary(args))

    model.to(device=device, dtype=model_dtype)
    if args.channels_last:
        model.to(memory_format=torch.channels_last)

    if not args.lr:
        global_batch_size = args.batch_size * args.world_size * args.grad_accum_steps
        batch_ratio = global_batch_size / args.lr_base_size
        if not args.lr_base_scale:
            on = args.opt.lower()
            args.lr_base_scale = 'sqrt' if any(o in on for o in ('ada', 'lamb')) else 'linear'
        if args.lr_base_scale == 'sqrt':
            batch_ratio = batch_ratio ** 0.5
        args.lr = args.lr_base * batch_ratio
        if utils.is_primary(args):
            _logger.info(
                f'Learning rate ({args.lr}) calculated from base learning rate ({args.lr_base}) '
                f'and effective global batch size ({global_batch_size}) with {args.lr_base_scale} scaling.')

    optimizer = create_optimizer_v2(model, **optimizer_kwargs(cfg=args), **args.opt_kwargs)
    if utils.is_primary(args):
        defaults = copy.deepcopy(optimizer.defaults)
        defaults['weight_decay'] = args.weight_decay
        defaults = ', '.join([f'{k}: {v}' for k, v in defaults.items()])
        _logger.info(f'Created {type(optimizer).__name__} ({args.opt}) optimizer: {defaults}')

    amp_autocast = suppress
    loss_scaler = None
    if args.amp:
        amp_autocast = partial(torch.autocast, device_type=device.type, dtype=amp_dtype)
        if device.type == 'cuda' and amp_dtype == torch.float16:
            loss_scaler = NativeScaler(device=device.type)
        if utils.is_primary(args):
            _logger.info('Using native Torch AMP. Training in mixed precision.')
    elif utils.is_primary(args):
        _logger.info(f'AMP not enabled. Training in {model_dtype or torch.float32}.')

    resume_epoch = None
    if args.resume:
        resume_epoch = resume_checkpoint(
            model,
            args.resume,
            optimizer=None if args.no_resume_opt else optimizer,
            loss_scaler=None if args.no_resume_opt else loss_scaler,
            log_info=utils.is_primary(args),
        )

    if args.distributed:
        if utils.is_primary(args):
            _logger.info('Wrapping MAE model with DistributedDataParallel')
        ddp_device_ids = [device.index] if device.type == 'cuda' else None
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=ddp_device_ids,
            broadcast_buffers=not args.no_ddp_bb,
        )

    if args.data and not args.data_dir:
        args.data_dir = args.data
    input_img_mode = args.input_img_mode or ('RGB' if data_config['input_size'][0] == 3 else 'L')
    dataset_train = create_dataset(
        args.dataset,
        root=args.data_dir,
        split=args.train_split,
        is_training=True,
        class_map=args.class_map,
        download=args.dataset_download,
        batch_size=args.batch_size,
        seed=args.seed,
        repeats=args.epoch_repeats,
        input_img_mode=input_img_mode,
        input_key=args.input_key,
        target_key=args.target_key,
        num_samples=args.train_num_samples,
        trust_remote_code=args.dataset_trust_remote_code,
    )

    train_interpolation = args.train_interpolation
    if args.no_aug or not train_interpolation:
        train_interpolation = data_config['interpolation']

    loader_train = create_loader(
        dataset_train,
        input_size=data_config['input_size'],
        batch_size=args.batch_size,
        is_training=True,
        no_aug=args.no_aug,
        re_prob=args.reprob,
        re_mode=args.remode,
        re_count=args.recount,
        re_split=args.resplit,
        train_crop_mode=args.train_crop_mode,
        scale=args.scale,
        ratio=args.ratio,
        hflip=args.hflip,
        vflip=args.vflip,
        color_jitter=args.color_jitter,
        color_jitter_prob=args.color_jitter_prob,
        grayscale_prob=args.grayscale_prob,
        gaussian_blur_prob=args.gaussian_blur_prob,
        auto_augment=args.aa,
        num_aug_repeats=args.aug_repeats,
        num_aug_splits=0,
        interpolation=train_interpolation,
        mean=data_config['mean'],
        std=data_config['std'],
        num_workers=args.workers,
        worker_seeding=args.worker_seeding,
        pin_memory=args.pin_mem,
        img_dtype=model_dtype or torch.float32,
        device=device,
        distributed=args.distributed,
        use_prefetcher=args.prefetcher,
        use_multi_epochs_loader=args.use_multi_epochs_loader,
    )

    output_dir = None
    saver = None
    if utils.is_primary(args):
        exp_name = args.experiment or '-'.join([
            datetime.now().strftime('%Y%m%d-%H%M%S'),
            safe_model_name('mae_vit_base_patch16_224'),
            str(data_config['input_size'][-1]),
        ])
        output_dir = utils.get_outdir(args.output if args.output else './output/train', exp_name)
        saver = utils.CheckpointSaver(
            model=model,
            optimizer=optimizer,
            args=args,
            amp_scaler=loss_scaler,
            checkpoint_dir=output_dir,
            recovery_dir=output_dir,
            decreasing=True,
            max_history=args.checkpoint_hist,
        )
        with open(os.path.join(output_dir, 'args.yaml'), 'w') as f:
            f.write(args_text)

    updates_per_epoch = (len(loader_train) + args.grad_accum_steps - 1) // args.grad_accum_steps
    lr_scheduler, num_epochs = create_scheduler_v2(
        optimizer,
        **scheduler_kwargs(args, decreasing_metric=True),
        updates_per_epoch=updates_per_epoch,
    )
    start_epoch = args.start_epoch if args.start_epoch is not None else (resume_epoch or 0)
    if lr_scheduler is not None and start_epoch > 0:
        if args.sched_on_updates:
            lr_scheduler.step_update(start_epoch * updates_per_epoch)
        else:
            lr_scheduler.step(start_epoch)

    best_metric = None
    best_epoch = None
    results = []
    total_updates = 0
    try:
        for epoch in range(start_epoch, num_epochs):
            if hasattr(dataset_train, 'set_epoch'):
                dataset_train.set_epoch(epoch)
            elif args.distributed and hasattr(loader_train.sampler, 'set_epoch'):
                loader_train.sampler.set_epoch(epoch)

            train_metrics, total_updates, reached_max_steps = train_one_epoch(
                epoch,
                model,
                loader_train,
                optimizer,
                args,
                device=device,
                lr_scheduler=lr_scheduler,
                saver=saver,
                output_dir=output_dir,
                amp_autocast=amp_autocast,
                loss_scaler=loss_scaler,
                model_dtype=model_dtype,
                total_updates=total_updates,
            )

            if output_dir is not None:
                lrs = [param_group['lr'] for param_group in optimizer.param_groups]
                utils.update_summary(
                    epoch,
                    train_metrics,
                    None,
                    filename=os.path.join(output_dir, 'summary.csv'),
                    lr=sum(lrs) / len(lrs),
                    write_header=best_metric is None,
                )

            latest_metric = train_metrics['loss']
            if saver is not None:
                best_metric, best_epoch = saver.save_checkpoint(epoch, metric=latest_metric)
                if args.save_encoder_only:
                    save_encoder_checkpoint(model, args, epoch, output_dir)

            if lr_scheduler is not None:
                lr_scheduler.step(epoch + 1, latest_metric)

            latest_results = {'epoch': epoch, 'train': train_metrics}
            results.append(latest_results)
            if reached_max_steps:
                if utils.is_primary(args):
                    _logger.info(f'Reached max_train_steps={args.max_train_steps}; stopping.')
                break
    except KeyboardInterrupt:
        pass

    if args.distributed:
        torch.distributed.destroy_process_group()

    if best_metric is not None:
        _logger.info('*** Best loss: {0} (epoch {1})'.format(best_metric, best_epoch))
    if utils.is_primary(args):
        print(f'--result\n{json.dumps(results[-10:], indent=4)}')


def train_one_epoch(
        epoch,
        model,
        loader,
        optimizer,
        args,
        device=torch.device('cuda'),
        lr_scheduler=None,
        saver=None,
        output_dir=None,
        amp_autocast=suppress,
        loss_scaler=None,
        model_dtype=None,
        total_updates=0,
):
    second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
    has_no_sync = hasattr(model, 'no_sync')
    update_time_m = utils.AverageMeter()
    data_time_m = utils.AverageMeter()
    losses_m = utils.AverageMeter()

    model.train()

    accum_steps = args.grad_accum_steps
    last_accum_steps = len(loader) % accum_steps
    updates_per_epoch = (len(loader) + accum_steps - 1) // accum_steps
    num_updates = epoch * updates_per_epoch
    last_batch_idx = len(loader) - 1
    last_batch_idx_to_accum = len(loader) - last_accum_steps

    data_start_time = update_start_time = time.time()
    optimizer.zero_grad()
    update_sample_count = 0
    reached_max_steps = False

    for batch_idx, (_key, input, _target) in enumerate(loader):
        last_batch = batch_idx == last_batch_idx
        need_update = last_batch or (batch_idx + 1) % accum_steps == 0
        update_idx = batch_idx // accum_steps
        if batch_idx >= last_batch_idx_to_accum:
            accum_steps = last_accum_steps

        if not args.prefetcher:
            input = input.to(device=device, dtype=model_dtype)
        if args.channels_last:
            input = input.contiguous(memory_format=torch.channels_last)

        data_time_m.update(accum_steps * (time.time() - data_start_time))

        def _forward():
            with amp_autocast():
                loss, _pred, _mask = model(input, mask_ratio=args.mask_ratio)
            if accum_steps > 1:
                loss = loss / accum_steps
            return loss

        def _backward(_loss):
            if loss_scaler is not None:
                loss_scaler(
                    _loss,
                    optimizer,
                    clip_grad=args.clip_grad,
                    clip_mode=args.clip_mode,
                    parameters=model_parameters(model, exclude_head='agc' in args.clip_mode),
                    create_graph=second_order,
                    need_update=need_update,
                )
            else:
                _loss.backward(create_graph=second_order)
                if need_update:
                    if args.clip_grad is not None:
                        utils.dispatch_clip_grad(
                            model_parameters(model, exclude_head='agc' in args.clip_mode),
                            value=args.clip_grad,
                            mode=args.clip_mode,
                        )
                    optimizer.step()

        global_batch_size = batch_size = input.shape[0]
        if args.distributed:
            global_batch_size *= args.world_size

        if has_no_sync and not need_update:
            with model.no_sync():
                loss = _forward()
                _backward(loss)
        else:
            loss = _forward()
            _backward(loss)

        losses_m.update(loss.item() * accum_steps, batch_size)
        update_sample_count += global_batch_size

        if not need_update:
            data_start_time = time.time()
            continue

        num_updates += 1
        total_updates += 1
        optimizer.zero_grad()

        if args.synchronize_step and device.type == 'cuda':
            torch.cuda.synchronize()
        time_now = time.time()
        update_time_m.update(time_now - update_start_time)
        update_start_time = time_now

        if update_idx % args.log_interval == 0 or last_batch:
            lrl = [param_group['lr'] for param_group in optimizer.param_groups]
            lr = sum(lrl) / len(lrl)
            loss_avg, loss_now = losses_m.avg, losses_m.val
            if args.distributed:
                loss_avg = utils.reduce_tensor(loss.new([loss_avg]), args.world_size).item()
                loss_now = utils.reduce_tensor(loss.new([loss_now]), args.world_size).item()
            if utils.is_primary(args):
                _logger.info(
                    f'Train: {epoch} [{update_idx:>4d}/{updates_per_epoch} '
                    f'({100. * (update_idx + 1) / updates_per_epoch:>3.0f}%)]  '
                    f'Updates: {total_updates}  '
                    f'Loss: {loss_now:#.3g} ({loss_avg:#.3g})  '
                    f'Time: {update_time_m.val:.3f}s, {update_sample_count / update_time_m.val:>7.2f}/s  '
                    f'({update_time_m.avg:.3f}s, {update_sample_count / update_time_m.avg:>7.2f}/s)  '
                    f'LR: {lr:.3e}  '
                    f'Data: {data_time_m.val:.3f} ({data_time_m.avg:.3f})'
                )

        if saver is not None and args.recovery_interval and ((update_idx + 1) % args.recovery_interval == 0):
            saver.save_recovery(epoch, batch_idx=update_idx)

        if lr_scheduler is not None:
            lr_scheduler.step_update(num_updates=num_updates, metric=losses_m.avg)

        update_sample_count = 0
        data_start_time = time.time()

        if args.max_train_steps is not None and total_updates >= args.max_train_steps:
            reached_max_steps = True
            break

    if hasattr(optimizer, 'sync_lookahead'):
        optimizer.sync_lookahead()

    loss_avg = losses_m.avg
    if args.distributed:
        loss_avg = torch.tensor([loss_avg], device=device, dtype=torch.float32)
        loss_avg = utils.reduce_tensor(loss_avg, args.world_size).item()
    return OrderedDict([('loss', loss_avg)]), total_updates, reached_max_steps


def save_encoder_checkpoint(model, args, epoch, output_dir):
    unwrapped = utils.unwrap_model(model)
    save_state = {
        'epoch': epoch,
        'arch': 'mae_vit_base_patch16_224_encoder',
        'state_dict': unwrapped.encoder_state_dict(),
        'args': args,
        'version': 2,
    }
    torch.save(save_state, os.path.join(output_dir, 'encoder_last.pth.tar'))


if __name__ == '__main__':
    main()
