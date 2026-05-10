#!/usr/bin/env python3
import argparse
import csv
import json
import logging
import os
from contextlib import suppress
from functools import partial
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from PIL import Image
from sklearn.cluster import KMeans, MiniBatchKMeans
from timm.data import create_transform, resolve_data_config
from timm.models import create_model, load_checkpoint


_LOG = logging.getLogger("cluster_representatives")
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def parse_args():
    parser = argparse.ArgumentParser(description="Cluster representative images for AgriNet classes.")
    parser.add_argument("--data-dir", type=str, required=True, help="Dataset root organized as class/image files.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Checkpoint path.")
    parser.add_argument("--model", type=str, default="vit_base_patch16_224.orig_in21k_ft_in1k", help="timm model.")
    parser.add_argument("--num-classes", type=int, default=1020, help="Model classification head size.")
    parser.add_argument("--clusters-per-class", type=int, default=10, help="Requested clusters per class.")
    parser.add_argument("--batch-size", type=int, default=512, help="Per-process batch size.")
    parser.add_argument("--workers", type=int, default=16, help="DataLoader workers per process.")
    parser.add_argument("--device", type=str, default="cuda", help="Inference device.")
    parser.add_argument("--amp", action="store_true", default=False, help="Enable AMP inference.")
    parser.add_argument("--amp-dtype", type=str, default="float16", choices=("float16", "bfloat16"))
    parser.add_argument("--channels-last", action="store_true", default=False, help="Use channels_last.")
    parser.add_argument("--pin-mem", action="store_true", default=False, help="Pin CPU memory.")
    parser.add_argument("--out-dir", type=str, required=True, help="Output directory.")
    parser.add_argument("--features-dtype", type=str, default="float16", choices=("float16", "float32"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-interval", type=int, default=50)
    parser.add_argument("--max-classes", type=int, default=None, help="Limit number of classes for dry-runs.")
    parser.add_argument("--max-images-per-class", type=int, default=None, help="Limit images per class for dry-runs.")
    parser.add_argument("--skip-existing-features", action="store_true", default=False, help="Skip extraction if shards exist.")
    parser.add_argument("--symlink-dirname", type=str, default="representatives", help="Representative image link dir.")
    return parser.parse_args()


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )


def init_distributed(args):
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    distributed = world_size > 1

    if distributed:
        backend = "nccl" if args.device.startswith("cuda") else "gloo"
        dist.init_process_group(backend=backend)

    if args.device.startswith("cuda"):
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device(args.device)

    return distributed, rank, local_rank, world_size, device


def cleanup_distributed(distributed):
    if distributed and dist.is_initialized():
        dist.destroy_process_group()


def barrier(distributed):
    if distributed and dist.is_initialized():
        dist.barrier()


def seed_everything(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def list_class_dirs(root, max_classes=None):
    class_dirs = []
    with os.scandir(root) as it:
        for entry in it:
            if entry.is_dir():
                class_dirs.append((entry.name, entry.path))
    class_dirs.sort(key=lambda item: item[0])
    if max_classes is not None:
        class_dirs = class_dirs[:max_classes]
    return class_dirs


def list_images(class_dir, max_images=None):
    images = []
    with os.scandir(class_dir) as it:
        for entry in it:
            if entry.is_file() and Path(entry.name).suffix.lower() in IMAGE_EXTENSIONS:
                images.append(entry.path)
    images.sort()
    if max_images is not None:
        images = images[:max_images]
    return images


class RepresentativeDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, transform, max_classes=None, max_images_per_class=None):
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.class_ids = []
        self.class_to_idx = {}
        self.samples = []
        self.class_counts = {}

        class_dirs = list_class_dirs(self.data_dir, max_classes=max_classes)
        for class_idx, (class_id, class_dir) in enumerate(class_dirs):
            self.class_ids.append(class_id)
            self.class_to_idx[class_id] = class_idx
            image_paths = list_images(class_dir, max_images=max_images_per_class)
            self.class_counts[class_id] = len(image_paths)
            for image_path in image_paths:
                self.samples.append((image_path, class_idx))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        image_path, class_idx = self.samples[index]
        with Image.open(image_path) as image:
            image = image.convert("RGB")
            tensor = self.transform(image)
        return image_path, class_idx, tensor


def collate_batch(batch):
    paths, class_indices, images = zip(*batch)
    return list(paths), torch.tensor(class_indices, dtype=torch.int64), torch.stack(images, dim=0)


def build_model(args, device):
    model = create_model(
        args.model,
        pretrained=False,
        num_classes=args.num_classes,
    )
    load_checkpoint(model, args.checkpoint)
    model.eval()
    model.to(device=device)
    if args.channels_last:
        model = model.to(memory_format=torch.channels_last)
    return model


def build_transform(args, model):
    data_config = resolve_data_config(vars(args), model=model, use_test_size=True, verbose=True)
    transform = create_transform(
        input_size=data_config["input_size"],
        interpolation=data_config["interpolation"],
        mean=data_config["mean"],
        std=data_config["std"],
        crop_pct=data_config["crop_pct"],
        crop_mode=data_config["crop_mode"],
        is_training=False,
    )
    return data_config, transform


def shard_indices(total, rank, world_size):
    return list(range(rank, total, world_size))


def make_loader(dataset, indices, args):
    subset = torch.utils.data.Subset(dataset, indices)
    return torch.utils.data.DataLoader(
        subset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=args.pin_mem,
        drop_last=False,
        persistent_workers=args.workers > 0,
        collate_fn=collate_batch,
    )


def run_feature_extraction(args, dataset, model, loader, device, rank, world_size):
    out_dir = Path(args.out_dir)
    shard_path = out_dir / f"features_rank{rank:02d}.pt"
    if args.skip_existing_features and shard_path.exists():
        _LOG.info("Rank %d reusing existing shard %s", rank, shard_path)
        return shard_path

    if args.amp:
        amp_dtype = torch.bfloat16 if args.amp_dtype == "bfloat16" else torch.float16
        amp_autocast = partial(torch.autocast, device_type=device.type, dtype=amp_dtype)
    else:
        amp_autocast = suppress

    path_buffer = []
    class_buffer = []
    feature_batches = []
    total_seen = 0

    with torch.inference_mode():
        for batch_idx, (paths, class_indices, images) in enumerate(loader):
            images = images.to(device=device, non_blocking=args.pin_mem)
            if args.channels_last:
                images = images.contiguous(memory_format=torch.channels_last)

            with amp_autocast():
                features = model.forward_features(images)
                features = model.forward_head(features, pre_logits=True)

            if args.features_dtype == "float16":
                features = features.to(dtype=torch.float16)
            else:
                features = features.to(dtype=torch.float32)

            feature_batches.append(features.cpu())
            class_buffer.append(class_indices.clone())
            path_buffer.extend(paths)
            total_seen += len(paths)

            if batch_idx % args.log_interval == 0:
                _LOG.info(
                    "Rank %d/%d processed %d samples (%d batches)",
                    rank,
                    world_size,
                    total_seen,
                    batch_idx + 1,
                )

    features_tensor = torch.cat(feature_batches, dim=0) if feature_batches else torch.empty((0, 0))
    class_tensor = torch.cat(class_buffer, dim=0) if class_buffer else torch.empty((0,), dtype=torch.int64)
    payload = {
        "paths": path_buffer,
        "class_indices": class_tensor,
        "features": features_tensor,
        "class_ids": dataset.class_ids,
        "class_counts": dataset.class_counts,
    }
    torch.save(payload, shard_path)
    _LOG.info("Rank %d wrote feature shard to %s", rank, shard_path)
    return shard_path


def load_feature_shards(out_dir, world_size):
    all_paths = []
    all_class_indices = []
    feature_tensors = []
    class_ids = None
    class_counts = None

    for rank in range(world_size):
        shard_path = Path(out_dir) / f"features_rank{rank:02d}.pt"
        shard = torch.load(shard_path, map_location="cpu", weights_only=False)
        all_paths.extend(shard["paths"])
        all_class_indices.append(shard["class_indices"].numpy())
        feature_tensors.append(shard["features"].numpy())
        if class_ids is None:
            class_ids = shard["class_ids"]
            class_counts = shard["class_counts"]

    if feature_tensors:
        features = np.concatenate(feature_tensors, axis=0)
        class_indices = np.concatenate(all_class_indices, axis=0)
    else:
        features = np.zeros((0, 0), dtype=np.float16)
        class_indices = np.zeros((0,), dtype=np.int64)

    return all_paths, class_indices, features, class_ids, class_counts


def l2_normalize(features):
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    norms = np.clip(norms, a_min=1e-12, a_max=None)
    return features / norms


def choose_clusterer(class_features, actual_k, seed):
    sample_count = class_features.shape[0]
    if sample_count <= max(2048, actual_k * 64):
        return KMeans(n_clusters=actual_k, n_init=3, random_state=seed)
    batch_size = min(sample_count, max(2048, actual_k * 128))
    return MiniBatchKMeans(
        n_clusters=actual_k,
        batch_size=batch_size,
        n_init=3,
        max_no_improvement=20,
        reassignment_ratio=0.01,
        random_state=seed,
    )


def pick_representatives(class_features, centers):
    # Small center count makes a dense distance matrix cheaper than repeated argmins.
    distances = np.linalg.norm(class_features[None, :, :] - centers[:, None, :], axis=2)
    order = np.argsort(distances, axis=1)
    selected_samples = set()
    selected = [None] * centers.shape[0]

    for center_idx in range(centers.shape[0]):
        for sample_idx in order[center_idx]:
            sample_idx = int(sample_idx)
            if sample_idx not in selected_samples:
                selected[center_idx] = sample_idx
                selected_samples.add(sample_idx)
                break

    return [(idx, float(distances[cluster_idx, idx])) for cluster_idx, idx in enumerate(selected)]


def cluster_class(class_features, requested_k, seed):
    sample_count = class_features.shape[0]
    actual_k = min(requested_k, sample_count)
    if sample_count == 0:
        return [], 0
    if sample_count <= actual_k:
        return [(idx, 0.0) for idx in range(sample_count)], actual_k

    normalized = l2_normalize(class_features.astype(np.float32, copy=False))
    clusterer = choose_clusterer(normalized, actual_k, seed)
    clusterer.fit(normalized)
    return pick_representatives(normalized, clusterer.cluster_centers_), actual_k


def ensure_clean_symlink(target_path, source_path):
    if target_path.exists() or target_path.is_symlink():
        target_path.unlink()
    target_path.symlink_to(source_path)


def write_outputs(args, all_paths, class_indices, features, class_ids, class_counts):
    out_dir = Path(args.out_dir)
    links_root = out_dir / args.symlink_dirname
    links_root.mkdir(parents=True, exist_ok=True)

    rows = []
    json_payload = {}
    class_indices = np.asarray(class_indices)

    for class_idx, class_id in enumerate(class_ids):
        mask = class_indices == class_idx
        class_paths = [all_paths[i] for i in np.flatnonzero(mask)]
        class_features = features[mask]
        selected, actual_k = cluster_class(class_features, args.clusters_per_class, args.seed)

        class_dir = links_root / class_id
        class_dir.mkdir(parents=True, exist_ok=True)
        class_records = []

        for cluster_id, (sample_idx, distance_to_center) in enumerate(selected):
            image_path = class_paths[sample_idx]
            suffix = Path(image_path).suffix.lower()
            link_path = class_dir / f"cluster_{cluster_id:02d}{suffix}"
            ensure_clean_symlink(link_path, Path(image_path).resolve())

            record = {
                "class_id": class_id,
                "cluster_id": cluster_id,
                "image_path": image_path,
                "feature_index": int(np.flatnonzero(mask)[sample_idx]),
                "distance_to_center": distance_to_center,
                "class_size": int(class_counts[class_id]),
                "requested_k": int(args.clusters_per_class),
                "actual_k": int(actual_k),
            }
            rows.append(record)
            class_records.append(
                {
                    "cluster_id": cluster_id,
                    "image_path": image_path,
                    "distance_to_center": distance_to_center,
                    "symlink_path": str(link_path),
                }
            )

        json_payload[class_id] = {
            "class_size": int(class_counts[class_id]),
            "requested_k": int(args.clusters_per_class),
            "actual_k": int(actual_k),
            "representatives": class_records,
        }

    csv_path = out_dir / "representatives.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "class_id",
                "cluster_id",
                "image_path",
                "feature_index",
                "distance_to_center",
                "class_size",
                "requested_k",
                "actual_k",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    json_path = out_dir / "representatives.json"
    with json_path.open("w") as handle:
        json.dump(json_payload, handle, ensure_ascii=False, indent=2)

    _LOG.info("Wrote %d representative rows to %s and %s", len(rows), csv_path, json_path)


def write_run_summary(out_dir, world_size, total_images, total_classes):
    summary_path = Path(out_dir) / "run_summary.json"
    payload = {
        "world_size": int(world_size),
        "total_images": int(total_images),
        "total_classes": int(total_classes),
        "feature_shards": [f"features_rank{rank:02d}.pt" for rank in range(world_size)],
    }
    with summary_path.open("w") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
    _LOG.info("Wrote run summary to %s", summary_path)


def main():
    args = parse_args()
    setup_logging()
    seed_everything(args.seed)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    distributed, rank, _, world_size, device = init_distributed(args)
    _LOG.info("Starting with rank=%d world_size=%d device=%s", rank, world_size, device)

    try:
        model = build_model(args, device)
        _, transform = build_transform(args, model)
        dataset = RepresentativeDataset(
            data_dir=args.data_dir,
            transform=transform,
            max_classes=args.max_classes,
            max_images_per_class=args.max_images_per_class,
        )
        indices = shard_indices(len(dataset), rank, world_size)
        loader = make_loader(dataset, indices, args)

        _LOG.info(
            "Rank %d sees %d / %d samples across %d classes",
            rank,
            len(indices),
            len(dataset),
            len(dataset.class_ids),
        )
        shard_path = run_feature_extraction(args, dataset, model, loader, device, rank, world_size)

        barrier(distributed)
        if rank == 0:
            all_paths, class_indices, features, class_ids, class_counts = load_feature_shards(out_dir, world_size)
            write_run_summary(out_dir, world_size, len(all_paths), len(class_ids))
            write_outputs(args, all_paths, class_indices, features, class_ids, class_counts)
            _LOG.info("Completed representative clustering using shards rooted at %s", shard_path.parent)
    finally:
        cleanup_distributed(distributed)


if __name__ == "__main__":
    main()
