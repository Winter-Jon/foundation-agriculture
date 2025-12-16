import torch
from contextlib import suppress
from functools import partial
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.data.random_erasing import RandomErasing
from .collate import FastCollateMixupWithKey

def adapt_to_chs(x, n):
    if not isinstance(x, (list, tuple)):
        x = tuple(x)
    return x

class PrefetchLoaderWithKey:

    def __init__(
            self,
            loader: torch.utils.data.DataLoader,
            mean: tuple[float, ...] = IMAGENET_DEFAULT_MEAN,
            std: tuple[float, ...] = IMAGENET_DEFAULT_STD,
            channels: int = 3,
            device: torch.device = torch.device('cuda'),
            img_dtype: torch.dtype | None = None,
            fp16: bool = False,
            re_prob: float = 0.,
            re_mode: str = 'const',
            re_count: int = 1,
            re_num_splits: int = 0,
    ):
        mean = adapt_to_chs(mean, channels)
        std = adapt_to_chs(std, channels)
        normalization_shape = (1, channels, 1, 1)

        self.loader = loader
        self.device = device
        if fp16:
            # fp16 arg is deprecated, but will override dtype arg if set for bwd compat
            img_dtype = torch.float16
        self.img_dtype = img_dtype or torch.float32
        self.mean = torch.tensor(
            [x * 255 for x in mean], device=device, dtype=img_dtype).view(normalization_shape)
        self.std = torch.tensor(
            [x * 255 for x in std], device=device, dtype=img_dtype).view(normalization_shape)
        if re_prob > 0.:
            self.random_erasing = RandomErasing(
                probability=re_prob,
                mode=re_mode,
                max_count=re_count,
                num_splits=re_num_splits,
                device=device,
            )
        else:
            self.random_erasing = None
        self.is_cuda = device.type == 'cuda' and torch.cuda.is_available()
        self.is_npu = device.type == 'npu' and torch.npu.is_available()

    def __iter__(self):
        first = True
        if self.is_cuda:
            stream = torch.cuda.Stream(device=self.device)
            stream_context = partial(torch.cuda.stream, stream=stream)
        elif self.is_npu:
            stream = torch.npu.Stream(device=self.device)
            stream_context = partial(torch.npu.stream, stream=stream)
        else:
            stream = None
            stream_context = suppress

        # 【修改点 1】不再使用 for next_input, next_target 解包
        # 而是直接获取 batch 变量，适配长度变化
        for next_data in self.loader:
            
            # 【修改点 2】检测是否包含 keys
            if len(next_data) == 3:
                next_keys, next_input, next_target = next_data
            else:
                next_keys = None
                next_input, next_target = next_data

            with stream_context():
                next_input = next_input.to(device=self.device, non_blocking=True)
                next_target = next_target.to(device=self.device, non_blocking=True)
                next_input = next_input.to(self.img_dtype).sub_(self.mean).div_(self.std)
                if self.random_erasing is not None:
                    next_input = self.random_erasing(next_input)

            if not first:
                # 【修改点 3】根据是否有 keys 决定 yield 内容
                if keys is not None:
                    yield keys, input, target
                else:
                    yield input, target
            else:
                first = False

            if stream is not None:
                if self.is_cuda:
                    torch.cuda.current_stream(device=self.device).wait_stream(stream)
                elif self.is_npu:
                    torch.npu.current_stream(device=self.device).wait_stream(stream)

            input = next_input
            target = next_target
            keys = next_keys # 缓存当前的 keys 供下一次 yield 使用

        # 循环结束，yield 最后一个 batch
        if keys is not None:
            yield keys, input, target
        else:
            yield input, target

    def __len__(self):
        return len(self.loader)

    @property
    def sampler(self):
        return self.loader.sampler

    @property
    def dataset(self):
        return self.loader.dataset

    @property
    def mixup_enabled(self):
        if isinstance(self.loader.collate_fn, FastCollateMixupWithKey):
            return self.loader.collate_fn.mixup_enabled
        else:
            return False

    @mixup_enabled.setter
    def mixup_enabled(self, x):
        if isinstance(self.loader.collate_fn, FastCollateMixupWithKey):
            self.loader.collate_fn.mixup_enabled = x