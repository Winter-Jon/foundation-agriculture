
"""
mda_timm_inject.py

将 MDATransUNet_v2.py 中的“记忆增强注意力”思想（Q + memory, feature<->latent 投影）
注入到 timm 的 ViT / Swin Transformer 中，而不需要改 timm 源码。

核心对应 MDATransUNet_v2.py Attention.forward 中的逻辑：
- 若 memory_q_LTM is None: 用 zeros_like(query)
- 否则：memory_q_LTM 先从 latent -> feature，再与 query 相加
- 输出 memory_q_STM：feature -> latent 的投影，供下一层作为 memory_q_LTM
见原始实现：MDATransUNet_v2.py Attention 类。 (用户文件)

注意：
- timm 版本差异较大，下面实现尽量兼容常见 ViT/Swin（0.9.x~0.10.x）。
- 这里默认采用“全模型共享一条 memory 链”，即按 block 顺序更新 memory，
  以最大程度贴近原 Encoder 循环传递 memory_q_LTM 的行为。
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Dict, Any, Callable, Tuple

import torch
import torch.nn as nn


@dataclass
class MDAMemoryConfig:
    """控制记忆增强注意力的配置。"""
    latent_dim: Optional[int] = None  # None => latent_dim = dim
    enabled: bool = True              # 关闭则等价于原 timm attention
    detach_memory: bool = False       # 是否对 memory_q_STM 做 detach（避免跨层梯度耦合）
    # 注：MDATransUNet_v2 中 is_start/is_end 控制 Identity/Linear；这里保留同样语义
    use_identity_at_start: bool = True
    use_identity_at_end: bool = True


class _MemoryContainer:
    """跨多个注意力模块共享的 memory 容器。"""
    def __init__(self):
        self.mem: Optional[torch.Tensor] = None

    def reset(self):
        self.mem = None


class MDAInjectedViTAttention(nn.Module):
    """
    把 MDATransUNet_v2 的 memory 机制注入到 timm ViT Attention。
    兼容 timm 常见实现：base_attn 具备 qkv/proj/attn_drop/proj_drop/num_heads/scale 等属性。
    """
    def __init__(
        self,
        base_attn: nn.Module,
        mem_container: _MemoryContainer,
        dim: int,
        is_start: bool,
        is_end: bool,
        cfg: MDAMemoryConfig,
    ):
        super().__init__()
        self.base = base_attn
        self.mem_container = mem_container
        self.dim = dim
        self.cfg = cfg

        latent_dim = cfg.latent_dim or dim

        # 与 MDATransUNet_v2.py 对齐：is_end => Identity(feature->latent); is_start => Identity(latent->feature)
        self.proj_feature_to_latent = nn.Identity() if (cfg.use_identity_at_end and is_end) else nn.Linear(dim, latent_dim, bias=True)
        self.proj_latent_to_feature = nn.Identity() if (cfg.use_identity_at_start and is_start) else nn.Linear(latent_dim, dim, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, N, C)
        if not self.cfg.enabled:
            return self.base(x)

        B, N, C = x.shape
        assert C == self.dim, f"Unexpected dim: got {C}, expected {self.dim}"

        # timm ViT Attention 常见实现：qkv = Linear(C, 3C)
        qkv = self.base.qkv(x)
        # (B, N, 3, heads, head_dim)
        num_heads = getattr(self.base, "num_heads")
        head_dim = C // num_heads
        qkv = qkv.reshape(B, N, 3, num_heads, head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # (B, heads, N, head_dim)

        # 可选 q_norm/k_norm（timm 新版部分模型会有）
        if hasattr(self.base, "q_norm") and self.base.q_norm is not None:
            q = self.base.q_norm(q)
        if hasattr(self.base, "k_norm") and self.base.k_norm is not None:
            k = self.base.k_norm(k)

        # ---- MDA memory 注入（对齐 MDATransUNet_v2 Attention.forward）----
        # mixed_query_layer 在原实现里是 (B, N, C)。这里把 q 还原到 (B, N, C) 注入 memory，再 reshape 回去。
        q_flat = q.permute(0, 2, 1, 3).reshape(B, N, C)  # (B, N, C)

        mem = self.mem_container.mem
        if mem is None or mem.shape[:2] != (B, N):
            mem_feat = torch.zeros_like(q_flat)
        else:
            # latent -> feature
            mem_feat = self.proj_latent_to_feature(mem)
            if mem_feat.shape != q_flat.shape:
                # 维度不匹配时回退到 zeros（与原逻辑保持一致的“安全退化”）
                mem_feat = torch.zeros_like(q_flat)

        q_flat = q_flat + mem_feat  # Q + memory

        # memory_q_STM: feature -> latent
        new_mem = self.proj_feature_to_latent(q_flat)
        if self.cfg.detach_memory:
            new_mem = new_mem.detach()
        self.mem_container.mem = new_mem

        # reshape 回 (B, heads, N, head_dim)
        q = q_flat.reshape(B, N, num_heads, head_dim).permute(0, 2, 1, 3)

        # ---- 原 timm attention 计算 ----
        scale = getattr(self.base, "scale", head_dim ** -0.5)
        attn = (q * scale) @ k.transpose(-2, -1)
        attn = attn.softmax(dim=-1)
        attn = self.base.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.base.proj(x)
        x = self.base.proj_drop(x)
        return x


class MDAInjectedSwinWindowAttention(nn.Module):
    """
    把 MDA memory 注入到 timm Swin 的 WindowAttention。
    兼容 timm Swin WindowAttention 常见属性：
      qkv/proj/attn_drop/proj_drop/relative_position_bias_table/relative_position_index/num_heads/scale
    """
    def __init__(
        self,
        base_attn: nn.Module,
        mem_container: _MemoryContainer,
        dim: int,
        is_start: bool,
        is_end: bool,
        cfg: MDAMemoryConfig,
    ):
        super().__init__()
        self.base = base_attn
        self.mem_container = mem_container
        self.dim = dim
        self.cfg = cfg

        latent_dim = cfg.latent_dim or dim
        self.proj_feature_to_latent = nn.Identity() if (cfg.use_identity_at_end and is_end) else nn.Linear(dim, latent_dim, bias=True)
        self.proj_latent_to_feature = nn.Identity() if (cfg.use_identity_at_start and is_start) else nn.Linear(latent_dim, dim, bias=True)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # x: (B_, N, C), B_ = num_windows * batch
        if not self.cfg.enabled:
            # timm 原 WindowAttention.forward(x, mask)
            return self.base(x, mask=mask) if "mask" in self.base.forward.__code__.co_varnames else self.base(x, mask)

        B_, N, C = x.shape
        assert C == self.dim, f"Unexpected dim: got {C}, expected {self.dim}"

        num_heads = getattr(self.base, "num_heads")
        head_dim = C // num_heads

        qkv = self.base.qkv(x).reshape(B_, N, 3, num_heads, head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # (B_, heads, N, head_dim)

        # ---- MDA memory 注入（对齐原 Attention.forward）----
        q_flat = q.permute(0, 2, 1, 3).reshape(B_, N, C)  # (B_, N, C)

        mem = self.mem_container.mem
        if mem is None or mem.shape[:2] != (B_, N):
            mem_feat = torch.zeros_like(q_flat)
        else:
            mem_feat = self.proj_latent_to_feature(mem)
            if mem_feat.shape != q_flat.shape:
                mem_feat = torch.zeros_like(q_flat)

        q_flat = q_flat + mem_feat

        new_mem = self.proj_feature_to_latent(q_flat)
        if self.cfg.detach_memory:
            new_mem = new_mem.detach()
        self.mem_container.mem = new_mem

        q = q_flat.reshape(B_, N, num_heads, head_dim).permute(0, 2, 1, 3)

        # ---- 原 Swin WindowAttention 计算（含 relative position bias）----
        scale = getattr(self.base, "scale", head_dim ** -0.5)
        q = q * scale
        attn = q @ k.transpose(-2, -1)  # (B_, heads, N, N)

        # relative position bias
        if hasattr(self.base, "relative_position_bias_table") and hasattr(self.base, "relative_position_index"):
            relative_position_bias = self.base.relative_position_bias_table[self.base.relative_position_index.view(-1)].view(
                N, N, -1
            )  # (N, N, heads)
            relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # (heads, N, N)
            attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            # mask: (num_windows, N, N)
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, num_heads, N, N)

        attn = attn.softmax(dim=-1)
        attn = self.base.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.base.proj(x)
        x = self.base.proj_drop(x)
        return x


def inject_mda_into_timm_vit(model: nn.Module, cfg: Optional[MDAMemoryConfig] = None) -> nn.Module:
    """
    对 timm VisionTransformer 系列模型注入 MDA memory attention。

    用法：
      import timm
      m = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=100)
      m = inject_mda_into_timm_vit(m, MDAMemoryConfig(latent_dim=256))

    说明：
      - 会替换每个 block.attn 为 MDAInjectedViTAttention(wrapper)
      - 会包装 model.forward，在每次 forward 开始重置 memory（避免跨 batch 泄露）
    """
    cfg = cfg or MDAMemoryConfig()
    container = _MemoryContainer()

    if not hasattr(model, "blocks"):
        raise ValueError("This model does not look like a timm ViT (missing `blocks`).")

    blocks = model.blocks
    # blocks 可能是 nn.Sequential 或 ModuleList
    block_list = list(blocks) if isinstance(blocks, (nn.Sequential, nn.ModuleList)) else None
    if block_list is None:
        raise ValueError("Unsupported `model.blocks` type.")

    n_blocks = len(block_list)
    for i, blk in enumerate(block_list):
        if not hasattr(blk, "attn"):
            continue
        base_attn = blk.attn
        dim = getattr(base_attn, "dim", None) or getattr(base_attn.qkv, "in_features", None)
        if dim is None:
            raise ValueError("Cannot infer attention dim for ViT block.")
        blk.attn = MDAInjectedViTAttention(
            base_attn=base_attn,
            mem_container=container,
            dim=dim,
            is_start=(i == 0),
            is_end=(i == n_blocks - 1),
            cfg=cfg,
        )

    # attach and wrap forward to reset memory each call
    model._mda_mem_container = container  # type: ignore[attr-defined]
    orig_forward = model.forward

    def _forward_with_reset(x, *args, **kwargs):
        container.reset()
        return orig_forward(x, *args, **kwargs)

    model.forward = _forward_with_reset  # type: ignore[method-assign]
    return model


def inject_mda_into_timm_swin(model: nn.Module, cfg: Optional[MDAMemoryConfig] = None) -> nn.Module:
    """
    对 timm SwinTransformer 系列模型注入 MDA memory attention。

    用法：
      import timm
      m = timm.create_model('swin_base_patch4_window7_224', pretrained=True, num_classes=100)
      m = inject_mda_into_timm_swin(m, MDAMemoryConfig(latent_dim=256))

    说明：
      - 会遍历 model.layers[*].blocks[*].attn 替换为 MDAInjectedSwinWindowAttention(wrapper)
      - 会包装 model.forward，在每次 forward 开始重置 memory
      - 如果某一层输入 token 形状变化导致 memory 不匹配，会自动回退到 zeros_like（与原实现一致）
    """
    cfg = cfg or MDAMemoryConfig()
    container = _MemoryContainer()

    if not hasattr(model, "layers"):
        raise ValueError("This model does not look like a timm Swin (missing `layers`).")

    # layers: list of BasicLayer
    layers = model.layers
    all_blocks = []
    for layer in layers:
        if hasattr(layer, "blocks"):
            for blk in layer.blocks:
                all_blocks.append(blk)

    n_blocks = len(all_blocks)
    for i, blk in enumerate(all_blocks):
        if not hasattr(blk, "attn"):
            continue
        base_attn = blk.attn
        dim = getattr(base_attn, "dim", None) or getattr(base_attn.qkv, "in_features", None)
        if dim is None:
            raise ValueError("Cannot infer attention dim for Swin WindowAttention.")
        blk.attn = MDAInjectedSwinWindowAttention(
            base_attn=base_attn,
            mem_container=container,
            dim=dim,
            is_start=(i == 0),
            is_end=(i == n_blocks - 1),
            cfg=cfg,
        )

    model._mda_mem_container = container  # type: ignore[attr-defined]
    orig_forward = model.forward

    def _forward_with_reset(x, *args, **kwargs):
        container.reset()
        return orig_forward(x, *args, **kwargs)

    model.forward = _forward_with_reset  # type: ignore[method-assign]
    return model


if __name__ == '__main__':
    import timm
    from thop import profile
    import time


    cfg = MDAMemoryConfig(
        latent_dim=224,        # 对应你代码里的 config.latent_space_dim
        enabled=True,
        detach_memory=False,   # 是否让 memory 不跨层反传梯度（可按训练稳定性调整）
    )

    # 1) ViT
    vit = timm.create_model("vit_base_patch16_224", pretrained=True, num_classes=100)
    vit = inject_mda_into_timm_vit(vit, cfg)

    # 2) Swin
    swin = timm.create_model("swin_tiny_patch4_window7_224", pretrained=True, num_classes=100)
    swin = inject_mda_into_timm_swin(swin, cfg).cuda()

    print(torch.__version__)

    image = torch.rand(1, 3, 224, 224).cuda()

    f, p = profile(swin, inputs=(image,))
    # f, p = summary(net, (image, time_step))
    print('flops:%f' % f)
    print('params:%f' % p)
    print('flops: %.1f G, params: %.1f M' % (f / 1e9, p / 1e6))

    s = time.time()

    with torch.no_grad():
        out = swin(image, )

    print('infer_time:', time.time() - s)
    print("FPS:%f" % (1 / (time.time() - s)))

    print(out.shape)