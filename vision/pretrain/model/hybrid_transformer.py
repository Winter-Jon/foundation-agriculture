# --------------------------------------------------------
# Swin Transformer
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ze Liu
# --------------------------------------------------------

"""Dynamic Swin-style Transformer.

Notes
- This implementation keeps the original model initialization parameters intact.
- Comments are intentionally minimal and only highlight key logic.
"""

from __future__ import annotations

import os
import sys

import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
from timm.layers import DropPath, to_2tuple, trunc_normal_

from timm.models import register_model

# Optional fused kernels (window shift + partition) for speed.
try:
    kernel_path = os.path.abspath(os.path.join(".."))
    sys.path.append(kernel_path)
    from kernels.window_process.window_process import (  # type: ignore
        WindowProcess,
        WindowProcessReverse,
    )
except Exception:
    WindowProcess = None
    WindowProcessReverse = None
    print(
        "[Warning] Fused window process have not been installed. "
        "Please refer to get_started.md for installation."
    )


class Mlp(nn.Module):
    """Feed-forward network used in Transformer blocks."""

    def __init__(
        self,
        in_features: int,
        hidden_features: int | None = None,
        out_features: int | None = None,
        act_layer=nn.GELU,
        drop: float = 0.0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


def window_partition(x: torch.Tensor, window_size: int) -> torch.Tensor:
    """Split (B, H, W, C) into non-overlapping windows."""

    b, h, w, c = x.shape
    x = x.view(b, h // window_size, window_size, w // window_size, window_size, c)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, c)
    return windows


def window_reverse(windows: torch.Tensor, window_size: int, h: int, w: int) -> torch.Tensor:
    """Merge windows back to (B, H, W, C)."""

    b = int(windows.shape[0] / (h * w / window_size / window_size))
    x = windows.view(b, h // window_size, w // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(b, h, w, -1)
    return x


class Self_Attention(nn.Module):
    """Self-attention producing updated (K, V) for the subsequent cross-attention."""

    def __init__(self, dim: int, num_heads: int = 8, attn_drop_ratio: float = 0.0):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5

        self.attn_drop = nn.Dropout(attn_drop_ratio)
        self.cross_key = nn.Linear(dim, dim)
        self.cross_value = nn.Linear(dim, dim)

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        residual_q = q

        b, n_q, c = q.shape
        n_k = k.shape[1]
        n_v = v.shape[1]

        q = q.reshape(b, n_q, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k = k.reshape(b, n_k, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v = v.reshape(b, n_v, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = self.attn_drop(attn.softmax(dim=-1))

        x = (attn @ v).transpose(1, 2).reshape(b, n_q, c)

        k_out = self.cross_key(x + residual_q)
        v_out = self.cross_value(x + residual_q)
        return k_out, v_out


class Cross_Attention(nn.Module):
    """Standard multi-head cross-attention."""

    def __init__(self, dim: int, num_heads: int = 8, attn_drop_ratio: float = 0.0):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5

        self.attn_drop = nn.Dropout(attn_drop_ratio)

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        b, n_q, c = q.shape
        n_k = k.shape[1]
        n_v = v.shape[1]

        q = q.reshape(b, n_q, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k = k.reshape(b, n_k, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v = v.reshape(b, n_v, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = self.attn_drop(attn.softmax(dim=-1))

        x = (attn @ v).transpose(1, 2).reshape(b, n_q, c)
        return x


class Query_Attention(nn.Module):
    """Learnable query tokens (Q-Former style)."""

    def __init__(self, num_query_tokens: int, hidden_size: int):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_size, eps=1e-6)

        self.query_tokens = nn.Parameter(torch.zeros(1, num_query_tokens, hidden_size))
        trunc_normal_(self.query_tokens, std=0.02)

    def forward(self, batch_size: int) -> torch.Tensor:
        query_tokens = self.query_tokens.expand(batch_size, -1, -1)
        return self.norm(query_tokens)


class WindowAttention(nn.Module):
    """Window based multi-head self attention (W-MSA/SW-MSA) with relative position bias."""

    def __init__(
        self,
        dim: int,
        window_size: tuple[int, int],
        num_heads: int,
        qkv_bias: bool = True,
        qk_scale: float | None = None,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads

        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5

        # Relative position bias table.
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads)
        )

        # Pair-wise relative position index.
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing="ij"))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += self.window_size[0] - 1
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        trunc_normal_(self.relative_position_bias_table, std=0.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        b_, n, c = x.shape
        qkv = (
            self.qkv(x)
            .reshape(b_, n, 3, self.num_heads, c // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = q * self.scale
        attn = q @ k.transpose(-2, -1)

        rel_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1],
            self.window_size[0] * self.window_size[1],
            -1,
        )
        rel_bias = rel_bias.permute(2, 0, 1).contiguous()
        attn = attn + rel_bias.unsqueeze(0)

        if mask is not None:
            n_w = mask.shape[0]
            attn = attn.view(b_ // n_w, n_w, self.num_heads, n, n) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, n, n)

        attn = self.softmax(attn)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(b_, n, c)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    def extra_repr(self) -> str:
        return f"dim={self.dim}, window_size={self.window_size}, num_heads={self.num_heads}"

    def flops(self, n: int) -> int:
        # Standard W-MSA FLOPs (approx.).
        flops = 0
        flops += n * self.dim * 3 * self.dim
        flops += self.num_heads * n * (self.dim // self.num_heads) * n
        flops += self.num_heads * n * n * (self.dim // self.num_heads)
        flops += n * self.dim * self.dim
        return flops


class prepare(nn.Module):
    """Generate (Q, K, V) for dynamic global attention."""

    def __init__(
        self,
        dim: int,
        num_patches: int = 56 * 56,
        poolstep: int = 8,
        drop_rate: float = 0.0,
        ape: bool = True,
    ):
        super().__init__()
        self.ape = ape

        if self.ape:
            self.absolute_pos_embed = nn.Parameter(torch.zeros(1, num_patches, dim))
            trunc_normal_(self.absolute_pos_embed, std=0.02)

        self.pos_drop = nn.Dropout(p=drop_rate)
        self.avg_pool = nn.AvgPool2d(kernel_size=poolstep, stride=poolstep)

        self.query_proj = nn.Conv2d(dim, dim, kernel_size=1)
        self.key_proj = nn.Conv2d(dim, dim, kernel_size=1)
        self.value_proj = nn.Conv2d(dim, dim, kernel_size=1)

    def forward(self, featuremap: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
        # featuremap: [B, C, H, W]
        b = featuremap.shape[0]

        q = self.query_proj(featuremap).flatten(2).transpose(1, 2)
        if self.ape:
            q = q + self.absolute_pos_embed
        q = self.pos_drop(q)

        pooled = self.avg_pool(featuremap)
        k = self.key_proj(pooled).flatten(2).transpose(1, 2)
        v = self.value_proj(pooled).flatten(2).transpose(1, 2)
        return q, k, v, b


class DynamicGlobalAttention(nn.Module):
    """Dynamic global attention: pooled KV + learned queries."""

    def __init__(
        self,
        dim: int,
        input_resolution: tuple[int, int],
        num_heads: int = 8,
        attn_drop: float = 0.0,
        pool_step: int = 8,
        ape: bool = True,
    ):
        super().__init__()
        self.input_dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads

        self.qkv1 = prepare(
            dim=dim,
            poolstep=pool_step,
            num_patches=input_resolution[0] * input_resolution[1],
            ape=ape,
        )
        self.self_attention = Self_Attention(dim=dim, num_heads=num_heads, attn_drop_ratio=attn_drop)
        self.qformer = Query_Attention(
            num_query_tokens=input_resolution[0] * input_resolution[1],
            hidden_size=dim,
        )
        self.cross_attention = Cross_Attention(dim=dim, num_heads=num_heads, attn_drop_ratio=attn_drop)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, H, W]
        q1, k1, v1, b = self.qkv1(x)
        k2, v2 = self.self_attention(q1, k1, v1)
        q2 = self.qformer(b)
        x_cross = self.cross_attention(q2, k2, v2)
        return self.proj(x_cross)  # [B, H*W, C]


class DynamicTransformerBlock(nn.Module):
    """Swin Transformer block with an additional dynamic global attention branch."""

    def __init__(
        self,
        pool_step,
        dynamic_ape,
        dim,
        input_resolution,
        num_heads,
        window_size=7,
        shift_size=0,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        fused_window_process=False,
    ):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio

        if min(self.input_resolution) <= self.window_size:
            self.shift_size = 0
            self.window_size = min(self.input_resolution)

        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        self.norm1 = norm_layer(dim)
        self.norm2 = norm_layer(dim)
        self.dynamic_attn = DynamicGlobalAttention(
            dim=dim,
            input_resolution=input_resolution,
            num_heads=num_heads,
            attn_drop=attn_drop,
            pool_step=pool_step,
            ape=dynamic_ape,
        )
        self.attn = WindowAttention(
            dim,
            window_size=to_2tuple(self.window_size),
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
        )

        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm3 = norm_layer(dim)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            act_layer=act_layer,
            drop=drop,
        )

        # Attention mask for SW-MSA.
        if self.shift_size > 0:
            h, w = self.input_resolution
            img_mask = torch.zeros((1, h, w, 1))
            h_slices = (
                slice(0, -self.window_size),
                slice(-self.window_size, -self.shift_size),
                slice(-self.shift_size, None),
            )
            w_slices = (
                slice(0, -self.window_size),
                slice(-self.window_size, -self.shift_size),
                slice(-self.shift_size, None),
            )
            cnt = 0
            for hs in h_slices:
                for ws in w_slices:
                    img_mask[:, hs, ws, :] = cnt
                    cnt += 1

            mask_windows = window_partition(img_mask, self.window_size)
            mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, 0.0)
        else:
            attn_mask = None

        self.register_buffer("attn_mask", attn_mask)
        self.fused_window_process = fused_window_process

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, w = self.input_resolution
        b, l, c = x.shape
        assert l == h * w, "input feature has wrong size"

        # Dynamic global attention branch.
        dy_shortcut = x
        x = self.norm1(x)
        x_dyn = self.dynamic_attn(x.view(b, h, w, c).permute(0, 3, 1, 2))
        x = self.drop_path(x_dyn) + dy_shortcut

        # Window attention branch.
        shortcut = x
        x = self.norm2(x).view(b, h, w, c)

        if self.shift_size > 0:
            if not self.fused_window_process:
                shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
                x_windows = window_partition(shifted_x, self.window_size)
            else:
                x_windows = WindowProcess.apply(x, b, h, w, c, -self.shift_size, self.window_size)
        else:
            shifted_x = x
            x_windows = window_partition(shifted_x, self.window_size)

        x_windows = x_windows.view(-1, self.window_size * self.window_size, c)
        attn_windows = self.attn(x_windows, mask=self.attn_mask)
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, c)

        if self.shift_size > 0:
            if not self.fused_window_process:
                shifted_x = window_reverse(attn_windows, self.window_size, h, w)
                x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
            else:
                x = WindowProcessReverse.apply(attn_windows, b, h, w, c, self.shift_size, self.window_size)
        else:
            x = window_reverse(attn_windows, self.window_size, h, w)

        x = x.view(b, h * w, c)
        x = shortcut + self.drop_path(x)

        # FFN.
        x = x + self.drop_path(self.mlp(self.norm3(x)))
        return x


class PatchMerging(nn.Module):
    """Patch merging layer (downsample by 2)."""

    def __init__(self, input_resolution, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim

        self.down = nn.Conv2d(
            in_channels=dim,
            out_channels=2 * dim,
            kernel_size=2,
            stride=2,
            padding=0,
        )
        self.norm = norm_layer(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, H*W, C]
        h, w = self.input_resolution
        b, _, c = x.shape

        x = self.norm(x.reshape(b, c, -1).permute(0, 2, 1)).reshape(b, h, w, c).permute(0, 3, 1, 2)
        x = self.down(x)
        x = x.reshape(b, 2 * c, -1).permute(0, 2, 1)
        return x


class BasicLayer(nn.Module):
    """A basic stage (multiple Transformer blocks + optional downsample)."""

    def __init__(
        self,
        pool_step,
        dynamic_ape,
        dim,
        input_resolution,
        depth,
        num_heads,
        window_size,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        norm_layer=nn.LayerNorm,
        downsample=None,
        use_checkpoint=False,
        fused_window_process=False,
    ):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        self.blocks = nn.ModuleList(
            [
                DynamicTransformerBlock(
                    pool_step=pool_step,
                    dynamic_ape=dynamic_ape,
                    dim=dim,
                    input_resolution=input_resolution,
                    num_heads=num_heads,
                    window_size=window_size,
                    shift_size=0 if (i % 2 == 0) else window_size // 2,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop,
                    attn_drop=attn_drop,
                    drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                    norm_layer=norm_layer,
                    fused_window_process=fused_window_process,
                )
                for i in range(depth)
            ]
        )

        self.downsample = downsample(input_resolution, dim=dim, norm_layer=norm_layer) if downsample else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for blk in self.blocks:
            x = checkpoint.checkpoint(blk, x) if self.use_checkpoint else blk(x)
        if self.downsample is not None:
            x = self.downsample(x)
        return x

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, depth={self.depth}"


class PatchEmbed(nn.Module):
    """Image to Patch Embedding."""

    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)

        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = norm_layer(embed_dim) if norm_layer is not None else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, _, h, w = x.shape
        assert (h, w) == (self.img_size[0], self.img_size[1]), (
            f"Input image size ({h}*{w}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        )
        x = self.proj(x).flatten(2).transpose(1, 2)
        if self.norm is not None:
            x = self.norm(x)
        return x


class DynamicTransformer(nn.Module):
    def __init__(
        self,
        pool_step=[8, 4, 2, 1],
        dynamic_ape=True,
        img_size=224,
        patch_size=4,
        in_chans=3,
        num_classes=1000,
        embed_dim=96,
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 24],
        window_size=7,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.1,
        norm_layer=nn.LayerNorm,
        ape=False,
        patch_norm=True,
        use_checkpoint=False,
        fused_window_process=False,
        **kwargs,
    ):
        super().__init__()

        self.num_classes = num_classes
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.ape = ape
        self.patch_norm = patch_norm
        self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))
        self.mlp_ratio = mlp_ratio

        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None,
        )
        num_patches = self.patch_embed.num_patches
        patches_resolution = self.patch_embed.patches_resolution
        self.patches_resolution = patches_resolution

        if self.ape:
            self.absolute_pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
            trunc_normal_(self.absolute_pos_embed, std=0.02)

        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = BasicLayer(
                pool_step=pool_step[i_layer],
                dynamic_ape=dynamic_ape,
                dim=int(embed_dim * 2**i_layer),
                input_resolution=(
                    patches_resolution[0] // (2**i_layer),
                    patches_resolution[1] // (2**i_layer),
                ),
                depth=depths[i_layer],
                num_heads=num_heads[i_layer],
                window_size=window_size,
                mlp_ratio=self.mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[sum(depths[:i_layer]) : sum(depths[: i_layer + 1])],
                norm_layer=norm_layer,
                downsample=PatchMerging if (i_layer < self.num_layers - 1) else None,
                use_checkpoint=use_checkpoint,
                fused_window_process=fused_window_process,
            )
            self.layers.append(layer)

        self.norm = norm_layer(self.num_features)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patch_embed(x)
        if self.ape:
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x)

        for layer in self.layers:
            x = layer(x)

        x = self.norm(x)
        x = self.avgpool(x.transpose(1, 2))
        x = torch.flatten(x, 1)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.forward_features(x)
        return self.head(x)


def creat_model(num_classes):
    # Keep the original API name for compatibility.
    model = DynamicTransformer(pool_step=[8, 4, 2, 1], dynamic_ape=True, num_classes=num_classes)
    return model


@register_model
def hybrid_transformer_base(pretrained=False, **kwargs):
    model = DynamicTransformer(
        pool_step=[8, 4, 2, 1],
        dynamic_ape=True,
        **kwargs,
    )
    return model

if __name__ == "__main__":
    dummy_input = torch.randn(2, 3, 224, 224)

    model = creat_model(1000)
    output = model(dummy_input)
    print(model)
    print("Output shape:", output.shape)

    # Optional profiling (requires thop)
    try:
        from thop import profile  # type: ignore

        flops, params = profile(model, inputs=(dummy_input,))
        print(f"flops: {flops}")
        print(f"params: {params}")
        print(f"flops: {flops / 1e9:.1f} G, params: {params / 1e6:.1f} M")
    except Exception:
        pass

    import time

    s = time.time()
    with torch.no_grad():
        _ = model(dummy_input)

    elapsed = time.time() - s
    print("infer_time:", elapsed)
    if elapsed > 0:
        print("FPS:", 1 / elapsed)
