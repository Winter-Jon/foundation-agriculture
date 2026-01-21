# --------------------------------------------------------
# Swin Transformer
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ze Liu
# --------------------------------------------------------
import math

import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
from timm.layers import DropPath, to_2tuple, trunc_normal_
from timm.models import register_model

try:
    import os, sys

    kernel_path = os.path.abspath(os.path.join('../..'))
    sys.path.append(kernel_path)
    from kernels.window_process.window_process import WindowProcess, WindowProcessReverse

except Exception:
    WindowProcess = None
    WindowProcessReverse = None
    # print("[Warning] Fused window process have not been installed. Please refer to get_started.md for installation.")

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

def window_partition(x, window_size):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size

    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows

def window_reverse(windows, window_size, H, W):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image

    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x

class PatchEmbed(nn.Module):

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
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        B, C, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x).flatten(2).transpose(1, 2)  # B Ph*Pw C
        if self.norm is not None:
            x = self.norm(x)
        return x

class PatchMerging(nn.Module):
    r""" Patch Merging Layer.

    Args:
        input_resolution (tuple[int]): Resolution of input feature.
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, input_resolution, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x):
        """
        x: B, H*W, C
        """
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"
        assert H % 2 == 0 and W % 2 == 0, f"x size ({H}*{W}) are not even."

        x = x.view(B, H, W, C)

        x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
        x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
        x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
        x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C
        x = torch.cat([x0, x1, x2, x3], -1)  # B H/2 W/2 4*C
        x = x.view(B, -1, 4 * C)  # B H/2*W/2 4*C

        x = self.norm(x)
        x = self.reduction(x)

        return x

class Multi_Scale_Awareness(nn.Module):
    def __init__(self, dim, ca_num_heads=4, qkv_bias=False):
        super().__init__()
        self.dim = dim
        self.ca_num_heads = ca_num_heads

        if dim % ca_num_heads != 0:
            self.padded_dim = dim + (ca_num_heads - dim % ca_num_heads)
        else:
            self.padded_dim = dim

        self.C_per_head = self.padded_dim // ca_num_heads

        self.s = nn.Linear(dim, self.padded_dim, bias=qkv_bias)

        for i in range(ca_num_heads):
            local_conv = nn.Conv2d(
                in_channels=self.C_per_head, out_channels=self.C_per_head,
                kernel_size=(1 + i * 2), padding=i, stride=1,
                groups=self.C_per_head
            )
            setattr(self, f"local_conv_{i + 1}", local_conv)

        self.bn = nn.BatchNorm2d(self.padded_dim)
        self.act = nn.GELU()

        self.cross_key = nn.Linear(dim, dim)
        self.cross_value = nn.Linear(dim, dim)

    def forward(self, x, H, W):
        B, N, C = x.shape

        x_padded = self.s(x)

        s = x_padded.reshape(B, H, W, self.ca_num_heads, self.C_per_head).permute(3, 0, 4, 1, 2)  # [heads, B, C_per_head, H, W]

        outputs = []
        for i in range(self.ca_num_heads):
            local_conv = getattr(self, f"local_conv_{i + 1}")
            s_i = local_conv(s[i])  # [B, C_per_head, H, W]
            outputs.append(s_i)

        s_out = torch.cat(outputs, dim=1)
        s_out = self.act(self.bn(s_out))
        s_out = s_out.flatten(2).transpose(1, 2)
        s_out = s_out[:, :, :self.dim]

        k = self.cross_key(s_out)
        v = self.cross_value(s_out)

        return k, v

class Global_Cross_Attention(nn.Module):
    def __init__(self, dim, num_heads=8, attn_drop_ratio=0.):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.attn_drop = nn.Dropout(attn_drop_ratio)
        self.proj = nn.Linear(dim, dim)

    def forward(self, q, k, v):
        B, N_q, C = q.shape
        B, N_k, C = k.shape
        q = q.reshape(B, N_q, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k = k.reshape(B, N_k, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v = v.reshape(B, N_k, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = self.attn_drop(attn.softmax(dim=-1))
        x = (attn @ v).transpose(1, 2).reshape(B, N_q, C)

        Wq = int(math.sqrt(N_q))
        Hq = Wq
        Wk = int(math.sqrt(N_k))
        Hk = Wk
        H = Hk // Hq

        x = x.reshape(B, Hq, Wq, C)
        x = x.repeat_interleave(H, dim=1)
        x = x.repeat_interleave(H, dim=2)

        x = self.proj(x.reshape(B, N_k, C))

        return x

class DynamicAttention(nn.Module):
    """Dynamic attention with optional recurrent (memory) query.

    This is adapted from the RecA idea in the paper: keep a running query (Q_hat)
    and add it to the current query before attention.

    Here, the query is the learnable query_tokens (reduced tokens). We keep a
    per-forward memory tensor with the same shape as the expanded query.
    """

    def __init__(self, dim=96, ca_num_heads=4, num_heads=8,
                 input_resolution=56 * 56, query_ratio=8, use_memory: bool = True):
        super().__init__()
        self.use_memory = use_memory

        self.query_tokens = nn.Parameter(torch.zeros(1, input_resolution // query_ratio, dim))
        trunc_normal_(self.query_tokens, std=0.02)

        self.attn = Multi_Scale_Awareness(dim=dim, ca_num_heads=ca_num_heads)
        self.cross_attn = Global_Cross_Attention(dim=dim, num_heads=num_heads)

    def forward(self, x, H, W, q_hat: Optional[torch.Tensor] = None):
        """Args:
            x: (B, N, C)
            q_hat: (B, Nq, C) or None
        Returns:
            x_cross: (B, N, C)
            q_hat_new: (B, Nq, C) or None
        """
        B, N, C = x.shape
        k, v = self.attn(x, H, W)

        q = self.query_tokens.expand(B, -1, -1)

        if not self.use_memory:
            q_hat_new = None
            x_cross = self.cross_attn(q, k, v)
            return x_cross, q_hat_new

        # init / shape-guard
        # 记忆查询：q_hat 保存上一轮的 query（RecA），用于增强跨层/跨块特征交互
        if (q_hat is None) or (q_hat.shape != q.shape):
            q_hat = torch.zeros_like(q)

        # 记忆融合：当前 query + 历史记忆
        q = q + q_hat
        q_hat_new = q

        x_cross = self.cross_attn(q, k, v)
        return x_cross, q_hat_new


class WindowAttention(nn.Module):
    r""" Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.

    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):

        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = qk_scale or self.head_dim ** -0.5

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))  # 2*Wh-1 * 2*Ww-1, nH

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        try:
            coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing="ij"))  # 2, Wh, Ww
        except TypeError:
            coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None, q_hat: Optional[torch.Tensor] = None, use_memory: bool = True):
        """Window-based multi-head self-attention with optional recurrent query (RecA).

        Args:
            x: (num_windows*B, N, C)
            mask: attention mask or None
            q_hat: (num_windows*B, N, C) recurrent query memory (previous Q), or None
            use_memory: if False, behaves like standard W-MSA

        Returns:
            out: (num_windows*B, N, C)
            q_hat_new: (num_windows*B, N, C) if use_memory else None
        """
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # [B_, heads, N, head_dim]

        if not use_memory:
            q_hat_new = None
        else:
            # q_hat lives in the (B_, N, C) space (before splitting heads)
            # 窗口内记忆：每个窗口保存自己的 q_hat（RecA）
            if (q_hat is None) or (q_hat.shape != x.shape):
                q_hat = torch.zeros_like(x)

            q_flat = q.transpose(1, 2).reshape(B_, N, C)  # [B_, N, C]
            # 记忆融合：当前 query + 历史记忆
            q_flat = q_flat + q_hat
            q_hat_new = q_flat
            q = q_flat.reshape(B_, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        out = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        out = self.proj(out)
        out = self.proj_drop(out)
        return out, q_hat_new

class DynamicTransformerBlock(nn.Module):

    def __init__(self,ca_num_heads,query_ratio,
                 dim, input_resolution, num_heads,
                 mlp_ratio=4.,  drop=0.,  drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                 ):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads

        self.mlp_ratio = mlp_ratio

        self.norm1 = norm_layer(dim)
        self.norm2 = norm_layer(dim)
        self.dynamic_attn = DynamicAttention(dim=dim, ca_num_heads=ca_num_heads,
                                               query_ratio=query_ratio,
                                               num_heads=num_heads,
                                               input_resolution=input_resolution[0] * input_resolution[1],
                                               use_memory=True)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, memory_state: Dict[str, Any]):
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        # 取出动态注意力的记忆
        q_hat_dyn = memory_state.get("dyn", None)

        dy_shortcut = x
        x_norm = self.norm1(x)
        x_dyn, q_hat_dyn = self.dynamic_attn(x_norm, H, W, q_hat=q_hat_dyn)
        x = self.drop_path(x_dyn) + dy_shortcut

        # FFN
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        # 写回动态注意力记忆
        memory_state["dyn"] = q_hat_dyn
        return x, memory_state

class DynamicAwareWindowBlock(nn.Module):

    def __init__(self,ca_num_heads, query_ratio, dim, input_resolution, num_heads, window_size=7, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                 fused_window_process=False):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        if min(self.input_resolution) <= self.window_size:
            # if window size is larger than input resolution, we don't partition windows
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        self.norm1 = norm_layer(dim)

        self.dynamic_attn = DynamicAttention(dim=dim, ca_num_heads=ca_num_heads,
                                               query_ratio=query_ratio,
                                               num_heads=num_heads,
                                               input_resolution=input_resolution[0] * input_resolution[1],
                                               use_memory=True)

        self.attn = WindowAttention(
            dim, window_size=to_2tuple(self.window_size), num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        if self.shift_size > 0:
            # calculate attention mask for SW-MSA
            H, W = self.input_resolution
            img_mask = torch.zeros((1, H, W, 1))  # 1 H W 1
            h_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            w_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1

            mask_windows = window_partition(img_mask, self.window_size)  # nW, window_size, window_size, 1
            mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        else:
            attn_mask = None

        self.register_buffer("attn_mask", attn_mask)
        self.fused_window_process = fused_window_process

    def forward(self, x, memory_state: Dict[str, Any]):
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        # -------- Dynamic (cross) attention with recurrent query on query_tokens --------
        # 动态注意力记忆（跨层/跨块）
        q_hat_dyn = memory_state.get("dyn", None)
        dy_shortcut = x
        x_norm = self.norm1(x)
        x_dyn, q_hat_dyn = self.dynamic_attn(x_norm, H, W, q_hat=q_hat_dyn)
        x = self.drop_path(x_dyn) + dy_shortcut
        # 写回动态注意力记忆
        memory_state["dyn"] = q_hat_dyn

        # -------- Window self-attention with recurrent query (RecA) --------
        # 窗口注意力记忆（局部窗口内的 q_hat）
        q_hat_win = memory_state.get("win", None)
        if (q_hat_win is None) or (q_hat_win.shape != x.shape):
            q_hat_win = torch.zeros_like(x)

        shortcut = x
        x = self.norm2(x)
        x = x.view(B, H, W, C)

        q_hat_map = q_hat_win.view(B, H, W, C)

        # cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
            shifted_q = torch.roll(q_hat_map, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
            if not self.fused_window_process:
                x_windows = window_partition(shifted_x, self.window_size)
            else:
                x_windows = WindowProcess.apply(x, B, H, W, C, -self.shift_size, self.window_size)
            q_windows = window_partition(shifted_q, self.window_size)
        else:
            shifted_x = x
            x_windows = window_partition(shifted_x, self.window_size)
            q_windows = window_partition(q_hat_map, self.window_size)

        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)
        q_hat_windows = q_windows.view(-1, self.window_size * self.window_size, C)

        attn_windows, q_hat_windows_new = self.attn(x_windows, mask=self.attn_mask, q_hat=q_hat_windows, use_memory=True)

        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        q_hat_windows_new = q_hat_windows_new.view(-1, self.window_size, self.window_size, C)

        # reverse cyclic shift (feature)
        if self.shift_size > 0:
            if not self.fused_window_process:
                shifted_x = window_reverse(attn_windows, self.window_size, H, W)
                x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
            else:
                x = WindowProcessReverse.apply(attn_windows, B, H, W, C, self.shift_size, self.window_size)
        else:
            x = window_reverse(attn_windows, self.window_size, H, W)

        # reverse cyclic shift (query memory)
        if self.shift_size > 0:
            q_shifted = window_reverse(q_hat_windows_new, self.window_size, H, W)
            q_map_new = torch.roll(q_shifted, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            q_map_new = window_reverse(q_hat_windows_new, self.window_size, H, W)

        q_hat_win_new = q_map_new.view(B, H * W, C)
        # 写回窗口注意力记忆
        memory_state["win"] = q_hat_win_new

        x = x.view(B, H * W, C)
        x = shortcut + self.drop_path(x)

        # FFN
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x, memory_state

class WindowBlock(nn.Module):
    def __init__(self, dim, input_resolution, num_heads, window_size=7,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm, fused_window_process=False):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.mlp_ratio = mlp_ratio

        if min(self.input_resolution) <= self.window_size:
            self.window_size = min(self.input_resolution)

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim, window_size=to_2tuple(self.window_size), num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.fused_window_process = fused_window_process

    def forward(self, x, memory_state: Dict[str, Any]):
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        # 窗口注意力记忆
        q_hat_win = memory_state.get("win", None)
        if (q_hat_win is None) or (q_hat_win.shape != x.shape):
            q_hat_win = torch.zeros_like(x)

        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)

        q_hat_map = q_hat_win.view(B, H, W, C)

        x_windows = window_partition(x, self.window_size)
        q_windows = window_partition(q_hat_map, self.window_size)

        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)
        q_hat_windows = q_windows.view(-1, self.window_size * self.window_size, C)

        attn_windows, q_hat_windows_new = self.attn(x_windows, mask=None, q_hat=q_hat_windows, use_memory=True)
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        q_hat_windows_new = q_hat_windows_new.view(-1, self.window_size, self.window_size, C)

        x = window_reverse(attn_windows, self.window_size, H, W)
        q_map_new = window_reverse(q_hat_windows_new, self.window_size, H, W)

        x = x.view(B, H * W, C)
        # 写回窗口注意力记忆
        memory_state["win"] = q_map_new.view(B, H * W, C)

        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x, memory_state

class DynamicBasicLayer(nn.Module):

    def __init__(self,ca_num_heads,  query_ratio,dim, input_resolution, depth, num_heads, window_size,
                 mlp_ratio=4.,  drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, downsample=None, use_checkpoint=False,
                 ):

        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        # build blocks
        self.blocks = nn.ModuleList([
            DynamicTransformerBlock(ca_num_heads=ca_num_heads,
                                    query_ratio= query_ratio,
                                 dim=dim, input_resolution=input_resolution,
                                 num_heads=num_heads,
                                 mlp_ratio=mlp_ratio,
                                 drop=drop,
                                 drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                                 norm_layer=norm_layer,
                                 )
            for i in range(depth)])

        # patch merging layer
        if downsample is not None:
            self.downsample = downsample(input_resolution, dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None

    def forward(self, x, memory_state: Dict[str, Any]):
        for blk in self.blocks:
            # NOTE: torch.utils.checkpoint only supports Tensor inputs/outputs;
            # memory_state is a dict, so we disable checkpointing in memory-attention mode.
            x, memory_state = blk(x, memory_state)
        if self.downsample is not None:
            x = self.downsample(x)
            # resolution / token count changes: reset memories
            # 下采样后 token 数变化，记忆需要重置
            memory_state["dyn"] = None
            memory_state["win"] = None
        return x, memory_state

class HybridBasicLayer(nn.Module):

    def __init__(self,ca_num_heads,query_ratio,dim, input_resolution, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, downsample=None, use_checkpoint=False,
                 fused_window_process=False):

        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        # build blocks
        self.blocks = nn.ModuleList([
            DynamicAwareWindowBlock(ca_num_heads=ca_num_heads,
                                    query_ratio= query_ratio,
                                 dim=dim, input_resolution=input_resolution,
                                 num_heads=num_heads, window_size=window_size,
                                 shift_size=0 if (i % 2 == 0) else window_size // 2,
                                 mlp_ratio=mlp_ratio,
                                 qkv_bias=qkv_bias, qk_scale=qk_scale,
                                 drop=drop, attn_drop=attn_drop,
                                 drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                                 norm_layer=norm_layer,
                                 fused_window_process=fused_window_process)
            for i in range(depth)])

        # patch merging layer
        if downsample is not None:
            self.downsample = downsample(input_resolution, dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None

    def forward(self, x, memory_state: Dict[str, Any]):
        for blk in self.blocks:
            # NOTE: torch.utils.checkpoint only supports Tensor inputs/outputs;
            # memory_state is a dict, so we disable checkpointing in memory-attention mode.
            x, memory_state = blk(x, memory_state)
        if self.downsample is not None:
            x = self.downsample(x)
            # resolution / token count changes: reset memories
            # 下采样后 token 数变化，记忆需要重置
            memory_state["dyn"] = None
            memory_state["win"] = None
        return x, memory_state

class WindowBasicLayer(nn.Module):

    def __init__(self,  dim, input_resolution, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, downsample=None, use_checkpoint=False,
                 fused_window_process=False):

        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        # build blocks
        self.blocks = nn.ModuleList([
                    WindowBlock(
                                 dim=dim, input_resolution=input_resolution,
                                 num_heads=num_heads, window_size=window_size,
                                 mlp_ratio=mlp_ratio,
                                 qkv_bias=qkv_bias, qk_scale=qk_scale,
                                 drop=drop, attn_drop=attn_drop,
                                 drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                                 norm_layer=norm_layer,
                                 fused_window_process=fused_window_process)
            for i in range(depth)])

        # patch merging layer
        if downsample is not None:
            self.downsample = downsample(input_resolution, dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None

    def forward(self, x, memory_state: Dict[str, Any]):
        for blk in self.blocks:
            # NOTE: torch.utils.checkpoint only supports Tensor inputs/outputs;
            # memory_state is a dict, so we disable checkpointing in memory-attention mode.
            x, memory_state = blk(x, memory_state)
        if self.downsample is not None:
            x = self.downsample(x)
            # resolution / token count changes: reset memories
            # 下采样后 token 数变化，记忆需要重置
            memory_state["dyn"] = None
            memory_state["win"] = None
        return x, memory_state

class DynamicTransformer(nn.Module):
    r""" Swin Transformer
        A PyTorch impl of : `Swin Transformer: Hierarchical Vision Transformer using Shifted Windows`  -
          https://arxiv.org/pdf/2103.14030

    Args:
        img_size (int | tuple(int)): Input image size. Default 224
        patch_size (int | tuple(int)): Patch size. Default: 4
        in_chans (int): Number of input image channels. Default: 3
        num_classes (int): Number of classes for classification head. Default: 1000
        embed_dim (int): Patch embedding dimension. Default: 96
        depths (tuple(int)): Depth of each Swin Transformer layer.
        num_heads (tuple(int)): Number of attention heads in different layers.
        window_size (int): Window size. Default: 7
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4
        qkv_bias (bool): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float): Override default qk scale of head_dim ** -0.5 if set. Default: None
        drop_rate (float): Dropout rate. Default: 0
        attn_drop_rate (float): Attention dropout rate. Default: 0
        drop_path_rate (float): Stochastic depth rate. Default: 0.1
        norm_layer (nn.Module): Normalization layer. Default: nn.LayerNorm.
        ape (bool): If True, add absolute position embedding to the patch embedding. Default: False
        patch_norm (bool): If True, add normalization after patch embedding. Default: True
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False
        fused_window_process (bool, optional): If True, use one kernel to fused window shift & window partition for acceleration, similar for the reversed part. Default: False
    """

    def __init__(self, ca_num_heads=[6, 6, 6, -1],
                 query_ratio = [64, 16, 4, 1],
                 img_size=224, patch_size=4, in_chans=3, num_classes=1000,
                 embed_dim=96, depths=[2, 2, 6, 2], num_heads=[3, 6, 12, 24],
                 window_size=7, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm, ape=False, patch_norm=True,
                 use_checkpoint=False, fused_window_process=False, **kwargs):
        super().__init__()

        self.num_classes = num_classes
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.ape = ape
        self.patch_norm = patch_norm
        self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))
        self.mlp_ratio = mlp_ratio

        # split image into non-overlapping patches
        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)
        num_patches = self.patch_embed.num_patches
        patches_resolution = self.patch_embed.patches_resolution
        self.patches_resolution = patches_resolution

        # absolute position embedding
        if self.ape:
            self.absolute_pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
            trunc_normal_(self.absolute_pos_embed, std=.02)

        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule

        # build layers
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            if i_layer in [0, 1]:
                layer = DynamicBasicLayer(
                    ca_num_heads=ca_num_heads[i_layer],
                   query_ratio = query_ratio[i_layer],
                   dim=int(embed_dim * 2 ** i_layer),
                   input_resolution=(patches_resolution[0] // (2 ** i_layer),
                                     patches_resolution[1] // (2 ** i_layer)),
                   depth=depths[i_layer],
                   num_heads=num_heads[i_layer],
                   window_size=window_size,
                   mlp_ratio=self.mlp_ratio,
                   drop=drop_rate,
                   drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                   norm_layer=norm_layer,
                   downsample=PatchMerging if (i_layer < self.num_layers - 1) else None,
                   use_checkpoint=use_checkpoint,
                               )
            elif i_layer == 2:
                layer = HybridBasicLayer(
                    ca_num_heads=ca_num_heads[i_layer],
                   dim=int(embed_dim * 2 ** i_layer), query_ratio=query_ratio[i_layer],
                   input_resolution=(patches_resolution[0] // (2 ** i_layer),
                                     patches_resolution[1] // (2 ** i_layer)),
                   depth=depths[i_layer],
                   num_heads=num_heads[i_layer],
                   window_size=window_size,
                   mlp_ratio=self.mlp_ratio,
                   qkv_bias=qkv_bias, qk_scale=qk_scale,
                   drop=drop_rate, attn_drop=attn_drop_rate,
                   drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                   norm_layer=norm_layer,
                   downsample=PatchMerging if (i_layer < self.num_layers - 1) else None,
                   use_checkpoint=use_checkpoint,
                   fused_window_process=fused_window_process)
            elif i_layer == 3:
                layer = WindowBasicLayer(
                       dim=int(embed_dim * 2 ** i_layer),
                       input_resolution=(patches_resolution[0] // (2 ** i_layer),
                                         patches_resolution[1] // (2 ** i_layer)),
                       depth=depths[i_layer],
                       num_heads=num_heads[i_layer],
                       window_size=window_size,
                       mlp_ratio=self.mlp_ratio,
                       qkv_bias=qkv_bias, qk_scale=qk_scale,
                       drop=drop_rate, attn_drop=attn_drop_rate,
                       drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                       norm_layer=norm_layer,
                       downsample=PatchMerging if (i_layer < self.num_layers - 1) else None,
                       use_checkpoint=use_checkpoint,
                       fused_window_process=fused_window_process)

            self.layers.append(layer)

        self.norm = norm_layer(self.num_features)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'absolute_pos_embed'}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'relative_position_bias_table'}

    def forward_features(self, x):
        x = self.patch_embed(x)
        if self.ape:
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x)

        # 记忆状态在层间传递（dyn: 动态注意力；win: 窗口注意力）
        memory_state: Dict[str, Any] = {"dyn": None, "win": None}

        for layer in self.layers:
            x, memory_state = layer(x, memory_state)

        x = self.norm(x)  # B L C
        x = self.avgpool(x.transpose(1, 2))  # B C 1
        x = torch.flatten(x, 1)
        return x

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x

def creat_model():
    model = DynamicTransformer(embed_dim=96,ca_num_heads=[6, 6, 6, -1])
    return model


@register_model
def dynamic_transformer_v2(pretrained=False, **kwargs):
    model = DynamicTransformer(embed_dim=96,ca_num_heads=[6, 6, 6, -1], **kwargs)
    return model

if __name__ == '__main__':
    dummy_input = torch.randn(2, 3, 224, 224)  # [B, C, H, W]
    from thop import profile
    model = creat_model()
    output = model(dummy_input)
    print(model)
    f, p = profile(model, inputs=(dummy_input,))
    print('flops:%f' % f)
    print('params:%f' % p)
    print('flops: %.1f G, params: %.1f M' % (f / 1e9, p / 1e6))
    import time
    s = time.time()
    with torch.no_grad():
        out = model(dummy_input, )

    print('infer_time:', time.time() - s)
    print("FPS:%f" % (1 / (time.time() - s)))

    print(out.shape)
    print("Output shape:", output.shape)