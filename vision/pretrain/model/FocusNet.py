import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint

from timm.models import register_model

def drop_path(x: torch.Tensor, drop_prob: float = 0.0, training: bool = False) -> torch.Tensor:
    """
    Stochastic Depth / DropPath（按样本丢弃残差分支）。
    - drop_prob=0 或 eval 模式：恒等映射
    - 否则：按 batch 维度生成 [B,1,1,1...] mask
    """
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1.0 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()
    return x.div(keep_prob) * random_tensor


class DropPath(nn.Module):
    """DropPath 模块封装。"""

    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = float(drop_prob)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return drop_path(x, self.drop_prob, self.training)

    def extra_repr(self) -> str:
        return f"drop_prob={self.drop_prob}"


# class LayerNorm(nn.Module):
#     """
#     ConvNeXt 风格 LayerNorm，支持：
#     - channels_last: (B,H,W,C)
#     - channels_first: (B,C,H,W)
#     """

#     def __init__(self, normalized_shape: int, eps: float = 1e-6, data_format: str = "channels_last"):
#         super().__init__()
#         if data_format not in ("channels_last", "channels_first"):
#             raise NotImplementedError(f"Unsupported data_format={data_format}")

#         self.weight = nn.Parameter(torch.ones(normalized_shape))
#         self.bias = nn.Parameter(torch.zeros(normalized_shape))
#         self.eps = eps
#         self.data_format = data_format
#         self.normalized_shape = (normalized_shape,)

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         if self.data_format == "channels_last":
#             return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)

#         # channels_first: 沿 channel 做 LN
#         u = x.mean(1, keepdim=True)
#         s = (x - u).pow(2).mean(1, keepdim=True)
#         x = (x - u) / torch.sqrt(s + self.eps)
#         return self.weight[:, None, None] * x + self.bias[:, None, None]

class LayerNorm(nn.GroupNorm):
    def __init__(self, num_channels, eps=1e-6):
        super().__init__(num_groups=1, num_channels=num_channels, eps=eps)

    def forward(self, x):
        return super().forward(x)

class PatchEmbed(nn.Module):
    """
    Patch Embedding（用于第一层输入）：
    - 通过 stride=patch_size 的 Conv 做 patch partition + linear embedding（下采样倍率=patch_size）
    - 额外叠加 2 层 3x3 Conv 做局部增强
    输出: [B, embed_dim, H/patch, W/patch]
    """

    def __init__(self, patch_size: int = 4, in_chans: int = 3, embed_dim: int = 96, use_norm: bool = True):
        super().__init__()
        k = (patch_size, patch_size)
        self.proj1 = nn.Conv2d(in_chans, embed_dim, kernel_size=k, stride=k, bias=False)
        self.proj2 = nn.Conv2d(embed_dim, embed_dim, kernel_size=3, stride=1, padding=1, bias=False)
        self.proj3 = nn.Conv2d(embed_dim, embed_dim, kernel_size=3, stride=1, padding=1, bias=False)

        self.norm1 = LayerNorm(embed_dim, eps=1e-6) if use_norm else nn.Identity()
        self.norm2 = LayerNorm(embed_dim, eps=1e-6) if use_norm else nn.Identity()
        self.norm3 = LayerNorm(embed_dim, eps=1e-6) if use_norm else nn.Identity()

        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.act(self.norm1(self.proj1(x)))
        x = self.act(self.norm2(self.proj2(x)))
        x = self.act(self.norm3(self.proj3(x)))
        return x


class PatchEmbedConv1(nn.Module):
    """
    精简版 Patch Embedding：仅一层 stride=patch_size 的 Conv。
    输出: [B, embed_dim, H/patch, W/patch]
    """

    def __init__(self, patch_size: int = 4, in_chans: int = 3, embed_dim: int = 96, use_norm: bool = True):
        super().__init__()
        k = (patch_size, patch_size)
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=k, stride=k, bias=False)
        self.norm = LayerNorm(embed_dim, eps=1e-6) if use_norm else nn.Identity()
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.norm(self.proj(x)))


class PatchMerging(nn.Module):
    """
    下采样模块（2x 下采样）：
    - 2x2 stride=2 Conv：空间降采样，同时通道扩张到 2C
    - 叠加 2 个 3x3 Conv 做局部增强（与你原实现一致）
    输出: [B, 2C, H/2, W/2]
    """

    def __init__(self, dim: int):
        super().__init__()
        self.proj1 = nn.Conv2d(dim, dim * 2, kernel_size=2, stride=2, bias=False)
        self.proj2 = nn.Conv2d(dim * 2, dim * 2, kernel_size=3, stride=1, padding=1, bias=False)
        self.proj3 = nn.Conv2d(dim * 2, dim * 2, kernel_size=3, stride=1, padding=1, bias=False)

        self.norm1 = LayerNorm(dim * 2, eps=1e-6)
        self.norm2 = LayerNorm(dim * 2, eps=1e-6)
        self.norm3 = LayerNorm(dim * 2, eps=1e-6)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.act(self.norm1(self.proj1(x)))
        x = self.act(self.norm2(self.proj2(x)))
        x = self.act(self.norm3(self.proj3(x)))
        return x


class PatchMergingConv1(nn.Module):
    """
    精简版下采样：仅 2x2 stride=2 Conv。
    输出: [B, 2C, H/2, W/2]
    """

    def __init__(self, dim: int):
        super().__init__()
        self.proj = nn.Conv2d(dim, dim * 2, kernel_size=2, stride=2, bias=False)
        self.norm = LayerNorm(dim * 2, eps=1e-6)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.norm(self.proj(x)))


class ConvMlp(nn.Module):
    """
    Conv 形式 MLP（用于 block 内的 FFN）：
    - LN -> 1x1 Conv 扩维 -> GELU
    - DWConv(3x3) 做局部位置增强 -> GELU
    - 1x1 Conv 恢复维度
    - layer_scale + DropPath（与原逻辑一致）
    """

    def __init__(self, dim: int, hidden_dim: int, drop_path_prob: float = 0.0):
        super().__init__()
        self.norm = LayerNorm(dim, eps=1e-6)
        self.fc1 = nn.Conv2d(dim, hidden_dim, kernel_size=1)
        self.pos = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1, groups=hidden_dim)
        self.fc2 = nn.Conv2d(hidden_dim, dim, kernel_size=1)
        self.act = nn.GELU()

        layer_scale_init = 1e-6
        self.layer_scale = nn.Parameter(layer_scale_init * torch.ones((dim,)), requires_grad=True)
        self.drop_path = DropPath(drop_path_prob) if drop_path_prob > 0.0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skip = x
        x = self.norm(x)
        x = self.act(self.fc1(x))
        x = x + self.act(self.pos(x))
        x = self.fc2(x)
        x = self.drop_path(self.layer_scale[:, None, None] * x)
        return skip + x


class WindowFocus(nn.Module):
    """
    Window Focus（核心注意力/门控模块，与你原实现一致但更清晰）：
    1) KV 分支：1x1 Conv + GELU + DWConv(kxk)（提取局部上下文）
    2) AreaMap 分支：把特征按窗口划分后在通道维拼接，再用 group 1x1 Conv 做区域选择
    3) ChannelMap 分支：全局均值池化得到 [B,C,1,1]，再 1x1 Conv 得到通道门控
    4) 三者相乘融合，再 1x1 Conv + layer_scale + DropPath + residual
    """

    def __init__(
        self,
        image_size: int,
        dim: int,
        window_size: int,
        kernel_size: int,
        num_heads: int,  # 目前未显式用到，多头参数保留以兼容你的接口
        qkv_bias: bool = True,
        proj_drop: float = 0.0,
    ):
        super().__init__()
        self.dim = dim
        self.window_size = int(window_size)
        self.kernel_size = int(kernel_size)
        self.num_heads = int(num_heads)

        # pad 后的 size 用于估算窗口数量
        padded_size = (self.window_size - image_size % self.window_size) % self.window_size + image_size
        self.num_port = (padded_size // self.window_size) ** 2 if (self.window_size <= padded_size) else 1

        self.norm = LayerNorm(dim, eps=1e-6)

        self.kv = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=1, bias=qkv_bias),
            nn.GELU(),
            nn.Conv2d(dim, dim, kernel_size=self.kernel_size, padding=(self.kernel_size - 1) // 2, groups=dim, bias=qkv_bias),
        )

        self.area_map = nn.Conv2d(dim * self.num_port, dim * self.num_port, kernel_size=1, groups=self.num_port, bias=qkv_bias)
        self.channel_map = nn.Conv2d(dim, dim, kernel_size=1, bias=qkv_bias)
        self.proj = nn.Conv2d(dim, dim, kernel_size=1, bias=qkv_bias)

        self.drop_path = DropPath(proj_drop) if proj_drop > 0.0 else nn.Identity()

        layer_scale_init = 1e-6
        self.layer_scale = nn.Parameter(layer_scale_init * torch.ones((dim,)), requires_grad=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B,C,H,W]
        B, C, H, W = x.shape
        skip = x

        x = self.norm(x)
        kv = self.kv(x)  # [B,C,H,W]

        # channel gate
        channel_feature = x.mean((2, 3), keepdim=True)  # [B,C,1,1]
        channel_gate = self.channel_map(channel_feature)  # [B,C,1,1]

        # window partition: 推导窗口划分尺寸
        if self.window_size == 7:
            win_h = self.window_size
            win_w = self.window_size
        else:
            grid = int(self.num_port ** 0.5)
            win_h = H // grid
            win_w = W // grid

        # geometry_feature: [B, C*num_port, win_h, win_w]
        geometry_feature = (
            x.reshape(B, C, H // win_h, win_h, W // win_w, win_w)
             .permute(0, 2, 4, 1, 3, 5)
             .flatten(1, 3)
        )

        # area_gate: [B,C,H,W]
        area_gate = (
            self.area_map(geometry_feature)
            .reshape(B, H // win_h, W // win_w, C, win_h, win_w)
            .permute(0, 3, 1, 4, 2, 5)
            .flatten(2, 3)
            .flatten(-2, -1)
        )

        out = kv * area_gate * channel_gate
        out = self.layer_scale[:, None, None] * self.proj(out)
        out = self.drop_path(out)
        out = out.contiguous()
        return out + skip


class FocusBlock(nn.Module):
    """
    一个 Focus Block：
    - 先做 WindowFocus（门控/注意力融合）
    - 再做 ConvMlp（FFN）
    注意：padding 仅用于确保 H/W 可被 window_size 整除，最后裁剪回原尺寸
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        image_size: int,
        window_size: int = 7,
        kernel_size: int = 11,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        drop_path_prob: float = 0.0,
    ):
        super().__init__()
        self.window_size = int(window_size)

        self.attn = WindowFocus(
            image_size=image_size,
            dim=dim,
            window_size=self.window_size,
            kernel_size=kernel_size,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            proj_drop=drop_path_prob,
        )

        hidden_dim = int(dim * mlp_ratio)
        self.mlp = ConvMlp(dim=dim, hidden_dim=hidden_dim, drop_path_prob=drop_path_prob)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape

        pad_r = (self.window_size - W % self.window_size) % self.window_size
        pad_b = (self.window_size - H % self.window_size) % self.window_size
        if pad_r > 0 or pad_b > 0:
            # NCHW pad 顺序: (left, right, top, bottom)
            x = F.pad(x, (0, pad_r, 0, pad_b))

        x = self.attn(x)

        if pad_r > 0 or pad_b > 0:
            x = x[:, :, :H, :W].contiguous()

        x = self.mlp(x)
        return x


class BasicLayer(nn.Module):
    """
    一个 stage：
    - 堆叠 depth 个 FocusBlock
    - 可选下采样（PatchMergingConv1 / PatchMerging）
    - 可选 checkpoint 节省显存
    """

    def __init__(
        self,
        dim: int,
        depth: int,
        num_heads: int,
        window_size: int,
        image_size: int,
        kernel_size: int,
        mlp_ratio: float,
        qkv_bias: bool,
        drop_path_probs,
        downsample,
        use_checkpoint: bool,
    ):
        super().__init__()
        self.use_checkpoint = bool(use_checkpoint)

        self.blocks = nn.ModuleList([
            FocusBlock(
                dim=dim,
                num_heads=num_heads,
                image_size=image_size,
                window_size=window_size,
                kernel_size=kernel_size,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop_path_prob=drop_path_probs[i] if isinstance(drop_path_probs, list) else float(drop_path_probs),
            )
            for i in range(int(depth))
        ])

        self.downsample = downsample(dim) if downsample is not None else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for blk in self.blocks:
            if (not torch.jit.is_scripting()) and self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x)
            else:
                x = blk(x)

        if self.downsample is not None:
            x = self.downsample(x)
        return x


class FocusNet(nn.Module):
    """
    FocusNet Backbone（你的主网络）：
    PatchEmbed -> 多个 stage(BasicLayer) -> LN -> GAP -> FC
    """

    def __init__(
        self,
        image_size: int = 224,
        patch_size: int = 4,
        in_chans: int = 3,
        num_classes: int = 1000,
        embed_dim: int = 96,
        depths=(2, 2, 6, 2),
        num_heads=(3, 6, 12, 24),
        window_size: int = 7,
        kernel_size: int = 11,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        drop_rate: float = 0.0,        # 注意：此处 drop_rate 在原实现中被 DropPath 使用（非标准 dropout）
        drop_path_rate: float = 0.1,
        patch_norm: bool = True,
        use_checkpoint: bool = False,
        **kwargs
    ):
        super().__init__()
        self.num_classes = int(num_classes)
        self.num_layers = len(depths)
        self.embed_dim = int(embed_dim)
        self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))

        # 你的原实现实际用的是 PatchEmbed_guorun_conv1，这里保持一致
        self.patch_embed = PatchEmbedConv1(
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            use_norm=patch_norm,
        )

        # 与原实现保持行为一致：用 DropPath 作为 pos_drop
        self.pos_drop = DropPath(drop_rate) if drop_rate > 0.0 else nn.Identity()

        # stochastic depth：线性递增
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        # build stages
        image_size_layer = image_size // patch_size
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = BasicLayer(
                dim=int(embed_dim * 2 ** i_layer),
                depth=depths[i_layer],
                num_heads=num_heads[i_layer],
                window_size=window_size,
                image_size=image_size_layer,
                kernel_size=kernel_size,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop_path_probs=dpr[sum(depths[:i_layer]): sum(depths[:i_layer + 1])],
                downsample=PatchMergingConv1 if (i_layer < self.num_layers - 1) else None,
                use_checkpoint=use_checkpoint,
            )
            self.layers.append(layer)
            image_size_layer //= 2

        self.norm = LayerNorm(self.num_features, eps=1e-6)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Linear(self.num_features, self.num_classes) if self.num_classes > 0 else nn.Identity()

        self.apply(self._init_weights)

    def _init_weights(self, m: nn.Module):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patch_embed(x)
        x = self.pos_drop(x)

        for layer in self.layers:
            x = layer(x)

        x = self.norm(x)                       # [B,C,H,W]
        x = self.avgpool(x.flatten(-2, -1))     # [B,C,1]
        x = torch.flatten(x, 1)                # [B,C]
        return self.head(x)


# -------------------------
# 预设配置（保留你原先的使用习惯，但全部指向 FocusNet，避免 FocusEnd 未定义）
# -------------------------

@register_model
def focusnet_tiny_224(pretrained=False, **kwargs) -> FocusNet:
    return FocusNet(
        patch_size=4,
        window_size=7,
        embed_dim=96,
        depths=(2, 2, 8, 2),
        num_heads=(3, 6, 12, 24),
        **kwargs,
    )


def focusnet_debug(num_classes: int = 1000, **kwargs) -> FocusNet:
    # 对应你原来的 test() 设定（仅用于调试/实验）
    return FocusNet(
        image_size=224,
        in_chans=3,
        patch_size=4,
        window_size=7,
        kernel_size=15,
        embed_dim=96,
        depths=(3, 3, 27, 3),
        num_heads=(3, 6, 12, 24),
        num_classes=num_classes,
        drop_path_rate=0.15,
        **kwargs,
    )


if __name__ == "__main__":
    print("torch:", torch.__version__)
    net = focusnet_debug().cuda().eval()
    print(net)

    image = torch.rand(1, 3, 224, 224).cuda()

    # 可选：thop 复杂度统计（不作为依赖）
    try:
        from thop import profile
        flops, params = profile(net, inputs=(image,))
        print(f"FLOPs: {flops / 1e9:.2f} G, Params: {params / 1e6:.2f} M")
    except Exception as e:
        print(f"[Skip profiling] thop not available or failed: {e}")

    with torch.no_grad():
        t0 = time.time()
        out = net(image)
        dt = time.time() - t0

    print("Output:", out.shape)
    print("Infer time (s):", dt)
    if dt > 0:
        print("FPS:", 1.0 / dt)
