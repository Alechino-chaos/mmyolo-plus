# Copyright (c) OpenMMLab. All rights reserved.
"""
YOLOv12-specific building blocks.

This module implements the core innovations of YOLOv12:
- AAttn: Area-Attention mechanism for efficient spatial attention
- ABlock: Area Attention Block combining AAttn + MLP
- A2C2f: R-ELAN block with residual area-attention feature extraction
- C3k2: Enhanced CSP Bottleneck with optional C3k inner modules
- C3k: CSP Bottleneck with customizable kernel sizes
"""

import logging
from typing import Optional

import torch
import torch.nn as nn
from mmcv.cnn import ConvModule
from mmengine.model import BaseModule
from torch import Tensor

from mmyolo.registry import MODELS

logger = logging.getLogger(__name__)

# Flash Attention availability check
USE_FLASH_ATTN = False
try:
    if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8:
        from flash_attn.flash_attn_interface import flash_attn_func
        USE_FLASH_ATTN = True
    else:
        from torch.nn.functional import scaled_dot_product_attention as sdpa
        logger.warning(
            'FlashAttention is not available on this device. '
            'Using scaled_dot_product_attention instead.')
except Exception:
    from torch.nn.functional import scaled_dot_product_attention as sdpa
    logger.warning(
        'FlashAttention is not available. '
        'Using scaled_dot_product_attention instead.')


@MODELS.register_module()
class AAttn(BaseModule):
    """Area-Attention module with Flash Attention support.

    Efficient attention mechanism that divides the feature map into areas
    for localized attention computation, reducing complexity while
    maintaining global receptive field.

    Args:
        dim (int): Number of input channels.
        num_heads (int): Number of attention heads. dim // num_heads
            should be a multiple of 32.
        area (int): Number of areas to divide the feature map into.
            Defaults to 1 (no division).
        init_cfg (dict or list[dict], optional): Initialization config.
            Defaults to None.
    """

    def __init__(self,
                 dim: int,
                 num_heads: int,
                 area: int = 1,
                 norm_cfg: dict = dict(
                     type='BN', momentum=0.03, eps=0.001),
                 init_cfg: Optional[dict] = None):
        super().__init__(init_cfg)
        self.area = area
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        all_head_dim = self.head_dim * self.num_heads

        self.qk = ConvModule(
            dim,
            all_head_dim * 2,
            kernel_size=1,
            norm_cfg=norm_cfg,
            act_cfg=None)
        self.v = ConvModule(
            dim,
            all_head_dim,
            kernel_size=1,
            norm_cfg=norm_cfg,
            act_cfg=None)
        self.proj = ConvModule(
            all_head_dim,
            dim,
            kernel_size=1,
            norm_cfg=norm_cfg,
            act_cfg=None)
        # Position encoding via depthwise conv
        self.pe = ConvModule(
            all_head_dim,
            dim,
            kernel_size=5,
            stride=1,
            padding=2,
            groups=dim,
            norm_cfg=norm_cfg,
            act_cfg=None)

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass through area-attention.

        Args:
            x (Tensor): Input tensor of shape (B, C, H, W).

        Returns:
            Tensor: Output tensor of shape (B, C, H, W).
        """
        B, C, H, W = x.shape
        N = H * W

        qk = self.qk(x).flatten(2).transpose(1, 2)
        v = self.v(x)
        pp = self.pe(v)
        v = v.flatten(2).transpose(1, 2)

        if self.area > 1:
            qk = qk.reshape(B * self.area, N // self.area, C * 2)
            v = v.reshape(B * self.area, N // self.area, C)
            B, N, _ = qk.shape

        q, k = qk.split([C, C], dim=2)

        if x.is_cuda and USE_FLASH_ATTN:
            q = q.view(B, N, self.num_heads, self.head_dim)
            k = k.view(B, N, self.num_heads, self.head_dim)
            v = v.view(B, N, self.num_heads, self.head_dim)

            x_out = flash_attn_func(
                q.contiguous().half(),
                k.contiguous().half(),
                v.contiguous().half()).to(q.dtype)
        else:
            q = q.transpose(1, 2).view(
                B, self.num_heads, self.head_dim, N)
            k = k.transpose(1, 2).view(
                B, self.num_heads, self.head_dim, N)
            v = v.transpose(1, 2).view(
                B, self.num_heads, self.head_dim, N)

            attn = (q.transpose(-2, -1) @ k) * (self.head_dim ** -0.5)
            max_attn = attn.max(dim=-1, keepdim=True).values
            exp_attn = torch.exp(attn - max_attn)
            attn = exp_attn / exp_attn.sum(dim=-1, keepdim=True)
            x_out = (v @ attn.transpose(-2, -1))
            x_out = x_out.permute(0, 3, 1, 2)

        if self.area > 1:
            x_out = x_out.reshape(B // self.area, N * self.area, C)
            B, N, _ = x_out.shape

        x_out = x_out.reshape(B, H, W, C).permute(0, 3, 1, 2)
        return self.proj(x_out + pp)


@MODELS.register_module()
class ABlock(BaseModule):
    """Area-Attention Block combining AAttn and MLP.

    Applies area-attention followed by a feed-forward MLP, both with
    residual connections.

    Args:
        dim (int): Number of input channels.
        num_heads (int): Number of attention heads.
        mlp_ratio (float): Expansion ratio for MLP hidden dimension.
            Defaults to 1.2.
        area (int): Number of areas for area-attention. Defaults to 1.
        init_cfg (dict or list[dict], optional): Initialization config.
            Defaults to None.
    """

    def __init__(self,
                 dim: int,
                 num_heads: int,
                 mlp_ratio: float = 1.2,
                 area: int = 1,
                 norm_cfg: dict = dict(
                     type='BN', momentum=0.03, eps=0.001),
                 act_cfg: dict = dict(type='SiLU', inplace=True),
                 init_cfg: Optional[dict] = None):
        super().__init__(init_cfg)
        self.attn = AAttn(
            dim, num_heads=num_heads, area=area, norm_cfg=norm_cfg)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            ConvModule(
                dim, mlp_hidden_dim, kernel_size=1,
                norm_cfg=norm_cfg, act_cfg=act_cfg),
            ConvModule(
                mlp_hidden_dim, dim, kernel_size=1,
                norm_cfg=norm_cfg, act_cfg=None))
        if self.init_cfg is None:
            self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m):
        """Initialize convolution weights like the official ABlock."""
        if isinstance(m, nn.Conv2d):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def init_weights(self):
        """Initialize weights with truncated normal distribution."""
        super().init_weights()
        if self.init_cfg is None:
            self.apply(self._init_weights)

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass through ABlock.

        Args:
            x (Tensor): Input tensor.

        Returns:
            Tensor: Output tensor.
        """
        x = x + self.attn(x)
        x = x + self.mlp(x)
        return x


@MODELS.register_module()
class C3k(BaseModule):
    """CSP Bottleneck with customizable kernel sizes for YOLOv12.

    Args:
        in_channels (int): Input channels.
        out_channels (int): Output channels.
        num_blocks (int): Number of Bottleneck blocks. Defaults to 2.
        shortcut (bool): Whether to use shortcut in Bottleneck.
            Defaults to True.
        groups (int): Groups for grouped convolution. Defaults to 1.
        expansion (float): Expansion ratio for hidden channels.
            Defaults to 0.5.
        kernel_size (int or tuple): Kernel size for Bottleneck conv.
            Defaults to 3.
        norm_cfg (dict): Normalization config.
            Defaults to dict(type='BN', momentum=0.03, eps=0.001).
        act_cfg (dict): Activation config.
            Defaults to dict(type='SiLU', inplace=True).
        init_cfg (dict or list[dict], optional): Initialization config.
            Defaults to None.
    """

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 num_blocks: int = 2,
                 shortcut: bool = True,
                 groups: int = 1,
                 expansion: float = 0.5,
                 kernel_size: int = 3,
                 norm_cfg: dict = dict(
                     type='BN', momentum=0.03, eps=0.001),
                 act_cfg: dict = dict(type='SiLU', inplace=True),
                 init_cfg: Optional[dict] = None):
        super().__init__(init_cfg)
        hidden_channels = int(out_channels * expansion)

        self.cv1 = ConvModule(
            in_channels,
            hidden_channels,
            kernel_size=1,
            stride=1,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)
        self.cv2 = ConvModule(
            in_channels,
            hidden_channels,
            kernel_size=1,
            stride=1,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)
        self.cv3 = ConvModule(
            hidden_channels * 2,
            out_channels,
            kernel_size=1,
            stride=1,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)

        self.m = nn.Sequential(*[
            DarknetBottleneckK(
                hidden_channels,
                hidden_channels,
                shortcut=shortcut,
                groups=groups,
                kernel_size=kernel_size,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg)
            for _ in range(num_blocks)
        ])

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass."""
        y1 = self.cv1(x)
        y2 = self.cv2(x)
        y2 = self.m(y2)
        return self.cv3(torch.cat([y1, y2], dim=1))


class DarknetBottleneckK(BaseModule):
    """Darknet Bottleneck with customizable kernel size.

    Args:
        in_channels (int): Input channels.
        out_channels (int): Output channels.
        shortcut (bool): Whether to use residual connection.
            Defaults to True.
        groups (int): Groups for grouped convolution.
            Defaults to 1.
        expansion (float): Expansion ratio. Defaults to 1.0.
        kernel_size (int or tuple): Kernel size. Defaults to 3.
        norm_cfg (dict): Normalization config.
        act_cfg (dict): Activation config.
        init_cfg (dict, optional): Initialization config.
    """

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 shortcut: bool = True,
                 groups: int = 1,
                 expansion: float = 1.0,
                 kernel_size: int = 3,
                 norm_cfg: dict = dict(
                     type='BN', momentum=0.03, eps=0.001),
                 act_cfg: dict = dict(type='SiLU', inplace=True),
                 init_cfg: Optional[dict] = None):
        super().__init__(init_cfg)
        hidden_channels = int(out_channels * expansion)

        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)

        self.cv1 = ConvModule(
            in_channels,
            hidden_channels,
            kernel_size=kernel_size[0],
            stride=1,
            padding=kernel_size[0] // 2,
            groups=groups,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)
        self.cv2 = ConvModule(
            hidden_channels,
            out_channels,
            kernel_size=kernel_size[1],
            stride=1,
            padding=kernel_size[1] // 2,
            groups=groups,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)
        self.add_shortcut = shortcut and in_channels == out_channels

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass."""
        return x + self.cv2(self.cv1(x)) if self.add_shortcut else self.cv2(
            self.cv1(x))


@MODELS.register_module()
class C3k2(BaseModule):
    """Enhanced CSP Bottleneck with 2 convolutions for YOLOv12.

    Uses C3k blocks or standard Bottleneck as inner modules.

    Args:
        in_channels (int): Input channels.
        out_channels (int): Output channels.
        num_blocks (int): Number of inner blocks. Defaults to 1.
        use_c3k (bool): Whether to use C3k inner blocks. If False,
            uses standard Bottleneck. Defaults to False.
        expansion (float): Expansion ratio. Defaults to 0.5.
        groups (int): Groups for grouped convolution. Defaults to 1.
        shortcut (bool): Whether to use shortcut. Defaults to True.
        norm_cfg (dict): Normalization config.
            Defaults to dict(type='BN', momentum=0.03, eps=0.001).
        act_cfg (dict): Activation config.
            Defaults to dict(type='SiLU', inplace=True).
        init_cfg (dict or list[dict], optional): Initialization config.
            Defaults to None.
    """

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 num_blocks: int = 1,
                 use_c3k: bool = False,
                 expansion: float = 0.5,
                 groups: int = 1,
                 shortcut: bool = True,
                 norm_cfg: dict = dict(
                     type='BN', momentum=0.03, eps=0.001),
                 act_cfg: dict = dict(type='SiLU', inplace=True),
                 init_cfg: Optional[dict] = None):
        super().__init__(init_cfg)
        hidden_channels = int(out_channels * expansion)

        self.cv1 = ConvModule(
            in_channels,
            hidden_channels,
            kernel_size=1,
            stride=1,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)
        self.cv2 = ConvModule(
            in_channels,
            hidden_channels,
            kernel_size=1,
            stride=1,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)
        self.cv3 = ConvModule(
            (2 + num_blocks) * hidden_channels,
            out_channels,
            kernel_size=1,
            stride=1,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)

        if use_c3k:
            self.m = nn.ModuleList([
                C3k(
                    hidden_channels,
                    hidden_channels,
                    num_blocks=2,
                    shortcut=shortcut,
                    groups=groups,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg)
                for _ in range(num_blocks)
            ])
        else:
            self.m = nn.ModuleList([
                DarknetBottleneckK(
                    hidden_channels,
                    hidden_channels,
                    shortcut=shortcut,
                    groups=groups,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg)
                for _ in range(num_blocks)
            ])

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass."""
        y = [self.cv2(x), self.cv1(x)]
        y.extend(m(y[-1]) for m in self.m)
        return self.cv3(torch.cat(y, 1))


@MODELS.register_module()
class A2C2f(BaseModule):
    """R-ELAN block with Area-Attention for YOLOv12.

    Core innovation of YOLOv12: Residual Efficient Layer Aggregation
    Network with area-based attention mechanisms.

    Args:
        in_channels (int): Input channels.
        out_channels (int): Output channels.
        num_ablocks (int): Number of ABlock/C3k stacks.
            Defaults to 1.
        use_area_attn (bool): Whether to use area-attention (ABlock).
            If False, uses C3k instead. Defaults to True.
        area (int): Number of areas for area-attention.
            Defaults to 1.
        use_residual (bool): Whether to use learnable layer-scale
            residual connection. Defaults to False.
        mlp_ratio (float): MLP expansion ratio for ABlock.
            Defaults to 2.0.
        expansion (float): Channel expansion ratio.
            Defaults to 0.5.
        groups (int): Groups for grouped convolution.
            Defaults to 1.
        shortcut (bool): Whether to use shortcut in C3k fallback.
            Defaults to True.
        norm_cfg (dict): Normalization config.
            Defaults to dict(type='BN', momentum=0.03, eps=0.001).
        act_cfg (dict): Activation config.
            Defaults to dict(type='SiLU', inplace=True).
        init_cfg (dict or list[dict], optional): Initialization config.
            Defaults to None.
    """

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 num_ablocks: int = 1,
                 use_area_attn: bool = True,
                 area: int = 1,
                 use_residual: bool = False,
                 mlp_ratio: float = 2.0,
                 expansion: float = 0.5,
                 groups: int = 1,
                 shortcut: bool = True,
                 norm_cfg: dict = dict(
                     type='BN', momentum=0.03, eps=0.001),
                 act_cfg: dict = dict(type='SiLU', inplace=True),
                 init_cfg: Optional[dict] = None):
        super().__init__(init_cfg)
        hidden_channels = int(out_channels * expansion)
        assert hidden_channels % 32 == 0, \
            f'Hidden channels ({hidden_channels}) must be multiple of 32.'

        num_heads = hidden_channels // 32

        self.cv1 = ConvModule(
            in_channels,
            hidden_channels,
            kernel_size=1,
            stride=1,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)
        self.cv2 = ConvModule(
            (1 + num_ablocks) * hidden_channels,
            out_channels,
            kernel_size=1,
            stride=1,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)

        # Learnable layer-scale for residual
        if use_area_attn and use_residual:
            init_values = 0.01
            self.gamma = nn.Parameter(
                init_values * torch.ones((out_channels)), requires_grad=True)
        else:
            self.gamma = None

        # Build inner blocks
        self.m = nn.ModuleList()
        for _ in range(num_ablocks):
            if use_area_attn:
                self.m.append(
                    nn.Sequential(
                        ABlock(
                            hidden_channels, num_heads, mlp_ratio, area,
                            norm_cfg=norm_cfg, act_cfg=act_cfg),
                        ABlock(
                            hidden_channels, num_heads, mlp_ratio, area,
                            norm_cfg=norm_cfg, act_cfg=act_cfg)))
            else:
                self.m.append(
                    C3k(
                        hidden_channels,
                        hidden_channels,
                        num_blocks=2,
                        shortcut=shortcut,
                        groups=groups,
                        norm_cfg=norm_cfg,
                        act_cfg=act_cfg))

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass through R-ELAN layer.

        Args:
            x (Tensor): Input tensor.

        Returns:
            Tensor: Output tensor.
        """
        y = [self.cv1(x)]
        y.extend(m(y[-1]) for m in self.m)

        if self.gamma is not None:
            return x + self.gamma.view(1, -1, 1, 1) * self.cv2(
                torch.cat(y, 1))
        return self.cv2(torch.cat(y, 1))
