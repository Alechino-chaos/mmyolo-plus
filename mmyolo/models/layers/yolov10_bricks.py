# Copyright (c) OpenMMLab. All rights reserved.
"""YOLOv10 building blocks for MMYOLO rotated experiments."""

from typing import Optional

import torch
import torch.nn as nn
from mmcv.cnn import ConvModule
from mmengine.model import BaseModule
from torch import Tensor

from mmyolo.registry import MODELS


@MODELS.register_module()
class YOLOv10CIB(BaseModule):
    """Compact inverted block used by YOLOv10 C2fCIB layers."""

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 shortcut: bool = True,
                 expansion: float = 0.5,
                 use_large_kernel: bool = False,
                 norm_cfg: dict = dict(
                     type='BN', momentum=0.03, eps=0.001),
                 act_cfg: dict = dict(type='SiLU', inplace=True),
                 init_cfg: Optional[dict] = None):
        super().__init__(init_cfg)
        hidden_channels = int(out_channels * expansion)
        mid_channels = hidden_channels * 2
        dw_kernel = 7 if use_large_kernel else 3

        self.block = nn.Sequential(
            ConvModule(
                in_channels,
                in_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                groups=in_channels,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg),
            ConvModule(
                in_channels,
                mid_channels,
                kernel_size=1,
                stride=1,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg),
            ConvModule(
                mid_channels,
                mid_channels,
                kernel_size=dw_kernel,
                stride=1,
                padding=dw_kernel // 2,
                groups=mid_channels,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg),
            ConvModule(
                mid_channels,
                out_channels,
                kernel_size=1,
                stride=1,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg),
            ConvModule(
                out_channels,
                out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                groups=out_channels,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg))
        self.add_shortcut = shortcut and in_channels == out_channels

    def forward(self, x: Tensor) -> Tensor:
        out = self.block(x)
        return x + out if self.add_shortcut else out


@MODELS.register_module()
class YOLOv10C2fCIB(BaseModule):
    """C2f layer with YOLOv10 CIB inner blocks."""

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 num_blocks: int = 1,
                 shortcut: bool = True,
                 expansion: float = 0.5,
                 use_large_kernel: bool = False,
                 norm_cfg: dict = dict(
                     type='BN', momentum=0.03, eps=0.001),
                 act_cfg: dict = dict(type='SiLU', inplace=True),
                 init_cfg: Optional[dict] = None):
        super().__init__(init_cfg)
        self.mid_channels = int(out_channels * expansion)
        self.main_conv = ConvModule(
            in_channels,
            2 * self.mid_channels,
            kernel_size=1,
            stride=1,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)
        self.blocks = nn.ModuleList([
            YOLOv10CIB(
                self.mid_channels,
                self.mid_channels,
                shortcut=shortcut,
                expansion=1.0,
                use_large_kernel=use_large_kernel,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg) for _ in range(num_blocks)
        ])
        self.final_conv = ConvModule(
            (2 + num_blocks) * self.mid_channels,
            out_channels,
            kernel_size=1,
            stride=1,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)

    def forward(self, x: Tensor) -> Tensor:
        parts = list(self.main_conv(x).split(
            (self.mid_channels, self.mid_channels), 1))
        parts.extend(block(parts[-1]) for block in self.blocks)
        return self.final_conv(torch.cat(parts, 1))


@MODELS.register_module()
class YOLOv10SCDown(BaseModule):
    """Spatial-channel downsampling block used in YOLOv10."""

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int = 3,
                 norm_cfg: dict = dict(
                     type='BN', momentum=0.03, eps=0.001),
                 act_cfg: dict = dict(type='SiLU', inplace=True),
                 init_cfg: Optional[dict] = None):
        super().__init__(init_cfg)
        self.pointwise = ConvModule(
            in_channels,
            out_channels,
            kernel_size=1,
            stride=1,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)
        self.depthwise = ConvModule(
            out_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=2,
            padding=kernel_size // 2,
            groups=out_channels,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)

    def forward(self, x: Tensor) -> Tensor:
        return self.depthwise(self.pointwise(x))


class YOLOv10Attention(BaseModule):
    """Lightweight spatial self-attention used inside YOLOv10 PSA."""

    def __init__(self,
                 channels: int,
                 num_heads: int,
                 norm_cfg: dict = dict(
                     type='BN', momentum=0.03, eps=0.001),
                 init_cfg: Optional[dict] = None):
        super().__init__(init_cfg)
        self.num_heads = max(num_heads, 1)
        self.head_dim = channels // self.num_heads
        self.qkv = ConvModule(
            channels,
            channels * 3,
            kernel_size=1,
            norm_cfg=norm_cfg,
            act_cfg=None)
        self.proj = ConvModule(
            channels,
            channels,
            kernel_size=1,
            norm_cfg=norm_cfg,
            act_cfg=None)
        self.pe = ConvModule(
            channels,
            channels,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=channels,
            norm_cfg=norm_cfg,
            act_cfg=None)

    def forward(self, x: Tensor) -> Tensor:
        b, c, h, w = x.shape
        qkv = self.qkv(x).reshape(b, 3, self.num_heads, self.head_dim, h * w)
        q, k, v = qkv[:, 0], qkv[:, 1], qkv[:, 2]
        attn = torch.matmul(q.transpose(-2, -1), k) * (self.head_dim**-0.5)
        attn = attn.softmax(dim=-1)
        out = torch.matmul(v, attn.transpose(-2, -1))
        out = out.reshape(b, c, h, w)
        return self.proj(out + self.pe(x))


@MODELS.register_module()
class YOLOv10PSA(BaseModule):
    """Partial self-attention block used at the end of YOLOv10 backbone."""

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 expansion: float = 0.5,
                 norm_cfg: dict = dict(
                     type='BN', momentum=0.03, eps=0.001),
                 act_cfg: dict = dict(type='SiLU', inplace=True),
                 init_cfg: Optional[dict] = None):
        super().__init__(init_cfg)
        assert in_channels == out_channels
        hidden_channels = int(in_channels * expansion)
        self.pre_conv = ConvModule(
            in_channels,
            hidden_channels * 2,
            kernel_size=1,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)
        self.attn = YOLOv10Attention(
            hidden_channels,
            num_heads=max(hidden_channels // 64, 1),
            norm_cfg=norm_cfg)
        self.ffn = nn.Sequential(
            ConvModule(
                hidden_channels,
                hidden_channels * 2,
                kernel_size=1,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg),
            ConvModule(
                hidden_channels * 2,
                hidden_channels,
                kernel_size=1,
                norm_cfg=norm_cfg,
                act_cfg=None))
        self.post_conv = ConvModule(
            hidden_channels * 2,
            out_channels,
            kernel_size=1,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)

    def forward(self, x: Tensor) -> Tensor:
        a, b = self.pre_conv(x).chunk(2, 1)
        b = b + self.attn(b)
        b = b + self.ffn(b)
        return self.post_conv(torch.cat((a, b), 1))
