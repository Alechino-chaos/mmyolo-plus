# Copyright (c) OpenMMLab. All rights reserved.
"""YOLOv12 PAFPN neck.

Uses A2C2f blocks in C3k fallback mode for top-down and bottom-up
fusion pathways, with C3k2 for the final P5/32-large block.
The down-sampling is done via Conv (k=3, s=2) as in the original
YOLOv12 implementation.
"""

from typing import List, Union

import torch
import torch.nn as nn
from mmcv.cnn import ConvModule
from mmdet.utils import ConfigType, OptMultiConfig

from mmyolo.registry import MODELS
from ..layers.yolov12_bricks import A2C2f, C3k2
from ..utils import make_divisible, make_round
from .yolov5_pafpn import YOLOv5PAFPN


def _make_divisible_max(x: float,
                        widen_factor: float,
                        divisor: int = 8,
                        max_channels: int = 1024) -> int:
    """Scale channels with YOLOv12/Ultralytics max-channel behavior."""
    return make_divisible(min(x, max_channels), widen_factor, divisor)


@MODELS.register_module()
class YOLOv12PAFPN(YOLOv5PAFPN):
    """Path Aggregation Network used in YOLOv12.

    Args:
        in_channels (List[int]): Number of input channels per scale.
        out_channels (Union[List[int], int]): Number of output channels.
        deepen_factor (float): Depth multiplier. Defaults to 1.0.
        widen_factor (float): Width multiplier. Defaults to 1.0.
        num_csp_blocks (int): Number of blocks per fusion layer.
            Defaults to 2.
        max_channels (int): Maximum base channel count before widening.
            Defaults to 1024.
        freeze_all (bool): Whether to freeze the model.
        norm_cfg (dict): Normalization config.
        act_cfg (dict): Activation config.
        init_cfg (dict or list[dict], optional): Initialization config.
    """

    def __init__(self,
                 in_channels: List[int],
                 out_channels: Union[List[int], int],
                 deepen_factor: float = 1.0,
                 widen_factor: float = 1.0,
                 num_csp_blocks: int = 2,
                 max_channels: int = 1024,
                 use_residual: bool = False,
                 mlp_ratio: float = 2.0,
                 freeze_all: bool = False,
                 norm_cfg: ConfigType = dict(
                     type='BN', momentum=0.03, eps=0.001),
                 act_cfg: ConfigType = dict(type='SiLU', inplace=True),
                 init_cfg: OptMultiConfig = None):
        self.max_channels = max_channels
        self.use_residual = use_residual
        self.mlp_ratio = mlp_ratio
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            deepen_factor=deepen_factor,
            widen_factor=widen_factor,
            num_csp_blocks=num_csp_blocks,
            freeze_all=freeze_all,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg,
            init_cfg=init_cfg)

    def build_reduce_layer(self, idx: int) -> nn.Module:
        """Build reduce layer (identity for YOLOv12)."""
        return nn.Identity()

    def build_upsample_layer(self, idx: int) -> nn.Module:
        """Build upsample layer (nearest neighbor 2x)."""
        return nn.Upsample(scale_factor=2, mode='nearest')

    def build_top_down_layer(self, idx: int) -> nn.Module:
        """Build top-down fusion layer using A2C2f (no area-attn).

        Uses A2C2f with use_area_attn=False (falls back to C3k blocks).
        """
        in_ch = _make_divisible_max(
            self.in_channels[idx - 1], self.widen_factor,
            max_channels=self.max_channels) + _make_divisible_max(
                self.in_channels[idx], self.widen_factor,
                max_channels=self.max_channels)
        out_ch = _make_divisible_max(
            self.out_channels[idx - 1], self.widen_factor,
            max_channels=self.max_channels)

        return A2C2f(
            in_ch,
            out_ch,
            num_ablocks=make_round(self.num_csp_blocks, self.deepen_factor),
            use_area_attn=False,
            area=-1,
            use_residual=self.use_residual,
            mlp_ratio=self.mlp_ratio,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)

    def build_downsample_layer(self, idx: int) -> nn.Module:
        """Build downsample layer (Conv k=3, s=2).

        YOLOv12 uses standard Conv for down-sampling in the neck.
        """
        channels = _make_divisible_max(
            self.out_channels[idx], self.widen_factor,
            max_channels=self.max_channels)
        return ConvModule(
            channels,
            channels,
            kernel_size=3,
            stride=2,
            padding=1,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)

    def build_bottom_up_layer(self, idx: int) -> nn.Module:
        """Build bottom-up fusion layer.

        Uses A2C2f for most layers, C3k2 for the final P5/32-large block.
        """
        out_ch = _make_divisible_max(
            self.out_channels[idx + 1], self.widen_factor,
            max_channels=self.max_channels)
        in_ch = _make_divisible_max(
            self.out_channels[idx], self.widen_factor,
            max_channels=self.max_channels) + _make_divisible_max(
                self.out_channels[idx + 1], self.widen_factor,
                max_channels=self.max_channels)

        # Last bottom-up layer uses C3k2 (P5/32-large)
        if idx == len(self.out_channels) - 2:
            return C3k2(
                in_ch,
                out_ch,
                num_blocks=make_round(
                    self.num_csp_blocks, self.deepen_factor),
                use_c3k=True,
                shortcut=True,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg)

        return A2C2f(
            in_ch,
            out_ch,
            num_ablocks=make_round(self.num_csp_blocks, self.deepen_factor),
            use_area_attn=False,
            area=-1,
            use_residual=self.use_residual,
            mlp_ratio=self.mlp_ratio,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)

    def init_weights(self):
        """Initialize the parameters."""
        if self.init_cfg is None:
            for m in self.modules():
                if isinstance(m, torch.nn.Conv2d):
                    m.reset_parameters()
        else:
            super().init_weights()
