# Copyright (c) OpenMMLab. All rights reserved.
"""YOLOv10 backbone for MMYOLO."""

from typing import List, Tuple, Union

import torch
import torch.nn as nn
from mmcv.cnn import ConvModule
from mmdet.utils import ConfigType, OptMultiConfig

from mmyolo.registry import MODELS
from ..layers import CSPLayerWithTwoConv, SPPFBottleneck
from ..layers.yolov10_bricks import YOLOv10C2fCIB, YOLOv10PSA, YOLOv10SCDown
from ..utils import make_divisible, make_round
from .base_backbone import BaseBackbone


def make_yolov10_channel(channel: int,
                         widen_factor: float,
                         max_channels: int,
                         divisor: int = 8) -> int:
    """Apply YOLOv10 width and max-channel scaling."""
    return make_divisible(min(channel, max_channels), widen_factor, divisor)


@MODELS.register_module()
class YOLOv10CSPDarknet(BaseBackbone):
    """YOLOv10 CSP-Darknet backbone.

    The structure follows the official YOLOv10 P5 backbone and exposes the
    standard P3/P4/P5 feature maps for downstream rotated heads.
    """

    # in_channels, out_channels, num_blocks, use_scdown, block_type
    arch_settings = {
        'P5': [[64, 128, 3, False, 'c2f'], [128, 256, 6, False, 'c2f'],
               [256, 512, 6, True, 'c2f'],
               [512, 1024, 3, True, 'c2fcib']]
    }

    def __init__(self,
                 arch: str = 'P5',
                 deepen_factor: float = 1.0,
                 widen_factor: float = 1.0,
                 max_channels: int = 1024,
                 input_channels: int = 3,
                 out_indices: Tuple[int] = (2, 3, 4),
                 frozen_stages: int = -1,
                 plugins: Union[dict, List[dict]] = None,
                 use_large_kernel: bool = False,
                 norm_cfg: ConfigType = dict(
                     type='BN', momentum=0.03, eps=0.001),
                 act_cfg: ConfigType = dict(type='SiLU', inplace=True),
                 norm_eval: bool = False,
                 init_cfg: OptMultiConfig = None):
        self.max_channels = max_channels
        self.use_large_kernel = use_large_kernel
        super().__init__(
            self.arch_settings[arch],
            deepen_factor=deepen_factor,
            widen_factor=widen_factor,
            input_channels=input_channels,
            out_indices=out_indices,
            frozen_stages=frozen_stages,
            plugins=plugins,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg,
            norm_eval=norm_eval,
            init_cfg=init_cfg)

    def _make_channel(self, channel: int) -> int:
        return make_yolov10_channel(channel, self.widen_factor,
                                    self.max_channels)

    def build_stem_layer(self) -> nn.Module:
        return ConvModule(
            self.input_channels,
            self._make_channel(self.arch_setting[0][0]),
            kernel_size=3,
            stride=2,
            padding=1,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)

    def build_stage_layer(self, stage_idx: int, setting: list) -> list:
        in_channels, out_channels, num_blocks, use_scdown, block_type = setting
        in_channels = self._make_channel(in_channels)
        out_channels = self._make_channel(out_channels)
        num_blocks = make_round(num_blocks, self.deepen_factor)

        downsample_cls = YOLOv10SCDown if use_scdown else ConvModule
        if use_scdown:
            downsample = downsample_cls(
                in_channels,
                out_channels,
                kernel_size=3,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg)
        else:
            downsample = downsample_cls(
                in_channels,
                out_channels,
                kernel_size=3,
                stride=2,
                padding=1,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg)

        if block_type == 'c2fcib':
            csp = YOLOv10C2fCIB(
                out_channels,
                out_channels,
                num_blocks=num_blocks,
                shortcut=True,
                use_large_kernel=self.use_large_kernel,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg)
        else:
            csp = CSPLayerWithTwoConv(
                out_channels,
                out_channels,
                num_blocks=num_blocks,
                add_identity=True,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg)

        stage = [downsample, csp]
        if stage_idx == len(self.arch_setting) - 1:
            stage.extend([
                SPPFBottleneck(
                    out_channels,
                    out_channels,
                    kernel_sizes=5,
                    norm_cfg=self.norm_cfg,
                    act_cfg=self.act_cfg),
                YOLOv10PSA(
                    out_channels,
                    out_channels,
                    norm_cfg=self.norm_cfg,
                    act_cfg=self.act_cfg)
            ])
        return stage

    def init_weights(self):
        if self.init_cfg is None:
            for module in self.modules():
                if isinstance(module, torch.nn.Conv2d):
                    module.reset_parameters()
        else:
            super().init_weights()
