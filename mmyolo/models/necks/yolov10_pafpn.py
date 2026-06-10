# Copyright (c) OpenMMLab. All rights reserved.
"""YOLOv10 PAFPN for rotated detection experiments."""

from typing import List, Union

import torch
import torch.nn as nn
from mmcv.cnn import ConvModule
from mmdet.utils import ConfigType, OptMultiConfig

from mmyolo.registry import MODELS
from ..layers import CSPLayerWithTwoConv
from ..layers.yolov10_bricks import YOLOv10C2fCIB, YOLOv10SCDown
from ..utils import make_round
from .base_yolo_neck import BaseYOLONeck


@MODELS.register_module()
class YOLOv10PAFPN(BaseYOLONeck):
    """YOLOv10-style PAFPN with uniform output channels.

    The internal top-down and bottom-up path follows the YOLOv10 P5 feature
    routing. The final output convolutions adapt P3/P4/P5 features to a single
    channel width so the shared rotated head remains identical across sizes.
    """

    def __init__(self,
                 in_channels: List[int],
                 out_channels: Union[List[int], int],
                 head_channels: int = 256,
                 deepen_factor: float = 1.0,
                 widen_factor: float = 1.0,
                 max_channels: int = 1024,
                 num_csp_blocks: int = 3,
                 use_large_kernel: bool = False,
                 freeze_all: bool = False,
                 norm_cfg: ConfigType = dict(
                     type='BN', momentum=0.03, eps=0.001),
                 act_cfg: ConfigType = dict(type='SiLU', inplace=True),
                 init_cfg: OptMultiConfig = None):
        assert isinstance(out_channels, list) and len(out_channels) == 3
        self.head_channels = head_channels
        self.max_channels = max_channels
        self.num_csp_blocks = num_csp_blocks
        self.use_large_kernel = use_large_kernel
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            deepen_factor=deepen_factor,
            widen_factor=widen_factor,
            freeze_all=freeze_all,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg,
            init_cfg=init_cfg)

    def init_weights(self):
        if self.init_cfg is None:
            for module in self.modules():
                if isinstance(module, torch.nn.Conv2d):
                    module.reset_parameters()
        else:
            super().init_weights()

    def build_reduce_layer(self, idx: int) -> nn.Module:
        return nn.Identity()

    def build_upsample_layer(self, *args, **kwargs) -> nn.Module:
        return nn.Upsample(scale_factor=2, mode='nearest')

    def build_top_down_layer(self, idx: int) -> nn.Module:
        in_channels = self.in_channels[idx] + self.in_channels[idx - 1]
        out_channels = self.out_channels[idx - 1]
        return CSPLayerWithTwoConv(
            in_channels,
            out_channels,
            num_blocks=make_round(self.num_csp_blocks, self.deepen_factor),
            add_identity=False,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)

    def build_downsample_layer(self, idx: int) -> nn.Module:
        return YOLOv10SCDown(
            self.out_channels[idx],
            self.out_channels[idx],
            kernel_size=3,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)

    def build_bottom_up_layer(self, idx: int) -> nn.Module:
        in_channels = self.out_channels[idx] + self.out_channels[idx + 1]
        out_channels = self.out_channels[idx + 1]
        return YOLOv10C2fCIB(
            in_channels,
            out_channels,
            num_blocks=make_round(self.num_csp_blocks, self.deepen_factor),
            shortcut=True,
            use_large_kernel=self.use_large_kernel,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)

    def build_out_layer(self, idx: int) -> nn.Module:
        return ConvModule(
            self.out_channels[idx],
            self.head_channels,
            kernel_size=1,
            stride=1,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)
