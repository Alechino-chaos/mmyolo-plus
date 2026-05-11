# Copyright (c) OpenMMLab. All rights reserved.
"""YOLOv12 CSP-Darknet backbone.

The YOLOv12 backbone uses C3k2 blocks in early stages and A2C2f blocks
(with area-attention) in later stages for efficient feature extraction.
"""

from copy import deepcopy
from typing import List, Tuple, Union

import torch
import torch.nn as nn
from mmcv.cnn import ConvModule
from mmdet.utils import ConfigType, OptMultiConfig

from mmyolo.registry import MODELS
from ..layers import A2C2f, ABlock, C3k2
from ..utils import make_divisible, make_round
from .base_backbone import BaseBackbone


def _make_divisible_max(x: float,
                        widen_factor: float,
                        divisor: int = 8,
                        max_channels: int = 1024) -> int:
    """Scale channels with YOLOv12/Ultralytics max-channel behavior."""
    return make_divisible(min(x, max_channels), widen_factor, divisor)


@MODELS.register_module()
class YOLOv12CSPDarknet(BaseBackbone):
    """CSP-Darknet backbone used in YOLOv12.

    Args:
        arch (str): Architecture type, from {'P5'}. Defaults to 'P5'.
        last_stage_out_channels (int): Output channels of last stage.
            Defaults to 1024.
        max_channels (int): Maximum base channel count before widening.
            Defaults to 1024.
        plugins (list[dict]): List of plugins for stages.
        deepen_factor (float): Depth multiplier. Defaults to 1.0.
        widen_factor (float): Width multiplier. Defaults to 1.0.
        input_channels (int): Input image channels. Defaults to 3.
        out_indices (Tuple[int]): Output stage indices.
            Defaults to (2, 3, 4).
        frozen_stages (int): Stages to freeze. -1 = none.
        norm_cfg (dict): Normalization config.
        act_cfg (dict): Activation config.
        norm_eval (bool): Whether to set norm layers to eval mode.
        init_cfg (dict or list[dict], optional): Initialization config.
    """
    # arch_settings: [down_in, down_out, block_out, num_blocks, block_type,
    #                  down_groups, c3k2_exp, a2c2f_area]
    # block_type: 0=C3k2, 1=A2C2f
    arch_settings = {
        'P5': [
            [64, 128, 256, 2, 0, 2, 0.25, 0],
            [256, 256, 512, 2, 0, 4, 0.25, 0],
            [512, 512, 512, 4, 1, 1, 0.0, 4],
            [512, None, None, 4, 1, 1, 0.0, 1],
        ]
    }

    def __init__(self,
                 arch: str = 'P5',
                 last_stage_out_channels: int = 1024,
                 max_channels: int = 1024,
                 use_c3k: bool = False,
                 use_residual: bool = False,
                 mlp_ratio: float = 2.0,
                 plugins: Union[dict, List[dict]] = None,
                 deepen_factor: float = 1.0,
                 widen_factor: float = 1.0,
                 input_channels: int = 3,
                 out_indices: Tuple[int] = (2, 3, 4),
                 frozen_stages: int = -1,
                 norm_cfg: ConfigType = dict(
                     type='BN', momentum=0.03, eps=0.001),
                 act_cfg: ConfigType = dict(type='SiLU', inplace=True),
                 norm_eval: bool = False,
                 init_cfg: OptMultiConfig = None):
        self.max_channels = max_channels
        self.use_c3k = use_c3k
        self.use_residual = use_residual
        self.mlp_ratio = mlp_ratio
        arch_setting = deepcopy(self.arch_settings[arch])
        arch_setting[-1][1] = last_stage_out_channels
        arch_setting[-1][2] = last_stage_out_channels
        super().__init__(
            arch_setting,
            deepen_factor,
            widen_factor,
            input_channels=input_channels,
            out_indices=out_indices,
            plugins=plugins,
            frozen_stages=frozen_stages,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg,
            norm_eval=norm_eval,
            init_cfg=init_cfg)

    def build_stem_layer(self) -> nn.Module:
        """Build stem: Conv(3->64, k=3, s=2)."""
        return ConvModule(
            self.input_channels,
            _make_divisible_max(
                self.arch_setting[0][0], self.widen_factor,
                max_channels=self.max_channels),
            kernel_size=3,
            stride=2,
            padding=1,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)

    def build_stage_layer(self, stage_idx: int, setting: list) -> list:
        """Build a stage: downsample conv + one repeated YOLOv12 block."""
        (down_in, down_out, block_out, num_blocks, block_type,
         down_groups, c3k2_exp, a2c2f_area) = setting

        down_in = _make_divisible_max(
            down_in, self.widen_factor, max_channels=self.max_channels)
        down_out = _make_divisible_max(
            down_out, self.widen_factor, max_channels=self.max_channels)
        block_out = _make_divisible_max(
            block_out, self.widen_factor, max_channels=self.max_channels)
        num_blocks = make_round(num_blocks, self.deepen_factor)

        stage = [ConvModule(
            down_in, down_out, kernel_size=3, stride=2, padding=1,
            groups=down_groups,
            norm_cfg=self.norm_cfg, act_cfg=self.act_cfg)]

        if block_type == 0:  # C3k2
            stage.append(C3k2(
                down_out, block_out, num_blocks=num_blocks,
                use_c3k=self.use_c3k, expansion=c3k2_exp, shortcut=True,
                norm_cfg=self.norm_cfg, act_cfg=self.act_cfg))
        else:  # A2C2f
            stage.append(A2C2f(
                down_out, block_out, num_ablocks=num_blocks,
                use_area_attn=True, area=a2c2f_area,
                use_residual=self.use_residual, mlp_ratio=self.mlp_ratio,
                norm_cfg=self.norm_cfg, act_cfg=self.act_cfg))

        return stage

    def init_weights(self):
        """Initialize the parameters."""
        if self.init_cfg is None:
            for m in self.modules():
                if isinstance(m, torch.nn.Conv2d):
                    m.reset_parameters()
            for m in self.modules():
                if isinstance(m, ABlock):
                    m.init_weights()
        else:
            super().init_weights()
