# Copyright (c) OpenMMLab. All rights reserved.
from unittest import TestCase

import torch
from torch.nn.modules.batchnorm import _BatchNorm

from mmyolo.models.layers import A2C2f, AAttn, ABlock, C3k, C3k2
from mmyolo.utils import register_all_modules

register_all_modules()


class TestYOLOv12Bricks(TestCase):

    def test_attention_blocks_forward(self):
        x = torch.randn(1, 64, 4, 4)

        attn = AAttn(64, num_heads=2, area=4)
        self.assertEqual(attn(x).shape, x.shape)
        self.assertIsInstance(attn.qk.bn, _BatchNorm)

        block = ABlock(64, num_heads=2, area=4)
        self.assertEqual(block(x).shape, x.shape)
        self.assertIsInstance(block.mlp[1].bn, _BatchNorm)

    def test_csp_blocks_forward(self):
        x = torch.randn(1, 64, 8, 8)

        c3k = C3k(64, 64, num_blocks=2)
        self.assertEqual(c3k(x).shape, x.shape)

        c3k2 = C3k2(64, 128, num_blocks=2, use_c3k=True)
        self.assertEqual(c3k2(x).shape, torch.Size((1, 128, 8, 8)))
        self.assertIsInstance(c3k2.m[0], C3k)

    def test_a2c2f_forward(self):
        x = torch.randn(1, 64, 8, 8)

        block = A2C2f(64, 64, num_ablocks=2, area=4, use_residual=True)
        self.assertEqual(block(x).shape, x.shape)
        self.assertIsNotNone(block.gamma)

        fallback = A2C2f(64, 128, num_ablocks=1, use_area_attn=False)
        self.assertEqual(fallback(x).shape, torch.Size((1, 128, 8, 8)))
        self.assertIsNone(fallback.gamma)
