# Copyright (c) OpenMMLab. All rights reserved.
from unittest import TestCase

import torch

from mmyolo.models.backbones import YOLOv12CSPDarknet
from mmyolo.models.layers import C3k
from mmyolo.utils import register_all_modules

register_all_modules()


class TestYOLOv12CSPDarknet(TestCase):

    def test_forward_nano_scale(self):
        model = YOLOv12CSPDarknet(
            deepen_factor=0.5,
            widen_factor=0.25,
            max_channels=1024,
            last_stage_out_channels=1024,
            out_indices=range(0, 5))
        model.train()

        feats = model(torch.randn(1, 3, 64, 64))

        self.assertEqual(len(feats), 5)
        self.assertEqual(feats[0].shape, torch.Size((1, 16, 32, 32)))
        self.assertEqual(feats[1].shape, torch.Size((1, 64, 16, 16)))
        self.assertEqual(feats[2].shape, torch.Size((1, 128, 8, 8)))
        self.assertEqual(feats[3].shape, torch.Size((1, 128, 4, 4)))
        self.assertEqual(feats[4].shape, torch.Size((1, 256, 2, 2)))
        self.assertEqual(len(model.stage4), 2)

    def test_forward_xlarge_scale(self):
        model = YOLOv12CSPDarknet(
            deepen_factor=1.0,
            widen_factor=1.5,
            max_channels=512,
            last_stage_out_channels=1024,
            use_c3k=True,
            use_residual=True,
            mlp_ratio=1.5)
        model.train()

        feats = model(torch.randn(1, 3, 64, 64))

        self.assertEqual(len(feats), 3)
        self.assertEqual(feats[0].shape, torch.Size((1, 768, 8, 8)))
        self.assertEqual(feats[1].shape, torch.Size((1, 768, 4, 4)))
        self.assertEqual(feats[2].shape, torch.Size((1, 768, 2, 2)))
        self.assertIsInstance(model.stage1[1].m[0], C3k)
        self.assertIsNotNone(model.stage3[1].gamma)
        self.assertIsNotNone(model.stage4[1].gamma)
