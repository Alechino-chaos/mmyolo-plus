# Copyright (c) OpenMMLab. All rights reserved.
from unittest import TestCase

import torch

from mmyolo.models import YOLOv12PAFPN
from mmyolo.utils import register_all_modules

register_all_modules()


class TestYOLOv12PAFPN(TestCase):

    def test_forward_nano_scale(self):
        in_channels = [512, 512, 1024]
        out_channels = [256, 512, 1024]
        feats = [
            torch.rand(1, 128, 8, 8),
            torch.rand(1, 128, 4, 4),
            torch.rand(1, 256, 2, 2)
        ]
        neck = YOLOv12PAFPN(
            in_channels=in_channels,
            out_channels=out_channels,
            deepen_factor=0.5,
            widen_factor=0.25,
            max_channels=1024)

        outs = neck(feats)

        self.assertEqual(len(outs), 3)
        self.assertEqual(outs[0].shape, torch.Size((1, 64, 8, 8)))
        self.assertEqual(outs[1].shape, torch.Size((1, 128, 4, 4)))
        self.assertEqual(outs[2].shape, torch.Size((1, 256, 2, 2)))

    def test_forward_xlarge_scale(self):
        in_channels = [512, 512, 1024]
        out_channels = [256, 512, 1024]
        feats = [
            torch.rand(1, 768, 8, 8),
            torch.rand(1, 768, 4, 4),
            torch.rand(1, 768, 2, 2)
        ]
        neck = YOLOv12PAFPN(
            in_channels=in_channels,
            out_channels=out_channels,
            deepen_factor=1.0,
            widen_factor=1.5,
            max_channels=512,
            use_residual=True,
            mlp_ratio=1.5)

        outs = neck(feats)

        self.assertEqual(len(outs), 3)
        self.assertEqual(outs[0].shape, torch.Size((1, 384, 8, 8)))
        self.assertEqual(outs[1].shape, torch.Size((1, 768, 4, 4)))
        self.assertEqual(outs[2].shape, torch.Size((1, 768, 2, 2)))
