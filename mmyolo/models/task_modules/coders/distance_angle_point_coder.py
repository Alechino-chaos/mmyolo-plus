# Copyright (c) OpenMMLab. All rights reserved.
from typing import Optional, Sequence, Union

import numpy as np
import torch
from mmdet.models.task_modules.coders import BaseBBoxCoder

from mmyolo.registry import TASK_UTILS


@TASK_UTILS.register_module()
class DistanceAnglePointCoder(BaseBBoxCoder):
    """Distance Angle Point BBox coder.

    This coder encodes gt bboxes (x, y, w, h, theta) into (left, top, right,
    bottom, theta) and decodes them back to the original representation.
    """

    def __init__(self, clip_border=True, angle_version='oc'):
        self.clip_border = clip_border
        self.angle_version = angle_version

    @staticmethod
    def norm_angle(angle: torch.Tensor, angle_version: str) -> torch.Tensor:
        """Normalize angles without depending on MMRotate coder modules."""
        if angle_version == 'oc':
            return angle
        if angle_version == 'le135':
            return (angle + np.pi / 4) % np.pi - np.pi / 4
        if angle_version == 'le90':
            return (angle + np.pi / 2) % np.pi - np.pi / 2
        if angle_version == 'r360':
            return (angle + np.pi) % (2 * np.pi) - np.pi
        raise ValueError(f'Unsupported angle version: {angle_version}')

    @classmethod
    def distance2obb(cls,
                     points: torch.Tensor,
                     distance: torch.Tensor,
                     max_shape=None,
                     angle_version: str = 'oc') -> torch.Tensor:
        """Convert left/top/right/bottom/angle distances to rotated boxes."""
        distance, angle = distance.split([4, 1], dim=-1)
        cos_angle, sin_angle = torch.cos(angle), torch.sin(angle)
        rotation = torch.cat(
            [cos_angle, -sin_angle, sin_angle, cos_angle], dim=-1)
        rotation = rotation.reshape(*rotation.shape[:-1], 2, 2)
        wh = distance[..., :2] + distance[..., 2:]
        offset = (distance[..., 2:] - distance[..., :2]) / 2
        offset = torch.matmul(rotation, offset[..., None]).squeeze(-1)
        center = points[..., :2] + offset
        angle = cls.norm_angle(angle, angle_version)
        return torch.cat([center, wh, angle], dim=-1)

    @staticmethod
    def obb2distance(points: torch.Tensor,
                     bboxes: torch.Tensor,
                     max_dis: Optional[float] = None,
                     eps: float = 0.1) -> torch.Tensor:
        """Convert rotated boxes to left/top/right/bottom/angle distances."""
        center, wh, angle = bboxes.split([2, 2, 1], dim=-1)
        cos_angle, sin_angle = torch.cos(angle), torch.sin(angle)
        rotation = torch.cat(
            [cos_angle, sin_angle, -sin_angle, cos_angle], dim=-1)
        rotation = rotation.reshape(*rotation.shape[:-1], 2, 2)
        offset = torch.matmul(rotation,
                              (points - center)[..., None]).squeeze(-1)
        left = wh[..., 0] / 2 + offset[..., 0]
        right = wh[..., 0] / 2 - offset[..., 0]
        top = wh[..., 1] / 2 + offset[..., 1]
        bottom = wh[..., 1] / 2 - offset[..., 1]
        if max_dis is not None:
            left = left.clamp(min=0, max=max_dis - eps)
            top = top.clamp(min=0, max=max_dis - eps)
            right = right.clamp(min=0, max=max_dis - eps)
            bottom = bottom.clamp(min=0, max=max_dis - eps)
        return torch.stack(
            [left, top, right, bottom, angle.squeeze(-1)], dim=-1)

    def decode(
        self,
        points: torch.Tensor,
        pred_bboxes: torch.Tensor,
        stride: torch.Tensor,
        max_shape: Optional[Union[Sequence[int], torch.Tensor,
                                  Sequence[Sequence[int]]]] = None,
    ) -> torch.Tensor:
        """Decode distance prediction to bounding box.

        Args:
            points (Tensor): Shape (B, N, 2) or (N, 2).
            pred_bboxes (Tensor): Distance from the given point to 4
                boundaries and angle (left, top, right, bottom, angle).
                Shape (B, N, 5) or (N, 5)
            max_shape (Sequence[int] or torch.Tensor or Sequence[
                Sequence[int]],optional): Maximum bounds for boxes, specifies
                (H, W, C) or (H, W). If priors shape is (B, N, 4), then
                the max_shape should be a Sequence[Sequence[int]],
                and the length of max_shape should also be B.
                Default None.
        Returns:
            Tensor: Boxes with shape (N, 5) or (B, N, 5)
        """
        assert points.size(-2) == pred_bboxes.size(-2)
        assert points.size(-1) == 2
        assert pred_bboxes.size(-1) == 5
        if self.clip_border is False:
            max_shape = None

        if pred_bboxes.dim() == 2:
            stride = stride[:, None]
        else:
            stride = stride[None, :, None]
        pred_bboxes[..., :4] = pred_bboxes[..., :4] * stride

        return self.distance2obb(points, pred_bboxes, max_shape,
                                 self.angle_version)

    def encode(self,
               points: torch.Tensor,
               gt_bboxes: torch.Tensor,
               max_dis: float = 16.,
               eps: float = 0.01) -> torch.Tensor:
        """Encode bounding box to distances.

        Args:
            points (Tensor): Shape (N, 2), The format is [x, y].
            gt_bboxes (Tensor): Shape (N, 5), The format is "xywha"
            max_dis (float): Upper bound of the distance. Defaults to 16.
            eps (float): a small value to ensure target < max_dis, instead <=.
                Defaults to 0.01.

        Returns:
            Tensor: Box transformation deltas. The shape is (N, 5).
        """

        assert points.size(-2) == gt_bboxes.size(-2)
        assert points.size(-1) == 2
        assert gt_bboxes.size(-1) == 5
        return self.obb2distance(points, gt_bboxes, max_dis, eps)
