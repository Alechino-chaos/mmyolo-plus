# Copyright (c) OpenMMLab. All rights reserved.
"""Dataset adapter for Ultralytics-style YOLO HBB text annotations."""

import os
import os.path as osp
from typing import List

import mmcv

from ..registry import DATASETS
from .yolov5_coco import BatchShapePolicyDataset


@DATASETS.register_module()
class YOLOv5YOLOTxtDataset(BatchShapePolicyDataset):
    """Read ``images/<split>`` and ``labels/<split>`` YOLO datasets.

    Each label line must use normalized horizontal-box format::

        class_id center_x center_y width height

    The dataset emits absolute ``xyxy`` boxes. A rotated pipeline can convert
    these horizontal boxes to zero-angle rotated boxes with ConvertBoxType.
    """

    METAINFO = {}
    IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff')

    def load_data_list(self) -> List[dict]:
        img_dir = self.data_prefix.get(
            'img_path', self.data_prefix.get('img', ''))
        label_dir = self.ann_file
        if not osp.isdir(img_dir):
            raise FileNotFoundError(f'Image directory does not exist: {img_dir}')
        if not osp.isdir(label_dir):
            raise FileNotFoundError(
                f'YOLO label directory does not exist: {label_dir}')

        image_paths = []
        for root, _, files in os.walk(img_dir):
            for filename in files:
                if filename.lower().endswith(self.IMG_EXTENSIONS):
                    image_paths.append(osp.join(root, filename))
        image_paths.sort()

        data_list = []
        for img_id, img_path in enumerate(image_paths):
            image = mmcv.imread(img_path)
            if image is None:
                raise ValueError(f'Failed to read image: {img_path}')
            height, width = image.shape[:2]
            relative_path = osp.relpath(img_path, img_dir)
            label_path = osp.join(
                label_dir, osp.splitext(relative_path)[0] + '.txt')
            instances = self._load_yolo_labels(label_path, width, height)
            data_list.append(
                dict(
                    img_id=img_id,
                    img_path=img_path,
                    height=height,
                    width=width,
                    instances=instances))
        return data_list

    def _load_yolo_labels(self, label_path: str, image_width: int,
                          image_height: int) -> List[dict]:
        if not osp.isfile(label_path):
            return []

        instances = []
        num_classes = len(self.metainfo.get('classes', ()))
        with open(label_path, encoding='utf-8') as label_file:
            for line_number, raw_line in enumerate(label_file, 1):
                values = raw_line.strip().split()
                if not values:
                    continue
                if len(values) != 5:
                    raise ValueError(
                        f'{label_path}:{line_number} expects 5 values for '
                        f'YOLO HBB labels, but got {len(values)}.')

                class_id = int(values[0])
                if class_id < 0 or (num_classes and class_id >= num_classes):
                    raise ValueError(
                        f'{label_path}:{line_number} has invalid class id '
                        f'{class_id} for {num_classes} classes.')
                center_x, center_y, box_width, box_height = map(
                    float, values[1:])
                x1 = max(0.0, (center_x - box_width / 2) * image_width)
                y1 = max(0.0, (center_y - box_height / 2) * image_height)
                x2 = min(float(image_width),
                         (center_x + box_width / 2) * image_width)
                y2 = min(float(image_height),
                         (center_y + box_height / 2) * image_height)
                if x2 <= x1 or y2 <= y1:
                    continue
                instances.append(
                    dict(
                        bbox=[x1, y1, x2, y2],
                        bbox_label=class_id,
                        ignore_flag=0))
        return instances
