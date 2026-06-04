# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os
import sys
from collections import OrderedDict
from pathlib import Path
from typing import Dict, Optional
from urllib.parse import urlparse

import torch


YOLOV12N_URL = (
    'https://github.com/sunsmarterjie/yolov12/releases/download/'
    'turbo/yolov12n.pt')

CONVERT_DICT = {
    # Backbone
    'model.0': 'backbone.stem',
    'model.1': 'backbone.stage1.0',
    'model.2': 'backbone.stage1.1',
    'model.3': 'backbone.stage2.0',
    'model.4': 'backbone.stage2.1',
    'model.5': 'backbone.stage3.0',
    'model.6': 'backbone.stage3.1',
    'model.7': 'backbone.stage4.0',
    'model.8': 'backbone.stage4.1',

    # Neck. model.9/10/12/13/16/19 are upsample/concat layers.
    'model.11': 'neck.top_down_layers.0',
    'model.14': 'neck.top_down_layers.1',
    'model.15': 'neck.downsample_layers.0',
    'model.17': 'neck.bottom_up_layers.0',
    'model.18': 'neck.downsample_layers.1',
    'model.20': 'neck.bottom_up_layers.1',

    # Detection head
    'model.21': 'bbox_head.head_module',
}

C3K2_PREFIXES = {'model.2', 'model.4', 'model.20'}


def _torch_load(path: str):
    """Load old Ultralytics checkpoints across PyTorch versions."""
    try:
        return torch.load(path, map_location='cpu', weights_only=False)
    except TypeError:
        return torch.load(path, map_location='cpu')


def _download(url: str, dst: str):
    from torch.hub import download_url_to_file

    os.makedirs(os.path.dirname(os.path.abspath(dst)), exist_ok=True)
    print(f'Download {url} to {dst}')
    download_url_to_file(url, dst, progress=True)


def _is_url(path: str) -> bool:
    parsed = urlparse(path)
    return parsed.scheme in {'http', 'https'}


def _resolve_src(src: str, cache_dir: str) -> str:
    if not _is_url(src):
        return src

    filename = os.path.basename(urlparse(src).path)
    dst = os.path.join(cache_dir, filename)
    if not os.path.exists(dst):
        _download(src, dst)
    else:
        print(f'Use cached checkpoint: {dst}')
    return dst


def _load_ultralytics_state_dict(src: str, official_repo: Optional[str]):
    if official_repo:
        sys.path.insert(0, os.path.abspath(official_repo))

    try:
        checkpoint = _torch_load(src)
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            'Loading official YOLOv12 .pt requires the sunsmarterjie/yolov12 '
            'code or an installed matching ultralytics package. Pass '
            '`--official-repo /path/to/yolov12` if it is not installed.'
        ) from exc

    model = checkpoint.get('ema') or checkpoint.get('model')
    if model is not None and hasattr(model, 'state_dict'):
        return model.float().state_dict()
    if isinstance(model, dict):
        return model
    if 'state_dict' in checkpoint:
        return checkpoint['state_dict']
    raise RuntimeError('Cannot find a model state_dict in the source file.')


def _split_first_dim(tensor: torch.Tensor, part: int) -> torch.Tensor:
    """Split an Ultralytics C3k2 cv1 tensor into MMYOLO branches.

    Ultralytics C3k2 uses one cv1 with 2*c output channels and then chunks it.
    This MMYOLO implementation uses two separate c-channel conv branches:
    official first half -> MMYOLO cv2, official second half -> MMYOLO cv1.
    """
    if tensor.ndim == 0:
        return tensor
    chunks = tensor.chunk(2, dim=0)
    return chunks[part].contiguous()


def _convert_c3k2_top_branch(key: str, weight: torch.Tensor,
                             prefix: str) -> Dict[str, torch.Tensor]:
    branch_key = key[len(prefix) + 1:]
    target_prefix = CONVERT_DICT[prefix]
    converted = OrderedDict()

    if branch_key.startswith('cv1.'):
        suffix = branch_key[len('cv1.'):]
        # official cv1 first half feeds the first concat branch
        converted[f'{target_prefix}.cv2.{suffix}'] = _split_first_dim(
            weight, 0)
        # official cv1 second half feeds the branch that enters inner blocks
        converted[f'{target_prefix}.cv1.{suffix}'] = _split_first_dim(
            weight, 1)
    elif branch_key.startswith('cv2.'):
        suffix = branch_key[len('cv2.'):]
        converted[f'{target_prefix}.cv3.{suffix}'] = weight
    else:
        converted[f'{target_prefix}.{branch_key}'] = weight
    return converted


def _convert_regular_key(key: str, weight: torch.Tensor,
                         prefix: str) -> Dict[str, torch.Tensor]:
    new_key = key.replace(prefix, CONVERT_DICT[prefix], 1)

    if new_key.startswith('bbox_head.head_module.'):
        if '.dfl.conv.weight' in new_key:
            print(f'Skip {key}: MMYOLO YOLOv8Head uses a proj buffer.')
            return {}
        if '.cv3.' in new_key:
            print(f'Skip {key}: YOLOv12 official cls branch is not '
                  'architecturally compatible with this MMYOLO head.')
            return {}
        new_key = new_key.replace('.cv2.', '.reg_preds.')
        new_key = new_key.replace('.cv3.', '.cls_preds.')

    return {new_key: weight}


def convert_state_dict(blobs: Dict[str, torch.Tensor]) -> OrderedDict:
    state_dict = OrderedDict()
    skipped = 0

    for key, weight in blobs.items():
        parts = key.split('.')
        if len(parts) < 3 or parts[0] != 'model':
            skipped += 1
            print(f'Skip {key}: not an Ultralytics model key.')
            continue

        prefix = f'model.{parts[1]}'
        if prefix not in CONVERT_DICT:
            skipped += 1
            print(f'Skip {key}: no trainable MMYOLO counterpart.')
            continue

        if prefix in C3K2_PREFIXES:
            converted = _convert_c3k2_top_branch(key, weight, prefix)
        else:
            converted = _convert_regular_key(key, weight, prefix)

        if not converted:
            skipped += 1
            continue

        for new_key, new_weight in converted.items():
            state_dict[new_key] = new_weight
            print(f'Convert {key} -> {new_key}')

    print(f'Converted tensors: {len(state_dict)}; skipped tensors: {skipped}')
    return state_dict


def _build_target_state_dict(config: str) -> Optional[Dict[str, torch.Tensor]]:
    if not config:
        return None

    try:
        from mmengine.config import Config

        from mmyolo.registry import MODELS
        from mmyolo.utils import register_all_modules
    except ImportError as exc:
        raise RuntimeError(
            'Shape filtering requires MMEngine/MMCV/MMDetection/MMYOLO. '
            'Install the training environment first, or pass '
            '`--no-shape-filter`.') from exc

    register_all_modules()
    cfg = Config.fromfile(config)
    model = MODELS.build(cfg.model)
    return model.state_dict()


def filter_by_shape(state_dict: OrderedDict,
                    target_state_dict: Dict[str, torch.Tensor]) -> OrderedDict:
    filtered = OrderedDict()
    dropped = []

    for key, weight in state_dict.items():
        if key not in target_state_dict:
            dropped.append((key, tuple(weight.shape), None))
            continue
        target_shape = tuple(target_state_dict[key].shape)
        if tuple(weight.shape) != target_shape:
            dropped.append((key, tuple(weight.shape), target_shape))
            continue
        filtered[key] = weight

    if dropped:
        print('Dropped incompatible tensors:')
        for key, src_shape, dst_shape in dropped:
            print(f'  {key}: source={src_shape}, target={dst_shape}')
    print(f'Kept tensors after shape filtering: {len(filtered)}')
    return filtered


def convert(src: str,
            dst: str,
            config: str = '',
            official_repo: Optional[str] = None,
            cache_dir: str = 'work_dirs/pretrain',
            shape_filter: bool = True):
    src = _resolve_src(src, cache_dir)
    blobs = _load_ultralytics_state_dict(src, official_repo)
    state_dict = convert_state_dict(blobs)

    if shape_filter and config:
        target_state_dict = _build_target_state_dict(config)
        state_dict = filter_by_shape(state_dict, target_state_dict)

    os.makedirs(os.path.dirname(os.path.abspath(dst)), exist_ok=True)
    checkpoint = dict(state_dict=state_dict)
    torch.save(checkpoint, dst)
    print(f'Saved MMYOLO checkpoint to {dst}')


def _default_config() -> str:
    config = Path('configs/yolov12/yolov12_n.py')
    return str(config) if config.exists() else ''


def main():
    parser = argparse.ArgumentParser(
        description='Convert official YOLOv12-n COCO weights to MMYOLO.')
    parser.add_argument(
        '--src',
        default=YOLOV12N_URL,
        help='Official YOLOv12 .pt path or URL.')
    parser.add_argument(
        '--dst',
        default='work_dirs/pretrain/yolov12n_coco_mmyolo.pth',
        help='Destination MMYOLO .pth path.')
    parser.add_argument(
        '--config',
        default=_default_config(),
        help='Target MMYOLO config used to drop shape-incompatible tensors.')
    parser.add_argument(
        '--official-repo',
        default=None,
        help='Path to the official sunsmarterjie/yolov12 repo if needed.')
    parser.add_argument(
        '--cache-dir',
        default='work_dirs/pretrain',
        help='Directory used when --src is a URL.')
    parser.add_argument(
        '--no-shape-filter',
        action='store_true',
        help='Do not build the target MMYOLO model to filter tensor shapes.')
    args = parser.parse_args()

    convert(
        src=args.src,
        dst=args.dst,
        config=args.config,
        official_repo=args.official_repo,
        cache_dir=args.cache_dir,
        shape_filter=not args.no_shape_filter)


if __name__ == '__main__':
    main()
