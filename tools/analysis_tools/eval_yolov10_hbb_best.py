#!/usr/bin/env python
"""Evaluate completed YOLOv10-H runs at IoU 0.50:0.05:0.95.

The script finds the best checkpoint from a completed 80-epoch run, invokes
``tools/dist_test.sh``, and summarizes AP50, AP75, and the mean across all ten
IoU thresholds. The mean is VOC-style because the dataset uses VOCMetric; it
is not the full COCO area/maxDets metric.
"""

import argparse
import csv
import re
import subprocess
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


IOU_THRESHOLDS = [round(0.50 + 0.05 * index, 2) for index in range(10)]
SIZES = ('s', 'm', 'l', 'x')
BEST_CKPT_PATTERN = re.compile(
    r'best_pascal_voc_mAP_epoch_(\d+)\.pth$')
COMPLETED_PATTERN = re.compile(r'Epoch\(val\)\s+\[80\]\[\d+/\d+\]')
BEST_LOG_PATTERN = re.compile(
    r'best checkpoint with ([0-9.]+) pascal_voc/mAP at (\d+) epoch')


def parse_args(args: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=('Evaluate best completed YOLOv10-H checkpoints at '
                     'IoU 0.50:0.05:0.95.'))
    parser.add_argument(
        '--sizes',
        nargs='+',
        choices=SIZES,
        default=list(SIZES),
        help='Model sizes to evaluate. Defaults to s m l x.')
    parser.add_argument(
        '--gpus', type=int, default=3, help='Number of GPUs. Defaults to 3.')
    parser.add_argument(
        '--work-root',
        type=Path,
        default=Path('work_dirs'),
        help='Training work directory root. Defaults to work_dirs.')
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Print selected checkpoints and commands without testing.')
    return parser.parse_args(args)


def _read_text(path: Path) -> str:
    return path.read_text(encoding='utf-8', errors='replace')


def _completed_logs(work_dir: Path) -> List[Path]:
    completed = []
    for log_path in work_dir.rglob('*.log'):
        if COMPLETED_PATTERN.search(_read_text(log_path)):
            completed.append(log_path)
    return sorted(completed, key=lambda path: path.stat().st_mtime,
                  reverse=True)


def _best_from_log(log_path: Path) -> Optional[Tuple[float, int]]:
    matches = BEST_LOG_PATTERN.findall(_read_text(log_path))
    if not matches:
        return None
    best_map, epoch = matches[-1]
    return float(best_map), int(epoch)


def _checkpoint_epoch(path: Path) -> Optional[int]:
    match = BEST_CKPT_PATTERN.search(path.name)
    return int(match.group(1)) if match else None


def find_best_checkpoint(work_root: Path,
                         size: str) -> Tuple[Path, Path, int, Optional[float]]:
    pattern = f'dota10_yolov10-h_{size}_640_80e_rand_*_fp32'
    candidates = []
    for work_dir in work_root.glob(pattern):
        for log_path in _completed_logs(work_dir):
            candidates.append((log_path.stat().st_mtime, work_dir, log_path))
    if not candidates:
        raise FileNotFoundError(
            f'No completed 80-epoch YOLOv10-H-{size} run found under '
            f'{work_root}.')

    _, work_dir, log_path = max(candidates, key=lambda item: item[0])
    logged_best = _best_from_log(log_path)
    checkpoints = sorted(
        work_dir.glob('best_pascal_voc_mAP_epoch_*.pth'),
        key=lambda path: path.stat().st_mtime,
        reverse=True)
    if not checkpoints:
        raise FileNotFoundError(
            f'No best checkpoint found in completed run {work_dir}.')

    if logged_best is not None:
        best_map, best_epoch = logged_best
        matching = [
            path for path in checkpoints
            if _checkpoint_epoch(path) == best_epoch
        ]
        if matching:
            return matching[0], log_path, best_epoch, best_map

    checkpoint = checkpoints[0]
    epoch = _checkpoint_epoch(checkpoint)
    if epoch is None:
        raise ValueError(f'Cannot parse best epoch from {checkpoint}.')
    return checkpoint, log_path, epoch, None


def build_test_command(repo_root: Path, size: str, checkpoint: Path,
                       eval_dir: Path, gpus: int) -> List[str]:
    config = repo_root / 'configs' / 'yolov10' / 'hbb' / (
        f'yolov10-h_{size}_syncbn_fast_3xb2-accum6-80e_dota10-640.py')
    if not config.is_file():
        raise FileNotFoundError(f'Missing evaluation config: {config}')
    threshold_value = '[' + ','.join(map(str, IOU_THRESHOLDS)) + ']'
    return [
        'bash', str(repo_root / 'tools' / 'dist_test.sh'), str(config),
        str(checkpoint), str(gpus), '--work-dir', str(eval_dir),
        '--cfg-options', f'test_evaluator.iou_thrs={threshold_value}'
    ]


def _latest_log(directory: Path) -> Optional[Path]:
    logs = list(directory.rglob('*.log'))
    return max(logs, key=lambda path: path.stat().st_mtime) if logs else None


def parse_metrics(log_path: Path) -> Dict[str, Optional[float]]:
    text = _read_text(log_path)
    metrics = {}
    for key in ('mAP', 'AP50', 'AP75'):
        matches = re.findall(
            rf'pascal_voc/{key}:\s*([0-9.]+)', text)
        metrics[key] = float(matches[-1]) if matches else None
    return metrics


def _write_summary(rows: Iterable[Dict[str, object]], path: Path) -> None:
    fields = [
        'model', 'checkpoint', 'best_epoch', 'training_best_ap50',
        'reeval_ap50', 'ap75', 'voc_map50_95', 'evaluation_log'
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main(args: Optional[Sequence[str]] = None) -> int:
    parsed = parse_args(args)
    if parsed.gpus < 1:
        print('ERROR: --gpus must be positive.', file=sys.stderr)
        return 2

    repo_root = Path(__file__).resolve().parents[2]
    work_root = (repo_root / parsed.work_root).resolve()
    rows = []
    for size in parsed.sizes:
        try:
            checkpoint, train_log, best_epoch, best_map = (
                find_best_checkpoint(work_root, size))
            eval_dir = work_root / (
                f'eval_yolov10-h_{size}_best_voc_ap50-95')
            command = build_test_command(repo_root, size, checkpoint,
                                         eval_dir, parsed.gpus)
        except (OSError, ValueError) as error:
            print(f'ERROR [{size}]: {error}', file=sys.stderr)
            return 2

        print(f'\nYOLOv10-H-{size}')
        print(f'  completed training log: {train_log}')
        print(f'  best checkpoint: {checkpoint}')
        print(f'  best epoch: {best_epoch}')
        if best_map is not None:
            print(f'  training AP50: {best_map:.4f}')
        print('  command: ' + ' '.join(command))
        if parsed.dry_run:
            continue

        result = subprocess.run(command, cwd=repo_root, check=False)
        if result.returncode != 0:
            print(f'ERROR [{size}]: evaluation failed with exit code '
                  f'{result.returncode}.', file=sys.stderr)
            return result.returncode

        eval_log = _latest_log(eval_dir)
        metrics = parse_metrics(eval_log) if eval_log else {
            'mAP': None,
            'AP50': None,
            'AP75': None
        }
        rows.append({
            'model': f'yolov10-h_{size}',
            'checkpoint': checkpoint,
            'best_epoch': best_epoch,
            'training_best_ap50': best_map,
            'reeval_ap50': metrics['AP50'],
            'ap75': metrics['AP75'],
            'voc_map50_95': metrics['mAP'],
            'evaluation_log': eval_log or ''
        })
        print(f"  re-eval AP50: {metrics['AP50']}")
        print(f"  AP75: {metrics['AP75']}")
        print(f"  VOC-style mAP50:95: {metrics['mAP']}")

    if not parsed.dry_run:
        summary_path = work_root / 'yolov10_hbb_ap50_95_summary.csv'
        _write_summary(rows, summary_path)
        print(f'\nSummary written to {summary_path}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
