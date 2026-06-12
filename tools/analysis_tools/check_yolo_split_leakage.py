#!/usr/bin/env python
"""Audit a YOLO train/val split for patch-level data leakage.

The dataset is never modified. Reports are written outside the dataset by
default, under ``work_dirs/dataset_audits/<dataset-name>``.
"""

import argparse
import csv
import hashlib
import json
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple


IMAGE_SUFFIXES = {
    '.bmp', '.jpeg', '.jpg', '.png', '.tif', '.tiff', '.webp'
}


def parse_args(args: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Check a YOLO train/val split for data leakage.')
    parser.add_argument(
        'data_root',
        type=Path,
        help=('Dataset root containing images/{train,val} and '
              'labels/{train,val}.'))
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=None,
        help=('Report directory. Defaults to '
              'work_dirs/dataset_audits/<dataset-name>.'))
    return parser.parse_args(args)


def _emit(progress: Optional[Callable[[str], None]], message: str) -> None:
    if progress is not None:
        progress(message)


def _list_files(directory: Path,
                suffixes: Optional[Iterable[str]] = None) -> List[Path]:
    allowed = set(suffixes) if suffixes is not None else None
    files = []
    if directory.is_dir():
        for path in directory.rglob('*'):
            if path.is_file() and (allowed is None
                                   or path.suffix.lower() in allowed):
                files.append(path)
    return sorted(files)


def _relative(path: Path, root: Path) -> str:
    return path.relative_to(root).as_posix()


def _relative_stem(path: Path, root: Path) -> str:
    return path.relative_to(root).with_suffix('').as_posix()


def _source_image_id(path: Path) -> str:
    return path.stem.split('__', 1)[0]


def _group_by_name(files: Iterable[Path], root: Path) -> Dict[str, List[str]]:
    groups = defaultdict(list)
    for path in files:
        groups[path.name].append(_relative(path, root))
    return dict(groups)


def _group_by_source_id(files: Iterable[Path],
                        root: Path) -> Dict[str, List[str]]:
    groups = defaultdict(list)
    for path in files:
        groups[_source_image_id(path)].append(_relative(path, root))
    return dict(groups)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _group_by_hash(files: Iterable[Path], root: Path,
                   progress: Optional[Callable[[str], None]],
                   split: str) -> Dict[str, List[str]]:
    files = list(files)
    groups = defaultdict(list)
    for index, path in enumerate(files, 1):
        groups[_sha256(path)].append(_relative(path, root))
        if index % 1000 == 0 or index == len(files):
            _emit(progress,
                  f'  hashed {split}: {index}/{len(files)} images')
    return dict(groups)


def _cross_split_groups(train_groups: Dict[str, List[str]],
                        val_groups: Dict[str, List[str]],
                        key_name: str) -> List[Dict[str, object]]:
    overlap = sorted(set(train_groups) & set(val_groups))
    return [{
        key_name: key,
        'train_images': train_groups[key],
        'val_images': val_groups[key]
    } for key in overlap]


def _parse_labels(label_files: Iterable[Path], root: Path,
                  split: str) -> Tuple[Counter, List[Dict[str, object]]]:
    class_counts = Counter()
    malformed = []
    for path in label_files:
        with path.open('r', encoding='utf-8-sig') as file:
            for line_number, raw_line in enumerate(file, 1):
                line = raw_line.strip()
                if not line:
                    continue
                token = line.split()[0]
                try:
                    class_id = int(token)
                    if class_id < 0:
                        raise ValueError
                except ValueError:
                    malformed.append({
                        'split': split,
                        'path': _relative(path, root),
                        'line': line_number,
                        'reason': f'invalid class id: {token!r}'
                    })
                    continue
                class_counts[class_id] += 1
    return class_counts, malformed


def _find_unpaired(images: Iterable[Path], labels: Iterable[Path],
                   image_root: Path,
                   label_root: Path) -> Tuple[List[str], List[str]]:
    image_stems = {
        _relative_stem(path, image_root): _relative(path, image_root)
        for path in images
    }
    label_stems = {
        _relative_stem(path, label_root): _relative(path, label_root)
        for path in labels
    }
    images_without_labels = [
        image_stems[key] for key in sorted(set(image_stems) - set(label_stems))
    ]
    labels_without_images = [
        label_stems[key] for key in sorted(set(label_stems) - set(image_stems))
    ]
    return images_without_labels, labels_without_images


def _class_distribution(train_counts: Counter,
                        val_counts: Counter) -> List[Dict[str, object]]:
    train_total = sum(train_counts.values())
    val_total = sum(val_counts.values())
    rows = []
    for class_id in sorted(set(train_counts) | set(val_counts)):
        train_count = train_counts[class_id]
        val_count = val_counts[class_id]
        rows.append({
            'class_id': class_id,
            'train_instances': train_count,
            'val_instances': val_count,
            'train_share': train_count / train_total if train_total else 0.0,
            'val_share': val_count / val_total if val_total else 0.0,
            'val_to_train_ratio': (val_count / train_count
                                   if train_count else None)
        })
    return rows


def _suspicious_rows(report: Dict[str, object]) -> List[Dict[str, object]]:
    rows = []
    overlap_specs = (
        ('same_image_name', 'same_image_names', 'name'),
        ('duplicate_sha256', 'duplicate_image_hashes', 'sha256'),
        ('shared_source_id', 'source_image_ids', 'source_id'),
    )
    overlaps = report['overlaps']
    for issue_type, section, key_name in overlap_specs:
        for group in overlaps[section]:
            for split in ('train', 'val'):
                for path in group[f'{split}_images']:
                    rows.append({
                        'issue_type': issue_type,
                        'key': group[key_name],
                        'split': split,
                        'path': path,
                        'details': ''
                    })

    integrity = report['integrity']
    for split in ('train', 'val'):
        for path in integrity[split]['images_without_labels']:
            rows.append({
                'issue_type': 'image_without_label',
                'key': '',
                'split': split,
                'path': path,
                'details': ''
            })
        for path in integrity[split]['labels_without_images']:
            rows.append({
                'issue_type': 'label_without_image',
                'key': '',
                'split': split,
                'path': path,
                'details': ''
            })
    for item in integrity['malformed_labels']:
        rows.append({
            'issue_type': 'malformed_label',
            'key': item['line'],
            'split': item['split'],
            'path': item['path'],
            'details': item['reason']
        })
    return rows


def _write_reports(report: Dict[str, object], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / 'audit_report.json').open(
            'w', encoding='utf-8') as file:
        json.dump(report, file, ensure_ascii=False, indent=2)
        file.write('\n')

    class_fields = [
        'class_id', 'train_instances', 'val_instances', 'train_share',
        'val_share', 'val_to_train_ratio'
    ]
    with (output_dir / 'class_distribution.csv').open(
            'w', encoding='utf-8', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=class_fields)
        writer.writeheader()
        writer.writerows(report['class_distribution'])

    suspicious_fields = ['issue_type', 'key', 'split', 'path', 'details']
    with (output_dir / 'suspicious_files.csv').open(
            'w', encoding='utf-8', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=suspicious_fields)
        writer.writeheader()
        writer.writerows(_suspicious_rows(report))


def audit_dataset(data_root: Path,
                  output_dir: Path,
                  progress: Optional[Callable[[str], None]] = None
                  ) -> Dict[str, object]:
    data_root = data_root.resolve()
    output_dir = output_dir.resolve()
    roots = {
        'train_images': data_root / 'images' / 'train',
        'val_images': data_root / 'images' / 'val',
        'train_labels': data_root / 'labels' / 'train',
        'val_labels': data_root / 'labels' / 'val',
    }
    missing_dirs = [str(path) for path in roots.values() if not path.is_dir()]
    if missing_dirs:
        raise FileNotFoundError('Missing required directories: ' +
                                ', '.join(missing_dirs))

    _emit(progress, '[1/4] Scanning image and label files')
    train_images = _list_files(roots['train_images'], IMAGE_SUFFIXES)
    val_images = _list_files(roots['val_images'], IMAGE_SUFFIXES)
    train_labels = _list_files(roots['train_labels'], {'.txt'})
    val_labels = _list_files(roots['val_labels'], {'.txt'})

    _emit(progress, '[2/4] Computing SHA-256 image hashes')
    train_hashes = _group_by_hash(train_images, roots['train_images'],
                                  progress, 'train')
    val_hashes = _group_by_hash(val_images, roots['val_images'], progress,
                                'val')

    _emit(progress, '[3/4] Checking names, source IDs, labels, and classes')
    train_names = _group_by_name(train_images, roots['train_images'])
    val_names = _group_by_name(val_images, roots['val_images'])
    train_sources = _group_by_source_id(train_images, roots['train_images'])
    val_sources = _group_by_source_id(val_images, roots['val_images'])
    train_counts, train_malformed = _parse_labels(
        train_labels, roots['train_labels'], 'train')
    val_counts, val_malformed = _parse_labels(val_labels,
                                              roots['val_labels'], 'val')
    train_unlabelled, train_orphan_labels = _find_unpaired(
        train_images, train_labels, roots['train_images'],
        roots['train_labels'])
    val_unlabelled, val_orphan_labels = _find_unpaired(
        val_images, val_labels, roots['val_images'], roots['val_labels'])

    overlaps = {
        'same_image_names': _cross_split_groups(train_names, val_names,
                                                'name'),
        'duplicate_image_hashes': _cross_split_groups(
            train_hashes, val_hashes, 'sha256'),
        'source_image_ids': _cross_split_groups(train_sources, val_sources,
                                                'source_id')
    }
    integrity = {
        'train': {
            'images_without_labels': train_unlabelled,
            'labels_without_images': train_orphan_labels
        },
        'val': {
            'images_without_labels': val_unlabelled,
            'labels_without_images': val_orphan_labels
        },
        'malformed_labels': train_malformed + val_malformed
    }
    leakage_pass = not any(overlaps.values())
    data_integrity_pass = not (
        train_unlabelled or train_orphan_labels or val_unlabelled
        or val_orphan_labels or integrity['malformed_labels'])
    report = {
        'dataset_root': str(data_root),
        'generated_at_utc': datetime.now(timezone.utc).isoformat(),
        'counts': {
            'train': {
                'images': len(train_images),
                'labels': len(train_labels),
                'instances': sum(train_counts.values())
            },
            'val': {
                'images': len(val_images),
                'labels': len(val_labels),
                'instances': sum(val_counts.values())
            }
        },
        'overlaps': overlaps,
        'integrity': integrity,
        'class_distribution': _class_distribution(train_counts, val_counts),
        'decisions': {
            'same_name_overlap_count': len(overlaps['same_image_names']),
            'sha256_overlap_count': len(overlaps['duplicate_image_hashes']),
            'source_id_overlap_count': len(overlaps['source_image_ids']),
            'leakage_pass': leakage_pass,
            'data_integrity_pass': data_integrity_pass,
            'overall_pass': leakage_pass and data_integrity_pass,
            'verdict': ('PASS' if leakage_pass and data_integrity_pass else
                        'FAIL')
        }
    }

    _emit(progress, '[4/4] Writing JSON and CSV reports')
    _write_reports(report, output_dir)
    return report


def _print_summary(report: Dict[str, object], output_dir: Path) -> None:
    counts = report['counts']
    decisions = report['decisions']
    integrity = report['integrity']
    images_without_labels = sum(
        len(integrity[split]['images_without_labels'])
        for split in ('train', 'val'))
    labels_without_images = sum(
        len(integrity[split]['labels_without_images'])
        for split in ('train', 'val'))
    print('\nYOLO split leakage audit')
    print(f"  train: {counts['train']['images']} images, "
          f"{counts['train']['labels']} labels, "
          f"{counts['train']['instances']} instances")
    print(f"  val:   {counts['val']['images']} images, "
          f"{counts['val']['labels']} labels, "
          f"{counts['val']['instances']} instances")
    print(f"  same image names: {decisions['same_name_overlap_count']}")
    print(f"  duplicate SHA-256 groups: {decisions['sha256_overlap_count']}")
    print(f"  shared source image IDs: {decisions['source_id_overlap_count']}")
    print(f'  images without labels: {images_without_labels}')
    print(f'  labels without images: {labels_without_images}')
    print(f"  malformed label rows: {len(integrity['malformed_labels'])}")
    print(f"  verdict: {decisions['verdict']}")
    print(f'  reports: {output_dir.resolve()}')


def main(args: Optional[Sequence[str]] = None) -> int:
    parsed = parse_args(args)
    data_root = parsed.data_root
    output_dir = parsed.output_dir
    if output_dir is None:
        output_dir = (Path.cwd() / 'work_dirs' / 'dataset_audits' /
                      data_root.resolve().name)
    try:
        report = audit_dataset(data_root, output_dir, progress=print)
    except (OSError, ValueError) as error:
        print(f'ERROR: {error}', file=sys.stderr)
        return 2
    _print_summary(report, output_dir)
    return 0 if report['decisions']['overall_pass'] else 1


if __name__ == '__main__':
    sys.exit(main())
