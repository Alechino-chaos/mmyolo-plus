---
name: dota-yolov12-experiment
description: Use this skill when planning YOLOv12-n DOTA1.5 experiments, formal baselines, ablations, COCO pretrain fine-tuning, work_dir naming, or experiment comparison.
---

# DOTA YOLOv12 Experiment Planning

Experiment rules:

1. Formal baseline must use MMYOLO-compatible YOLOv12-n COCO `.pth` pretrained checkpoint.
2. `load_from=None` is only a sanity run, not a formal baseline.
3. Training environment is single-node 3 x RTX 2080.
4. Use batch_size_per_gpu=4, total batch=12, base_lr=0.0009375 unless explicitly changed.
5. Keep DOTA 16-class order fixed.
6. Separate sanity runs, formal baselines, plugin experiments, and ablations into different work_dirs.
7. Every proposed experiment should include purpose, changed files, command, expected log signs, and success criteria.