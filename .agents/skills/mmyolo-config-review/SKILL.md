---
name: mmyolo-config-review
description: Use this skill when reviewing MMYOLO config files, especially YOLOv12, DOTA, dataset, dataloader, evaluator, load_from, scheduler, and custom_hooks settings.
---

# MMYOLO Config Review

When reviewing MMYOLO configs:

1. Check `_base_`, `num_classes`, `metainfo`, dataset type, data_root, ann_file, img paths, train/val/test dataloaders.
2. Check whether evaluator ann_file matches val/test JSON.
3. Check whether COCO pretrained checkpoint is MMYOLO/MMEngine-compatible `.pth`, not raw Ultralytics `.pt`.
4. Check whether DOTA class order is preserved.
5. Check whether batch size, base_lr, warmup, scheduler, close_mosaic, AMP, EMA, and custom_hooks are consistent.
6. Never change files before explaining the issue and showing evidence.
7. Prefer small patches and show diff after modification.
8. Do not start long training unless explicitly requested.