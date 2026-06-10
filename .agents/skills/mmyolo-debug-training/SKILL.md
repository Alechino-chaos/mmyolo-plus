---
name: mmyolo-debug-training
description: Use this skill when diagnosing MMYOLO training failures, abnormal mAP, DOTA COCO JSON errors, distributed training errors, CUDA issues, checkpoint loading problems, or validation problems.
---

# MMYOLO Training Debug

When debugging training:

1. First classify the failure: config error, data path error, annotation/category error, checkpoint loading error, distributed error, CUDA/memory error, or metric/evaluator error.
2. Ask for or inspect logs from the first 200 lines and the final traceback.
3. Check whether DOTA categories are 16 classes and category ids match the model head.
4. Check whether bbox format is COCO xywh in JSON and MMYOLO expects the same.
5. Check whether `load_from` logs show backbone/neck loaded and only head mismatch is expected.
6. Prefer short validation commands: print_config, dataset visualization, one-epoch sanity run.
7. Do not recommend full 300-epoch training until config and data loading are verified.