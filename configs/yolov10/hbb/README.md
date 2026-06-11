# YOLOv10-H DOTA1.0-derived HBB 640/80e

This config group evaluates horizontal detection on the supplied Ultralytics
YOLO HBB dataset. It does not use MMRotate, rotated IoU, rotated NMS, angle
prediction, or HBB-to-OBB conversion.

## Protocol

- Classes: DOTA1.0 IDs 0-14; source class 15 (`container-crane`) is ignored.
- Labels: normalized `class cx cy width height`.
- Input: `640x640`.
- Schedule: `80` epochs with random initialization (`load_from=None`).
- Evaluator: `mmdet.VOCMetric`, mAP at horizontal IoU 0.5.
- Default effective batch: `4 x 3 GPUs x accumulation 3 = 36`.
- Precision: FP32 for consistent runs on the three RTX 2080 GPUs.
- LR warmup and cosine scheduling are epoch based.

## Smoke test

```bash
CUDA_VISIBLE_DEVICES=0,1,2 bash tools/dist_train.sh \
  configs/yolov10/hbb/yolov10-h_s_syncbn_fast_3xb4-accum3-80e_dota10-640.py \
  3 \
  --work-dir work_dirs/smoke_yolov10-h_s_3gpu_fp32 \
  --cfg-options train_cfg.max_epochs=1 default_hooks.logger.interval=1
```

Do not add `--amp` until a separate three-GPU AMP smoke test has completed
without NaN/Inf. FP32 is the reference precision for this experiment group.

## Formal training

```bash
CUDA_VISIBLE_DEVICES=0,1,2 bash tools/dist_train.sh \
  configs/yolov10/hbb/yolov10-h_s_syncbn_fast_3xb4-accum3-80e_dota10-640.py 3
```

Train in order `s -> m -> l -> x`. Dense aerial batches can make the dynamic
assigner use substantial memory even for the small model. If FP32 OOMs, use
the provided fallback:

- `s/m`: use `3xb2-accum6`, effective batch 36.
- `l`: `3xb2-accum6`, effective batch 36.
- `x`: first `3xb2-accum6`, then `3xb1-accum12`, effective batch 36.

## Measurement

```bash
python tools/analysis_tools/get_flops.py \
  configs/yolov10/hbb/yolov10-h_s_syncbn_fast_3xb4-accum3-80e_dota10-640.py \
  --shape 640 640
```

Record the actual fallback, peak memory, runtime, best epoch, mAP@0.5 and the
15 per-class AP values in `results_template.csv`.
