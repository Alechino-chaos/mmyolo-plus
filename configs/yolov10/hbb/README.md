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

Before training, audit the patch split. This checks same-name files, exact
SHA-256 duplicates, and patches derived from the same source image ID. A
non-zero exit status means the split must not be used for final metrics.

```bash
python tools/analysis_tools/check_yolo_split_leakage.py \
  /share/home/luofeiran/DOTA1_yolo12x_hbb_split1024_gap200
```

Reports are written to
`work_dirs/dataset_audits/DOTA1_yolo12x_hbb_split1024_gap200/`. The source ID
overlap, image hash overlap, and same-name overlap must all be zero. If source
IDs overlap, split the original DOTA images first and regenerate the patches;
do not randomly split an already patched dataset.

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

### AP75 and AP50:95 re-evaluation

Completed runs can be re-evaluated without retraining. The helper only accepts
work directories whose logs contain the epoch-80 validation result, selects
the saved best checkpoint, and evaluates IoU thresholds 0.50 through 0.95:

```bash
CUDA_VISIBLE_DEVICES=0,1,2 python \
  tools/analysis_tools/eval_yolov10_hbb_best.py --sizes s m l x --gpus 3
```

Use `--dry-run` first to inspect checkpoint selection. Results are written to
`work_dirs/yolov10_hbb_ap50_95_summary.csv`. `AP75` is the 0.75 threshold
result. `voc_map50_95` is the mean VOC AP over ten IoU thresholds from 0.50 to
0.95; it is not the full COCO area/maxDets metric.
