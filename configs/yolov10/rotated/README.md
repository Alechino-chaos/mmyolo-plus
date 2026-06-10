# YOLOv10-R DOTA1.0 640/80e

This config set runs YOLOv10-style backbone/neck models with a shared RTMDet-R
rotated head on DOTA1.0 15-class oriented detection.

## Protocol

- Dataset layout: Ultralytics YOLO HBB text labels under
  `images/train`, `images/val`, `labels/train`, and `labels/val`.
- Labels use normalized `class cx cy width height` format.
- DOTA1.0 has 15 classes and no `container-crane`.
- The supplied dataset contains class ID 15 (`container-crane`); the dataset
  adapter ignores that ID so experiments remain on the DOTA1.0 0-14 protocol.
- Input: `640x640`.
- Schedule: `80` epochs, random initialization, `load_from = None`.
- Default batch: `3 GPUs x 4 images/GPU`, `accumulative_counts=3`,
  effective batch `36`.
- Precision: use FP32 on RTX 2080. Do not pass `--amp`; the rotated IoU
  regression path is not numerically stable with FP16 on this GPU generation.
- OOM fallback:
  - `l`: use `yolov10-r_l_syncbn_fast_3xb2-accum6-80e_dota10-640.py`.
  - `x`: first use `yolov10-r_x_syncbn_fast_3xb2-accum6-80e_dota10-640.py`.
  - `x` if still OOM: use
    `yolov10-r_x_syncbn_fast_3xb1-accum12-80e_dota10-640.py`.

The LR schedule is epoch-based on purpose. Do not change it to an iter-based
schedule for OOM fallbacks, otherwise warmup/cosine progress will no longer be
aligned across models.

Set `data_root` in the base config to the actual dataset root. The default is
`data/DOTA1_yolo12x_hbb_split1024/`. HBB labels are converted to zero-angle
rotated boxes in the pipeline; they do not contain the original DOTA object
orientation and therefore cannot reproduce official DOTA OBB evaluation.

## Commands

```bash
bash tools/dist_train.sh configs/yolov10/rotated/yolov10-r_s_syncbn_fast_3xb4-accum3-80e_dota10-640.py 3
bash tools/dist_train.sh configs/yolov10/rotated/yolov10-r_m_syncbn_fast_3xb4-accum3-80e_dota10-640.py 3
bash tools/dist_train.sh configs/yolov10/rotated/yolov10-r_l_syncbn_fast_3xb4-accum3-80e_dota10-640.py 3
bash tools/dist_train.sh configs/yolov10/rotated/yolov10-r_x_syncbn_fast_3xb4-accum3-80e_dota10-640.py 3
```

Use fallback configs only after the matching default config OOMs. Keep FP32
and use the fallback accumulation settings to preserve effective batch 36.

## Measurement

```bash
python tools/analysis_tools/get_flops.py configs/yolov10/rotated/yolov10-r_s_syncbn_fast_3xb4-accum3-80e_dota10-640.py --shape 640 640
```

Record results in `results_template.csv`. `params` and `FLOPs at 640` should
come from `tools/analysis_tools/get_flops.py`; peak memory, training time,
best epoch, and mAP should come from the training/test logs.
