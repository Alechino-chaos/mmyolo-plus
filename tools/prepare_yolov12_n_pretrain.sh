#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

SRC="${SRC:-https://github.com/sunsmarterjie/yolov12/releases/download/turbo/yolov12n.pt}"
DST="${DST:-work_dirs/pretrain/yolov12n_coco_mmyolo.pth}"
CONFIG="${CONFIG:-configs/yolov12/yolov12_n.py}"

if [[ -z "${OFFICIAL_YOLOV12_REPO:-}" && -d "$ROOT_DIR/../yolov12/ultralytics" ]]; then
  OFFICIAL_YOLOV12_REPO="$ROOT_DIR/../yolov12"
fi

ARGS=(
  --src "$SRC"
  --dst "$DST"
  --config "$CONFIG"
)

if [[ -n "${OFFICIAL_YOLOV12_REPO:-}" ]]; then
  ARGS+=(--official-repo "$OFFICIAL_YOLOV12_REPO")
fi

python tools/model_converters/yolov12_to_mmyolo.py "${ARGS[@]}"

echo "Prepared YOLOv12-n MMYOLO checkpoint: $DST"
