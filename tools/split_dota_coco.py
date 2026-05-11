import argparse
import json
import shutil
from pathlib import Path
from collections import defaultdict

from PIL import Image
from tqdm import tqdm


def mkdir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def gen_starts(length, patch_size, stride):
    if length <= patch_size:
        return [0]

    starts = list(range(0, length - patch_size + 1, stride))
    last = length - patch_size

    if starts[-1] != last:
        starts.append(last)

    return starts


def clip_bbox_to_window(bbox, x0, y0, x1, y1):
    bx, by, bw, bh = bbox
    bx1 = bx + bw
    by1 = by + bh

    ix0 = max(bx, x0)
    iy0 = max(by, y0)
    ix1 = min(bx1, x1)
    iy1 = min(by1, y1)

    iw = ix1 - ix0
    ih = iy1 - iy0

    if iw <= 0 or ih <= 0:
        return None, 0.0

    origin_area = max(bw * bh, 1e-6)
    inter_area = iw * ih
    visibility = inter_area / origin_area

    new_bbox = [
        ix0 - x0,
        iy0 - y0,
        iw,
        ih,
    ]

    return new_bbox, visibility


def save_yolo_label(label_path, labels):
    with open(label_path, "w", encoding="utf-8") as f:
        for item in labels:
            f.write(
                f"{item[0]} "
                f"{item[1]:.6f} {item[2]:.6f} "
                f"{item[3]:.6f} {item[4]:.6f}\n"
            )


def process_split(
    src_root,
    dst_root,
    split,
    patch_size,
    gap,
    min_area,
    min_visibility,
    keep_empty,
):
    src_root = Path(src_root)
    dst_root = Path(dst_root)

    src_img_dir = src_root / split / "images"
    src_ann_path = src_root / split / "annotations" / f"{split}.json"

    dst_img_dir = dst_root / split / "images"
    dst_label_dir = dst_root / split / "labels"
    dst_ann_dir = dst_root / split / "annotations"

    mkdir(dst_img_dir)
    mkdir(dst_label_dir)
    mkdir(dst_ann_dir)

    if not src_ann_path.exists():
        raise FileNotFoundError(f"Cannot find annotation file: {src_ann_path}")

    with open(src_ann_path, "r", encoding="utf-8") as f:
        coco = json.load(f)

    images = coco["images"]
    annotations = coco["annotations"]
    categories = coco["categories"]

    anns_by_img = defaultdict(list)
    for ann in annotations:
        anns_by_img[ann["image_id"]].append(ann)

    cat_ids = sorted([cat["id"] for cat in categories])
    cat_id_to_yolo_id = {cat_id: idx for idx, cat_id in enumerate(cat_ids)}

    stride = patch_size - gap
    if stride <= 0:
        raise ValueError("gap must be smaller than patch_size")

    out_images = []
    out_annotations = []

    new_img_id = 1
    new_ann_id = 1

    print(f"\nProcessing {split}...")
    print(f"source images: {src_img_dir}")
    print(f"source json:   {src_ann_path}")
    print(f"patch_size={patch_size}, gap={gap}, stride={stride}")

    for img_info in tqdm(images):
        old_img_id = img_info["id"]
        file_name = img_info["file_name"]

        img_path = src_img_dir / file_name
        if not img_path.exists():
            img_path = src_img_dir / Path(file_name).name

        if not img_path.exists():
            print(f"[WARN] image not found, skip: {file_name}")
            continue

        img = Image.open(img_path).convert("RGB")
        width, height = img.size

        xs = gen_starts(width, patch_size, stride)
        ys = gen_starts(height, patch_size, stride)

        base_name = Path(file_name).stem
        ext = Path(file_name).suffix
        if ext == "":
            ext = ".png"

        old_anns = anns_by_img[old_img_id]

        for y0 in ys:
            for x0 in xs:
                x1 = x0 + patch_size
                y1 = y0 + patch_size

                patch_anns = []
                yolo_labels = []

                for ann in old_anns:
                    if "bbox" not in ann:
                        continue

                    new_bbox, visibility = clip_bbox_to_window(
                        ann["bbox"], x0, y0, x1, y1
                    )

                    if new_bbox is None:
                        continue

                    new_area = new_bbox[2] * new_bbox[3]

                    if new_area < min_area:
                        continue

                    if visibility < min_visibility:
                        continue

                    new_ann = {}

                    for k, v in ann.items():
                        if k in ["id", "image_id", "bbox", "area", "segmentation"]:
                            continue
                        new_ann[k] = v

                    new_ann["id"] = new_ann_id
                    new_ann["image_id"] = new_img_id
                    new_ann["bbox"] = [
                        round(float(new_bbox[0]), 3),
                        round(float(new_bbox[1]), 3),
                        round(float(new_bbox[2]), 3),
                        round(float(new_bbox[3]), 3),
                    ]
                    new_ann["area"] = round(float(new_area), 3)

                    patch_anns.append(new_ann)
                    new_ann_id += 1

                    cat_id = ann["category_id"]
                    yolo_cls = cat_id_to_yolo_id[cat_id]

                    cx = (new_bbox[0] + new_bbox[2] / 2) / patch_size
                    cy = (new_bbox[1] + new_bbox[3] / 2) / patch_size
                    bw = new_bbox[2] / patch_size
                    bh = new_bbox[3] / patch_size

                    yolo_labels.append([yolo_cls, cx, cy, bw, bh])

                if len(patch_anns) == 0 and not keep_empty:
                    continue

                out_name = f"{base_name}__x{x0}_y{y0}{ext}"

                crop_box = (
                    x0,
                    y0,
                    min(x1, width),
                    min(y1, height),
                )
                crop = img.crop(crop_box)

                if crop.size != (patch_size, patch_size):
                    padded = Image.new("RGB", (patch_size, patch_size), (114, 114, 114))
                    padded.paste(crop, (0, 0))
                    crop = padded

                crop.save(dst_img_dir / out_name)

                save_yolo_label(
                    dst_label_dir / f"{Path(out_name).stem}.txt",
                    yolo_labels,
                )

                out_img_info = {
                    "id": new_img_id,
                    "file_name": out_name,
                    "width": patch_size,
                    "height": patch_size,
                }

                out_images.append(out_img_info)

                for item in patch_anns:
                    out_annotations.append(item)

                new_img_id += 1

    out_coco = {
        "images": out_images,
        "annotations": out_annotations,
        "categories": categories,
    }

    if "info" in coco:
        out_coco["info"] = coco["info"]

    if "licenses" in coco:
        out_coco["licenses"] = coco["licenses"]

    dst_json_path = dst_ann_dir / f"{split}.json"

    with open(dst_json_path, "w", encoding="utf-8") as f:
        json.dump(out_coco, f, ensure_ascii=False)

    src_classes = src_root / split / "classes.txt"
    dst_classes = dst_root / split / "classes.txt"

    if src_classes.exists():
        shutil.copy(src_classes, dst_classes)

    print(f"\nDone {split}")
    print(f"output images:      {len(out_images)}")
    print(f"output annotations: {len(out_annotations)}")
    print(f"output json:        {dst_json_path}")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--src-root", type=str, required=True)
    parser.add_argument("--dst-root", type=str, required=True)
    parser.add_argument("--splits", nargs="+", default=["train", "val"])

    parser.add_argument("--patch-size", type=int, default=1024)
    parser.add_argument("--gap", type=int, default=200)

    parser.add_argument("--min-area", type=float, default=16)
    parser.add_argument("--min-visibility", type=float, default=0.25)

    parser.add_argument("--keep-empty", action="store_true")

    args = parser.parse_args()

    for split in args.splits:
        process_split(
            src_root=args.src_root,
            dst_root=args.dst_root,
            split=split,
            patch_size=args.patch_size,
            gap=args.gap,
            min_area=args.min_area,
            min_visibility=args.min_visibility,
            keep_empty=args.keep_empty,
        )


if __name__ == "__main__":
    main()