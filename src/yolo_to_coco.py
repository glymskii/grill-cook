"""Convert the combined YOLO-format patty dataset to COCO for RF-DETR.

Merges data/detector and data/detector_real; images are symlinked, not copied.
Usage: python src/yolo_to_coco.py --out data/detector_coco
"""
import argparse
import json
import shutil
from pathlib import Path

import cv2

SOURCES = ["data/detector", "data/detector_real"]
SPLITS = {"train": "train", "val": "valid"}      # rfdetr expects 'valid'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/detector_coco")
    args = ap.parse_args()
    out = Path(args.out)
    if out.exists():
        shutil.rmtree(out)

    for src_split, dst_split in SPLITS.items():
        d = out / dst_split
        d.mkdir(parents=True)
        images, annotations = [], []
        img_id = ann_id = 0
        for src in SOURCES:
            img_dir = Path(src) / "images" / src_split
            if not img_dir.exists():
                continue
            for img_path in sorted(img_dir.glob("*.jpg")):
                im = cv2.imread(str(img_path))
                if im is None:
                    continue
                fh, fw = im.shape[:2]
                img_id += 1
                link = d / f"{img_id:06d}_{img_path.name}"
                link.symlink_to(img_path.resolve())
                images.append({"id": img_id, "file_name": link.name,
                               "width": fw, "height": fh})
                lab = Path(str(img_path).replace("/images/", "/labels/")
                           .replace(".jpg", ".txt"))
                if not lab.exists():
                    continue
                for line in lab.read_text().splitlines():
                    _, cx, cy, w, h = map(float, line.split())
                    bw, bh = w * fw, h * fh
                    x, y = cx * fw - bw / 2, cy * fh - bh / 2
                    ann_id += 1
                    annotations.append({"id": ann_id, "image_id": img_id,
                                        "category_id": 1,
                                        "bbox": [round(x, 1), round(y, 1),
                                                 round(bw, 1), round(bh, 1)],
                                        "area": round(bw * bh, 1),
                                        "iscrowd": 0})
        coco = {"images": images, "annotations": annotations,
                "categories": [{"id": 1, "name": "patty"}]}
        (d / "_annotations.coco.json").write_text(json.dumps(coco))
        print(f"{dst_split}: {len(images)} изображений, {len(annotations)} боксов")


if __name__ == "__main__":
    main()
