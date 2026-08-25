"""Auto-label patty boxes with the open-vocabulary model to train a real one.

YOLO-World localises patties well at conf 0.02 — it just cannot score them
above its own noise floor, which is exactly why it is unusable in production and
exactly why it makes a good annotator. Run it low, keep only boxes that pass the
size and roundness gates, and the result is training data at no labelling cost.

Split is by source video, never by frame: frames from one clip are near
duplicates, so a random split would leak and report a score the model has not
earned.

Usage: python src/build_detector_dataset.py --out data/detector
"""
import argparse
import shutil
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLOWorld

from config import Config

# (path, sampling step in seconds, split)
SOURCES = [
    ("data/Котлета Жарка.mp4", 2.0, "train"),
    ("data/griddle_15patties_50s.mp4", 0.35, "train"),
    ("data/kling/place.mp4", 0.2, "train"),
    ("data/kling/flip.mp4", 0.2, "train"),
    ("data/kling/cheese.mp4", 0.2, "train"),
    ("data/kling/mixed.mp4", 0.2, "train"),
    ("data/frozen_patties_pan.mp4", 1.0, "train"),
    ("data/kotlety_long.mp4", 3.0, "train"),
    ("data/fiveguys.mp4", 4.0, "train"),
    # held out entirely: a different kitchen the model never trains on
    ("data/kling/remove.mp4", 0.2, "val"),
    ("data/whitemanna.mp4", 4.0, "val"),
    ("data/kotlety_chicken.mp4", 1.5, "val"),
]

LABEL_CONF = 0.02


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/detector")
    ap.add_argument("--max-per-video", type=int, default=400)
    args = ap.parse_args()

    out = Path(args.out)
    if out.exists():
        shutil.rmtree(out)
    for split in ("train", "val"):
        (out / "images" / split).mkdir(parents=True, exist_ok=True)
        (out / "labels" / split).mkdir(parents=True, exist_ok=True)

    cfg = Config()
    model = YOLOWorld(cfg.model_name)
    model.set_classes(list(cfg.classes))

    totals = {"train": [0, 0], "val": [0, 0]}   # [images, boxes]
    for path, step, split in SOURCES:
        if not Path(path).exists():
            print(f"  ПРОПУСК (нет файла): {path}")
            continue
        stem = Path(path).stem.replace(" ", "_")
        cap = cv2.VideoCapture(path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        dur = (cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0) / fps
        n_img = n_box = 0
        t = 0.0
        while t < dur and n_img < args.max_per_video:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(round(t * fps)))
            ok, frame = cap.read()
            if not ok:
                break
            t += step
            r = model.predict(frame, conf=LABEL_CONF, iou=cfg.iou, imgsz=cfg.imgsz,
                              device=cfg.device, agnostic_nms=True, verbose=False)[0]
            if r.boxes is None or not len(r.boxes):
                continue
            fh, fw = frame.shape[:2]
            lines = []
            for b in r.boxes.xyxy.tolist():
                x1, y1, x2, y2 = b
                w, h = x2 - x1, y2 - y1
                if w <= 4 or h <= 4:
                    continue
                if not (0.045 <= max(w, h) / max(fw, fh) <= cfg.max_size_frac):
                    continue
                if max(w / h, h / w) > cfg.max_aspect:
                    continue
                cx, cy = (x1 + x2) / 2 / fw, (y1 + y2) / 2 / fh
                lines.append(f"0 {cx:.6f} {cy:.6f} {w / fw:.6f} {h / fh:.6f}")
            if not lines:
                continue
            name = f"{stem}_{int(t * 100):07d}"
            cv2.imwrite(str(out / "images" / split / f"{name}.jpg"), frame,
                        [cv2.IMWRITE_JPEG_QUALITY, 88])
            (out / "labels" / split / f"{name}.txt").write_text("\n".join(lines))
            n_img += 1
            n_box += len(lines)
        cap.release()
        totals[split][0] += n_img
        totals[split][1] += n_box
        print(f"  {stem:28s} [{split}] {n_img:4d} кадров, {n_box:5d} боксов")

    (out / "data.yaml").write_text(
        f"path: {out.resolve()}\ntrain: images/train\nval: images/val\n"
        "names:\n  0: patty\n")
    print(f"\ntrain: {totals['train'][0]} кадров / {totals['train'][1]} боксов")
    print(f"val:   {totals['val'][0]} кадров / {totals['val'][1]} боксов")
    print(f"wrote {out}/data.yaml")


if __name__ == "__main__":
    main()
