"""Label the real multi-patty videos by propagating detections through time.

Neither detector sees a patty in every frame: the fine-tuned one loses dark
crusts and cheese-covered patties, the zero-shot one scores at its noise floor.
But patties do not move. A position confirmed by detections earlier in the
video is still a patty when the crust darkens or a cheese slice lands on it —
so we cluster the union of both detectors' boxes into stationary slots and
stamp every sampled frame inside a slot's lifetime, detection or not.

Split is by time, not chance: the first 70% of each video trains, the last 30%
validates. Empty-griddle frames become explicit negatives.

Usage: python src/build_real_dataset.py --out data/detector_real
"""
import argparse
import math
import shutil
from pathlib import Path

import cv2

import sys
sys.path.insert(0, str(Path(__file__).parent))
from config import Config

VIDEOS = [("data/IMG_6635.mov", 1.8), ("data/IMG_6637.mov", 0.7)]
GAP_TOL_S = 18.0          # long bridges glue removed->new placements into phantoms
TRAIN_FRAC = 0.7


class Slot:
    def __init__(self, t, box):
        self.dets = [(t, box)]           # (t, (cx, cy, w, h)) normalized

    def pos(self):
        return self.dets[-1][1][:2]

    def near(self, box):
        cx, cy = self.pos()
        w = self.dets[-1][1][2]
        return math.hypot(box[0] - cx, box[1] - cy) < 0.7 * w

    def box_at(self, t):
        prev = [d for d in self.dets if d[0] <= t]
        return (prev[-1] if prev else self.dets[0])[1]

    def active(self, t):
        t0, t1 = self.dets[0][0], self.dets[-1][0]
        if not (t0 <= t <= t1):
            return False
        return min(abs(t - td) for td, _ in self.dets) <= GAP_TOL_S


def collect(path, step, cfg, ft, world):
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    dur = cap.get(cv2.CAP_PROP_FRAME_COUNT) / fps
    frames, slots = [], []
    t = 0.0
    while t < dur:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(t * fps))
        ok, frame = cap.read()
        if not ok:
            break
        fh, fw = frame.shape[:2]
        boxes = []
        # third teacher: the far strip up-scaled. Perspective shrinks and
        # flattens distant patties below what full-frame passes can see.
        cx0, cy0 = int(0.35 * fw), 0
        crop = frame[0:int(0.60 * fh), cx0:fw]
        passes = [(ft, 0.10, frame, 0, 0), (world, cfg.conf, frame, 0, 0),
                  (world, cfg.conf, crop, cx0, cy0)]
        for model, conf, img, ox, oy in passes:
            r = model.predict(img, conf=conf, iou=cfg.iou, imgsz=960,
                              device=cfg.device, agnostic_nms=True, verbose=False)[0]
            if r.boxes is None:
                continue
            for bx1, by1, bx2, by2 in r.boxes.xyxy.tolist():
                x1, y1, x2, y2 = bx1 + ox, by1 + oy, bx2 + ox, by2 + oy
                w, h = x2 - x1, y2 - y1
                if w < 8 or h < 8 or max(w / h, h / w) > cfg.max_aspect:
                    continue
                if not (0.03 <= max(w, h) / max(fw, fh) <= cfg.max_size_frac):
                    continue
                b = ((x1 + x2) / 2 / fw, (y1 + y2) / 2 / fh, w / fw, h / fh)
                if not any(abs(b[0] - q[0]) < 0.4 * q[2] and abs(b[1] - q[1]) < 0.4 * q[3]
                           for q in boxes):
                    boxes.append(b)
        for b in boxes:
            for s in slots:
                if s.near(b):
                    s.dets.append((t, b))
                    break
            else:
                slots.append(Slot(t, b))
        frames.append((t, frame))
        t += step
    cap.release()
    # stationary patties collect dozens of confirmations; noise clusters don't
    slots = [s for s in slots if len(s.dets) >= 8]
    return frames, slots


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/detector_real")
    args = ap.parse_args()
    out = Path(args.out)
    if out.exists():
        shutil.rmtree(out)
    for split in ("train", "val"):
        (out / "images" / split).mkdir(parents=True)
        (out / "labels" / split).mkdir(parents=True)

    cfg = Config()
    from ultralytics import YOLO, YOLOWorld
    ft = YOLO("runs/detect/runs/detect/patty_v5/weights/best.pt")
    world = YOLOWorld(cfg.model_name)
    world.set_classes(list(cfg.classes))

    tot = {"train": [0, 0], "val": [0, 0]}
    for path, step in VIDEOS:
        stem = Path(path).stem
        frames, slots = collect(path, step, cfg, ft, world)
        dur = frames[-1][0]
        cut = dur * TRAIN_FRAC
        neg_budget = 25
        for t, frame in frames:
            split = "train" if t <= cut else "val"
            lines = [f"0 {b[0]:.6f} {b[1]:.6f} {b[2]:.6f} {b[3]:.6f}"
                     for b in (s.box_at(t) for s in slots if s.active(t))]
            if not lines:
                if split != "train" or neg_budget <= 0:
                    continue
                neg_budget -= 1                    # explicit empty-griddle negative
            name = f"{stem}_{int(t * 10):06d}"
            cv2.imwrite(str(out / "images" / split / f"{name}.jpg"), frame,
                        [cv2.IMWRITE_JPEG_QUALITY, 90])
            (out / "labels" / split / f"{name}.txt").write_text("\n".join(lines))
            tot[split][0] += 1
            tot[split][1] += len(lines)
        print(f"{stem}: слотов {len(slots)}, кадров {len(frames)}")
    print(f"train {tot['train'][0]} кадров / {tot['train'][1]} боксов; "
          f"val {tot['val'][0]} / {tot['val'][1]}")


if __name__ == "__main__":
    main()
