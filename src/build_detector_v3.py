"""Detector dataset v3: the face classifier cleans what the detector proposes.

Auto-labelling with the detector alone teaches it its own mistakes — grease
marks and spatula edges get boxed and reinforced. The face classifier knows a
patty from a griddle, so every proposal it calls "other" is dropped, and that
region becomes background the next model learns to reject.

Empty-griddle frames go in with no boxes at all: the strongest negative there is.

Usage: python src/build_detector_v3.py
"""
import json
import pickle
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import models, transforms

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "app"))
from config import Config                                # noqa: E402
from face_stats import stats_for                         # noqa: E402
from pipeline import in_poly                             # noqa: E402

OUT = ROOT / "data/detector_v3"
SOURCES = [
    # (video, roi, sample step, empty-until, split-block seconds)
    ("data/IMG_6635.mov", [[0.22,0.72],[0.48,0.09],[0.95,0.20],[0.70,0.97]], 2.0, 80.0),
    ("data/IMG_6637.mov", [[0.215,0.70],[0.415,0.085],[0.885,0.165],[0.695,0.92]], 1.0, 12.0),
]
TF = transforms.Compose([
    transforms.Resize((112, 112)), transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])


def main():
    if OUT.exists():
        shutil.rmtree(OUT)
    for sp in ("train", "val"):
        (OUT / "images" / sp).mkdir(parents=True)
        (OUT / "labels" / sp).mkdir(parents=True)

    cfg = Config()
    from ultralytics import YOLO
    det = YOLO("runs/detect/runs/detect/patty_v6/weights/best.pt")
    with open(ROOT / "data/face_clf.pkl", "rb") as f:
        clf = pickle.load(f)["clf"]
    ref = np.load(ROOT / "data/face_stats.npy")
    mu, sd = ref.mean(0), ref.std(0) + 1e-6
    dev = cfg.device
    net = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    net.fc = torch.nn.Identity(); net.eval().to(dev)

    stats = {"train": [0, 0, 0], "val": [0, 0, 0]}     # frames, boxes, dropped
    for vid, roi, step, empty_until in SOURCES:
        stem = Path(vid).stem
        cap = cv2.VideoCapture(str(ROOT / vid))
        fps = cap.get(cv2.CAP_PROP_FPS)
        dur = cap.get(cv2.CAP_PROP_FRAME_COUNT) / fps
        t = 0.0
        while t < dur:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(t * fps))
            ok, frame = cap.read()
            if not ok:
                break
            split = "train" if int(t // 60) % 4 else "val"   # 3:1 by minute blocks
            fh, fw = frame.shape[:2]
            lines, dropped = [], 0
            if t >= empty_until:
                r = det.predict(frame, conf=0.15, iou=cfg.iou, imgsz=960,
                                device=dev, agnostic_nms=True, verbose=False)[0]
                boxes = [] if r.boxes is None else r.boxes.xyxy.tolist()
                crops, keep = [], []
                for i, (x1, y1, x2, y2) in enumerate(boxes):
                    w, h = x2 - x1, y2 - y1
                    if w < 10 or h < 10 or max(w / h, h / w) > cfg.max_aspect:
                        continue
                    if max(w, h) / max(fw, fh) > cfg.max_size_frac:
                        continue
                    if not in_poly((x1 + x2) / 2 / fw, (y1 + y2) / 2 / fh, roi):
                        continue                      # off the cooking surface
                    R = int(max(w, h) * 0.575)
                    cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
                    c = frame[max(0, cy - R):cy + R, max(0, cx - R):cx + R]
                    if c.shape[0] < 24 or c.shape[1] < 24:
                        continue
                    crops.append(cv2.resize(c, (96, 96))); keep.append((x1, y1, x2, y2))
                if crops:
                    with torch.no_grad():
                        b = torch.stack([TF(Image.fromarray(cv2.cvtColor(c, cv2.COLOR_BGR2RGB)))
                                         for c in crops]).to(dev)
                        emb = net(b).cpu().numpy()
                    st = (np.array([stats_for(c) for c in crops], np.float32) - mu) / sd
                    pred = clf.predict(np.hstack([emb, st * 3.0]))
                    for (x1, y1, x2, y2), cls in zip(keep, pred):
                        if cls == 3:
                            dropped += 1
                            continue                 # griddle, glove, spatula
                        w, h = x2 - x1, y2 - y1
                        lines.append(f"0 {(x1+x2)/2/fw:.6f} {(y1+y2)/2/fh:.6f} "
                                     f"{w/fw:.6f} {h/fh:.6f}")
            name = f"{stem}_{int(t*10):06d}"
            cv2.imwrite(str(OUT / "images" / split / f"{name}.jpg"), frame,
                        [cv2.IMWRITE_JPEG_QUALITY, 88])
            (OUT / "labels" / split / f"{name}.txt").write_text("\n".join(lines))
            stats[split][0] += 1; stats[split][1] += len(lines); stats[split][2] += dropped
            t += step
        cap.release()
        print(f"{stem}: готово")

    (OUT / "data.yaml").write_text(
        f"path: {OUT.resolve()}\ntrain: images/train\nval: images/val\nnames:\n  0: patty\n")
    for sp in ("train", "val"):
        f, b, d = stats[sp]
        print(f"{sp}: {f} кадров, {b} боксов, отброшено классификатором {d}")


if __name__ == "__main__":
    main()
