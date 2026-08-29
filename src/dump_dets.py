"""Dump preprocessed detections for MOT labeling and offline tuning.

One GPU pass per window; every downstream step (GT building, IDF1 grid search)
then replays these at CPU speed. Preprocessing mirrors the live pipeline:
zone, dedup, size gate, Lab colour.
"""
import json
import os
import sys
from collections import deque
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "app"))
from config import Config                              # noqa: E402
from pipeline import dedup_dets, in_poly, lab_of, size_gate  # noqa: E402

ROI = [[0.215, 0.70], [0.415, 0.085], [0.885, 0.165], [0.695, 0.92]]
import sys as _sys
WINDOWS = ([(_sys.argv[1], float(_sys.argv[2]), float(_sys.argv[3]))]
           if len(_sys.argv) > 3 else
           [("tune", 120.0, 300.0), ("holdout", 620.0, 740.0)])
FPS = 5.0

cfg = Config()
from ultralytics import YOLO
model = YOLO("runs/detect/runs/detect/patty_v6/weights/best.pt")

cap = cv2.VideoCapture(os.environ.get("SRC", "data/IMG_6635.mov"))
src_fps = cap.get(cv2.CAP_PROP_FPS)
for name, t0, t1 in WINDOWS:
    radii = deque(maxlen=180)
    rows = []
    t = t0
    while t < t1:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(t * src_fps))
        ok, frame = cap.read()
        if not ok:
            break
        fh, fw = frame.shape[:2]
        r = model.predict(frame, conf=0.15, iou=cfg.iou, imgsz=960, device=cfg.device,
                          agnostic_nms=True, verbose=False)[0]
        dets = []
        if r.boxes is not None:
            for (x1, y1, x2, y2), cf in zip(r.boxes.xyxy.tolist(), r.boxes.conf.tolist()):
                w, h = x2 - x1, y2 - y1
                if w < 8 or h < 8 or max(w / h, h / w) > cfg.max_aspect:
                    continue
                if max(w, h) / max(fw, fh) > cfg.max_size_frac:
                    continue
                dets.append(((x1 + x2) / 2 / fw, (y1 + y2) / 2 / fh, (w + h) / 4 / fw,
                             round(cf, 3), lab_of(frame, x1, y1, x2, y2)))
        dets = dedup_dets(dets)
        dets, _ = size_gate(dets, radii)
        dets = [d for d in dets if in_poly(d[0], d[1], ROI)]
        rows.append({"t": round(t, 2), "dets": dets})
        t += 1.0 / FPS
    out = ROOT / f"data/mot_dets_{name}.json"
    out.write_text(json.dumps(rows))
    print(f"{name}: {len(rows)} кадров -> {out}")
cap.release()
