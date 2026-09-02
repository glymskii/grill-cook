"""Dump preprocessed detections with RF-DETR, in the exact shape dump_dets.py
produces for YOLO, so the same face labeller, engine and ground truth score both.

Preprocessing mirrors the live pipeline: aspect/size filters, Lab colour of the
box, dedup, size gate, ROI. RF-DETR wants RGB.

Usage: SRC=data/IMG_6637.mov WEIGHTS=runs/rfdetr_v4/checkpoint_best_total.pth \
         python src/dump_dets_rfdetr.py smash 0 170 data/mot_dets_smash_rfdetr.json
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
FPS = 5.0

name, t0, t1, out = sys.argv[1], float(sys.argv[2]), float(sys.argv[3]), sys.argv[4]
cfg = Config()
from rfdetr import RFDETRSmall                          # noqa: E402
model = RFDETRSmall(pretrain_weights=os.environ.get(
    "WEIGHTS", str(ROOT / "runs/rfdetr_v4/checkpoint_best_total.pth")))
model.optimize_for_inference() if hasattr(model, "optimize_for_inference") else None

cap = cv2.VideoCapture(os.environ.get("SRC", str(ROOT / "data/IMG_6635.mov")))
src_fps = cap.get(cv2.CAP_PROP_FPS)
radii = deque(maxlen=180)
rows = []
t = t0
while t < t1:
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(t * src_fps))
    ok, frame = cap.read()
    if not ok:
        break
    fh, fw = frame.shape[:2]
    det = model.predict(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), threshold=0.15)
    dets = []
    for (x1, y1, x2, y2), cf in zip(det.xyxy.tolist(), det.confidence.tolist()):
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
    if int(t) % 60 == 0 and abs(t - round(t)) < 1e-6:
        print(f"  {t:.0f}s", flush=True)
cap.release()
Path(out).write_text(json.dumps(rows))
print(f"{name}: {len(rows)} frames, {sum(len(r['dets']) for r in rows)} dets -> {out}")
