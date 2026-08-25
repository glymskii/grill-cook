"""Cut labelled patty crops out of the reference cook.

Labels come from the hand-verified timeline in data/kotleta_zharka_gt.json. Each
window says which face was pointing at the camera, so every crop taken inside it
inherits that label.

Sampling is deliberately uneven: face A is only ever visible for about two and a
half seconds in the whole cook, so those windows are sampled at full frame rate
while the long face-B and cheese stretches are thinned out.

Usage: python src/build_face_dataset.py --out data/faces
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLOWorld

from config import Config

VIDEO = "data/Котлета Жарка.mp4"

# (label, t_start, t_end, sampling step in seconds, episode id)
# episode ids keep temporally adjacent near-duplicate frames on the same side of
# the train/test split — a random split would leak and flatter the score
WINDOWS = [
    ("FACE_B",   10.0, 166.90, 1.00, "B_early"),
    ("FACE_B",  173.20, 216.30, 0.60, "B_late"),

    ("FACE_A",  167.55, 168.35, 0.02, "A_1"),
    ("FACE_A",  170.75, 171.35, 0.02, "A_2"),
    ("FACE_A",  217.05, 217.70, 0.02, "A_3"),

    ("HANDLING", 167.25, 167.50, 0.04, "H_1"),
    ("HANDLING", 168.55, 170.60, 0.04, "H_2"),
    ("HANDLING", 171.45, 172.70, 0.04, "H_3"),
    ("HANDLING", 216.50, 217.00, 0.04, "H_4"),
    ("HANDLING", 218.05, 218.80, 0.04, "H_5"),

    ("TOPPING",  219.00, 363.50, 1.50, "T_1"),
]

CROP = 128


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/faces")
    ap.add_argument("--video", default=VIDEO)
    args = ap.parse_args()

    out = Path(args.out)
    for lbl in {w[0] for w in WINDOWS}:
        (out / lbl).mkdir(parents=True, exist_ok=True)

    cfg = Config()
    model = YOLOWorld(cfg.model_name)
    model.set_classes(list(cfg.classes))

    cap = cv2.VideoCapture(args.video)
    fps = cap.get(cv2.CAP_PROP_FPS)
    index = []
    counts = {}

    for label, t0, t1, step, episode in WINDOWS:
        t = t0
        while t <= t1:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(round(t * fps)))
            ok, frame = cap.read()
            if not ok:
                t += step
                continue
            r = model.predict(frame, conf=0.04, imgsz=cfg.imgsz,
                              device=cfg.device, agnostic_nms=True, verbose=False)[0]
            if r.boxes is None or not len(r.boxes):
                t += step
                continue
            fh, fw = frame.shape[:2]
            i = int(np.argmax(r.boxes.conf.tolist()))
            x1, y1, x2, y2 = r.boxes.xyxy.tolist()[i]
            w, h = x2 - x1, y2 - y1
            if max(w, h) / max(fw, fh) > cfg.max_size_frac:
                t += step
                continue
            pad = 0.12 * max(w, h)
            cx1, cy1 = int(max(0, x1 - pad)), int(max(0, y1 - pad))
            cx2, cy2 = int(min(fw, x2 + pad)), int(min(fh, y2 + pad))
            crop = frame[cy1:cy2, cx1:cx2]
            if crop.size == 0:
                t += step
                continue
            crop = cv2.resize(crop, (CROP, CROP))
            name = f"{episode}_{t:08.2f}.jpg".replace(".", "_", 1)
            path = out / label / name
            cv2.imwrite(str(path), crop)
            index.append({"path": str(path), "label": label,
                          "episode": episode, "t": round(t, 2)})
            counts[label] = counts.get(label, 0) + 1
            t += step
    cap.release()

    (out / "index.json").write_text(json.dumps(index, indent=2))
    print("crops per class:", counts, "total", len(index))


if __name__ == "__main__":
    main()
