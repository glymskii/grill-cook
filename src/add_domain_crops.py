"""Harvest labelled crops from a clip where every patty shares one state.

The generated clips are each shot in a single condition — everything raw while
they are being laid down, everything seared while they are being lifted off — so
a time window plus one label covers every patty in frame. That turns a whole
clip into training data without per-object annotation.

Usage:
  python src/add_domain_crops.py data/kling/place.mp4 --label FACE_B \
      --from 3.5 --to 5.0 --episode kling_place --out data/faces
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLOWorld

from config import Config

CROP = 128


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--label", required=True)
    ap.add_argument("--from", dest="t0", type=float, default=0.0)
    ap.add_argument("--to", dest="t1", type=float, default=1e9)
    ap.add_argument("--step", type=float, default=0.2)
    ap.add_argument("--episode", required=True)
    ap.add_argument("--out", default="data/faces")
    ap.add_argument("--min-size-frac", type=float, default=0.05)
    ap.add_argument("--max-per-frame", type=int, default=20)
    args = ap.parse_args()

    out = Path(args.out)
    (out / args.label).mkdir(parents=True, exist_ok=True)
    index_path = out / "index.json"
    index = json.loads(index_path.read_text()) if index_path.exists() else []

    cfg = Config()
    cfg.min_size_frac = args.min_size_frac
    model = YOLOWorld(cfg.model_name)
    model.set_classes(list(cfg.classes))

    cap = cv2.VideoCapture(args.video)
    fps = cap.get(cv2.CAP_PROP_FPS)
    n = 0
    t = args.t0
    while t <= args.t1:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(round(t * fps)))
        ok, frame = cap.read()
        if not ok:
            break
        r = model.predict(frame, conf=0.05, imgsz=cfg.imgsz, device=cfg.device,
                          agnostic_nms=True, verbose=False)[0]
        if r.boxes is not None and len(r.boxes):
            fh, fw = frame.shape[:2]
            boxes = sorted(zip(r.boxes.xyxy.tolist(), r.boxes.conf.tolist()),
                           key=lambda b: -b[1])[:args.max_per_frame]
            for k, (b, _) in enumerate(boxes):
                w, h = b[2] - b[0], b[3] - b[1]
                if w <= 4 or h <= 4:
                    continue
                frac = max(w, h) / max(fw, fh)
                if not (cfg.min_size_frac <= frac <= cfg.max_size_frac):
                    continue
                pad = 0.12 * max(w, h)
                crop = frame[int(max(0, b[1] - pad)):int(min(fh, b[3] + pad)),
                             int(max(0, b[0] - pad)):int(min(fw, b[2] + pad))]
                if crop.size == 0:
                    continue
                crop = cv2.resize(crop, (CROP, CROP))
                name = f"{args.episode}_{t:06.2f}_{k}.jpg".replace(".", "_", 1)
                path = out / args.label / name
                cv2.imwrite(str(path), crop)
                index.append({"path": str(path), "label": args.label,
                              "episode": args.episode, "t": round(t, 2)})
                n += 1
        t += args.step
    cap.release()
    index_path.write_text(json.dumps(index, indent=2))
    print(f"{args.episode}: +{n} crops as {args.label} (index now {len(index)})")


if __name__ == "__main__":
    main()
