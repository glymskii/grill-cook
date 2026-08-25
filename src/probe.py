"""Probe detection quality: sample frames, try prompt sets, save annotated JPGs.

Usage: python probe.py video.mp4 --times 30,60,120 --out out/probe
"""
import argparse
from pathlib import Path

import cv2
from ultralytics import YOLOWorld

PROMPT_SETS = {
    "patty": ["burger patty", "meat patty", "cutlet"],
    "broad": ["burger patty", "piece of meat", "food on pan"],
    "raw": ["raw meat patty", "cooked burger patty", "meatball"],
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--times", default="15,45,90,150,240")
    ap.add_argument("--out", default="out/probe")
    ap.add_argument("--model", default="yolov8l-worldv2.pt")
    ap.add_argument("--conf", type=float, default=0.03)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(args.video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    frames = {}
    for t in [float(x) for x in args.times.split(",")]:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(t * fps))
        ok, fr = cap.read()
        if ok:
            frames[t] = fr
            cv2.imwrite(str(out / f"raw_t{int(t)}.jpg"), fr)
    cap.release()
    print(f"sampled {len(frames)} frames")

    for name, prompts in PROMPT_SETS.items():
        model = YOLOWorld(args.model)  # fresh CPU instance: set_classes+MPS conflict
        model.set_classes(prompts)
        for t, fr in frames.items():
            r = model.predict(fr, conf=args.conf, imgsz=960, device="cpu",
                              verbose=False)[0]
            vis = fr.copy()
            for box, conf, cls in zip(r.boxes.xyxy.tolist(),
                                      r.boxes.conf.tolist(),
                                      r.boxes.cls.int().tolist()):
                x1, y1, x2, y2 = [int(v) for v in box]
                cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 220, 0), 2)
                cv2.putText(vis, f"{prompts[cls]} {conf:.2f}", (x1, y1 - 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 220, 0), 2)
            cv2.imwrite(str(out / f"{name}_t{int(t)}.jpg"), vis)
            print(f"{name} t={t}: {len(r.boxes)} dets")


if __name__ == "__main__":
    main()
