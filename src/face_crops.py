"""Extract patty-face crops for the face classifier.

Positions come from the detection dumps, so crops land on patties rather than
on a grid. Sampling is sparse in time (neighbouring frames are near-duplicates
and would only inflate the set while leaking between splits).

Usage: python src/face_crops.py
"""
import json
import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parent.parent
SOURCES = [("data/mot_dets_full.json", "data/IMG_6635.mov", "long", 4.0),
           ("data/mot_dets_smash.json", "data/IMG_6637.mov", "smash", 2.0)]
OUT = ROOT / "data/faces"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    n = 0
    for dets_p, vid_p, tag, step in SOURCES:
        rows = json.loads((ROOT / dets_p).read_text())
        cap = cv2.VideoCapture(str(ROOT / vid_p))
        fps = cap.get(cv2.CAP_PROP_FPS)
        next_t = 0.0
        for r in rows:
            if r["t"] < next_t:
                continue
            next_t = r["t"] + step
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(r["t"] * fps))
            ok, fr = cap.read()
            if not ok:
                continue
            fh, fw = fr.shape[:2]
            for i, d in enumerate(r["dets"]):
                x, y, rad = d[0], d[1], d[2]
                R = int(rad * fw * 1.15)
                cx, cy = int(x * fw), int(y * fh)
                c = fr[max(0, cy - R):cy + R, max(0, cx - R):cx + R]
                if c.shape[0] < 24 or c.shape[1] < 24:
                    continue
                cv2.imwrite(str(OUT / f"{tag}_{int(r['t']*10):06d}_{i}.jpg"),
                            cv2.resize(c, (96, 96)), [cv2.IMWRITE_JPEG_QUALITY, 92])
                n += 1
        cap.release()
        print(f"{tag}: готово")
    print(f"кропов: {n} -> {OUT}")


if __name__ == "__main__":
    main()
