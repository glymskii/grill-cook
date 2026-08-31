"""Can a pinned crop tell us when someone touched the patty, and what changed?

This is the part of the proposed pipeline that has no detector behind it today:
the event detector. Once a place is fixed, it does not need one. The same pixels
are there every frame, so a plain difference against the slot's own quiet
reference says when a hand, a spatula or a press came over it — and the colour
of the same crop before and after that disturbance says whether the patty was
turned or merely passed by.

Writes data/anchor_probe_<tag>.json: per slot, a timeline of
(t, disturbance, L, a, b) plus the episodes the disturbance carves out.

Usage: python src/anchor_probe.py --video data/IMG_6637.mov \
           --dets data/mot_dets_smash_faces.json --tag smash
"""
import argparse
import json
import statistics as st
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "app"))
import slots                                           # noqa: E402


def crop_of(frame, x, y, r, pad=1.0):
    fh, fw = frame.shape[:2]
    R = max(6, int(r * fw * pad))
    cx, cy = int(x * fw), int(y * fh)
    c = frame[max(0, cy - R):cy + R, max(0, cx - R):cx + R]
    return cv2.resize(c, (64, 64)) if c.size else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--dets", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--fps", type=float, default=5.0)
    ap.add_argument("--targets", default="42,35,12,12")
    args = ap.parse_args()

    ta, tb, te, tl = (float(v) for v in args.targets.split(","))
    slots.EVENTS = Path(tempfile.mkdtemp()) / "e.jsonl"
    eng = slots.SlotEngine({"A": ta, "B": tb, "tol_early": te, "tol_late": tl})
    rows = json.loads(Path(args.dets).read_text())

    cap = cv2.VideoCapture(args.video)
    src_fps = cap.get(cv2.CAP_PROP_FPS)
    tl_by_slot: dict[int, list] = {}
    prev_crops: dict[int, np.ndarray] = {}
    src_i = 0
    for row in rows:
        want = int(row["t"] * src_fps)
        while src_i < want:
            cap.grab()
            src_i += 1
        ok, frame = cap.read()
        src_i += 1
        if not ok:
            break
        eng.update([tuple(d) for d in row["dets"]], row["t"])
        lab_full = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        for s in eng.slots.values():
            if not s.anchored:
                continue
            c = crop_of(frame, s.ax, s.ay, s.ar)
            if c is None:
                continue
            prev = prev_crops.get(s.sid)
            # normalise by the crop's own brightness: the camera's auto-exposure
            # swings the whole frame and must not read as someone touching this
            # patty
            dist = 0.0
            if prev is not None:
                a = c.astype(np.float32)
                b = prev.astype(np.float32)
                dist = float(np.abs(a - b).mean() / max(b.mean(), 1.0))
            prev_crops[s.sid] = c
            lab = crop_of(lab_full, s.ax, s.ay, s.ar, pad=0.6)
            m = lab.reshape(-1, 3).mean(axis=0) if lab is not None else (0, 0, 0)
            tl_by_slot.setdefault(s.sid, []).append(
                [round(row["t"], 2), round(dist, 4), round(float(m[0]), 1),
                 round(float(m[1]), 1), round(float(m[2]), 1)])
    cap.release()

    out = ROOT / f"data/anchor_probe_{args.tag}.json"
    out.write_text(json.dumps(tl_by_slot))
    quiet = [x[1] for tlv in tl_by_slot.values() for x in tlv]
    quiet.sort()
    print(f"{len(tl_by_slot)} anchored slots, {len(quiet)} samples -> {out.name}")
    print(f"disturbance: p50 {quiet[len(quiet)//2]:.3f}  p90 {quiet[int(.9*len(quiet))]:.3f}  "
          f"p99 {quiet[int(.99*len(quiet))]:.3f}  max {quiet[-1]:.3f}")


if __name__ == "__main__":
    main()
