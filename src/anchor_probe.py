"""Can a pinned crop tell us when someone touched the patty, and what changed?

This is the part of the proposed pipeline that has no detector behind it today:
the event detector. Once a place is fixed, it does not need one. The same pixels
are there every frame, so a plain difference against the slot's own quiet
reference says when a hand, a spatula or a press came over it — and the colour
of the same crop before and after that disturbance says whether the patty was
turned or merely passed by.

Writes data/anchor_probe_<tag>.json: per slot and per frame,
  t, disturbance, the crop's own L,a,b,spread,grain, the same five for the ring
  of bare griddle around it, and whether the slot was occupied.

Usage: python src/anchor_probe.py --video data/IMG_6637.mov \
           --dets data/mot_dets_smash_faces.json --tag smash
"""
import argparse
import json
import math
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


def patch_stats(small_lab, x, y, r):
    """The same five numbers for the patty's own crop: colour, spread, grain."""
    fh, fw = small_lab.shape[:2]
    R = max(3, int(r * fw))
    cx, cy = int(x * fw), int(y * fh)
    c = small_lab[max(0, cy - R):cy + R, max(0, cx - R):cx + R]
    if c.size == 0:
        return [None] * 5
    m = c.reshape(-1, 3).mean(axis=0)
    L = c[:, :, 0].astype(np.float32)
    return [round(float(m[0]), 1), round(float(m[1]), 1), round(float(m[2]), 1),
            round(float(L.std()), 1),
            round(float(np.abs(cv2.Laplacian(L, cv2.CV_32F)).mean()), 2)]


def bare_griddle(small_lab, slot, others, dets):
    """Bare griddle around this slot, measured in the same frame.

    The obvious reference — what the place looked like before the patty landed —
    does not survive contact with the footage: on the smash video the cook works
    that very spot with a spatula, and the slot is born when the DETECTOR first
    sees the patty, which is seconds after the meat is already there. Sampled
    references came back as spatula and as raw mince.

    The ring around the patty has neither problem. It is griddle by construction,
    it is measured in the same frame so exposure swings cancel, and neighbours
    are cut out of it. On a fully packed griddle too little of it survives, and
    then the slot honestly has no reference.
    """
    fh, fw = small_lab.shape[:2]
    R = slot.ar * fw
    cx, cy = slot.ax * fw, slot.ay * fh
    x0, x1 = int(max(0, cx - 2.1 * R)), int(min(fw, cx + 2.1 * R))
    y0, y1 = int(max(0, cy - 2.1 * R)), int(min(fh, cy + 2.1 * R))
    win = small_lab[y0:y1, x0:x1]
    if win.size == 0:
        return None
    yy, xx = np.ogrid[y0:y1, x0:x1]
    d = np.hypot(xx - cx, yy - cy)
    mask = (d >= 1.15 * R) & (d <= 1.9 * R)
    for o in others:
        if o.sid == slot.sid:
            continue
        od = np.hypot(xx - o.ax * fw, yy - o.ay * fh)
        mask &= od > 1.05 * o.ar * fw
    # neighbours without a slot of their own would otherwise leak meat into the
    # ring and make it rougher than the patty it is supposed to be compared with
    for dx, dy, dr in dets:
        if math.hypot(dx - slot.ax, dy - slot.ay) < 0.2 * slot.ar:
            continue
        od = np.hypot(xx - dx * fw, yy - dy * fh)
        mask &= od > 1.05 * dr * fw
    px = win[mask]
    if len(px) < 150:
        return None
    # colour alone cannot separate a dark seared patty from dark griddle, so the
    # reference carries texture too: meat is grainy, the plate is smooth
    L = win[:, :, 0].astype(np.float32)
    rough = float(np.abs(cv2.Laplacian(L, cv2.CV_32F))[mask].mean())
    m = np.median(px, axis=0)
    return [round(float(m[0]), 1), round(float(m[1]), 1), round(float(m[2]), 1),
            round(float(px[:, 0].std()), 1), round(rough, 2)]


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
    no_ref = 0
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
        small = cv2.resize(lab_full, (640, int(640 * frame.shape[0] / frame.shape[1])))
        anchored = [q for q in eng.slots.values() if q.anchored]
        for s in anchored:
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
            centre = patch_stats(small, s.ax, s.ay, s.ar * 0.6)
            ref = bare_griddle(small, s, anchored,
                               [(d[0], d[1], d[2]) for d in row["dets"]])
            if ref is None:
                no_ref += 1
            tl_by_slot.setdefault(s.sid, []).append(
                [round(row["t"], 2), round(dist, 4)] + centre + (ref or [None] * 5)
                + [int(s.absent_since is None)])
    cap.release()

    out = ROOT / f"data/anchor_probe_{args.tag}.json"
    out.write_text(json.dumps({"tl": tl_by_slot}))
    n = sum(len(v) for v in tl_by_slot.values())
    print(f"samples with no griddle ring left (packed in by neighbours): "
          f"{no_ref} of {n} ({100 * no_ref / max(n, 1):.0f}%)")
    quiet = [x[1] for tlv in tl_by_slot.values() for x in tlv]
    quiet.sort()
    print(f"{len(tl_by_slot)} anchored slots, {len(quiet)} samples -> {out.name}")
    print(f"disturbance: p50 {quiet[len(quiet)//2]:.3f}  p90 {quiet[int(.9*len(quiet))]:.3f}  "
          f"p99 {quiet[int(.99*len(quiet))]:.3f}  max {quiet[-1]:.3f}")


if __name__ == "__main__":
    main()
