"""Mine crops of the things that are NOT a patty but sit where one should be.

The slot engine gives this for free. A pinned place with a high disturbance is,
almost by definition, a place with a hand, a glove or a spatula over it — so the
episodes we already detect are a negative miner. What that misses is food that
is not a patty: on the long shift a sausage is rolled across the griddle and the
detector claims it, which is why two flip verdicts fired on it. Those come from
sweeping detection crops inside named windows.

Crops are framed exactly like the training crops (1.15 r) so the classifier sees
the same geometry it will see in production.

Usage: python src/mine_negatives.py --video data/IMG_6635.mov \
         --dets data/mot_dets_full_faces.json --tag long \
         --targets 270,150,15,20 --windows 440-470,530-575
"""
import argparse
import json
import sys
import tempfile
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "app"))
import slots                                           # noqa: E402

OUT = ROOT / "data/neg_candidates"


def save(frame, x, y, r, name):
    fh, fw = frame.shape[:2]
    R = int(r * fw * 1.15)
    cx, cy = int(x * fw), int(y * fh)
    c = frame[max(0, cy - R):cy + R, max(0, cx - R):cx + R]
    if c.size == 0 or min(c.shape[:2]) < 16:
        return False
    cv2.imwrite(str(OUT / name), c)
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--dets", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--targets", default="42,35,12,12")
    ap.add_argument("--windows", default="", help="t0-t1,t0-t1: sweep every detection here")
    ap.add_argument("--hot", type=float, default=0.6)
    ap.add_argument("--gap", type=float, default=1.5, help="seconds between crops of one slot")
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    ta, tb, te, tl = (float(v) for v in args.targets.split(","))
    slots.EVENTS = Path(tempfile.mkdtemp()) / "e.jsonl"
    eng = slots.SlotEngine({"A": ta, "B": tb, "tol_early": te, "tol_late": tl})
    windows = [tuple(float(v) for v in w.split("-"))
               for w in args.windows.split(",") if w]

    rows = json.loads(Path(args.dets).read_text())
    cap = cv2.VideoCapture(args.video)
    src_fps, src_i = cap.get(cv2.CAP_PROP_FPS), 0
    last: dict[int, float] = {}
    n_hot = n_win = 0
    for row in rows:
        want = int(row["t"] * src_fps)
        while src_i < want:
            cap.grab()
            src_i += 1
        ok, frame = cap.read()
        src_i += 1
        if not ok:
            break
        eng.update([tuple(d) for d in row["dets"]], row["t"], frame)
        t10 = int(row["t"] * 10)
        for s in eng.slots.values():
            if not s.anchored or s.disturb < args.hot:
                continue
            if row["t"] - last.get(s.sid, -99) < args.gap:
                continue
            last[s.sid] = row["t"]
            if save(frame, s.ax, s.ay, s.ar, f"hot{args.tag}_{t10:06d}_{s.sid}.jpg"):
                n_hot += 1
        if any(a <= row["t"] <= b for a, b in windows):
            for i, d in enumerate(row["dets"]):
                if save(frame, d[0], d[1], d[2], f"win{args.tag}_{t10:06d}_{i}.jpg"):
                    n_win += 1
    cap.release()
    print(f"{args.tag}: {n_hot} crops from disturbed places, {n_win} from windows "
          f"-> {OUT}")


if __name__ == "__main__":
    main()
