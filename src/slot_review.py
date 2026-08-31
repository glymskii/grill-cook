"""Run the slot engine on a video and lay its verdicts out for human review.

Each strip is the SAME place at -4s, -1s, +2s, +5s around the verdict, which is
what makes this review trustworthy: the old flip review centred its crops on a
drifting track, so a strip often showed a neighbour rather than the patty the
verdict was about. A pinned crop cannot do that.

Usage: python src/slot_review.py --video data/IMG_6635.mov \
         --dets data/mot_dets_full_faces.json --tag long \
         --targets 270,150,15,20 --pick flip
"""
import argparse
import json
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "app"))
import slots                                           # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--dets", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--targets", default="42,35,12,12")
    ap.add_argument("--pick", default="flip")
    ap.add_argument("--limit", type=int, default=12)
    args = ap.parse_args()

    ta, tb, te, tl = (float(v) for v in args.targets.split(","))
    slots.EVENTS = Path(tempfile.mkdtemp()) / "e.jsonl"
    eng = slots.SlotEngine({"A": ta, "B": tb, "tol_early": te, "tol_late": tl})
    events = []
    eng.on_event = lambda e: events.append(
        dict(e, x=getattr(eng.slots.get(e["pid"]), "ax", None),
             y=getattr(eng.slots.get(e["pid"]), "ay", None),
             r=getattr(eng.slots.get(e["pid"]), "ar", None)))

    rows = json.loads(Path(args.dets).read_text())
    cap = cv2.VideoCapture(args.video)
    src_fps, src_i = cap.get(cv2.CAP_PROP_FPS), 0
    for row in rows:
        want = int(row["t"] * src_fps)
        while src_i < want:
            cap.grab()
            src_i += 1
        ok, frame = cap.read()
        src_i += 1
        eng.update([tuple(d) for d in row["dets"]], row["t"], frame if ok else None)
    out = ROOT / f"data/slot_events_{args.tag}.json"
    out.write_text(json.dumps(events))
    kinds = {}
    for e in events:
        kinds[e["type"]] = kinds.get(e["type"], 0) + 1
    print(f"{args.tag}: {kinds} -> {out.name}")

    picked = [e for e in events if e["type"] == args.pick and e.get("x")][:args.limit]
    strips = []
    for e in picked:
        t0 = e["ts_video"]
        row = []
        for tag, t in (("-4s", t0 - 6), ("-1s", t0 - 3), ("+2s", t0 + 1), ("+5s", t0 + 4)):
            cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, int(t * src_fps)))
            ok, f = cap.read()
            if not ok:
                continue
            fh, fw = f.shape[:2]
            R = int(e["r"] * fw * 1.3)
            cx, cy = int(e["x"] * fw), int(e["y"] * fh)
            c = f[max(0, cy - R):cy + R, max(0, cx - R):cx + R]
            if c.size == 0:
                continue
            c = cv2.resize(c, (170, 170))
            cv2.putText(c, tag, (5, 20), cv2.FONT_HERSHEY_DUPLEX, 0.55, (255, 255, 255), 1)
            row.append(c)
        if len(row) < 4:
            continue
        img = np.hstack(row)
        label = np.zeros((28, img.shape[1], 3), np.uint8)
        cv2.putText(label, f"slot {e['pid']}  t={t0:.0f}s  {e['type']}"
                          f"  {e.get('via', '')} {e.get('grade', '')}",
                    (6, 20), cv2.FONT_HERSHEY_DUPLEX, 0.52, (255, 255, 255), 1)
        strips.append(np.vstack([label, img]))
    cap.release()
    if strips:
        path = f"/tmp/slotrev_{args.tag}_{args.pick}.jpg"
        cv2.imwrite(path, np.vstack(strips))
        print(f"{len(strips)} strips -> {path}")


if __name__ == "__main__":
    main()
