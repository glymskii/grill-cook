"""Per-patty side timing for a griddle full of patties.

Identity comes from the detector's tracker plus a stitching pass; the side
resting on the surface comes from the face classifier, exactly as in the
single-patty version. Each track keeps its own timers, so patties placed and
flipped at different moments are measured independently.

Usage:
  python src/analyze_multi.py video.mp4 --model models/face_clf.joblib \
      --out out/multi --fps 25 [--render]
"""
import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import joblib
import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms

from config import Config
from detector import PattyDetector

NORM = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
PREP = transforms.Compose([transforms.ToPILImage(), transforms.Resize((224, 224)),
                           transforms.ToTensor(), NORM])
SIDE_OF_FACE = {"FACE_A": "B", "FACE_B": "A"}
PALETTE = [(66, 133, 244), (52, 168, 83), (251, 188, 5), (234, 67, 53),
           (171, 71, 188), (0, 172, 193), (255, 112, 67), (158, 157, 36)]


def fmt(s):
    return f"{int(s // 60)}:{s % 60:04.1f}"


class Track:
    _next = 1

    def __init__(self, frame_idx, det):
        self.id = Track._next
        Track._next += 1
        self.first = self.last = frame_idx
        self.xyxy = det.xyxy
        self.obs = []          # (frame_idx, face label)
        self.alive = True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--model", default="models/face_clf.joblib")
    ap.add_argument("--out", default="out/multi")
    ap.add_argument("--fps", type=float, default=25.0)
    ap.add_argument("--conf", type=float, default=0.05)
    ap.add_argument("--min-size-frac", type=float, default=0.055)
    ap.add_argument("--smooth-s", type=float, default=0.32)
    ap.add_argument("--commit-s", type=float, default=0.40)
    ap.add_argument("--lost-s", type=float, default=3.0)
    ap.add_argument("--match-diam", type=float, default=0.7,
                    help="association radius, in patty diameters")
    ap.add_argument("--min-prob", type=float, default=0.55,
                    help="below this the crop is treated as unreadable, not as a face")
    ap.add_argument("--render", action="store_true")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = Config()
    cfg.min_size_frac = args.min_size_frac
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    backbone.fc = nn.Identity()
    backbone.eval().to(device)
    clf = joblib.load(args.model)["clf"]

    det = PattyDetector(cfg)
    tracks, raw2track = [], {}
    per_frame = defaultdict(list)      # frame -> [(track_id, box, face)]
    last_idx = 0

    for frame_idx, frame, dets in det.stream(args.video):
        last_idx = frame_idx
        for t in tracks:
            if t.alive and (frame_idx - t.last) / args.fps > args.lost_s:
                t.alive = False

        # Associate by position, not by the detector's own ids: ByteTrack recycles
        # them, and a recycled id silently teleports one patty's history onto
        # another. Distance is decisive here because patties do not move between
        # frames — they only vanish briefly while a hand passes over them.
        pairs = []
        for di, d in enumerate(dets):
            for ti, cand in enumerate(tracks):
                if not cand.alive or cand.last >= frame_idx:
                    continue
                cx = (cand.xyxy[0] + cand.xyxy[2]) / 2
                cy = (cand.xyxy[1] + cand.xyxy[3]) / 2
                dist = np.hypot(d.center[0] - cx, d.center[1] - cy)
                if dist < args.match_diam * d.diameter:
                    pairs.append((dist, di, ti))
        pairs.sort()
        det2track, taken = {}, set()
        for dist, di, ti in pairs:
            if di in det2track or ti in taken:
                continue
            det2track[di], _ = tracks[ti], taken.add(ti)

        for di, d in enumerate(dets):
            t = det2track.get(di)
            if t is None:
                t = Track(frame_idx, d)
                tracks.append(t)
            if d.raw_id != -1:
                raw2track[d.raw_id] = t
            t.last, t.xyxy = frame_idx, d.xyxy

            x1, y1, x2, y2 = d.xyxy
            pad = 0.12 * d.diameter
            fh, fw = frame.shape[:2]
            crop = frame[int(max(0, y1 - pad)):int(min(fh, y2 + pad)),
                         int(max(0, x1 - pad)):int(min(fw, x2 + pad))]
            face = None
            if crop.size:
                crop = cv2.resize(crop, (128, 128))
                x = PREP(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))[None].to(device)
                with torch.no_grad():
                    f = backbone(x).cpu().numpy()
                proba = clf.predict_proba(f)[0]
                k = int(np.argmax(proba))
                # a hand or spatula over the patty produces a crop that matches
                # nothing well; that low confidence is the handling signal now
                face = (str(clf.classes_[k]) if proba[k] >= args.min_prob
                        else "HANDLING")
            t.obs.append((frame_idx, face))
            per_frame[frame_idx].append((t.id, d.xyxy, face))

        if frame_idx % 250 == 0:
            print(f"  {fmt(frame_idx / args.fps)}  tracks={len(tracks)}", flush=True)

    half = max(1, int(args.smooth_s * args.fps / 2))
    results = []
    for t in sorted(tracks, key=lambda q: q.first):
        if (t.last - t.first) / args.fps < 1.0 or len(t.obs) < 8:
            continue
        labels = [f for _, f in t.obs]
        sm = []
        for i in range(len(labels)):
            win = [l for l in labels[max(0, i - half): i + half + 1] if l]
            sm.append(Counter(win).most_common(1)[0][0] if win else None)

        side, timers, off = None, {"A": 0.0, "B": 0.0}, 0.0
        flips, pending, pending_t = 0, None, None
        dt = 1.0 / args.fps
        prev_i = t.obs[0][0]
        for (fi, _), face in zip(t.obs, sm):
            gap = (fi - prev_i) * dt
            prev_i = fi
            if face in SIDE_OF_FACE:
                cand = SIDE_OF_FACE[face]
                if side is None:
                    side = cand
                elif cand != side:
                    if pending != cand:
                        pending, pending_t = cand, fi * dt
                    elif fi * dt - pending_t >= args.commit_s:
                        side, pending, flips = cand, None, flips + 1
                else:
                    pending = None
                timers[side] += min(gap, dt * 3)
            elif face == "TOPPING":
                if side:
                    timers[side] += min(gap, dt * 3)
            else:
                off += min(gap, dt * 3)
        results.append({
            "patty": t.id,
            "placed_t": round(t.first / args.fps, 2),
            "removed_t": round(t.last / args.fps, 2),
            "side_A_s": round(timers["A"], 1), "side_B_s": round(timers["B"], 1),
            "off_s": round(off, 1), "flips": flips,
            "x": round((t.xyxy[0] + t.xyxy[2]) / 2), "y": round((t.xyxy[1] + t.xyxy[3]) / 2),
        })

    print(f"\n=== {len(results)} PATTIES ===")
    for r in results:
        print(f"  #{r['patty']:<3d} placed {r['placed_t']:6.2f}  removed {r['removed_t']:6.2f}"
              f"  A {r['side_A_s']:6.1f}  B {r['side_B_s']:6.1f}  flips {r['flips']}")
    (out_dir / "result.json").write_text(json.dumps(results, indent=2, ensure_ascii=False))

    if args.render:
        render(args, per_frame, results, out_dir)
    print(f"\nwrote {out_dir}/result.json")


def render(args, per_frame, results, out_dir):
    cap = cv2.VideoCapture(args.video)
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    vw = cv2.VideoWriter(str(out_dir / "annotated_raw.mp4"),
                         cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (W, H))
    keep = {r["patty"] for r in results}
    s = H / 1920.0
    i = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        for tid, box, face in per_frame.get(i, []):
            if tid not in keep:
                continue
            x1, y1, x2, y2 = [int(v) for v in box]
            col = PALETTE[(tid - 1) % len(PALETTE)]
            col = (col[2], col[1], col[0])
            cv2.rectangle(frame, (x1, y1), (x2, y2), col, 2)
            lbl = f"#{tid} {(face or '?').replace('FACE_', '')}"
            cv2.putText(frame, lbl, (x1, y1 - 6), cv2.FONT_HERSHEY_SIMPLEX,
                        0.8 * s, (0, 0, 0), 4, cv2.LINE_AA)
            cv2.putText(frame, lbl, (x1, y1 - 6), cv2.FONT_HERSHEY_SIMPLEX,
                        0.8 * s, col, 2, cv2.LINE_AA)
        cv2.putText(frame, f"t={fmt(i / args.fps)}", (12, int(50 * s)),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.1 * s, (0, 0, 0), 5, cv2.LINE_AA)
        cv2.putText(frame, f"t={fmt(i / args.fps)}", (12, int(50 * s)),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.1 * s, (255, 255, 255), 2, cv2.LINE_AA)
        vw.write(frame)
        i += 1
    vw.release()
    cap.release()


if __name__ == "__main__":
    main()
