"""Side timing driven by the face classifier.

Reads which face is pointing at the camera on every frame and derives the side
resting on the griddle from it. Nothing here depends on measuring change over a
window, so a burst of flips half a second apart is no harder than a slow one --
the failure mode that made the colour-step detector mis-assign 44 seconds of
this cook.

Usage:
  python src/analyze_faces.py video.mp4 --model models/face_clf.joblib \
      --out out/faces --fps 25 [--render]
"""
import argparse
import json
from collections import Counter
from pathlib import Path

import cv2
import joblib
import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms
from ultralytics import YOLOWorld

from config import Config

NORM = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
PREP = transforms.Compose([transforms.ToPILImage(), transforms.Resize((224, 224)),
                           transforms.ToTensor(), NORM])

# which face is up -> which side is pressed against the griddle
SIDE_OF_FACE = {"FACE_A": "B", "FACE_B": "A"}
COL = {"A": (66, 133, 244), "B": (52, 168, 83)}


def fmt(s: float) -> str:
    return f"{int(s // 60)}:{s % 60:04.1f}"


def smooth(labels, half):
    """Majority vote in a sliding window; kills single-frame flicker."""
    out = []
    for i in range(len(labels)):
        win = [l for l in labels[max(0, i - half): i + half + 1] if l]
        out.append(Counter(win).most_common(1)[0][0] if win else None)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--model", default="models/face_clf.joblib")
    ap.add_argument("--out", default="out/faces")
    ap.add_argument("--fps", type=float, default=25.0)
    ap.add_argument("--start", type=float, default=0.0)
    ap.add_argument("--end", type=float, default=None)
    ap.add_argument("--conf", type=float, default=0.05)
    ap.add_argument("--smooth-s", type=float, default=0.32)
    ap.add_argument("--commit-s", type=float, default=0.40,
                    help="a new side must hold this long before a flip is logged")
    ap.add_argument("--min-prob", type=float, default=0.55,
                    help="below this the crop is unreadable — the handling signal")
    ap.add_argument("--render", action="store_true")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    import subprocess
    clip = str(out_dir / "clip.mp4")
    cmd = ["ffmpeg", "-y", "-v", "error", "-i", args.video, "-ss", str(args.start)]
    if args.end is not None:
        cmd += ["-to", str(args.end)]
    cmd += ["-r", str(args.fps), "-c:v", "libx264", "-preset", "fast",
            "-crf", "18", "-an", clip]
    subprocess.run(cmd, check=True)

    cfg = Config()
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    det = YOLOWorld(cfg.model_name)
    det.set_classes(list(cfg.classes))
    backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    backbone.fc = nn.Identity()
    backbone.eval().to(device)
    bundle = joblib.load(args.model)
    clf = bundle["clf"]

    frames = []
    cap = cv2.VideoCapture(clip)
    i = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        t = args.start + i / args.fps
        r = det.predict(frame, conf=args.conf, imgsz=cfg.imgsz, device=cfg.device,
                        agnostic_nms=True, verbose=False)[0]
        rec = {"t": t, "box": None, "face": None, "p": 0.0}
        if r.boxes is not None and len(r.boxes):
            fh, fw = frame.shape[:2]
            cands = []
            for b, c in zip(r.boxes.xyxy.tolist(), r.boxes.conf.tolist()):
                w, h = b[2] - b[0], b[3] - b[1]
                if w > 4 and h > 4 and \
                        cfg.min_size_frac <= max(w, h) / max(fw, fh) <= cfg.max_size_frac:
                    cands.append((c, b, w, h))
            if cands:
                c, b, w, h = max(cands, key=lambda x: x[0])
                pad = 0.12 * max(w, h)
                x1, y1 = int(max(0, b[0] - pad)), int(max(0, b[1] - pad))
                x2, y2 = int(min(fw, b[2] + pad)), int(min(fh, b[3] + pad))
                crop = frame[y1:y2, x1:x2]
                if crop.size:
                    crop = cv2.resize(crop, (128, 128))
                    x = PREP(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))[None].to(device)
                    with torch.no_grad():
                        f = backbone(x).cpu().numpy()
                    proba = clf.predict_proba(f)[0]
                    k = int(np.argmax(proba))
                    face = (str(clf.classes_[k]) if proba[k] >= args.min_prob
                            else "HANDLING")
                    rec.update(box=[round(v, 1) for v in b],
                               face=face, p=float(proba[k]))
        frames.append(rec)
        i += 1
        if i % 500 == 0:
            print(f"  {fmt(t)}  n={i}", flush=True)
    cap.release()

    half = max(1, int(args.smooth_s * args.fps / 2))
    faces = smooth([f["face"] for f in frames], half)

    # a readable face fixes the side; cheese hides it but cannot turn the patty,
    # and handling means the patty is off the surface, so both carry forward
    side, timers, off, events = None, {"A": 0.0, "B": 0.0}, 0.0, []
    pending, pending_since = None, None
    dt = 1.0 / args.fps
    t_placed = t_removed = None
    for rec, face in zip(frames, faces):
        t = rec["t"]
        if rec["box"] is None:
            if t_placed is not None:
                off += dt
            continue
        if t_placed is None:
            t_placed = t
        t_removed = t

        if face in SIDE_OF_FACE:
            cand = SIDE_OF_FACE[face]
            if side is None:
                side = cand
                events.append(("PLACED", t, f"side {side} down"))
            elif cand != side:
                if pending != cand:
                    pending, pending_since = cand, t
                elif t - pending_since >= args.commit_s:
                    events.append(("FLIP", pending_since, f"side {side} -> {cand}"))
                    side, pending = cand, None
            else:
                pending = None
            timers[side] += dt
        elif face == "TOPPING":
            if not events or events[-1][0] != "TOPPING":
                events.append(("TOPPING", t, ""))
            if side:
                timers[side] += dt
        else:                      # HANDLING: airborne or on its rim
            off += dt
            pending = None

    if t_removed is not None:
        events.append(("REMOVED", t_removed, ""))

    print("\n=== TIMELINE ===")
    for kind, t, note in events:
        print(f"  {fmt(t):>8s}  {kind:8s} {note}")
    print("\n=== TOTALS ===")
    print(f"  side A (first down): {timers['A']:7.1f} s  ({fmt(timers['A'])})")
    print(f"  side B            : {timers['B']:7.1f} s  ({fmt(timers['B'])})")
    print(f"  off surface       : {off:7.1f} s")
    print(f"  on griddle total  : {timers['A'] + timers['B'] + off:7.1f} s")

    (out_dir / "result.json").write_text(json.dumps({
        "side_A_s": round(timers["A"], 1), "side_B_s": round(timers["B"], 1),
        "off_surface_s": round(off, 1),
        "total_s": round(timers["A"] + timers["B"] + off, 1),
        "flips": sum(1 for e in events if e[0] == "FLIP"),
        "events": [{"type": k, "t": round(t, 2), "note": n} for k, t, n in events],
    }, indent=2, ensure_ascii=False))

    if args.render:
        render(clip, frames, faces, events, args, out_dir, timers)
    print(f"\nwrote {out_dir}/result.json")


def render(clip, frames, faces, events, args, out_dir, _t):
    cap = cv2.VideoCapture(clip)
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    vw = cv2.VideoWriter(str(out_dir / "annotated_raw.mp4"),
                         cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (W, H))
    s = H / 1080.0
    side, timers, off = None, {"A": 0.0, "B": 0.0}, 0.0
    pending, pending_since = None, None
    dt = 1.0 / args.fps
    ev_i = 0
    flash = None
    for rec, face in zip(frames, faces):
        ok, frame = cap.read()
        if not ok:
            break
        t = rec["t"]
        if rec["box"] is not None:
            if face in SIDE_OF_FACE:
                cand = SIDE_OF_FACE[face]
                if side is None:
                    side = cand
                elif cand != side:
                    if pending != cand:
                        pending, pending_since = cand, t
                    elif t - pending_since >= args.commit_s:
                        side, pending = cand, None
                        flash = (t, "FLIP")
                else:
                    pending = None
                timers[side] += dt
            elif face == "TOPPING":
                if side:
                    timers[side] += dt
            else:
                off += dt

            x1, y1, x2, y2 = [int(v) for v in rec["box"]]
            col = COL.get(side, (200, 200, 200))
            cv2.rectangle(frame, (x1, y1), (x2, y2), col, 3)
            lines = [
                f"#1  face:{(face or '?').replace('FACE_','')} p={rec['p']:.2f}",
                f"{'>' if side == 'A' else ' '}side A {fmt(timers['A'])}",
                f"{'>' if side == 'B' else ' '}side B {fmt(timers['B'])}",
            ]
            ty = y1 - int(12 * s) - int(38 * s) * len(lines)
            if ty < 0:
                ty = y2 + int(8 * s)
            for k, line in enumerate(lines):
                org = (x1, ty + int(38 * s) * (k + 1))
                cv2.putText(frame, line, org, cv2.FONT_HERSHEY_SIMPLEX, 0.95 * s,
                            (0, 0, 0), max(4, int(5 * s)), cv2.LINE_AA)
                cv2.putText(frame, line, org, cv2.FONT_HERSHEY_SIMPLEX, 0.95 * s,
                            col, max(2, int(2 * s)), cv2.LINE_AA)
        if flash and 0 <= t - flash[0] < 1.5:
            cv2.putText(frame, flash[1], (int(40 * s), int(140 * s)),
                        cv2.FONT_HERSHEY_SIMPLEX, 2.0 * s, (0, 0, 0), 8, cv2.LINE_AA)
            cv2.putText(frame, flash[1], (int(40 * s), int(140 * s)),
                        cv2.FONT_HERSHEY_SIMPLEX, 2.0 * s, (0, 240, 255), 3, cv2.LINE_AA)
        cv2.putText(frame, f"t={fmt(t)}  off:{off:.1f}s", (12, int(42 * s)),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.05 * s, (0, 0, 0), 5, cv2.LINE_AA)
        cv2.putText(frame, f"t={fmt(t)}  off:{off:.1f}s", (12, int(42 * s)),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.05 * s, (255, 255, 255), 2, cv2.LINE_AA)
        vw.write(frame)
    vw.release()
    cap.release()


if __name__ == "__main__":
    main()
