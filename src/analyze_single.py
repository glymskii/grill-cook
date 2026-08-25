"""Manipulation-anchored analysis of a single-patty cook.

Usage:
  python src/analyze_single.py video.mp4 --fps 25 --out out/run [--start S --end E]
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLOWorld

from config import Config
from detector import mean_lab
from manipulations import FrameObs, MpConfig, find_manipulations, side_timeline


def fmt(sec: float) -> str:
    return f"{int(sec // 60)}:{sec % 60:04.1f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--out", default="out/single")
    ap.add_argument("--fps", type=float, default=25.0)
    ap.add_argument("--start", type=float, default=0.0)
    ap.add_argument("--end", type=float, default=None)
    ap.add_argument("--conf", type=float, default=0.05)
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    src = args.video
    if args.start or args.end or args.fps:
        import subprocess
        clip = str(out_dir / "clip.mp4")
        cmd = ["ffmpeg", "-y", "-v", "error", "-i", args.video, "-ss", str(args.start)]
        if args.end is not None:
            cmd += ["-to", str(args.end)]
        cmd += ["-r", str(args.fps), "-c:v", "libx264", "-preset", "fast",
                "-crf", "18", "-an", clip]
        subprocess.run(cmd, check=True)
        src = clip

    cfg = Config()
    model = YOLOWorld(cfg.model_name)
    model.set_classes(list(cfg.classes))

    obs: list[FrameObs] = []
    cap = cv2.VideoCapture(src)
    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        t = args.start + idx / args.fps
        r = model.predict(frame, conf=args.conf, imgsz=cfg.imgsz,
                          device=cfg.device, agnostic_nms=True, verbose=False)[0]
        best = None
        if r.boxes is not None and len(r.boxes):
            fh, fw = frame.shape[:2]
            cands = []
            for b, c in zip(r.boxes.xyxy.tolist(), r.boxes.conf.tolist()):
                w, h = b[2] - b[0], b[3] - b[1]
                if w <= 4 or h <= 4:
                    continue
                # keep rim-on poses: only the size gate applies here, the
                # roundness gate would drop exactly the frames we need
                if not (cfg.min_size_frac <= max(w, h) / max(fw, fh) <= cfg.max_size_frac):
                    continue
                cands.append((c, b, w, h))
            if cands:
                c, b, w, h = max(cands, key=lambda x: x[0])
                best = FrameObs(
                    t=t, present=True,
                    center=((b[0] + b[2]) / 2, (b[1] + b[3]) / 2),
                    diameter=max(w, h), aspect=max(w / h, h / w),
                    lab=mean_lab(frame, tuple(b), cfg.feat_inner_frac),
                )
        obs.append(best or FrameObs(t=t, present=False))
        idx += 1
        if idx % 500 == 0:
            print(f"  {fmt(t)}  frames={idx}", flush=True)
    cap.release()

    present = [o for o in obs if o.present]
    if not present:
        print("no patty detected")
        return
    t_placed = present[0].t
    t_removed = present[-1].t

    mcfg = MpConfig()
    manips = find_manipulations(obs, mcfg)
    timers, airborne, events = side_timeline(obs, manips, t_placed, t_removed)

    print("\n=== MANIPULATIONS ===")
    for m in manips:
        print(f"  {fmt(m.t_start)} - {fmt(m.t_end)}  {m.kind:8s} dE={m.delta_e}")
    print("\n=== TIMELINE ===")
    for kind, t, note in events:
        print(f"  {fmt(t):>8s}  {kind:8s} {note}")
    print("\n=== TOTALS ===")
    print(f"  side 1 (down first): {timers[1]:7.1f} s  ({fmt(timers[1])})")
    print(f"  side 2            : {timers[2]:7.1f} s  ({fmt(timers[2])})")
    print(f"  off surface       : {airborne:7.1f} s")
    print(f"  on griddle total  : {timers[1] + timers[2] + airborne:7.1f} s")

    (out_dir / "result.json").write_text(json.dumps({
        "placed_t": round(t_placed, 2),
        "removed_t": round(t_removed, 2),
        "side1_s": round(timers[1], 1),
        "side2_s": round(timers[2], 1),
        "off_surface_s": round(airborne, 1),
        "total_s": round(timers[1] + timers[2] + airborne, 1),
        "manipulations": [
            {"t_start": round(m.t_start, 2), "t_end": round(m.t_end, 2),
             "kind": m.kind, "delta_e": m.delta_e} for m in manips],
        "events": [{"type": k, "t": round(t, 2), "note": n} for k, t, n in events],
    }, indent=2, ensure_ascii=False))
    print(f"\nwrote {out_dir}/result.json")


if __name__ == "__main__":
    main()
