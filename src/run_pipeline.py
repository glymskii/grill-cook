"""Run the patty side-timer pipeline on a video.

Usage:
  python run_pipeline.py video.mp4 --out out/run1 [--roi x1,y1,x2,y2]
      [--start 10 --end 300] [--conf 0.08] [--no-render] [--stride 1]
"""
import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np

from config import Config
from detector import PattyDetector
from patties import PattyTracker
from render import draw_frame, fmt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--out", default="out/run")
    ap.add_argument("--roi", default=None, help="x1,y1,x2,y2")
    ap.add_argument("--start", type=float, default=0, help="start time, s")
    ap.add_argument("--end", type=float, default=None, help="end time, s")
    ap.add_argument("--conf", type=float, default=None)
    ap.add_argument("--model", default=None)
    ap.add_argument("--classes", default=None, help="comma-separated open-vocab prompts")
    ap.add_argument("--no-render", action="store_true")
    ap.add_argument("--fps", type=float, default=None,
                    help="decimate to this frame rate (long videos); timings stay in source seconds")
    args = ap.parse_args()

    cfg = Config()
    if args.roi:
        cfg.roi = tuple(float(v) for v in args.roi.split(","))
    if args.conf is not None:
        cfg.conf = args.conf
    if args.model:
        cfg.model_name = args.model
    if args.classes:
        cfg.classes = tuple(s.strip() for s in args.classes.split(","))

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(args.video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    # optionally cut/decimate the segment to a temp file so ultralytics sees only it
    video_path = args.video
    if args.start > 0 or args.end is not None or args.fps:
        import subprocess
        clip = str(out_dir / "clip.mp4")
        cmd = ["ffmpeg", "-y", "-i", args.video, "-ss", str(args.start)]
        if args.end is not None:
            cmd += ["-to", str(args.end)]
        if args.fps:
            cmd += ["-r", str(args.fps)]
            fps = args.fps
        cmd += ["-c:v", "libx264", "-preset", "fast", "-crf", "18", "-an", clip]
        subprocess.run(cmd, check=True, capture_output=True)
        video_path = clip

    det = PattyDetector(cfg)
    tracker = PattyTracker(cfg, fps, (W ** 2 + H ** 2) ** 0.5)

    writer = None
    if not args.no_render:
        writer = cv2.VideoWriter(str(out_dir / "annotated_raw.mp4"),
                                 cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))

    # scene cuts: a hard cut / reframe changes the WHOLE frame at once, while a
    # hand or spatula crosses only one edge. Split the border into 4 side bands
    # and call it a cut only when >=3 sides change simultaneously.
    sides = {
        "top": (slice(0, 7), slice(None)),
        "bottom": (slice(47, 54), slice(None)),
        "left": (slice(None), slice(0, 12)),
        "right": (slice(None), slice(84, 96)),
    }

    last_idx = 0
    prev_small = None
    cuts = []
    for frame_idx, frame, dets in det.stream(video_path):
        last_idx = frame_idx
        small = cv2.cvtColor(cv2.resize(frame, (96, 54)),
                             cv2.COLOR_BGR2GRAY).astype(np.int16)
        scene_cut = False
        if prev_small is not None:
            d = np.abs(small - prev_small)
            hot = sum(1 for sl in sides.values()
                      if float(d[sl].mean()) > cfg.cut_diff_thr)
            if hot >= 3:
                scene_cut = True
                cuts.append(round(frame_idx / fps, 2))
        prev_small = small
        tracker.step(frame_idx, frame, dets, scene_cut)
        if writer is not None:
            writer.write(draw_frame(frame, tracker.patties, frame_idx, fps, cfg))
        if frame_idx % 500 == 0:
            print(f"frame {frame_idx} t={fmt(frame_idx / fps)} "
                  f"patties={len(tracker.patties)}", flush=True)
    if writer is not None:
        writer.release()

    patties = tracker.finish(last_idx)

    events = []
    rows = []
    for p in sorted(patties, key=lambda q: q.first_frame):
        for ev in p.events:
            events.append({"patty": p.id, "type": ev.type,
                           "t": round(ev.t, 2), "frame": ev.frame, **ev.detail})
        done = p.state.startswith("REMOVED")
        rows.append({
            "patty": p.id,
            "placed_t": round(p.first_frame / fps, 1),
            "removed_t": round(p.last_seen_frame / fps, 1) if done else "",
            "total_s": round(p.timers[1] + p.timers[2], 1),
            "side1_s": round(p.timers[1], 1),
            "side2_s": round(p.timers[2], 1),
            "flips": p.n_flips,
            "toppings": p.n_toppings,
            "status": p.state,
        })

    (out_dir / "events.json").write_text(json.dumps(
        {"events": events, "scene_cuts": cuts}, indent=2))
    with open(out_dir / "summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys() if rows else ["patty"])
        w.writeheader()
        w.writerows(rows)

    print("\n=== SUMMARY ===")
    for r in rows:
        print(r)
    print(f"\nwrote {out_dir}/events.json, summary.csv"
          + ("" if args.no_render else ", annotated_raw.mp4"))


if __name__ == "__main__":
    main()
