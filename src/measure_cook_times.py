"""Measure real per-side cook times by replaying a video through the live logic.

Linear pass, no looping: a looped stream restarts the scene and shreds tracks,
which is exactly why the live run reported nonsense. Here the same detector,
size gate and TimerEngine run once over the file, and the flip intervals that
come out are what the cook actually does.

Usage: python src/measure_cook_times.py data/IMG_6635.mov --fps 10
"""
import argparse
import statistics as st
import sys
from collections import deque
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "app"))

from config import Config              # noqa: E402
from pipeline import in_poly, lab_of, size_gate  # noqa: E402
from timers import TimerEngine         # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--fps", type=float, default=10.0)
    ap.add_argument("--roi", default="")     # "x,y;x,y;..." normalized
    ap.add_argument("--model", default="runs/detect/runs/detect/patty_v6/weights/best.pt")
    # a hand or spatula sweeping over a patty hides it for a moment; only a gap
    # that is long AND late enough to be a real flip should count as one
    ap.add_argument("--flip-gap", type=float, default=1.2)
    ap.add_argument("--min-side", type=float, default=25.0)
    ap.add_argument("--cooldown", type=float, default=30.0)
    args = ap.parse_args()

    roi = [[float(v) for v in pt.split(",")] for pt in args.roi.split(";")] if args.roi else []
    cfg = Config()
    from ultralytics import YOLO
    model = YOLO(args.model)

    cap = cv2.VideoCapture(args.video)
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30
    total = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    step = max(1, round(src_fps / args.fps))

    # targets are irrelevant to measurement — the engine records actual seconds
    eng = TimerEngine({"A": 999, "B": 999, "tol_early": 999, "tol_late": 999},
                      flip_gap_min=args.flip_gap, min_side_before_flip=args.min_side,
                      flip_cooldown=args.cooldown)
    flips: list[tuple[int, float]] = []       # (flip index, seconds on that side)
    eng.on_event = lambda ev: (flips.append((ev["flips"], ev["elapsed"]))
                               if ev["type"] == "flip" else None)
    radii: deque = deque(maxlen=180)

    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % step:
            idx += 1
            continue
        t = idx / src_fps
        fh, fw = frame.shape[:2]
        r = model.predict(frame, conf=0.15, iou=cfg.iou, imgsz=960, device=cfg.device,
                          agnostic_nms=True, verbose=False)[0]
        dets = []
        if r.boxes is not None:
            for (x1, y1, x2, y2), cf in zip(r.boxes.xyxy.tolist(), r.boxes.conf.tolist()):
                w, h = x2 - x1, y2 - y1
                if w < 8 or h < 8 or max(w / h, h / w) > cfg.max_aspect:
                    continue
                if max(w, h) / max(fw, fh) > cfg.max_size_frac:
                    continue
                dets.append(((x1 + x2) / 2 / fw, (y1 + y2) / 2 / fh,
                             (w + h) / 4 / fw, cf, lab_of(frame, x1, y1, x2, y2)))
        dets, _ = size_gate(dets, radii)
        if len(roi) >= 3:
            dets = [d for d in dets if in_poly(d[0], d[1], roi)]
        eng.update(dets, t)
        idx += 1
        if idx % (step * 300) == 0:
            print(f"  {t/60:.1f} мин / {total/src_fps/60:.1f}, "
                  f"на плите {len(eng.alive)}, переворотов {len(flips)}", flush=True)
    cap.release()

    done = eng.done + [{"side_a": p.side_time["A"], "side_b": p.side_time["B"],
                        "flips": p.flips} for p in eng.alive.values()]
    q = lambda v, p: sorted(v)[int(p * (len(v) - 1))] if v else 0

    first = [s for i, s in flips if i == 1 and s > 5]
    second = [s for i, s in flips if i == 2 and s > 5]
    print(f"\nвсего переворотов: {len(flips)}, котлет с историей: {len(done)}")
    if first:
        print(f"сторона A до 1-го переворота: медиана {st.median(first):.0f}с  "
              f"(25–75%: {q(first,.25):.0f}–{q(first,.75):.0f}, n={len(first)})")
    if second:
        print(f"сторона B до 2-го переворота: медиана {st.median(second):.0f}с  "
              f"(25–75%: {q(second,.25):.0f}–{q(second,.75):.0f}, n={len(second)})")
    full = [d["side_a"] + d["side_b"] for d in done if d["side_a"] > 5 and d["side_b"] > 5]
    if full:
        print(f"полный цикл (обе стороны): медиана {st.median(full):.0f}с (n={len(full)})")
    per_patty = [d["flips"] for d in done]
    if per_patty:
        print(f"переворотов на котлету: медиана {st.median(per_patty):.1f} "
              f"(ожидаем 1–2; больше — детектор путает перекрытия с переворотом)")
    import json
    Path("/tmp/cook_times.json").write_text(json.dumps(
        {"done": done, "flips": flips}, default=float))
    on_grill = [d["side_a"] + d["side_b"] for d in done if d["side_a"] + d["side_b"] > 20]
    if on_grill:
        print(f"время на плите (все котлеты, независимо от переворотов): "
              f"медиана {st.median(on_grill):.0f}с "
              f"(25–75%: {q(on_grill,.25):.0f}–{q(on_grill,.75):.0f}, n={len(on_grill)})")
    if first and second:
        print(f"\nРЕКОМЕНДАЦИЯ: target_a={round(st.median(first))} "
              f"target_b={round(st.median(second))} "
              f"tol={round(max(8, (q(first,.75)-q(first,.25))/2))}")


if __name__ == "__main__":
    main()
