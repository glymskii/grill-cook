"""Render a clean product demo: the HUD exactly as the cook sees it, to MP4.

The engine replays the pre-dumped detections at their own rate, while frames are
drawn at video rate — countdowns stay smooth because the HUD has always drawn
them from absolute deadlines rather than from engine ticks.

Usage: python src/render_hud_video.py --video data/IMG_6637.mov \
         --dets data/mot_dets_smash.json --out /tmp/demo.mp4 --seconds 100
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
sys.path.insert(0, str(ROOT / "app"))
import timers                                          # noqa: E402
from pipeline import in_poly                           # noqa: E402

BLUE, AMBER, RED = (201, 116, 60), (35, 166, 245), (77, 72, 229)   # BGR
GREEN, WHITE, DIM = (90, 220, 80), (245, 238, 232), (179, 161, 143)


def put(img, text, org, scale, colour, thick=2, centre=True):
    (w, h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_DUPLEX, scale, thick)
    x = org[0] - w // 2 if centre else org[0]
    cv2.putText(img, text, (x, org[1]), cv2.FONT_HERSHEY_DUPLEX, scale, colour,
                thick, cv2.LINE_AA)
    return w


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--dets", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seconds", type=float, default=100.0)
    ap.add_argument("--fps", type=float, default=30.0)
    ap.add_argument("--targets", default="42,35,12,12")
    ap.add_argument("--roi", default="0.215,0.70;0.415,0.085;0.885,0.165;0.695,0.92")
    ap.add_argument("--cook", default="US Partner")
    args = ap.parse_args()

    ta, tb, te, tl = (float(v) for v in args.targets.split(","))
    roi = [[float(v) for v in p.split(",")] for p in args.roi.split(";")]
    rows = json.loads(Path(args.dets).read_text())
    timers.EVENTS = Path(tempfile.mkdtemp()) / "e.jsonl"
    eng = timers.TimerEngine({"A": ta, "B": tb, "tol_early": te, "tol_late": tl})
    toasts = []          # (until_ts, text, colour)
    eng.on_event = lambda ev: toasts.append((
        ev["ts_video"] + 2.6,
        {"optimal": "PERFECT FLIP - RIGHT ON TIME",
         "early": f"TOO EARLY! +{ev.get('bonus', 0):.0f}s ADDED TO SIDE {ev['side']}",
         "late": "TOO LATE - WATCH THE COUNTDOWN"}.get(ev.get("grade"), ""),
        {"optimal": (95, 158, 29), "early": (27, 126, 194),
         "late": (64, 59, 196)}.get(ev.get("grade"), DIM))
    ) if ev["type"] == "flip" else None

    cap = cv2.VideoCapture(args.video)
    src_fps = cap.get(cv2.CAP_PROP_FPS)
    fw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    fh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    vw = cv2.VideoWriter(args.out, cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (fw, fh))
    poly_px = np.array([[int(x * fw), int(y * fh)] for x, y in roi], np.int32)

    n_frames = int(args.seconds * args.fps)
    ri = 0
    snap = {"patties": [], "session": {"streak": 0, "optimal": 0, "flips": 0}, "done": 0}
    for i in range(n_frames):
        t = i / args.fps
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(t * src_fps))
        ok, frame = cap.read()
        if not ok:
            break
        while ri < len(rows) and rows[ri]["t"] <= t:
            for d in rows[ri]["dets"]:
                pass
            eng._video_now = rows[ri]["t"]
            # stamp video time on events for toast timing
            orig_emit = eng._emit_raw
            eng.update([tuple(d) for d in rows[ri]["dets"]], rows[ri]["t"])
            ri += 1
        snap = eng.snapshot(t)

        # --- background: dim everything outside the working zone --------------
        overlay = frame.copy()
        cv2.fillPoly(overlay, [poly_px], (255, 255, 255))
        mask = (overlay == 255).all(axis=2)
        dark = (frame * 0.42).astype(np.uint8)
        frame = np.where(mask[:, :, None], frame, dark)
        cv2.polylines(frame, [poly_px], True, GREEN, 3, cv2.LINE_AA)

        any_over = False
        for p in [q for q in snap["patties"] if q.get("missing_for", 0) <= 0.4]:
            x, y = int(p["x"] * fw), int(p["y"] * fh)
            r = max(int(p["r"] * fw * 0.95), 46)
            remain = p["deadline"] - t
            ring, over = BLUE, False
            if remain <= -tl:
                ring, over, any_over = RED, True, True
            elif remain <= 10:
                ring = AMBER
            # dark disc + progress arc
            disc = frame.copy()
            cv2.circle(disc, (x, y), r, (34, 27, 20), -1)
            frame = cv2.addWeighted(disc, 0.55, frame, 0.45, 0)
            cv2.circle(frame, (x, y), r, (64, 52, 42), 3, cv2.LINE_AA)
            frac = min(1.0, max(0.0, p["elapsed"] / max(p["target"], 1)))
            cv2.ellipse(frame, (x, y), (r, r), -90, 0, 360 * frac, ring, 5, cv2.LINE_AA)
            if 0 < remain <= 3.5:
                put(frame, str(int(math.ceil(remain))), (x, y + r // 3), r / 42, AMBER, 4)
            elif remain <= 0 and not over:
                put(frame, "FLIP", (x, y + r // 4), r / 62, RED, 4)
            else:
                mm_, ss = divmod(int(abs(remain)), 60)
                put(frame, f"{'+' if remain < 0 else ''}{mm_}:{ss:02d}",
                    (x, y + r // 6), r / 78, WHITE, 3)
                put(frame, f"side {p['side']} - #{p['pid']}", (x, y + int(r * 0.56)),
                    r / 230, DIM, 2)
            if p.get("bonus", 0) > 0 and r > 70:
                put(frame, f"carry +{p['bonus']:.0f}s", (x, y + int(r * 0.8)),
                    r / 210, AMBER, 2)

        if any_over:                       # red vignette, as in the HUD
            v = frame.copy()
            cv2.rectangle(v, (0, 0), (fw, fh), (60, 55, 190), 46)
            frame = cv2.addWeighted(v, 0.5, frame, 0.5, 0)

        # --- top bar ---------------------------------------------------------
        bar = frame.copy()
        cv2.rectangle(bar, (0, 0), (fw, 74), (22, 16, 11), -1)
        frame = cv2.addWeighted(bar, 0.82, frame, 0.18, 0)
        s = snap["session"]
        pct = f"{100 * s['optimal'] // s['flips']}%" if s["flips"] else "-"
        chips = [("live", GREEN), (args.cook, WHITE),
                 (f"on grill {len(snap['patties'])}", WHITE),
                 (f"done {snap['done']}", WHITE),
                 (f"streak {s['streak']}", AMBER), (f"on-target {pct}", WHITE)]
        cx = 34
        for text, col in chips:
            cv2.circle(frame, (cx, 37), 7, col, -1, cv2.LINE_AA) if text == "live" else None
            off = 20 if text == "live" else 0
            w = put(frame, text, (cx + off, 46), 0.78, col, 2, centre=False)
            cx += w + off + 46

        # --- toast -----------------------------------------------------------
        live_toasts = [x for x in toasts if x[0] > t and x[1]]
        if live_toasts:
            _, text, col = live_toasts[-1]
            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_DUPLEX, 1.05, 2)
            x0, y0 = (fw - tw) // 2 - 34, 116
            box = frame.copy()
            cv2.rectangle(box, (x0, y0), (x0 + tw + 68, y0 + th + 44), col, -1)
            frame = cv2.addWeighted(box, 0.92, frame, 0.08, 0)
            put(frame, text, (fw // 2, y0 + th + 14), 1.05, (255, 255, 255), 2)

        vw.write(frame)
        if i % 300 == 0:
            print(f"  {t:.0f}с / {args.seconds:.0f}", flush=True)
    vw.release()
    cap.release()
    print("готово:", args.out)


if __name__ == "__main__":
    main()
