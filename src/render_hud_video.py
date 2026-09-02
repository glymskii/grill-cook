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
sys.path.insert(0, str(ROOT / "src"))
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
    ap.add_argument("--start", type=float, default=0.0,
                    help="first video second to draw; the engine still replays "
                         "everything before it, so the grill starts loaded")
    ap.add_argument("--fps", type=float, default=30.0)
    ap.add_argument("--width", type=int, default=0, help="0 = source width")
    ap.add_argument("--targets", default="42,35,12,12")
    ap.add_argument("--roi", default="0.215,0.70;0.415,0.085;0.885,0.165;0.695,0.92")
    ap.add_argument("--cook", default="US Partner")
    ap.add_argument("--engine", default="slots", choices=["slots", "production"])
    args = ap.parse_args()

    roi = [[float(v) for v in p.split(",")] for p in args.roi.split(";")]
    rows = json.loads(Path(args.dets).read_text())
    if ":" in args.targets:
        # per-SKU standards: "pale:320,195,45,65;dark:135,175,20,25" - a patty picks
        # its own by the colour it rests at; the first pair is the station default
        skus = {}
        for part in args.targets.split(";"):
            name, vals = part.split(":")
            a, b, e, l = (float(v) for v in vals.split(","))
            skus[name] = {"A": a, "B": b, "tol_early": e, "tol_late": l}
        first = next(iter(skus.values()))
        targets = dict(first, skus=dict(skus, a_below=132))
        ta, tb, te, tl = first["A"], first["B"], first["tol_early"], first["tol_late"]
    else:
        ta, tb, te, tl = (float(v) for v in args.targets.split(","))
        targets = {"A": ta, "B": tb, "tol_early": te, "tol_late": tl}
    if args.engine == "slots":
        import slots
        from face_reader import FaceReader
        slots.EVENTS = Path(tempfile.mkdtemp()) / "e.jsonl"
        eng = slots.SlotEngine(targets, face_model=FaceReader())
    else:
        timers.EVENTS = Path(tempfile.mkdtemp()) / "e.jsonl"
        eng = timers.TimerEngine(targets)
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
    panel_w = int(fw * 0.26)
    ow = args.width or (fw + panel_w)
    oh = int(round(fh * ow / (fw + panel_w)))
    vw = cv2.VideoWriter(args.out, cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (ow, oh))
    poly_px = np.array([[int(x * fw), int(y * fh)] for x, y in roi], np.int32)

    n_frames = int(args.seconds * args.fps)
    ri = 0
    snap = {"patties": [], "session": {"streak": 0, "optimal": 0, "flips": 0}, "done": 0}
    # Warm the engine on everything before the window, silently: a clip that
    # starts mid-shift must open with the grill already loaded, not with the
    # timers being born on screen. The slot engine needs the pictures for that,
    # not just the boxes — its anchors, its episodes and the face all live in
    # pixels — so the warm-up decodes too.
    src_i = 0
    while ri < len(rows) and rows[ri]["t"] <= args.start:
        warm = None
        if args.engine == "slots":
            want = int(rows[ri]["t"] * src_fps)
            while src_i < want:
                cap.grab()
                src_i += 1
            ok, warm = cap.read()
            src_i += 1
            if not ok:
                warm = None
        eng.update([tuple(d) for d in rows[ri]["dets"]], rows[ri]["t"],
                   *([warm] if args.engine == "slots" else []))
        ri += 1
    toasts.clear()
    # Sequential decode: one seek, then grab-and-drop. Seeking per frame in a
    # 1080p60 source costs more than decoding it.
    src_i = int(args.start * src_fps)
    cap.set(cv2.CAP_PROP_POS_FRAMES, src_i)
    for i in range(n_frames):
        t = args.start + i / args.fps
        want = int(t * src_fps)
        while src_i < want:
            cap.grab()
            src_i += 1
        ok, frame = cap.read()
        src_i += 1
        if not ok:
            break
        while ri < len(rows) and rows[ri]["t"] <= t:
            eng.update([tuple(d) for d in rows[ri]["dets"]], rows[ri]["t"],
                       *([frame] if args.engine == "slots" else []))
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
            tl_p = p.get("tol_late", tl)
            ring, over = BLUE, False
            if p.get("done"):
                # both sides cooked to the standard: green, and the overrun
                # counts up so a forgotten patty still shouts
                over_s = int(p.get("done_for", 0))
                cv2.circle(frame, (x, y), r, GREEN, 4, cv2.LINE_AA)
                put(frame, "DONE", (x, y - r // 8), r / 150, GREEN, 2)
                put(frame, f"+{over_s // 60}:{over_s % 60:02d}", (x, y + int(r * 0.32)), r / 110,
                    WHITE if over_s < 60 else AMBER, 2)
                continue
            if p.get("provisional"):
                # a slot still proving itself: a thin quiet ring, no countdown
                cv2.circle(frame, (x, y), r, (150, 140, 130), 2, cv2.LINE_AA)
                continue
            if remain <= -tl_p:
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
                put(frame, f"side {p['side']} - #{p.get('no', p['pid'])}",
                    (x, y + int(r * 0.56)), r / 230, DIM, 2)
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

        # --- log panel: every patty on the plate, oldest first -----------------
        # It lives beside the picture rather than on top of it: the griddle runs
        # to the right edge of this frame and a panel over it would hide the very
        # patties it is describing.
        live = sorted(snap["patties"], key=lambda q: -q.get("total", 0))
        pw = int(fw * 0.26)
        canvas = np.zeros((fh, fw + pw, 3), np.uint8)
        canvas[:, :fw] = frame
        canvas[:, fw:] = (22, 16, 11)
        frame = canvas
        px0 = fw
        yy = 74 + 44
        put(frame, "ON THE PLATE", (px0 + 22, yy), 0.66, DIM, 2, centre=False)
        yy += 16
        cv2.line(frame, (px0 + 22, yy), (px0 + pw - 22, yy), (70, 58, 46), 2)
        yy += 40
        col = [px0 + 22, px0 + int(pw * 0.30), px0 + int(pw * 0.50),
               px0 + int(pw * 0.70), px0 + int(pw * 0.87)]
        for label, cxx in zip(("#", "total", "side A", "side B", "state"), col):
            put(frame, label, (cxx, yy), 0.52, DIM, 1, centre=False)
        yy += 12
        mmss = lambda v: f"{int(v) // 60}:{int(v) % 60:02d}"
        # A log that quietly drops rows is worse than no log: the pitch shrinks
        # to fit a busy plate, and anything still left over is counted out loud.
        pitch = 40 if len(live) <= 20 else max(26, (fh - 60 - yy) // max(len(live), 1))
        shown = 0
        for p in live:
            yy += pitch
            if yy > fh - 34:
                break
            shown += 1
            hot = RED if p["deadline"] - t <= -p.get("tol_late", tl) and not p.get("done") else None
            tone = GREEN if p.get("done") else (hot or WHITE)
            put(frame, f"{p.get('no', p['pid'])}", (col[0], yy), 0.7, tone, 2, centre=False)
            put(frame, mmss(p.get("total", 0)), (col[1], yy), 0.66, WHITE, 2, centre=False)
            for key, cxx in (("side_a", col[2]), ("side_b", col[3])):
                v = p.get(key, 0)
                live_side = (key == "side_a") == (p["side"] == "A")
                put(frame, mmss(v) if v else "-", (cxx, yy), 0.62,
                    AMBER if live_side and not p.get("done") else DIM, 2, centre=False)
            state = "DONE" if p.get("done") else p["side"]
            put(frame, state, (col[4], yy), 0.62, GREEN if p.get("done") else DIM, 2, centre=False)
        if shown < len(live):
            put(frame, f"+{len(live) - shown} more on the plate",
                (col[0], min(yy + pitch, fh - 12)), 0.55, DIM, 1, centre=False)

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

        vw.write(frame if frame.shape[1] == ow else
                 cv2.resize(frame, (ow, oh), interpolation=cv2.INTER_AREA))
        if i % 300 == 0:
            print(f"  {t - args.start:.0f}с / {args.seconds:.0f}", flush=True)
    vw.release()
    cap.release()
    print("готово:", args.out)


if __name__ == "__main__":
    main()
