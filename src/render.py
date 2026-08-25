"""Overlay: track id, active side, both side timers, event flashes."""
import cv2

from config import Config

PALETTE = [(66, 133, 244), (52, 168, 83), (251, 188, 5), (234, 67, 53),
           (171, 71, 188), (0, 172, 193), (255, 112, 67), (158, 157, 36)]


def _color(pid: int):
    c = PALETTE[(pid - 1) % len(PALETTE)]
    return (c[2], c[1], c[0])  # BGR


def fmt(sec: float) -> str:
    return f"{int(sec // 60)}:{sec % 60:04.1f}"


def draw_frame(frame, patties, frame_idx: int, fps: float, cfg: Config):
    now = frame_idx / fps
    s = frame.shape[0] / 1080.0  # scale typography with resolution
    fs = 0.95 * s
    lh = int(38 * s)
    for p in patties:
        if p.state != "ACTIVE" or frame_idx - p.last_seen_frame > 2 * fps:
            continue
        x1, y1, x2, y2 = [int(v) for v in p.last_xyxy]
        col = _color(p.id)
        visible = p.last_seen_frame == frame_idx
        cv2.rectangle(frame, (x1, y1), (x2, y2), col, 3 if visible else 1)

        a_on = p.side_down == 1
        lines = [
            f"#{p.id}  flips:{p.n_flips}",
            f"{'>' if a_on else ' '}S1 {fmt(p.timers[1])}",
            f"{'>' if not a_on else ' '}S2 {fmt(p.timers[2])}",
        ]
        ty = y1 - int(10 * s) - lh * len(lines)
        if ty < 0:
            ty = y2 + int(6 * s)
        for i, line in enumerate(lines):
            org = (x1, ty + lh * (i + 1))
            cv2.putText(frame, line, org, cv2.FONT_HERSHEY_SIMPLEX,
                        fs, (0, 0, 0), max(4, int(5 * s)), cv2.LINE_AA)
            cv2.putText(frame, line, org, cv2.FONT_HERSHEY_SIMPLEX,
                        fs, col, max(2, int(2 * s)), cv2.LINE_AA)

        # flash recent events
        for ev in p.events[-3:]:
            if 0 <= now - ev.t < 1.2 and ev.type != "PLACED":
                org = (x1, max(int(30 * s), y1 - int(12 * s)))
                cv2.putText(frame, ev.type, org, cv2.FONT_HERSHEY_SIMPLEX,
                            1.7 * s, (0, 0, 0), max(5, int(7 * s)), cv2.LINE_AA)
                cv2.putText(frame, ev.type, org, cv2.FONT_HERSHEY_SIMPLEX,
                            1.7 * s, (0, 240, 255), max(2, int(3 * s)), cv2.LINE_AA)

    if cfg.roi is not None:
        x1, y1, x2, y2 = [int(v) for v in cfg.roi]
        cv2.rectangle(frame, (x1, y1), (x2, y2), (200, 200, 200), 1)
    cv2.putText(frame, f"t={fmt(now)}", (12, int(40 * s)), cv2.FONT_HERSHEY_SIMPLEX,
                1.1 * s, (0, 0, 0), max(4, int(5 * s)), cv2.LINE_AA)
    cv2.putText(frame, f"t={fmt(now)}", (12, int(40 * s)), cv2.FONT_HERSHEY_SIMPLEX,
                1.1 * s, (255, 255, 255), max(2, int(2 * s)), cv2.LINE_AA)
    return frame
