"""Live per-side timer state machine for the pilot HUD.

Deliberately simpler than the offline pipeline: no face classifier, no track
stitching across scene cuts. A flip is a disappearance of at least flip_gap_min
seconds that ends near where it started; the gap itself belongs to neither side
(the patty is in the air or under the spatula). Sub-gap dropouts are detector
jitter and accrue to the current side untouched.

All timestamps are epoch seconds so the HUD can dead-reckon deadlines against
its own clock.
"""
import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path

EVENTS = Path(__file__).parent / "state" / "events.jsonl"


@dataclass
class LivePatty:
    pid: int
    cx: float
    cy: float
    r: float
    placed_ts: float
    side: str = "A"
    side_started: float = 0.0
    side_time: dict = field(default_factory=lambda: {"A": 0.0, "B": 0.0})
    flips: int = 0
    last_seen: float = 0.0
    missing_since: float | None = None
    last_flip_ts: float = 0.0
    bonus: float = 0.0            # early-flip shortfall carried onto the new side
    flip_feedback: str | None = None       # optimal | early | late, for the HUD

    def elapsed(self, now: float) -> float:
        if self.missing_since is not None:
            return self.side_time[self.side] + (self.missing_since - self.side_started)
        return self.side_time[self.side] + (now - self.side_started)


class TimerEngine:
    def __init__(self, targets: dict, flip_gap_min=1.2, removed_after=9.0,
                 flip_cooldown=30.0, min_side_before_flip=25.0, assoc_frac=1.6,
                 birth_conf=0.30):
        self.targets = targets                  # {"A": s, "B": s, "tol_early": s, "tol_late": s}
        self.flip_gap_min = flip_gap_min
        self.removed_after = removed_after
        self.flip_cooldown = flip_cooldown
        self.min_side = min_side_before_flip
        self.assoc_frac = assoc_frac
        self.birth_conf = birth_conf   # hysteresis: new tracks need this much,
                                       # existing ones survive on far less
        self.alive: dict[int, LivePatty] = {}
        self.done: list[dict] = []
        self.next_pid = 1
        self.session = {"flips": 0, "optimal": 0, "early": 0, "late": 0, "streak": 0}
        self.on_event = None                    # optional hook for the edge worker
        self.freeze_until = 0.0                 # settle window after a scene cut
        self.scene_cut_ts = 0.0
        self.match_stat = (0, 0)                # (matched alive, total alive) per frame

    # ---- events -------------------------------------------------------------
    def _emit_raw(self, ev: dict):
        EVENTS.parent.mkdir(parents=True, exist_ok=True)
        with EVENTS.open("a") as f:
            f.write(json.dumps(ev, ensure_ascii=False) + "\n")
        if self.on_event:
            self.on_event(ev)

    def _emit(self, kind: str, p: LivePatty, extra: dict | None = None):
        ev = {"ts": round(time.time(), 2), "type": kind, "pid": p.pid,
              "side": p.side, "flips": p.flips,
              "side_a": round(p.side_time["A"], 1), "side_b": round(p.side_time["B"], 1),
              "ta": self.targets["A"], "tb": self.targets["B"],
              "te": self.targets["tol_early"], "tl": self.targets["tol_late"]}
        if extra:
            ev.update(extra)
        self._emit_raw(ev)

    def scene_reset(self, now: float):
        """The view changed globally: positional identity is void. Archive the
        current patties as invalidated (their timers must not enter analytics
        as completed cooks) and hold off new tracks while the scene settles."""
        for p in self.alive.values():
            if p.missing_since is None:
                p.side_time[p.side] += now - p.side_started
            else:
                p.side_time[p.side] += p.missing_since - p.side_started
            self._emit("invalidated", p)
        n = len(self.alive)
        self.alive.clear()
        self.freeze_until = now + 1.2
        self.scene_cut_ts = now
        self._emit_raw({"ts": round(time.time(), 2), "type": "scene_cut",
                        "n_dropped": n})

    def _grade_flip(self, side: str, elapsed: float) -> str:
        t = self.targets
        if elapsed < t[side] - t["tol_early"]:
            return "early"
        if elapsed > t[side] + t["tol_late"]:
            return "late"
        return "optimal"

    # ---- per-frame update ---------------------------------------------------
    def update(self, dets: list[tuple[float, float, float]], now: float):
        """dets: (cx, cy, r[, conf]) in normalized coords, r = radius/frame_w."""
        if now < self.freeze_until:
            return
        dets = [(d + (1.0,))[:4] for d in dets]
        used = set()
        matched = 0
        # associate greedily: nearest detection within assoc_frac * diameter
        for p in sorted(self.alive.values(), key=lambda q: q.placed_ts):
            best, best_d = None, 1e9
            for i, (cx, cy, r, _cf) in enumerate(dets):
                if i in used:
                    continue
                d = math.hypot(cx - p.cx, cy - p.cy)
                if d < best_d:
                    best, best_d = i, d
            if best is not None and best_d <= self.assoc_frac * 2 * p.r:
                used.add(best)
                matched += 1
                cx, cy, r, _cf = dets[best]
                p.cx = 0.7 * p.cx + 0.3 * cx
                p.cy = 0.7 * p.cy + 0.3 * cy
                p.r = 0.8 * p.r + 0.2 * r
                if p.missing_since is not None:
                    gap = now - p.missing_since
                    long_enough = gap >= self.flip_gap_min
                    cooled = now - max(p.last_flip_ts, p.placed_ts) >= self.flip_cooldown
                    seasoned = (p.missing_since - p.side_started) >= self.min_side
                    if long_enough and cooled and seasoned:
                        elapsed = p.side_time[p.side] + (p.missing_since - p.side_started)
                        grade = self._grade_flip(p.side, elapsed)
                        # an early flip is fixable: the heat the old side missed
                        # is owed by the new side, so its target grows by the gap
                        p.bonus = round(self.targets[p.side] - elapsed, 1) if grade == "early" else 0.0
                        p.side_time[p.side] = elapsed
                        p.side = "B" if p.side == "A" else "A"
                        p.side_started = now
                        p.flips += 1
                        p.last_flip_ts = now
                        p.flip_feedback = grade
                        self.session["flips"] += 1
                        self.session[grade] += 1
                        self.session["streak"] = self.session["streak"] + 1 if grade == "optimal" else 0
                        self._emit("flip", p, {"grade": grade, "elapsed": round(elapsed, 1),
                                                "bonus": p.bonus})
                    else:
                        # short dropout: fold the gap back into the running side
                        pass
                    p.missing_since = None
                p.last_seen = now
            else:
                if p.missing_since is None:
                    p.missing_since = now

        self.match_stat = (matched, len(self.alive))

        # unmatched detections become new patties — but only confident ones;
        # a flickering low-conf blob may extend a track, never found one
        for i, (cx, cy, r, cf) in enumerate(dets):
            if i in used or cf < self.birth_conf:
                continue
            # ignore rebirth right on top of an existing patty
            if any(math.hypot(cx - q.cx, cy - q.cy) < 1.2 * q.r for q in self.alive.values()):
                continue
            p = LivePatty(self.next_pid, cx, cy, r, placed_ts=now,
                          side_started=now, last_seen=now)
            self.next_pid += 1
            self.alive[p.pid] = p
            self._emit("placed", p)

        # removals
        for pid in [pid for pid, p in self.alive.items()
                    if p.missing_since and now - p.missing_since > self.removed_after]:
            p = self.alive.pop(pid)
            p.side_time[p.side] += p.missing_since - p.side_started
            total = p.side_time["A"] + p.side_time["B"]
            self._emit("removed", p, {"total": round(total, 1)})
            self.done.append({"pid": p.pid, "side_a": round(p.side_time["A"], 1),
                              "side_b": round(p.side_time["B"], 1),
                              "flips": p.flips, "total": round(total, 1)})

    # ---- HUD snapshot -------------------------------------------------------
    def snapshot(self, now: float) -> dict:
        t = self.targets
        out = []
        for p in self.alive.values():
            target = t[p.side] + p.bonus
            elapsed = p.elapsed(now)
            fb, p.flip_feedback = p.flip_feedback, None    # one-shot to the HUD
            out.append({
                "pid": p.pid, "x": round(p.cx, 4), "y": round(p.cy, 4),
                "r": round(p.r, 4), "side": p.side, "flips": p.flips,
                "elapsed": round(elapsed, 1), "target": target,
                "bonus": p.bonus,
                "deadline": round(now - elapsed + target, 2),
                "missing": p.missing_since is not None,
                "feedback": fb,
            })
        return {"patties": out, "session": dict(self.session), "done": len(self.done),
                "scene_cut_ts": round(self.scene_cut_ts, 2)}
