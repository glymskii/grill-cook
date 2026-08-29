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


class KF:
    """Steady-state Kalman (alpha-beta) over the normalized centre.

    Patties are static until a spatula shoves them. Velocity must learn a real
    slide yet stay dead to box jitter — innovations inside the deadband update
    only the position, and speed is clamped to what a spatula can plausibly do.
    Anything cleverer turns detector noise into motion (measured: jitter of
    0.1r random-walked velocity to 1.6 r/s and shredded the tracker).
    """

    TAU = 1.2            # seconds of velocity memory while coasting
    ALPHA = 0.5          # position: split the difference with the measurement
    BETA = 0.18          # velocity: learn slowly
    DEAD = 0.006         # ~0.15 r of box jitter is noise, not motion
    VMAX = 0.20          # a spatula slide peaks around 4-5 r/s; cap below chaos

    def __init__(self, x: float, y: float, t: float):
        self.x = [x, y, 0.0, 0.0]
        self.t = t

    def peek(self, now: float):
        """Predicted position without mutating state."""
        dt = max(0.0, min(now - self.t, 2.0))
        damp = math.exp(-dt / self.TAU)
        return (self.x[0] + self.x[2] * damp * dt,
                self.x[1] + self.x[3] * damp * dt)

    def update(self, mx: float, my: float, now: float):
        dt = max(0.05, min(now - self.t, 2.0))
        damp = math.exp(-dt / self.TAU)
        out = []
        for pos, vel, m in ((self.x[0], self.x[2], mx), (self.x[1], self.x[3], my)):
            pred = pos + vel * damp * dt
            inn = m - pred
            new_pos = pred + self.ALPHA * inn
            new_vel = vel * damp
            if abs(inn) > self.DEAD:
                step = inn - math.copysign(self.DEAD, inn)
                new_vel += self.BETA * step / dt
            new_vel = max(-self.VMAX, min(self.VMAX, new_vel))
            out += [new_pos, new_vel]
        self.x = [out[0], out[2], out[1], out[3]]
        self.t = now


def _lab_dist(a, b) -> float:
    return math.dist(a, b) if a is not None and b is not None else 0.0


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
    lab: tuple | None = None               # mean colour, for identity across gaps
    kf: KF | None = None

    def elapsed(self, now: float) -> float:
        if self.missing_since is not None:
            return self.side_time[self.side] + (self.missing_since - self.side_started)
        return self.side_time[self.side] + (now - self.side_started)


class TimerEngine:
    def __init__(self, targets: dict, flip_gap_min=1.2, removed_after=9.0,
                 flip_cooldown=30.0, min_side_before_flip=25.0, assoc_frac=1.6,
                 birth_conf=0.30, gate_base=1.45, gate_max=2.6, size_gate=1.55,
                 colour_scale=26.0, revive_window=60.0, revive_colour=22.0,
                 overlap_frac=0.55, use_kf=False, birth_suppress=1.2):
        self.targets = targets                  # {"A": s, "B": s, "tol_early": s, "tol_late": s}
        self.flip_gap_min = flip_gap_min
        self.removed_after = removed_after
        self.flip_cooldown = flip_cooldown
        self.min_side = min_side_before_flip
        self.assoc_frac = assoc_frac
        self.birth_conf = birth_conf   # hysteresis: new tracks need this much,
                                       # existing ones survive on far less
        # identity gates. Patties sit edge to edge, so a centre may be barely
        # more than one radius away from the WRONG patty: the gate has to stay
        # tight, and only widen for a track that has been missing (the cook may
        # have slid it) rather than for everyone.
        self.gate_base = gate_base
        self.gate_max = gate_max
        self.size_gate = size_gate
        self.colour_scale = colour_scale
        self.revive_window = revive_window
        self.revive_colour = revive_colour
        self.dead: list[tuple[float, LivePatty]] = []   # graveyard for re-id
        self.revived = 0
        self.overlap_frac = overlap_frac
        self.merged = 0
        # KF-assisted association tripled track life offline but costs identity
        # correctness on dense layouts (teleports 9, undercount 14/16); stays a
        # research flag until it wins on labeled MOT ground truth.
        self.use_kf = use_kf
        self.birth_suppress = birth_suppress
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

    # ---- association ---------------------------------------------------------
    @staticmethod
    def _norm_det(d):
        """(cx, cy, r[, conf][, lab]) -> a uniform 5-tuple."""
        d = tuple(d)
        return (d[0], d[1], d[2],
                d[3] if len(d) > 3 else 1.0,
                d[4] if len(d) > 4 else None)

    def _cost(self, p: "LivePatty", det, now: float):
        """Match cost for one track/detection pair, or None if impossible."""
        cx, cy, r, _cf, lab = det
        ref = max((p.r + r) / 2, 1e-6)
        ex, ey = p.kf.peek(now) if (self.use_kf and p.kf) else (p.cx, p.cy)
        dist = math.hypot(cx - ex, cy - ey) / ref
        gap = 0.0 if p.missing_since is None else now - p.missing_since
        # the gate widens only for a track that has been missing — the cook may
        # have slid that one; everyone else stays pinned to their spot
        if dist > min(self.gate_base + 0.25 * gap, self.gate_max):
            return None
        ratio = max(p.r, r) / max(min(p.r, r), 1e-6)
        if ratio > self.size_gate:
            return None
        cost = dist + 1.5 * (ratio - 1.0)
        if lab and p.lab:
            cost += min(_lab_dist(lab, p.lab) / self.colour_scale, 1.5)
        return cost

    def _assign(self, dets, now: float):
        """Globally optimal track -> detection assignment."""
        tracks = sorted(self.alive.values(), key=lambda q: q.placed_ts)
        if not tracks or not dets:
            return [(p, None) for p in tracks]
        import numpy as np
        from scipy.optimize import linear_sum_assignment
        BIG = 1e6
        cost = np.full((len(tracks), len(dets)), BIG)
        feasible = set()
        for ti, p in enumerate(tracks):
            for di, d in enumerate(dets):
                c = self._cost(p, d, now)
                if c is not None:
                    cost[ti, di] = c
                    feasible.add((ti, di))
        rows, cols = linear_sum_assignment(cost)
        chosen = {ti: di for ti, di in zip(rows, cols) if (ti, di) in feasible}
        return [(p, chosen.get(ti)) for ti, p in enumerate(tracks)]

    def _revive(self, det, now: float):
        """A patty hidden by a hand is the same patty when it reappears.

        Without this, every occlusion longer than removed_after mints a fresh id
        with reset timers — the single biggest source of track fragments.
        """
        cx, cy, r, _cf, lab = det
        best, best_cost = None, 1e9
        for i, (died, q) in enumerate(self.dead):
            gap = now - died
            if gap > self.revive_window:
                continue
            ref = max((q.r + r) / 2, 1e-6)
            dist = math.hypot(cx - q.cx, cy - q.cy) / ref
            ratio = max(q.r, r) / max(min(q.r, r), 1e-6)
            if dist > 1.5 or ratio > 1.5:
                continue
            # a long absence can hide a swap: raw meat dropped where a cooked
            # patty left looks nothing like it, so demand colour agreement once
            # the gap grows past a plausible occlusion
            if gap > 12 and lab and q.lab and _lab_dist(lab, q.lab) > self.revive_colour:
                continue
            c = dist + 1.5 * (ratio - 1.0)
            if c < best_cost:
                best, best_cost = i, c
        if best is None:
            return None
        _, q = self.dead.pop(best)
        self.done = [d for d in self.done if d["pid"] != q.pid]
        self.revived += 1
        return q

    def _on_return(self, p: "LivePatty", now: float):
        """A patty is visible again: decide whether the gap was a flip.

        Shared by the normal match path and by revival, so a track that came
        back from the graveyard is judged exactly like one that never died.
        """
        if p.missing_since is None:
            return
        gap = now - p.missing_since
        long_enough = gap >= self.flip_gap_min
        cooled = now - max(p.last_flip_ts, p.placed_ts) >= self.flip_cooldown
        seasoned = (p.missing_since - p.side_started) >= self.min_side
        if long_enough and cooled and seasoned:
            elapsed = p.side_time[p.side] + (p.missing_since - p.side_started)
            grade = self._grade_flip(p.side, elapsed)
            # an early flip is fixable: the heat the old side missed is owed by
            # the new side, so its target grows by the gap
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
        # else: a short dropout simply folds back into the running side
        p.missing_since = None

    # ---- per-frame update ---------------------------------------------------
    def update(self, dets: list[tuple[float, float, float]], now: float):
        """dets: (cx, cy, r[, conf]) in normalized coords, r = radius/frame_w."""
        if now < self.freeze_until:
            return
        dets = [self._norm_det(d) for d in dets]
        used = set()
        matched = 0
        # Solve the assignment globally, not greedily: with patties packed edge
        # to edge, letting the oldest track pick first hands it a neighbour's
        # detection and orphans the rightful owner — that is where identity
        # fragments come from.
        for p, best in self._assign(dets, now):
            if best is not None:
                used.add(best)
                matched += 1
                cx, cy, r, _cf, lab = dets[best]
                if lab:
                    p.lab = lab if not p.lab else tuple(
                        0.85 * o + 0.15 * n for o, n in zip(p.lab, lab))
                if p.kf is None:
                    p.kf = KF(cx, cy, now)
                p.kf.update(cx, cy, now)
                if self.use_kf:
                    p.cx, p.cy = p.kf.x[0], p.kf.x[1]
                else:
                    p.cx = 0.7 * p.cx + 0.3 * cx
                    p.cy = 0.7 * p.cy + 0.3 * cy
                p.r = 0.8 * p.r + 0.2 * r
                self._on_return(p, now)
                p.last_seen = now
            else:
                if p.missing_since is None:
                    p.missing_since = now

        self.match_stat = (matched, len(self.alive))

        # unmatched detections become new patties — but only confident ones;
        # a flickering low-conf blob may extend a track, never found one
        for i, det in enumerate(dets):
            cx, cy, r, cf, lab = det
            if i in used or cf < self.birth_conf:
                continue
            # no births inside an existing track's reach: a detection that
            # missed its gate by a hair must wait for the track to catch up
            # (the Kalman prediction closes that distance next frame), not
            # mint a duplicate id. Touching neighbours sit 2r apart — safe.
            def near(q):
                qx, qy = q.kf.peek(now) if (self.use_kf and q.kf) else (q.cx, q.cy)
                return math.hypot(cx - qx, cy - qy) < self.birth_suppress * max(q.r, r)
            if any(near(q) for q in self.alive.values()):
                continue
            revived = self._revive(det, now)
            if revived is not None:
                revived.cx, revived.cy, revived.r = cx, cy, r
                self._on_return(revived, now)
                revived.last_seen = now
                self.alive[revived.pid] = revived
                self._emit("revived", revived)
                continue
            p = LivePatty(self.next_pid, cx, cy, r, placed_ts=now,
                          side_started=now, last_seen=now, lab=lab,
                          kf=KF(cx, cy, now))
            self.next_pid += 1
            self.alive[p.pid] = p
            self._emit("placed", p)

        self._enforce_physics(now)

        # removals
        for pid in [pid for pid, p in self.alive.items()
                    if p.missing_since and now - p.missing_since > self.removed_after]:
            p = self.alive.pop(pid)
            p.side_time[p.side] += p.missing_since - p.side_started
            p.side_started = p.missing_since
            total = p.side_time["A"] + p.side_time["B"]
            # keep it in the graveyard: an occlusion longer than removed_after
            # must not cost the patty its history if it comes back
            self.dead.append((now, p))
            self.dead = [d for d in self.dead if now - d[0] <= self.revive_window]
            self._emit("removed", p, {"total": round(total, 1)})
            self.done.append({"pid": p.pid, "side_a": round(p.side_time["A"], 1),
                              "side_b": round(p.side_time["B"], 1),
                              "flips": p.flips, "total": round(total, 1)})

    def _enforce_physics(self, now: float):
        """Patties are solid: two tracks overlapping by half are one patty.

        Tracks drift together when a neighbour steals a detection for a few
        frames; without this pass both keep ticking on the same spot. The
        survivor is the one that is currently visible (a parked ghost loses to
        a live track), ties go to the older history.
        """
        alive = sorted(self.alive.values(), key=lambda q: q.placed_ts)
        doomed = set()
        for i in range(len(alive)):
            for j in range(i + 1, len(alive)):
                a, b = alive[i], alive[j]
                if a.pid in doomed or b.pid in doomed:
                    continue
                # two tracks each holding their own detection this frame are two
                # real patties no matter how close the filter drew them — the
                # merge is for ghosts parked on top of a live track
                if a.last_seen == now and b.last_seen == now:
                    continue
                if math.hypot(a.cx - b.cx, a.cy - b.cy) < self.overlap_frac * (a.r + b.r):
                    a_vis = a.missing_since is None
                    b_vis = b.missing_since is None
                    victim = (b if a_vis else a) if a_vis != b_vis else b
                    doomed.add(victim.pid)
        for pid in doomed:
            q = self.alive.pop(pid)
            self.merged += 1
            self._emit("merged", q)

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
