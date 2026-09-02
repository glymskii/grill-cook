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
    lab_hist: list = field(default_factory=list)   # (t, lab) for crust-change flips
    crust_pending: tuple | None = None      # (t_change, ref_L, crowd_L) awaiting proof
    face_hist: list = field(default_factory=list)   # recent face classes (0 raw/1 cooked/2 cheese)
    face: int | None = None                 # stable face, changes only on agreement
    cheesed: bool = False                   # cheese went on: this patty is finished
    last_det: tuple | None = None           # raw detection centre, unsmoothed
    motion: float = 0.0                     # EMA of raw displacement, in radii
    handling: bool = False                  # in the cook's hands right now
    episode_gap: float = 0.0                # longest disappearance in this episode
    face_before: int | None = None          # face it showed before this episode
    settled_since: float = 0.0              # when it last came to rest
    other_since: float = 0.0                # first frame the crop stopped being a patty
    kf: KF | None = None
    sku: str | None = None                  # which standard applies, once the patty has shown its colour
    a_hist: list = field(default_factory=list)   # (t, raw a* of the matched box), for the SKU only
    cheesed_ts: float = 0.0                 # when the cheese flag went up, for the log panel

    def elapsed(self, now: float) -> float:
        if self.missing_since is not None:
            return self.side_time[self.side] + (self.missing_since - self.side_started)
        return self.side_time[self.side] + (now - self.side_started)


class TimerEngine:
    def __init__(self, targets: dict, flip_gap_min=1.2, removed_after=6.0,
                 flip_cooldown=45.0, min_side_before_flip=25.0, assoc_frac=1.6,
                 birth_conf=0.40, gate_base=1.45, gate_max=1.8, size_gate=1.55,
                 colour_scale=26.0, revive_window=60.0, revive_colour=22.0,
                 overlap_frac=0.55, use_kf="motion", birth_suppress=1.2,
                 kf_min_age=20.0, colour_win=2.5, flip_dl=22.0, topping_db=12.0,
                 crust_confirm=4.0, face_window=3.0, face_agree=0.85,
                 static_motion=0.12, lifted_after=2.0, handle_motion=0.35,
                 settle_time=4.0, other_grace=2.0):
        self.targets = targets                  # {"A": s, "B": s, "tol_early": s, "tol_late": s}
        self.flip_gap_min = flip_gap_min
        self.removed_after = removed_after
        self.flip_cooldown = flip_cooldown
        self.min_side = min_side_before_flip
        self.assoc_frac = assoc_frac
        # Hysteresis: a new track has to clear a bar, an established one survives
        # on far less. The bar was raised to 0.55 to hide the detector's 0.2
        # grease blobs; v7 no longer produces them, so it comes back down —
        # a cheese-covered patty can read 0.5, and missing a real patty is a
        # worse failure than a ring that lives for a moment.
        self.birth_conf = birth_conf
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
        self.use_kf = use_kf              # False | True | "hybrid"
        self.birth_suppress = birth_suppress
        # hybrid: trust the filter only for tracks that have proven themselves —
        # transitions (empty start, cleaning, re-layouts) stay on the plain
        # engine where the filter's momentum does more harm than good
        self.kf_min_age = kf_min_age
        # A smash flip never lifts the patty clear of the frame, so waiting for
        # it to disappear misses the event entirely. The crust does the talking:
        # the face swaps from raw to seared within a couple of seconds. Cheese
        # lands just as fast, but it drives b* up (yellow) while a flip drives
        # L* down (dark) — direction separates them.
        self.colour_win = colour_win
        # the camera rides its own auto-exposure: when the cook leans in, every
        # patty darkens at once. Only a change RELATIVE to the rest of the
        # griddle is a flip — absolute darkening is the room, not the meat.
        self.lum_hist: list[tuple[float, float]] = []
        self.flip_dl = flip_dl
        self.topping_db = topping_db
        # A spatula sliding under a patty to lift it off darkens it just like a
        # flip does — for a second or two. A real flip leaves it dark for good,
        # so a candidate has to still be dark after this long to count.
        self.crust_confirm = crust_confirm
        # Brightness thresholds could not tell a flip from cheese landing or a
        # spatula shadow — 85% of their flips were false. The face classifier
        # names what it sees instead, and a face only changes when most of the
        # recent votes agree, so one bad frame cannot flip a patty.
        self.static_motion = static_motion
        # a static patty that disappears has been lifted off; holding its track
        # open for the full removed_after only gives it time to steal a neighbour
        self.lifted_after = lifted_after
        self.other_grace = other_grace
        self.handle_motion = handle_motion
        self.settle_time = settle_time
        self.face_window = face_window
        self.face_agree = face_agree
        self.alive: dict[int, LivePatty] = {}
        self.done: list[dict] = []
        self.next_pid = 1
        self.session = {"flips": 0, "optimal": 0, "early": 0, "late": 0, "streak": 0}
        self.on_event = None                    # optional hook for the edge worker
        self.freeze_until = 0.0                 # settle window after a scene cut
        self.scene_cut_ts = 0.0
        self.has_faces = False
        self.rejected_other = 0
        self.match_stat = (0, 0)                # (matched alive, total alive) per frame
        self._now = 0.0                         # engine clock, for offline replay

    # ---- events -------------------------------------------------------------
    def _emit_raw(self, ev: dict):
        EVENTS.parent.mkdir(parents=True, exist_ok=True)
        with EVENTS.open("a") as f:
            f.write(json.dumps(ev, ensure_ascii=False) + "\n")
        if self.on_event:
            self.on_event(ev)

    def targets_for(self, p: LivePatty) -> dict:
        """The standard for THIS patty. A station cooks more than one product;
        with `skus` in the targets a patty is assigned by the colour it rests at
        (pale chicken vs pink beef: redness a* below/above `a_below`, measured
        138-144 against 123-126 with nothing in between) once it has shown it
        for 20 quiet readings. Until then, and without skus, the station-wide
        targets apply. Ported from the slot engine, where 13 of 13 patties on
        the long shift were assigned right."""
        skus = self.targets.get("skus")
        if not skus:
            return self.targets
        if len(p.a_hist) >= 20 and self._now - p.placed_ts >= 20:
            # The median of the raw box redness over the last 15 s - not the
            # smoothed colour: a track that flickers between two touching
            # patties (this engine still does that) averages the two into a
            # value between the classes, while the median follows the majority.
            # Re-read on every call, with hysteresis.
            a = sorted(v for _, v in p.a_hist)[len(p.a_hist) // 2]
            mid = skus.get("a_below", 132)
            if p.sku is None:
                p.sku = "pale" if a < mid else "dark"
            elif p.sku == "pale" and a > mid + 3:
                p.sku = "dark"
            elif p.sku == "dark" and a < mid - 3:
                p.sku = "pale"
        return skus.get(p.sku, self.targets) if p.sku else self.targets

    def _emit(self, kind: str, p: LivePatty, extra: dict | None = None):
        tg = self.targets_for(p)
        ev = {"ts": round(time.time(), 2), "ts_video": round(self._now, 2),
              "type": kind, "pid": p.pid,
              "side": p.side, "flips": p.flips,
              "side_a": round(p.side_time["A"], 1), "side_b": round(p.side_time["B"], 1),
              "ta": tg["A"], "tb": tg["B"],
              "te": tg["tol_early"], "tl": tg["tol_late"], "sku": p.sku}
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

    def _grade_flip(self, side: str, elapsed: float, t: dict | None = None) -> str:
        t = t or self.targets
        if elapsed < t[side] - t["tol_early"]:
            return "early"
        if elapsed > t[side] + t["tol_late"]:
            return "late"
        return "optimal"

    # ---- association ---------------------------------------------------------
    @staticmethod
    def _norm_det(d):
        """(cx, cy, r[, conf][, lab][, face]) -> a uniform 6-tuple."""
        d = tuple(d)
        return (d[0], d[1], d[2],
                d[3] if len(d) > 3 else 1.0,
                d[4] if len(d) > 4 else None,
                d[5] if len(d) > 5 else None)

    def _cost(self, p: "LivePatty", det, now: float):
        """Match cost for one track/detection pair, or None if impossible."""
        cx, cy, r, _cf, lab, _fc = det
        ref = max((p.r + r) / 2, 1e-6)
        # Prediction is for FINDING the patty, smoothing is for reporting where it
        # is. The filter barely stirs for a patty at rest (0.01r of drift over a
        # noisy minute), so leaning on it here costs nothing and keeps a carried
        # patty attached instead of snapping and being born again.
        ex, ey = p.kf.peek(now) if p.kf else (p.cx, p.cy)
        dist = math.hypot(cx - ex, cy - ey) / ref
        gap = 0.0 if p.missing_since is None else now - p.missing_since
        # The gate widens while a track is missing — but only if that patty was
        # actually being pushed around. A patty that sat still and then vanished
        # was lifted off the griddle, and searching wider for it just hands the
        # track its neighbour: 129 of those jumps happened in one shift.
        widen = 0.25 * gap if p.motion > self.static_motion else 0.0
        if dist > min(self.gate_base + widen, self.gate_max):
            return None
        ratio = max(p.r, r) / max(min(p.r, r), 1e-6)
        if ratio > self.size_gate:
            return None
        cost = dist + 1.5 * (ratio - 1.0)
        if lab and p.lab:
            cost += min(_lab_dist(lab, p.lab) / self.colour_scale, 1.5)
        return cost

    def _trust_kf(self, p: "LivePatty", now: float) -> bool:
        """Predict only for a patty that is actually being carried.

        The filter's momentum is a liability on a patty sitting still — box
        jitter becomes phantom velocity — and an asset on one riding a spatula,
        where the smoothed centre lags so far behind that the track snaps and
        the same patty is born again. Motion, not age, is the axis that
        separates the two cases.
        """
        if p.kf is None:
            return False
        if self.use_kf == "motion":
            return p.motion > self.handle_motion
        if self.use_kf == "hybrid":
            return now - p.placed_ts >= self.kf_min_age
        return bool(self.use_kf)

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
        cx, cy, r, _cf, lab, _fc = det
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

    def _dominant(self, hist, t0: float, t1: float):
        """The face this patty held between t0 and t1, if it held one at all."""
        votes = [f for (t, f) in hist if t0 <= t <= t1]
        if len(votes) < 4:
            return None
        top = max(set(votes), key=votes.count)
        return top if votes.count(top) / len(votes) >= self.face_agree else None

    def _face_pass(self, now: float):
        """A flip is a face that changed ACROSS a handling episode.

        Judging the face at an arbitrary moment was the mistake: a shadow, an
        exposure swing or a slice of cheese all change what the camera sees
        without anyone turning the patty. A flip has a shape — the patty is
        picked up or shoved, then it settles showing the other side — so the
        comparison is anchored to that episode and nothing else can imitate it.
        """
        doomed = []
        for p in self.alive.values():
            # The face is read every frame, not only when an episode ends: cheese
            # is laid on a patty that never moves, so waiting for a handling
            # episode left the HUD counting down and shouting FLIP at a patty
            # that was already finished.
            stable = self._dominant(p.face_hist, now - self.face_window, now)
            if stable is not None:
                p.face = stable
                if stable == 2:
                    p.cheesed = True
                    p.cheesed_ts = p.cheesed_ts or self._now
                if stable == 3:
                    p.other_since = p.other_since or now
                    if now - p.other_since > self.other_grace:
                        doomed.append(p.pid)   # the crop is griddle, not a patty
                else:
                    p.other_since = 0.0
            busy = p.missing_since is not None or p.motion > self.handle_motion
            if busy:
                if not p.handling:
                    p.handling = True
                    p.episode_gap = 0.0
                    p.face_before = self._dominant(p.face_hist, now - 6.0, now - 0.3)
                if p.missing_since is not None:
                    p.episode_gap = max(p.episode_gap, now - p.missing_since)
                p.settled_since = 0.0
                continue
            if not p.handling:
                continue
            if not p.settled_since:
                p.settled_since = now
                continue
            if now - p.settled_since < self.settle_time:
                continue
            after = self._dominant(p.face_hist, now - self.face_window, now)
            if after is None:
                continue
            before = p.face_before
            gap = p.episode_gap
            p.handling, p.face_before, p.episode_gap = False, None, 0.0
            if after == 2:
                p.cheesed = True
                p.cheesed_ts = p.cheesed_ts or self._now
            if p.cheesed:
                continue
            # Once both sides are seared they look the same, so a second flip is
            # invisible to the face alone. A patty that genuinely left the
            # griddle and came back settled is one either way.
            turned = (before is not None and after != before) or gap >= self.flip_gap_min
            if not turned:
                continue
            if after in (2, 3) or before in (2, 3):
                continue                      # cheese landed, or the crop lost the patty
            if now - max(p.last_flip_ts, p.placed_ts) < self.flip_cooldown:
                continue
            if now - p.side_started < self.min_side:
                continue
            self._do_flip(p, now, p.side_time[p.side] + (now - p.side_started), "face")

        for pid in doomed:
            q = self.alive.pop(pid, None)
            if q is not None:
                self._emit("dropped", q, {"reason": "not-a-patty"})

    def _crust_pass(self, now: float):
        """Flip the patties whose crust darkened against their neighbours — and stayed dark.

        Absolute darkening is the camera's auto-exposure: when the cook leans
        in, the whole griddle dims and every patty looks flipped. Comparing each
        patty to the median change of the others cancels that. A spatula lifting
        a patty off the griddle also darkens it, so the change must still hold
        crust_confirm seconds later before it counts as a flip.
        """
        vis = [p for p in self.alive.values()
               if p.missing_since is None and p.lab and p.lab_hist]
        if not vis:
            return
        crowd_now = sorted(p.lab[0] for p in vis)[len(vis) // 2]

        # --- confirm or drop candidates raised earlier -----------------------
        for p in vis:
            if not p.crust_pending:
                continue
            t0, ref_l, crowd_ref = p.crust_pending
            if now - t0 < self.crust_confirm:
                continue
            still = (ref_l - p.lab[0]) - (crowd_ref - crowd_now)
            p.crust_pending = None
            if still >= self.flip_dl * 0.7:
                p.side_started = min(p.side_started, t0)      # credit from the turn
                self._do_flip(p, now, p.side_time[p.side] + (t0 - p.side_started),
                              "crust")

        # --- raise new candidates -------------------------------------------
        cands = []
        for p in vis:
            if p.crust_pending:
                continue
            ref = None
            for (t, lab) in p.lab_hist:
                if now - t >= self.colour_win:
                    ref = lab
                else:
                    break
            if ref is not None:
                cands.append((p, ref[0] - p.lab[0], p.lab[2] - ref[2], ref[0]))
        if not cands:
            return
        drops = sorted(d for _, d, _, _ in cands)
        crowd = drops[len(drops) // 2] if len(cands) >= 3 else 0.0
        for p, drop, db, ref_l in cands:
            if db > self.topping_db:                  # cheese or sauce, not a flip
                continue
            if drop - crowd < self.flip_dl:
                continue
            if now - max(p.last_flip_ts, p.placed_ts) < self.flip_cooldown:
                continue
            if now - p.side_started < self.min_side:
                continue
            p.crust_pending = (now, ref_l, crowd_now)

    def _do_flip(self, p: "LivePatty", now: float, elapsed: float, source: str):
        tg = self.targets_for(p)
        grade = self._grade_flip(p.side, elapsed, tg)
        p.bonus = round(tg[p.side] - elapsed, 1) if grade == "early" else 0.0
        p.side_time[p.side] = elapsed
        p.side = "B" if p.side == "A" else "A"
        p.side_started = now
        p.flips += 1
        p.last_flip_ts = now
        p.flip_feedback = grade
        p.lab_hist.clear()                            # new face, new baseline
        p.crust_pending = None
        p.face_hist.clear()
        self.session["flips"] += 1
        self.session[grade] += 1
        self.session["streak"] = self.session["streak"] + 1 if grade == "optimal" else 0
        self._emit("flip", p, {"grade": grade, "elapsed": round(elapsed, 1),
                               "bonus": p.bonus, "via": source})

    def _on_return(self, p: "LivePatty", now: float):
        """A patty is visible again: decide whether the gap was a flip.

        Shared by the normal match path and by revival, so a track that came
        back from the graveyard is judged exactly like one that never died.
        """
        if p.missing_since is None:
            return
        gap = now - p.missing_since
        # Both signals stay on: the face names a flip the moment a raw side is
        # turned down, and the gap catches the ones the face cannot see at all —
        # once both sides are seared they look identical. Measured against hand
        # labels, the pair beats either alone (F1 0.49 vs 0.45 and 0.38).
        long_enough = gap >= self.flip_gap_min
        cooled = now - max(p.last_flip_ts, p.placed_ts) >= self.flip_cooldown
        seasoned = (p.missing_since - p.side_started) >= self.min_side
        if long_enough and cooled and seasoned:
            elapsed = p.side_time[p.side] + (p.missing_since - p.side_started)
            self._do_flip(p, now, elapsed, "gap")
        # else: a short dropout simply folds back into the running side
        p.missing_since = None

    # ---- per-frame update ---------------------------------------------------
    def update(self, dets: list[tuple[float, float, float]], now: float):
        """dets: (cx, cy, r[, conf]) in normalized coords, r = radius/frame_w."""
        self._now = now
        lums = sorted(d[4][0] for d in dets if len(d) > 4 and d[4])
        if lums:
            self.lum_hist.append((now, lums[len(lums) // 2]))
            cut = now - 4 * self.colour_win
            while self.lum_hist and self.lum_hist[0][0] < cut:
                self.lum_hist.pop(0)
        if now < self.freeze_until:
            return
        dets = [self._norm_det(d) for d in dets]
        self.rejected_other = sum(1 for d in dets if d[5] == 3)
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
                cx, cy, r, _cf, lab, face = dets[best]
                if p.last_det:
                    step = math.hypot(cx - p.last_det[0], cy - p.last_det[1]) / max(r, 1e-6)
                    p.motion = 0.75 * p.motion + 0.25 * min(step, 3.0)
                p.last_det = (cx, cy)
                if lab:
                    p.a_hist.append((now, lab[1]))
                    p.a_hist = [x for x in p.a_hist if x[0] >= now - 15]
                    p.lab = lab if not p.lab else tuple(
                        0.85 * o + 0.15 * n for o, n in zip(p.lab, lab))
                    p.lab_hist.append((now, p.lab))
                    cut = now - 6 * self.colour_win
                    while p.lab_hist and p.lab_hist[0][0] < cut:
                        p.lab_hist.pop(0)
                if face is not None:
                    p.face_hist.append((now, face))
                    cut = now - 12.0
                    while p.face_hist and p.face_hist[0][0] < cut:
                        p.face_hist.pop(0)
                if p.kf is None:
                    p.kf = KF(cx, cy, now)
                p.kf.update(cx, cy, now)
                if self._trust_kf(p, now):
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
        if any(d[5] is not None for d in dets):
            self.has_faces = True
            self._face_pass(now)
        else:
            self._crust_pass(now)

        # unmatched detections become new patties — but only confident ones;
        # a flickering low-conf blob may extend a track, never found one
        for i, det in enumerate(dets):
            cx, cy, r, cf, lab, face = det
            if i in used or cf < self.birth_conf:
                continue
            if face == 3:
                # griddle, a glove or a spatula: it may keep an existing track
                # alive through a bad frame, but it must never start a new one
                continue
            # no births inside an existing track's reach: a detection that
            # missed its gate by a hair must wait for the track to catch up
            # (the Kalman prediction closes that distance next frame), not
            # mint a duplicate id. Touching neighbours sit 2r apart — safe.
            def near(q):
                qx, qy = q.kf.peek(now) if q.kf else (q.cx, q.cy)
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
                          kf=KF(cx, cy, now), face=face)
            self.next_pid += 1
            self.alive[p.pid] = p
            self._emit("placed", p)

        self._enforce_physics(now)

        # removals
        for pid in [pid for pid, p in self.alive.items()
                    if p.missing_since and now - p.missing_since >
                    (self.removed_after if p.motion > self.static_motion else self.lifted_after)]:
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
            tg = self.targets_for(p)
            target = tg[p.side] + p.bonus
            elapsed = p.elapsed(now)
            fb, p.flip_feedback = p.flip_feedback, None    # one-shot to the HUD
            out.append({
                "pid": p.pid, "x": round(p.cx, 4), "y": round(p.cy, 4),
                "r": round(p.r, 4), "side": p.side, "flips": p.flips,
                "elapsed": round(elapsed, 1), "target": target,
                "bonus": p.bonus,
                "deadline": round(now - elapsed + target, 2),
                "cheesed": p.cheesed,
                "face": p.face,
                "missing": p.missing_since is not None,
                "missing_for": round(now - p.missing_since, 1) if p.missing_since else 0.0,
                "feedback": fb,
                "sku": p.sku, "tol_late": tg["tol_late"],
                # the log panel: what this patty has spent so far, side by side
                "side_a": round(p.side_time["A"] + (elapsed - p.side_time[p.side]
                                                    if p.side == "A" else 0), 1),
                "side_b": round(p.side_time["B"] + (elapsed - p.side_time[p.side]
                                                    if p.side == "B" else 0), 1),
                "total": round(now - p.placed_ts, 1),
                "cheese_for": round(now - p.cheesed_ts, 1) if p.cheesed_ts else 0.0,
            })
        return {"patties": out, "session": dict(self.session), "done": len(self.done),
                "scene_cut_ts": round(self.scene_cut_ts, 2)}
