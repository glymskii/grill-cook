"""Manipulation-anchored event detection.

The windowed colour change-point detector in patties.py cannot resolve a burst of
flips: a cook can turn a patty three times in five seconds, which is faster than
the comparison windows themselves. It also merges a flip with a topping dropped
on right after it.

This module anchors the analysis to the physical event instead. A patty can only
change the side it rests on while it is off the surface, so:

  1. find every *manipulation* — a run of frames where the patty is airborne,
     occluded, or standing on its rim;
  2. compare the visible face immediately before and immediately after it;
  3. a moderate colour step means the patty turned over, a large one means
     something was laid on it, a small one means it was only nudged.

Appearance changes with no manipulation around them are toppings by definition —
nothing can turn a patty that was never picked up.
"""
from dataclasses import dataclass, field

import numpy as np


@dataclass
class FrameObs:
    """What the detector saw on one frame."""
    t: float
    present: bool
    center: tuple = (0.0, 0.0)
    diameter: float = 0.0
    aspect: float = 1.0
    lab: np.ndarray | None = None


@dataclass
class Manipulation:
    t_start: float
    t_end: float
    kind: str = "?"          # FLIP | TOPPING | NUDGE
    delta_e: float = 0.0
    before: np.ndarray | None = None
    after: np.ndarray | None = None


@dataclass
class MpConfig:
    # --- what counts as "off the surface" ---
    lift_drop_l: float = 30.0     # lightness collapse vs resting level
    lift_rise_frac: float = 0.35  # upward jump, in patty diameters
    edge_aspect_mult: float = 1.2 # aspect above resting * this => standing on rim
    min_manip_s: float = 0.12     # ignore single-frame detector hiccups
    merge_gap_s: float = 0.45     # runs closer than this are one manoeuvre
    rest_window_s: float = 4.0    # trailing window defining the resting baseline

    # --- reading the face before/after ---
    face_window_s: float = 0.45   # how much settled footage to average
    settle_s: float = 0.25        # skip this much right after landing
    min_face_samples: int = 4

    # --- classifying the step ---
    flip_delta_e: float = 10.0
    topping_delta_e: float = 30.0


def _median_lab(obs: list[FrameObs], t_from: float, t_to: float, cfg: MpConfig):
    vals = [o.lab for o in obs
            if o.present and o.lab is not None and t_from <= o.t <= t_to]
    if len(vals) < cfg.min_face_samples:
        return None
    return np.median(np.stack(vals), axis=0)


def _settled_face(obs, off, start_i: int, step: int, cfg: MpConfig):
    """Average the face over the nearest genuinely settled stretch.

    Sampling at a fixed offset from the manoeuvre lands mid-tumble whenever the
    cook holds the patty on its rim, which reads as a huge colour step. Walking
    out to the first run of on-surface frames avoids that.
    """
    i = start_i
    n = len(obs)
    run = []
    while 0 <= i < n:
        o = obs[i]
        if not off[i] and o.present and o.lab is not None:
            run.append(o)
            if run[-1].t - run[0].t >= cfg.face_window_s or \
                    run[0].t - run[-1].t >= cfg.face_window_s:
                break
        elif run:
            run = []  # the stretch was interrupted, start over
        i += step
    if len(run) < cfg.min_face_samples:
        return None
    return np.median(np.stack([o.lab for o in run]), axis=0)


def find_manipulations(obs: list[FrameObs], cfg: MpConfig) -> list[Manipulation]:
    """Segment the observation series into manoeuvres and classify each one."""
    if not obs:
        return []

    # rolling baseline of what "resting on the surface" looks like, so a slowly
    # darkening patty or a drifting camera does not accumulate false lifts
    off = []
    for i, o in enumerate(obs):
        if not o.present:
            off.append(True)
            continue
        past = [p for p in obs[max(0, i - 400):i]
                if p.present and p.lab is not None]
        past = [p for p in past if o.t - p.t <= cfg.rest_window_s]
        if len(past) < 5:
            off.append(False)
            continue
        base_l = float(np.median([p.lab[0] for p in past]))
        base_y = float(np.median([p.center[1] for p in past]))
        base_asp = float(np.median([p.aspect for p in past]))
        d = o.diameter or 1.0
        lifted = (o.lab is not None and o.lab[0] < base_l - cfg.lift_drop_l) \
            or (base_y - o.center[1] > cfg.lift_rise_frac * d)
        on_rim = o.aspect > base_asp * cfg.edge_aspect_mult
        off.append(bool(lifted or on_rim))

    # contiguous off-surface runs -> candidate manoeuvres
    runs = []
    i = 0
    while i < len(obs):
        if off[i]:
            j = i
            while j + 1 < len(obs) and off[j + 1]:
                j += 1
            runs.append((obs[i].t, obs[j].t))
            i = j + 1
        else:
            i += 1

    merged = []
    for r in runs:
        if merged and r[0] - merged[-1][1] <= cfg.merge_gap_s:
            merged[-1] = (merged[-1][0], r[1])
        else:
            merged.append(list(r) if False else (r[0], r[1]))
    merged = [m for m in merged if m[1] - m[0] >= cfg.min_manip_s]

    t_index = {round(o.t, 4): i for i, o in enumerate(obs)}
    out = []
    for t0, t1 in merged:
        i0 = t_index.get(round(t0, 4), 0)
        i1 = t_index.get(round(t1, 4), len(obs) - 1)
        before = _settled_face(obs, off, max(0, i0 - 1), -1, cfg)
        after = _settled_face(obs, off, min(len(obs) - 1, i1 + 1), +1, cfg)
        if before is None or after is None:
            continue
        delta = float(np.linalg.norm(after - before))
        kind = ("TOPPING" if delta > cfg.topping_delta_e
                else "FLIP" if delta > cfg.flip_delta_e else "NUDGE")
        out.append(Manipulation(t0, t1, kind, round(delta, 1), before, after))
    return out


def side_timeline(obs: list[FrameObs], manips: list[Manipulation],
                  t_placed: float, t_removed: float):
    """Accumulate per-side seconds, alternating sides on every FLIP.

    Time spent off the surface belongs to neither side and is reported apart.
    """
    timers = {1: 0.0, 2: 0.0}
    side = 1
    airborne = 0.0
    cursor = t_placed
    events = [("PLACED", t_placed, "")]

    for m in manips:
        if m.t_end <= t_placed or m.t_start >= t_removed:
            continue
        start = max(m.t_start, cursor)
        timers[side] += max(0.0, start - cursor)
        airborne += max(0.0, min(m.t_end, t_removed) - start)
        cursor = min(m.t_end, t_removed)
        if m.kind == "FLIP":
            side = 3 - side
            events.append(("FLIP", m.t_start, f"-> side {side}, dE {m.delta_e}"))
        elif m.kind == "TOPPING":
            events.append(("TOPPING", m.t_start, f"dE {m.delta_e}"))
    timers[side] += max(0.0, t_removed - cursor)
    events.append(("REMOVED", t_removed, ""))
    return timers, airborne, events
