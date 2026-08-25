"""Logical patty tracking on top of raw tracker ids + per-side frying timers.

State machine per patty:
  PLACED  -> side 1 down, timer(1) runs
  FLIP    -> active timer switches side (any number of flips)
  REMOVED -> both timers frozen

Flip detection: step change in the patty's central Lab color, tested by comparing
a short "recent" window against an earlier "baseline" window. A tracking gap or a
positional jump (spatula lift) lowers the required color threshold.
"""
from collections import deque
from dataclasses import dataclass, field

import numpy as np

from config import Config
from detector import Detection, mean_lab


@dataclass
class Event:
    type: str            # PLACED | FLIP | REMOVED
    t: float             # video time, seconds
    frame: int
    detail: dict = field(default_factory=dict)


class Patty:
    _next_id = 1

    def __init__(self, cfg: Config, fps: float, frame_idx: int, det: Detection):
        self.cfg = cfg
        self.fps = fps
        self.id = Patty._next_id
        Patty._next_id += 1

        self.raw_ids = {det.raw_id}
        self.state = "ACTIVE"
        self.side_down = 1
        self.timers = {1: 0.0, 2: 0.0}
        self.n_flips = 0
        self.n_toppings = 0

        self.first_frame = frame_idx
        self.last_seen_frame = frame_idx
        self.last_xyxy = det.xyxy
        self.n_dets = 1

        self.feats = deque()          # (frame_idx, lab ndarray)
        self.last_flip_t = -1e9
        self.gap_boost_until = -1.0   # time until which the lowered flip threshold applies
        self.no_flip_until = -1.0     # scene-cut suppression window

        self.events = [Event("PLACED", frame_idx / fps, frame_idx)]

    # ---------- helpers ----------
    def t(self, frame_idx: int) -> float:
        return frame_idx / self.fps

    def _window_median(self, t_from: float, t_to: float):
        vals = [f for fi, f in self.feats if t_from <= self.t(fi) <= t_to]
        # demand the window be mostly filled — a half-empty window right after a
        # scene cut or occlusion gives an unreliable baseline
        need = max(3, int((t_to - t_from) * self.fps * self.cfg.win_fill_frac))
        if len(vals) < need:
            return None
        return np.median(np.stack(vals), axis=0)

    # ---------- updates ----------
    def update(self, frame_idx: int, det: Detection, frame_bgr, scene_cut=False):
        cfg = self.cfg
        now = self.t(frame_idx)

        # accumulate frying time for the side currently down
        dt_frames = frame_idx - self.last_seen_frame
        dt = min(dt_frames / self.fps, cfg.stitch_max_gap_s)
        if dt > 0:
            self.timers[self.side_down] += dt

        # positional jump => displacement cue (spatula action);
        # ignored around scene cuts where reframing moves everything
        jump = np.hypot(det.center[0] - (self.last_xyxy[0] + self.last_xyxy[2]) / 2,
                        det.center[1] - (self.last_xyxy[1] + self.last_xyxy[3]) / 2)
        if not scene_cut and now >= self.no_flip_until and \
                (dt_frames > 1 or jump > cfg.jump_dist_frac * det.diameter):
            self.gap_boost_until = now + cfg.recent_win_s + cfg.baseline_gap_s

        self.last_seen_frame = frame_idx
        self.last_xyxy = det.xyxy
        self.n_dets += 1

        feat = mean_lab(frame_bgr, det.xyxy, cfg.feat_inner_frac)
        if feat is not None:
            self.feats.append((frame_idx, feat))
            horizon = cfg.baseline_win_s + cfg.baseline_gap_s + cfg.recent_win_s + 1.0
            while self.feats and self.t(self.feats[0][0]) < now - horizon:
                self.feats.popleft()

        self._check_flip(now)

    def _check_flip(self, now: float):
        cfg = self.cfg
        if now < self.no_flip_until or now - self.last_flip_t < cfg.flip_cooldown_s:
            return
        recent = self._window_median(now - cfg.recent_win_s, now)
        b_to = now - cfg.recent_win_s - cfg.baseline_gap_s
        baseline = self._window_median(b_to - cfg.baseline_win_s, b_to)
        if recent is None or baseline is None:
            return
        delta = float(np.linalg.norm(recent - baseline))
        disturbed = now <= self.gap_boost_until
        # a patty cannot turn over without being physically disturbed; without
        # this gate, anything dropped on top (cheese, sauce) reads as a flip
        if cfg.require_disturbance and not disturbed:
            return
        thr = cfg.flip_gap_delta_e if disturbed else cfg.flip_delta_e
        if delta < thr:
            return

        t_step = now - cfg.recent_win_s - cfg.baseline_gap_s / 2
        if delta > cfg.topping_delta_e:
            # something was placed on the patty: the side facing the surface is
            # unchanged, so the running timer keeps going
            self.n_toppings += 1
            self.last_flip_t = now
            self.feats.clear()
            self.gap_boost_until = -1.0
            self.events.append(Event("TOPPING", t_step, int(round(t_step * self.fps)),
                                     {"delta_e": round(delta, 1)}))
            return

        # flip happened roughly at the start of the transient
        t_flip = max(t_step, self.last_flip_t + 0.1)
        retro = max(0.0, now - t_flip)
        old, new = self.side_down, 3 - self.side_down
        moved = min(retro, self.timers[old])
        self.timers[old] -= moved
        self.timers[new] += moved
        self.side_down = new
        self.n_flips += 1
        self.last_flip_t = now
        # reset the appearance history: the pre-flip baseline would otherwise
        # keep matching against the new look and fire a second, phantom flip
        self.feats.clear()
        self.gap_boost_until = -1.0
        frame = int(round(t_flip * self.fps))
        self.events.append(Event("FLIP", t_flip, frame,
                                 {"from_side": old, "to_side": new, "delta_e": round(delta, 1)}))

    def mark_removed(self):
        if self.state == "ACTIVE":
            self.state = "REMOVED"
            t = self.last_seen_frame / self.fps
            self.events.append(Event("REMOVED", t, self.last_seen_frame))


class PattyTracker:
    """Stitches raw tracker ids into logical patties and drives their state machines."""

    def __init__(self, cfg: Config, fps: float, frame_diag: float):
        self.cfg = cfg
        self.fps = fps
        self.frame_diag = frame_diag
        self.patties: list[Patty] = []
        self.raw2patty: dict[int, Patty] = {}
        self.last_cut_frame = -10 ** 9

    def step(self, frame_idx: int, frame_bgr, dets: list[Detection],
             scene_cut: bool = False):
        cfg = self.cfg
        if scene_cut:
            self.last_cut_frame = frame_idx
            now = frame_idx / self.fps
            for p in self.patties:
                if p.state == "ACTIVE":
                    p.feats.clear()  # appearance not comparable across the cut
                    p.no_flip_until = now + cfg.cut_suppress_s
                    p.gap_boost_until = -1.0
        # reframing may displace everything: widen the stitch radius briefly
        near_cut = (frame_idx - self.last_cut_frame) / self.fps < 1.0
        radius_mult = cfg.cut_stitch_mult if near_cut else 1.0

        # pass 1: known raw ids (id -1 = untracked detection, never map it)
        unmatched = []
        for det in dets:
            p = self.raw2patty.get(det.raw_id) if det.raw_id != -1 else None
            if p is not None and p.state == "ACTIVE":
                p.update(frame_idx, det, frame_bgr, scene_cut)
            else:
                unmatched.append(det)

        # pass 2: stitch to a recently lost patty, else new patty
        for det in unmatched:
            best, best_d = None, 1e18
            max_d = cfg.stitch_max_dist_diam * det.diameter * radius_mult
            for p in self.patties:
                if p.state != "ACTIVE" or p.last_seen_frame >= frame_idx:
                    continue
                gap_s = (frame_idx - p.last_seen_frame) / self.fps
                if gap_s > cfg.stitch_max_gap_s:
                    continue
                lx = (p.last_xyxy[0] + p.last_xyxy[2]) / 2
                ly = (p.last_xyxy[1] + p.last_xyxy[3]) / 2
                d = np.hypot(det.center[0] - lx, det.center[1] - ly)
                if d < max_d and d < best_d:
                    best, best_d = p, d
            if best is not None:
                if det.raw_id != -1:
                    best.raw_ids.add(det.raw_id)
                    self.raw2patty[det.raw_id] = best
                best.update(frame_idx, det, frame_bgr, scene_cut)
            elif self._duplicates_active(frame_idx, det):
                continue  # second box on a patty already tracked this frame
            else:
                p = Patty(cfg, self.fps, frame_idx, det)
                self.patties.append(p)
                if det.raw_id != -1:
                    self.raw2patty[det.raw_id] = p

        # removal check
        for p in self.patties:
            if p.state == "ACTIVE":
                lost_s = (frame_idx - p.last_seen_frame) / self.fps
                if lost_s > cfg.removed_after_s:
                    p.mark_removed()

    def _duplicates_active(self, frame_idx: int, det: Detection) -> bool:
        """True if this box overlaps a patty already updated on this frame."""
        ax1, ay1, ax2, ay2 = det.xyxy
        area_a = max(1e-6, (ax2 - ax1) * (ay2 - ay1))
        for p in self.patties:
            if p.state != "ACTIVE" or p.last_seen_frame != frame_idx:
                continue
            bx1, by1, bx2, by2 = p.last_xyxy
            iw = max(0.0, min(ax2, bx2) - max(ax1, bx1))
            ih = max(0.0, min(ay2, by2) - max(ay1, by1))
            inter = iw * ih
            area_b = max(1e-6, (bx2 - bx1) * (by2 - by1))
            if inter / (area_a + area_b - inter) > self.cfg.dup_iou:
                return True
        return False

    def finish(self, last_frame_idx: int):
        """End of video: close out active patties and drop noise tracks."""
        for p in self.patties:
            if p.state == "ACTIVE":
                lost_s = (last_frame_idx - p.last_seen_frame) / self.fps
                if lost_s > self.cfg.removed_after_s:
                    p.mark_removed()
                elif lost_s >= self.cfg.eos_removed_after_s:
                    # the recording stopped before the removal timeout could
                    # elapse; the patty did leave the surface, so keep the
                    # measurement but flag that it is unconfirmed
                    p.mark_removed()
                    p.state = "REMOVED_EOS"
                else:
                    p.state = "CENSORED"  # still frying when video ended
        keep = []
        for p in self.patties:
            life_s = (p.last_seen_frame - p.first_frame) / self.fps
            if life_s >= self.cfg.min_track_len_s and p.n_dets >= 5:
                keep.append(p)
        self.patties = keep
        return self.patties
