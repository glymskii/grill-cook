"""Slot engine: the griddle is a board of fixed places, not a video of moving things.

The camera is bolted down and a frying patty does not go anywhere — measured on
the hand-labelled windows, a patty is at rest 82-99% of its life, makes one
move, and 90-100% of those moves are under half a radius, i.e. a nudge inside
the same place. What looks like motion is the detector: 97-98% of consecutive
detections sit within 0.06 r of each other, and the remaining 1-3% jump up to
1.5 r on smoke, grease and hands.

So position is not something to track frame by frame. It is decided once, when
the patty is placed and comes to rest, and then held. Detections stop being
things to associate and become answers to a local question — "is my patty still
in its place?". A hand covering the slot no longer costs identity, because
identity is the place, not the box.

That buys the thing the old design could not have: whatever happens over a slot,
the SAME pixels are there before and after. A flip is then decided by looking at
the patty, not by noticing that a track went missing.

Prototype: scored against the production engine before being wired in.
"""
import json
import math
import statistics as st
import time
from dataclasses import dataclass, field
from pathlib import Path

EVENTS = Path(__file__).resolve().parent.parent / "app" / "state" / "slot_events.jsonl"


def _lab_dist(a, b) -> float:
    return math.dist(a, b) if a and b else 0.0


@dataclass
class Slot:
    sid: int
    ax: float                      # anchor: frozen once the patty settles
    ay: float
    ar: float
    placed_ts: float
    side: str = "A"
    side_started: float = 0.0
    side_time: dict = field(default_factory=lambda: {"A": 0.0, "B": 0.0})
    flips: int = 0
    anchored: bool = False
    births: list = field(default_factory=list)     # (t, x, y, r) while settling
    seen_ts: float = 0.0
    absent_since: float | None = None
    face_hist: list = field(default_factory=list)  # (t, class)
    lab_hist: list = field(default_factory=list)   # (t, lab)
    face: int | None = None
    cheesed: bool = False
    episode: dict | None = None    # open manipulation, or one awaiting its verdict
    offsets: list = field(default_factory=list)    # (t, dx, dy) of claims, for re-anchoring
    bonus: float = 0.0
    flip_feedback: str | None = None
    last_flip_ts: float = 0.0

    def elapsed(self, now: float) -> float:
        ref = self.absent_since if self.absent_since is not None else now
        return self.side_time[self.side] + (ref - self.side_started)


class SlotEngine:
    def __init__(self, targets: dict, claim_frac=0.75, settle_time=2.0,
                 settle_tol=0.25, birth_conf=0.40, birth_suppress=1.3,
                 removed_after=6.0, verdict_delay=2.5, face_window=2.5,
                 face_agree=0.6, flip_dl=18.0, flip_cooldown=45.0,
                 min_side_before_flip=25.0, reanchor_tol=0.35, migrate_frac=3.0,
                 migrate_window=4.0, other_grace=2.0, episode_min=0.6):
        self.targets = targets
        self.claim_frac = claim_frac       # how far from its anchor a patty may be found
        self.settle_time = settle_time     # rest needed before the anchor is frozen
        self.settle_tol = settle_tol
        self.birth_conf = birth_conf
        self.birth_suppress = birth_suppress
        self.removed_after = removed_after
        # A verdict is never taken on the frame the patty reappears: one frame of
        # a hand pulling away reads as anything. The slot waits until the crop has
        # been quiet this long, then judges.
        self.verdict_delay = verdict_delay
        self.face_window = face_window
        self.face_agree = face_agree
        self.flip_dl = flip_dl
        self.flip_cooldown = flip_cooldown
        self.min_side = min_side_before_flip
        self.reanchor_tol = reanchor_tol
        self.migrate_frac = migrate_frac
        self.migrate_window = migrate_window
        self.other_grace = other_grace
        self.episode_min = episode_min
        self.slots: dict[int, Slot] = {}
        self.next_sid = 1
        self.done: list[dict] = []
        self.session = {"flips": 0, "optimal": 0, "early": 0, "late": 0, "streak": 0}
        self.on_event = None
        self.lum_hist: list[tuple[float, float]] = []
        self._now = 0.0
        self.freeze_until = 0.0
        self.scene_cut_ts = 0.0
        # why a verdict said "nothing happened here" — the tuning is done on this
        self.stats: dict[str, int] = {}

    # ---- events -------------------------------------------------------------
    def _emit(self, kind: str, s: Slot, extra: dict | None = None):
        ev = {"ts": round(time.time(), 2), "ts_video": round(self._now, 2),
              "type": kind, "pid": s.sid, "side": s.side, "flips": s.flips,
              "side_a": round(s.side_time["A"], 1), "side_b": round(s.side_time["B"], 1),
              "ta": self.targets["A"], "tb": self.targets["B"],
              "te": self.targets["tol_early"], "tl": self.targets["tol_late"]}
        if extra:
            ev.update(extra)
        EVENTS.parent.mkdir(parents=True, exist_ok=True)
        with EVENTS.open("a") as f:
            f.write(json.dumps(ev, ensure_ascii=False) + "\n")
        if self.on_event:
            self.on_event(ev)

    def _grade(self, side: str, elapsed: float) -> str:
        t = self.targets
        if elapsed < t[side] - t["tol_early"]:
            return "early"
        if elapsed > t[side] + t["tol_late"]:
            return "late"
        return "optimal"

    # ---- helpers ------------------------------------------------------------
    @staticmethod
    def _norm(d):
        d = tuple(d)
        return (d[0], d[1], d[2],
                d[3] if len(d) > 3 else 1.0,
                d[4] if len(d) > 4 else None,
                d[5] if len(d) > 5 else None)

    def _dominant(self, hist, t0: float, t1: float):
        votes = [c for t, c in hist if t0 <= t <= t1 and c is not None]
        if len(votes) < 2:
            return None
        top = max(set(votes), key=votes.count)
        return top if votes.count(top) / len(votes) >= self.face_agree else None

    def _mean_lab(self, hist, t0: float, t1: float):
        labs = [l for t, l in hist if t0 <= t <= t1 and l]
        if not labs:
            return None
        return tuple(st.mean(v[i] for v in labs) for i in range(3))

    def _room_shift(self, t0: float, t1: float, t2: float, t3: float) -> float:
        """How much the whole griddle changed brightness between two windows.

        The camera rides its own auto-exposure: when the cook leans in, every
        patty darkens at once. Only a change beyond the room's own is the meat.
        """
        a = [l for t, l in self.lum_hist if t0 <= t <= t1]
        b = [l for t, l in self.lum_hist if t2 <= t <= t3]
        return st.median(b) - st.median(a) if a and b else 0.0

    # ---- the flip decision ---------------------------------------------------
    def _verdict(self, s: Slot, now: float):
        """An episode is over and the crop has been quiet: what happened here?

        Absence alone is never enough. The old engine had to accept it — once a
        track died under a hand there was nothing left to compare — and that is
        exactly why 33 of its 38 flips came from a disappearance and only 24% of
        them were real. Here the place survives the hand, so a flip has to show
        itself in the patty.
        """
        ep = s.episode
        s.episode = None
        bump = lambda k: self.stats.__setitem__(k, self.stats.get(k, 0) + 1)
        bump("verdicts")
        before_face, before_lab = ep["face_before"], ep["lab_before"]
        after_face = self._dominant(s.face_hist, now - self.face_window, now)
        after_lab = self._mean_lab(s.lab_hist, now - self.face_window, now)
        if after_face == 3 or before_face == 3:
            return bump("no/other-crop")            # the crop lost the patty
        if after_face == 2:
            s.cheesed = True                        # dressed, not turned
            return bump("no/cheese")
        via = None
        if before_face is not None and after_face is not None and after_face != before_face:
            via = "face"
        elif before_lab and after_lab:
            # both sides seared look the same to the classifier; the crust still
            # differs in lightness, measured against the room's own swing
            room = self._room_shift(ep["t0"] - self.face_window, ep["t0"], now - self.face_window, now)
            if (before_lab[0] - after_lab[0]) - room >= self.flip_dl:
                via = "crust"
        if via is None:
            if before_face is None:
                return bump("no/before-unknown")
            if after_face is None:
                return bump("no/after-unknown")
            return bump("no/looks-the-same")
        if s.cheesed:
            return bump("no/already-cheesed")
        elapsed = s.side_time[s.side] + (ep["t0"] - s.side_started)
        if elapsed < self.min_side:
            return bump("no/too-soon")
        if ep["t0"] - max(s.last_flip_ts, s.placed_ts) < self.flip_cooldown:
            return bump("no/cooldown")
        bump(f"flip/{via}")
        grade = self._grade(s.side, elapsed)
        s.bonus = round(self.targets[s.side] - elapsed, 1) if grade == "early" else 0.0
        s.side_time[s.side] = elapsed
        s.side = "B" if s.side == "A" else "A"
        s.side_started = now
        s.flips += 1
        s.last_flip_ts = now
        s.flip_feedback = grade
        s.lab_hist.clear()
        s.face_hist = [(t, c) for t, c in s.face_hist if t >= now - 0.5]
        self.session["flips"] += 1
        self.session[grade] += 1
        self.session["streak"] = self.session["streak"] + 1 if grade == "optimal" else 0
        self._emit("flip", s, {"grade": grade, "elapsed": round(elapsed, 1),
                               "bonus": s.bonus, "via": via,
                               "gap": round(ep.get("gap", 0.0), 1)})

    # ---- per-frame update ----------------------------------------------------
    def update(self, dets, now: float):
        self._now = now
        dets = [self._norm(d) for d in dets]
        lums = sorted(d[4][0] for d in dets if d[4])
        if lums:
            self.lum_hist.append((now, lums[len(lums) // 2]))
            self.lum_hist = [x for x in self.lum_hist if x[0] >= now - 60]

        # --- claims: every slot asks whether its patty is still in its place ---
        pairs = []
        for s in self.slots.values():
            for i, d in enumerate(dets):
                dist = math.hypot(d[0] - s.ax, d[1] - s.ay) / max(s.ar, 1e-6)
                if dist <= self.claim_frac:
                    pairs.append((dist, s.sid, i))
        pairs.sort()
        claimed, taken = {}, set()
        for dist, sid, i in pairs:
            if sid in claimed or i in taken:
                continue
            claimed[sid], _ = i, taken.add(i)

        for s in self.slots.values():
            i = claimed.get(s.sid)
            if i is None:
                if s.absent_since is None:
                    s.absent_since = now
                continue
            d = dets[i]
            s.face_hist.append((now, d[5]))
            s.lab_hist.append((now, d[4]))
            s.face_hist = [x for x in s.face_hist if x[0] >= now - 12]
            s.lab_hist = [x for x in s.lab_hist if x[0] >= now - 12]
            s.offsets.append((now, d[0] - s.ax, d[1] - s.ay))
            s.offsets = [x for x in s.offsets if x[0] >= now - 6]
            face = self._dominant(s.face_hist, now - self.face_window, now)
            if face is not None:
                s.face = face
                if face == 2:
                    s.cheesed = True
            if not s.anchored:
                s.births.append((now, d[0], d[1], d[2]))
                self._settle(s, now)
            if s.absent_since is not None:
                gap = now - s.absent_since
                s.absent_since = None
                if gap >= self.episode_min and s.episode is None:
                    self.stats["episodes"] = self.stats.get("episodes", 0) + 1
                    # the patty was out of its place: open an episode and judge it
                    # once the crop has been quiet again
                    s.episode = {"t0": s.seen_ts, "gap": gap, "due": now + self.verdict_delay,
                                 "face_before": self._dominant(s.face_hist, s.seen_ts - 6.0,
                                                               s.seen_ts + 0.01),
                                 "lab_before": self._mean_lab(s.lab_hist, s.seen_ts - 6.0,
                                                              s.seen_ts + 0.01)}
                elif s.episode is not None:
                    s.episode["due"] = now + self.verdict_delay   # still being handled
                    s.episode["gap"] = max(s.episode["gap"], gap)
            s.seen_ts = now
            self._reanchor(s, now)

        # --- verdicts that have come due --------------------------------------
        for s in list(self.slots.values()):
            if s.episode and s.absent_since is None and now >= s.episode["due"]:
                self._verdict(s, now)

        # --- detections nobody claimed: a new patty, or one that was moved -----
        for i, d in enumerate(dets):
            if i in taken or d[3] < self.birth_conf or d[5] == 3:
                continue
            cx, cy, r = d[0], d[1], d[2]
            if any(math.hypot(cx - s.ax, cy - s.ay) < self.birth_suppress * max(s.ar, r)
                   for s in self.slots.values()):
                continue
            if self._migrate(d, now):
                continue
            s = Slot(self.next_sid, cx, cy, r, placed_ts=now, side_started=now,
                     seen_ts=now, births=[(now, cx, cy, r)])
            s.face_hist.append((now, d[5]))
            s.lab_hist.append((now, d[4]))
            self.next_sid += 1
            self.slots[s.sid] = s
            self._emit("placed", s)

        # --- removals ----------------------------------------------------------
        for sid in [k for k, s in self.slots.items()
                    if s.absent_since and now - s.absent_since > self.removed_after]:
            s = self.slots.pop(sid)
            s.side_time[s.side] += s.absent_since - s.side_started
            total = s.side_time["A"] + s.side_time["B"]
            self._emit("removed", s, {"total": round(total, 1)})
            self.done.append({"pid": s.sid, "side_a": round(s.side_time["A"], 1),
                              "side_b": round(s.side_time["B"], 1),
                              "flips": s.flips, "total": round(total, 1)})

    def _settle(self, s: Slot, now: float):
        """Freeze the anchor once the patty has held still — after the smash, not
        before it: a ball of mince spreading under a spatula moves for seconds."""
        s.births = [b for b in s.births if b[0] >= now - self.settle_time * 2]
        recent = [b for b in s.births if b[0] >= now - self.settle_time]
        if len(recent) < 4 or recent[-1][0] - recent[0][0] < self.settle_time:
            return
        mx = st.median([b[1] for b in recent])
        my = st.median([b[2] for b in recent])
        mr = st.median([b[3] for b in recent])
        if any(math.dist((b[1], b[2]), (mx, my)) / max(mr, 1e-6) > self.settle_tol
               for b in recent):
            return
        s.ax, s.ay, s.ar, s.anchored = mx, my, mr, True

    def _reanchor(self, s: Slot, now: float):
        """A nudge that persists is a new place, a nudge that does not is smoke."""
        if not s.anchored:
            return
        recent = [o for o in s.offsets if o[0] >= now - 3.0]
        if len(recent) < 8:
            return
        dx = st.median([o[1] for o in recent])
        dy = st.median([o[2] for o in recent])
        if math.hypot(dx, dy) / max(s.ar, 1e-6) < self.reanchor_tol:
            return
        s.ax += dx
        s.ay += dy
        s.offsets.clear()

    def _migrate(self, d, now: float) -> bool:
        """The cook slid a patty to another spot: the place moves, the timer stays."""
        best, bd = None, 1e9
        for s in self.slots.values():
            if s.absent_since is None or now - s.absent_since > self.migrate_window:
                continue
            dist = math.hypot(d[0] - s.ax, d[1] - s.ay) / max(s.ar, 1e-6)
            if dist < bd and dist <= self.migrate_frac:
                best, bd = s, dist
        if best is None:
            return False
        gap = now - best.absent_since
        best.episode = {"t0": best.seen_ts, "gap": gap, "due": now + self.verdict_delay,
                        "face_before": self._dominant(best.face_hist, best.seen_ts - 6.0,
                                                      best.seen_ts + 0.01),
                        "lab_before": self._mean_lab(best.lab_hist, best.seen_ts - 6.0,
                                                     best.seen_ts + 0.01)}
        best.ax, best.ay, best.ar = d[0], d[1], d[2]
        best.absent_since = None
        best.seen_ts = now
        best.offsets.clear()
        best.face_hist.append((now, d[5]))
        best.lab_hist.append((now, d[4]))
        self._emit("moved", best)
        return True

    def scene_reset(self, now: float):
        for s in self.slots.values():
            s.side_time[s.side] += now - s.side_started
            self._emit("invalidated", s)
        self.slots.clear()
        self.freeze_until = now + 1.2
        self.scene_cut_ts = now

    # ---- HUD snapshot --------------------------------------------------------
    def snapshot(self, now: float) -> dict:
        t = self.targets
        out = []
        for s in self.slots.values():
            target = t[s.side] + s.bonus
            elapsed = s.elapsed(now)
            fb, s.flip_feedback = s.flip_feedback, None
            out.append({
                "pid": s.sid, "x": round(s.ax, 4), "y": round(s.ay, 4),
                "r": round(s.ar, 4), "side": s.side, "flips": s.flips,
                "elapsed": round(elapsed, 1), "target": target, "bonus": s.bonus,
                "deadline": round(now - elapsed + target, 2),
                "cheesed": s.cheesed, "face": s.face,
                "missing": s.absent_since is not None,
                "missing_for": round(now - s.absent_since, 1) if s.absent_since else 0.0,
                "feedback": fb,
            })
        return {"patties": out, "session": dict(self.session), "done": len(self.done),
                "scene_cut_ts": round(self.scene_cut_ts, 2)}
