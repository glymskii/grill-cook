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
    ring_hist: list = field(default_factory=list)  # (t, lab of the griddle around it)
    claim_hist: list = field(default_factory=list)  # (t, was the patty found here)
    face: int | None = None
    cheesed: bool = False
    episode: dict | None = None    # open manipulation, or one awaiting its verdict
    offsets: list = field(default_factory=list)    # (t, dx, dy) of claims, for re-anchoring
    bonus: float = 0.0
    flip_feedback: str | None = None
    last_flip_ts: float = 0.0
    prev_crop: object = None       # last frame's pixels at this place
    disturb: float = 0.0           # how much they just changed
    quiet_ts: float = 0.0          # last moment nothing was happening here
    face_quiet: int = 0            # readings taken while settled and occupied
    face_quiet_other: int = 0      # ...of which called this patty 'not a patty'
    cheesed_ts: float = 0.0        # when the slice landed
    last_xy: tuple | None = None   # where the patty was last actually seen
    pre_cheese_lab: tuple | None = None   # the crop before the slice landed
    cheese_pending: tuple | None = None   # (t0, opened, before): a jump awaiting confirmation

    def elapsed(self, now: float) -> float:
        ref = self.absent_since if self.absent_since is not None else now
        return self.side_time[self.side] + (ref - self.side_started)


class SlotEngine:
    def __init__(self, targets: dict, claim_frac=0.75, settle_time=2.0,
                 settle_tol=0.25, birth_conf=0.40, birth_suppress=1.6,
                 removed_after=6.0, verdict_delay=2.5, face_window=2.5,
                 face_agree=0.6, flip_dl=18.0, flip_cooldown=20.0,
                 min_side_before_flip=25.0, reanchor_tol=0.35, migrate_frac=3.0,
                 migrate_window=4.0, other_grace=2.0, episode_min=0.6,
                 hot=0.35, topping_db=15.0, room_max=25.0, hold_frac=0.8,
                 cheese_window=4.0, occupied_dist=40.0, max_backdate=30.0,
                 history_step=0.5, shadow_time=10.0, show_after=4.0,
                 before_window=3.0, strong_dl=30.0, cheese_jump=20.0,
                 cheese_min_age=20.0, undress_after=10.0, cheese_confirm=8.0,
                 face_model=None):
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
        self.hot = hot                     # crop change that counts as a hand
        self.topping_db = topping_db       # yellow jump that means cheese, not a turn
        self.room_max = room_max           # light moved this much: do not judge
        self.hold_frac = hold_frac         # the patty must hold its place to be judged
        self.cheese_window = cheese_window # sustained 'cheese' that retires a slot
        self.occupied_dist = occupied_dist # crop this far from the ring = occupied
        self.max_backdate = max_backdate
        self.history_step = history_step
        self.shadow_time = shadow_time     # a vacated anchor keeps suppressing births
        self.show_after = show_after       # a ring is provisional until this age
        self.before_window = before_window # quiet seconds before an episode that count as 'before'
        self.strong_dl = strong_dl         # a colour step this big outranks the classifier
        self.cheese_jump = cheese_jump     # brighter by this much across an episode = a slice
        self.cheese_min_age = cheese_min_age   # nobody dresses a patty this young
        self.undress_after = undress_after     # the place looked pre-cheese this long: flag off
        self.cheese_confirm = cheese_confirm   # seconds the light must hold before a slice is believed
        # optional callable: list of BGR crops -> list of class ids. Given one,
        # the face is read at the anchor instead of at the detection box.
        self.face_model = face_model
        self.history: list = []            # (t, small Lab frame) for back-dating
        # Ticket numbers the cook sees. Assigned after back-dating and ordered by
        # when the meat landed, not by when we recognised it — the two patties put
        # down first were being shown as #2 and #3 because a ball of mince takes
        # seconds to become a patty to the detector.
        self.order: list = []              # (placed_ts, sid), sorted
        self.shadows: list = []            # (t, x, y, r) places an anchor just left
        self.plate_hist: list = []         # (t, median ring L over the plate)
        self.slots: dict[int, Slot] = {}
        self.next_sid = 1
        self.done: list[dict] = []
        self.session = {"flips": 0, "optimal": 0, "early": 0, "late": 0, "streak": 0}
        self.on_event = None
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

    def _backdate(self, s: Slot, now: float):
        """Start the clock when the meat landed, not when we recognised it.

        A ball of mince is not a patty to the detector — it is pale, lumpy and
        half under the cook's hand — so a slot is often born many seconds after
        the meat hits the plate. Measured on the smash session: two patties were
        on the griddle from 8 s and 13 s but their slots opened at 21.4 s and
        22.6 s, so their cook time was short by 9-13 seconds. Once the place is
        known, its own past answers the question: walk back while the crop is
        still unlike the bare griddle around it, and stop where it is not.
        """
        import numpy as np
        first = s.placed_ts
        for t, sm in reversed(self.history):
            if t >= s.placed_ts or s.placed_ts - t > self.max_backdate:
                continue
            sh, sw = sm.shape[:2]
            R, cx, cy = s.ar * sw, s.ax * sw, s.ay * sh
            yy, xx = np.ogrid[0:sh, 0:sw]
            d = np.hypot(xx - cx, yy - cy)
            inner, ring = sm[d <= 0.6 * R], sm[(d >= 1.15 * R) & (d <= 1.9 * R)]
            if len(inner) < 20 or len(ring) < 40:
                break
            if math.dist(inner.mean(0), np.median(ring, 0)) < self.occupied_dist:
                break                      # here the place was still bare plate
            first = t
        if first < s.placed_ts:
            s.placed_ts = s.side_started = first

    def _lab_before(self, hist, t1: float):
        """The crop's colour before an episode: the quiet seconds just before it,
        and when a busy minute left none there, the last quiet seconds it has at
        all (the history keeps 12 s). Only quiet frames ever enter the history, so
        this is never a hand - but it must not be empty either: ten verdicts on
        real flips died as 'before-unknown' for want of three quiet seconds."""
        recent = [l for t, l in hist if t <= t1][-10:]          # ~2 quiet seconds at 5 fps
        if len(recent) < 3:
            return None
        return tuple(st.median(v[i] for v in recent) for i in range(3))

    def _swap(self, s: Slot, now: float):
        """The patty in this place was replaced: close its cook, start a new one."""
        s.side_time[s.side] += now - s.side_started
        total = s.side_time["A"] + s.side_time["B"]
        self._emit("removed", s, {"total": round(total, 1), "swap": True})
        self.done.append({"pid": s.sid, "side_a": round(s.side_time["A"], 1),
                          "side_b": round(s.side_time["B"], 1), "flips": s.flips,
                          "total": round(total, 1)})
        s.side, s.side_started, s.side_time = "A", now, {"A": 0.0, "B": 0.0}
        s.flips, s.bonus, s.last_flip_ts = 0, 0.0, 0.0
        s.cheesed, s.cheesed_ts, s.pre_cheese_lab = False, 0.0, None
        s.placed_ts = now
        self.order = sorted([(t, sid) for t, sid in self.order if sid != s.sid] + [(now, s.sid)])
        self._emit("placed", s, {"swap": True})

    def _mean_plate(self, t0: float, t1: float):
        v = [L for t, L in self.plate_hist if t0 <= t <= t1]
        return st.mean(v) if v else None

    def _trust_face(self, s: Slot) -> bool:
        """Does the face model make sense at this place at all?"""
        if s.face_quiet < 40:
            return True                    # not enough evidence to distrust it
        return s.face_quiet_other / s.face_quiet < 0.5

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
        # The colour of the crop only means something if the crop is the patty.
        # Reviewed by eye, the false verdicts that survived everything else were
        # all the same mistake: at the moment of judging, the place was under a
        # forearm or a sausage. One frame of presence is not enough — the patty
        # has to be back and stay back for the whole window we measure.
        held = [c for t, c in s.claim_hist if t >= now - self.face_window]
        if held and sum(held) / len(held) < self.hold_frac:
            return bump("no/not-back-yet")
        # Colour first, the classifier second. On the smash session the face
        # classifier calls pale raw mince "cheese" for 42% of detections — it was
        # trained on the other kitchen — and that veto silently swallowed the
        # whole batch flip. The colour of the pinned crop has no such opinion:
        # cheese moves b* by 29-41 points, a turn by under 6.
        via, dL = None, None
        if before_lab and after_lab:
            # The griddle ring is a good witness and a bad accountant. Subtracting
            # it corrected nothing and swung the answer by up to 25 Lab points in
            # both directions — killing a real flip on one slot and inventing one
            # on another. Used as a veto it is solid: when the light at this place
            # really moved, refuse to judge instead of pretending to compensate.
            plate_b = self._mean_plate(ep["t0"] - self.before_window, ep["t0"] + 0.01)
            plate_a = self._mean_plate(now - self.face_window, now)
            if plate_b is not None and plate_a is not None and abs(plate_b - plate_a) > self.room_max:
                return bump("no/light-moved")
            db = before_lab[2] - after_lab[2]
            dL = before_lab[0] - after_lab[0]
            # Cheese is an event, not a state. Measured across every cheese
            # placement on both sessions the crop gets BRIGHTER by 20-110 L (white
            # cheese) or yellower by 15-40 b* (American); a chicken patty turning
            # golden goes the other way, -14..-50 L. Reading "cheese" off the
            # classifier at rest is what dressed a raw chicken patty for ten
            # minutes: white on white. Where the classifier is trusted it is the
            # second witness; a slot younger than 30 s at rest cannot be dressed.
            # The classifier cannot be the second witness here: at the moment a
            # white slice lands it says "not a patty" or "cooked" (15@424 was
            # refused at +108 L). Brightness alone decides above the jump; the
            # classifier only breaks ties for a small jump, where a slice on an
            # already-golden patty adds little light.
            jump = max(-dL, 0.0)
            brighter = jump >= self.cheese_jump
            yellower = -db >= self.topping_db
            age_ok = ep["t0"] - s.placed_ts >= self.cheese_min_age
            if (brighter or yellower) and age_ok and not s.cheesed:
                # A jump opens a candidacy, it does not dress the patty: a spatula
                # resting over the crop, a paler patty carried across it and a
                # sausage all jump the same way and are gone in seconds. The slice
                # stays - so the light must still be there in 8 s, and where the
                # classifier is trusted it must by then read cheese at rest.
                s.cheese_pending = (ep["t0"], now, before_lab)
                return bump("no/cheese-pending")
            # A dressed patty is finished; if its place went dark by this much
            # the dressed patty left and whatever sits here now is a new one.
            # Read as a flip, that was three of the four false flips on the shift.
            if s.cheesed and dL >= self.cheese_jump:
                self._swap(s, now)
                return bump("no/swap")
            if dL >= self.flip_dl:
                via = "crust"
        # The classifier is a witness, not a judge: "not a patty" at the anchor
        # refuses a verdict only when the colour said nothing. It refused a +62
        # flip on 19@477 once, with a golden crust in plain view.
        if (after_face == 3 or before_face == 3) and self._trust_face(s) \
                and (dL is None or dL < self.strong_dl):
            return bump("no/other-crop")            # the crop lost the patty
        if via is None and before_face is not None and after_face is not None \
                and after_face != before_face and 2 not in (before_face, after_face):
            via = "face"
        if via is None:
            if before_face is None:
                return bump("no/before-unknown")
            if after_face is None:
                return bump("no/after-unknown")
            return bump("no/looks-the-same")
        if s.cheesed:
            return bump("no/already-cheesed")
        elapsed = s.side_time[s.side] + (ep["t0"] - s.side_started)
        # Both guards scale with the standard rather than sitting at a constant:
        # 45 s of cooldown was tuned on 150-270 s sides and silently forbade
        # every flip on the smash session, where a side lasts 35-42 s.
        shortest = min(self.targets["A"], self.targets["B"])
        if elapsed < min(self.min_side, 0.5 * shortest):
            return bump("no/too-soon")
        if ep["t0"] - max(s.last_flip_ts, s.placed_ts) < min(self.flip_cooldown,
                                                             0.6 * shortest):
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

    # ---- the pinned crop -----------------------------------------------------
    def _sense(self, frame, now: float, dets):
        """Watch each anchored place directly, in pixels.

        Absence cannot be the only trigger: a smash patty is turned where it lies
        and the detector often never loses it, so the whole batch flip at 57-71 s
        produced no episode at all. The crop does not miss it — a hand or a
        spatula over a fixed place is a large, unmistakable change against a very
        quiet baseline (measured: p50 0.02-0.045 against peaks of 0.4-4.9).
        """
        import cv2                                   # only needed with a frame
        import numpy as np
        fh, fw = frame.shape[:2]
        crops = []                                   # for the face model, if any
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        # The ring is a median over an annulus; at 1080p that is a quarter of a
        # million pixels per slot per frame and the whole pass grinds to a halt.
        # A 640-wide copy answers the same question for 1/9th of the work, and
        # the edge worker has to hold 25 fps with this in the loop.
        sm = cv2.resize(lab, (640, max(1, int(640 * fh / fw))))
        # a thin, slow trail of the plate, kept only so a new slot can ask what
        # its own place looked like before the meat arrived
        if not self.history or now - self.history[-1][0] >= self.history_step:
            self.history.append((now, cv2.resize(lab, (320, max(1, int(320 * fh / fw))))))
            self.history = [h for h in self.history if now - h[0] <= self.max_backdate]
        plate_rings = []
        for s in self.slots.values():
            if not s.anchored:
                continue
            R = max(6, int(s.ar * fw))
            cx, cy = int(s.ax * fw), int(s.ay * fh)
            c = frame[max(0, cy - R):cy + R, max(0, cx - R):cx + R]
            if c.size == 0:
                continue
            c = cv2.resize(c, (64, 64)).astype(np.float32)
            if s.prev_crop is not None:
                s.disturb = float(np.abs(c - s.prev_crop).mean() /
                                  max(s.prev_crop.mean(), 1.0))
            s.prev_crop = c
            r6 = max(4, int(s.ar * fw * 0.6))
            p = lab[max(0, cy - r6):cy + r6, max(0, cx - r6):cx + r6]
            if p.size:
                m = p.reshape(-1, 3).mean(axis=0)
                if s.disturb < self.hot:      # a hand in the crop is not its colour
                    s.lab_hist.append((now, (float(m[0]), float(m[1]), float(m[2]))))
                s.lab_hist = [x for x in s.lab_hist if x[0] >= now - 12]
            ring = self._ring(sm, s, dets, sm.shape[1], sm.shape[0])
            if ring is not None:
                s.ring_hist.append((now, ring))
                s.ring_hist = [x for x in s.ring_hist if x[0] >= now - 12]
                plate_rings.append(ring[0])
            if self.face_model is not None:
                R2 = max(6, int(s.ar * fw * 1.15))
                c2 = frame[max(0, cy - R2):cy + R2, max(0, cx - R2):cx + R2]
                if c2.size:
                    crops.append((s, c2))
        for s in self.slots.values():
            if s.anchored and s.cheese_pending and s.disturb < self.hot:
                t0, opened, before = s.cheese_pending
                if now - opened >= self.cheese_confirm:
                    recent = self._mean_lab(s.lab_hist, now - 4.0, now)
                    held = recent is not None and (recent[0] - before[0] >= self.cheese_jump or
                                                   before[2] - recent[2] >= self.topping_db)
                    face = self._dominant(s.face_hist, now - 6.0, now)
                    face_ok = (not self._trust_face(s)) or face == 2
                    if held and face_ok:
                        s.cheesed, s.cheesed_ts, s.pre_cheese_lab = True, t0, before
                        self._emit("dressed", s)
                        self.stats["cheese/confirmed"] = self.stats.get("cheese/confirmed", 0) + 1
                    elif now - opened >= self.cheese_confirm + 6.0 or not held:
                        self.stats["cheese/cancelled"] = self.stats.get("cheese/cancelled", 0) + 1
                        s.cheese_pending = None
                    if s.cheesed:
                        s.cheese_pending = None
            if not (s.anchored and s.cheesed and s.pre_cheese_lab and s.disturb < self.hot):
                continue
            if True:
                recent = self._mean_lab(s.lab_hist, now - self.undress_after, now)
                quiet = [t for t, _ in s.lab_hist if t >= now - self.undress_after]
                if recent and len(quiet) >= 8 and \
                        abs(recent[0] - s.pre_cheese_lab[0]) <= 10 and \
                        abs(recent[2] - s.pre_cheese_lab[2]) <= 8:
                    s.cheesed, s.cheesed_ts, s.pre_cheese_lab = False, 0.0, None
                    self._emit("undressed", s)
        if plate_rings:
            # Light moves the whole plate at once; a neighbour being flipped
            # moves one ring. The verdict asks the plate, not the ring.
            self.plate_hist.append((now, float(st.median(plate_rings))))
            self.plate_hist = [x for x in self.plate_hist if x[0] >= now - 12]
        if crops:
            # Read the face where the engine actually reasons — at the anchor.
            # Classifying the detection box instead was hiding the retrained
            # negatives from the verdict: with a sausage lying across a patty the
            # box still reads "raw", while the same model on the pinned crop says
            # "not a patty" five frames out of five.
            for (s, _), cls in zip(crops, self.face_model([c for _, c in crops])):
                s.face_hist.append((now, int(cls)))
                s.face_hist = [x for x in s.face_hist if x[0] >= now - 12]
                # A place that is occupied and quiet holds a patty, by
                # construction. If the model calls that "not a patty" it is out
                # of its training distribution here — measured on the smash
                # session, where it says "other" for 85-100% of settled patties
                # and with the same 0.95+ confidence it uses on a real sausage,
                # so no threshold separates them. Rather than veto every verdict
                # on that station, the slot stops trusting the reading.
                if s.absent_since is None and s.disturb < self.hot and not s.episode:
                    s.face_quiet += 1
                    s.face_quiet_other += int(cls == 3)

    @staticmethod
    def _ring(lab, s, dets, fw, fh):
        """Bare griddle immediately around the slot, cleared of everything the
        detector can see. This is the honest exposure reference: it does not
        change when the patty is turned, so subtracting it removes the camera's
        auto-exposure swing without removing the flip along with it."""
        import numpy as np
        R = s.ar * fw
        cx, cy = s.ax * fw, s.ay * fh
        x0, x1 = int(max(0, cx - 2.1 * R)), int(min(fw, cx + 2.1 * R))
        y0, y1 = int(max(0, cy - 2.1 * R)), int(min(fh, cy + 2.1 * R))
        win = lab[y0:y1, x0:x1]
        if win.size == 0:
            return None
        yy, xx = np.ogrid[y0:y1, x0:x1]
        d = np.hypot(xx - cx, yy - cy)
        mask = (d >= 1.15 * R) & (d <= 1.9 * R)
        for dx, dy, dr in dets:
            mask &= np.hypot(xx - dx * fw, yy - dy * fh) > 1.05 * dr * fw
        px = win[mask]
        if len(px) < 40:
            return None
        m = np.median(px, axis=0)
        return (float(m[0]), float(m[1]), float(m[2]))

    # ---- per-frame update ----------------------------------------------------
    def update(self, dets, now: float, frame=None):
        self._now = now
        dets = [self._norm(d) for d in dets]

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
            s.claim_hist.append((now, i is not None))
            s.claim_hist = [x for x in s.claim_hist if x[0] >= now - 12]
            if i is None:
                if s.absent_since is None:
                    s.absent_since = now
                continue
            d = dets[i]
            if frame is None or self.face_model is None:
                s.face_hist.append((now, d[5]))
            if frame is None:                      # no pixels: fall back to the box
                s.lab_hist.append((now, d[4]))
                s.lab_hist = [x for x in s.lab_hist if x[0] >= now - 12]
            s.face_hist = [x for x in s.face_hist if x[0] >= now - 12]
            s.last_xy = (d[0], d[1])
            s.offsets.append((now, d[0] - s.ax, d[1] - s.ay))
            s.offsets = [x for x in s.offsets if x[0] >= now - 6]
            face = self._dominant(s.face_hist, now - self.face_window, now)
            if face is not None:
                s.face = face
                if face == 2 and frame is None:
                    s.cheesed = True       # with pixels, colour decides this
            if not s.anchored:
                s.births.append((now, d[0], d[1], d[2]))
                was = s.anchored
                self._settle(s, now)
                if s.anchored and not was:
                    if self.history:
                        self._backdate(s, now)
                    self.order.append((s.placed_ts, s.sid))
                    self.order.sort()
            s.absent_since = None
            s.seen_ts = now
            self._reanchor(s, now)

        if frame is not None:
            self._sense(frame, now, [(d[0], d[1], d[2]) for d in dets])

        # --- episodes: someone is working this place --------------------------
        for s in self.slots.values():
            if not s.anchored:
                continue
            busy = s.absent_since is not None or s.disturb >= self.hot
            if busy:
                if s.episode is None:
                    self.stats["episodes"] = self.stats.get("episodes", 0) + 1
                    q1 = (s.quiet_ts or now) + 0.01
                    s.episode = {"t0": s.quiet_ts or s.seen_ts, "gap": 0.0,
                                 "face_before": self._dominant(s.face_hist, q1 - 6.0, q1),
                                 "lab_before": self._lab_before(s.lab_hist, q1)}
                if s.absent_since is not None:
                    s.episode["gap"] = max(s.episode["gap"], now - s.absent_since)
                s.episode["due"] = now + self.verdict_delay
            else:
                s.quiet_ts = s.quiet_ts if s.episode else now

        # --- verdicts that have come due --------------------------------------
        for s in list(self.slots.values()):
            if s.episode and s.absent_since is None and now >= s.episode["due"]:
                self._verdict(s, now)
                s.quiet_ts = now

        # --- detections nobody claimed: a new patty, or one that was moved -----
        for i, d in enumerate(dets):
            if i in taken or d[3] < self.birth_conf:
                continue
            if d[5] == 3:
                # a glove, a spatula, the grate edge: never a birth. The mince
                # ball this once let through is recovered by back-dating from
                # the pixels instead, which needs no slot to exist early
                continue
            cx, cy, r = d[0], d[1], d[2]
            # One case may continue a slot before the solid-body rule gets a say:
            # a patty just PRESSED - the disc grew by a third or more where a slot
            # went missing a moment ago. Anything else migrating first turned out
            # to chase the spatula: a lifted patty's box walked five slots off
            # their places in thirty seconds and the verdicts were judged off-site.
            if self._migrate(d, now, grown_only=True):
                continue
            # Patties are solid. A new disc may not overlap an existing slot -
            # measured against the anchor AND the last place the patty was
            # actually seen, because a patty nudged during a flip sits between
            # the two for a while. The false slot #10 was born exactly there:
            # 1.05 r from a neighbour whose anchor had just moved 0.02 away.
            if any(self._overlaps(cx, cy, r, s) for s in self.slots.values()):
                continue
            # ...and not on a place an anchor left in the last few seconds
            self.shadows = [sh for sh in self.shadows if now - sh[0] <= self.shadow_time]
            if any(math.hypot(cx - sx, cy - sy) < self.birth_suppress * max(sr, r)
                   for _, sx, sy, sr in self.shadows):
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
            self.shadows.append((now, s.ax, s.ay, s.ar))
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

    def _overlaps(self, cx, cy, r, s: Slot) -> bool:
        lim = self.birth_suppress * max(s.ar, r)
        if math.hypot(cx - s.ax, cy - s.ay) < lim:
            return True
        if s.last_xy is not None and math.hypot(cx - s.last_xy[0], cy - s.last_xy[1]) < lim:
            return True
        return False

    def _migrate(self, d, now: float, grown_only: bool = False) -> bool:
        """The cook slid a patty to another spot: the place moves, the timer stays."""
        best, bd = None, 1e9
        for s in self.slots.values():
            if s.absent_since is None or now - s.absent_since > self.migrate_window:
                continue
            if grown_only and d[2] < 1.3 * s.ar:
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
        self.shadows.append((now, best.ax, best.ay, best.ar))
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
            nos = {sid: i + 1 for i, (_, sid) in enumerate(self.order)}
            out.append({
                "pid": s.sid, "no": nos.get(s.sid, s.sid),
                "x": round(s.ax, 4), "y": round(s.ay, 4),
                "r": round(s.ar, 4), "side": s.side, "flips": s.flips,
                "elapsed": round(elapsed, 1), "target": target, "bonus": s.bonus,
                "deadline": round(now - elapsed + target, 2),
                "cheesed": s.cheesed, "face": s.face,
                "provisional": (now - s.placed_ts) < self.show_after,
                "side_a": round(s.side_time["A"] + (elapsed - s.side_time[s.side]
                                                    if s.side == "A" else 0), 1),
                "side_b": round(s.side_time["B"] + (elapsed - s.side_time[s.side]
                                                    if s.side == "B" else 0), 1),
                "total": round(now - s.placed_ts, 1),
                "cheese_for": round(now - s.cheesed_ts, 1) if s.cheesed_ts else 0.0,
                "missing": s.absent_since is not None,
                "missing_for": round(now - s.absent_since, 1) if s.absent_since else 0.0,
                "feedback": fb,
            })
        return {"patties": out, "session": dict(self.session), "done": len(self.done),
                "scene_cut_ts": round(self.scene_cut_ts, 2)}
