"""Score the slot engine against the production one, on the same evidence.

Three questions, all answered from the manual ground truth we already paid for:
  1. flips  - precision/recall against data/flip_verdicts.json (64 candidates
              reviewed frame by frame, 6 of them real)
  2. churn  - how far the ring the cook stares at travels per minute; this is the
              complaint that started the rework
  3. shape  - rings vs detections, how many ids the session burns

Usage: python src/eval_slots.py            # both videos, both engines
"""
import json
import math
import statistics as st
import sys
import tempfile
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "app"))
sys.path.insert(0, str(ROOT / "src"))
import timers                                          # noqa: E402
import slots                                           # noqa: E402

TOL = 6.0          # s: how close a produced flip must be to a reviewed moment


def load_gt():
    """Reviewed moments, each with the place it happened.

    Time alone cannot match a flip to a review: on a loaded griddle several
    patties are handled within the same few seconds, so the nearest candidate in
    time is often a different patty. The review carries xy — use it.
    """
    cand = json.loads((ROOT / "data/flip_candidates.json").read_text())
    v = json.loads((ROOT / "data/flip_verdicts.json").read_text())
    by_id = {c["id"]: c for c in cand}
    pick = lambda ids: [(by_id[i]["t"], by_id[i]["xy"]) for i in ids if i in by_id]
    real, false, unclear = pick(v["real"]), pick(v["false"]), pick(v["unclear"])
    # reviewed later, straight from the v7 audit: no candidate record, so they
    # match on time alone (r=None means "do not check the place")
    false += [(float(t), None) for t in v.get("_v7_extra_false", {})]
    return real, false, unclear


def _near(flip, cand) -> float:
    """Distance in (seconds, radii) collapsed to a rank; 1e9 = not the same event."""
    (t, x, y), (ct, cxy) = flip, cand
    if abs(ct - t) > TOL:
        return 1e9
    if cxy is None or x is None:
        return abs(ct - t)
    d = math.hypot(x - cxy[0], y - cxy[1]) / max(cxy[2], 1e-6)
    return 1e9 if d > 1.5 else abs(ct - t) + d


def score_flips(flips, gt):
    """flips: [(t, x, y)]"""
    real, false, unclear = gt
    hit, fp, unjudged = set(), 0, []
    for f in flips:
        dr = [(_near(f, c), i) for i, c in enumerate(real)]
        df = [(_near(f, c), i) for i, c in enumerate(false)]
        du = [(_near(f, c), i) for i, c in enumerate(unclear)]
        br = min(dr) if dr else (1e9, -1)
        bf = min(df) if df else (1e9, -1)
        bu = min(du) if du else (1e9, -1)
        best = min(br[0], bf[0], bu[0])
        if best >= 1e9:
            unjudged.append(round(f[0], 1))
        elif br[0] == best:
            if br[1] in hit:
                fp += 1                     # the same turn reported twice
            else:
                hit.add(br[1])
        elif bf[0] == best:
            fp += 1
        # else: reviewed and found unjudgeable — out of both counts
    tp = len(hit)
    judged = tp + fp
    prec = tp / judged if judged else 0.0
    rec = tp / len(real) if real else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return {"flips": len(flips), "tp": tp, "fp": fp, "precision": round(prec, 3),
            "recall": f"{tp}/{len(real)}", "f1": round(f1, 3), "unjudged": unjudged}


def replay(engine, rows, targets, video=None):
    """Run a dump through an engine, collecting what the cook would have seen.

    The slot engine is given frames because in production it has them: the same
    worker that runs the detector holds the picture. The production engine is
    run exactly as it ships, on detections alone.
    """
    ev = []

    def record(e):
        # stamp the place: the engine emits identity, the scorer needs geometry
        if e["type"] == "flip":
            live = getattr(engine, "alive", None) or getattr(engine, "slots", {})
            p = live.get(e["pid"])
            e["x"] = getattr(p, "cx", None) if p else None
            e["y"] = getattr(p, "cy", None) if p else None
            if p is not None and e["x"] is None:
                e["x"], e["y"] = p.ax, p.ay
        ev.append(e)

    engine.on_event = record
    paths, rings, dets_n = {}, [], []
    cap = cv2.VideoCapture(video) if video else None
    src_fps, src_i = (cap.get(cv2.CAP_PROP_FPS) if cap else 0), 0
    for r in rows:
        frame = None
        if cap:
            want = int(r["t"] * src_fps)
            while src_i < want:
                cap.grab()
                src_i += 1
            ok, frame = cap.read()
            src_i += 1
            if not ok:
                frame = None
        engine.update([tuple(d) for d in r["dets"]], r["t"], *([frame] if cap else []))
        snap = engine.snapshot(r["t"])
        vis = [p for p in snap["patties"] if p.get("missing_for", 0) <= 0.4]
        rings.append(len(vis))
        dets_n.append(len(r["dets"]))
        for p in vis:
            paths.setdefault(p["pid"], []).append((r["t"], p["x"], p["y"], p["r"]))
    if cap:
        cap.release()
    return ev, paths, rings, dets_n


def churn(paths):
    """Distance the drawn ring walks, per minute of the patty's life."""
    per_min, jumps = [], []
    for pid, pts in paths.items():
        if len(pts) < 25 or pts[-1][0] - pts[0][0] < 10:
            continue
        r = st.median([p[3] for p in pts]) or 1e-6
        walk = sum(math.dist(a[1:3], b[1:3]) for a, b in zip(pts, pts[1:])) / r
        life = pts[-1][0] - pts[0][0]
        per_min.append(walk / life * 60)
        jumps.append(sum(1 for a, b in zip(pts, pts[1:])
                         if math.dist(a[1:3], b[1:3]) / r > 0.1) / life * 60)
    return per_min, jumps


def run(name, dump, targets, gt=None, video=None):
    rows = json.loads((ROOT / dump).read_text())
    out = {}
    for label, eng in (("production", timers.TimerEngine(targets)),
                       ("slots", slots.SlotEngine(targets))):
        mod = timers if label == "production" else slots
        mod.EVENTS = Path(tempfile.mkdtemp()) / "e.jsonl"
        ev, paths, rings, dets_n = replay(
            eng, rows, targets, str(ROOT / video) if video and label == "slots" else None)
        flips = [(e["ts_video"], e.get("x"), e.get("y")) for e in ev if e["type"] == "flip"]
        walk, jumps = churn(paths)
        out[label] = {
            "ids": sum(1 for e in ev if e["type"] == "placed"),
            "rings_med": st.median(rings), "dets_med": st.median(dets_n),
            "extra_ring_frames": round(100 * sum(1 for a, b in zip(rings, dets_n)
                                                 if a > b) / len(rings)),
            "walk_r_per_min": round(st.median(walk), 2) if walk else None,
            "jumps_per_min": round(st.median(jumps), 1) if jumps else None,
            "n_flips": len(flips),
        }
        out[label]["stats"] = dict(getattr(eng, "stats", {}))
        if gt:
            out[label]["flip_score"] = score_flips(flips, gt)
        else:
            out[label]["flip_times"] = [round(t, 1) for t, _, _ in flips]
    print(f"\n===== {name}")
    for label, d in out.items():
        print(f"  {label:11s} ids {d['ids']:3d}  rings {d['rings_med']:.0f} "
              f"(dets {d['dets_med']:.0f}, extra-ring frames {d['extra_ring_frames']}%)  "
              f"ring walk {d['walk_r_per_min']} r/min, jumps {d['jumps_per_min']}/min")
        if "flip_score" in d:
            f = d["flip_score"]
            print(f"              flips {f['flips']:3d}  precision {f['precision']:.2f}  "
                  f"recall {f['recall']}  F1 {f['f1']:.2f}  "
                  f"(fp {f['fp']}, unjudged {f['unjudged']})")
        else:
            print(f"              flips {d['n_flips']}: {d['flip_times']}")
        if d["stats"]:
            print(f"              {d['stats']}")
    return out


if __name__ == "__main__":
    gt = load_gt()
    run("long shift 16.8 min (manual flip ground truth)",
        "data/mot_dets_full_faces.json",
        {"A": 270, "B": 150, "tol_early": 15, "tol_late": 20}, gt,
        video="data/IMG_6635.mov")
    run("smash 2.8 min", "data/mot_dets_smash_faces.json",
        {"A": 42, "B": 35, "tol_early": 12, "tol_late": 12},
        video="data/IMG_6637.mov")
