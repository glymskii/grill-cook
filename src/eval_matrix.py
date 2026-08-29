"""CPU evaluation matrix over the pre-dumped full-video detections.

Replays engines over data/mot_dets_full.json (no GPU), reporting per-engine:
teleport timestamps (identity errors), fragments, merges — plus IDF1 on every
labeled GT window. One command answers both axes for any engine config.

Usage: python src/eval_matrix.py
"""
import json
import math
import statistics as st
import sys
import tempfile
from pathlib import Path

import motmetrics as mm
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "app"))
import timers                                   # noqa: E402

timers.EVENTS = Path(tempfile.mkdtemp()) / "e.jsonl"
timers.KF.BETA, timers.KF.DEAD, timers.KF.VMAX = 0.3, 0.009, 0.1
DIAM = 0.085
T = {"A": 120, "B": 90, "tol_early": 30, "tol_late": 30}

FULL = json.loads((ROOT / "data/mot_dets_full.json").read_text())
GTS = {}
for w in ("tune", "holdout", "trans1", "trans2"):
    f = ROOT / f"data/mot_gt_{w}.json"
    if f.exists():
        GTS[w] = json.loads(f.read_text())["tracks"]


def replay(**kw):
    eng = timers.TimerEngine(dict(T), **kw)
    prev = {}
    teleports = []
    hyp = []
    for row in FULL:
        t = row["t"]
        eng.update([tuple(d) for d in row["dets"]], t)
        cur = {p.pid: (p.cx, p.cy, p.r) for p in eng.alive.values()
               if p.missing_since is None}
        for pid, (x, y, r) in cur.items():
            if pid in prev:
                px, py, _ = prev[pid]
                if math.hypot(x - px, y - py) > 0.8 * r:
                    teleports.append(round(t, 1))
        prev = cur
        hyp.append((t, [(pid, x, y) for pid, (x, y, _) in cur.items()]))
    lives = [d["side_a"] + d["side_b"] for d in eng.done]
    lives += [p.side_time["A"] + p.side_time["B"] for p in eng.alive.values()]
    frags = sum(1 for x in lives if x < 5)
    return {"hyp": hyp, "teleports": teleports, "ids": eng.next_pid - 1,
            "frags": frags, "merges": eng.merged,
            "med_life": st.median(lives) if lives else 0}


def idf1_on(hyp, gt_tracks):
    gt_by_t = {}
    t_lo = min(t for tr in gt_tracks for (t, *_ ) in tr["frames"])
    t_hi = max(t for tr in gt_tracks for (t, *_ ) in tr["frames"])
    for tr in gt_tracks:
        for (t, x, y, r) in tr["frames"]:
            gt_by_t.setdefault(round(t, 2), []).append((tr["id"], x, y))
    acc = mm.MOTAccumulator(auto_id=True)
    for (t, hs) in hyp:
        if not (t_lo - 0.01 <= t <= t_hi + 0.01):
            continue
        gts = gt_by_t.get(round(t, 2), [])
        if gts or hs:
            d = np.full((len(gts), len(hs)), np.nan)
            for i, (_, gx, gy) in enumerate(gts):
                for j, (_, hx, hy) in enumerate(hs):
                    dd = math.hypot(gx - hx, gy - hy)
                    if dd < DIAM:
                        d[i, j] = dd
            acc.update([g[0] for g in gts], [h[0] for h in hs], d)
    mh = mm.metrics.create()
    s = mh.compute(acc, metrics=["idf1", "num_switches"], name="x")
    return float(s["idf1"].iloc[0]), int(s["num_switches"].iloc[0])


def main():
    configs = [
        ("ЯДРО (прод)", {}),
        ("КФ полный", {"use_kf": True}),
        ("ГИБРИД age>=10с", {"use_kf": "hybrid", "kf_min_age": 10.0}),
        ("ГИБРИД age>=20с", {"use_kf": "hybrid", "kf_min_age": 20.0}),
        ("ГИБРИД age>=40с", {"use_kf": "hybrid", "kf_min_age": 40.0}),
    ]
    for name, kw in configs:
        r = replay(**kw)
        line = (f"{name:18s} телепорты {len(r['teleports']):3d} "
                f"id {r['ids']:3d} обрывки {r['frags']:2d} "
                f"слияния {r['merges']:3d} жизнь {r['med_life']:.0f}с")
        for w, gt in GTS.items():
            f1, sw = idf1_on(r["hyp"], gt)
            line += f" | {w}: IDF1 {f1:.3f} sw {sw}"
        print(line, flush=True)
        if r["teleports"]:
            print(f"    моменты телепортов: {r['teleports'][:14]}")


if __name__ == "__main__":
    main()
