"""Tune the KF branch against IDF1 on labeled ground truth.

Detections are precomputed, so every config replays at CPU speed. IDF1 comes
from py-motmetrics (the reference implementation); the distance gate is one
patty diameter in normalized units. The core engine (use_kf=False) is the
baseline to beat on the tune window AND the holdout.

Usage: python src/tune_kf_idf1.py            # grid search + report
"""
import itertools
import json
import sys
import tempfile
from pathlib import Path

import motmetrics as mm
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "app"))
import timers                                   # noqa: E402

timers.EVENTS = Path(tempfile.mkdtemp()) / "e.jsonl"
DIAM = 0.085                                    # typical patty diameter (norm.)


def run_engine(rows, **kw):
    eng = timers.TimerEngine({"A": 120, "B": 90, "tol_early": 30, "tol_late": 30}, **kw)
    hyp = []
    for row in rows:
        eng.update([tuple(d) for d in row["dets"]], row["t"])
        hyp.append((row["t"], [(p.pid, p.cx, p.cy) for p in eng.alive.values()
                               if p.missing_since is None]))
    return hyp


def idf1(gt_tracks, hyp, times):
    gt_by_t = {}
    for tr in gt_tracks:
        for (t, x, y, r) in tr["frames"]:
            gt_by_t.setdefault(round(t, 2), []).append((tr["id"], x, y))
    acc = mm.MOTAccumulator(auto_id=True)
    for (t, hs) in hyp:
        gts = gt_by_t.get(round(t, 2), [])
        gt_ids = [g[0] for g in gts]
        hp_ids = [h[0] for h in hs]
        if gts and hs:
            d = np.full((len(gts), len(hs)), np.nan)
            for i, (_, gx, gy) in enumerate(gts):
                for j, (_, hx, hy) in enumerate(hs):
                    dist = ((gx - hx) ** 2 + (gy - hy) ** 2) ** 0.5
                    if dist < DIAM:
                        d[i, j] = dist
        else:
            d = np.zeros((len(gts), len(hs)))
        acc.update(gt_ids, hp_ids, d)
    mh = mm.metrics.create()
    s = mh.compute(acc, metrics=["idf1", "mota", "num_switches",
                                 "mostly_tracked", "num_fragmentations"], name="x")
    return {k: float(s[k].iloc[0]) for k in s.columns}


def main():
    data = {}
    for w in ("tune", "holdout"):
        rows = json.loads((ROOT / f"data/mot_dets_{w}.json").read_text())
        gt = json.loads((ROOT / f"data/mot_gt_{w}.json").read_text())["tracks"]
        data[w] = (rows, gt)

    def score(name, w, **kw):
        rows, gt = data[w]
        hyp = run_engine(rows, **kw)
        m = idf1(gt, hyp, [r["t"] for r in rows])
        return m

    base_t = score("core", "tune")
    print(f"BASELINE core   tune: IDF1 {base_t['idf1']:.3f}  MOTA {base_t['mota']:.3f}  "
          f"switches {base_t['num_switches']:.0f}  frag {base_t['num_fragmentations']:.0f}")

    grid = list(itertools.product(
        [0.10, 0.18, 0.30],          # BETA
        [0.004, 0.006, 0.009],       # DEAD
        [0.10, 0.20],                # VMAX
        [1.2, 1.55],                 # birth_suppress
    ))
    results = []
    for beta, dead, vmax, bs in grid:
        timers.KF.BETA, timers.KF.DEAD, timers.KF.VMAX = beta, dead, vmax
        m = score("kf", "tune", use_kf=True, birth_suppress=bs)
        results.append(((beta, dead, vmax, bs), m))
        print(f"kf b={beta} d={dead} v={vmax} bs={bs}: IDF1 {m['idf1']:.3f} "
              f"sw {m['num_switches']:.0f} frag {m['num_fragmentations']:.0f}", flush=True)
    results.sort(key=lambda x: -x[1]["idf1"])
    best_cfg, best_m = results[0]
    print(f"\nЛУЧШИЙ КФ на tune: {best_cfg} -> IDF1 {best_m['idf1']:.3f} "
          f"(база {base_t['idf1']:.3f})")

    # holdout: базис и лучший КФ
    timers.KF.BETA, timers.KF.DEAD, timers.KF.VMAX = 0.18, 0.006, 0.20
    base_h = score("core", "holdout")
    timers.KF.BETA, timers.KF.DEAD, timers.KF.VMAX = best_cfg[0], best_cfg[1], best_cfg[2]
    kf_h = score("kf", "holdout", use_kf=True, birth_suppress=best_cfg[3])
    print(f"HOLDOUT: core IDF1 {base_h['idf1']:.3f} sw {base_h['num_switches']:.0f} | "
          f"КФ IDF1 {kf_h['idf1']:.3f} sw {kf_h['num_switches']:.0f}")
    verdict = "КФ ПОБЕЖДАЕТ" if (best_m["idf1"] > base_t["idf1"] + 0.01
                                  and kf_h["idf1"] >= base_h["idf1"] - 0.005) else "ОСТАЁМСЯ НА ЯДРЕ"
    print("ВЕРДИКТ:", verdict)


if __name__ == "__main__":
    main()
