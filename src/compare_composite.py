"""Score the composite run against the timings it was built from.

Tracks are matched to ground-truth patties by position: the scene is static, so
each slot is unambiguous. Reports per-patty and aggregate error for placement,
removal and the two side timers.

Usage: python src/compare_composite.py out/multi15_v3/result.json data/composite15_gt.json
"""
import argparse
import json

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("result")
    ap.add_argument("gt")
    ap.add_argument("--max-dist", type=float, default=180.0)
    args = ap.parse_args()

    pred = json.load(open(args.result))
    gt = json.load(open(args.gt))["patties"]

    used, rows = set(), []
    for g in gt:
        best, best_d = None, 1e18
        for p in pred:
            if p["patty"] in used:
                continue
            d = np.hypot(p["x"] - g["x"], p["y"] - g["y"])
            if d < best_d:
                best, best_d = p, d
        if best is None or best_d > args.max_dist:
            rows.append((g, None, best_d))
            continue
        used.add(best["patty"])
        rows.append((g, best, best_d))

    print(f"{'gt':>3} {'trk':>4} {'placed':>14} {'removed':>14} "
          f"{'side A':>14} {'side B':>14}")
    errs = {"placed": [], "removed": [], "A": [], "B": []}
    matched = 0
    for g, p, d in rows:
        if p is None:
            print(f"{g['id']:>3}   --   NOT DETECTED")
            continue
        matched += 1
        e_pl = p["placed_t"] - g["placed_t"]
        e_rm = p["removed_t"] - g["removed_t"]
        e_a = p["side_A_s"] - g["side_A_s"]
        e_b = p["side_B_s"] - g["side_B_s"]
        for k, v in (("placed", e_pl), ("removed", e_rm), ("A", e_a), ("B", e_b)):
            errs[k].append(v)
        print(f"{g['id']:>3} {p['patty']:>4} "
              f"{g['placed_t']:6.1f}->{p['placed_t']:6.1f} "
              f"{g['removed_t']:6.1f}->{p['removed_t']:6.1f} "
              f"{g['side_A_s']:6.1f}->{p['side_A_s']:6.1f} "
              f"{g['side_B_s']:6.1f}->{p['side_B_s']:6.1f}")

    print(f"\nmatched {matched}/{len(gt)}   extra tracks {len(pred) - matched}")
    for k, label in (("placed", "укладка"), ("removed", "снятие"),
                     ("A", "сторона A"), ("B", "сторона B")):
        v = np.array(errs[k]) if errs[k] else np.array([np.nan])
        print(f"  {label:10s} медиана |ошибки| {np.nanmedian(np.abs(v)):6.2f} s   "
              f"среднее {np.nanmean(v):+6.2f} s   макс |{np.nanmax(np.abs(v)):.2f}| s")


if __name__ == "__main__":
    main()
