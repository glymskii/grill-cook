"""Frame-by-frame audit of a dumped session: phantoms, flips, gaps.

Aggregate metrics (teleports, IDF1) missed two real defects in the smash demo,
so every demo now gets this pass: how many rings the cook would actually see
versus how many patties are there, and when the engine thinks a flip happened
so the timestamps can be checked against the footage.

Usage: python src/audit_video.py data/mot_dets_full.json
"""
import json
import statistics as st
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "app"))
import timers                                          # noqa: E402

DRAW_LAG = 0.4


def main():
    dets_path = sys.argv[1]
    ta, tb = (float(x) for x in (sys.argv[2:4] or [120, 90]))
    rows = json.loads(Path(dets_path).read_text())
    timers.EVENTS = Path(tempfile.mkdtemp()) / "e.jsonl"
    eng = timers.TimerEngine({"A": ta, "B": tb, "tol_early": 30, "tol_late": 30})
    evs = []
    eng.on_event = evs.append

    excess, drawn_n, det_n = [], [], []
    for row in rows:
        eng.update([tuple(d) for d in row["dets"]], row["t"])
        snap = eng.snapshot(row["t"])
        drawn = [q for q in snap["patties"] if q.get("missing_for", 0) <= DRAW_LAG]
        excess.append(len(drawn) - len(row["dets"]))
        drawn_n.append(len(drawn))
        det_n.append(len(row["dets"]))

    dur = rows[-1]["t"]
    print(f"=== {dets_path}: {len(rows)} кадров, {dur/60:.1f} мин")
    print(f"детекций/кадр: медиана {st.median(det_n):.0f}, пик {max(det_n)}")
    print(f"колец на экране: медиана {st.median(drawn_n):.0f}, пик {max(drawn_n)}")
    pos = [e for e in excess if e > 0]
    print(f"лишних колец: среднее {st.mean(excess):+.2f}, максимум {max(excess)}, "
          f"кадров с лишними {len(pos)}/{len(excess)} ({100*len(pos)/len(excess):.0f}%)")

    flips = [e for e in evs if e["type"] == "flip"]
    by_src = {}
    for f in flips:
        by_src.setdefault(f.get("via", "?"), []).append(round(f["ts_video"]))
    print(f"\nпереворотов: {len(flips)}  " +
          " ".join(f"{k}={len(v)}" for k, v in by_src.items()))
    for k, v in by_src.items():
        print(f"  {k}: {sorted(v)}")
    sides = [round(f["elapsed"]) for f in flips if f.get("elapsed")]
    if sides:
        print(f"  длительность стороны до переворота: медиана {st.median(sides):.0f}с, "
              f"диапазон {min(sides)}–{max(sides)}с")
    other = {}
    for e in evs:
        if e["type"] != "flip":
            other[e["type"]] = other.get(e["type"], 0) + 1
    print("прочие события:", other)


if __name__ == "__main__":
    main()
