"""Manual ground truth for flips: generate candidates, render review strips.

Production settings are deliberately conservative, so judging them against
themselves would only measure self-consistency. Candidates come from a much
more sensitive pass (both signals, low thresholds) — every real flip the colour
or disappearance channel can see at all lands in the pool, and the verdicts in
data/flip_verdicts.json turn it into precision and recall.

Usage: python src/flip_gt.py candidates      # build pool + review sheets
       python src/flip_gt.py score           # after filling verdicts
"""
import json
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "app"))
import timers                                              # noqa: E402

DETS = ROOT / "data/mot_dets_full.json"
VIDEO = ROOT / "data/IMG_6635.mov"
POOL = ROOT / "data/flip_candidates.json"
VERDICTS = ROOT / "data/flip_verdicts.json"
T = {"A": 120, "B": 90, "tol_early": 30, "tol_late": 30}


def run(**kw):
    """Replay the session, stamping each flip with where that patty actually was."""
    timers.EVENTS = Path(tempfile.mkdtemp()) / "e.jsonl"
    eng = timers.TimerEngine(dict(T), **kw)
    out, pending = [], []
    eng.on_event = lambda ev: pending.append(ev) if ev["type"] == "flip" else None
    last_pos = {}
    for r in json.loads(DETS.read_text()):
        eng.update([tuple(d) for d in r["dets"]], r["t"])
        for q in eng.alive.values():
            last_pos[q.pid] = (q.cx, q.cy, q.r)
        while pending:
            ev = pending.pop(0)
            ev["xy"] = last_pos.get(ev["pid"], (0.5, 0.5, 0.05))
            out.append(ev)
    return out


def candidates():
    prod = run()
    sens = run(flip_dl=11.0, crust_confirm=2.5, min_side_before_flip=15.0,
               flip_cooldown=15.0, flip_gap_min=0.8)
    pool = []
    for src, evs in (("prod", prod), ("sens", sens)):
        for e in evs:
            t = round(e["ts_video"], 1)
            dup = next((c for c in pool if abs(c["t"] - t) < 5), None)
            if dup:
                dup["sources"] = sorted(set(dup["sources"] + [src]))
                continue
            pool.append({"id": len(pool) + 1, "t": t, "pid": e["pid"],
                         "via": e.get("via"), "sources": [src], "xy": e["xy"]})
    pool.sort(key=lambda c: c["t"])
    for i, c in enumerate(pool):
        c["id"] = i + 1
    POOL.write_text(json.dumps(pool, indent=1))
    print(f"кандидатов: {len(pool)} "
          f"(только прод: {sum(1 for c in pool if c['sources']==['prod'])}, "
          f"только чувствительный: {sum(1 for c in pool if c['sources']==['sens'])}, "
          f"оба: {sum(1 for c in pool if len(c['sources'])==2)})")

    cap = cv2.VideoCapture(str(VIDEO))
    fps = cap.get(cv2.CAP_PROP_FPS)
    outdir = ROOT / "out" / "flip_review"
    outdir.mkdir(parents=True, exist_ok=True)

    def crop(t, x, y, r, tag):
        cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, int(t * fps)))
        ok, fr = cap.read()
        if not ok:
            return np.zeros((150, 150, 3), np.uint8)
        fh, fw = fr.shape[:2]
        R = int(max(r * fw * 1.9, 95))
        cx, cy = int(x * fw), int(y * fh)
        c = fr[max(0, cy - R):cy + R, max(0, cx - R):cx + R]
        if c.size == 0:
            return np.zeros((150, 150, 3), np.uint8)
        c = cv2.resize(c, (150, 150))
        cv2.putText(c, tag, (4, 15), 0, 0.42, (60, 255, 90), 1)
        return c

    rows_img, sheet_i = [], 1
    for c in pool:
        x, y, r = c["xy"]
        strip = [crop(c["t"] + dt, x, y, r, f"{dt:+d}s") for dt in (-8, -3, 1, 6, 12)]
        label = np.zeros((150, 118, 3), np.uint8)
        cv2.putText(label, f"#{c['id']}", (6, 40), 0, 0.7, (255, 255, 255), 2)
        cv2.putText(label, f"t={int(c['t'])}", (6, 72), 0, 0.5, (200, 200, 200), 1)
        cv2.putText(label, "+".join(c["sources"]), (6, 100), 0, 0.42, (120, 200, 255), 1)
        rows_img.append(np.hstack([label] + strip))
        if len(rows_img) == 8:
            cv2.imwrite(str(outdir / f"sheet_{sheet_i:02d}.jpg"), np.vstack(rows_img),
                        [cv2.IMWRITE_JPEG_QUALITY, 86])
            rows_img, sheet_i = [], sheet_i + 1
    if rows_img:
        cv2.imwrite(str(outdir / f"sheet_{sheet_i:02d}.jpg"), np.vstack(rows_img),
                    [cv2.IMWRITE_JPEG_QUALITY, 86])
    cap.release()
    print(f"листов на ревью: {sheet_i} в {outdir}")


def score():
    pool = json.loads(POOL.read_text())
    v = json.loads(VERDICTS.read_text())
    real = set(v.get("real", []))
    false_ = set(v.get("false", []))
    judged = real | false_
    prod = [c for c in pool if "prod" in c["sources"]]
    prod_ids = {c["id"] for c in prod}
    tp = len(prod_ids & real)
    fp = len(prod_ids & false_)
    missed = real - prod_ids
    print(f"размечено: {len(judged)} из {len(pool)} кандидатов "
          f"({len(real)} настоящих, {len(false_)} ложных)")
    print(f"\nПРОДАКШН-ДЕТЕКТОР:")
    print(f"  верных переворотов:  {tp}")
    print(f"  ложных срабатываний: {fp}")
    if tp + fp:
        print(f"  точность (precision): {100*tp/(tp+fp):.0f}%  "
              f"→ доля ложных {100*fp/(tp+fp):.0f}%")
    if real:
        print(f"  пропущено настоящих: {len(missed)} {sorted(missed)}")
        print(f"  полнота (recall):     {100*tp/len(real):.0f}%")


if __name__ == "__main__":
    (candidates if sys.argv[1:2] == ["candidates"] else score)()
