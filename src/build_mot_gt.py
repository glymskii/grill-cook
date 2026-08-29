"""Build MOT ground truth from dumped detections with full hindsight.

Offline we know the future: tracklets are grown with a tight gate, then merged
across gaps only when the spot stays empty in between — and every merge longer
than a hand-wave, plus every track boundary, is rendered to a review image so
a human (me) can veto errors via data/mot_overrides_<window>.json:
  {"forbid_merges": [[idA, idB]], "force_merges": [[idA, idB]], "drop": [id]}

Usage: python src/build_mot_gt.py tune
"""
import json
import math
import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parent.parent
WINDOW = sys.argv[1] if len(sys.argv) > 1 else "tune"
GATE = 0.7          # frame-to-frame, in units of r
MERGE_DIST = 0.9    # tracklet stitch distance
MERGE_GAP = 30.0    # max seconds between tracklets to stitch
AUTO_GAP = 6.0      # stitches shorter than this need no review

rows = json.loads((ROOT / f"data/mot_dets_{WINDOW}.json").read_text())
overrides_f = ROOT / f"data/mot_overrides_{WINDOW}.json"
ov = json.loads(overrides_f.read_text()) if overrides_f.exists() else {}
forbid = {tuple(sorted(x)) for x in ov.get("forbid_merges", [])}
force = {tuple(sorted(x)) for x in ov.get("force_merges", [])}
drop = set(ov.get("drop", []))

# --- tracklets ---------------------------------------------------------------
tracklets = []      # {"id", "frames": [(t, x, y, r)], "last": (x, y, r)}
next_id = 1
for row in rows:
    t = row["t"]
    used = set()
    for tr in tracklets:
        if t - tr["frames"][-1][0] > 1.2:      # only currently-live tracklets bid
            continue
        best, bd = None, 1e9
        lx, ly, lr = tr["last"]
        for i, d in enumerate(row["dets"]):
            if i in used:
                continue
            dist = math.hypot(d[0] - lx, d[1] - ly) / max(lr, 1e-6)
            if dist < bd:
                best, bd = i, dist
        if best is not None and bd <= GATE:
            used.add(best)
            d = row["dets"][best]
            tr["frames"].append((t, d[0], d[1], d[2]))
            tr["last"] = (d[0], d[1], d[2])
    for i, d in enumerate(row["dets"]):
        if i not in used:
            tracklets.append({"id": next_id, "frames": [(t, d[0], d[1], d[2])],
                              "last": (d[0], d[1], d[2])})
            next_id += 1

# --- hindsight stitching -----------------------------------------------------
tracklets.sort(key=lambda tr: tr["frames"][0][0])
def occupied(x, y, r, t0, t1, exclude):
    for tr in tracklets:
        if tr["id"] in exclude:
            continue
        for (tt, xx, yy, rr) in tr["frames"]:
            if t0 < tt < t1 and math.hypot(xx - x, yy - y) < 0.8 * r:
                return True
    return False

merges = []
merged_into = {}
changed = True
while changed:
    changed = False
    tracklets.sort(key=lambda tr: tr["frames"][0][0])
    for a in tracklets:
        for b in tracklets:
            if a is b:
                continue
            a_end, b_start = a["frames"][-1], b["frames"][0]
            gap = b_start[0] - a_end[0]
            key = tuple(sorted((a["id"], b["id"])))
            if key in forbid:
                continue
            dist = math.hypot(b_start[1] - a_end[1], b_start[2] - a_end[2]) / max(a_end[3], 1e-6)
            ok = (0 < gap <= MERGE_GAP and dist <= MERGE_DIST
                  and not occupied(a_end[1], a_end[2], a_end[3],
                                   a_end[0], b_start[0], {a["id"], b["id"]}))
            if key in force:
                ok = 0 < gap
            if ok:
                merges.append((a["id"], b["id"], round(gap, 1), round(dist, 2)))
                a["frames"] += b["frames"]
                a["last"] = b["last"]
                merged_into[b["id"]] = a["id"]
                tracklets.remove(b)
                changed = True
                break
        if changed:
            break

# --- filter noise ------------------------------------------------------------
gt = [tr for tr in tracklets
      if tr["id"] not in drop
      and tr["frames"][-1][0] - tr["frames"][0][0] >= 4.0 and len(tr["frames"]) >= 12]
short = len(tracklets) - len(gt)

out = {"window": WINDOW,
       "tracks": [{"id": tr["id"], "frames": tr["frames"]} for tr in gt],
       "merges": merges}
(ROOT / f"data/mot_gt_{WINDOW}.json").write_text(json.dumps(out))
review = [m for m in merges if m[2] > AUTO_GAP]
print(f"{WINDOW}: треклетов {next_id-1} -> GT-треков {len(gt)} (шумовых убрано {short})")
print(f"склеек {len(merges)}, из них на ревью (пауза > {AUTO_GAP}с): {len(review)}")
for m in review:
    print(f"  merge {m[0]}<-{m[1]}: пауза {m[2]}с, дистанция {m[3]}r")

# --- review images -----------------------------------------------------------
cap = cv2.VideoCapture(str(ROOT / "data/IMG_6635.mov"))
src_fps = cap.get(cv2.CAP_PROP_FPS)
outdir = ROOT / "out" / f"mot_review_{WINDOW}"
outdir.mkdir(parents=True, exist_ok=True)
def crop_at(t, x, y, r):
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(t * src_fps))
    ok, fr = cap.read()
    if not ok:
        return None
    fh, fw = fr.shape[:2]
    R = int(max(r * fw * 2.2, 90))
    cx, cy = int(x * fw), int(y * fh)
    return cv2.resize(fr[max(0, cy - R):cy + R, max(0, cx - R):cx + R], (220, 220))

import numpy as np
for (aid, bid, gap, dist) in review:
    tr = next((q for q in gt if q["id"] == aid), None)
    if tr is None:
        continue
    fr_list = tr["frames"]
    idx = next(i for i, f in enumerate(fr_list) if f[0] - fr_list[0][0] >= 0) 
    # найдём точку склейки: кадр с паузой
    for i in range(1, len(fr_list)):
        if fr_list[i][0] - fr_list[i-1][0] >= gap - 0.5:
            before, after = fr_list[i-1], fr_list[i]
            a_img = crop_at(before[0], before[1], before[2], before[3])
            b_img = crop_at(after[0], after[1], after[2], after[3])
            if a_img is not None and b_img is not None:
                canvas = np.hstack([a_img, b_img])
                cv2.putText(canvas, f"{aid}<-{bid} gap {gap}s", (6, 18), 0, 0.55, (0, 255, 0), 2)
                cv2.imwrite(str(outdir / f"merge_{aid}_{bid}.jpg"), canvas)
            break
cap.release()
print(f"ревью-кадры: {outdir}")
