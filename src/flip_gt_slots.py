"""Exhaustive flip ground truth, per slot, along the whole life of the slot.

The earlier ground truth judged only the moments the old engine had proposed, so
it could measure precision and never recall. This one starts from the place,
not from the engine: every anchored slot gets a 1-second trace of what its own
pixels did (brightness, yellowness, texture, disturbance), candidates come from
that trace (a lasting colour step, or a burst of disturbance), each candidate is
rendered as a strip and judged by eye, and only then is any engine scored.

  python src/flip_gt_slots.py trace long        -> data/slot_trace_long.json
  python src/flip_gt_slots.py candidates long   -> out/flip_gt/long_*.jpg + data/flip_cands_long.json
  python src/flip_gt_slots.py score long        -> recall / precision of data/slot_events_long.json
                                                   against data/flip_gt_slots_long.json
"""
import json
import math
import statistics as st
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
SRC = {"long": ("data/IMG_6635.mov", "out/v9_review/life_long.json"),
       "smash": ("data/IMG_6637.mov", "out/v9_review/life_smash.json")}


def stats(lab, x, y, r):
    """Inner-disc mean Lab, its texture, and how far it sits from the ring."""
    sh, sw = lab.shape[:2]
    R, cx, cy = r * sw, x * sw, y * sh
    x0, x1 = int(max(0, cx - 2 * R)), int(min(sw, cx + 2 * R))
    y0, y1 = int(max(0, cy - 2 * R)), int(min(sh, cy + 2 * R))
    win = lab[y0:y1, x0:x1]
    if win.size == 0:
        return None
    yy, xx = np.ogrid[y0:y1, x0:x1]
    d = np.hypot(xx - cx, yy - cy)
    inner = win[d <= 0.6 * R]
    ring = win[(d >= 1.15 * R) & (d <= 1.9 * R)]
    if len(inner) < 20 or len(ring) < 40:
        return None
    m = inner.mean(0)
    return [round(float(m[0]), 1), round(float(m[1]), 1), round(float(m[2]), 1),
            round(float(inner[:, 0].std()), 1),
            round(float(math.dist(m, np.median(ring, 0))), 1)]


def trace(tag):
    video, life_f = SRC[tag]
    life = json.loads((ROOT / life_f).read_text())
    cap = cv2.VideoCapture(str(ROOT / video))
    fps = cap.get(cv2.CAP_PROP_FPS)
    n_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    out = {sid: [] for sid in life}
    prev = {}
    t, i = 0.0, 0
    while True:
        want = int(t * fps)
        if want >= n_total:
            break
        while i < want:
            if not cap.grab():
                break
            i += 1
        ok, f = cap.read()
        i += 1
        if not ok:
            break
        fh, fw = f.shape[:2]
        lab = cv2.cvtColor(cv2.resize(f, (960, int(960 * fh / fw))), cv2.COLOR_BGR2LAB)
        for sid, s in life.items():
            if not (s["start"] - 5 <= t <= s["start"] + s["life"] + 5):
                continue
            x, y, r = s["anchor"]
            st_ = stats(lab, x, y, r)
            if st_ is None:
                continue
            # disturbance: how much the inner disc changed since last second
            dist = 0.0
            if sid in prev:
                dist = round(float(math.dist(st_[:3], prev[sid][:3])), 1)
            prev[sid] = st_
            out[sid].append([round(t, 1)] + st_ + [dist])
        t += 1.0
        if int(t) % 120 == 0:
            print(f"  {t:.0f}s", flush=True)
    cap.release()
    (ROOT / f"data/slot_trace_{tag}.json").write_text(json.dumps(out))
    print("trace:", {sid: len(v) for sid, v in out.items()})


def candidates(tag, step_L=14.0, step_b=10.0, burst=22.0, merge=10.0):
    video, life_f = SRC[tag]
    life = json.loads((ROOT / life_f).read_text())
    tr = json.loads((ROOT / f"data/slot_trace_{tag}.json").read_text())
    cands = []          # (sid, t, kind, dL, db)
    for sid, rows in tr.items():
        if len(rows) < 25:
            continue
        ts = [r[0] for r in rows]
        L = [r[1] for r in rows]
        B = [r[3] for r in rows]
        D = [r[6] for r in rows]
        # a lasting colour step: median of the 10 s before vs the 10 s after,
        # with the 3 s around the point left out (that is where the hand is)
        got = []
        for k in range(12, len(rows) - 12):
            b_L = st.median(L[k - 12:k - 2]); a_L = st.median(L[k + 2:k + 12])
            b_B = st.median(B[k - 12:k - 2]); a_B = st.median(B[k + 2:k + 12])
            dL, db = a_L - b_L, a_B - b_B
            if abs(dL) >= step_L or abs(db) >= step_b:
                got.append((ts[k], "step", round(dL), round(db), abs(dL) + abs(db)))
        # a burst of disturbance: the cook is doing something here
        for k in range(1, len(rows) - 1):
            if D[k] >= burst and D[k] >= D[k - 1] and D[k] >= D[k + 1]:
                got.append((ts[k], "burst", 0, 0, D[k]))
        got.sort()
        # merge neighbours: keep the strongest inside each window
        merged = []
        for g in got:
            if merged and g[0] - merged[-1][0] <= merge:
                if g[4] > merged[-1][4] or (merged[-1][1] == "burst" and g[1] == "step"):
                    merged[-1] = g
                continue
            merged.append(g)
        for g in merged:
            cands.append({"sid": int(sid), "t": g[0], "kind": g[1], "dL": g[2], "db": g[3]})
    cands.sort(key=lambda c: (c["sid"], c["t"]))
    for k, c in enumerate(cands, 1):
        c["id"] = k
    (ROOT / f"data/flip_cands_{tag}.json").write_text(json.dumps(cands, indent=0))
    print(f"{tag}: {len(cands)} candidates over {len(tr)} slots "
          f"(steps {sum(1 for c in cands if c['kind']=='step')}, "
          f"bursts {sum(1 for c in cands if c['kind']=='burst')})")

    # review strips
    cap = cv2.VideoCapture(str(ROOT / video))
    fps = cap.get(cv2.CAP_PROP_FPS)
    outdir = ROOT / "out/flip_gt"
    outdir.mkdir(parents=True, exist_ok=True)
    OFFS = (-8, -3, 3, 8, 15)
    strips = []
    for c in cands:
        x, y, r = life[str(c["sid"])]["anchor"]
        tiles = []
        for dt in OFFS:
            cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, int((c["t"] + dt) * fps)))
            ok, f = cap.read()
            if not ok:
                tiles.append(np.zeros((130, 130, 3), np.uint8))
                continue
            fh, fw = f.shape[:2]
            R = int(r * fw * 1.4)
            cx, cy = int(x * fw), int(y * fh)
            crop = f[max(0, cy - R):cy + R, max(0, cx - R):cx + R]
            crop = cv2.resize(crop, (130, 130)) if crop.size else np.zeros((130, 130, 3), np.uint8)
            cv2.putText(crop, f"{dt:+d}s", (4, 16), cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255), 1)
            tiles.append(crop)
        row = np.hstack(tiles)
        lab = np.zeros((24, row.shape[1], 3), np.uint8)
        cv2.putText(lab, f"id {c['id']}  slot {c['sid']}  t={c['t']:.0f}s  {c['kind']}  dL{c['dL']:+d} db{c['db']:+d}",
                    (4, 17), cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255), 1)
        strips.append(np.vstack([lab, row]))
    cap.release()
    per = 16
    for k in range(0, len(strips), per):
        sheet = np.vstack(strips[k:k + per])
        cv2.imwrite(str(outdir / f"{tag}_{k // per + 1:02d}.jpg"), sheet)
    print(f"{math.ceil(len(strips) / per)} sheets -> out/flip_gt/{tag}_*.jpg")


def score(tag, window=12.0):
    gt = json.loads((ROOT / f"data/flip_gt_slots_{tag}.json").read_text())
    truth = [(int(sid), t) for sid, ts in gt["flips"].items() for t in ts]
    extra = gt.get("extra", {})
    truth += [(("xy", x, y), t) for x, y, t in extra.get("flips", [])]
    ev = json.loads((ROOT / f"data/slot_events_{tag}.json").read_text())
    life = json.loads((ROOT / SRC[tag][1]).read_text())
    # events from slot_life carry no coordinates: resolve the place through the
    # life file of the SAME run, by the id that run handed out
    new_life_f = ROOT / f"out/v9_review/life_{tag}_new.json"
    new_life = json.loads(new_life_f.read_text()) if new_life_f.exists() else {}
    def places(e):
        """Both places a flip can be matched at: where the slot sits now (the
        event) and where it was founded (the life file) - a patty nudged during
        its life is the same patty."""
        out = []
        if e.get("x") is not None:
            out.append((e["x"], e["y"]))
        s = new_life.get(str(e["pid"]))
        if s:
            out.append((s["anchor"][0], s["anchor"][1]))
        return out
    found = [(e["pid"], e["ts_video"], places(e)) for e in ev if e["type"] == "flip"]
    # match by PLACE and time, not by slot id: engine ids need not equal the
    # ids the truth was labelled with
    def place(sid):
        if isinstance(sid, tuple):          # ("xy", x, y): addressed by place
            return sid[1], sid[2]
        return life[str(sid)]["anchor"][:2]
    unclear = [(int(sid), place(int(sid)), t) for sid, ts in gt.get("unclear", {}).items() for t in ts]
    unclear += [(None, (x, y), t) for x, y, t in extra.get("unclear", [])]
    # an engine flip at a moment the eye could not judge is neither a hit nor
    # a false alarm: it leaves the score entirely
    def dist(pl, pxy):
        return min((math.hypot(x - pxy[0], y - pxy[1]) for x, y in pl), default=9.0)
    found = [f for f in found
             if not any(abs(f[1] - t) <= window and dist(f[2], pxy) <= 0.05
                        for sid, pxy, t in unclear)]
    # Match by PLACE only, nearest pair first across the whole set: slot ids
    # do not survive an engine change, neighbours touch, and a greedy walk in
    # truth order hands a flip to the wrong neighbour when two anchors sit
    # within a radius of each other
    pairs = []
    for i, (sid, t) in enumerate(truth):
        px, py = place(sid)
        for j, (pid, ft, pl) in enumerate(found):
            if abs(ft - t) > window:
                continue
            d = dist(pl, (px, py))
            if d <= 0.05:
                pairs.append((d, abs(ft - t), i, j))
    pairs.sort()
    hit_truth, hit_found = set(), set()
    for d, dt, i, j in pairs:
        if i in hit_truth or j in hit_found:
            continue
        hit_truth.add(i); hit_found.add(j)
    rec = len(hit_truth) / max(1, len(truth))
    prec = len(hit_found) / max(1, len(found))
    print(f"{tag}: truth {len(truth)} flips, engine {len(found)} flips -> "
          f"recall {100*rec:.0f}% ({len(hit_truth)}/{len(truth)}), "
          f"precision {100*prec:.0f}% ({len(hit_found)}/{len(found)})")
    missed = [(sid, t) for i, (sid, t) in enumerate(truth) if i not in hit_truth]
    false = [(pid, round(ft, 1)) for j, (pid, ft, _) in enumerate(found) if j not in hit_found]
    print("  missed:", missed)
    print("  false :", false)


if __name__ == "__main__":
    cmd, tag = sys.argv[1], sys.argv[2]
    {"trace": trace, "candidates": candidates, "score": score}[cmd](tag)
