"""A/B the track-identity rework on real footage: one detector pass, two engines.

Fragmentation is the metric that matters — a fragment is a "patty" the system
believed in for a few seconds before losing it and minting a new id. The guard
against cheating (merging genuinely distinct patties into one) is the median
alive count: it must stay level while the number of ids created drops.

Usage: python src/ab_identity.py data/IMG_6635.mov --fps 10
"""
import argparse
import importlib.util
import statistics as st
import sys
import tempfile
from collections import deque
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "app"))

from config import Config                       # noqa: E402
from pipeline import dedup_dets, in_poly, lab_of, size_gate  # noqa: E402
import timers as timers_new                     # noqa: E402


def load_old(path: str):
    spec = importlib.util.spec_from_file_location("timers_old", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.EVENTS = Path(tempfile.mkdtemp()) / "events.jsonl"
    return mod


class Churn:
    """Identity-error meter: a physical patty cannot jump.

    A tracked centre moving more than 0.8 r between consecutive sightings means
    the track grabbed a different patty — the exact failure that credits frying
    time to the wrong burger. Id counters cannot see it; this does.
    """

    def __init__(self):
        self.prev = {}
        self.teleports = 0

    def feed(self, eng):
        cur = {p.pid: (p.cx, p.cy, p.r) for p in eng.alive.values()
               if p.missing_since is None}
        for pid, (x, y, r) in cur.items():
            if pid in self.prev:
                px, py, _ = self.prev[pid]
                import math
                if math.hypot(x - px, y - py) > 0.8 * r:
                    self.teleports += 1
        self.prev = cur


def report(name: str, eng, alive_counts):
    lives = [d["side_a"] + d["side_b"] for d in eng.done]
    lives += [p.side_time["A"] + p.side_time["B"] for p in eng.alive.values()]
    frags = [x for x in lives if x < 5]
    print(f"\n{name}")
    print(f"  всего id создано:        {eng.next_pid - 1}")
    print(f"  обрывков (<5с жизни):    {len(frags)} ({100*len(frags)/max(len(lives),1):.0f}%)")
    print(f"  медиана жизни трека:     {st.median(lives):.0f}с" if lives else "")
    tot = [a for a, _ in alive_counts]
    vis = [v for _, v in alive_counts]
    print(f"  медиана треков (всех/видимых):  {st.median(tot):.0f} / {st.median(vis):.0f}")
    if hasattr(eng, "revived"):
        print(f"  воскрешений:             {eng.revived}")
    if hasattr(eng, "merged"):
        print(f"  слияний дублей:          {eng.merged}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--fps", type=float, default=10.0)
    ap.add_argument("--old", default="/tmp/timers_old.py")
    ap.add_argument("--model", default="runs/detect/runs/detect/patty_v6/weights/best.pt")
    ap.add_argument("--roi", default="")
    args = ap.parse_args()

    cfg = Config()
    from ultralytics import YOLO
    model = YOLO(args.model)
    roi = [[float(v) for v in pt.split(",")] for pt in args.roi.split(";")] if args.roi else []
    old_mod = load_old(args.old)
    timers_new.EVENTS = Path(tempfile.mkdtemp()) / "events.jsonl"

    T = {"A": 120, "B": 90, "tol_early": 30, "tol_late": 30}
    v2 = load_old("/tmp/timers_v2.py")     # вчерашняя версия: венгр без КФ
    engines = [
        ("ЖАДНАЯ (исходная)", old_mod.TimerEngine(dict(T))),
        ("ВЕНГР (вчера, прод)", v2.TimerEngine(dict(T))),
        ("ЯДРО: венгр + физика (кандидат)", timers_new.TimerEngine(dict(T))),
        ("ИССЛЕД.: + КФ (флаг)", timers_new.TimerEngine(dict(T), use_kf=True,
                                                        birth_suppress=1.55)),
    ]
    churns = [Churn() for _ in engines]
    alives = [[] for _ in engines]

    cap = cv2.VideoCapture(args.video)
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30
    total = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    step = max(1, round(src_fps / args.fps))
    radii: deque = deque(maxlen=180)

    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % step:
            idx += 1
            continue
        t = idx / src_fps
        fh, fw = frame.shape[:2]
        r = model.predict(frame, conf=0.15, iou=cfg.iou, imgsz=960, device=cfg.device,
                          agnostic_nms=True, verbose=False)[0]
        dets = []
        if r.boxes is not None:
            for (x1, y1, x2, y2), cf in zip(r.boxes.xyxy.tolist(), r.boxes.conf.tolist()):
                w, h = x2 - x1, y2 - y1
                if w < 8 or h < 8 or max(w / h, h / w) > cfg.max_aspect:
                    continue
                if max(w, h) / max(fw, fh) > cfg.max_size_frac:
                    continue
                dets.append(((x1 + x2) / 2 / fw, (y1 + y2) / 2 / fh, (w + h) / 4 / fw,
                             cf, lab_of(frame, x1, y1, x2, y2)))
        dets = dedup_dets(dets)
        dets, _ = size_gate(dets, radii)
        if roi:
            dets = [d for d in dets if in_poly(d[0], d[1], roi)]
        for k, (name, eng) in enumerate(engines):
            eng.update([d[:4] for d in dets] if k == 0 else list(dets), t)
            churns[k].feed(eng)
            alives[k].append((len(eng.alive),
                              sum(1 for q in eng.alive.values()
                                  if q.missing_since is None)))
        idx += 1
        if idx % (step * 400) == 0:
            print(f"  {t/60:.1f} / {total/src_fps/60:.1f} мин", flush=True)
    cap.release()

    for k, (name, eng) in enumerate(engines):
        report(name, eng, alives[k])
        print(f"  ТЕЛЕПОРТОВ (ошибки идентичности): {churns[k].teleports}")


if __name__ == "__main__":
    main()
