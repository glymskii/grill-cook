"""RTSP -> detector -> tracker -> TimerEngine, as a restartable thread.

Process-latest-drop-stale: the reader thread keeps only the newest frame, the
worker takes it when free. Reconnects the stream after 3 s of silence — a
webcam pushed over the internet will hiccup and must not require a restart.
"""
import threading
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
BEST = ROOT / "runs/detect/runs/detect/patty_v7/weights/best.pt"


def lab_of(frame, x1, y1, x2, y2):
    """Mean Lab colour of the patty's middle — its identity across occlusions.

    Only the central 60% is sampled: the rim blends into the griddle and would
    drag every patty's colour toward the same grey.
    """
    fh, fw = frame.shape[:2]
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    hw, hh = (x2 - x1) * 0.3, (y2 - y1) * 0.3
    a_, b_ = max(0, int(cx - hw)), max(0, int(cy - hh))
    c_, d_ = min(fw, int(cx + hw)), min(fh, int(cy + hh))
    if c_ - a_ < 2 or d_ - b_ < 2:
        return None
    m = cv2.cvtColor(frame[b_:d_, a_:c_], cv2.COLOR_BGR2LAB).reshape(-1, 3).mean(axis=0)
    return (float(m[0]), float(m[1]), float(m[2]))


def dedup_dets(dets):
    """Two boxes whose centres nearly coincide are one object twice.

    Agnostic NMS works on IoU, and a squat box plus a tall box over the same
    patty can slip under the threshold together. Physical objects cannot
    overlap, so keep the confident one.
    """
    import math
    dets = sorted(dets, key=lambda d: -d[3])
    kept = []
    for d in dets:
        if all(math.hypot(d[0] - k[0], d[1] - k[1]) > 0.8 * max(d[2], k[2])
               for k in kept):
            kept.append(d)
    return kept


def size_gate(dets, radii, lo=0.58, hi=1.7, warmup=12):
    """Patties on one griddle are the same size — outliers are not patties.

    The reference is a running median of confidently detected radii, so it
    adapts to the camera's framing instead of assuming pixel sizes. Until the
    median has `warmup` samples everything passes: a cold start must not
    swallow the first real patties.
    """
    for d in dets:
        if d[3] >= 0.6:
            radii.append(d[2])
    if len(radii) < warmup:
        return dets, 0
    ref = sorted(radii)[len(radii) // 2]
    kept = [d for d in dets if lo * ref <= d[2] <= hi * ref]
    return kept, len(dets) - len(kept)


def in_poly(x: float, y: float, poly) -> bool:
    """Ray-cast point-in-polygon over normalized vertices."""
    inside = False
    n = len(poly)
    for i in range(n):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % n]
        if (y1 > y) != (y2 > y) and x < (x2 - x1) * (y - y1) / (y2 - y1) + x1:
            inside = not inside
    return inside


class Pipeline(threading.Thread):
    def __init__(self, settings: dict, engine):
        super().__init__(daemon=True)
        self.s = settings
        self.engine = engine
        self.stop_flag = threading.Event()
        self.engine_lock = threading.Lock()
        self.lock = threading.Lock()
        self.latest = None                    # (frame, t_arr)
        self.jpeg = None                      # annotated preview
        self.clipbuf = deque(maxlen=64)       # ~16s of clean frames at 4fps
        self.radii = deque(maxlen=180)        # running size reference
        self.frame_wh = (0, 0)
        self.status = {"stream": "connecting", "source_fps": 0.0,
                       "processed_fps": 0.0, "detect_ms": 0.0, "last_frame_age": None,
                       "scene_cuts": 0, "size_rejected": 0}
        self._in_stamps, self._out_stamps = [], []

    # ---- model --------------------------------------------------------------
    def _load_model(self):
        import sys
        sys.path.insert(0, str(ROOT / "src"))
        from config import Config
        cfg = Config()
        if self.s.get("model") == "world":
            from ultralytics import YOLOWorld
            m = YOLOWorld(cfg.model_name)
            m.set_classes(list(cfg.classes))
            return m, cfg, cfg.conf, cfg.conf
        from ultralytics import YOLO
        birth = float(self.s.get("conf", 0.30))
        return YOLO(str(BEST)), cfg, min(0.15, birth), birth

    # ---- reader -------------------------------------------------------------
    def _reader(self):
        url = self.s["rtsp_url"]
        cap = None
        last_ok = 0.0
        while not self.stop_flag.is_set():
            if cap is None:
                cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
                if not cap.isOpened():
                    cap.release(); cap = None
                    self.status["stream"] = "no stream"
                    time.sleep(1.0)
                    continue
                self.status["stream"] = "ok"
                last_ok = time.time()
            ok, frame = cap.read()
            now = time.time()
            if ok:
                with self.lock:
                    self.latest = (frame, now)
                    self.frame_wh = (frame.shape[1], frame.shape[0])
                self._in_stamps.append(now)
                last_ok = now
            elif now - last_ok > 3.0:
                cap.release(); cap = None
                self.status["stream"] = "reconnecting"
            else:
                time.sleep(0.005)

    # ---- worker -------------------------------------------------------------
    def run(self):
        model, cfg, conf, birth = self._load_model()
        self.engine.birth_conf = birth
        reader = threading.Thread(target=self._reader, daemon=True)
        reader.start()
        common = dict(conf=conf, iou=cfg.iou, imgsz=int(self.s.get("imgsz", 960)),
                      device=cfg.device, verbose=False, agnostic_nms=True,
                      persist=True, tracker=cfg.tracker)
        last_jpeg = 0.0
        # scene-cut detector: same border-band scheme the offline pipeline uses.
        # Bands sit outside the cooking surface, so hands and patties do not
        # touch them; when >=3 of 4 bands jump at once the whole view moved.
        bands = {"top": (slice(0, 7), slice(None)),
                 "bottom": (slice(47, 54), slice(None)),
                 "left": (slice(None), slice(0, 12)),
                 "right": (slice(None), slice(84, 96))}
        prev_small = None
        last_cut = 0.0
        band_ema = 0.0     # rolling noise floor of the trigger statistic
        pending_cut = None # candidate awaiting association evidence
        probe = []
        while not self.stop_flag.is_set():
            with self.lock:
                item, self.latest = self.latest, None
            if item is None:
                time.sleep(0.005)
                continue
            frame, t_arr = item
            t1 = time.time()
            if not self.clipbuf or t1 - self.clipbuf[-1][0] >= 0.25:
                ch, cw = frame.shape[:2]
                small_clip = frame if cw <= 960 else cv2.resize(
                    frame, (960, int(ch * 960 / cw)))
                okc, jb = cv2.imencode(".jpg", small_clip,
                                       [cv2.IMWRITE_JPEG_QUALITY, 70])
                if okc:
                    self.clipbuf.append((t1, jb.tobytes()))
            small = cv2.cvtColor(cv2.resize(frame, (96, 54)),
                                 cv2.COLOR_BGR2GRAY).astype(np.float32)
            small -= float(small.mean())   # auto-exposure steps vanish here
            cut = False
            if prev_small is not None:
                d = np.abs(small - prev_small)
                # 3rd-highest band mean == "at least 3 of 4 bands this hot".
                # Threshold adapts to the stream's own noise floor: a real fixed
                # camera sits near 0.2 and keeps the absolute floor; synthetic or
                # vibrating footage raises the floor instead of spamming cuts.
                stat = sorted(float(d[sl].mean()) for sl in bands.values())[1]
                cut = stat > max(8.0, band_ema * 5.0)
                if not cut:
                    band_ema = 0.95 * band_ema + 0.05 * stat
            prev_small = small
            r = model.track(frame, **common)[0]
            t2 = time.time()
            # a band spike alone is a CANDIDATE: the cook leaning in or an
            # exposure step also lights the borders. Only a real view change
            # breaks track association — so demand that evidence first.
            if cut and pending_cut is None and t2 - last_cut > 3.0:
                pending_cut = t2
                probe = []
                self.status["scene_candidates"] = self.status.get("scene_candidates", 0) + 1
            if pending_cut is not None:
                if t2 - pending_cut > 0.3:
                    with self.engine_lock:
                        m, a = self.engine.match_stat
                    probe.append(m / a if a else 1.0)
                if t2 - pending_cut > 1.5:
                    broke = probe and sum(probe) / len(probe) < 0.4
                    with self.engine_lock:
                        has_alive = bool(self.engine.alive)
                        if broke and has_alive:
                            last_cut = t2
                            self.status["scene_cuts"] += 1
                            self.engine.scene_reset(t2)
                    pending_cut = None
            fw, fh = frame.shape[1], frame.shape[0]
            dets = []
            if r.boxes is not None:
                for (x1, y1, x2, y2), cf in zip(r.boxes.xyxy.tolist(),
                                                r.boxes.conf.tolist()):
                    w, h = x2 - x1, y2 - y1
                    if w < 8 or h < 8 or max(w / h, h / w) > cfg.max_aspect:
                        continue
                    if max(w, h) / max(fw, fh) > cfg.max_size_frac:
                        continue
                    dets.append((((x1 + x2) / 2) / fw, ((y1 + y2) / 2) / fh,
                                 (w + h) / 4 / fw, round(cf, 3),
                                 lab_of(frame, x1, y1, x2, y2)))
            dets = dedup_dets(dets)
            dets, n_size = size_gate(dets, self.radii)
            self.status["size_rejected"] = n_size
            roi = self.s.get("roi") or []
            if len(roi) >= 3:
                # timers exist only inside the work zone: the pass tray and
                # prep boards may hold patties, but nobody cooks there
                kept = [d for d in dets if in_poly(d[0], d[1], roi)]
                self.status["outside_roi"] = len(dets) - len(kept)
                dets = kept
            else:
                self.status["outside_roi"] = 0
            with self.engine_lock:
                self.engine.update(dets, t2)
            self._out_stamps.append(t2)
            self.status["detect_ms"] = round((t2 - t1) * 1e3, 1)
            self.status["last_frame_age"] = round(time.time() - t_arr, 2)
            for stamps, key in ((self._in_stamps, "source_fps"),
                                (self._out_stamps, "processed_fps")):
                cut = t2 - 5.0
                while stamps and stamps[0] < cut:
                    stamps.pop(0)
                self.status[key] = round(len(stamps) / 5.0, 1)
            if t2 - last_jpeg > 0.5:
                last_jpeg = t2
                vis = r.plot(line_width=2)
                if len(roi) >= 3:
                    pts = np.array([[int(px * fw), int(py * fh)] for px, py in roi],
                                   dtype=np.int32)
                    cv2.polylines(vis, [pts], True, (80, 220, 90), 2)
                scale = 640 / max(fw, fh)
                vis = cv2.resize(vis, (int(fw * scale), int(fh * scale)))
                okj, buf = cv2.imencode(".jpg", vis, [cv2.IMWRITE_JPEG_QUALITY, 70])
                if okj:
                    self.jpeg = buf.tobytes()


    def stop(self):
        self.stop_flag.set()
