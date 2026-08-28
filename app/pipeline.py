"""RTSP -> detector -> tracker -> TimerEngine, as a restartable thread.

Process-latest-drop-stale: the reader thread keeps only the newest frame, the
worker takes it when free. Reconnects the stream after 3 s of silence — a
webcam pushed over the internet will hiccup and must not require a restart.
"""
import threading
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
BEST = ROOT / "runs/detect/runs/detect/patty_v2/weights/best.pt"


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
        self.frame_wh = (0, 0)
        self.status = {"stream": "connecting", "source_fps": 0.0,
                       "processed_fps": 0.0, "detect_ms": 0.0, "last_frame_age": None,
                       "scene_cuts": 0}
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
            return m, cfg, cfg.conf
        from ultralytics import YOLO
        return YOLO(str(BEST)), cfg, float(self.s.get("conf", 0.25))

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
        model, cfg, conf = self._load_model()
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
        while not self.stop_flag.is_set():
            with self.lock:
                item, self.latest = self.latest, None
            if item is None:
                time.sleep(0.005)
                continue
            frame, t_arr = item
            t1 = time.time()
            small = cv2.cvtColor(cv2.resize(frame, (96, 54)),
                                 cv2.COLOR_BGR2GRAY).astype(np.int16)
            cut = False
            if prev_small is not None:
                d = np.abs(small - prev_small)
                # 3rd-highest band mean == "at least 3 of 4 bands this hot".
                # Threshold adapts to the stream's own noise floor: a real fixed
                # camera sits near 0.2 and keeps the absolute 9.0; synthetic or
                # vibrating footage raises the floor instead of spamming cuts.
                stat = sorted(float(d[sl].mean()) for sl in bands.values())[1]
                cut = stat > max(9.0, band_ema * 5.0)
                if not cut:
                    band_ema = 0.95 * band_ema + 0.05 * stat
            prev_small = small
            r = model.track(frame, **common)[0]
            t2 = time.time()
            if cut and t2 - last_cut > 3.0:
                last_cut = t2
                self.status["scene_cuts"] += 1
                with self.engine_lock:
                    self.engine.scene_reset(t2)
            fw, fh = frame.shape[1], frame.shape[0]
            dets = []
            if r.boxes is not None:
                for x1, y1, x2, y2 in r.boxes.xyxy.tolist():
                    w, h = x2 - x1, y2 - y1
                    if w < 8 or h < 8 or max(w / h, h / w) > cfg.max_aspect:
                        continue
                    if max(w, h) / max(fw, fh) > cfg.max_size_frac:
                        continue
                    dets.append((((x1 + x2) / 2) / fw, ((y1 + y2) / 2) / fh,
                                 (w + h) / 4 / fw))
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
                scale = 640 / max(fw, fh)
                vis = cv2.resize(vis, (int(fw * scale), int(fh * scale)))
                okj, buf = cv2.imencode(".jpg", vis, [cv2.IMWRITE_JPEG_QUALITY, 70])
                if okj:
                    self.jpeg = buf.tobytes()


    def stop(self):
        self.stop_flag.set()
