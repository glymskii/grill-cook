"""Real-time RTSP benchmark on the pilot server.

One question: at what sustained fps does ingest -> decode -> detect -> track
run against a live stream, and with what end-to-end latency. Production policy
is process-latest-drop-stale, so the reader keeps only the newest frame and we
count how many source frames each processed frame skipped.

The script owns its RTSP source: mediamtx relays, ffmpeg publishes the bench
clip at real-time pace with -c copy, so serving costs nothing and the client
pays the same decode bill it would pay a real camera. Same topology as the
pilot (camera -> relay -> consumers).

Usage: python src/bench_rtsp.py --model runs/detect/runs/patty_v1/weights/best.pt \
           --imgsz 960 --duration 300 --tag ft960
"""
import argparse
import json
import statistics as st
import subprocess
import threading
import time
from pathlib import Path

import cv2

from config import Config

URL = "rtsp://127.0.0.1:8555/cam"


class LatestFrame:
    """Single-slot buffer: the reader overwrites, the consumer drains."""

    def __init__(self):
        self.lock = threading.Lock()
        self.frame = None
        self.seq = -1
        self.t_arr = 0.0
        self.frames_in = 0
        self.stop = False

    def run(self, cap):
        while not self.stop:
            ok, frame = cap.read()
            if not ok:
                time.sleep(0.005)
                continue
            with self.lock:
                self.frame = frame
                self.seq += 1
                self.t_arr = time.monotonic()
                self.frames_in += 1

    def take(self):
        with self.lock:
            if self.frame is None:
                return None
            f, s, t = self.frame, self.seq, self.t_arr
            self.frame = None
            return f, s, t


def open_stream():
    for _ in range(40):
        cap = cv2.VideoCapture(URL, cv2.CAP_FFMPEG)
        if cap.isOpened():
            return cap
        cap.release()
        time.sleep(0.5)
    raise RuntimeError("RTSP source never came up")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--imgsz", type=int, default=960)
    ap.add_argument("--duration", type=float, default=90.0)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--source", default="data/bench_cam.mp4")
    args = ap.parse_args()

    cfg = Config()
    if "world" in args.model:
        from ultralytics import YOLOWorld
        model = YOLOWorld(args.model)
        model.set_classes(list(cfg.classes))
        conf = cfg.conf                       # its scores live at the noise floor
    else:
        from ultralytics import YOLO
        model = YOLO(args.model)
        conf = args.conf

    relay = subprocess.Popen(
        ["/opt/homebrew/opt/mediamtx/bin/mediamtx",
         "data/bench_mediamtx.yml"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    time.sleep(0.7)
    server = subprocess.Popen(
        ["ffmpeg", "-v", "error", "-re", "-stream_loop", "-1", "-i", args.source,
         "-c", "copy", "-bsf:v", "h264_mp4toannexb", "-an", "-f", "rtsp", "-rtsp_transport", "tcp", URL])
    try:
        time.sleep(1.0)
        cap = open_stream()
        slot = LatestFrame()
        reader = threading.Thread(target=slot.run, args=(cap,), daemon=True)
        reader.start()

        common = dict(conf=conf, iou=cfg.iou, imgsz=args.imgsz, device=cfg.device,
                      verbose=False, agnostic_nms=True, persist=True,
                      tracker=cfg.tracker)

        # warmup outside the measured window
        for _ in range(10):
            got = slot.take()
            if got:
                model.track(got[0], **common)
            time.sleep(0.05)

        det_ms, e2e_ms, skips, boxes, stamps = [], [], [], [], []
        last_seq = None
        t0 = time.monotonic()
        in0 = slot.frames_in
        while time.monotonic() - t0 < args.duration:
            got = slot.take()
            if got is None:
                time.sleep(0.002)
                continue
            frame, seq, t_arr = got
            t1 = time.monotonic()
            r = model.track(frame, **common)[0]
            t2 = time.monotonic()
            det_ms.append((t2 - t1) * 1e3)
            e2e_ms.append((t2 - t_arr) * 1e3)
            boxes.append(0 if r.boxes is None else len(r.boxes))
            if last_seq is not None:
                skips.append(seq - last_seq - 1)
            last_seq = seq
            stamps.append(t2)
        elapsed = time.monotonic() - t0
        slot.stop = True

        n = len(det_ms)
        per_min = []
        for w in range(int(elapsed // 60) or 1):
            lo, hi = t0 + 60 * w, t0 + 60 * (w + 1)
            per_min.append(round(sum(1 for s in stamps if lo <= s < hi) / 60, 2))
        out = {
            "tag": args.tag, "model": args.model, "imgsz": args.imgsz,
            "duration_s": round(elapsed, 1),
            "source_fps": round((slot.frames_in - in0) / elapsed, 2),
            "processed_fps": round(n / elapsed, 2),
            "detect_ms_p50": round(st.median(det_ms), 1),
            "detect_ms_p95": round(sorted(det_ms)[int(0.95 * n)], 1),
            "e2e_ms_p50": round(st.median(e2e_ms), 1),
            "e2e_ms_p95": round(sorted(e2e_ms)[int(0.95 * n)], 1),
            "skip_ratio": round(sum(skips) / max(1, len(skips)), 2),
            "boxes_avg": round(sum(boxes) / n, 1),
            "fps_per_minute": per_min,
        }
        print(json.dumps(out, indent=1))
        res = Path("data/bench_results.json")
        hist = json.loads(res.read_text()) if res.exists() else []
        hist.append(out)
        res.write_text(json.dumps(hist, indent=1))
    finally:
        server.terminate()
        relay.terminate()


if __name__ == "__main__":
    main()
