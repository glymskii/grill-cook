"""Edge worker: the MacBook side of the split pilot.

Connects OUTBOUND to the cloud hub over WebSocket (nothing on this machine is
exposed to the internet), receives settings and the desired run state, runs the
vision pipeline locally, and streams back status, snapshots, preview frames and
events. The worker's disk is the durable copy of settings/shifts/events — the
hub's filesystem is ephemeral, so after a hub redeploy the worker restores them.

Run: .venv/bin/python app/edge_worker.py --hub wss://<hub>/ws/agent --token <AGENT_TOKEN>
"""
import argparse
import asyncio
import base64
import json
import os
import sys
import time
from pathlib import Path

import websockets

from pipeline import Pipeline
from timers import EVENTS, TimerEngine

STATE = Path(__file__).parent / "state"
CLIPS = STATE / "clips"
CLIP_TYPES = {"flip", "removed", "invalidated", "scene_cut"}
CACHE = STATE / "hub_cache.json"


class Worker:
    def __init__(self):
        self.pipe = None
        self.engine = None
        self.settings = None
        self.pending_events = []
        self._clip_jobs = []

    def cache(self, data: dict):
        STATE.mkdir(exist_ok=True)
        old = json.loads(CACHE.read_text()) if CACHE.exists() else {}
        old.update(data)
        CACHE.write_text(json.dumps(old, ensure_ascii=False, indent=1))

    def cached(self) -> dict:
        return json.loads(CACHE.read_text()) if CACHE.exists() else {}

    def _on_event(self, ev: dict):
        if ev.get("type") in CLIP_TYPES:
            ev = dict(ev)
            ev["clip"] = f"{int(ev['ts'] * 10)}_{ev['type']}_{ev.get('pid', 0)}.mp4"
            self._clip_jobs.append((ev["ts"], ev["clip"]))
        self.pending_events.append(ev)

    def write_clip(self, ts: float, name: str):
        """Cut [-10s..+3s] around the event from the rolling buffer."""
        if not (self.pipe and self.pipe.is_alive()):
            return
        import cv2
        import numpy as np
        frames = [(t, j) for t, j in list(self.pipe.clipbuf)
                  if ts - 10 <= t <= ts + 3]
        if len(frames) < 4:
            return
        CLIPS.mkdir(parents=True, exist_ok=True)
        first = cv2.imdecode(np.frombuffer(frames[0][1], np.uint8), cv2.IMREAD_COLOR)
        h, w = first.shape[:2]
        vw = cv2.VideoWriter(str(CLIPS / name), cv2.VideoWriter_fourcc(*"avc1"),
                             4, (w, h))
        if not vw.isOpened():
            vw = cv2.VideoWriter(str(CLIPS / name), cv2.VideoWriter_fourcc(*"mp4v"),
                                 4, (w, h))
        for _, j in frames:
            img = cv2.imdecode(np.frombuffer(j, np.uint8), cv2.IMREAD_COLOR)
            if img is not None:
                vw.write(img)
        vw.release()
        # retention: keep the newest 300 clips
        clips = sorted(CLIPS.glob("*.mp4"), key=lambda f: f.stat().st_mtime)
        for old in clips[:-300]:
            old.unlink(missing_ok=True)

    def apply(self, settings: dict, run: bool):
        self.settings = settings
        self.cache({"settings": settings})
        if self.engine:
            self.engine.targets.update({
                "A": settings["target_a"], "B": settings["target_b"],
                "tol_early": settings["tol_early"], "tol_late": settings["tol_late"],
                "skus": settings.get("_skus") or None})
        running = self.pipe and self.pipe.is_alive()
        if running:
            self.pipe.s = settings         # ROI/targets apply live, no restart
        if run and not running:
            self.engine = TimerEngine({"A": settings["target_a"], "B": settings["target_b"],
                                       "tol_early": settings["tol_early"],
                                       "tol_late": settings["tol_late"],
                                       "skus": settings.get("_skus") or None})
            self.engine.on_event = self._on_event
            self.pipe = Pipeline(settings, self.engine)
            self.pipe.start()
        elif not run and running:
            self.pipe.stop()
            self.pipe = None

    async def talk(self, url: str, token: str, station: str = "main"):
        async with websockets.connect(url,
                                      additional_headers={"x-agent-token": token,
                                                          "x-station": station},
                                      max_size=2 ** 22) as ws:
            print("hub connected")
            hello = json.loads(await ws.recv())
            need = not hello.get("have_settings") or not hello.get("have_shifts")
            if need and self.cached().get("settings"):
                await ws.send(json.dumps({"type": "restore", **self.cached()}))
                print("hub state restored from local cache (run=%s)"
                      % self.cached().get("run"))
            if EVENTS.exists():
                lines = EVENTS.read_text().strip().splitlines()[-5000:]
                evs = [json.loads(l) for l in lines]
                for i in range(0, len(evs), 500):
                    await ws.send(json.dumps({"type": "backfill",
                                              "events": evs[i:i + 500]}))
                print(f"backfilled {len(evs)} events to hub")
            last = {"snap": 0.0, "prev": 0.0}

            async def sender():
                while True:
                    now = time.time()
                    while self._clip_jobs and now - self._clip_jobs[0][0] >= 3.5:
                        ts_, name_ = self._clip_jobs.pop(0)
                        await asyncio.get_event_loop().run_in_executor(
                            None, self.write_clip, ts_, name_)
                    running = bool(self.pipe and self.pipe.is_alive())
                    if now - last["snap"] >= 0.2:
                        last["snap"] = now
                        msg = {"type": "state", "ts": round(now, 3), "running": running}
                        if running:
                            with self.pipe.engine_lock:
                                msg["snapshot"] = self.engine.snapshot(now)
                            msg["status"] = dict(self.pipe.status)
                            msg["frame_wh"] = self.pipe.frame_wh
                        while self.pending_events:
                            await ws.send(json.dumps(
                                {"type": "event", "event": self.pending_events.pop(0)}))
                        await ws.send(json.dumps(msg))
                    if running and now - last["prev"] >= 0.5 and self.pipe.jpeg:
                        last["prev"] = now
                        await ws.send(json.dumps(
                            {"type": "preview",
                             "jpg": base64.b64encode(self.pipe.jpeg).decode()}))
                    await asyncio.sleep(0.05)

            send_task = asyncio.create_task(sender())
            try:
                async for raw in ws:
                    m = json.loads(raw)
                    if m["type"] == "get_clip":
                        f = CLIPS / Path(m["name"]).name
                        if f.exists() and f.stat().st_size < 3_500_000:
                            await ws.send(json.dumps(
                                {"type": "clip", "name": m["name"],
                                 "b64": base64.b64encode(f.read_bytes()).decode()}))
                        else:
                            await ws.send(json.dumps(
                                {"type": "clip_missing", "name": m["name"]}))
                    elif m["type"] == "config":
                        self.apply(m["settings"], m["run"])
                        self.cache({"run": m["run"], "shifts": m.get("shifts", [])})
            finally:
                send_task.cancel()


PIDFILE = Path("/tmp/grill_edge_worker.pid")


def acquire_singleton():
    """Two workers fighting over the hub flap the HUD; never allow it."""
    if PIDFILE.exists():
        try:
            old = int(PIDFILE.read_text())
            os.kill(old, 0)
        except (ValueError, ProcessLookupError, PermissionError):
            pass
        else:
            sys.exit(f"edge worker already running (pid {old}); "
                     "kill it first or use that instance")
    PIDFILE.write_text(str(os.getpid()))


async def main():
    acquire_singleton()
    ap = argparse.ArgumentParser()
    ap.add_argument("--hub", required=True)
    ap.add_argument("--token", required=True)
    ap.add_argument("--station", default="main")
    args = ap.parse_args()
    w = Worker()
    while True:
        try:
            await w.talk(args.hub, args.token, args.station)
        except Exception as e:
            print("hub link lost:", e)
        await asyncio.sleep(2)


if __name__ == "__main__":
    asyncio.run(main())
