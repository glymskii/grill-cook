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
CACHE = STATE / "hub_cache.json"


class Worker:
    def __init__(self):
        self.pipe = None
        self.engine = None
        self.settings = None
        self.pending_events = []

    def cache(self, data: dict):
        STATE.mkdir(exist_ok=True)
        old = json.loads(CACHE.read_text()) if CACHE.exists() else {}
        old.update(data)
        CACHE.write_text(json.dumps(old, ensure_ascii=False, indent=1))

    def cached(self) -> dict:
        return json.loads(CACHE.read_text()) if CACHE.exists() else {}

    def apply(self, settings: dict, run: bool):
        self.settings = settings
        self.cache({"settings": settings})
        if self.engine:
            self.engine.targets.update({
                "A": settings["target_a"], "B": settings["target_b"],
                "tol_early": settings["tol_early"], "tol_late": settings["tol_late"]})
        running = self.pipe and self.pipe.is_alive()
        if running:
            self.pipe.s = settings         # ROI/targets apply live, no restart
        if run and not running:
            self.engine = TimerEngine({"A": settings["target_a"], "B": settings["target_b"],
                                       "tol_early": settings["tol_early"],
                                       "tol_late": settings["tol_late"]})
            self.engine.on_event = lambda ev: self.pending_events.append(ev)
            self.pipe = Pipeline(settings, self.engine)
            self.pipe.start()
        elif not run and running:
            self.pipe.stop()
            self.pipe = None

    async def talk(self, url: str, token: str):
        async with websockets.connect(url, additional_headers={"x-agent-token": token},
                                      max_size=2 ** 22) as ws:
            print("hub connected")
            hello = json.loads(await ws.recv())
            if not hello.get("have_settings") and self.cached().get("settings"):
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
                    if m["type"] == "config":
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
    args = ap.parse_args()
    w = Worker()
    while True:
        try:
            await w.talk(args.hub, args.token)
        except Exception as e:
            print("hub link lost:", e)
        await asyncio.sleep(2)


if __name__ == "__main__":
    asyncio.run(main())
