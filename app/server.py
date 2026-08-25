"""Pilot server: admin + HUD + WebSocket state feed.

Single process. The vision pipeline runs as a thread the admin starts and
stops; settings and shifts persist as JSON files next to this script. Spawns
its own mediamtx relay (TCP-only, :8555) so the partner's webcam has somewhere
to push.

Run: .venv/bin/python app/server.py   -> http://<host>:8080/admin
"""
import asyncio
import json
import subprocess
import time
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, RedirectResponse, Response

from pipeline import Pipeline
from recommend import recommend
from timers import EVENTS, TimerEngine

ROOT = Path(__file__).resolve().parent
STATE = ROOT / "state"
STATE.mkdir(exist_ok=True)

DEFAULTS = {
    "rtsp_url": "rtsp://127.0.0.1:8555/cam",
    "model": "finetuned", "imgsz": 960, "conf": 0.25,
    "diameter_cm": 10.0, "thickness_cm": 1.2, "griddle_c": 190.0, "freezer_c": -18.0,
    "target_a": 180, "target_b": 150, "tol_early": 15, "tol_late": 15,
}


def load(name, fallback):
    f = STATE / name
    return json.loads(f.read_text()) if f.exists() else fallback


def save(name, data):
    (STATE / name).write_text(json.dumps(data, ensure_ascii=False, indent=1))


settings = {**DEFAULTS, **load("settings.json", {})}
shifts = load("shifts.json", [])
app = FastAPI()
pipe: Pipeline | None = None
engine: TimerEngine | None = None
relay: subprocess.Popen | None = None


def make_engine():
    return TimerEngine({"A": settings["target_a"], "B": settings["target_b"],
                        "tol_early": settings["tol_early"], "tol_late": settings["tol_late"]})


@app.on_event("startup")
def start_relay():
    global relay
    cfg = ROOT / "mediamtx.yml"
    cfg.write_text("rtspAddress: :8555\nrtspTransports: [tcp]\nrtmp: no\n"
                   "hls: no\nwebrtc: no\nsrt: no\nmoq: no\npaths:\n  all_others:\n")
    try:
        relay = subprocess.Popen(["/opt/homebrew/opt/mediamtx/bin/mediamtx", str(cfg)],
                                 stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except FileNotFoundError:
        relay = None


@app.get("/")
def root():
    return RedirectResponse("/admin")


@app.get("/admin")
def admin_page():
    return FileResponse(ROOT / "static/admin.html")


@app.get("/hud")
def hud_page():
    return FileResponse(ROOT / "static/hud.html")


@app.get("/api/settings")
def get_settings():
    return settings


@app.post("/api/settings")
async def post_settings(body: dict):
    settings.update({k: body[k] for k in DEFAULTS if k in body})
    save("settings.json", settings)
    if engine:
        engine.targets.update({"A": settings["target_a"], "B": settings["target_b"],
                               "tol_early": settings["tol_early"], "tol_late": settings["tol_late"]})
    return settings


@app.get("/api/recommend")
def api_recommend(d: float, h: float, tg: float, tf: float):
    return recommend(d, h, tg, tf)


@app.post("/api/pipeline/start")
def start_pipeline():
    global pipe, engine
    if pipe and pipe.is_alive():
        return {"running": True}
    engine = make_engine()
    pipe = Pipeline(settings, engine)
    pipe.start()
    return {"running": True}


@app.post("/api/pipeline/stop")
def stop_pipeline():
    global pipe
    if pipe:
        pipe.stop()
        pipe = None
    return {"running": False}


@app.get("/api/status")
def status():
    running = bool(pipe and pipe.is_alive())
    st = dict(pipe.status) if running else {}
    st.update({"running": running, "shift": active_shift(),
               "frame_wh": pipe.frame_wh if running else (0, 0)})
    return st


def active_shift():
    """Shifts are weekly: cook + ISO weekdays (0=Mon) + hour range."""
    now = time.localtime()
    hhmm = time.strftime("%H:%M", now)
    for s in shifts:
        if now.tm_wday in s.get("days", []) and s["start"] <= hhmm <= s["end"]:
            return s
    return None


@app.get("/api/shifts")
def get_shifts():
    return shifts


@app.post("/api/shifts")
async def post_shift(body: dict):
    if body.get("delete") is not None:
        idx = int(body["delete"])
        if 0 <= idx < len(shifts):
            shifts.pop(idx)
    else:
        shifts.append({"cook": body["cook"], "days": [int(d) for d in body["days"]],
                       "start": body["start"], "end": body["end"]})
    save("shifts.json", shifts)
    return shifts


@app.get("/api/events")
def events(limit: int = 100):
    if not EVENTS.exists():
        return []
    lines = EVENTS.read_text().strip().splitlines()[-limit:]
    return [json.loads(l) for l in reversed(lines)]


@app.get("/preview.jpg")
def preview():
    if pipe and pipe.jpeg:
        return Response(pipe.jpeg, media_type="image/jpeg",
                        headers={"Cache-Control": "no-store"})
    return Response(status_code=404)


@app.websocket("/ws")
async def ws(sock: WebSocket):
    await sock.accept()
    try:
        while True:
            now = time.time()
            if pipe and engine and pipe.is_alive():
                with pipe.engine_lock:
                    snap = engine.snapshot(now)
                snap.update({"server_ts": round(now, 3), "running": True,
                             "frame_wh": pipe.frame_wh, "stream": pipe.status["stream"],
                             "shift": active_shift(),
                             "targets": {"A": settings["target_a"], "B": settings["target_b"],
                                         "tol_early": settings["tol_early"],
                                         "tol_late": settings["tol_late"]}})
            else:
                snap = {"server_ts": round(now, 3), "running": False, "patties": []}
            await sock.send_text(json.dumps(snap))
            await asyncio.sleep(0.2)
    except WebSocketDisconnect:
        pass


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080, log_level="warning")
