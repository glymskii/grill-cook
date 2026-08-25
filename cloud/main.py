"""Cloud hub: the public face of the pilot.

Partner's browser and the admin talk to this service; the MacBook edge worker
connects outbound over /ws/agent and does all vision work. The hub keeps state
in memory (Railway's filesystem is ephemeral) — the worker holds the durable
copy and restores it after a hub redeploy.

Auth: ?k=<ADMIN_KEY> once per browser (sets a cookie), x-agent-token for the
edge worker. Set ADMIN_KEY, AGENT_TOKEN and TZ (shift matching runs in the
kitchen's local time) in the environment.
"""
import asyncio
import base64
import json
import os
import time
from collections import deque
from pathlib import Path

from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, HTMLResponse, RedirectResponse, Response

ROOT = Path(__file__).resolve().parent
ADMIN_KEY = os.environ.get("ADMIN_KEY", "dev")
AGENT_TOKEN = os.environ.get("AGENT_TOKEN", "dev")

DEFAULTS = {
    "rtsp_url": "", "model": "finetuned", "imgsz": 960, "conf": 0.25,
    "diameter_cm": 10.0, "thickness_cm": 1.2, "griddle_c": 190.0, "freezer_c": -18.0,
    "target_a": 180, "target_b": 150, "tol_early": 15, "tol_late": 15,
}

settings = dict(DEFAULTS)
settings_touched = False
shifts: list = []
events: deque = deque(maxlen=2000)
desired_run = False
agent_ws: WebSocket | None = None
agent_state: dict = {}
agent_seen = 0.0
preview: bytes | None = None

app = FastAPI()


# ---- auth ------------------------------------------------------------------
def authed(request: Request) -> bool:
    return (request.query_params.get("k") == ADMIN_KEY
            or request.cookies.get("gk") == ADMIN_KEY)


@app.middleware("http")
async def gate(request: Request, call_next):
    if request.url.path == "/health":
        return await call_next(request)
    if not authed(request):
        return HTMLResponse("<h3>Access key required — open the link with ?k=…</h3>",
                            status_code=401)
    resp = await call_next(request)
    if request.query_params.get("k") == ADMIN_KEY:
        resp.set_cookie("gk", ADMIN_KEY, max_age=7 * 86400, httponly=False)
    return resp


@app.get("/health")
def health():
    return {"ok": True, "agent": agent_connected()}


def agent_connected() -> bool:
    return agent_ws is not None and time.time() - agent_seen < 5.0


def active_shift():
    now = time.localtime()
    hhmm = time.strftime("%H:%M", now)
    for s in shifts:
        if now.tm_wday in s.get("days", []) and s["start"] <= hhmm <= s["end"]:
            return s
    return None


async def push_config():
    if agent_ws is not None:
        try:
            await agent_ws.send_text(json.dumps(
                {"type": "config", "settings": settings, "run": desired_run,
                 "shifts": shifts}))
        except Exception:
            pass


# ---- pages -----------------------------------------------------------------
@app.get("/")
def root():
    return RedirectResponse("/admin")


@app.get("/admin")
def admin_page():
    return FileResponse(ROOT / "static/admin.html")


@app.get("/hud")
def hud_page():
    return FileResponse(ROOT / "static/hud.html")


@app.get("/preview.jpg")
def preview_jpg():
    if preview:
        return Response(preview, media_type="image/jpeg",
                        headers={"Cache-Control": "no-store"})
    return Response(status_code=404)


# ---- REST ------------------------------------------------------------------
@app.get("/api/settings")
def get_settings():
    return settings


@app.post("/api/settings")
async def post_settings(body: dict):
    global settings_touched
    settings.update({k: body[k] for k in DEFAULTS if k in body})
    settings_touched = True
    await push_config()
    return settings


@app.get("/api/recommend")
def api_recommend(d: float, h: float, tg: float, tf: float):
    from recommend import recommend
    return recommend(d, h, tg, tf)


@app.post("/api/pipeline/start")
async def start_pipeline():
    global desired_run
    desired_run = True
    await push_config()
    return {"running": True}


@app.post("/api/pipeline/stop")
async def stop_pipeline():
    global desired_run
    desired_run = False
    await push_config()
    return {"running": False}


@app.get("/api/status")
def status():
    st = dict(agent_state.get("status", {}))
    st.update({"running": bool(agent_state.get("running")) and agent_connected(),
               "agent_connected": agent_connected(),
               "shift": active_shift(),
               "frame_wh": agent_state.get("frame_wh", (0, 0))})
    return st


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
        shifts.append({"cook": body["cook"], "days": [int(x) for x in body["days"]],
                       "start": body["start"], "end": body["end"]})
    await push_config()
    return shifts


@app.get("/api/events")
def get_events(limit: int = 100):
    return list(events)[-limit:][::-1]


# ---- agent link ------------------------------------------------------------
@app.websocket("/ws/agent")
async def ws_agent(sock: WebSocket):
    global agent_ws, agent_state, agent_seen, preview, settings, shifts, settings_touched
    if sock.headers.get("x-agent-token") != AGENT_TOKEN:
        await sock.close(code=4403)
        return
    await sock.accept()
    agent_ws = sock
    agent_seen = time.time()
    await sock.send_text(json.dumps({"type": "hello", "have_settings": settings_touched}))
    await push_config()
    try:
        while True:
            m = json.loads(await sock.receive_text())
            agent_seen = time.time()
            if m["type"] == "state":
                agent_state = m
                if not m.get("running"):
                    preview = None
            elif m["type"] == "preview":
                preview = base64.b64decode(m["jpg"])
            elif m["type"] == "event":
                events.append(m["event"])
            elif m["type"] == "restore":
                global desired_run
                if not settings_touched and m.get("settings"):
                    settings.update(m["settings"])
                    settings_touched = True
                if not shifts and m.get("shifts"):
                    shifts.extend(m["shifts"])
                if m.get("run"):
                    desired_run = True       # the kitchen was live — stay live
                await push_config()
    except WebSocketDisconnect:
        pass
    finally:
        if agent_ws is sock:
            agent_ws = None
            preview = None


# ---- browsers --------------------------------------------------------------
@app.websocket("/ws")
async def ws_browser(sock: WebSocket):
    if sock.cookies.get("gk") != ADMIN_KEY and sock.query_params.get("k") != ADMIN_KEY:
        await sock.close(code=4403)
        return
    await sock.accept()
    try:
        while True:
            now = time.time()
            snap = dict(agent_state.get("snapshot", {"patties": []}))
            snap.update({"server_ts": round(now, 3),
                         "running": bool(agent_state.get("running")) and agent_connected(),
                         "agent_connected": agent_connected(),
                         "frame_wh": agent_state.get("frame_wh", (0, 0)),
                         "stream": agent_state.get("status", {}).get("stream", "—"),
                         "shift": active_shift(),
                         "targets": {"A": settings["target_a"], "B": settings["target_b"],
                                     "tol_early": settings["tol_early"],
                                     "tol_late": settings["tol_late"]}})
            await sock.send_text(json.dumps(snap))
            await asyncio.sleep(0.2)
    except WebSocketDisconnect:
        pass
