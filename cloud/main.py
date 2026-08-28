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
    "roi": [],
}

settings = dict(DEFAULTS)
settings_touched = False
shifts: list = []
events: deque = deque(maxlen=20000)
desired_run = False
agent_ws: WebSocket | None = None
agent_state: dict = {}
agent_seen = 0.0
preview: bytes | None = None
still: bytes | None = None       # survives pipeline stops: the ROI editor needs a frame

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


@app.get("/still.jpg")
def still_jpg():
    if still:
        return Response(still, media_type="image/jpeg",
                        headers={"Cache-Control": "no-store"})
    return Response(status_code=404)


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


def cook_for_ts(ts: float):
    lt = time.localtime(ts)
    hhmm = time.strftime("%H:%M", lt)
    for s in shifts:
        if lt.tm_wday in s.get("days", []) and s["start"] <= hhmm <= s["end"]:
            return s["cook"]
    return "—"


@app.get("/api/analytics")
def analytics(period: str = "today"):
    now = time.time()
    lt = time.localtime(now)
    day_start = now - (lt.tm_hour * 3600 + lt.tm_min * 60 + lt.tm_sec)
    since = {"today": day_start, "7d": now - 7 * 86400, "all": 0.0}.get(period, day_start)
    evs = [e for e in events if e["ts"] >= since]

    ta, tb = settings["target_a"], settings["target_b"]
    tol_e, tol_l = settings["tol_early"], settings["tol_late"]

    def verdict(e):
        # judge each patty by the targets that were in force when it cooked;
        # older events without stamps fall back to the current settings
        a, b = e.get("ta", ta), e.get("tb", tb)
        te_, tl_ = e.get("te", tol_e), e.get("tl", tol_l)
        if e["side_a"] < a - te_ or e["side_b"] < b - te_:
            return "under"
        if e["side_a"] > a + tl_ or e["side_b"] > b + tl_:
            return "over"
        return "ok"

    done = [e for e in evs if e["type"] == "removed"]
    flips = [e for e in evs if e["type"] == "flip"]

    def stats(vals):
        if not vals:
            return None
        n = len(vals)
        mean = sum(vals) / n
        sd = (sum((v - mean) ** 2 for v in vals) / n) ** 0.5
        return {"avg": round(mean, 1), "sd": round(sd, 1)}

    verdicts = {"ok": 0, "under": 0, "over": 0}
    for e in done:
        verdicts[verdict(e)] += 1

    hours = [{"n": 0, "ok": 0} for _ in range(24)]
    for e in done:
        h = time.localtime(e["ts"]).tm_hour
        hours[h]["n"] += 1
        hours[h]["ok"] += verdict(e) == "ok"

    cooks: dict = {}
    for e in done:
        c = cooks.setdefault(cook_for_ts(e["ts"]),
                             {"done": 0, "ok": 0, "a": [], "b": []})
        c["done"] += 1
        c["ok"] += verdict(e) == "ok"
        c["a"].append(e["side_a"]); c["b"].append(e["side_b"])
    grades = {"optimal": 0, "early": 0, "late": 0}
    for f in flips:
        if f.get("grade") in grades:
            grades[f["grade"]] += 1

    return {
        "period": period, "done": len(done),
        "compliance": round(100 * verdicts["ok"] / len(done), 1) if done else None,
        "verdicts": verdicts,
        "side_a": stats([e["side_a"] for e in done]),
        "side_b": stats([e["side_b"] for e in done]),
        "total": stats([e["side_a"] + e["side_b"] for e in done]),
        "flips": {"n": len(flips), **grades,
                  "on_time_pct": round(100 * grades["optimal"] / len(flips), 1) if flips else None},
        "resets": sum(1 for e in evs if e["type"] == "scene_cut"),
        "invalidated": sum(1 for e in evs if e["type"] == "invalidated"),
        "hours": hours,
        "cooks": [{"cook": k, "done": v["done"],
                   "compliance": round(100 * v["ok"] / v["done"], 1),
                   "avg_a": round(sum(v["a"]) / len(v["a"]), 1),
                   "avg_b": round(sum(v["b"]) / len(v["b"]), 1)}
                  for k, v in sorted(cooks.items(), key=lambda x: -x[1]["done"])],
        "targets": {"a": ta, "b": tb, "tol_early": tol_e, "tol_late": tol_l},
    }


@app.get("/api/events")
def get_events(limit: int = 100):
    return list(events)[-limit:][::-1]


# ---- agent link ------------------------------------------------------------
@app.websocket("/ws/agent")
async def ws_agent(sock: WebSocket):
    global agent_ws, agent_state, agent_seen, preview, still, settings, shifts, settings_touched
    if sock.headers.get("x-agent-token") != AGENT_TOKEN:
        await sock.close(code=4403)
        return
    await sock.accept()
    if agent_ws is not None:
        try:
            await agent_ws.close(code=4409)   # a newer edge instance took over
        except Exception:
            pass
    agent_ws = sock
    agent_seen = time.time()
    await sock.send_text(json.dumps({"type": "hello", "have_settings": settings_touched,
                                     "have_events": len(events) > 0}))
    await push_config()
    try:
        while True:
            m = json.loads(await sock.receive_text())
            if agent_ws is not sock:
                break                          # superseded: drop stale sender
            agent_seen = time.time()
            if m["type"] == "state":
                agent_state = m
                if not m.get("running"):
                    preview = None
            elif m["type"] == "preview":
                preview = base64.b64decode(m["jpg"])
                still = preview
            elif m["type"] == "event":
                events.append(m["event"])
            elif m["type"] == "backfill":
                seen = {(e["ts"], e["type"], e.get("pid")) for e in events}
                fresh = [ev for ev in m["events"]
                         if (ev["ts"], ev["type"], ev.get("pid")) not in seen]
                if fresh:
                    merged = sorted(list(events) + fresh, key=lambda e: e["ts"])
                    events.clear()
                    events.extend(merged)
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
