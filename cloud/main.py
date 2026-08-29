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

import auth
import db
import secrets
from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, HTMLResponse, RedirectResponse, Response

ROOT = Path(__file__).resolve().parent
ADMIN_KEY = os.environ.get("ADMIN_KEY", "dev")
VIEW_KEY = os.environ.get("VIEW_KEY", ADMIN_KEY)   # cook tablet: read-only surfaces
AGENT_TOKEN = os.environ.get("AGENT_TOKEN", "dev")
RELAY_PUBLIC = os.environ.get("RELAY_PUBLIC", "")
LOCATION_NAME = os.environ.get("LOCATION_NAME", "Pilot Kitchen")
OWNER_EMAIL = os.environ.get("OWNER_EMAIL", "").lower()

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
stations: dict = {}                     # name -> {"seen": ts}; "main" drives the UI
preview: bytes | None = None
still: bytes | None = None       # survives pipeline stops: the ROI editor needs a frame

app = FastAPI()


# ---- auth ------------------------------------------------------------------
VIEW_PATHS = {"/hud", "/preview.jpg"}
OWNER_ONLY = ("/api/users",)
MUTATION_MIN_ROLE = "manager"


def actor_of(request: Request) -> dict | None:
    """Who is calling: {'email', 'role'} with role in viewer/manager/owner."""
    sess = auth.read_session(request.cookies.get("gs"))
    if sess:
        return sess
    if (request.query_params.get("k") == ADMIN_KEY
            or request.cookies.get("gk") == ADMIN_KEY):
        return {"email": "legacy-key", "role": "owner"}
    if (request.query_params.get("v") == VIEW_KEY
            or request.cookies.get("gv") == VIEW_KEY):
        return {"email": "tablet", "role": "tablet"}
    return None


@app.middleware("http")
async def gate(request: Request, call_next):
    path = request.url.path
    if path in ("/health", "/login", "/api/login"):
        return await call_next(request)
    actor = actor_of(request)
    if actor is None:
        if path == "/admin" or path == "/":
            return RedirectResponse("/login")
        return HTMLResponse("<h3>Sign in required.</h3>", status_code=401)
    role = actor["role"]
    if role == "tablet" and path not in VIEW_PATHS:
        return HTMLResponse("<h3>This link opens the cook display only.</h3>",
                            status_code=403)
    if role == "viewer" and request.method != "GET" and path.startswith("/api/"):
        return HTMLResponse("viewer role is read-only", status_code=403)
    if any(path.startswith(p) for p in OWNER_ONLY) and role != "owner":
        return HTMLResponse("owner only", status_code=403)
    request.state.actor = actor
    resp = await call_next(request)
    if request.query_params.get("k") == ADMIN_KEY:
        resp.set_cookie("gk", ADMIN_KEY, max_age=7 * 86400, httponly=False)
    if request.query_params.get("v") == VIEW_KEY:
        resp.set_cookie("gv", VIEW_KEY, max_age=30 * 86400, httponly=False)
    return resp


def actor_email(request: Request) -> str:
    return getattr(request.state, "actor", {"email": "?"})["email"]


@app.get("/login")
def login_page():
    return FileResponse(ROOT / "static/login.html")


@app.post("/api/login")
async def api_login(request: Request, body: dict):
    email = (body.get("email") or "").strip().lower()
    password = body.get("password") or ""
    u = await db.get_user(email)
    if not u or not auth.verify_password(password, u["pass_hash"]):
        await db.audit(email or "?", "login.failed")
        return HTMLResponse("invalid credentials", status_code=401)
    resp = Response(json.dumps({"ok": True, "role": u["role"]}),
                    media_type="application/json")
    resp.set_cookie("gs", auth.make_session(email, u["role"]),
                    max_age=auth.SESSION_DAYS * 86400, httponly=True)
    await db.audit(email, "login.ok")
    return resp


@app.post("/api/logout")
async def api_logout():
    resp = Response(json.dumps({"ok": True}), media_type="application/json")
    resp.delete_cookie("gs")
    return resp


@app.get("/api/me")
async def api_me(request: Request):
    return getattr(request.state, "actor", None) or {}


@app.get("/api/users")
async def api_users():
    return await db.list_users()


@app.post("/api/users")
async def api_users_post(request: Request, body: dict):
    action = body.get("action", "create")
    email = (body.get("email") or "").strip().lower()
    if action == "delete":
        if email == OWNER_EMAIL:
            return HTMLResponse("cannot delete the owner", status_code=400)
        await db.delete_user(email)
        await db.audit(actor_email(request), "user.delete", {"email": email})
        return {"ok": True}
    if action == "reset":
        temp = secrets.token_urlsafe(9)
        await db.set_password(email, auth.hash_password(temp))
        await db.audit(actor_email(request), "user.reset", {"email": email})
        return {"ok": True, "temp_password": temp}
    role = body.get("role", "viewer")
    if role not in ("viewer", "manager", "owner"):
        return HTMLResponse("bad role", status_code=400)
    temp = secrets.token_urlsafe(9)
    try:
        await db.create_user(email, body.get("name", ""), role, auth.hash_password(temp))
    except Exception:
        return HTMLResponse("user exists", status_code=400)
    await db.audit(actor_email(request), "user.create", {"email": email, "role": role})
    return {"ok": True, "temp_password": temp}


history: deque = deque(maxlen=2880)          # 24h of 30s health samples
alerts: deque = deque(maxlen=200)
clip_cache: dict = {}
clip_waiters: dict = {}


@app.get("/api/clip/{name}")
async def get_clip(name: str):
    name = name.split("/")[-1]
    if name in clip_cache:
        return Response(clip_cache[name], media_type="video/mp4")
    if agent_ws is None:
        return Response(status_code=503)
    ev = clip_waiters.setdefault(name, asyncio.Event())
    try:
        await agent_ws.send_text(json.dumps({"type": "get_clip", "name": name}))
    except Exception:
        return Response(status_code=503)
    try:
        await asyncio.wait_for(ev.wait(), timeout=12)
    except asyncio.TimeoutError:
        clip_waiters.pop(name, None)
        return Response(status_code=404)
    clip_waiters.pop(name, None)
    data = clip_cache.get(name)
    if not data:
        return Response(status_code=404)
    return Response(data, media_type="video/mp4")
_alert_state: dict = {}


async def fire_alert(kind: str, level: str, message: str):
    ev = {"ts": time.time(), "kind": kind, "level": level, "message": message,
          "ack": False}
    alerts.append(ev)
    if db.pool():
        await db.pool().execute(
            "INSERT INTO alerts (ts,kind,level,message) VALUES ($1,$2,$3,$4)",
            ev["ts"], kind, level, message)
    cfg = await db.kv_get("alert_cfg") or {}
    tok, chat = cfg.get("telegram_token"), cfg.get("telegram_chat")
    if tok and chat:
        import urllib.parse
        import urllib.request

        def _send():
            try:
                data = urllib.parse.urlencode(
                    {"chat_id": chat,
                     "text": f"🍔 Grill Cook · {LOCATION_NAME}\n{message}"}).encode()
                urllib.request.urlopen(
                    f"https://api.telegram.org/bot{tok}/sendMessage",
                    data=data, timeout=10)
            except Exception as e:
                print("telegram send failed:", e)
        await asyncio.get_event_loop().run_in_executor(None, _send)


async def _alert_watcher():
    prev_shift = None
    while True:
        try:
            cfg = await db.kv_get("alert_cfg") or {}
            now = time.time()
            st = agent_state.get("status", {})

            def edge_bad():
                return desired_run and not agent_connected()

            def cam_bad():
                return (desired_run and agent_connected()
                        and bool(agent_state.get("running"))
                        and st.get("stream") != "ok")

            for kind, cond, grace, level, msg in (
                    ("edge-offline", edge_bad, 120, "bad",
                     "Edge worker is offline while the station is running."),
                    ("camera-offline", cam_bad, 300, "bad",
                     "No camera stream for 5 minutes while running.")):
                stt = _alert_state.setdefault(kind, {"since": None, "fired": False})
                if cond():
                    stt["since"] = stt["since"] or now
                    if not stt["fired"] and now - stt["since"] > grace:
                        stt["fired"] = True
                        await fire_alert(kind, level, msg)
                else:
                    if stt["fired"]:
                        await fire_alert(kind, "ok", kind.replace("-", " ") + " recovered.")
                    _alert_state[kind] = {"since": None, "fired": False}

            cur = active_shift()
            if prev_shift and (not cur or cur.get("cook") != prev_shift.get("cook")):
                lt = time.localtime(now)
                day0 = now - (lt.tm_hour * 3600 + lt.tm_min * 60 + lt.tm_sec)
                done = [e for e in events
                        if e["type"] == "removed" and e["ts"] >= day0
                        and cook_for_ts(e["ts"]) == prev_shift["cook"]]
                if done:
                    ok_n = 0
                    for e in done:
                        ta_, tb_ = e.get("ta", settings["target_a"]), e.get("tb", settings["target_b"])
                        te_, tl_ = e.get("te", settings["tol_early"]), e.get("tl", settings["tol_late"])
                        if not (e["side_a"] < ta_ - te_ or e["side_b"] < tb_ - te_
                                or e["side_a"] > ta_ + tl_ or e["side_b"] > tb_ + tl_):
                            ok_n += 1
                    pct = 100 * ok_n / len(done)
                    thr = cfg.get("compliance_min", 60)
                    if pct < thr:
                        await fire_alert("low-compliance", "warn",
                                         f"Shift of {prev_shift['cook']} ended at "
                                         f"{pct:.0f}% compliance ({len(done)} patties, "
                                         f"threshold {thr}%).")
            prev_shift = cur
        except Exception as e:
            print("alert watcher error:", e)
        await asyncio.sleep(30)


@app.get("/api/alerts")
def get_alerts():
    return {"alerts": list(alerts)[-50:][::-1],
            "unack": sum(1 for a in alerts if not a["ack"] and a["level"] != "ok")}


@app.post("/api/alerts")
async def post_alerts(body: dict):
    if body.get("action") == "ack_all":
        for a in alerts:
            a["ack"] = True
    return {"ok": True}


@app.get("/api/alert_cfg")
async def get_alert_cfg():
    cfg = await db.kv_get("alert_cfg") or {}
    return {"telegram_chat": cfg.get("telegram_chat", ""),
            "compliance_min": cfg.get("compliance_min", 60),
            "telegram_token_set": bool(cfg.get("telegram_token"))}


@app.post("/api/alert_cfg")
async def post_alert_cfg(request: Request, body: dict):
    cfg = await db.kv_get("alert_cfg") or {}
    for k in ("telegram_token", "telegram_chat", "compliance_min"):
        if k in body and body[k] != "":
            cfg[k] = body[k]
    await db.kv_set("alert_cfg", cfg)
    await db.audit(actor_email(request), "alert_cfg.update",
                   {k: v for k, v in body.items() if k != "telegram_token"})
    return {"ok": True}


@app.post("/api/alert_test")
async def alert_test(request: Request):
    await fire_alert("test", "warn", f"Test alert requested by {actor_email(request)}.")
    return {"ok": True}


@app.on_event("startup")
async def _init_db():
    global settings_touched, desired_run
    try:
        ok = await db.connect()
    except Exception as e:
        print("db connect failed, memory mode:", e)
        ok = False
    if not ok:
        return
    for ev in await db.load_events():
        events.append(ev)
    saved = await db.kv_get("settings")
    if saved:
        settings.update({k: saved[k] for k in DEFAULTS if k in saved})
        settings_touched = True
    saved_shifts = await db.kv_get("shifts")
    if saved_shifts is not None:
        shifts.clear()
        shifts.extend(saved_shifts)
    run = await db.kv_get("desired_run")
    if run is not None:
        desired_run = bool(run)
    if OWNER_EMAIL and await db.count_users() == 0:
        await db.create_user(OWNER_EMAIL, "Owner", "owner",
                             auth.hash_password(ADMIN_KEY))
        print("owner seeded:", OWNER_EMAIL)
    print(f"db ready: {len(events)} events restored")


@app.on_event("startup")
async def _start_alert_watcher():
    asyncio.get_event_loop().create_task(_alert_watcher())


@app.on_event("startup")
async def _start_health_sampler():
    async def _health_sampler():
        while True:
            st = agent_state.get("status", {})
            history.append({"t": int(time.time()), "a": agent_connected(),
                            "s": bool(agent_state.get("running")) and st.get("stream") == "ok",
                            "r": bool(agent_state.get("running"))})
            await asyncio.sleep(30)
    asyncio.get_event_loop().create_task(_health_sampler())


@app.get("/api/stations")
def get_stations():
    now = time.time()
    return [{"name": n, "online": now - v.get("seen", 0) < 6}
            for n, v in sorted(stations.items())]


@app.get("/api/history")
def get_history():
    return list(history)


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
    return {**settings, "_relay_public": RELAY_PUBLIC, "_location": LOCATION_NAME,
            "_view_key": VIEW_KEY}


@app.post("/api/settings")
async def post_settings(request: Request, body: dict):
    global settings_touched
    settings.update({k: body[k] for k in DEFAULTS if k in body})
    settings_touched = True
    await db.kv_set("settings", {k: settings[k] for k in DEFAULTS})
    await db.audit(actor_email(request), "settings.update", body)
    await push_config()
    return settings


TARGET_KEYS = ("target_a", "target_b", "tol_early", "tol_late")


@app.get("/api/skus")
async def get_skus():
    skus = await db.kv_get("skus") or []
    return {"skus": skus, "active": await db.kv_get("active_sku")}


@app.post("/api/skus")
async def post_skus(request: Request, body: dict):
    skus = await db.kv_get("skus") or []
    action = body.get("action")
    if action == "save":
        sku = body["sku"]
        sku.setdefault("id", secrets.token_hex(4))
        skus = [x for x in skus if x["id"] != sku["id"]] + [sku]
    elif action == "delete":
        skus = [x for x in skus if x["id"] != body["id"]]
    elif action == "activate":
        sku = next((x for x in skus if x["id"] == body["id"]), None)
        if not sku:
            return HTMLResponse("no such sku", status_code=404)
        settings.update({k: sku[k] for k in TARGET_KEYS})
        await db.kv_set("settings", {k: settings[k] for k in DEFAULTS})
        await db.kv_set("active_sku", sku["id"])
        await push_config()
    await db.kv_set("skus", skus)
    await db.audit(actor_email(request), f"sku.{action}", body)
    return {"skus": skus, "active": await db.kv_get("active_sku")}


@app.get("/api/recommend")
def api_recommend(d: float, h: float, tg: float, tf: float):
    from recommend import recommend
    return recommend(d, h, tg, tf)


@app.post("/api/pipeline/start")
async def start_pipeline(request: Request):
    global desired_run
    desired_run = True
    await db.kv_set("desired_run", True)
    await db.audit(actor_email(request), "pipeline.start")
    await push_config()
    return {"running": True}


@app.post("/api/pipeline/stop")
async def stop_pipeline(request: Request):
    global desired_run
    desired_run = False
    await db.kv_set("desired_run", False)
    await db.audit(actor_email(request), "pipeline.stop")
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
async def post_shift(request: Request, body: dict):
    if body.get("delete") is not None:
        idx = int(body["delete"])
        if 0 <= idx < len(shifts):
            shifts.pop(idx)
    else:
        shifts.append({"cook": body["cook"], "days": [int(x) for x in body["days"]],
                       "start": body["start"], "end": body["end"]})
    await db.kv_set("shifts", shifts)
    await db.audit(actor_email(request), "shifts.update", body)
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

    day_rows = []
    for back in range(6, -1, -1):
        d0 = day_start - back * 86400
        d1 = d0 + 86400
        dd = [e for e in events if d0 <= e["ts"] < d1 and e["type"] == "removed"]
        ok_n = sum(1 for e in dd if verdict(e) == "ok")
        day_rows.append({"label": time.strftime("%a %d", time.localtime(d0)),
                         "done": len(dd),
                         "ok_pct": round(100 * ok_n / len(dd), 0) if dd else None})

    return {
        "period": period, "done": len(done), "days": day_rows,
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


@app.get("/api/audit")
async def get_audit():
    return await db.audit_tail(100)


@app.get("/api/events.csv")
def events_csv(period: str = "all"):
    now = time.time()
    lt = time.localtime(now)
    day_start = now - (lt.tm_hour * 3600 + lt.tm_min * 60 + lt.tm_sec)
    since = {"today": day_start, "7d": now - 7 * 86400, "all": 0.0}.get(period, 0.0)
    lines = ["time,type,patty,side,grade,side_a_s,side_b_s,target_a,target_b,cook"]
    for e in events:
        if e["ts"] < since:
            continue
        lines.append(",".join(str(x) for x in (
            time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(e["ts"])),
            e.get("type", ""), e.get("pid", ""), e.get("side", ""),
            e.get("grade", ""), e.get("side_a", ""), e.get("side_b", ""),
            e.get("ta", ""), e.get("tb", ""), cook_for_ts(e["ts"]))))
    return Response("\n".join(lines), media_type="text/csv",
                    headers={"Content-Disposition": "attachment; filename=grill-events.csv"})


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
    station = sock.headers.get("x-station", "main")
    await sock.accept()
    stations[station] = {"seen": time.time()}
    if agent_ws is not None:
        try:
            await agent_ws.close(code=4409)   # a newer edge instance took over
        except Exception:
            pass
    agent_ws = sock
    agent_seen = time.time()
    await sock.send_text(json.dumps({"type": "hello", "have_settings": settings_touched,
                                     "have_shifts": len(shifts) > 0,
                                     "have_events": len(events) > 0}))
    await push_config()
    try:
        while True:
            m = json.loads(await sock.receive_text())
            if agent_ws is not sock:
                break                          # superseded: drop stale sender
            agent_seen = time.time()
            stations.setdefault(station, {})["seen"] = agent_seen
            if m["type"] == "state":
                agent_state = m
                if not m.get("running"):
                    preview = None
            elif m["type"] == "preview":
                preview = base64.b64decode(m["jpg"])
                still = preview
            elif m["type"] == "event":
                events.append(m["event"])
                asyncio.create_task(db.insert_event(m["event"]))
            elif m["type"] == "clip":
                clip_cache[m["name"]] = base64.b64decode(m["b64"])
                while len(clip_cache) > 20:
                    clip_cache.pop(next(iter(clip_cache)))
                if m["name"] in clip_waiters:
                    clip_waiters[m["name"]].set()
            elif m["type"] == "clip_missing":
                if m["name"] in clip_waiters:
                    clip_waiters[m["name"]].set()
            elif m["type"] == "backfill":
                seen = {(e["ts"], e["type"], e.get("pid")) for e in events}
                fresh = [ev for ev in m["events"]
                         if (ev["ts"], ev["type"], ev.get("pid")) not in seen]
                if fresh:
                    merged = sorted(list(events) + fresh, key=lambda e: e["ts"])
                    events.clear()
                    events.extend(merged)
                    for ev in fresh:
                        asyncio.create_task(db.insert_event(ev))
            elif m["type"] == "restore":
                global desired_run
                if not settings_touched and m.get("settings"):
                    settings.update(m["settings"])
                    settings_touched = True
                if not shifts and m.get("shifts"):
                    shifts.extend(m["shifts"])
                    await db.kv_set("shifts", shifts)
                if m.get("run"):
                    desired_run = True       # the kitchen was live — stay live
                await db.kv_set("settings", {k: settings[k] for k in DEFAULTS})
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
