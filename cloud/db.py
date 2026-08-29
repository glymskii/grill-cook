"""Persistence for the hub: Postgres when DATABASE_URL is set, memory otherwise.

The hub stays fully functional without a database (pilot fallback) — every
caller treats this module as best-effort durability, not as the hot path.
In-memory mirrors remain the source for request handling; the DB makes them
survive redeploys and outgrow the 20k-event deque.
"""
import json
import os

import asyncpg

_pool: asyncpg.Pool | None = None

SCHEMA = """
CREATE TABLE IF NOT EXISTS events (
  id bigserial PRIMARY KEY,
  ts double precision NOT NULL,
  type text NOT NULL,
  pid int,
  side text, grade text,
  side_a real, side_b real, ta real, tb real, te real, tl real,
  extra jsonb,
  station text NOT NULL DEFAULT 'main',
  UNIQUE (ts, type, pid)
);
CREATE INDEX IF NOT EXISTS events_ts ON events (ts);
CREATE TABLE IF NOT EXISTS kv (key text PRIMARY KEY, value jsonb NOT NULL);
CREATE TABLE IF NOT EXISTS shifts (
  id serial PRIMARY KEY, station text NOT NULL DEFAULT 'main',
  cook text NOT NULL, days int[] NOT NULL, start_t text NOT NULL, end_t text NOT NULL
);
CREATE TABLE IF NOT EXISTS users (
  id serial PRIMARY KEY, email text UNIQUE NOT NULL, name text NOT NULL DEFAULT '',
  role text NOT NULL, pass_hash text NOT NULL, created_at double precision NOT NULL
);
CREATE TABLE IF NOT EXISTS audit (
  id serial PRIMARY KEY, ts double precision NOT NULL,
  user_email text NOT NULL, action text NOT NULL, payload jsonb
);
CREATE TABLE IF NOT EXISTS skus (
  id serial PRIMARY KEY, station text NOT NULL DEFAULT 'main', name text NOT NULL,
  target_a real NOT NULL, target_b real NOT NULL,
  tol_early real NOT NULL, tol_late real NOT NULL, active bool NOT NULL DEFAULT false
);
CREATE TABLE IF NOT EXISTS alerts (
  id serial PRIMARY KEY, ts double precision NOT NULL, kind text NOT NULL,
  level text NOT NULL, message text NOT NULL, ack bool NOT NULL DEFAULT false
);
"""


async def connect() -> bool:
    global _pool
    url = os.environ.get("DATABASE_URL")
    if not url:
        return False
    _pool = await asyncpg.create_pool(url, min_size=1, max_size=4)
    async with _pool.acquire() as c:
        await c.execute(SCHEMA)
    return True


def ready() -> bool:
    return _pool is not None


async def insert_event(ev: dict):
    if not _pool:
        return
    known = {"ts", "type", "pid", "side", "grade", "side_a", "side_b",
             "ta", "tb", "te", "tl"}
    extra = {k: v for k, v in ev.items() if k not in known}
    await _pool.execute(
        """INSERT INTO events (ts,type,pid,side,grade,side_a,side_b,ta,tb,te,tl,extra)
           VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12)
           ON CONFLICT (ts,type,pid) DO NOTHING""",
        ev["ts"], ev["type"], ev.get("pid"), ev.get("side"), ev.get("grade"),
        ev.get("side_a"), ev.get("side_b"), ev.get("ta"), ev.get("tb"),
        ev.get("te"), ev.get("tl"), json.dumps(extra) if extra else None)


async def load_events(limit=50000) -> list[dict]:
    if not _pool:
        return []
    rows = await _pool.fetch(
        "SELECT * FROM (SELECT * FROM events ORDER BY ts DESC LIMIT $1) s ORDER BY ts",
        limit)
    out = []
    for r in rows:
        ev = {k: r[k] for k in ("ts", "type", "pid", "side", "grade", "side_a",
                                "side_b", "ta", "tb", "te", "tl") if r[k] is not None}
        if r["extra"]:
            ev.update(json.loads(r["extra"]))
        out.append(ev)
    return out


async def kv_set(key: str, value):
    if _pool:
        await _pool.execute(
            "INSERT INTO kv (key,value) VALUES ($1,$2) "
            "ON CONFLICT (key) DO UPDATE SET value=$2", key, json.dumps(value))


async def kv_get(key: str):
    if not _pool:
        return None
    v = await _pool.fetchval("SELECT value FROM kv WHERE key=$1", key)
    return json.loads(v) if v else None


async def audit(user: str, action: str, payload=None):
    if _pool:
        import time
        await _pool.execute(
            "INSERT INTO audit (ts,user_email,action,payload) VALUES ($1,$2,$3,$4)",
            time.time(), user, action, json.dumps(payload) if payload else None)


async def audit_tail(limit=100):
    if not _pool:
        return []
    rows = await _pool.fetch("SELECT ts,user_email,action,payload FROM audit "
                             "ORDER BY id DESC LIMIT $1", limit)
    return [{"ts": r["ts"], "user": r["user_email"], "action": r["action"],
             "payload": json.loads(r["payload"]) if r["payload"] else None} for r in rows]


def pool():
    return _pool


# ---- users ------------------------------------------------------------------
async def create_user(email: str, name: str, role: str, pass_hash: str):
    import time
    if _pool:
        await _pool.execute(
            "INSERT INTO users (email,name,role,pass_hash,created_at) "
            "VALUES ($1,$2,$3,$4,$5)", email.lower(), name, role, pass_hash, time.time())


async def get_user(email: str):
    if not _pool:
        return None
    r = await _pool.fetchrow("SELECT * FROM users WHERE email=$1", email.lower())
    return dict(r) if r else None


async def list_users():
    if not _pool:
        return []
    rows = await _pool.fetch("SELECT email,name,role,created_at FROM users ORDER BY id")
    return [dict(r) for r in rows]


async def delete_user(email: str):
    if _pool:
        await _pool.execute("DELETE FROM users WHERE email=$1", email.lower())


async def set_password(email: str, pass_hash: str):
    if _pool:
        await _pool.execute("UPDATE users SET pass_hash=$2 WHERE email=$1",
                            email.lower(), pass_hash)


async def count_users() -> int:
    if not _pool:
        return 0
    return await _pool.fetchval("SELECT count(*) FROM users")
