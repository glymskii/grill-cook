"""Accounts, sessions and roles — stdlib crypto only.

Passwords: scrypt with a per-user salt. Sessions: HMAC-signed cookie holding
email/role/expiry. Roles: owner > manager > viewer. The legacy ?k= key keeps
working as an owner-equivalent so the pilot never locks itself out; the ?v=
key stays a tablet-only view credential.
"""
import hashlib
import hmac
import os
import secrets
import time

SECRET = os.environ.get("SESSION_SECRET", "dev-secret").encode()
SESSION_DAYS = 14
ROLE_RANK = {"viewer": 0, "manager": 1, "owner": 2}


def hash_password(password: str) -> str:
    salt = secrets.token_hex(16)
    h = hashlib.scrypt(password.encode(), salt=bytes.fromhex(salt),
                       n=2 ** 14, r=8, p=1).hex()
    return f"scrypt${salt}${h}"


def verify_password(password: str, stored: str) -> bool:
    try:
        _, salt, h = stored.split("$")
        calc = hashlib.scrypt(password.encode(), salt=bytes.fromhex(salt),
                              n=2 ** 14, r=8, p=1).hex()
        return hmac.compare_digest(calc, h)
    except Exception:
        return False


def make_session(email: str, role: str) -> str:
    exp = int(time.time()) + SESSION_DAYS * 86400
    body = f"{email}|{role}|{exp}"
    sig = hmac.new(SECRET, body.encode(), hashlib.sha256).hexdigest()
    return f"{body}|{sig}"


def read_session(cookie: str | None):
    if not cookie:
        return None
    try:
        email, role, exp, sig = cookie.rsplit("|", 3)
        body = f"{email}|{role}|{exp}"
        if not hmac.compare_digest(
                hmac.new(SECRET, body.encode(), hashlib.sha256).hexdigest(), sig):
            return None
        if int(exp) < time.time() or role not in ROLE_RANK:
            return None
        return {"email": email, "role": role}
    except Exception:
        return None


def at_least(role: str | None, needed: str) -> bool:
    return role is not None and ROLE_RANK.get(role, -1) >= ROLE_RANK[needed]
