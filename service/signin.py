"""Passwordless sign-in for the pi plugin.

Self-contained: yaeger-pi mints its own tokens in its own table and exchanges
them for Supabase sessions directly, so it no longer depends on the
download-gate's DMG onboarding flow or its email template.

Flow:
    POST /v1/auth/email    {email}  -> emails a sign-in link, returns nothing useful
    POST /v1/auth/resolve  {token}  -> claims the token, returns a Supabase session

The session-minting technique mirrors the download-gate: create-or-rotate the
user's password to a value we generate, immediately trade it for a session, and
never reveal it. Each token rotates the password, so a leaked link is the only
credential and it is single-use.
"""

from __future__ import annotations

import os
import secrets
import smtplib
import ssl
from datetime import datetime, timedelta, timezone
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

import httpx
from fastapi import HTTPException

SUPABASE_URL = os.environ.get("YAEGER_SUPABASE_URL", "").rstrip("/")
SUPABASE_SERVICE_ROLE = os.environ.get("YAEGER_SUPABASE_SERVICE_ROLE_KEY", "")
SUPABASE_ANON_KEY = os.environ.get("YAEGER_SUPABASE_ANON_KEY", "")

SMTP_HOST = os.environ.get("YAEGER_SMTP_HOST", "")
SMTP_PORT = int(os.environ.get("YAEGER_SMTP_PORT", "465"))
SMTP_USER = os.environ.get("YAEGER_SMTP_USER", "")
SMTP_FROM = os.environ.get("YAEGER_SMTP_FROM", SMTP_USER)
SECRET_KEY_PATH = os.environ.get("YAEGER_SECRET_KEY_PATH", "/etc/yaeger-secret.key")

TOKEN_TTL = timedelta(minutes=30)  # short: this is a terminal paste, not a saved link


def _smtp_password() -> str:
    """Decrypt the SMTP password with the same Fernet key the download-gate uses."""
    from cryptography.fernet import Fernet

    enc = os.environ.get("YAEGER_SMTP_PASSWORD_ENC")
    if not enc:
        raise HTTPException(500, "SMTP password not configured")
    with open(SECRET_KEY_PATH, "rb") as kf:
        return Fernet(kf.read().strip()).decrypt(enc.encode()).decode()


def render_signin_email(to_email: str, token: str) -> MIMEMultipart:
    """A sign-in email for the plugin: no DMG, no product pitch, just the token."""
    msg = MIMEMultipart("alternative")
    msg["Subject"] = "Your yaeger-pi sign-in code"
    msg["From"] = SMTP_FROM
    msg["To"] = to_email
    msg["Reply-To"] = SMTP_USER

    minutes = int(TOKEN_TTL.total_seconds() // 60)

    text = f"""\
Sign in to yaeger-pi

Paste this code back into your terminal:

    {token}

It expires in {minutes} minutes and can only be used once.

If you did not ask to sign in, ignore this email - nothing has happened.
"""

    html = f"""\
<html><body style="margin:0;padding:0;background:#0d0f15;">
  <div style="max-width:520px;margin:0 auto;padding:40px 28px;
              font-family:-apple-system,Segoe UI,Roboto,sans-serif;color:#e9ebf2;">
    <p style="font:500 12px/1 ui-monospace,SFMono-Regular,Menlo,monospace;
              letter-spacing:.14em;text-transform:uppercase;color:#767d90;margin:0 0 20px;">
      yaeger-pi
    </p>
    <h1 style="font-size:26px;line-height:1.2;letter-spacing:-.02em;margin:0 0 12px;">
      Sign in to yaeger-pi
    </h1>
    <p style="font-size:15px;line-height:1.6;color:#a7aec0;margin:0 0 26px;">
      Paste this code back into your terminal.
    </p>
    <div style="background:#161923;border:1px solid #2a2f3d;border-radius:8px;
                padding:18px 20px;margin:0 0 22px;">
      <code style="font:600 16px/1.4 ui-monospace,SFMono-Regular,Menlo,monospace;
                   color:#8a88f5;word-break:break-all;">{token}</code>
    </div>
    <p style="font-size:13px;line-height:1.6;color:#767d90;margin:0;">
      Expires in {minutes} minutes, single use.
      If you did not ask to sign in, ignore this email &mdash; nothing has happened.
    </p>
  </div>
</body></html>
"""
    msg.attach(MIMEText(text, "plain"))
    msg.attach(MIMEText(html, "html"))
    return msg


def send_email(to_email: str, msg: MIMEMultipart) -> None:
    password = _smtp_password()
    ctx = ssl.create_default_context()
    if SMTP_PORT == 465:
        with smtplib.SMTP_SSL(SMTP_HOST, SMTP_PORT, context=ctx, timeout=20) as s:
            s.login(SMTP_USER, password)
            s.send_message(msg, from_addr=SMTP_USER, to_addrs=[to_email])
    else:
        with smtplib.SMTP(SMTP_HOST, SMTP_PORT, timeout=20) as s:
            s.starttls(context=ctx)
            s.login(SMTP_USER, password)
            s.send_message(msg, from_addr=SMTP_USER, to_addrs=[to_email])


def new_token() -> tuple[str, datetime]:
    return secrets.token_urlsafe(24), datetime.now(timezone.utc) + TOKEN_TTL


async def mint_session(email: str) -> dict:
    """Create or rotate the Supabase user, then trade for a real session."""
    if not (SUPABASE_URL and SUPABASE_SERVICE_ROLE and SUPABASE_ANON_KEY):
        raise HTTPException(500, "Supabase admin credentials not configured")

    admin_headers = {
        "Authorization": f"Bearer {SUPABASE_SERVICE_ROLE}",
        "apikey": SUPABASE_SERVICE_ROLE,
        "Content-Type": "application/json",
    }
    # Used only for the immediate grant below; the user never sees or types it.
    one_shot = secrets.token_urlsafe(32)

    async with httpx.AsyncClient(timeout=20.0) as c:
        # Look the user up through the admin API rather than a direct DB query,
        # which keeps this service free of Postgres credentials.
        found = await c.get(
            f"{SUPABASE_URL}/auth/v1/admin/users",
            headers=admin_headers,
            params={"filter": email},
        )
        user_id = None
        if found.status_code == 200:
            for u in found.json().get("users", []):
                if (u.get("email") or "").lower() == email:
                    user_id = u["id"]
                    break

        if user_id:
            r = await c.put(
                f"{SUPABASE_URL}/auth/v1/admin/users/{user_id}",
                headers=admin_headers,
                json={"password": one_shot},
            )
            if r.status_code not in (200, 201):
                raise HTTPException(502, f"password rotate failed ({r.status_code})")
            registered = True
        else:
            r = await c.post(
                f"{SUPABASE_URL}/auth/v1/admin/users",
                headers=admin_headers,
                json={"email": email, "email_confirm": True, "password": one_shot},
            )
            if r.status_code not in (200, 201):
                raise HTTPException(502, f"create-user failed ({r.status_code}): {r.text[:200]}")
            registered = False

        grant = await c.post(
            f"{SUPABASE_URL}/auth/v1/token?grant_type=password",
            headers={"apikey": SUPABASE_ANON_KEY, "Content-Type": "application/json"},
            json={"email": email, "password": one_shot},
        )
        if grant.status_code != 200:
            raise HTTPException(502, f"token grant failed ({grant.status_code})")
        session = grant.json()

    return {
        "email": email,
        "registered": registered,
        "access_token": session["access_token"],
        "refresh_token": session.get("refresh_token"),
        "expires_in": session.get("expires_in", 3600),
    }


async def refresh_session(refresh_token: str) -> dict:
    """Exchange a refresh token for a new access token.

    Lives server-side so the plugin never needs the Supabase anon key.
    """
    if not (SUPABASE_URL and SUPABASE_ANON_KEY):
        raise HTTPException(500, "Supabase not configured")
    async with httpx.AsyncClient(timeout=20.0) as c:
        r = await c.post(
            f"{SUPABASE_URL}/auth/v1/token?grant_type=refresh_token",
            headers={"apikey": SUPABASE_ANON_KEY, "Content-Type": "application/json"},
            json={"refresh_token": refresh_token},
        )
        if r.status_code != 200:
            raise HTTPException(401, "refresh token rejected - sign in again")
        s = r.json()
    return {
        "access_token": s["access_token"],
        "refresh_token": s.get("refresh_token", refresh_token),
        "expires_in": s.get("expires_in", 3600),
    }
