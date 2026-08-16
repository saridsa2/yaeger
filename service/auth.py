"""Supabase JWT verification.

Deliberately mirrors /opt/yaeger-download-gate/app.py so both services accept
exactly the same tokens: ES256 via the public JWKS (GoTrue's default for new
installs), falling back to HS256 for older symmetric setups. Reusing the same
env var names means one configuration for both.
"""

from __future__ import annotations

import json
import os

import jwt
from fastapi import Header, HTTPException


def verify_supabase_jwt(token: str) -> dict:
    # A malformed token is a client error, not a server error: decoding the
    # header throws before any of the verification branches run.
    try:
        header = jwt.get_unverified_header(token)
    except Exception as e:
        raise HTTPException(401, f"malformed token: {e}")
    alg = header.get("alg", "HS256")
    kid = header.get("kid")

    if alg == "ES256":
        jwks_str = os.environ.get("YAEGER_SUPABASE_JWT_JWKS", "")
        if not jwks_str:
            raise HTTPException(500, "JWT_JWKS not configured for ES256 verification")
        try:
            jwks = json.loads(jwks_str)
        except Exception as e:
            raise HTTPException(500, f"JWT_JWKS bad JSON: {e}")

        ec_key = None
        for k in jwks.get("keys", []):
            if k.get("kty") == "EC" and (kid is None or k.get("kid") == kid):
                ec_key = k
                break
        if not ec_key:
            raise HTTPException(401, f"no matching ES256 key in JWKS for kid={kid}")
        try:
            public_key = jwt.PyJWK(ec_key).key
            return jwt.decode(token, public_key, algorithms=["ES256"], audience="authenticated")
        except Exception as e:
            raise HTTPException(401, f"ES256 verification failed: {e}")

    if alg == "HS256":
        secret = os.environ.get("YAEGER_SUPABASE_JWT_SECRET")
        if not secret:
            raise HTTPException(500, "JWT_SECRET not configured for HS256 verification")
        try:
            return jwt.decode(token, secret, algorithms=["HS256"], audience="authenticated")
        except Exception as e:
            raise HTTPException(401, f"HS256 verification failed: {e}")

    raise HTTPException(401, f"unsupported algorithm: {alg}")


class Principal:
    def __init__(self, user_id: str, email: str | None):
        self.user_id = user_id
        self.email = email


async def current_user(authorization: str | None = Header(default=None)) -> Principal:
    """FastAPI dependency. Every route that costs money or reveals the store uses this.

    YAEGERPI_DEV_USER short-circuits verification for local development only. It
    is ignored unless YAEGERPI_ENV=dev, so shipping it to prod cannot silently
    disable auth.
    """
    dev_user = os.environ.get("YAEGERPI_DEV_USER")
    if dev_user and os.environ.get("YAEGERPI_ENV") == "dev":
        # Derive the email from the dev user so multi-user flows (teams) are testable.
        return Principal(user_id=dev_user, email=f"{dev_user}@localhost")

    if not authorization or not authorization.lower().startswith("bearer "):
        raise HTTPException(401, "missing bearer token")

    claims = verify_supabase_jwt(authorization[7:].strip())
    sub = claims.get("sub")
    if not sub:
        raise HTTPException(401, "token has no subject")
    return Principal(user_id=sub, email=claims.get("email"))
