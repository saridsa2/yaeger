"""yaeger-pi harness service.

Serves deployment specs for the pi plugin. Gated by the same Supabase identity
as the rest of yaeger, so a yaeger user is a yaeger-pi user.

    GET  /healthz              unauthenticated liveness
    GET  /v1/catalog           models we have verified harnesses for (the picker)
    POST /v1/resolve           model -> harness spec (tier 1/2/3)
    POST /v1/events/deploy     report outcome; drives verification + the KB
    GET  /v1/kb                inspect the gotcha KB
"""

from __future__ import annotations

import os
import secrets
from datetime import datetime, timezone

from fastapi import Depends, FastAPI, Header, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy import func, select
from sqlalchemy.orm import Session

import resolver
import signin
import teams
from auth import Principal, current_user
from db import (
    DeployEvent,
    Entitlement,
    Harness,
    KBEntry,
    ModelRequest,
    SessionLocal,
    SigninToken,
    Usage,
    init_db,
    utcnow,
)

app = FastAPI(title="yaeger-pi harness service", version="1.0.0")

# Tier-3 generation is the only path that spends GPU. Cap it per user per day.
DAILY_GENERATION_LIMIT = int(os.environ.get("YAEGERPI_DAILY_GENERATION_LIMIT", "20"))


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@app.on_event("startup")
def _startup() -> None:
    init_db()


# Teams live in their own module; it takes the session dependency so it does not
# need to import back into app.py.
app.include_router(teams.build_router(get_db))


# ---------------------------------------------------------------- models

class ResolveRequest(BaseModel):
    model_repo: str = Field(..., description="Hugging Face repo id")
    revision: str = "main"
    architecture: str | None = None
    param_count_b: float | None = Field(None, description="Billions of params, if known")
    quantization: str = "none"
    gpu: str | None = Field(None, description="Force a GPU class; otherwise chosen by fit")
    gpu_count: int = 1
    max_usd_per_hour: float | None = Field(None, description="Refuse configs above this")


class DeployEventRequest(BaseModel):
    model_repo: str
    harness_id: str | None = None
    tier: str = "cold"
    outcome: str = Field(..., pattern="^(booted|failed|aborted)$")
    phase: str | None = None
    error_excerpt: str | None = Field(None, max_length=8000)


# ---------------------------------------------------------------- routes

@app.get("/healthz")
async def healthz():
    return {"ok": True, "service": "yaeger-pi", "ts": datetime.now(timezone.utc).isoformat()}


@app.get("/v1/catalog")
def catalog(user: Principal = Depends(current_user), db: Session = Depends(get_db)):
    """What the plugin's picker shows. Verified first - those are the free, proven path."""
    rows = db.scalars(
        select(Harness).order_by(Harness.verified.desc(), Harness.boot_count.desc())
    ).all()
    return {
        "models": [
            {
                "model_repo": h.model_repo,
                "architecture": h.architecture,
                "quantization": h.quantization,
                "gpu": h.gpu,
                "gpu_count": h.gpu_count,
                "est_usd_per_hour": resolver.price(h.gpu, h.gpu_count),
                "verified": h.verified,
                "boot_count": h.boot_count,
            }
            for h in rows
        ]
    }


@app.post("/v1/resolve")
def resolve_harness(
    req: ResolveRequest,
    user: Principal = Depends(current_user),
    db: Session = Depends(get_db),
):
    tier, harness = resolver.resolve(
        db, model_repo=req.model_repo, gpu=req.gpu, gpu_count=req.gpu_count
    )

    today = utcnow().strftime("%Y-%m-%d")
    # Column defaults fire at INSERT, so a freshly constructed row has None
    # counters until flush. Initialise them explicitly.
    usage = db.get(Usage, (user.user_id, today)) or Usage(
        user_id=user.user_id, day=today, resolves=0, generations=0, est_cost_usd=0.0
    )
    usage.resolves = (usage.resolves or 0) + 1

    if tier == "exact" and harness:
        db.add(usage)
        db.commit()
        spec = dict(harness.spec)
        spec["tier"] = "exact"
        spec.setdefault("provenance", {}).update(
            {"harness_id": harness.id, "verified": True, "boot_count": harness.boot_count}
        )
        return spec

    # Tiers 2 and 3 both build a spec here. The difference is the starting point:
    # a verified sibling, or nothing.
    weights = resolver.estimate_weights_gb(req.param_count_b, req.quantization)
    gpu, count = resolver.pick_gpu(weights, req.gpu)
    hourly = resolver.price(gpu, count)

    if req.max_usd_per_hour is not None and hourly > req.max_usd_per_hour:
        raise HTTPException(
            409,
            f"cheapest fitting config is {count}x{gpu} at ${hourly}/hr, "
            f"above your ${req.max_usd_per_hour}/hr ceiling",
        )

    if tier == "near" and harness:
        spec = dict(harness.spec)
        spec["model"] = {
            "repo": req.model_repo,
            "revision": req.revision,
            "architecture": req.architecture or harness.architecture,
            "family": harness.family,
            "quantization": req.quantization,
        }
        spec.setdefault("provenance", {})["derived_from"] = harness.id
        warnings = [
            f"Adapted from {harness.model_repo}, which has booted "
            f"{harness.boot_count}x. Not yet verified for {req.model_repo}."
        ]
    else:
        # Tier 3 spends our GPU, so it is invite-only. A refusal still records
        # the request: what uninvited users ask for is the demand signal for
        # what to generate next.
        ent = db.get(Entitlement, (user.email or "").lower()) if user.email else None
        if not (ent and ent.can_generate):
            existing = db.scalars(
                select(ModelRequest).where(
                    ModelRequest.model_repo == req.model_repo,
                    ModelRequest.user_id == user.user_id,
                )
            ).first()
            if existing:
                existing.requests = (existing.requests or 0) + 1
            else:
                db.add(
                    ModelRequest(
                        model_repo=req.model_repo,
                        user_id=user.user_id,
                        email=user.email,
                        architecture=req.architecture,
                        requests=1,
                    )
                )
            db.add(usage)
            db.commit()
            raise HTTPException(
                403,
                f"{req.model_repo} has no verified harness yet. Building one runs on "
                "yaeger's GPUs, which is invite-only for now - your request has been "
                "recorded and counts toward what gets built next. Everything in "
                "/v1/catalog is available to you today.",
            )

        limit = ent.daily_generation_limit or DAILY_GENERATION_LIMIT
        if (usage.generations or 0) >= limit:
            raise HTTPException(
                429,
                f"daily generation limit reached ({limit}). "
                "Verified models in /v1/catalog remain available.",
            )
        usage.generations = (usage.generations or 0) + 1
        spec = {
            "spec_version": 1,
            "model": {
                "repo": req.model_repo,
                "revision": req.revision,
                "architecture": req.architecture,
                "quantization": req.quantization,
            },
            "engine": {"name": "vllm", "version": "0.27.1"},
            "serving": {"max_model_len": 32768},
            "modal": {"app_name": req.model_repo.split("/")[-1].lower().replace(".", "-")},
        }
        warnings = ["Generated for an architecture we have not verified. Expect to iterate."]

    spec["tier"] = tier
    spec["hardware"] = {
        "gpu": gpu,
        "count": count,
        "est_weights_gb": weights,
        "est_usd_per_hour": hourly,
    }

    entries = resolver.match_kb(
        db,
        architecture=req.architecture,
        model_repo=req.model_repo,
        quantization=req.quantization,
    )
    spec, applied = resolver.apply_kb_fixes(spec, entries)
    spec.setdefault("provenance", {}).update({"verified": False, "kb_entries_applied": applied})
    spec["warnings"] = warnings + [
        e.title for e in entries if not (e.fix or {}).get("spec_patch")
    ]

    hid = resolver.harness_id(req.model_repo, gpu, count)
    if not db.get(Harness, hid):
        db.add(
            Harness(
                id=hid,
                model_repo=req.model_repo,
                revision=req.revision,
                architecture=req.architecture,
                quantization=req.quantization,
                gpu=gpu,
                gpu_count=count,
                spec=spec,
                verified=False,
                derived_from=harness.id if (tier == "near" and harness) else None,
            )
        )
    spec["provenance"]["harness_id"] = hid

    db.add(usage)
    db.commit()
    return spec


@app.post("/v1/events/deploy")
def deploy_event(
    req: DeployEventRequest,
    user: Principal = Depends(current_user),
    db: Session = Depends(get_db),
):
    """The flywheel. A boot promotes a harness to verified; a failure gets diagnosed."""
    matched = None
    guidance = None

    if req.outcome == "failed" and req.error_excerpt:
        entry = resolver.diagnose(db, req.error_excerpt)
        if entry:
            matched = entry.id
            entry.hit_count = (entry.hit_count or 0) + 1
            guidance = {
                "kb_entry": entry.id,
                "title": entry.title,
                "cause": entry.cause,
                "fix": entry.fix,
            }

    db.add(
        DeployEvent(
            user_id=user.user_id,
            harness_id=req.harness_id,
            model_repo=req.model_repo,
            tier=req.tier,
            outcome=req.outcome,
            phase=req.phase,
            error_excerpt=req.error_excerpt,
            matched_kb=matched,
        )
    )

    if req.harness_id and (h := db.get(Harness, req.harness_id)):
        if req.outcome == "booted":
            h.verified = True
            h.boot_count = (h.boot_count or 0) + 1
            h.last_verified_at = utcnow()
        elif req.outcome == "failed":
            h.fail_count = (h.fail_count or 0) + 1

    db.commit()
    return {"recorded": True, "guidance": guidance}


@app.get("/v1/kb")
def list_kb(user: Principal = Depends(current_user), db: Session = Depends(get_db)):
    rows = db.scalars(select(KBEntry).order_by(KBEntry.hit_count.desc())).all()
    return {
        "entries": [
            {
                "id": e.id,
                "title": e.title,
                "category": e.category,
                "confidence": e.confidence,
                "silent": (e.symptom or {}).get("silent", False),
                "hit_count": e.hit_count,
            }
            for e in rows
        ]
    }


# ---------------------------------------------------------------- sign-in

class EmailRequest(BaseModel):
    email: str


class TokenRequest(BaseModel):
    token: str


@app.post("/v1/auth/email")
def auth_email(req: EmailRequest, db: Session = Depends(get_db)):
    """Email a single-use sign-in code. Always reports success: whether an
    address has an account is not something an unauthenticated caller should
    be able to probe."""
    email = req.email.strip().lower()
    if "@" not in email or len(email) > 255:
        raise HTTPException(400, "invalid email")

    token, expires = signin.new_token()
    db.add(SigninToken(token=token, email=email, expires_at=expires))
    db.commit()

    try:
        signin.send_email(email, signin.render_signin_email(email, token))
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(502, f"could not send email: {e}")

    return {"sent": True, "expires_in_minutes": int(signin.TOKEN_TTL.total_seconds() // 60)}


@app.post("/v1/auth/resolve")
async def auth_resolve(req: TokenRequest, db: Session = Depends(get_db)):
    """Claim a code and return a Supabase session."""
    row = db.get(SigninToken, req.token.strip())
    now = utcnow()
    if not row or row.consumed_at is not None or row.expires_at.replace(tzinfo=timezone.utc) < now:
        raise HTTPException(410, "code invalid, expired, or already used")

    # Claim before minting so a retry cannot reuse the same code.
    row.consumed_at = now
    db.commit()

    return await signin.mint_session(row.email)


class RefreshRequest(BaseModel):
    refresh_token: str


@app.post("/v1/auth/refresh")
async def auth_refresh(req: RefreshRequest):
    """Renew an expired session without another email round trip."""
    return await signin.refresh_session(req.refresh_token.strip())


# ---------------------------------------------------------------- admin
# Guarded by a shared secret, not by Supabase identity: these routes decide who
# can spend our GPU, so they must not be reachable by anyone who merely holds a
# valid user token. Unset ADMIN_TOKEN disables them entirely.

def require_admin(x_admin_token: str | None = Header(default=None)) -> None:
    expected = os.environ.get("YAEGERPI_ADMIN_TOKEN")
    if not expected:
        raise HTTPException(404, "admin API disabled")
    if not x_admin_token or not secrets.compare_digest(x_admin_token, expected):
        raise HTTPException(403, "bad admin token")


class InviteRequest(BaseModel):
    email: str
    can_generate: bool = True
    daily_generation_limit: int = 20
    note: str | None = None


@app.post("/v1/admin/invite", dependencies=[Depends(require_admin)])
def invite(req: InviteRequest, db: Session = Depends(get_db)):
    email = req.email.strip().lower()
    ent = db.get(Entitlement, email) or Entitlement(email=email)
    ent.can_generate = req.can_generate
    ent.daily_generation_limit = req.daily_generation_limit
    ent.note = req.note
    db.add(ent)

    # Any pending requests from this person are now servable by them.
    for r in db.scalars(select(ModelRequest).where(ModelRequest.email == email)).all():
        r.fulfilled = True
    db.commit()
    return {"email": email, "can_generate": ent.can_generate, "limit": ent.daily_generation_limit}


@app.delete("/v1/admin/invite/{email}", dependencies=[Depends(require_admin)])
def revoke(email: str, db: Session = Depends(get_db)):
    ent = db.get(Entitlement, email.strip().lower())
    if not ent:
        raise HTTPException(404, "not invited")
    db.delete(ent)
    db.commit()
    return {"revoked": email}


@app.get("/v1/admin/invites", dependencies=[Depends(require_admin)])
def list_invites(db: Session = Depends(get_db)):
    rows = db.scalars(select(Entitlement).order_by(Entitlement.created_at)).all()
    return {
        "invites": [
            {
                "email": e.email,
                "can_generate": e.can_generate,
                "limit": e.daily_generation_limit,
                "note": e.note,
            }
            for e in rows
        ]
    }


@app.get("/v1/admin/requests", dependencies=[Depends(require_admin)])
def demand(db: Session = Depends(get_db)):
    """What people want that we cannot serve, ranked by distinct askers."""
    rows = db.execute(
        select(
            ModelRequest.model_repo,
            ModelRequest.architecture,
            func.count(func.distinct(ModelRequest.user_id)).label("people"),
            func.sum(ModelRequest.requests).label("asks"),
        )
        .where(ModelRequest.fulfilled.is_(False))
        .group_by(ModelRequest.model_repo, ModelRequest.architecture)
        .order_by(func.count(func.distinct(ModelRequest.user_id)).desc())
    ).all()
    return {
        "wanted": [
            {"model_repo": r[0], "architecture": r[1], "people": r[2], "asks": r[3]} for r in rows
        ]
    }


@app.get("/v1/me")
def whoami(user: Principal = Depends(current_user), db: Session = Depends(get_db)):
    """Lets the plugin tell the user what they can do before they try it."""
    ent = db.get(Entitlement, (user.email or "").lower()) if user.email else None
    return {
        "email": user.email,
        "can_generate": bool(ent and ent.can_generate),
        "daily_generation_limit": ent.daily_generation_limit if ent else 0,
    }


@app.get("/v1/stats")
def stats(user: Principal = Depends(current_user), db: Session = Depends(get_db)):
    """Is the flywheel turning? Tier mix over time is the number that matters."""
    total = db.scalar(select(func.count()).select_from(Harness)) or 0
    verified = db.scalar(
        select(func.count()).select_from(Harness).where(Harness.verified.is_(True))
    ) or 0
    return {
        "harnesses": total,
        "verified": verified,
        "kb_entries": db.scalar(select(func.count()).select_from(KBEntry)) or 0,
        "deploys": db.scalar(select(func.count()).select_from(DeployEvent)) or 0,
    }
