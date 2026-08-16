"""Team endpoints: shared inference, shared context, shared bill.

The service is a control plane. It decides who may fetch a team's endpoint
credentials; it never proxies inference. Prompts and completions go straight
from a member's machine to the team's own Modal endpoint and are never seen
here - that is the sovereignty property, and it is why usage is self-reported.
"""

from __future__ import annotations

import secrets
from datetime import datetime, timedelta, timezone

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from auth import Principal, current_user
from db import Team, TeamGpuTime, TeamMember, TeamUsage, utcnow

router = APIRouter(prefix="/v1/teams", tags=["teams"])

# What the same tokens would have cost on a frontier API, for the comparison the
# whole exercise exists to answer. Rough per-million-token rates, USD.
API_REFERENCE = {"input": 3.00, "output": 15.00}


def _db():  # replaced by app.py's dependency at import time
    raise NotImplementedError


def new_id() -> str:
    return secrets.token_urlsafe(9).replace("-", "").replace("_", "")[:12]


def email_of(user: Principal) -> str:
    if not user.email:
        raise HTTPException(400, "account has no email")
    return user.email.lower()


def membership(db: Session, team_id: str, email: str) -> TeamMember:
    m = db.scalars(
        select(TeamMember).where(TeamMember.team_id == team_id, TeamMember.email == email)
    ).first()
    if not m:
        raise HTTPException(403, "you are not a member of this team")
    return m


def require_owner(db: Session, team_id: str, email: str) -> Team:
    team = db.get(Team, team_id)
    if not team:
        raise HTTPException(404, "no such team")
    if team.owner_email != email:
        raise HTTPException(403, "only the team owner can do that")
    return team


# ---------------------------------------------------------------- models

class CreateTeam(BaseModel):
    name: str = Field(..., min_length=1, max_length=120)


class InviteMember(BaseModel):
    email: str


class SetContext(BaseModel):
    context: str = Field(..., max_length=8000)


class PublishEndpoint(BaseModel):
    url: str
    key: str
    model: str
    gpu: str
    gpu_count: int = 1
    usd_per_hour: float = 0.0
    sandbox_id: str | None = None
    # When the sandbox was CREATED, not when it became ready. A cold start can
    # be ~10 minutes of billed GPU that produces no tokens; starting the clock
    # at publish would hide exactly the cost this feature exists to expose.
    started_at: datetime | None = None


class ReportUsage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    requests: int = 1


def build_router(get_db):
    """app.py passes its session dependency in, so this module stays importable."""

    @router.post("")
    def create_team(req: CreateTeam, user: Principal = Depends(current_user),
                    db: Session = Depends(get_db)):
        """Anyone signed in can start a team. Starting one costs nothing."""
        email = email_of(user)
        team = Team(id=new_id(), name=req.name.strip(), owner_email=email)
        db.add(team)
        db.add(TeamMember(team_id=team.id, email=email, role="owner", joined_at=utcnow()))
        db.commit()
        return {"team_id": team.id, "name": team.name, "role": "owner"}

    @router.get("")
    def my_teams(user: Principal = Depends(current_user), db: Session = Depends(get_db)):
        email = email_of(user)
        rows = db.scalars(select(TeamMember).where(TeamMember.email == email)).all()
        out = []
        for m in rows:
            t = db.get(Team, m.team_id)
            if not t:
                continue
            out.append({
                "team_id": t.id,
                "name": t.name,
                "role": m.role,
                "joined": m.joined_at is not None,
                "endpoint_live": bool(t.endpoint_url and not t.endpoint_stopped_at),
                "model": t.endpoint_model,
                "members": db.scalar(
                    select(func.count()).select_from(TeamMember).where(TeamMember.team_id == t.id)
                ),
            })
        return {"teams": out}

    @router.post("/{team_id}/invite")
    def invite(team_id: str, req: InviteMember, user: Principal = Depends(current_user),
               db: Session = Depends(get_db)):
        require_owner(db, team_id, email_of(user))
        email = req.email.strip().lower()
        if "@" not in email:
            raise HTTPException(400, "invalid email")
        existing = db.scalars(
            select(TeamMember).where(TeamMember.team_id == team_id, TeamMember.email == email)
        ).first()
        if existing:
            return {"email": email, "already_member": True}
        db.add(TeamMember(team_id=team_id, email=email, role="member"))
        db.commit()
        return {"email": email, "invited": True}

    @router.delete("/{team_id}/members/{email}")
    def remove_member(team_id: str, email: str, user: Principal = Depends(current_user),
                      db: Session = Depends(get_db)):
        """Revoking membership revokes endpoint access at the next fetch.

        It does not rotate the key, so a removed member keeps working until the
        endpoint restarts. Rotate by restarting if that matters.
        """
        team = require_owner(db, team_id, email_of(user))
        target = email.strip().lower()
        if target == team.owner_email:
            raise HTTPException(400, "cannot remove the owner")
        m = db.scalars(
            select(TeamMember).where(TeamMember.team_id == team_id, TeamMember.email == target)
        ).first()
        if not m:
            raise HTTPException(404, "not a member")
        db.delete(m)
        db.commit()
        return {"removed": target, "note": "restart the endpoint to rotate the shared key"}

    @router.put("/{team_id}/endpoint")
    def publish_endpoint(team_id: str, req: PublishEndpoint,
                         user: Principal = Depends(current_user),
                         db: Session = Depends(get_db)):
        """The owner publishes the endpoint after launching it on their Modal."""
        team = require_owner(db, team_id, email_of(user))
        team.endpoint_url = req.url
        team.endpoint_key = req.key
        team.endpoint_model = req.model
        team.endpoint_gpu = req.gpu
        team.endpoint_gpu_count = req.gpu_count
        team.endpoint_usd_per_hour = req.usd_per_hour
        team.endpoint_sandbox_id = req.sandbox_id
        team.endpoint_started_at = req.started_at or utcnow()
        team.endpoint_stopped_at = None
        db.commit()
        return {"published": True, "team_id": team_id, "model": req.model}

    @router.delete("/{team_id}/endpoint")
    def unpublish_endpoint(team_id: str, user: Principal = Depends(current_user),
                           db: Session = Depends(get_db)):
        """Marks the endpoint stopped and books the GPU time it consumed."""
        team = require_owner(db, team_id, email_of(user))
        if team.endpoint_started_at and not team.endpoint_stopped_at:
            secs = (utcnow() - team.endpoint_started_at.replace(tzinfo=timezone.utc)).total_seconds()
            day = utcnow().strftime("%Y-%m-%d")
            row = db.scalars(
                select(TeamGpuTime).where(TeamGpuTime.team_id == team_id, TeamGpuTime.day == day)
            ).first() or TeamGpuTime(team_id=team_id, day=day, gpu_seconds=0.0, usd=0.0)
            row.gpu_seconds = (row.gpu_seconds or 0) + secs
            row.usd = (row.usd or 0) + secs / 3600 * (team.endpoint_usd_per_hour or 0)
            db.add(row)
        team.endpoint_stopped_at = utcnow()
        db.commit()
        return {"stopped": True}

    @router.get("/{team_id}/endpoint")
    def get_endpoint(team_id: str, user: Principal = Depends(current_user),
                     db: Session = Depends(get_db)):
        """Members fetch the shared endpoint. This is the only gate that matters
        for inference access - the service never sees the traffic itself."""
        email = email_of(user)
        m = membership(db, team_id, email)
        team = db.get(Team, team_id)
        if not team:
            raise HTTPException(404, "no such team")
        if not m.joined_at:
            m.joined_at = utcnow()
            db.commit()
        if not team.endpoint_url or team.endpoint_stopped_at:
            raise HTTPException(
                409,
                "the team endpoint is not running. The owner starts it with /yaeger-team-start.",
            )
        return {
            "url": team.endpoint_url,
            "key": team.endpoint_key,
            "model": team.endpoint_model,
            "gpu": f"{team.endpoint_gpu_count}x{team.endpoint_gpu}",
            "usd_per_hour": team.endpoint_usd_per_hour,
            "context": team.context,
            "context_updated_at": (
                team.context_updated_at.isoformat() if team.context_updated_at else None
            ),
            # Members must know their prompts are recorded. This is surfaced by
            # the plugin on join; silent tracing of a teammate is not acceptable.
            "tracing": {
                "enabled": True,
                "stored": "your team's own Modal volume - never sent to yaeger",
                "visible_to": team.owner_email,
            },
        }

    @router.get("/{team_id}/context")
    def get_context(team_id: str, user: Principal = Depends(current_user),
                    db: Session = Depends(get_db)):
        membership(db, team_id, email_of(user))
        team = db.get(Team, team_id)
        return {"context": team.context if team else None}

    @router.put("/{team_id}/context")
    def set_context(team_id: str, req: SetContext, user: Principal = Depends(current_user),
                    db: Session = Depends(get_db)):
        """What the team is working on. Any member can update it; a joiner's
        model reads it immediately, so they arrive with context instead of cold."""
        membership(db, team_id, email_of(user))
        team = db.get(Team, team_id)
        if not team:
            raise HTTPException(404, "no such team")
        team.context = req.context
        team.context_updated_at = utcnow()
        db.commit()
        return {"updated": True, "chars": len(req.context)}

    @router.post("/{team_id}/usage")
    def report_usage(team_id: str, req: ReportUsage, user: Principal = Depends(current_user),
                     db: Session = Depends(get_db)):
        email = email_of(user)
        membership(db, team_id, email)
        day = utcnow().strftime("%Y-%m-%d")
        row = db.scalars(
            select(TeamUsage).where(
                TeamUsage.team_id == team_id, TeamUsage.email == email, TeamUsage.day == day
            )
        ).first() or TeamUsage(
            team_id=team_id, email=email, day=day,
            prompt_tokens=0, completion_tokens=0, requests=0,
        )
        row.prompt_tokens = (row.prompt_tokens or 0) + max(0, req.prompt_tokens)
        row.completion_tokens = (row.completion_tokens or 0) + max(0, req.completion_tokens)
        row.requests = (row.requests or 0) + max(0, req.requests)
        db.add(row)
        db.commit()
        return {"recorded": True}

    @router.get("/{team_id}/economics")
    def economics(team_id: str, days: int = 7, user: Principal = Depends(current_user),
                  db: Session = Depends(get_db)):
        """The question this whole thing exists to answer: is it worth it?

        Owner-only: this exposes what every member did, and it is the owner who
        carries the bill. Members see their own usage via /me/usage.

        Counts GPU time still accruing on a live endpoint, so idle time shows up
        rather than being quietly excluded.
        """
        team = require_owner(db, team_id, email_of(user))

        since = (utcnow() - timedelta(days=days)).strftime("%Y-%m-%d")

        usage = db.execute(
            select(
                func.coalesce(func.sum(TeamUsage.prompt_tokens), 0),
                func.coalesce(func.sum(TeamUsage.completion_tokens), 0),
                func.coalesce(func.sum(TeamUsage.requests), 0),
                func.count(func.distinct(TeamUsage.email)),
            ).where(TeamUsage.team_id == team_id, TeamUsage.day >= since)
        ).one()
        prompt_tokens, completion_tokens, requests, people = usage

        gpu_seconds = db.scalar(
            select(func.coalesce(func.sum(TeamGpuTime.gpu_seconds), 0.0)).where(
                TeamGpuTime.team_id == team_id, TeamGpuTime.day >= since
            )
        ) or 0.0
        usd = db.scalar(
            select(func.coalesce(func.sum(TeamGpuTime.usd), 0.0)).where(
                TeamGpuTime.team_id == team_id, TeamGpuTime.day >= since
            )
        ) or 0.0

        # Include the currently-running session, otherwise a live endpoint looks free.
        if team.endpoint_started_at and not team.endpoint_stopped_at:
            live = (utcnow() - team.endpoint_started_at.replace(tzinfo=timezone.utc)).total_seconds()
            gpu_seconds += live
            usd += live / 3600 * (team.endpoint_usd_per_hour or 0)

        total_tokens = prompt_tokens + completion_tokens
        # Below a few minutes of GPU time the derived ratios are dominated by
        # noise and read as absurdly favourable. Report nothing rather than
        # something flattering and wrong.
        meaningful = gpu_seconds >= 300
        api_equiv = (
            prompt_tokens / 1_000_000 * API_REFERENCE["input"]
            + completion_tokens / 1_000_000 * API_REFERENCE["output"]
        )
        gpu_hours = gpu_seconds / 3600

        return {
            "window_days": days,
            "people_active": people,
            "requests": requests,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "gpu_hours": round(gpu_hours, 2),
            "gpu_usd": round(usd, 2),
            "api_equivalent_usd": round(api_equiv, 2),
            "saving_usd": round(api_equiv - usd, 2),
            "usd_per_million_tokens": round(usd / (total_tokens / 1_000_000), 2)
            if (total_tokens and meaningful)
            else None,
            "tokens_per_gpu_hour": int(total_tokens / gpu_hours)
            if (gpu_hours and meaningful)
            else None,
            "verdict": _verdict(api_equiv, usd, total_tokens, gpu_hours),
        }

    @router.get("/{team_id}/insights")
    def insights(team_id: str, days: int = 30, user: Principal = Depends(current_user),
                 db: Session = Depends(get_db)):
        """Per-member activity for the admin, in the shape a contributor graph wants.

        Owner-only by design: this is who-did-what, which members should not be
        able to read about each other.
        """
        require_owner(db, team_id, email_of(user))
        since = (utcnow() - timedelta(days=days)).strftime("%Y-%m-%d")

        rows = db.execute(
            select(
                TeamUsage.email,
                TeamUsage.day,
                func.sum(TeamUsage.prompt_tokens),
                func.sum(TeamUsage.completion_tokens),
                func.sum(TeamUsage.requests),
            )
            .where(TeamUsage.team_id == team_id, TeamUsage.day >= since)
            .group_by(TeamUsage.email, TeamUsage.day)
            .order_by(TeamUsage.day)
        ).all()

        by_member: dict[str, dict] = {}
        for email, day, pt, ct, rq in rows:
            m = by_member.setdefault(
                email, {"email": email, "days": {}, "tokens": 0, "requests": 0, "active_days": 0}
            )
            m["days"][day] = {"tokens": (pt or 0) + (ct or 0), "requests": rq or 0}
            m["tokens"] += (pt or 0) + (ct or 0)
            m["requests"] += rq or 0
            m["active_days"] += 1

        members = db.scalars(select(TeamMember).where(TeamMember.team_id == team_id)).all()
        for mem in members:
            by_member.setdefault(
                mem.email,
                {"email": mem.email, "days": {}, "tokens": 0, "requests": 0, "active_days": 0},
            )["role"] = mem.role

        return {
            "window_days": days,
            "contributors": sorted(
                by_member.values(), key=lambda m: m["tokens"], reverse=True
            ),
        }

    @router.get("/{team_id}/me/usage")
    def my_usage(team_id: str, days: int = 30, user: Principal = Depends(current_user),
                 db: Session = Depends(get_db)):
        """A member can always see their own numbers - just not everyone else's."""
        email = email_of(user)
        membership(db, team_id, email)
        since = (utcnow() - timedelta(days=days)).strftime("%Y-%m-%d")
        r = db.execute(
            select(
                func.coalesce(func.sum(TeamUsage.prompt_tokens), 0),
                func.coalesce(func.sum(TeamUsage.completion_tokens), 0),
                func.coalesce(func.sum(TeamUsage.requests), 0),
            ).where(
                TeamUsage.team_id == team_id,
                TeamUsage.email == email,
                TeamUsage.day >= since,
            )
        ).one()
        return {"email": email, "prompt_tokens": r[0], "completion_tokens": r[1], "requests": r[2]}

    return router


def _verdict(api_usd: float, gpu_usd: float, tokens: int, gpu_hours: float) -> str:
    if gpu_hours < 0.084:  # < ~5 min of GPU time
        return "not enough GPU time yet to judge - come back after a real session"
    if tokens == 0:
        return f"${gpu_usd:.2f} of GPU time and no tokens served - the endpoint is idle, stop it"
    if api_usd > gpu_usd * 1.2:
        return f"self-hosting is winning: ${api_usd - gpu_usd:.2f} cheaper over this window"
    if gpu_usd > api_usd * 1.2:
        return (
            f"the API would be ${gpu_usd - api_usd:.2f} cheaper - utilisation is too low. "
            "Add people to the team or stop the endpoint between sessions."
        )
    return "roughly break-even with the API"
