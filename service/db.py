"""Storage for the harness store, the gotcha KB, and deploy telemetry.

Portable across SQLite (local dev) and Postgres (the yaeger box) so the same
code runs in both: JSON columns instead of JSONB, no server-side defaults.
"""

from __future__ import annotations

import os
from datetime import datetime, timezone

from sqlalchemy import (
    JSON,
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    UniqueConstraint,
    create_engine,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, sessionmaker

DATABASE_URL = os.environ.get("YAEGERPI_DATABASE_URL", "sqlite:///./yaegerpi.db")


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Base(DeclarativeBase):
    pass


class Harness(Base):
    """A deployment spec for one model on one GPU class.

    `verified` is the whole point of the store: it means this exact spec booted
    on real hardware at least once. An unverified row is still a hypothesis and
    must be labelled as such when served.
    """

    __tablename__ = "harnesses"
    __table_args__ = (UniqueConstraint("model_repo", "gpu", "gpu_count", name="uq_harness_target"),)

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    model_repo: Mapped[str] = mapped_column(String(255), index=True)
    revision: Mapped[str] = mapped_column(String(64), default="main")
    architecture: Mapped[str | None] = mapped_column(String(128), index=True)
    family: Mapped[str | None] = mapped_column(String(64), index=True)
    quantization: Mapped[str] = mapped_column(String(32), default="none")

    gpu: Mapped[str] = mapped_column(String(32), index=True)
    gpu_count: Mapped[int] = mapped_column(Integer, default=1)

    spec: Mapped[dict] = mapped_column(JSON)

    verified: Mapped[bool] = mapped_column(Boolean, default=False, index=True)
    boot_count: Mapped[int] = mapped_column(Integer, default=0)
    fail_count: Mapped[int] = mapped_column(Integer, default=0)
    derived_from: Mapped[str | None] = mapped_column(String(64))

    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)
    last_verified_at: Mapped[datetime | None] = mapped_column(DateTime)


class KBEntry(Base):
    """One hard-won gotcha: what breaks, why, and which spec field fixes it.

    Entries are matched two ways - proactively by `applies_to` when building a
    spec, and reactively by `error_regex` when diagnosing a failed deploy.
    """

    __tablename__ = "kb_entries"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    title: Mapped[str] = mapped_column(Text)
    category: Mapped[str] = mapped_column(String(32), index=True)

    applies_to: Mapped[dict] = mapped_column(JSON)
    symptom: Mapped[dict] = mapped_column(JSON)
    cause: Mapped[str] = mapped_column(Text)
    fix: Mapped[dict] = mapped_column(JSON)

    confidence: Mapped[str] = mapped_column(String(16), default="reported")
    source: Mapped[str] = mapped_column(Text, default="")
    hit_count: Mapped[int] = mapped_column(Integer, default=0)

    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)


class DeployEvent(Base):
    """Outcome telemetry. Drives the flywheel and doubles as the metering log."""

    __tablename__ = "deploy_events"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[str] = mapped_column(String(64), index=True)
    harness_id: Mapped[str | None] = mapped_column(ForeignKey("harnesses.id"))
    model_repo: Mapped[str] = mapped_column(String(255), index=True)
    tier: Mapped[str] = mapped_column(String(8))

    outcome: Mapped[str] = mapped_column(String(16), index=True)  # booted | failed | aborted
    phase: Mapped[str | None] = mapped_column(String(32))
    error_excerpt: Mapped[str | None] = mapped_column(Text)
    matched_kb: Mapped[str | None] = mapped_column(String(64))

    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow, index=True)


class Entitlement(Base):
    """The invite list.

    Generation spends OUR GPU, so it is invite-only. Everything else - browsing
    the catalog and deploying a verified harness to the user's own Modal - is
    open to any authenticated account, because it costs us nothing.

    Keyed by email rather than Supabase user_id so an invite can be issued
    before the person has ever signed in.
    """

    __tablename__ = "entitlements"

    email: Mapped[str] = mapped_column(String(255), primary_key=True)
    can_generate: Mapped[bool] = mapped_column(Boolean, default=True)
    daily_generation_limit: Mapped[int] = mapped_column(Integer, default=20)
    note: Mapped[str | None] = mapped_column(Text)
    invited_by: Mapped[str | None] = mapped_column(String(255))
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)


class ModelRequest(Base):
    """Demand signal: models people asked for that we cannot yet serve.

    A refusal is wasted information. Recording it tells us what to generate
    next, ranked by how many distinct people wanted it.
    """

    __tablename__ = "model_requests"
    __table_args__ = (UniqueConstraint("model_repo", "user_id", name="uq_request_user"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    model_repo: Mapped[str] = mapped_column(String(255), index=True)
    user_id: Mapped[str] = mapped_column(String(64), index=True)
    email: Mapped[str | None] = mapped_column(String(255))
    architecture: Mapped[str | None] = mapped_column(String(128))
    requests: Mapped[int] = mapped_column(Integer, default=1)
    fulfilled: Mapped[bool] = mapped_column(Boolean, default=False, index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)


class Team(Base):
    """A group sharing one endpoint, one index, and one bill.

    The owner's Modal account runs the GPU; members just get the URL and key.
    That is the point: only one person in a team needs a Modal account at all.
    """

    __tablename__ = "teams"

    id: Mapped[str] = mapped_column(String(32), primary_key=True)
    name: Mapped[str] = mapped_column(String(120))
    owner_email: Mapped[str] = mapped_column(String(255), index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)

    # What the team is currently working on. Injected into member sessions so a
    # joiner's model has context immediately. The codegraph replaces this later.
    context: Mapped[str | None] = mapped_column(Text)
    context_updated_at: Mapped[datetime | None] = mapped_column(DateTime)

    # The shared endpoint, published by the owner after launching it.
    endpoint_url: Mapped[str | None] = mapped_column(String(512))
    endpoint_key: Mapped[str | None] = mapped_column(String(255))
    endpoint_model: Mapped[str | None] = mapped_column(String(255))
    endpoint_gpu: Mapped[str | None] = mapped_column(String(32))
    endpoint_gpu_count: Mapped[int] = mapped_column(Integer, default=1)
    endpoint_usd_per_hour: Mapped[float] = mapped_column(Float, default=0.0)
    endpoint_sandbox_id: Mapped[str | None] = mapped_column(String(64))
    endpoint_started_at: Mapped[datetime | None] = mapped_column(DateTime)
    endpoint_stopped_at: Mapped[datetime | None] = mapped_column(DateTime)


class TeamMember(Base):
    """Membership, keyed by email so an invite can precede a first sign-in."""

    __tablename__ = "team_members"
    __table_args__ = (UniqueConstraint("team_id", "email", name="uq_team_member"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    team_id: Mapped[str] = mapped_column(ForeignKey("teams.id"), index=True)
    email: Mapped[str] = mapped_column(String(255), index=True)
    role: Mapped[str] = mapped_column(String(16), default="member")  # owner | member
    joined_at: Mapped[datetime | None] = mapped_column(DateTime)
    invited_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)


class TeamUsage(Base):
    """Token economics.

    Traffic goes straight from member to the team's endpoint - it never crosses
    this service - so these numbers are self-reported by the plugin. Good enough
    for a team measuring itself; not a billing ledger for strangers.
    """

    __tablename__ = "team_usage"
    __table_args__ = (UniqueConstraint("team_id", "email", "day", name="uq_team_usage"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    team_id: Mapped[str] = mapped_column(ForeignKey("teams.id"), index=True)
    email: Mapped[str] = mapped_column(String(255), index=True)
    day: Mapped[str] = mapped_column(String(10), index=True)
    prompt_tokens: Mapped[int] = mapped_column(Integer, default=0)
    completion_tokens: Mapped[int] = mapped_column(Integer, default=0)
    requests: Mapped[int] = mapped_column(Integer, default=0)


class TeamGpuTime(Base):
    """Wall-clock GPU seconds the team paid for, including idle.

    Idle time is the number that decides whether self-hosting is worth it, and
    the one every vendor comparison leaves out.
    """

    __tablename__ = "team_gpu_time"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    team_id: Mapped[str] = mapped_column(ForeignKey("teams.id"), index=True)
    day: Mapped[str] = mapped_column(String(10), index=True)
    gpu_seconds: Mapped[float] = mapped_column(Float, default=0.0)
    usd: Mapped[float] = mapped_column(Float, default=0.0)


class SigninToken(Base):
    """Single-use sign-in codes, minted and claimed by this service alone."""

    __tablename__ = "signin_tokens"

    token: Mapped[str] = mapped_column(String(64), primary_key=True)
    email: Mapped[str] = mapped_column(String(255), index=True)
    expires_at: Mapped[datetime] = mapped_column(DateTime)
    consumed_at: Mapped[datetime | None] = mapped_column(DateTime)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)


class Usage(Base):
    """Per-user counters, so tier-3 generation can be rate-limited by identity."""

    __tablename__ = "usage"

    user_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    day: Mapped[str] = mapped_column(String(10), primary_key=True)  # YYYY-MM-DD
    resolves: Mapped[int] = mapped_column(Integer, default=0)
    generations: Mapped[int] = mapped_column(Integer, default=0)
    est_cost_usd: Mapped[float] = mapped_column(Float, default=0.0)


engine = create_engine(
    DATABASE_URL,
    future=True,
    connect_args={"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {},
)
SessionLocal = sessionmaker(bind=engine, expire_on_commit=False, future=True)


def init_db() -> None:
    Base.metadata.create_all(engine)
