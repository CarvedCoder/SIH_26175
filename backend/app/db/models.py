"""SQLAlchemy entities.

Design notes:
    * Authentication identity is OWNED BY SUPABASE. The application stores
      only the stable Supabase user id (JWT ``sub``) as ``owner_id`` — no
      passwords, no duplicated credentials.
    * Scenes reference artifacts by object-storage KEY, never by absolute
      local paths.
    * Jobs are the durable record of the processing state machine; the
      in-memory coordination layer in backend.app.jobs.manager is a cache
      over these rows, so state survives backend restarts.
"""

from __future__ import annotations

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
    Index,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Base(DeclarativeBase):
    pass


class SceneRow(Base):
    __tablename__ = "scenes"

    scene_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    # Supabase auth user id (JWT "sub"); "local" when auth is disabled.
    owner_id: Mapped[str] = mapped_column(String(128), nullable=False, index=True)
    filename: Mapped[str] = mapped_column(Text, nullable=False)
    # Designated input extension, e.g. ".tif" (determinism contract).
    input_ext: Mapped[str] = mapped_column(String(16), nullable=False)
    # Object key of the uploaded input raster in the object store.
    input_object_key: Mapped[str | None] = mapped_column(Text, nullable=True)
    # The rasterio inspection payload captured at upload time.
    raster_metadata: Mapped[dict] = mapped_column(JSON, nullable=False, default=dict)
    # name -> {"key": ..., "size": ...} for every generated result artifact.
    artifacts: Mapped[dict] = mapped_column(JSON, nullable=False, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, onupdate=utcnow
    )

    __table_args__ = (
        Index("ix_scenes_owner_created", "owner_id", "created_at"),
    )


class JobRow(Base):
    __tablename__ = "jobs"

    job_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    scene_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("scenes.scene_id", ondelete="CASCADE"), index=True
    )
    owner_id: Mapped[str] = mapped_column(String(128), nullable=False, index=True)
    status: Mapped[str] = mapped_column(String(32), nullable=False, default="queued")
    stage: Mapped[str] = mapped_column(String(64), nullable=False, default="queued")
    progress: Mapped[float | None] = mapped_column(Float, nullable=True)
    cancel_requested: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    result: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    error: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    message: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    # PID of the process running inference (orphan liveness detection).
    owner_pid: Mapped[int | None] = mapped_column(Integer, nullable=True)
