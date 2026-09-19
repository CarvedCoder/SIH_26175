"""PostgresJobRepository — JobRepository over SQLAlchemy/PostgreSQL.

Implements the same protocol and the SAME state-machine policy
(``job_ops.py``) as FileJobRepository / SqliteJobRepository, but the
storage is the application's PostgreSQL database (Supabase in production)
via the shared SQLAlchemy engine (backend.app.db.database). Selected with
``DW_JOB_STORE=postgres``.

Claims use SELECT ... FOR UPDATE inside the session transaction, so two
workers cannot both claim one queued job. Honest storage note: switching
between the file/sqlite/postgres stores does NOT migrate existing records
— pick one per deployment.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Callable
from uuid import uuid4

from sqlalchemy import delete, select

from backend.app.core.logging import logger
from backend.app.db.database import get_engine, session_scope
from backend.app.db.models import JobRow
from backend.app.domain.entities import Job, utc_now
from backend.app.infrastructure.persistence import job_ops

_JSON_COLS = ("result", "error", "request")


def _row_to_job(row: JobRow) -> Job:
    data = {
        "job_id": row.job_id,
        "scene_id": row.scene_id,
        "status": row.status,
        "stage": row.stage,
        "progress": row.progress,
        "result": row.result,
        "error": row.error,
        "message": row.message,
        "cancel_requested": row.cancel_requested,
        "created_at": _iso(row.created_at) or utc_now(),
        "started_at": _iso(row.started_at),
        "completed_at": _iso(row.completed_at),
        "owner_pid": row.owner_pid,
        "worker_id": row.worker_id,
        "lease_until": row.lease_until,
        "attempt": row.attempt,
        "request": row.request,
    }
    return Job.from_document(data)


def _iso(value: datetime | None) -> str | None:
    if value is None:
        return None
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.isoformat()


def _apply_row(row: JobRow, job: Job) -> None:
    doc = job.to_document()
    row.job_id = doc["job_id"]
    row.scene_id = doc["scene_id"]
    row.status = doc["status"]
    row.stage = doc["stage"]
    row.progress = doc["progress"]
    row.result = doc["result"]
    row.error = doc["error"]
    row.message = doc["message"]
    row.cancel_requested = bool(doc["cancel_requested"])
    row.created_at = datetime.fromisoformat(doc["created_at"])
    row.started_at = (
        datetime.fromisoformat(doc["started_at"]) if doc["started_at"] else None
    )
    row.completed_at = (
        datetime.fromisoformat(doc["completed_at"]) if doc["completed_at"] else None
    )
    row.owner_pid = doc["owner_pid"]
    row.worker_id = doc["worker_id"]
    row.lease_until = doc["lease_until"]
    row.attempt = int(doc["attempt"])
    row.request = doc["request"]


class PostgresJobRepository:
    """JobRepository over the application's PostgreSQL database."""

    def __init__(
        self,
        *,
        retention_limit: int = 500,
        ttl_seconds: int = 24 * 3600,
        lease_seconds: int = 1800,
        queued_grace_seconds: int = 60,
        worker_id: str | None = None,
        clock: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
    ) -> None:
        self._retention_limit = retention_limit
        self._ttl_seconds = ttl_seconds
        self._lease_seconds = lease_seconds
        self._queued_grace_seconds = queued_grace_seconds
        self.worker_id = worker_id or f"worker-{uuid4().hex[:8]}"
        self._clock = clock
        from backend.app.db.database import init_db

        init_db()  # fail fast + create schema

    # -- row helpers ---------------------------------------------------------

    def _fetch(self, session, job_id: str) -> Job | None:
        row = session.get(JobRow, job_id)
        return _row_to_job(row) if row is not None else None

    def _apply_liveness(self, job: Job, *, from_disk: bool) -> Job:
        reason = job_ops.interrupt_reason(
            job,
            now=self._clock(),
            queued_grace_seconds=self._queued_grace_seconds,
            from_disk=from_disk,
        )
        if reason is None and job.status == "processing" and job.lease_until is None:
            reason = "Job record predates durable leasing and cannot be trusted."
        if reason is not None:
            job_ops.finalize_interrupted(job, reason)
            with session_scope() as session:
                row = session.get(JobRow, job.job_id)
                if row is not None:
                    _apply_row(row, job)
        return job

    def _evict(self, session) -> None:
        """TTL + retention, applied lazily on create."""
        now = self._clock()
        rows = session.scalars(
            select(JobRow).order_by(JobRow.created_at.asc())
        ).all()
        expired, alive = [], []
        for row in rows:
            created = _iso(row.created_at)
            created_dt = datetime.fromisoformat(created)
            if (now - created_dt).total_seconds() > self._ttl_seconds:
                expired.append(row.job_id)
            else:
                alive.append(row)
        for job_id in expired:
            session.delete(session.get(JobRow, job_id))
        if expired:
            logger.info("evicted %d expired jobs", len(expired))
        overflow = len(alive) - self._retention_limit
        for row in alive:
            if overflow <= 0:
                break
            if row.status in ("completed", "failed", "cancelled"):
                session.delete(row)
                overflow -= 1

    # -- JobRepository protocol ----------------------------------------------

    def create(self, scene_id: str) -> Job:
        job = Job(job_id=f"job_{uuid4().hex[:12]}", scene_id=scene_id)
        with session_scope() as session:
            self._evict(session)
            session.add(JobRow(job_id=job.job_id, scene_id=scene_id))
            session.flush()
            row = session.get(JobRow, job.job_id)
            _apply_row(row, job)
        logger.info(
            "job created: %s scene=%s worker=%s",
            job.job_id, scene_id, self.worker_id,
        )
        return job

    def get(self, job_id: str) -> Job | None:
        with session_scope() as session:
            job = self._fetch(session, job_id)
        if job is None:
            return None
        return self._apply_liveness(job, from_disk=True)

    def list_for_scene(self, scene_id: str) -> list[Job]:
        with session_scope() as session:
            rows = session.scalars(
                select(JobRow)
                .where(JobRow.scene_id == scene_id)
                .order_by(JobRow.created_at.asc())
            ).all()
            jobs = [_row_to_job(r) for r in rows]
        return [self._apply_liveness(j, from_disk=True) for j in jobs]

    def get_latest_for_scene(self, scene_id: str) -> Job | None:
        jobs = self.list_for_scene(scene_id)
        return max(jobs, key=lambda j: j.created_at) if jobs else None

    def get_active_for_scene(self, scene_id: str) -> Job | None:
        active = [j for j in self.list_for_scene(scene_id) if not j.is_terminal]
        return max(active, key=lambda j: j.created_at) if active else None

    def update(self, job_id: str, **fields: Any) -> Job | None:
        allowed = {
            k: fields[k]
            for k in ("status", "stage", "progress", "result", "error",
                      "message", "request")
            if k in fields
        }
        with session_scope() as session:
            row = session.get(JobRow, job_id)
            if row is None:
                return None
            job = _row_to_job(row)
            if not job_ops.mutate(
                job, worker_id=self.worker_id,
                lease_seconds=self._lease_seconds, now=self._clock(), **allowed,
            ):
                return job  # refused (terminal lock); nothing persisted
            _apply_row(row, job)
        return job

    def renew_lease(self, job_id: str) -> Job | None:
        with session_scope() as session:
            row = session.get(JobRow, job_id)
            if row is None:
                return None
            job = _row_to_job(row)
            if not job_ops.renew_lease(
                job, lease_seconds=self._lease_seconds, now=self._clock()
            ):
                return job
            _apply_row(row, job)
        return job

    def list_queued(self) -> list[Job]:
        with session_scope() as session:
            rows = session.scalars(
                select(JobRow)
                .where(JobRow.status == "queued")
                .order_by(JobRow.created_at.asc())
            ).all()
            jobs = [_row_to_job(r) for r in rows]
        return [self._apply_liveness(j, from_disk=True) for j in jobs]

    def claim_queued(self, job_id: str) -> Job | None:
        """Exactly-once claim: SELECT ... FOR UPDATE serializes claimers;
        the status check and the processing update commit atomically."""
        engine = get_engine()
        with engine.begin() as conn:
            row = conn.execute(
                select(JobRow).where(JobRow.job_id == job_id).with_for_update()
            ).scalar_one_or_none()
            if row is None or row.status != "queued":
                return None
            job = _row_to_job(row)
            if not job_ops.mutate(
                job,
                status="processing",
                stage="depth_inference",
                progress=5.0,
                worker_id=self.worker_id,
                lease_seconds=self._lease_seconds,
                now=self._clock(),
            ):
                return None
            _apply_row(row, job)
        return job

    def request_cancel(self, job_id: str) -> Job | None:
        with session_scope() as session:
            row = session.get(JobRow, job_id)
            if row is None:
                return None
            job = _row_to_job(row)
            if job.is_terminal:
                return job
            job.cancel_requested = True
            if job.status == "queued":
                job.status = "cancelled"
                job.stage = "cancelled"
                job.completed_at = utc_now()
            _apply_row(row, job)
        logger.info("cancel requested: %s", job_id)
        return job

    def delete(self, job_id: str) -> bool:
        with session_scope() as session:
            row = session.get(JobRow, job_id)
            if row is None:
                return False
            session.delete(row)
            return True

    def delete_for_scene(self, scene_id: str) -> int:
        with session_scope() as session:
            result = session.execute(
                delete(JobRow).where(JobRow.scene_id == scene_id)
            )
            return int(result.rowcount or 0)
