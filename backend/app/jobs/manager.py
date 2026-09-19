"""Durable job manager — PostgreSQL is the source of truth.

MIGRATION NOTE: jobs were previously persisted as JSON files under the
scene's process directory; they now live in the ``jobs`` SQL table
(backend.app.db.models.JobRow) via SQLAlchemy. All public semantics are
preserved:

    * a server restart loses nothing: completed/failed/cancelled jobs are
      still readable and the frontend can still show their results;
    * polling can hit any worker process;
    * a job orphaned by a crash/restart is detected honestly via PID
      liveness: a non-terminal job whose owning process is gone is marked
      FAILED (JOB_INTERRUPTED) on the next read instead of pretending to
      still run.

State machine (terminal statuses LOCKED):
    queued -> processing -> completed | failed
    queued -> cancelled
    processing -> cancelled (cancel_requested honored at a checkpoint)
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from threading import Lock
from typing import Any
from uuid import uuid4

from sqlalchemy import delete, select

from backend.app.core.config import settings
from backend.app.core.logging import logger
from backend.app.db.database import session_scope
from backend.app.db.models import JobRow


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _dt_to_iso(value: datetime | None) -> str | None:
    if value is None:
        return None
    if value.tzinfo is None:
        # SQLite returns naive UTC datetimes; re-anchor them so aware/naive
        # arithmetic can never mix.
        value = value.replace(tzinfo=timezone.utc)
    return value.isoformat()


@dataclass
class Job:
    job_id: str
    scene_id: str
    owner_id: str = "local"
    status: str = "queued"
    stage: str = "queued"
    progress: float | None = None
    result: dict[str, Any] | None = None
    error: dict[str, Any] | None = None
    message: str | None = None
    cancel_requested: bool = False
    created_at: str = field(default_factory=_utc_now)
    started_at: str | None = None
    completed_at: str | None = None
    owner_pid: int | None = None  # process running the inference (liveness)

    @property
    def is_terminal(self) -> bool:
        return self.status in {"completed", "failed", "cancelled"}

    @property
    def created_dt(self) -> datetime:
        return datetime.fromisoformat(self.created_at)


TERMINAL_STATUSES = {"completed", "failed", "cancelled"}


def _pid_alive(pid: int | None) -> bool:
    """True when a process with this PID exists on this machine."""
    if pid is None or pid <= 0:
        return False
    try:
        os.kill(pid, 0)  # signal 0 = existence check, no signal sent
    except ProcessLookupError:
        return False
    except PermissionError:
        return True  # exists, owned by someone else
    return True


def _row_to_job(row: JobRow) -> Job:
    return Job(
        job_id=row.job_id,
        scene_id=row.scene_id,
        owner_id=row.owner_id,
        status=row.status,
        stage=row.stage,
        progress=row.progress,
        result=row.result,
        error=row.error,
        message=row.message,
        cancel_requested=row.cancel_requested,
        created_at=_dt_to_iso(row.created_at) or _utc_now(),
        started_at=_dt_to_iso(row.started_at),
        completed_at=_dt_to_iso(row.completed_at),
        owner_pid=row.owner_pid,
    )


def _apply_update(row: JobRow, job: Job) -> None:
    row.status = job.status
    row.stage = job.stage
    row.progress = job.progress
    row.result = job.result
    row.error = job.error
    row.message = job.message
    row.cancel_requested = job.cancel_requested
    row.started_at = datetime.fromisoformat(job.started_at) if job.started_at else None
    row.completed_at = (
        datetime.fromisoformat(job.completed_at) if job.completed_at else None
    )
    row.owner_pid = job.owner_pid


class JobManager:
    """Thread-safe store of asynchronous jobs backed by the SQL database.

    Every mutation is a transaction; reads always hit the database, so
    restarts and worker processes all observe the same job state.
    """

    def __init__(
        self,
        retention_limit: int = 500,
        ttl_seconds: int = 24 * 3600,
    ) -> None:
        self._retention_limit = retention_limit
        self._ttl_seconds = ttl_seconds
        # How long a queued job is trusted to start (create→start window)
        # before it's declared interrupted.
        self._queued_grace_seconds = 60
        self._lock = Lock()  # serializes in-process read-modify-write cycles

    # -- persistence helpers ----------------------------------------------

    def _revive_or_fail(self, job: Job) -> Job:
        """Fail a non-terminal job whose owning process is gone — honest
        terminal state instead of a stuck 'processing' record."""
        if job.is_terminal:
            return job
        interrupted: str | None = None
        if job.status == "processing" and not _pid_alive(job.owner_pid):
            interrupted = "The server restarted while this job was running."
        elif (
            job.status == "queued"
            and (datetime.now(timezone.utc) - job.created_dt).total_seconds()
            > self._queued_grace_seconds
        ):
            interrupted = "The server restarted before this job could start."
        if interrupted:
            job.status = "failed"
            job.stage = "failed"
            job.error = {
                "code": "JOB_INTERRUPTED",
                "message": interrupted,
                "recoverable": True,
            }
            job.message = interrupted
            job.completed_at = _utc_now()
            with session_scope() as session:
                row = session.get(JobRow, job.job_id)
                if row is not None:
                    _apply_update(row, job)
            logger.info("job %s marked interrupted: %s", job.job_id, interrupted)
        return job

    def _scan_scene_jobs(self, scene_id: str) -> list[Job]:
        """All persisted jobs for a scene, with TTL + retention applied."""
        now = datetime.now(timezone.utc)
        with session_scope() as session:
            rows = session.scalars(
                select(JobRow)
                .where(JobRow.scene_id == scene_id)
                .order_by(JobRow.created_at.asc())
            ).all()
            jobs = [_row_to_job(row) for row in rows]
            doomed: list[str] = []
            for job in jobs:
                if (now - job.created_dt).total_seconds() > self._ttl_seconds:
                    doomed.append(job.job_id)
            if len(jobs) > self._retention_limit:
                doomed.extend(
                    job.job_id for job in jobs[: len(jobs) - self._retention_limit]
                )
            if doomed:
                session.execute(delete(JobRow).where(JobRow.job_id.in_(set(doomed))))
                jobs = [job for job in jobs if job.job_id not in set(doomed)]
        return [self._revive_or_fail(job) for job in jobs]

    # -- creation ---------------------------------------------------------

    def create_job(self, scene_id: str, owner_id: str = "local") -> Job:
        job = Job(
            job_id=f"job_{uuid4().hex[:12]}",
            scene_id=scene_id,
            owner_id=owner_id,
        )
        with session_scope() as session:
            session.add(
                JobRow(
                    job_id=job.job_id,
                    scene_id=scene_id,
                    owner_id=owner_id,
                    status=job.status,
                    stage=job.stage,
                    created_at=datetime.fromisoformat(job.created_at),
                )
            )
        logger.info("job created: %s scene=%s", job.job_id, scene_id)
        return job

    # -- reads ------------------------------------------------------------

    def get_job(self, job_id: str) -> Job | None:
        with session_scope() as session:
            row = session.get(JobRow, job_id)
            job = _row_to_job(row) if row is not None else None
            if job is not None and (
                datetime.now(timezone.utc) - job.created_dt
            ).total_seconds() > self._ttl_seconds:
                # TTL-expired: evict (the old disk store evicted on the
                # next create; a direct read is the equivalent trigger).
                session.delete(row)
                return None
        return self._revive_or_fail(job) if job is not None else None

    def get_latest_job_for_scene(self, scene_id: str) -> Job | None:
        scene_jobs = self._scan_scene_jobs(scene_id)
        if not scene_jobs:
            return None
        return max(scene_jobs, key=lambda job: job.created_at)

    def get_active_job_for_scene(self, scene_id: str) -> Job | None:
        active = [
            job for job in self._scan_scene_jobs(scene_id) if not job.is_terminal
        ]
        if not active:
            return None
        return max(active, key=lambda job: job.created_at)

    # -- mutation ----------------------------------------------------------

    def update_job(
        self,
        job_id: str,
        *,
        status: str | None = None,
        stage: str | None = None,
        progress: float | None = None,
        result: dict[str, Any] | None = None,
        error: dict[str, Any] | None = None,
        message: str | None = None,
    ) -> Job | None:
        """Update a job. Terminal statuses are locked: an update that tries
        to move a terminal job (or finish a cancelled one) is refused and
        the record keeps its current state."""
        with self._lock:
            with session_scope() as session:
                row = session.get(JobRow, job_id)
                if row is None:
                    return None
                if row.status in TERMINAL_STATUSES:
                    return _row_to_job(row)

                if status is not None:
                    if (
                        status in TERMINAL_STATUSES
                        and row.cancel_requested
                        and status != "cancelled"
                    ):
                        # a cancelled job can only end as cancelled
                        return _row_to_job(row)
                    row.status = status
                    if status == "processing" and row.started_at is None:
                        row.started_at = datetime.now(timezone.utc)
                        row.owner_pid = os.getpid()
                    if status in TERMINAL_STATUSES:
                        row.completed_at = datetime.now(timezone.utc)

                if stage is not None:
                    row.stage = stage
                if progress is not None:
                    row.progress = progress
                if result is not None:
                    row.result = result
                if error is not None:
                    row.error = error
                if message is not None:
                    row.message = message
                return _row_to_job(row)

    def request_cancel(self, job_id: str) -> Job | None:
        """Request cancellation: queued -> cancelled immediately; running
        -> cancel_requested flagged (worker honors at next checkpoint)."""
        with self._lock:
            with session_scope() as session:
                row = session.get(JobRow, job_id)
                if row is None or row.status in TERMINAL_STATUSES:
                    return _row_to_job(row) if row is not None else None
                row.cancel_requested = True
                if row.status == "queued":
                    row.status = "cancelled"
                    row.stage = "cancelled"
                    row.completed_at = datetime.now(timezone.utc)
                return _row_to_job(row)

    # -- deletion ----------------------------------------------------------

    def delete_job(self, job_id: str) -> bool:
        with session_scope() as session:
            row = session.get(JobRow, job_id)
            if row is None:
                return False
            session.delete(row)
            return True

    def delete_jobs_for_scene(self, scene_id: str) -> int:
        with session_scope() as session:
            result = session.execute(
                delete(JobRow).where(JobRow.scene_id == scene_id)
            )
            return int(result.rowcount or 0)


job_manager = JobManager(
    retention_limit=settings.job_retention_limit,
    ttl_seconds=settings.job_ttl_seconds,
)
