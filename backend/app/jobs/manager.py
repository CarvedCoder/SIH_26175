"""Bounded in-memory job manager with cancellation semantics.

PROTOTYPE LIMITATION (explicit): jobs live in process memory and do NOT
survive a restart. A single-node MVP does not justify a persistence layer —
in-flight inference cannot survive a restart either, so persisted "queued"
rows would lie about resumability. The store is BOUNDED though: the oldest
terminal jobs are evicted past ``job_retention_limit``, and any job older
than ``job_ttl_seconds`` is evicted, so memory cannot grow without bound.

State machine (explicit, one-way through terminal states):
    queued -> processing -> completed
                        \-> failed
    queued -> cancelled
    processing -> cancelled (cancel requested; worker honors it at the
                 next checkpoint and the result is discarded)

Terminal statuses are LOCKED: a late worker returning after a cancel or a
failure can never overwrite a terminal record.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from threading import Lock
from typing import Any
from uuid import uuid4

from backend.app.core.config import settings
from backend.app.core.logging import logger


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class Job:
    job_id: str
    scene_id: str
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

    @property
    def is_terminal(self) -> bool:
        return self.status in {"completed", "failed", "cancelled"}

    @property
    def created_dt(self) -> datetime:
        return datetime.fromisoformat(self.created_at)


TERMINAL_STATUSES = {"completed", "failed", "cancelled"}


class JobManager:
    """Bounded, thread-safe store of asynchronous processing jobs."""

    def __init__(
        self,
        retention_limit: int = 500,
        ttl_seconds: int = 24 * 3600,
    ) -> None:
        self._jobs: dict[str, Job] = {}
        self._lock = Lock()
        self._retention_limit = retention_limit
        self._ttl_seconds = ttl_seconds

    # -- creation ---------------------------------------------------------

    def create_job(self, scene_id: str) -> Job:
        job = Job(
            job_id=f"job_{uuid4().hex[:12]}",
            scene_id=scene_id,
        )

        with self._lock:
            self._evict_expired_locked()
            self._jobs[job.job_id] = job
            self._evict_overflow_locked()

        logger.info("job created: %s scene=%s", job.job_id, scene_id)
        return job

    # -- reads ------------------------------------------------------------

    def get_job(self, job_id: str) -> Job | None:
        with self._lock:
            return self._jobs.get(job_id)

    def get_latest_job_for_scene(self, scene_id: str) -> Job | None:
        """Return the most recently created job for a scene."""

        with self._lock:
            scene_jobs = [
                job for job in self._jobs.values() if job.scene_id == scene_id
            ]

        if not scene_jobs:
            return None

        return max(scene_jobs, key=lambda job: job.created_at)

    def get_active_job_for_scene(self, scene_id: str) -> Job | None:
        """Return the queued/processing job for a scene, if any."""

        with self._lock:
            active = [
                job
                for job in self._jobs.values()
                if job.scene_id == scene_id and not job.is_terminal
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
            job = self._jobs.get(job_id)

            if job is None:
                return None

            if job.is_terminal:
                return job

            if status is not None:
                if status in TERMINAL_STATUSES and job.cancel_requested and status != "cancelled":
                    # a cancelled job can only end as cancelled
                    return job
                job.status = status
                if status == "processing" and job.started_at is None:
                    job.started_at = _utc_now()
                if status in TERMINAL_STATUSES:
                    job.completed_at = _utc_now()

            if stage is not None:
                job.stage = stage
            if progress is not None:
                job.progress = progress
            if result is not None:
                job.result = result
            if error is not None:
                job.error = error
            if message is not None:
                job.message = message

            return job

    def request_cancel(self, job_id: str) -> Job | None:
        """Request cancellation.

        * queued  -> cancelled immediately (never started).
        * running -> cancel_requested flagged; the worker checks it at its
          next checkpoint and the job ends cancelled (results discarded).
        * terminal-> unchanged.
        """

        with self._lock:
            job = self._jobs.get(job_id)
            if job is None or job.is_terminal:
                return job

            job.cancel_requested = True
            if job.status == "queued":
                job.status = "cancelled"
                job.stage = "cancelled"
                job.completed_at = _utc_now()

        logger.info("cancel requested: %s", job_id)
        return job

    # -- deletion / retention ---------------------------------------------

    def delete_job(self, job_id: str) -> bool:
        with self._lock:
            return self._jobs.pop(job_id, None) is not None

    def delete_jobs_for_scene(self, scene_id: str) -> int:
        with self._lock:
            doomed = [
                job_id for job_id, job in self._jobs.items() if job.scene_id == scene_id
            ]
            for job_id in doomed:
                del self._jobs[job_id]
            return len(doomed)

    def _evict_expired_locked(self) -> None:
        now = datetime.now(timezone.utc)
        expired = [
            job_id
            for job_id, job in self._jobs.items()
            if (now - job.created_dt).total_seconds() > self._ttl_seconds
        ]
        for job_id in expired:
            del self._jobs[job_id]
        if expired:
            logger.info("evicted %d expired jobs", len(expired))

    def _evict_overflow_locked(self) -> None:
        if len(self._jobs) <= self._retention_limit:
            return
        terminal = sorted(
            (job for job in self._jobs.values() if job.is_terminal),
            key=lambda job: job.created_at,
        )
        overflow = len(self._jobs) - self._retention_limit
        for job in terminal[:overflow]:
            del self._jobs[job.job_id]


job_manager = JobManager(
    retention_limit=settings.job_retention_limit,
    ttl_seconds=settings.job_ttl_seconds,
)
