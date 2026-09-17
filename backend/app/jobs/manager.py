"""Job store — compatibility facade over the tranche-1 architecture.

Since the statelessness refactor (docs/backend_architecture_audit.md,
tranche 1) the implementation lives in:

    domain:         backend.app.domain.entities.Job      (canonical entity)
    protocol:       backend.app.domain.protocols.JobRepository
    persistence:    backend.app.infrastructure.persistence.file_job_repository
                    .FileJobRepository                     (disk-backed impl)
    application:    backend.app.application.jobs.service.JobService

``JobManager`` below is a thin compatibility subclass kept so existing
imports (``from backend.app.jobs.manager import job_manager``) and tests
keep working while callers migrate to the service/repository imports.

STATELESSNESS: disk is the source of truth; every read goes to disk, so a
server restart loses nothing, polling can hit any instance, and a job
orphaned by a crash is finalized honestly. Liveness is carried by the
DURABLE lease/heartbeat fields (a processing job whose lease expired is
marked failed JOB_INTERRUPTED on the next read); ``owner_pid`` is
diagnostics only. Records written by the pre-lease manager (PID-era) are
finalized through a one-time legacy shim on read.

State machine (explicit, terminal states LOCKED):
    queued -> processing -> completed | failed
    queued -> cancelled
    processing -> cancelled (cancel_requested honored at a checkpoint)
"""

from __future__ import annotations

from pathlib import Path

from backend.app.application.jobs.service import JobService
from backend.app.core.config import settings
from backend.app.domain.entities import Job, TERMINAL_STATUSES
from backend.app.infrastructure.persistence.file_job_repository import (
    FileJobRepository,
)

__all__ = ["Job", "JobManager", "TERMINAL_STATUSES", "job_manager"]


class JobManager(JobService):
    """Backward-compatible JobService bound to the file repository.

    Retained attributes used by the test suite:
      * ``_jobs``      — the repository's write-through index (bookkeeping
        only; deleting it cannot change any read result);
      * ``_ttl_seconds`` — TTL knob;
      * ``_job_path``  — persistence-path helper.
    """

    def __init__(
        self,
        retention_limit: int | None = None,
        ttl_seconds: int | None = None,
    ) -> None:
        repo = FileJobRepository(
            retention_limit=(
                settings.job_retention_limit
                if retention_limit is None
                else retention_limit
            ),
            ttl_seconds=(
                settings.job_ttl_seconds if ttl_seconds is None else ttl_seconds
            ),
            lease_seconds=settings.job_lease_seconds,
        )
        super().__init__(repo)
        self._repo = repo

    # -- test/compat surface ------------------------------------------------

    @property
    def _jobs(self) -> dict:
        """Write-through index of the underlying repository (NOT a read
        cache — every get/update reads the disk record)."""
        return self._repo._index

    @property
    def _ttl_seconds(self) -> int:
        return self._repo._ttl_seconds

    @staticmethod
    def _job_path(scene_id: str, job_id: str) -> Path:
        return FileJobRepository._job_path(scene_id, job_id)


job_manager = JobManager()
