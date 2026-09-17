"""Persistence-agnostic Job state-machine operations.

SHARED by every JobRepository implementation (FileJobRepository,
SqliteJobRepository, ...). Persisting is the implementation's job; WHAT a
legal state transition is lives here, once, so implementations cannot
drift (contract tests in backend_tests/test_statelessness.py run against
every implementation).

Two policies:

``mutate``    — the update policy: terminal states are LOCKED, a flagged
                (cancel_requested) job can only end cancelled, claiming
                (status -> processing) stamps the durable lease + attempt
                and diagnostics.

``interrupt_reason`` — the liveness policy: a processing job whose durable
                lease expired is dead (whatever host it ran on); a queued
                job past its grace window never started. Returns an honest
                interruption message or None. PIDs are NEVER consulted —
                the legacy PID shim lives only in the file repository's
                record migration path.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

from backend.app.domain.entities import Job, TERMINAL_STATUSES, utc_now


def mutate(
    job: Job,
    *,
    status: str | None = None,
    stage: str | None = None,
    progress: float | None = None,
    result: dict[str, Any] | None = None,
    error: dict[str, Any] | None = None,
    message: str | None = None,
    request: dict[str, Any] | None = None,
    worker_id: str | None = None,
    lease_seconds: int = 1800,
    now: datetime | None = None,
) -> bool:
    """Apply an update to the job IN PLACE. Returns False when the update
    was refused (terminal lock / cancelled-only-cancelled); the job is
    then untouched and the caller persists nothing."""
    if job.is_terminal:
        return False
    now = now or datetime.now(timezone.utc)

    if status is not None:
        if status in TERMINAL_STATUSES and job.cancel_requested and status != "cancelled":
            return False
        job.status = status
        if status == "processing":
            if job.started_at is None:
                job.started_at = utc_now()
            # Claim: durable lease + attempt; worker identity is diagnostic.
            job.attempt += 1
            if worker_id is not None:
                job.worker_id = worker_id
            job.lease_until = (now + timedelta(seconds=lease_seconds)).isoformat()
        if status in TERMINAL_STATUSES:
            job.completed_at = utc_now()
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
    if request is not None:
        job.request = request
    return True


def renew_lease(job: Job, *, lease_seconds: int = 1800,
                now: datetime | None = None) -> bool:
    """Extend the durable lease. False for terminal jobs (locked)."""
    if job.is_terminal:
        return False
    now = now or datetime.now(timezone.utc)
    job.lease_until = (now + timedelta(seconds=lease_seconds)).isoformat()
    return True


def interrupt_reason(
    job: Job,
    *,
    now: datetime | None = None,
    queued_grace_seconds: int = 60,
    from_disk: bool = False,
) -> str | None:
    """Honest liveness: why this non-terminal job is dead, or None.

    ``from_disk`` means the evaluating process has no prior memory of the
    job: a disk-only queued job past the grace window therefore never
    started (or its starter died) and is failed the same way. The grace
    window keeps a multi-worker race safe — worker B never condemns a job
    worker A created moments ago and is about to claim.
    """
    if job.is_terminal:
        return None
    now = now or datetime.now(timezone.utc)
    if job.status == "processing":
        if job.lease_until is not None:
            if datetime.fromisoformat(job.lease_until) < now:
                return (
                    "The worker's lease expired without a heartbeat — "
                    "the job was interrupted."
                )
            return None
        # Pre-lease records: the caller may apply its migration shim
        # (file repository consults owner_pid there, once).
        return None
    if (
        job.status == "queued"
        and from_disk
        and (now - job.created_dt).total_seconds() > queued_grace_seconds
    ):
        return "The server restarted before this job could start."
    return None


def finalize_interrupted(job: Job, message: str) -> Job:
    """Stamp the honest terminal failure (caller persists it)."""
    job.status = "failed"
    job.stage = "failed"
    job.error = {
        "code": "JOB_INTERRUPTED",
        "message": message,
        "recoverable": True,
    }
    job.message = message
    job.completed_at = utc_now()
    return job
