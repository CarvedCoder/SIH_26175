"""External GPU worker — claims and executes durable job records.

Run a worker OUTSIDE the API process:

    python -m backend.app.worker

With DW_WORKER_MODE=external the API only creates durable job records;
this loop claims queued jobs (FileJobRepository.list_queued), executes
them through the certified pipeline with a durable lease heartbeat, and
lets the terminal-state lock make late/retrying workers harmless.

Delivery semantics: at-least-once with idempotent outputs — the
deterministic pipeline writes identical artifacts, and the terminal
lock refuses a loser's completion. Exactly-once requires the planned
database-backed JobRepository (see docs/backend_architecture.md).

Concurrency policy: one worker per GPU (DW_WORKER_CONCURRENCY=1 default)
— the process-local semaphore only guards a single process; horizontal
scaling means MORE WORKER PROCESSES, not bigger semaphores.
"""

from __future__ import annotations

import os
import time

from backend.app.application.jobs.service import JobService
from backend.app.core.config import get_settings
from backend.app.core.logging import logger
from backend.app.infrastructure.persistence.file_job_repository import (
    FileJobRepository,
)
from backend.app.services.processing_service import ProcessingService


def run_worker(poll_seconds: float | None = None, once: bool = False) -> None:
    settings = get_settings()
    poll = (
        settings.worker_poll_seconds
        if poll_seconds is None
        else poll_seconds
    )
    concurrency = max(1, int(os.environ.get("DW_WORKER_CONCURRENCY", "1")))

    repository = FileJobRepository(
        retention_limit=settings.job_retention_limit,
        ttl_seconds=settings.job_ttl_seconds,
        lease_seconds=settings.job_lease_seconds,
    )
    jobs = JobService(repository)
    processing = ProcessingService()

    logger.info(
        "worker starting: worker=%s poll=%.1fs concurrency=%d lease=%ds",
        repository.worker_id, poll, concurrency, settings.job_lease_seconds,
    )
    while True:
        executed = 0
        for job in repository.list_queued()[:concurrency]:
            claimed = repository.claim_queued(job.job_id)
            if claimed is None:
                continue  # another worker won the claim
            logger.info("worker claimed job %s scene=%s",
                        job.job_id, job.scene_id)
            # Heartbeat for the WHOLE execution: other instances never see
            # this job as interrupted while we live; if we die, the lease
            # expires and any reader finalizes it honestly.
            with jobs.lease_heartbeat(job.job_id):
                processing.execute_job_record(job.job_id)
            executed += 1
        if once:
            return
        if not executed:
            time.sleep(poll)


if __name__ == "__main__":
    try:
        run_worker()
    except KeyboardInterrupt:
        logger.info("worker stopped")
