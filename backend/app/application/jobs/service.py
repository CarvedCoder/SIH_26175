"""JobService — the application layer over the durable JobRepository.

The API routes and the processing worker talk to THIS class, never to a
storage implementation. It adds the worker-side heartbeat (a durable
lease renewed while inference runs) on top of the repository protocol.

Old import path: ``backend.app.jobs.manager.JobManager`` remains as a
thin compatibility subclass — existing callers and tests keep working
while they migrate to the service/repository imports.
"""

from __future__ import annotations

import threading
from contextlib import contextmanager
from typing import Any, Iterator

from backend.app.core.logging import logger
from backend.app.domain.entities import Job
from backend.app.domain.protocols import JobRepository


class JobService:
    """Application service for the job lifecycle."""

    def __init__(
        self,
        repository: JobRepository,
        *,
        heartbeat_interval_seconds: float | None = None,
    ) -> None:
        self._repo = repository
        # Heartbeat cadence: a third of the lease by default (the repo
        # owns the lease duration) so two missed beats still don't expire
        # a live worker.
        self._heartbeat_interval = heartbeat_interval_seconds

    # -- reads (delegate) ---------------------------------------------------

    def get_job(self, job_id: str) -> Job | None:
        return self._repo.get(job_id)

    def get_latest_job_for_scene(self, scene_id: str) -> Job | None:
        return self._repo.get_latest_for_scene(scene_id)

    def get_active_job_for_scene(self, scene_id: str) -> Job | None:
        return self._repo.get_active_for_scene(scene_id)

    def list_jobs_for_scene(self, scene_id: str) -> list[Job]:
        return self._repo.list_for_scene(scene_id)

    # -- mutations (delegate) -------------------------------------------------

    def create_job(self, scene_id: str) -> Job:
        return self._repo.create(scene_id)

    def update_job(self, job_id: str, **fields: Any) -> Job | None:
        return self._repo.update(job_id, **fields)

    def request_cancel(self, job_id: str) -> Job | None:
        return self._repo.request_cancel(job_id)

    def delete_job(self, job_id: str) -> bool:
        return self._repo.delete(job_id)

    def delete_jobs_for_scene(self, scene_id: str) -> int:
        return self._repo.delete_for_scene(scene_id)

    # -- worker-side heartbeat -------------------------------------------------

    @property
    def heartbeat_interval(self) -> float:
        if self._heartbeat_interval is not None:
            return self._heartbeat_interval
        lease = getattr(self._repo, "_lease_seconds", 1800)
        return max(5.0, lease / 3.0)

    @contextmanager
    def lease_heartbeat(self, job_id: str) -> Iterator[None]:
        """Keep the job's durable lease renewed while the worker runs.

        Usage: ``with job_service.lease_heartbeat(job_id): <long work>``.
        The heartbeat is a PERFORMANCE-RESPONSIVENESS mechanism over
        durable state — if the worker dies mid-run, the lease simply
        expires and the next reader finalizes the job honestly. Stopping
        the thread can never corrupt anything.
        """
        stop = threading.Event()

        def _beat() -> None:
            while not stop.wait(self.heartbeat_interval):
                try:
                    self._repo.renew_lease(job_id)
                except Exception:  # noqa: BLE001 — a failed beat is retried
                    logger.exception("lease heartbeat failed for %s", job_id)

        thread = threading.Thread(
            target=_beat, name=f"lease-heartbeat-{job_id}", daemon=True
        )
        thread.start()
        try:
            yield
        finally:
            stop.set()
            thread.join(timeout=self.heartbeat_interval + 5.0)
