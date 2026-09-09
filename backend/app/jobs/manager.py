from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from threading import Lock
from typing import Any, Optional
from uuid import uuid4


@dataclass
class Job:
    job_id: str
    scene_id: str
    status: str = "queued"
    stage: str = "queued"
    progress: Optional[float] = None
    result: Optional[dict[str, Any]] = None
    error: Optional[dict[str, Any]] = None
    created_at: str = field(default_factory=lambda: _utc_now())
    started_at: Optional[str] = None
    completed_at: Optional[str] = None


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class JobManager:
    """In-memory manager for asynchronous DepthWizard processing jobs."""

    def __init__(self) -> None:
        self._jobs: dict[str, Job] = {}
        self._lock = Lock()

    def create_job(self, scene_id: str) -> Job:
        job = Job(
            job_id=f"job_{uuid4().hex[:12]}",
            scene_id=scene_id,
        )

        with self._lock:
            self._jobs[job.job_id] = job

        return job

    def get_job(self, job_id: str) -> Optional[Job]:
        with self._lock:
            return self._jobs.get(job_id)

    def get_latest_job_for_scene(self, scene_id: str) -> Optional[Job]:
        """Return the most recently created job for a scene."""

        with self._lock:
            scene_jobs = [
                job
                for job in self._jobs.values()
                if job.scene_id == scene_id
            ]

        if not scene_jobs:
            return None

        return max(
            scene_jobs,
            key=lambda job: job.created_at,
        )

    def update_job(
        self,
        job_id: str,
        *,
        status: Optional[str] = None,
        stage: Optional[str] = None,
        progress: Optional[float] = None,
        result: Optional[dict[str, Any]] = None,
        error: Optional[dict[str, Any]] = None,
    ) -> Optional[Job]:
        with self._lock:
            job = self._jobs.get(job_id)

            if job is None:
                return None

            if status is not None:
                job.status = status

                if status == "processing" and job.started_at is None:
                    job.started_at = _utc_now()

                if status in {"completed", "failed", "cancelled"}:
                    job.completed_at = _utc_now()

            if stage is not None:
                job.stage = stage

            if progress is not None:
                job.progress = progress

            if result is not None:
                job.result = result

            if error is not None:
                job.error = error

            return job

    def delete_job(self, job_id: str) -> bool:
        with self._lock:
            return self._jobs.pop(job_id, None) is not None


job_manager = JobManager()