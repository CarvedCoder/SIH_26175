"""Disk-backed job manager with cancellation semantics.

STATELESSNESS: every job is persisted as a JSON document under the scene's
process directory (``data/process/scenes/<scene_id>/jobs/<job_id>.json``).
The in-process ``_jobs`` dict is only a write-through CACHE — the file on
disk is the source of truth. Consequences:

    * a server restart loses nothing: completed/failed/cancelled jobs are
      still readable and the frontend can still show their results;
    * polling can hit any worker process (multi-worker deployments work);
    * a job orphaned by a crash/restart is detected honestly: its owning
      PID is recorded when processing starts, and any reader that finds a
      non-terminal job whose owner PID is no longer alive marks it FAILED
      (JOB_INTERRUPTED) instead of leaving it stuck at "processing" —
      in-flight torch work cannot be resumed, so "still running" would lie.

State machine (explicit, one-way through terminal states):
    queued -> processing -> completed
                        \\-> failed
    queued -> cancelled
    processing -> cancelled (cancel requested; worker honors it at the
                 next checkpoint and the result is discarded)

Terminal statuses are LOCKED: a late worker returning after a cancel or a
failure can never overwrite a terminal record.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import Any
from uuid import uuid4

from backend.app.core.config import settings
from backend.app.core.logging import logger
from backend.app.core.paths import get_scene_process_dir


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


class JobManager:
    """Bounded, thread-safe, DISK-BACKED store of asynchronous jobs.

    All mutations write through to ``<scene process dir>/jobs/<id>.json``;
    reads fall back to disk when a job is not in this process's cache, so
    restarts and worker processes all observe the same job state.
    """

    def __init__(
        self,
        retention_limit: int = 500,
        ttl_seconds: int = 24 * 3600,
    ) -> None:
        self._jobs: dict[str, Job] = {}
        self._lock = Lock()
        self._retention_limit = retention_limit
        self._ttl_seconds = ttl_seconds
        # How long a disk-only "queued" job is trusted to start (multi-
        # worker create→start window) before it's declared interrupted.
        self._queued_grace_seconds = 60

    # -- persistence helpers ----------------------------------------------

    @staticmethod
    def _jobs_dir(scene_id: str) -> Path:
        return get_scene_process_dir(scene_id) / "jobs"

    @staticmethod
    def _job_path(scene_id: str, job_id: str) -> Path:
        return JobManager._jobs_dir(scene_id) / f"{job_id}.json"

    @staticmethod
    def _load_job_file(path: Path) -> Job | None:
        try:
            with path.open("r", encoding="utf-8") as f:
                data = json.load(f)
            return Job(**data)
        except (OSError, json.JSONDecodeError, TypeError, ValueError):
            return None

    @staticmethod
    def _write_job_file(job: Job) -> None:
        path = JobManager._job_path(job.scene_id, job.job_id)
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(".json.tmp")
            with tmp.open("w", encoding="utf-8") as f:
                json.dump(asdict(job), f)
            tmp.replace(path)
        except OSError:
            logger.exception("failed to persist job %s", job.job_id)

    def _revive_or_fail(self, job: Job, from_disk: bool = False) -> Job:
        """Normalize a job loaded from disk.

        A non-terminal job whose owning process is gone can never finish
        (inference is not resumable) — mark it failed once, persist that,
        and every future reader sees the honest terminal state.

        ``from_disk`` means THIS process has no memory of the job: it was
        created by a previous lifetime. Queued-in-disk jobs older than the
        grace window therefore never started (or their starter died) and
        are failed the same way. The grace window keeps a multi-worker
        race safe — worker B never condemns a job worker A created mere
        moments ago and is about to start.
        """
        if job.is_terminal:
            return job
        interrupted: str | None = None
        if job.status == "processing" and not _pid_alive(job.owner_pid):
            interrupted = "The server restarted while this job was running."
        elif (
            job.status == "queued"
            and from_disk
            and (datetime.now(timezone.utc) - job.created_dt).total_seconds()
            > self._queued_grace_seconds
        ):
            interrupted = (
                "The server restarted before this job could start."
            )
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
            self._write_job_file(job)
            logger.info("job %s marked interrupted: %s", job.job_id, interrupted)
        return job

    def _scan_scene_jobs(self, scene_id: str) -> list[Job]:
        """All persisted jobs for a scene (cache-miss reads go to disk)."""
        jobs_dir = self._jobs_dir(scene_id)
        if not jobs_dir.is_dir():
            return []
        now = datetime.now(timezone.utc)
        loaded: list[Job] = []
        for path in jobs_dir.glob("*.json"):
            cached = self._jobs.get(path.stem)
            job = cached if cached is not None else self._load_job_file(path)
            if job is None:
                continue
            # TTL eviction, applied lazily — old job files cannot grow
            # without bound.
            if (now - job.created_dt).total_seconds() > self._ttl_seconds:
                path.unlink(missing_ok=True)
                with self._lock:
                    self._jobs.pop(job.job_id, None)
                continue
            job = self._revive_or_fail(job, from_disk=cached is None)
            with self._lock:
                self._jobs[job.job_id] = job
            loaded.append(job)
        # Retention: keep the newest N, drop oldest terminal first.
        if len(loaded) > self._retention_limit:
            ordered = sorted(loaded, key=lambda j: j.created_at)
            for job in ordered[: len(loaded) - self._retention_limit]:
                self._job_path(scene_id, job.job_id).unlink(missing_ok=True)
                with self._lock:
                    self._jobs.pop(job.job_id, None)
            loaded = ordered[len(loaded) - self._retention_limit:]
        return loaded

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
        self._write_job_file(job)

        logger.info("job created: %s scene=%s", job.job_id, scene_id)
        return job

    # -- reads ------------------------------------------------------------

    def get_job(self, job_id: str) -> Job | None:
        with self._lock:
            job = self._jobs.get(job_id)
        if job is not None:
            return self._revive_or_fail(job)
        # Cache miss: the job may belong to a previous process lifetime —
        # job_id encodes no scene, so scan is the only honest lookup. The
        # scenes root is small (per-scene jobs/ dirs); scanning it is cheap.
        return self._find_job_anywhere(job_id)

    def _find_job_anywhere(self, job_id: str) -> Job | None:
        from backend.app.core.paths import SCENES_PROCESS_DIR

        base = SCENES_PROCESS_DIR
        if not base.is_dir():
            return None
        for jobs_dir in base.glob("*/jobs"):
            path = jobs_dir / f"{job_id}.json"
            if path.is_file():
                job = self._load_job_file(path)
                if job is None:
                    return None
                with self._lock:
                    self._jobs[job.job_id] = job
                return self._revive_or_fail(job, from_disk=True)
        return None

    def get_latest_job_for_scene(self, scene_id: str) -> Job | None:
        """Return the most recently created job for a scene."""
        scene_jobs = self._scan_scene_jobs(scene_id)
        if not scene_jobs:
            return None
        return max(scene_jobs, key=lambda job: job.created_at)

    def get_active_job_for_scene(self, scene_id: str) -> Job | None:
        """Return the queued/processing job for a scene, if any."""
        active = [
            job
            for job in self._scan_scene_jobs(scene_id)
            if not job.is_terminal
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
                    job.owner_pid = os.getpid()
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

        # Write-through outside the lock: disk I/O must not serialise
        # concurrent readers; worst case a poller reads the previous
        # revision, which is exactly what polling tolerates.
        self._write_job_file(job)
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

        self._write_job_file(job)
        logger.info("cancel requested: %s", job_id)
        return job

    # -- deletion / retention ---------------------------------------------

    def delete_job(self, job_id: str) -> bool:
        with self._lock:
            job = self._jobs.pop(job_id, None)
        if job is not None:
            self._job_path(job.scene_id, job_id).unlink(missing_ok=True)
            return True
        return False

    def delete_jobs_for_scene(self, scene_id: str) -> int:
        with self._lock:
            doomed = [
                job_id for job_id, job in self._jobs.items() if job.scene_id == scene_id
            ]
            for job_id in doomed:
                del self._jobs[job_id]
        removed = len(doomed)
        jobs_dir = self._jobs_dir(scene_id)
        if jobs_dir.is_dir():
            for path in jobs_dir.glob("*.json"):
                path.unlink(missing_ok=True)
                removed += 1
        return removed

    # -- in-memory eviction (keeps the cache bounded; evicted jobs' files
    # are removed too so disk and cache agree) ----------------------------

    def _evict_expired_locked(self) -> None:
        now = datetime.now(timezone.utc)
        expired = [
            job_id
            for job_id, job in self._jobs.items()
            if (now - job.created_dt).total_seconds() > self._ttl_seconds
        ]
        for job_id in expired:
            job = self._jobs.pop(job_id)
            self._job_path(job.scene_id, job_id).unlink(missing_ok=True)
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
            self._job_path(job.scene_id, job.job_id).unlink(missing_ok=True)


job_manager = JobManager(
    retention_limit=settings.job_retention_limit,
    ttl_seconds=settings.job_ttl_seconds,
)
