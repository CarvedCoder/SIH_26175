"""File-backed JobRepository — the durable, cross-process job store.

Layout: one JSON document per job under the scene's process directory
(``<process root>/<scene_id>/jobs/<job_id>.json``), written atomically
(tmp + replace). DISK IS THE SOURCE OF TRUTH: every read goes to disk.
Process-local state in this class is limited to a threading.Lock around
read-modify-write mutations and a small write-through index used only for
retention/eviction bookkeeping — deleting it cannot change any read
result (pinned by backend_tests/test_statelessness.py).

Liveness model (replaces the pre-tranche owner-PID check):
    a processing job carries ``lease_until`` + ``attempt``; its worker
    renews the lease via :meth:`renew_lease` (a heartbeat well inside the
    lease window). Any reader that finds a non-terminal job whose lease
    expired finalizes it as an honest JOB_INTERRUPTED failure — no PID
    liveness anywhere in the correctness path. ``owner_pid`` /
    ``worker_id`` are recorded for logs only.

    Legacy migration shim: records written before the lease fields
    existed have ``lease_until is None``; they are finalized using the
    old PID check exactly once, then persist their terminal state. New
    records never depend on PIDs.

Durability boundary (honest limitation, tranche 2+): mutations are
read-modify-write per file with last-writer-wins across processes. For
the current deployment (one API process; concurrent mutation of the SAME
job by two processes is not a designed flow) this is sound; a database
implementation of the same protocol is the upgrade path when
multi-writer conflicts become real.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from threading import RLock
from typing import Any, Callable
from uuid import uuid4

from backend.app.core.logging import logger
from backend.app.core.paths import get_scene_process_dir
from backend.app.domain.entities import Job, TERMINAL_STATUSES, utc_now


def _pid_alive(pid: int | None) -> bool:
    """Legacy migration shim only — True when a PID exists on this host."""
    if pid is None or pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


class FileJobRepository:
    """JobRepository over the per-scene JSON job documents."""

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
        # RLock: _locate_locked runs inside update/cancel's lock scope and
        # may itself consult the index/scan helpers.
        self._lock = RLock()
        self._retention_limit = retention_limit
        self._ttl_seconds = ttl_seconds
        self._lease_seconds = lease_seconds
        self._queued_grace_seconds = queued_grace_seconds
        # Logical identity of THIS worker/process — diagnostic only.
        self.worker_id = worker_id or f"worker-{os.getpid()}-{uuid4().hex[:8]}"
        self._clock = clock
        # Write-through index for eviction bookkeeping ONLY; never read
        # back job state from it.
        self._index: dict[str, Job] = {}

    # -- persistence helpers ----------------------------------------------

    @staticmethod
    def _jobs_dir(scene_id: str) -> Path:
        return get_scene_process_dir(scene_id) / "jobs"

    @classmethod
    def _job_path(cls, scene_id: str, job_id: str) -> Path:
        return cls._jobs_dir(scene_id) / f"{job_id}.json"

    def _load_job_file(self, path: Path) -> Job | None:
        try:
            with path.open("r", encoding="utf-8") as f:
                data = json.load(f)
            return Job.from_document(data)
        except (OSError, json.JSONDecodeError, TypeError, ValueError):
            return None

    def _write_job_file(self, job: Job) -> None:
        path = self._job_path(job.scene_id, job.job_id)
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(".json.tmp")
            with tmp.open("w", encoding="utf-8") as f:
                json.dump(job.to_document(), f)
            tmp.replace(path)
        except OSError:
            logger.exception("failed to persist job %s", job.job_id)

    # -- liveness -----------------------------------------------------------

    def _finalize_interrupted(self, job: Job, message: str) -> Job:
        job.status = "failed"
        job.stage = "failed"
        job.error = {
            "code": "JOB_INTERRUPTED",
            "message": message,
            "recoverable": True,
        }
        job.message = message
        job.completed_at = utc_now()
        self._write_job_file(job)
        logger.info("job %s finalized as interrupted: %s", job.job_id, message)
        return job

    def _revive_or_fail(self, job: Job, *, from_disk: bool) -> Job:
        """Honest liveness check, applied to every non-terminal read."""
        if job.is_terminal:
            return job
        now = self._clock()
        if job.status == "processing":
            if job.lease_until is not None:
                if datetime.fromisoformat(job.lease_until) < now:
                    return self._finalize_interrupted(
                        job,
                        "The worker's lease expired without a heartbeat — "
                        "the job was interrupted.",
                    )
                return job
            # Legacy migration shim (pre-lease records only): fall back to
            # the old PID check exactly once, then persist the outcome.
            if not _pid_alive(job.owner_pid):
                return self._finalize_interrupted(
                    job, "The server restarted while this job was running."
                )
            return job
        if (
            job.status == "queued"
            and from_disk
            and (now - job.created_dt).total_seconds() > self._queued_grace_seconds
        ):
            return self._finalize_interrupted(
                job, "The server restarted before this job could start."
            )
        return job

    # -- scanning / retention -------------------------------------------------

    def _scan_scene_jobs(self, scene_id: str) -> list[Job]:
        """All persisted jobs for a scene. Reads ALWAYS come from disk —
        this is what makes multi-instance deployments observe one truth."""
        jobs_dir = self._jobs_dir(scene_id)
        if not jobs_dir.is_dir():
            return []
        now = self._clock()
        loaded: list[Job] = []
        for path in jobs_dir.glob("*.json"):
            job = self._load_job_file(path)
            if job is None:
                self._index.pop(path.stem, None)
                continue
            # TTL eviction, applied lazily — old job files cannot grow
            # without bound.
            if (now - job.created_dt).total_seconds() > self._ttl_seconds:
                path.unlink(missing_ok=True)
                self._index.pop(job.job_id, None)
                continue
            job = self._revive_or_fail(job, from_disk=True)
            with self._lock:
                self._index[job.job_id] = job
            loaded.append(job)
        # Retention: keep the newest N, drop oldest terminal first.
        if len(loaded) > self._retention_limit:
            ordered = sorted(loaded, key=lambda j: j.created_at)
            for job in ordered[: len(loaded) - self._retention_limit]:
                self._job_path(scene_id, job.job_id).unlink(missing_ok=True)
                with self._lock:
                    self._index.pop(job.job_id, None)
            loaded = ordered[len(loaded) - self._retention_limit :]
        return loaded

    def _evict_expired_index_locked(self) -> None:
        now = self._clock()
        expired = [
            job_id
            for job_id, job in self._index.items()
            if (now - job.created_dt).total_seconds() > self._ttl_seconds
        ]
        for job_id in expired:
            job = self._index.pop(job_id)
            self._job_path(job.scene_id, job_id).unlink(missing_ok=True)

    def _evict_overflow_index_locked(self) -> None:
        """Retention on the write-through index + disk: keep the newest N,
        evicting only TERMINAL jobs (active jobs are being polled)."""
        if len(self._index) <= self._retention_limit:
            return
        terminal = sorted(
            (job for job in self._index.values() if job.is_terminal),
            key=lambda job: job.created_at,
        )
        overflow = len(self._index) - self._retention_limit
        for job in terminal[:overflow]:
            self._index.pop(job.job_id, None)
            self._job_path(job.scene_id, job.job_id).unlink(missing_ok=True)

    # -- JobRepository protocol ------------------------------------------------

    def create(self, scene_id: str) -> Job:
        job = Job(job_id=f"job_{uuid4().hex[:12]}", scene_id=scene_id)
        with self._lock:
            self._evict_expired_index_locked()
            self._index[job.job_id] = job
            self._evict_overflow_index_locked()
        self._write_job_file(job)
        logger.info("job created: %s scene=%s worker=%s",
                    job.job_id, scene_id, self.worker_id)
        return job

    def get(self, job_id: str) -> Job | None:
        """Read from DISK. The job_id encodes no scene, so the first lookup
        needs a scan of the (small) per-scene jobs/ directories."""
        with self._lock:
            indexed = self._index.get(job_id)
        if indexed is not None:
            path = self._job_path(indexed.scene_id, job_id)
            if path.is_file():
                job = self._load_job_file(path)
                if job is not None:
                    job = self._revive_or_fail(job, from_disk=False)
                    return job
            # fell through: indexed copy is stale (file gone) — rescan
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
                job = self._revive_or_fail(job, from_disk=True)
                with self._lock:
                    self._index[job.job_id] = job
                return job
        return None

    def list_for_scene(self, scene_id: str) -> list[Job]:
        return self._scan_scene_jobs(scene_id)

    def get_latest_for_scene(self, scene_id: str) -> Job | None:
        jobs = self._scan_scene_jobs(scene_id)
        if not jobs:
            return None
        return max(jobs, key=lambda job: job.created_at)

    def get_active_for_scene(self, scene_id: str) -> Job | None:
        active = [job for job in self._scan_scene_jobs(scene_id)
                  if not job.is_terminal]
        if not active:
            return None
        return max(active, key=lambda job: job.created_at)

    def update(
        self,
        job_id: str,
        *,
        status: str | None = None,
        stage: str | None = None,
        progress: float | None = None,
        result: dict[str, Any] | None = None,
        error: dict[str, Any] | None = None,
        message: str | None = None,
        request: dict[str, Any] | None = None,
    ) -> Job | None:
        with self._lock:
            job = self._locate_locked(job_id)
            if job is None:
                return None
            if job.is_terminal:
                return job
            if status is not None:
                if (status in TERMINAL_STATUSES
                        and job.cancel_requested and status != "cancelled"):
                    # a cancelled job can only end as cancelled
                    return job
                job.status = status
                if status == "processing":
                    if job.started_at is None:
                        job.started_at = utc_now()
                    # Claim: durable lease + diagnostics. attempt counts
                    # retries if the queue layer ever re-dispatches.
                    job.attempt += 1
                    job.worker_id = self.worker_id
                    job.owner_pid = os.getpid()
                    job.lease_until = (
                        self._clock() + timedelta(seconds=self._lease_seconds)
                    ).isoformat()
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
            with_job = job
        # Write-through outside the lock: disk I/O must not serialise
        # concurrent readers; a poller seeing the previous revision is
        # exactly what polling tolerates.
        self._write_job_file(with_job)
        with self._lock:
            self._index[with_job.job_id] = with_job
        return with_job

    def renew_lease(self, job_id: str) -> Job | None:
        """Worker heartbeat: extend the durable lease. No-op for terminal
        jobs. Returns the updated job, or None if unknown."""
        with self._lock:
            job = self._locate_locked(job_id)
            if job is None or job.is_terminal:
                return job
            job.lease_until = (
                self._clock() + timedelta(seconds=self._lease_seconds)
            ).isoformat()
        self._write_job_file(job)
        return job

    def _locate_locked(self, job_id: str) -> Job | None:
        """Read-modify-write within the process lock. Disk-first: the disk
        record is authoritative even when an index entry exists — but the
        INDEXED OBJECT is refreshed in place and returned, so every holder
        of a previously returned Job reference observes this manager's
        later mutations (the in-place mutation contract the old JobManager
        established, kept for callers). Cross-process visibility comes
        from the disk read, never from the index."""
        from dataclasses import fields as _dc_fields

        indexed = self._index.get(job_id)
        if indexed is not None:
            path = self._job_path(indexed.scene_id, job_id)
            if path.is_file():
                fresh = self._load_job_file(path)
                if fresh is not None:
                    for f in _dc_fields(Job):
                        setattr(indexed, f.name, getattr(fresh, f.name))
                    return indexed
        found = self._find_job_anywhere(job_id)
        if found is not None:
            self._index[job_id] = found
        return found

    def list_queued(self) -> list[Job]:
        """All claimable queued jobs across scenes (worker discovery).
        Revival applies first: a queued job past its grace window is
        finalized instead of being handed to a worker."""
        from backend.app.core.paths import SCENES_PROCESS_DIR

        base = SCENES_PROCESS_DIR
        if not base.is_dir():
            return []
        queued: list[Job] = []
        for jobs_dir in sorted(base.glob("*/jobs")):
            for path in sorted(jobs_dir.glob("*.json")):
                job = self._load_job_file(path)
                if job is None:
                    continue
                job = self._revive_or_fail(job, from_disk=True)
                if job.status == "queued":
                    queued.append(job)
        return queued

    def claim_queued(self, job_id: str) -> Job | None:
        """Claim a queued job: queued → processing, ONLY if the disk record
        is still queued (in-process check-then-write inside the lock).

        Delivery semantics (honest): at-least-once with idempotent outputs.
        Two workers racing the same claim in the millisecond window between
        read and write may both claim; the deterministic pipeline then
        writes identical artifacts and exactly one completion wins (the
        terminal-state lock refuses the loser). Exactly-once requires a
        database-backed repository — the upgrade path when multi-writer
        conflicts become real.
        """
        with self._lock:
            job = self._locate_locked(job_id)
            if job is None or job.status != "queued" or job.is_terminal:
                return None
            claimed = self.update(
                job_id, status="processing", stage="depth_inference", progress=5.0
            )
        return claimed

    def request_cancel(self, job_id: str) -> Job | None:
        with self._lock:
            job = self._locate_locked(job_id)
            if job is None or job.is_terminal:
                return job
            job.cancel_requested = True
            if job.status == "queued":
                job.status = "cancelled"
                job.stage = "cancelled"
                job.completed_at = utc_now()
        self._write_job_file(job)
        logger.info("cancel requested: %s", job_id)
        return job

    def delete(self, job_id: str) -> bool:
        with self._lock:
            job = self._index.pop(job_id, None)
        if job is not None:
            self._job_path(job.scene_id, job_id).unlink(missing_ok=True)
            return True
        return False

    def delete_for_scene(self, scene_id: str) -> int:
        with self._lock:
            doomed = [
                job_id for job_id, job in self._index.items()
                if job.scene_id == scene_id
            ]
            for job_id in doomed:
                del self._index[job_id]
        removed = len(doomed)
        jobs_dir = self._jobs_dir(scene_id)
        if jobs_dir.is_dir():
            for path in jobs_dir.glob("*.json"):
                path.unlink(missing_ok=True)
                removed += 1
        return removed
