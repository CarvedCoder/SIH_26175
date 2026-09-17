"""SqliteJobRepository — DB-backed JobRepository with exactly-once claims.

Same protocol, same state-machine policy (job_ops.py) as
FileJobRepository, but the storage is a single SQLite database with
transactional writes: ``claim_queued`` runs inside BEGIN IMMEDIATE, which
serializes claimers across PROCESSES — two workers can no longer both
claim the same job. This is the smallest reliable database-backed
mechanism for this project (stdlib; no new dependency, no server). The
same schema/SQL works unchanged on Postgres via a driver swap if the
deployment outgrows SQLite.

Selected with ``DW_JOB_STORE=sqlite`` (default "file"); the DB path is
``DW_JOB_DB`` (default ``<data dir>/jobs.db``).

Honest storage note: switching between the file and sqlite stores does
NOT migrate existing records — pick one per deployment.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import Any, Callable
from uuid import uuid4

from backend.app.core.logging import logger
from backend.app.domain.entities import Job, utc_now
from backend.app.infrastructure.persistence import job_ops

_SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs (
    job_id           TEXT PRIMARY KEY,
    scene_id         TEXT NOT NULL,
    status           TEXT NOT NULL,
    stage            TEXT,
    progress         REAL,
    result           TEXT,
    error            TEXT,
    message          TEXT,
    cancel_requested INTEGER NOT NULL DEFAULT 0,
    created_at       TEXT NOT NULL,
    started_at       TEXT,
    completed_at     TEXT,
    owner_pid        INTEGER,
    worker_id        TEXT,
    lease_until      TEXT,
    attempt          INTEGER NOT NULL DEFAULT 0,
    request          TEXT
);
CREATE INDEX IF NOT EXISTS idx_jobs_scene  ON jobs(scene_id);
CREATE INDEX IF NOT EXISTS idx_jobs_status ON jobs(status);
"""

_COLUMNS = (
    "job_id", "scene_id", "status", "stage", "progress", "result", "error",
    "message", "cancel_requested", "created_at", "started_at",
    "completed_at", "owner_pid", "worker_id", "lease_until", "attempt",
    "request",
)


def _to_job(row: sqlite3.Row) -> Job:
    data = {c: row[c] for c in _COLUMNS}
    for json_col in ("result", "error", "request"):
        if data[json_col] is not None:
            data[json_col] = json.loads(data[json_col])
    data["cancel_requested"] = bool(data["cancel_requested"])
    return Job.from_document(data)


def _to_row(job: Job) -> dict[str, Any]:
    data = job.to_document()
    for json_col in ("result", "error", "request"):
        if data[json_col] is not None:
            data[json_col] = json.dumps(data[json_col])
    data["cancel_requested"] = int(data["cancel_requested"])
    return data


class SqliteJobRepository:
    """JobRepository over SQLite (WAL, transactional claims)."""

    def __init__(
        self,
        *,
        db_path: Path | str,
        retention_limit: int = 500,
        ttl_seconds: int = 24 * 3600,
        lease_seconds: int = 1800,
        queued_grace_seconds: int = 60,
        worker_id: str | None = None,
        clock: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
    ) -> None:
        self._db_path = Path(db_path)
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._retention_limit = retention_limit
        self._ttl_seconds = ttl_seconds
        self._lease_seconds = lease_seconds
        self._queued_grace_seconds = queued_grace_seconds
        self.worker_id = worker_id or f"worker-{uuid4().hex[:8]}"
        self._clock = clock
        # sqlite3 connections are per-thread by contract; a small
        # thread-local pool keeps the worker + API threads isolated.
        # One shared connection, serialized by this lock (sqlite3 with
        # check_same_thread=False + an explicit mutex — the heartbeat
        # thread and API threads share this repository).
        self._lock = Lock()
        self._conn = self._connect()
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(
            str(self._db_path), timeout=30.0, check_same_thread=False
        )
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA busy_timeout=30000")
        return conn

    # -- row helpers ---------------------------------------------------------

    def _fetch(self, job_id: str) -> Job | None:
        row = self._conn.execute(
            "SELECT * FROM jobs WHERE job_id = ?", (job_id,)
        ).fetchone()
        return _to_job(row) if row is not None else None

    def _persist(self, job: Job) -> None:
        data = _to_row(job)
        cols = ", ".join(f"{c} = ?" for c in _COLUMNS)
        self._conn.execute(
            f"UPDATE jobs SET {cols} WHERE job_id = ?",
            [data[c] for c in _COLUMNS] + [job.job_id],
        )

    def _apply_liveness(self, job: Job, *, from_disk: bool) -> Job:
        reason = job_ops.interrupt_reason(
            job,
            now=self._clock(),
            queued_grace_seconds=self._queued_grace_seconds,
            from_disk=from_disk,
        )
        if reason is None and job.status == "processing" and job.lease_until is None:
            # No lease and no legacy PID record in the DB store: the record
            # predates the lease fields (imported from the file store).
            reason = "Job record predates durable leasing and cannot be trusted."
        if reason is not None:
            job_ops.finalize_interrupted(job, reason)
            with self._lock, self._conn:
                self._persist(job)
        return job

    def _evict_locked(self) -> None:
        """TTL + retention, applied lazily on create."""
        now = self._clock()
        rows = self._conn.execute(
            "SELECT job_id, created_at, status FROM jobs ORDER BY created_at"
        ).fetchall()
        expired, alive = [], []
        for row in rows:
            created = datetime.fromisoformat(row["created_at"])
            if (now - created).total_seconds() > self._ttl_seconds:
                expired.append(row["job_id"])
            else:
                alive.append(row)
        for job_id in expired:
            self._conn.execute("DELETE FROM jobs WHERE job_id = ?", (job_id,))
        if expired:
            logger.info("evicted %d expired jobs", len(expired))
        overflow = len(alive) - self._retention_limit
        if overflow > 0:
            for row in alive:
                if overflow <= 0:
                    break
                # retention drops oldest TERMINAL first
                if row["status"] in ("completed", "failed", "cancelled"):
                    self._conn.execute(
                        "DELETE FROM jobs WHERE job_id = ?", (row["job_id"],)
                    )
                    overflow -= 1

    # -- JobRepository protocol ----------------------------------------------

    def create(self, scene_id: str) -> Job:
        job = Job(job_id=f"job_{uuid4().hex[:12]}", scene_id=scene_id)
        with self._lock, self._conn:
            self._evict_locked()
            data = _to_row(job)
            self._conn.execute(
                f"INSERT INTO jobs ({', '.join(_COLUMNS)}) "
                f"VALUES ({', '.join('?' for _ in _COLUMNS)})",
                [data[c] for c in _COLUMNS],
            )
        logger.info("job created: %s scene=%s worker=%s",
                    job.job_id, scene_id, self.worker_id)
        return job

    def get(self, job_id: str) -> Job | None:
        job = self._fetch(job_id)
        if job is None:
            return None
        return self._apply_liveness(job, from_disk=True)

    def list_for_scene(self, scene_id: str) -> list[Job]:
        rows = self._conn.execute(
            "SELECT * FROM jobs WHERE scene_id = ? ORDER BY created_at",
            (scene_id,),
        ).fetchall()
        return [
            self._apply_liveness(_to_job(r), from_disk=True) for r in rows
        ]

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
        with self._lock, self._conn:  # transactional read-modify-write
            job = self._fetch(job_id)
            if job is None:
                return None
            if not job_ops.mutate(
                job, worker_id=self.worker_id,
                lease_seconds=self._lease_seconds, now=self._clock(), **allowed,
            ):
                return job  # refused (terminal lock); nothing persisted
            self._persist(job)
        return job

    def renew_lease(self, job_id: str) -> Job | None:
        with self._lock, self._conn:
            job = self._fetch(job_id)
            if job is None:
                return None
            if not job_ops.renew_lease(
                job, lease_seconds=self._lease_seconds, now=self._clock()
            ):
                return job
            self._persist(job)
        return job

    def list_queued(self) -> list[Job]:
        rows = self._conn.execute(
            "SELECT * FROM jobs WHERE status = 'queued' ORDER BY created_at"
        ).fetchall()
        return [
            self._apply_liveness(_to_job(r), from_disk=True) for r in rows
        ]

    def claim_queued(self, job_id: str) -> Job | None:
        """EXACTLY-ONCE claim: BEGIN IMMEDIATE serializes claimers across
        processes; the status check and the processing update commit
        atomically, so two workers can never both claim one job."""
        with self._lock, self._conn:
            self._conn.execute("BEGIN IMMEDIATE")
            row = self._conn.execute(
                "SELECT * FROM jobs WHERE job_id = ?", (job_id,)
            ).fetchone()
            if row is None or row["status"] != "queued":
                self._conn.rollback()
                return None
            job = _to_job(row)
            if not job_ops.mutate(
                job,
                status="processing",
                stage="depth_inference",
                progress=5.0,
                worker_id=self.worker_id,
                lease_seconds=self._lease_seconds,
                now=self._clock(),
            ):
                self._conn.rollback()
                return None
            self._persist(job)
            self._conn.commit()
            return job

    def request_cancel(self, job_id: str) -> Job | None:
        with self._lock, self._conn:
            job = self._fetch(job_id)
            if job is None or job.is_terminal:
                return job
            job.cancel_requested = True
            if job.status == "queued":
                job.status = "cancelled"
                job.stage = "cancelled"
                job.completed_at = utc_now()
            self._persist(job)
        logger.info("cancel requested: %s", job_id)
        return job

    def delete(self, job_id: str) -> bool:
        with self._lock, self._conn:
            cur = self._conn.execute(
                "DELETE FROM jobs WHERE job_id = ?", (job_id,)
            )
        return cur.rowcount > 0

    def delete_for_scene(self, scene_id: str) -> int:
        with self._lock, self._conn:
            cur = self._conn.execute(
                "DELETE FROM jobs WHERE scene_id = ?", (scene_id,)
            )
        return cur.rowcount
