"""Core domain entities (tranche 1: the Job aggregate).

The Job dataclass is the canonical representation; the API schema
(`schemas/job.py`) and the persistence document (JSON) both derive from
it. Legacy job files written before the lease fields existed load fine —
missing fields take their defaults (see FileJobRepository's migration
note).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any

TERMINAL_STATUSES = {"completed", "failed", "cancelled"}
ACTIVE_STATUSES = {"queued", "processing"}

# Explicit one-way state machine (terminal states are LOCKED):
#   queued -> processing -> completed | failed
#   queued -> cancelled
#   processing -> cancelled (cancel_requested honored at a checkpoint)


def utc_now() -> str:
    """ISO-8601 UTC timestamp (the persistence format for all datetimes)."""
    return datetime.now(timezone.utc).isoformat()


@dataclass
class Job:
    """A unit of asynchronous processing work.

    Durability contract (tranche 1):
      * persisted by the JobRepository as the source of truth;
      * ``worker_id`` / ``owner_pid`` are DIAGNOSTIC — they never decide
        correctness. Liveness is carried by the durable
        ``lease_until``/``attempt`` fields: a processing job whose lease
        expired (and was not renewed by its worker's heartbeat) is
        honestly failed on the next read, regardless of which process,
        host, or container the worker ran on.
    """

    job_id: str
    scene_id: str
    status: str = "queued"
    stage: str = "queued"
    progress: float | None = None
    result: dict[str, Any] | None = None
    error: dict[str, Any] | None = None
    message: str | None = None
    cancel_requested: bool = False
    created_at: str = field(default_factory=utc_now)
    started_at: str | None = None
    completed_at: str | None = None
    # -- execution bookkeeping (diagnostic, never truth) -------------------
    owner_pid: int | None = None  # process that claimed the job (logs only)
    worker_id: str | None = None  # logical worker identity (logs only)
    # -- durable liveness ---------------------------------------------------
    lease_until: str | None = None  # ISO ts; expired non-terminal => failed
    attempt: int = 0  # incremented on every claim
    # -- work payload ---------------------------------------------------------
    # The validated request parameters (kind + options), persisted so ANY
    # worker — not just the API process that accepted the HTTP request —
    # can execute the job. request + artifacts + config = everything a
    # fresh worker needs (statelessness definition, brief §2).
    request: dict[str, Any] | None = None

    @property
    def is_terminal(self) -> bool:
        return self.status in TERMINAL_STATUSES

    @property
    def is_active(self) -> bool:
        return self.status in ACTIVE_STATUSES

    @property
    def created_dt(self) -> datetime:
        return datetime.fromisoformat(self.created_at)

    def to_document(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_document(cls, data: dict[str, Any]) -> "Job":
        return cls(**data)
