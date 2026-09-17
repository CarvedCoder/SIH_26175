"""Statelessness contract tests (refactor brief §45).

These pin the properties the refactor exists to guarantee:

    Test A  — restart: job state survives process recreation.
    Test B  — two API instances: either instance reads/observes any job,
              including updates made by the other (stale-cache regression).
    Test C  — worker restart: expired-lease processing jobs finalize
              honestly; heartbeats keep live workers alive.
    Test F  — cancellation of a job created by another instance works
              through the durable store.
    Test H  — no global job truth: destroying process-local bookkeeping
              (the write-through index) changes no read result.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.app.infrastructure.persistence.file_job_repository import (
    FileJobRepository,
)


@pytest.fixture
def storage_root(tmp_path, monkeypatch):
    """Redirect scene storage into a per-test temp dir."""
    import backend.app.core.paths as paths

    monkeypatch.setattr(paths, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(paths, "DATA_DIR", tmp_path / "data")
    monkeypatch.setattr(paths, "SCENES_PROCESS_DIR",
                        tmp_path / "data" / "process" / "scenes")
    (tmp_path / "data" / "process" / "scenes").mkdir(parents=True, exist_ok=True)
    return tmp_path


def _repo(**kw) -> FileJobRepository:
    defaults = dict(retention_limit=100, ttl_seconds=3600, lease_seconds=1800)
    defaults.update(kw)
    return FileJobRepository(**defaults)


# ---------------------------------------------------------------------------
# Test A — restart
# ---------------------------------------------------------------------------


def test_job_survives_instance_recreation(storage_root):
    """A job created by one instance is fully readable by a NEW instance
    constructed later (simulated restart): no in-memory state needed."""
    first = _repo()
    job = first.create("scene_000000000001")
    first.update(job.job_id, status="processing", stage="depth_inference")
    first.update(job.job_id, status="completed", result={"ok": True})

    fresh = _repo()
    revived = fresh.get(job.job_id)
    assert revived is not None
    assert revived.status == "completed"
    assert revived.result == {"ok": True}
    assert revived.started_at is not None
    assert revived.completed_at is not None


# ---------------------------------------------------------------------------
# Test B — two API instances observe one truth
# ---------------------------------------------------------------------------


def test_cross_instance_update_visibility(storage_root):
    """REGRESSION (audit §1.2): instance A must see instance B's updates —
    the old in-memory cache served stale state forever after first read."""
    api_one = _repo()
    api_two = _repo()

    job = api_one.create("scene_000000000001")

    # A reads first (caching era: this would freeze the record for A)
    assert api_one.get(job.job_id).status == "queued"

    # B claims and completes the job
    api_two.update(job.job_id, status="processing", stage="depth_inference")
    api_two.update(job.job_id, status="completed", result={"ok": 1})

    # A MUST observe B's terminal state now
    seen_by_a = api_one.get(job.job_id)
    assert seen_by_a.status == "completed"
    assert seen_by_a.result == {"ok": 1}

    # and B must see a cancel request raised through A on a fresh job
    job2 = api_one.create("scene_000000000001")
    assert api_two.get(job2.job_id).status == "queued"
    cancelled = api_one.request_cancel(job2.job_id)
    assert cancelled.status == "cancelled"
    assert api_two.get(job2.job_id).status == "cancelled"


# ---------------------------------------------------------------------------
# Test C — worker crash / lease expiry
# ---------------------------------------------------------------------------


def test_expired_lease_finalizes_interrupted(storage_root):
    """A processing job whose lease expired (worker died, no heartbeat)
    becomes an honest failure for ANY reader — regardless of PID."""
    repo = _repo(lease_seconds=1800)
    job = repo.create("scene_000000000001")
    repo.update(job.job_id, status="processing")

    # simulate worker death: expire the lease on disk
    path = repo._job_path(job.scene_id, job.job_id)
    data = json.loads(path.read_text())
    data["lease_until"] = (datetime.now(timezone.utc) - timedelta(seconds=1)).isoformat()
    path.write_text(json.dumps(data))

    other_instance = _repo()  # different process, no shared memory
    revived = other_instance.get(job.job_id)
    assert revived.status == "failed"
    assert revived.error["code"] == "JOB_INTERRUPTED"
    assert revived.error["recoverable"] is True


def test_lease_claim_is_recorded_durably(storage_root):
    """Claiming a job writes the durable lease fields + diagnostics, and
    the attempt counter increments per claim."""
    repo = _repo()
    job = repo.create("scene_000000000001")
    claimed = repo.update(job.job_id, status="processing")
    assert claimed.lease_until is not None
    assert claimed.worker_id == repo.worker_id
    assert claimed.attempt == 1
    # renewal pushes the lease forward and keeps the record non-terminal
    before = datetime.fromisoformat(claimed.lease_until)
    renewed = repo.renew_lease(job.job_id)
    assert datetime.fromisoformat(renewed.lease_until) >= before


def test_legacy_pid_era_record_migrates_on_read(storage_root):
    """A job file written by the PRE-lease manager (owner_pid, no lease)
    is finalized via the one-time legacy shim when its PID is dead — and
    the terminal state persists."""
    repo = _repo()
    job = repo.create("scene_000000000001")
    repo.update(job.job_id, status="processing")

    path = repo._job_path(job.scene_id, job.job_id)
    data = json.loads(path.read_text())
    legacy = {k: v for k, v in data.items()
              if k not in ("lease_until", "worker_id", "attempt")}
    legacy["owner_pid"] = 2 ** 22  # cannot exist
    path.write_text(json.dumps(legacy))

    other_instance = _repo()
    revived = other_instance.get(job.job_id)
    assert revived.status == "failed"
    assert revived.error["code"] == "JOB_INTERRUPTED"


# ---------------------------------------------------------------------------
# Test F — cancel through the durable store
# ---------------------------------------------------------------------------


def test_terminal_lock_and_cancel_across_instances(storage_root):
    repo_a, repo_b = _repo(), _repo()
    job = repo_a.create("scene_000000000001")
    repo_a.update(job.job_id, status="processing")

    # cancelling a RUNNING job cross-instance sets the durable flag; the
    # late worker's completion attempt is refused (a flagged job can only
    # end cancelled) and the worker's next checkpoint ends it cancelled.
    repo_b.request_cancel(job.job_id)
    refused = repo_b.update(job.job_id, status="completed", result={"x": 1})
    assert refused.status == "processing"
    assert refused.cancel_requested is True
    ended = repo_b.update(job.job_id, status="cancelled")
    assert ended.status == "cancelled"

    # a QUEUED job created by another instance cancels immediately
    job2 = repo_a.create("scene_000000000001")
    cancelled = repo_b.request_cancel(job2.job_id)
    assert cancelled.status == "cancelled"
    assert repo_b.update(job2.job_id, status="completed").status == "cancelled"


# ---------------------------------------------------------------------------
# Test H — no global job truth
# ---------------------------------------------------------------------------


def test_destroying_process_local_index_changes_nothing(storage_root):
    """The write-through index is bookkeeping only: wiping it (and even
    planting stale entries in it) cannot corrupt a read."""
    repo = _repo()
    job = repo.create("scene_000000000001")
    repo.update(job.job_id, status="completed", result={"v": 1})

    repo._index.clear()
    assert repo.get(job.job_id).result == {"v": 1}

    # a stale index entry must not shadow the disk record
    from backend.app.domain.entities import Job

    repo._index[job.job_id] = Job(job_id=job.job_id,
                                  scene_id=job.scene_id, status="queued")
    assert repo.get(job.job_id).status == "completed"


def test_queued_grace_still_applies_to_disk_jobs(storage_root):
    """A disk-only queued job past the grace window is failed honestly
    (never started) — pre-existing semantics, now pinned here too."""
    repo = _repo(queued_grace_seconds=60)
    job = repo.create("scene_000000000001")
    path = repo._job_path(job.scene_id, job.job_id)
    data = json.loads(path.read_text())
    data["created_at"] = (
        datetime.now(timezone.utc) - timedelta(seconds=120)
    ).isoformat()
    path.write_text(json.dumps(data))

    fresh = _repo(queued_grace_seconds=60)
    assert fresh.get(job.job_id).status == "failed"
    assert fresh.get(job.job_id).error["code"] == "JOB_INTERRUPTED"


# ---------------------------------------------------------------------------
# Health split (brief §32): liveness touches nothing; readiness is cheap
# ---------------------------------------------------------------------------


def test_health_endpoints_split(client):
    live = client.get("/api/v1/health/live")
    assert live.status_code == 200
    assert live.json()["status"] == "ok"

    ready = client.get("/api/v1/health/ready")
    assert ready.status_code == 200
    assert ready.json()["status"] in ("ok", "degraded")

    # the canonical contract endpoint is unchanged
    canon = client.get("/api/v1/health")
    assert canon.status_code == 200
    assert set(canon.json()) == {"status", "version", "model_loaded"}


# ---------------------------------------------------------------------------
# Test E — two workers claim from one durable store
# ---------------------------------------------------------------------------


def test_two_workers_claim_disjoint_jobs(storage_root):
    """Two worker instances polling the same store claim jobs without
    corrupting state: each claim flips queued→processing exactly once, and
    the second worker's claim of the same job returns None."""
    repo_a, repo_b = _repo(), _repo()
    j1 = repo_a.create("scene_000000000001")
    j2 = repo_a.create("scene_000000000002")

    discovered = sorted(j.job_id for j in repo_b.list_queued())
    assert discovered == sorted([j1.job_id, j2.job_id])

    # both workers discover the same queue but each job is claimed once
    first = repo_b.claim_queued(j1.job_id)
    assert first.status == "processing"
    assert repo_b.claim_queued(j1.job_id) is None  # no longer queued

    # a claim of an unknown/cancelled job is refused
    assert repo_b.claim_queued("job_000000000000") is None
    repo_a.request_cancel(j2.job_id)
    assert repo_b.claim_queued(j2.job_id) is None


def test_worker_executes_job_from_record_alone(storage_root, monkeypatch):
    """A worker needs ONLY the durable record: execute_job_record reads
    request parameters from the job document, not from any in-memory
    request state."""
    from backend.app.services.processing_service import ProcessingService

    repo = _repo()
    job = repo.create("scene_000000000001")
    repo.update(job.job_id, request={
        "kind": "process", "mode": "auto", "ground_elev": None,
    })
    repo.claim_queued(job.job_id)

    svc = ProcessingService()
    calls: list[tuple[str, str, float | None]] = []

    def fake_process(job_id, scene_id, *, mode="auto", ground_elev=None):
        calls.append((job_id, scene_id, ground_elev))
        repo.update(job_id, status="completed", result={"cancelled": False})
        return {"cancelled": False}

    monkeypatch.setattr(svc, "process_scene", fake_process)
    result = svc.execute_job_record(job.job_id)
    assert result == {"cancelled": False}
    assert calls == [(job.job_id, "scene_000000000001", None)]
    assert repo.get(job.job_id).status == "completed"
