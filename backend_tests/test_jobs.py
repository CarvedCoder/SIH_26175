"""Job orchestration tests: process, poll, cancel, bounds, contract."""

from __future__ import annotations

import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def test_process_unknown_scene_404_no_job_created(client):
    """Audit H2: unknown scenes must 404 BEFORE any job exists."""
    response = client.post("/api/v1/scenes/scene_000000000000/process", json={})
    assert response.status_code == 404
    assert response.json()["error"]["code"] == "SCENE_NOT_FOUND"

    from backend.app.jobs.manager import job_manager

    assert job_manager.get_active_job_for_scene("scene_000000000000") is None


def test_process_known_scene_completes(client, uploaded_scene, mock_inference):
    scene_id = uploaded_scene["scene_id"]
    response = client.post(f"/api/v1/scenes/{scene_id}/process", json={})
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "queued"
    assert body["job_id"].startswith("job_")

    job = client.get(f"/api/v1/jobs/{body['job_id']}").json()
    assert job["status"] == "completed"
    assert job["progress"] == 100.0
    assert job["result"]["ok"] is True
    # the mocked pipeline was really called with the scene's input
    assert mock_inference[0][0].name == "input.tif"


def test_process_job_response_is_typed(client, uploaded_scene, mock_inference):
    response = client.post(f"/api/v1/scenes/{uploaded_scene['scene_id']}/process", json={})
    body = response.json()
    job = client.get(f"/api/v1/jobs/{body['job_id']}").json()
    for key in (
        "job_id",
        "scene_id",
        "status",
        "stage",
        "progress",
        "result",
        "error",
        "created_at",
        "started_at",
        "completed_at",
        "cancel_requested",
    ):
        assert key in job


def test_duplicate_process_conflicts(client, uploaded_scene, monkeypatch):
    """A scene with an active job cannot be processed again (409)."""
    import backend.app.api.routes.jobs as jobs_mod

    # keep the job queued forever (no worker runs) so the scene stays busy
    # tranche-2 seam: jobs dispatch through execute_job_record; patching it
    # keeps the job queued forever (the test wants a busy scene)
    from backend.app.services.processing_service import processing_service as _ps
    monkeypatch.setattr(_ps, "execute_job_record", lambda job_id: None)

    scene_id = uploaded_scene["scene_id"]
    first = client.post(f"/api/v1/scenes/{scene_id}/process", json={})
    assert first.status_code == 200

    second = client.post(f"/api/v1/scenes/{scene_id}/process", json={})
    assert second.status_code == 409
    assert second.json()["error"]["code"] == "SCENE_BUSY"


def test_process_rejects_unsupported_options(client, uploaded_scene):
    """The strict contract: phantom options get an explicit 422."""
    for payload in (
        {"model": "depth-anything-v2"},
        {"tile_size": 512},
        {"overlap": 0.15},
        {"enable_refinement": True},
        {"enable_reference_calibration": True},
    ):
        response = client.post(
            f"/api/v1/scenes/{uploaded_scene['scene_id']}/process", json=payload
        )
        assert response.status_code == 422, payload
        assert response.json()["error"]["code"] == "VALIDATION_ERROR"


def test_process_rejects_bad_mode(client, uploaded_scene):
    response = client.post(
        f"/api/v1/scenes/{uploaded_scene['scene_id']}/process",
        json={"mode": "mega"},
    )
    assert response.status_code == 422


def test_ground_elev_passed_to_inference(client, uploaded_scene, mock_inference):
    scene_id = uploaded_scene["scene_id"]
    client.post(f"/api/v1/scenes/{scene_id}/process", json={"ground_elev": 120.5})
    _input_path, kwargs = mock_inference[0]
    assert kwargs["ground_elev"] == 120.5


# ---------------------------------------------------------------------------
# Height-model backend switch (RDAH integration)
# ---------------------------------------------------------------------------

def test_architecture_passed_to_inference(client, uploaded_scene, mock_inference):
    """The frontend's backend switch rides through to run_inference."""
    scene_id = uploaded_scene["scene_id"]
    client.post(
        f"/api/v1/scenes/{scene_id}/process",
        json={"architecture": "calibration_net"},
    )
    _input_path, kwargs = mock_inference[0]
    assert kwargs["architecture"] == "calibration_net"


def test_architecture_auto_omitted_defaults_to_checkpoint(
    client, uploaded_scene, mock_inference
):
    """Legacy clients that omit architecture follow the configured
    checkpoint (auto) — never a silent backend swap."""
    scene_id = uploaded_scene["scene_id"]
    client.post(f"/api/v1/scenes/{scene_id}/process", json={})
    _input_path, kwargs = mock_inference[0]
    # the fixture's DW_CKPT is a calibration_net checkpoint
    assert kwargs["architecture"] == "calibration_net"


def test_rdah_with_calibration_env_checkpoint_fails_loudly(
    client, uploaded_scene, mock_inference
):
    """A DW_CKPT pointing at a calibration checkpoint while rdah is
    requested is a typed, recoverable failure — never a silent swap."""
    import time

    scene_id = uploaded_scene["scene_id"]
    response = client.post(
        f"/api/v1/scenes/{scene_id}/process",
        json={"architecture": "rdah"},
    )
    assert response.status_code == 200  # job accepted...
    job_id = response.json()["job_id"]

    # the inline worker runs on its own thread — poll to terminal state
    for _ in range(50):
        job = client.get(f"/api/v1/jobs/{job_id}").json()
        if job["status"] in ("failed", "completed", "cancelled"):
            break
        time.sleep(0.1)

    assert job["status"] == "failed"
    assert job["error"]["code"] == "INVALID_INPUT"
    assert "DW_CKPT" in job["error"]["message"]


def test_process_rejects_unknown_architecture(client, uploaded_scene):
    response = client.post(
        f"/api/v1/scenes/{uploaded_scene['scene_id']}/process",
        json={"architecture": "depth_anything"},
    )
    assert response.status_code == 422


def test_job_status_unknown_404(client):
    response = client.get("/api/v1/jobs/job_000000000000")
    assert response.status_code == 404
    assert response.json()["error"]["code"] == "JOB_NOT_FOUND"


def test_cancel_queued_job_immediately(client, uploaded_scene, monkeypatch):
    """A queued job is cancelled synchronously and never starts."""
    import backend.app.api.routes.jobs as jobs_mod

    # tranche-2 seam: jobs dispatch through execute_job_record; patching it
    # keeps the job queued forever (the test wants a busy scene)
    from backend.app.services.processing_service import processing_service as _ps
    monkeypatch.setattr(_ps, "execute_job_record", lambda job_id: None)

    scene_id = uploaded_scene["scene_id"]
    response = client.post(f"/api/v1/scenes/{scene_id}/process", json={})
    job_id = response.json()["job_id"]

    job = client.get(f"/api/v1/jobs/{job_id}").json()
    assert job["status"] == "queued"

    cancel = client.post(f"/api/v1/jobs/{job_id}/cancel")
    assert cancel.status_code == 200
    body = cancel.json()
    assert body["status"] == "cancelled"
    assert body["cancel_requested"] is True

    job = client.get(f"/api/v1/jobs/{job_id}").json()
    assert job["status"] == "cancelled"


def test_cancel_unknown_job_404(client):
    response = client.post("/api/v1/jobs/job_000000000000/cancel")
    assert response.status_code == 404


# ---------------------------------------------------------------------------
# Job store bounds (Phase 8)
# ---------------------------------------------------------------------------

def test_job_store_bounded_by_retention(fresh_job_manager):
    """Terminal jobs are evicted beyond the retention limit; ACTIVE jobs
    are never evicted (clients are polling them)."""
    # 20 completed jobs into a limit-8 store
    for _ in range(20):
        job = fresh_job_manager.create_job("scene_000000000001")
        fresh_job_manager.update_job(job.job_id, status="completed")
    assert len(fresh_job_manager._jobs) <= 8

    # a running job must survive an overflow eviction cycle
    active = fresh_job_manager.create_job("scene_000000000002")
    for _ in range(10):
        fresh_job_manager.create_job("scene_000000000003")
    assert fresh_job_manager.get_job(active.job_id) is not None


def test_job_store_ttl_eviction(fresh_job_manager):
    from datetime import datetime, timedelta, timezone

    job = fresh_job_manager.create_job("scene_000000000001")
    # age the job past the TTL
    stale = (
        datetime.now(timezone.utc) - timedelta(seconds=fresh_job_manager._ttl_seconds + 10)
    ).isoformat()
    job.created_at = stale

    fresh_job_manager.create_job("scene_000000000002")
    assert fresh_job_manager.get_job(job.job_id) is None


def test_terminal_status_locked_after_cancellation(fresh_job_manager):
    """A worker finishing after a cancel cannot flip the job to completed."""
    job = fresh_job_manager.create_job("scene_000000000001")
    cancelled = fresh_job_manager.request_cancel(job.job_id)
    assert cancelled is not None and cancelled.status == "cancelled"
    assert fresh_job_manager.get_job(job.job_id).status == "cancelled"

    updated = fresh_job_manager.update_job(
        job.job_id, status="completed", result={"ok": True}
    )
    assert updated.status == "cancelled"
    assert updated.result is None


def test_update_missing_job_returns_none(fresh_job_manager):
    assert fresh_job_manager.update_job("job_000000000000", status="completed") is None


# ---------------------------------------------------------------------------
# Cancellation of a RUNNING job (cooperative)
# ---------------------------------------------------------------------------

def test_cancel_during_inference_ends_cancelled(
    client, uploaded_scene, mock_inference, monkeypatch
):
    """A cancellation requested while the worker is inside inference must
    end the job cancelled, with no result and no partial outputs."""
    from backend.app.core.paths import get_scene_output_dir
    from backend.app.services import processing_service as ps

    scene_id = uploaded_scene["scene_id"]
    fake_run = ps.run_inference  # installed by the mock_inference fixture

    def run_and_cancel(*args, **kwargs):
        # simulate the user pressing cancel while inference is running
        job = ps.job_manager.get_active_job_for_scene(scene_id)
        assert job is not None
        ps.job_manager.request_cancel(job.job_id)
        return fake_run(*args, **kwargs)

    monkeypatch.setattr(ps, "run_inference", run_and_cancel)

    response = client.post(f"/api/v1/scenes/{scene_id}/process", json={})
    job_id = response.json()["job_id"]

    job = client.get(f"/api/v1/jobs/{job_id}").json()
    assert job["status"] == "cancelled"
    assert job["result"] is None

    # partial outputs were discarded
    assert not (get_scene_output_dir(scene_id) / "dsm.npy").exists()


# ---------------------------------------------------------------------------
# Disk-backed (stateless) semantics: restart survival + interrupted jobs
# ---------------------------------------------------------------------------


def test_completed_job_survives_manager_restart(fresh_job_manager):
    """A NEW JobManager instance (simulating a server restart) must read the
    same job state from disk — in-memory state is only a cache."""
    from backend.app.jobs.manager import JobManager

    job = fresh_job_manager.create_job("scene_000000000001")
    fresh_job_manager.update_job(
        job.job_id, status="completed", result={"elevation_mode": "absolute"}
    )

    restarted = JobManager(retention_limit=8, ttl_seconds=3600)
    revived = restarted.get_job(job.job_id)
    assert revived is not None
    assert revived.status == "completed"
    assert revived.result == {"elevation_mode": "absolute"}


def test_orphaned_processing_job_fails_on_read_after_restart(fresh_job_manager):
    """A job left 'processing' whose DURABLE LEASE expired must become an
    honest JOB_INTERRUPTED failure on the next read — never stuck at
    'processing' forever. (Tranche-1 contract: liveness is the lease, not
    a PID; the legacy PID-format migration has its own test.)"""
    from datetime import datetime, timedelta, timezone

    from backend.app.jobs.manager import JobManager

    job = fresh_job_manager.create_job("scene_000000000001")
    fresh_job_manager.update_job(job.job_id, status="processing")

    # Simulate the restart: the on-disk record's lease expired and the
    # fresh manager has no memory of the job.
    path = fresh_job_manager._job_path(job.scene_id, job.job_id)
    import json

    data = json.loads(path.read_text())
    assert data["lease_until"] is not None  # new records carry a lease
    data["lease_until"] = (
        datetime.now(timezone.utc) - timedelta(seconds=1)
    ).isoformat()
    path.write_text(json.dumps(data))

    restarted = JobManager(retention_limit=8, ttl_seconds=3600)
    revived = restarted.get_job(job.job_id)
    assert revived.status == "failed"
    assert revived.error["code"] == "JOB_INTERRUPTED"
    # and the failed state is persisted for every future reader
    assert restarted.get_job(job.job_id).status == "failed"


def test_heartbeat_keeps_leased_job_alive(fresh_job_manager, monkeypatch):
    """A worker renewing its lease is never finalized as interrupted —
    even past the original lease window."""
    import time

    from backend.app.jobs.manager import JobManager

    monkeypatch.setattr(fresh_job_manager._repo, "_lease_seconds", 0.2)
    job = fresh_job_manager.create_job("scene_000000000001")
    fresh_job_manager.update_job(job.job_id, status="processing")

    # simulate ~0.5s of "inference" with heartbeats every 0.05s
    deadline = time.monotonic() + 0.5
    while time.monotonic() < deadline:
        fresh_job_manager._repo.renew_lease(job.job_id)
        time.sleep(0.05)

    restarted = JobManager(retention_limit=8, ttl_seconds=3600)
    assert restarted.get_job(job.job_id).status == "processing"


def test_get_job_handles_disaster_stages(client, fresh_job_manager):
    """GET /api/v1/jobs/{job_id} must serialize disaster_assessment and disaster_complete stages."""
    from backend.app.jobs.manager import job_manager

    job = job_manager.create_job("scene_000000000001")
    job_manager.update_job(
        job.job_id,
        status="processing",
        stage="disaster_assessment",
        progress=85.0,
        message="Assessing building damage",
    )

    resp = client.get(f"/api/v1/jobs/{job.job_id}")
    assert resp.status_code == 200
    data = resp.json()
    assert data["stage"] == "disaster_assessment"
    assert data["status"] == "processing"

    job_manager.update_job(
        job.job_id,
        stage="disaster_complete",
        progress=95.0,
    )
    resp = client.get(f"/api/v1/jobs/{job.job_id}")
    assert resp.status_code == 200
    assert resp.json()["stage"] == "disaster_complete"

    # Also test unrecognized stage fallback gracefully maps without crashing
    job_manager.update_job(job.job_id, stage="some_unrecognized_custom_stage")
    resp = client.get(f"/api/v1/jobs/{job.job_id}")
    assert resp.status_code == 200
    assert resp.json()["stage"] == "finalizing"

