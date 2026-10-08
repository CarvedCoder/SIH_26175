"""Lazy two-phase processing contract.

Default behaviour (DWI "fast-first" delivery): POST /scenes/{id}/process
completes as soon as the elevation products exist (stage ``dsm_ready``),
and the slower analysis tail — reference validation, disaster assessment,
3D building reconstruction — runs ONLY when POST /scenes/{id}/analyze is
called. ``eager_analysis=true`` restores the single-job behaviour.
"""

from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def fresh_managers(monkeypatch):
    """Bind ONE per-test job manager everywhere the job store is resolved.

    The route modules (jobs, results) capture ``job_manager`` at first
    import, so without this the routes keep the FIRST test's manager —
    whose retention-limited index spans the whole session and can evict
    this test's job files from disk mid-test.
    """
    import backend.app.api.routes.jobs as jobs_routes
    import backend.app.api.routes.results as results_routes
    import backend.app.jobs.manager as manager_module

    mgr = manager_module.JobManager(retention_limit=50, ttl_seconds=3600)
    monkeypatch.setattr(manager_module, "job_manager", mgr)
    monkeypatch.setattr(jobs_routes, "job_manager", mgr)
    monkeypatch.setattr(results_routes, "job_manager", mgr)
    return mgr


def _wait_for_job(client, job_id: str) -> dict:
    for _ in range(100):
        job = client.get(f"/api/v1/jobs/{job_id}").json()
        if job["status"] in {"completed", "failed", "cancelled"}:
            return job
    raise AssertionError(f"job {job_id} never reached a terminal state")


def _stage_calls(processing_service, monkeypatch):
    """Replace every deferred analysis stage with a recorder."""
    calls: list[str] = []

    def fake_validation(scene_id, input_path, output_dir):
        calls.append("validation")

    def fake_disaster(job_id, scene_id, input_path, output_dir):
        calls.append("disaster")
        return {"buildings_available": False, "damage_available": False}

    def fake_buildings3d(job_id, scene_id, output_dir):
        calls.append("buildings3d")
        return {"available": False, "count": 0}

    monkeypatch.setattr(processing_service, "_write_validation_artifacts", fake_validation)
    monkeypatch.setattr(processing_service, "_run_disaster_stage", fake_disaster)
    monkeypatch.setattr(processing_service, "_run_buildings3d_stage", fake_buildings3d)
    return calls


def test_lazy_process_completes_at_dsm_ready(
    client, uploaded_scene, mock_inference, monkeypatch
):
    """Default process job: completed with stage dsm_ready, analysis
    deferred — no analysis stage may run."""
    import backend.app.services.processing_service as ps

    calls = _stage_calls(ps.processing_service, monkeypatch)

    scene_id = uploaded_scene["scene_id"]
    resp = client.post(f"/api/v1/scenes/{scene_id}/process", json={})
    assert resp.status_code == 200, resp.text
    job = _wait_for_job(client, resp.json()["job_id"])

    assert job["status"] == "completed"
    assert job["stage"] == "dsm_ready"
    assert job["result"]["analysis_status"] == "pending"
    assert job["result"]["analysis_pending"] is True
    assert calls == []  # the analysis tail never ran


def test_results_report_pending_analysis(
    client, uploaded_scene, mock_inference
):
    """GET /scenes/{id}/results surfaces the deferred-analysis state."""
    scene_id = uploaded_scene["scene_id"]
    resp = client.post(f"/api/v1/scenes/{scene_id}/process", json={})
    _wait_for_job(client, resp.json()["job_id"])

    results = client.get(f"/api/v1/scenes/{scene_id}/results").json()
    assert results["analysis_status"] == "pending"


def test_analyze_requires_processed_scene(client, uploaded_scene):
    """Analyzing a scene with no elevation products is a typed 409."""
    scene_id = uploaded_scene["scene_id"]
    resp = client.post(f"/api/v1/scenes/{scene_id}/analyze")
    assert resp.status_code == 409
    assert resp.json()["error"]["code"] == "SCENE_NOT_PROCESSED"


def test_analyze_runs_deferred_stages_and_completes(
    client, uploaded_scene, mock_inference, monkeypatch
):
    """POST /analyze on a dsm_ready scene runs validation → disaster →
    buildings3d, merges into the original payload and reports complete."""
    import backend.app.services.processing_service as ps

    calls = _stage_calls(ps.processing_service, monkeypatch)

    scene_id = uploaded_scene["scene_id"]
    resp = client.post(f"/api/v1/scenes/{scene_id}/process", json={})
    process_job = _wait_for_job(client, resp.json()["job_id"])
    assert process_job["stage"] == "dsm_ready"
    assert calls == []

    resp = client.post(f"/api/v1/scenes/{scene_id}/analyze")
    assert resp.status_code == 200, resp.text
    job = _wait_for_job(client, resp.json()["job_id"])

    assert job["status"] == "completed"
    assert job["stage"] == "completed"
    assert calls == ["validation", "disaster", "buildings3d"]
    assert job["result"]["analysis_status"] == "complete"
    assert "analysis_pending" not in job["result"]
    assert job["result"]["disaster"] == {
        "buildings_available": False, "damage_available": False,
    }
    # meta/provenance from the original processing job survives the merge
    assert job["result"]["meta"]["model_tag"] == "mock"

    results = client.get(f"/api/v1/scenes/{scene_id}/results").json()
    assert results["analysis_status"] == "complete"


def test_eager_analysis_restores_single_job(
    client, uploaded_scene, mock_inference, monkeypatch
):
    """eager_analysis=true runs everything in the process job (legacy)."""
    import backend.app.services.processing_service as ps

    calls = _stage_calls(ps.processing_service, monkeypatch)

    scene_id = uploaded_scene["scene_id"]
    resp = client.post(
        f"/api/v1/scenes/{scene_id}/process", json={"eager_analysis": True}
    )
    assert resp.status_code == 200, resp.text
    job = _wait_for_job(client, resp.json()["job_id"])

    assert job["status"] == "completed"
    assert job["stage"] == "completed"
    assert calls == ["validation", "disaster", "buildings3d"]
    assert job["result"]["analysis_status"] == "complete"


def test_analyze_is_rejected_while_job_active(
    client, uploaded_scene, mock_inference, monkeypatch
):
    """A queued analyze job takes the scene-busy guard like any other."""
    scene_id = uploaded_scene["scene_id"]
    resp = client.post(f"/api/v1/scenes/{scene_id}/process", json={})
    _wait_for_job(client, resp.json()["job_id"])

    # occupy the scene with a hung analyze job
    import backend.app.services.processing_service as ps

    def hung_analysis(job_id, scene_id):
        import time

        time.sleep(30)

    monkeypatch.setattr(ps.processing_service, "continue_analysis", hung_analysis)
    resp = client.post(f"/api/v1/scenes/{scene_id}/analyze")
    assert resp.status_code == 200
    busy = client.post(f"/api/v1/scenes/{scene_id}/analyze")
    assert busy.status_code == 409
