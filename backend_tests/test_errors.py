"""Error envelope contract tests (Phase 5)."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def _error_shape(body: dict) -> dict:
    error = body["error"]
    assert set(error.keys()) == {"code", "message", "details", "recoverable"}
    assert isinstance(error["code"], str)
    assert isinstance(error["message"], str)
    assert isinstance(error["recoverable"], bool)
    return error


def test_404_unknown_route_envelope(client):
    response = client.get("/api/v1/does-not-exist")
    assert response.status_code == 404
    error = _error_shape(response.json())
    assert error["code"] == "NOT_FOUND"
    assert "Traceback" not in error["message"]


def test_405_method_not_allowed_envelope(client):
    response = client.delete("/api/v1/health")
    assert response.status_code == 405
    error = _error_shape(response.json())
    assert error["code"] == "METHOD_NOT_ALLOWED"


def test_422_validation_error_envelope(client):
    response = client.post("/api/v1/scenes")
    assert response.status_code == 422
    error = _error_shape(response.json())
    assert error["code"] == "VALIDATION_ERROR"
    assert error["recoverable"] is True


def test_404_scene_envelope_has_no_filesystem_paths(client):
    response = client.get("/api/v1/scenes/scene_000000000000")
    error = _error_shape(response.json())
    assert "/home" not in error["message"]
    assert str(response.request.url) not in error["message"]


def test_error_response_carries_request_id_header(client):
    response = client.get("/api/v1/scenes/scene_000000000000")
    assert len(response.headers.get("X-Request-ID", "")) == 12


def test_client_supplied_request_id_is_honored(client):
    response = client.get(
        "/api/v1/scenes/scene_000000000000",
        headers={"X-Request-ID": "my-trace-id"},
    )
    assert response.headers["X-Request-ID"] == "my-trace-id"


def test_internal_error_sanitized_in_results(client, uploaded_scene, monkeypatch):
    """An unexpected exception in a SYNC route becomes a 500 envelope; the
    traceback stays server-side (logged), never in the response."""
    from backend.app.api.routes import results as results_mod

    def exploding(scene_id):
        raise RuntimeError("/secret/path/leaked via boom")

    monkeypatch.setattr(results_mod.result_service, "get_result_files", exploding)

    response = client.get(f"/api/v1/scenes/{uploaded_scene['scene_id']}/results")
    assert response.status_code == 500
    error = _error_shape(response.json())
    assert error["code"] == "INTERNAL_ERROR"
    assert "/secret/path" not in error["message"]
    assert error["recoverable"] is True


def test_background_failure_records_typed_job_error(
    client, uploaded_scene, mock_inference, monkeypatch
):
    """A worker crash must leave a FAILED job with a typed error — never a
    job stuck in 'processing'."""
    from backend.app.services import processing_service as ps

    def exploding(*args, **kwargs):
        raise RuntimeError("/secret/path/leaked via boom")

    monkeypatch.setattr(ps, "run_inference", exploding)

    response = client.post(f"/api/v1/scenes/{uploaded_scene['scene_id']}/process", json={})
    job_id = response.json()["job_id"]

    job = client.get(f"/api/v1/jobs/{job_id}").json()
    assert job["status"] == "failed"
    assert job["error"]["code"] == "PROCESSING_FAILED"
    # the raw exception text (and any paths in it) never reaches the client
    assert "/secret/path" not in job["error"]["message"]
