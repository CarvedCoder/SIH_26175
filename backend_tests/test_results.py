"""Results + export contract tests (Phase 22): honest state, allowlisted
files, real URLs."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def test_results_of_processed_scene(client, processed_scene):
    scene_id = processed_scene["scene"]["scene_id"]
    response = client.get(f"/api/v1/scenes/{scene_id}/results")
    assert response.status_code == 200
    body = response.json()

    # REAL job state — no fabricated "unknown" ids
    assert body["job_id"] == processed_scene["job_id"]
    assert body["status"] == "completed"
    assert body["depth"]["available"] is True
    assert body["dsm"]["available"] is True
    assert body["dsm"]["crs"] == "EPSG:32617"
    assert body["depth"]["statistics"]["units"] == "meters"

    # every advertised URL must be a real, servable endpoint
    for asset in body["assets"]:
        url = asset["url"]
        assert url.startswith("/api/v1/")
        status = client.get(url).status_code
        assert status == 200, url


def test_results_without_job_report_null_provenance(client, uploaded_scene, mock_inference):
    """Result files exist but the job record was removed everywhere (memory
    AND the on-disk job store): status reflects the artifacts, and job_id is
    honest null — never 'unknown'. Note: with the disk-backed job store,
    clearing only the in-memory cache no longer hides provenance — the
    persisted job is correctly still reported."""
    scene_id = uploaded_scene["scene_id"]
    client.post(f"/api/v1/scenes/{scene_id}/process", json={})

    from backend.app.jobs import manager as manager_module

    # SQL-backed store: deleting the persisted rows removes provenance —
    # there is no separate in-memory cache to clear any more.
    manager_module.job_manager.delete_jobs_for_scene(scene_id)

    body = client.get(f"/api/v1/scenes/{scene_id}/results").json()
    assert body["status"] == "completed"
    assert body["job_id"] is None


def test_results_unknown_scene_404(client):
    response = client.get("/api/v1/scenes/scene_000000000000/results")
    assert response.status_code == 404


def test_results_of_unprocessed_scene_404(client, uploaded_scene):
    response = client.get(f"/api/v1/scenes/{uploaded_scene['scene_id']}/results")
    assert response.status_code == 404
    assert response.json()["error"]["code"] == "RESULTS_NOT_FOUND"


def test_depth_dsm_metadata_endpoints(client, processed_scene):
    """The frontend layer-metadata contract (url + download_url)."""
    scene_id = processed_scene["scene"]["scene_id"]

    depth = client.get(f"/api/v1/scenes/{scene_id}/depth").json()
    assert depth["available"] is True
    # The interactive texture URL must be the Pillow-generated greyscale
    # depth texture — NEVER the Matplotlib diagnostic preview.
    assert depth["url"] == f"/api/v1/scenes/{scene_id}/results/depth-texture"
    assert depth["download_url"] == f"/api/v1/scenes/{scene_id}/results/depth"
    assert client.get(depth["download_url"]).status_code == 200
    assert client.get(depth["url"]).status_code == 200

    dsm = client.get(f"/api/v1/scenes/{scene_id}/dsm").json()
    assert dsm["available"] is True
    assert dsm["download_url"] == f"/api/v1/scenes/{scene_id}/results/dsm"
    assert dsm["crs"] == "EPSG:32617"
    assert client.get(dsm["download_url"]).status_code == 200


def test_result_file_allowlist_blocks_arbitrary_names(client, processed_scene):
    scene_id = processed_scene["scene"]["scene_id"]
    response = client.get(f"/api/v1/scenes/{scene_id}/results/scene.json")
    assert response.status_code == 404


def test_export_routes(client, processed_scene):
    scene_id = processed_scene["scene"]["scene_id"]

    dsm_export = client.get(f"/api/v1/scenes/{scene_id}/export/dsm")
    assert dsm_export.status_code == 200
    assert dsm_export.headers["content-type"].startswith("image/tiff")

    depth_export = client.get(f"/api/v1/scenes/{scene_id}/export/depth")
    assert depth_export.status_code == 200

    # unsupported type -> typed 400; missing artifact -> honest 404
    bad = client.get(f"/api/v1/scenes/{scene_id}/export/nonexistent")
    assert bad.status_code == 400
    assert bad.json()["error"]["code"] == "INVALID_EXPORT_TYPE"

    # terrain falls back to the georeferenced DSM raster (the pipeline's
    # actual terrain product) — it must be a real file, not a 404 lie:
    terrain_export = client.get(f"/api/v1/scenes/{scene_id}/export/terrain")
    assert terrain_export.status_code == 200


def test_export_unknown_scene_404(client):
    response = client.get("/api/v1/scenes/scene_000000000000/export/dsm")
    assert response.status_code == 404


# ---------------------------------------------------------------------------
# Validation / reference routes
# ---------------------------------------------------------------------------

def test_validation_honest_unavailable(client, processed_scene):
    """No reference data exists -> validation reports unavailable, not fake
    metrics."""
    scene_id = processed_scene["scene"]["scene_id"]
    body = client.get(f"/api/v1/scenes/{scene_id}/validation").json()
    assert body["scene_id"] == scene_id
    assert body["metrics"]["sample_count"] in (None, 0)


def test_validation_error_map_404_without_artifact(client, processed_scene):
    scene_id = processed_scene["scene"]["scene_id"]
    response = client.get(f"/api/v1/scenes/{scene_id}/validation/error-map")
    assert response.status_code == 404


def test_reference_honest_unavailable(client, processed_scene):
    scene_id = processed_scene["scene"]["scene_id"]
    body = client.get(f"/api/v1/scenes/{scene_id}/reference").json()
    assert body["available"] is False
    assert body["download_url"] is None


def test_validation_unknown_scene_404(client):
    assert (
        client.get("/api/v1/scenes/scene_000000000000/validation").status_code == 404
    )
