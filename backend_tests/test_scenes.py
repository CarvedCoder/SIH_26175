"""Scene lifecycle tests: upload, limits, get, list, delete, validate."""

from __future__ import annotations

import io
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def _tiff_bytes(width=256, height=256, georeferenced=True, dtype="uint8"):
    import rasterio
    from rasterio.transform import from_origin

    buffer = io.BytesIO()
    profile = dict(
        driver="GTiff",
        height=height,
        width=width,
        count=3,
        dtype=dtype,
    )
    if georeferenced:
        profile.update(crs="EPSG:32617", transform=from_origin(500000, 4000000, 0.5, 0.5))
    with rasterio.open(buffer, "w", **profile) as dst:
        dst.write(np.zeros((3, height, width), dtype=np.uint8))
    buffer.seek(0)
    return buffer


def test_create_scene(client, uploaded_scene):
    assert uploaded_scene["scene_id"].startswith("scene_")
    assert uploaded_scene["filename"] == "ortho.tif"
    assert uploaded_scene["status"] == "ready"
    assert uploaded_scene["georeference"]["available"] is True
    assert uploaded_scene["dimensions"] == {
        "width": 256,
        "height": 256,
        "channels": 3,
    }


def test_get_scene_matches_upload_metadata(client, uploaded_scene):
    scene_id = uploaded_scene["scene_id"]
    response = client.get(f"/api/v1/scenes/{scene_id}")
    assert response.status_code == 200
    body = response.json()
    assert body["filename"] == "ortho.tif"
    assert body["georeference"]["crs"] == uploaded_scene["georeference"]["crs"]


def test_get_unknown_scene_typed_404(client):
    response = client.get("/api/v1/scenes/scene_000000000000")
    assert response.status_code == 404
    assert response.json()["error"]["code"] == "SCENE_NOT_FOUND"


def test_malformed_scene_id_rejected(client):
    response = client.get("/api/v1/scenes/not-a-scene")
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "INVALID_SCENE_ID"


def test_list_scenes(client, uploaded_scene):
    response = client.get("/api/v1/scenes")
    assert response.status_code == 200
    scenes = response.json()
    assert len(scenes) == 1
    assert scenes[0]["scene_id"] == uploaded_scene["scene_id"]
    assert scenes[0]["has_results"] is False


def test_upload_rejects_wrong_extension(client):
    response = client.post(
        "/api/v1/scenes",
        files={"file": ("virus.exe", io.BytesIO(b"MZ..."), "application/octet-stream")},
    )
    assert response.status_code == 400
    assert "format" in response.json()["error"]["message"].lower()


def test_upload_rejects_corrupt_raster(client):
    """A .tif-named file that is not a raster: rejected, nothing persisted."""
    response = client.post(
        "/api/v1/scenes",
        files={"file": ("garbage.tif", io.BytesIO(b"not a tiff at all"), "image/tiff")},
    )
    assert response.status_code == 400

    from backend.app.core.paths import SCENES_RAW_DIR

    # failed upload must not leave orphaned scene state
    assert list(SCENES_RAW_DIR.iterdir()) == []


def test_upload_size_limit_enforced(client):
    """A valid TIFF header padded beyond DW_MAX_UPLOAD_BYTES (2 MB in
    tests) is rejected with 413 — enforced DURING streaming."""
    big = _tiff_bytes()
    padded = big.getvalue() + b"\0" * (3 * 1024 * 1024)
    response = client.post(
        "/api/v1/scenes",
        files={"file": ("big.tif", io.BytesIO(padded), "image/tiff")},
    )
    assert response.status_code == 413
    assert response.json()["error"]["code"] == "PAYLOAD_TOO_LARGE"


def test_upload_failure_cleans_temp_files(client):
    from backend.app.core.paths import UPLOAD_STAGING_DIR

    client.post(
        "/api/v1/scenes",
        files={"file": ("garbage.tif", io.BytesIO(b"junk"), "image/tiff")},
    )
    assert list(UPLOAD_STAGING_DIR.iterdir()) == []


def test_validate_scene(client, uploaded_scene):
    response = client.post(f"/api/v1/scenes/{uploaded_scene['scene_id']}/validate")
    assert response.status_code == 200
    body = response.json()
    assert body["valid"] is True
    assert body["issues"] == []
    assert body["georeferenced"] is True


def test_delete_scene_removes_everything(client, processed_scene, mock_inference):
    scene_id = processed_scene["scene"]["scene_id"]

    response = client.delete(f"/api/v1/scenes/{scene_id}")
    assert response.status_code == 200
    assert response.json()["deleted"] is True

    assert client.get(f"/api/v1/scenes/{scene_id}").status_code == 404
    assert client.get(f"/api/v1/scenes/{scene_id}/results").status_code == 404

    from backend.app.core.paths import (
        SCENES_OUTPUT_DIR,
        SCENES_PROCESS_DIR,
        SCENES_RAW_DIR,
    )

    assert not (SCENES_RAW_DIR / scene_id).exists()
    assert not (SCENES_PROCESS_DIR / scene_id).exists()
    assert not (SCENES_OUTPUT_DIR / scene_id).exists()


def test_delete_unknown_scene_404(client):
    response = client.delete("/api/v1/scenes/scene_000000000000")
    assert response.status_code == 404


def test_scene_input_resolution_is_deterministic(client, uploaded_scene):
    """The designated input.tif resolves identically on repeated calls even
    if other files appear in the directory."""
    from backend.app.core.paths import get_scene_raw_dir
    from backend.app.services.processing_service import processing_service

    scene_id = uploaded_scene["scene_id"]
    raw_dir = get_scene_raw_dir(scene_id)
    # a stray raster with a lexically "smaller" name must not win
    (raw_dir / "aaa.tif").write_bytes(_tiff_bytes().getvalue())

    first = processing_service.resolve_scene_input(scene_id)
    second = processing_service.resolve_scene_input(scene_id)
    assert first == second == raw_dir / "input.tif"
