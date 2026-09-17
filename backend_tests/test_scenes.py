"""Scene lifecycle tests: upload, limits, get, list, delete, validate."""

from __future__ import annotations

import io
import sys
from pathlib import Path

import numpy as np

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


# ---------------------------------------------------------------------------
# PNG / JPG inputs (first-class, matching the frontend's advertised formats)
# ---------------------------------------------------------------------------

def _png_bytes(width=512, height=384, mode="RGB"):
    from PIL import Image

    buffer = io.BytesIO()
    Image.new(mode, (width, height), color=(30, 90, 160)).save(buffer, format="PNG")
    buffer.seek(0)
    return buffer


def _jpeg_bytes(width=400, height=300):
    from PIL import Image

    buffer = io.BytesIO()
    Image.new("RGB", (width, height), color=(120, 40, 90)).save(
        buffer, format="JPEG"
    )
    buffer.seek(0)
    return buffer


def test_upload_png_scene(client):
    response = client.post(
        "/api/v1/scenes",
        files={"file": ("ortho.png", _png_bytes(), "image/png")},
    )
    assert response.status_code == 200, response.text
    scene = response.json()
    assert scene["scene_id"].startswith("scene_")
    assert scene["format"] == "PNG"
    # PNG has no CRS: honest relative-elevation scene
    assert scene["georeference"]["available"] is False
    assert scene["processing_path"] == "relative"
    assert scene["capabilities"]["absolute_elevation"] is False

    stored = client.get(f"/api/v1/scenes/{scene['scene_id']}").json()
    assert stored["format"] == "PNG"
    assert stored["dimensions"] == {"width": 512, "height": 384, "channels": 3}


def test_upload_jpeg_scene(client):
    response = client.post(
        "/api/v1/scenes",
        files={"file": ("ortho.jpg", _jpeg_bytes(), "image/jpeg")},
    )
    assert response.status_code == 200, response.text
    scene = response.json()
    assert scene["format"] == "JPEG"
    assert scene["dimensions"]["width"] == 400


def test_upload_fake_png_rejected(client):
    """A .png that is not an image: rejected by the raster parse gate."""
    response = client.post(
        "/api/v1/scenes",
        files={"file": ("fake.png", io.BytesIO(b"not an image"), "image/png")},
    )
    assert response.status_code == 400


def test_png_scene_processes_end_to_end(client, mock_inference):
    """PNG scenes run the SAME certified inference path and produce the
    same result contract (non-georeferenced: no dsm.tif, honest stats)."""
    upload = client.post(
        "/api/v1/scenes",
        files={"file": ("ortho.png", _png_bytes(), "image/png")},
    ).json()
    scene_id = upload["scene_id"]

    job = client.post(f"/api/v1/scenes/{scene_id}/process", json={}).json()
    status = client.get(f"/api/v1/jobs/{job['job_id']}").json()
    assert status["status"] == "completed", status

    # the pipeline received the stored PNG input
    assert mock_inference[0][0].name == "input.png"

    results = client.get(f"/api/v1/scenes/{scene_id}/results").json()
    assert results["job_id"] == job["job_id"]
    assert results["depth"]["available"] is True
    assert results["dsm"]["available"] is False  # no CRS -> no dsm.tif

    # point products work off the depth array
    elevation = client.get(f"/api/v1/scenes/{scene_id}/elevation?x=5&y=5")
    assert elevation.status_code == 200

    # slope is honestly refused (no GSD)
    slope = client.post(
        f"/api/v1/scenes/{scene_id}/measure/slope",
        json={"point_a": {"x": 0, "y": 0}, "point_b": {"x": 10, "y": 0}},
    ).json()
    assert slope["gsd_available"] is False
    assert slope["slope_degrees"] is None


def test_png_input_resolution_deterministic(client):
    from backend.app.core.paths import get_scene_raw_dir
    from backend.app.services.processing_service import processing_service

    upload = client.post(
        "/api/v1/scenes",
        files={"file": ("ortho.png", _png_bytes(), "image/png")},
    ).json()
    scene_id = upload["scene_id"]

    first = processing_service.resolve_scene_input(scene_id)
    second = processing_service.resolve_scene_input(scene_id)
    assert first == second == get_scene_raw_dir(scene_id) / "input.png"


def test_upload_rejects_unsupported_image_types(client):
    """Formats rasterio's contract doesn't cover for this product (e.g.
    webp) stay rejected with the honest message."""
    from PIL import Image

    buffer = io.BytesIO()
    Image.new("RGB", (64, 64)).save(buffer, format="WEBP")
    buffer.seek(0)
    response = client.post(
        "/api/v1/scenes",
        files={"file": ("pic.webp", buffer, "image/webp")},
    )
    assert response.status_code == 400
    assert "webp" in response.json()["error"]["message"].lower() or "format" in response.json()["error"]["message"].lower()
