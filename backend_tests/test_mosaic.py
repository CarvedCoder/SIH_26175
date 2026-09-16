"""API-level tests for the scene mosaic route (multi-file upload)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))



# ---------------------------------------------------------------------------
# API route: POST /api/v1/scenes/{id}/mosaic (opt-in multi-file upload)
# ---------------------------------------------------------------------------


def _api_upload(client, buffer, name="part.tif"):
    import io

    buffer.seek(0)
    return client.post(
        "/api/v1/scenes", files={"file": (name, io.BytesIO(buffer.getvalue()), "image/tiff")}
    ).json()


def test_mosaic_route_merges_and_updates_scene(client, mock_inference):
    import io

    import rasterio
    from rasterio.transform import from_origin

    buffer = io.BytesIO()
    with rasterio.open(
        buffer, "w", driver="GTiff", height=64, width=64, count=3, dtype="uint8",
        crs="EPSG:32617", transform=from_origin(500000, 4000000, 0.5, 0.5),
    ) as dst:
        dst.write(np.zeros((3, 64, 64), dtype=np.uint8))
    scene = _api_upload(client, buffer, "west.tif")

    part_b = io.BytesIO()
    with rasterio.open(
        part_b, "w", driver="GTiff", height=64, width=64, count=3, dtype="uint8",
        crs="EPSG:32617", transform=from_origin(500032, 4000000, 0.5, 0.5),
    ) as dst:
        dst.write(np.zeros((3, 64, 64), dtype=np.uint8))

    part_a = io.BytesIO()
    with rasterio.open(
        part_a, "w", driver="GTiff", height=64, width=64, count=3, dtype="uint8",
        crs="EPSG:32617", transform=from_origin(500000, 4000000, 0.5, 0.5),
    ) as dst:
        dst.write(np.zeros((3, 64, 64), dtype=np.uint8))

    response = client.post(
        f"/api/v1/scenes/{scene['scene_id']}/mosaic",
        files=[
            ("files", ("west.tif", part_a, "image/tiff")),
            ("files", ("east.tif", part_b, "image/tiff")),
        ],
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["dimensions"]["width"] == 128
    assert body["dimensions"]["height"] == 64
    assert body["georeference"]["crs"] == "EPSG:32617"

    # the mosaic IS the scene input: processing runs on the merged raster
    # and the world scale reflects 128 px x 0.5 m/px = 64 m
    job = client.post(f"/api/v1/scenes/{scene['scene_id']}/process", json={}).json()
    status = client.get(f"/api/v1/jobs/{job['job_id']}").json()["status"]
    assert status == "completed"
    terrain = client.get(f"/api/v1/scenes/{scene['scene_id']}/terrain").json()
    assert terrain["world_width_m"] == pytest.approx(64.0)
    assert terrain["is_georeferenced_scale"] is True


def test_mosaic_route_rejects_processed_scene(client, processed_scene):
    import io

    scene_id = processed_scene["scene"]["scene_id"]
    response = client.post(
        f"/api/v1/scenes/{scene_id}/mosaic",
        files=[
            ("files", ("a.tif", io.BytesIO(b"x"), "image/tiff")),
            ("files", ("b.tif", io.BytesIO(b"y"), "image/tiff")),
        ],
    )
    assert response.status_code == 409


def test_mosaic_route_rejects_non_geotiff(client, uploaded_scene):
    import io

    from PIL import Image

    buffer = io.BytesIO()
    Image.new("RGB", (32, 32)).save(buffer, format="PNG")
    response = client.post(
        f"/api/v1/scenes/{uploaded_scene['scene_id']}/mosaic",
        files=[
            ("files", ("a.png", buffer, "image/png")),
            ("files", ("b.png", buffer, "image/png")),
        ],
    )
    assert response.status_code == 400
