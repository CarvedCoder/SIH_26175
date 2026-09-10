"""Terrain, measurement, and refinement tests — including the GSD-honesty
refusal for slope on non-georeferenced scenes."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def test_terrain_representation(client, processed_scene):
    scene_id = processed_scene["scene"]["scene_id"]
    body = client.get(f"/api/v1/scenes/{scene_id}/terrain").json()
    assert body["available"] is True
    terrain = body["terrain"]
    assert terrain["dimensions"] == {"width": 256, "height": 256}
    assert terrain["coordinate_system"]["crs"] == "EPSG:32617"
    assert terrain["elevation_mode"] == "absolute"
    layer_urls = [layer["url"] for layer in terrain["layers"]]
    for url in layer_urls:
        assert client.get(url).status_code == 200


def test_terrain_unprocessed_404(client, uploaded_scene):
    response = client.get(f"/api/v1/scenes/{uploaded_scene['scene_id']}/terrain")
    assert response.status_code == 404


def test_terrain_unknown_scene_404(client):
    assert client.get("/api/v1/scenes/scene_000000000000/terrain").status_code == 404


def test_terrain_tiles(client, processed_scene):
    scene_id = processed_scene["scene"]["scene_id"]
    body = client.get(f"/api/v1/scenes/{scene_id}/terrain/tiles?tile_size=128").json()
    assert body["grid_x"] == 2 and body["grid_y"] == 2
    assert len(body["tiles"]) == 4
    tile = body["tiles"][0]
    assert tile["width"] == 128 and tile["height"] == 128


def test_minimap_endpoint(client, processed_scene):
    scene_id = processed_scene["scene"]["scene_id"]
    body = client.get(f"/api/v1/scenes/{scene_id}/minimap").json()
    assert body["url"] == f"/api/v1/scenes/{scene_id}/results/minimap"
    image = client.get(body["url"])
    assert image.status_code == 200
    assert image.headers["content-type"] == "image/png"


def test_elevation_probe(client, processed_scene, mock_inference):
    """The mock writes a DSM of zeros — the probe must return exactly that."""
    scene_id = processed_scene["scene"]["scene_id"]
    body = client.get(f"/api/v1/scenes/{scene_id}/elevation?x=10&y=20").json()
    assert body["elevation"] == 0.0
    assert body["units"] == "meters"


def test_elevation_out_of_bounds(client, processed_scene):
    scene_id = processed_scene["scene"]["scene_id"]
    response = client.get(f"/api/v1/scenes/{scene_id}/elevation?x=99999&y=0")
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "POINT_OUT_OF_BOUNDS"


def test_measure_height(client, processed_scene, mock_inference):
    scene_id = processed_scene["scene"]["scene_id"]
    response = client.post(
        f"/api/v1/scenes/{scene_id}/measure/height",
        json={"ground": {"x": 5, "y": 5}, "top": {"x": 10, "y": 10}},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["height"] == 0.0  # zeros DSM
    assert body["units"] == "meters"


def test_measure_height_out_of_bounds(client, processed_scene):
    scene_id = processed_scene["scene"]["scene_id"]
    response = client.post(
        f"/api/v1/scenes/{scene_id}/measure/height",
        json={"ground": {"x": 5, "y": 5}, "top": {"x": 9999, "y": 10}},
    )
    assert response.status_code == 400


def test_measure_slope_georeferenced(client, processed_scene):
    """Georeferenced scene -> real metric slope from the GSD."""
    import math

    scene_id = processed_scene["scene"]["scene_id"]
    response = client.post(
        f"/api/v1/scenes/{scene_id}/measure/slope",
        json={"point_a": {"x": 0, "y": 0}, "point_b": {"x": 100, "y": 0}},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["gsd_available"] is True
    assert math.isclose(body["horizontal_distance_m"], 50.0)  # 100 px * 0.5 m
    assert body["slope_degrees"] == 0.0  # zeros DSM -> flat


def test_measure_slope_non_georeferenced_refusal(client, uploaded_scene, mock_inference):
    """HONESTY CONTRACT: a scene without CRS has no metric scale — slope
    must be refused (null degrees), never guessed from pixels."""
    import io

    import rasterio

    # upload a NON-georeferenced scene and process it
    buffer = io.BytesIO()
    with rasterio.open(
        buffer,
        "w",
        driver="GTiff",
        height=256,
        width=256,
        count=3,
        dtype="uint8",
    ) as dst:
        dst.write(np.zeros((3, 256, 256), dtype=np.uint8))
    buffer.seek(0)

    scene = client.post(
        "/api/v1/scenes", files={"file": ("plain.tif", buffer, "image/tiff")}
    ).json()
    job = client.post(f"/api/v1/scenes/{scene['scene_id']}/process", json={}).json()
    status = client.get(f"/api/v1/jobs/{job['job_id']}").json()["status"]
    assert status == "completed"

    response = client.post(
        f"/api/v1/scenes/{scene['scene_id']}/measure/slope",
        json={"point_a": {"x": 0, "y": 0}, "point_b": {"x": 100, "y": 0}},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["gsd_available"] is False
    assert body["slope_degrees"] is None
    assert body["slope_percent"] is None
    assert body["elevation_change_m"] == 0.0


def test_refine_region(client, uploaded_scene, mock_inference):
    scene_id = uploaded_scene["scene_id"]
    response = client.post(
        f"/api/v1/scenes/{scene_id}/refine",
        json={
            "bbox": {"x_min": 0, "y_min": 0, "x_max": 128, "y_max": 128},
            "resolution": "high",
        },
    )
    assert response.status_code == 200
    job_id = response.json()["job_id"]

    job = client.get(f"/api/v1/jobs/{job_id}").json()
    assert job["status"] == "completed"

    # the refine crop really came from the scene input at the bbox window
    from backend.app.core.paths import get_scene_process_dir

    crop = get_scene_process_dir(scene_id) / "refine_crop.tif"
    assert crop.is_file()

    import rasterio

    with rasterio.open(crop) as ds:
        assert (ds.width, ds.height) == (128, 128)

    from backend.app.core.paths import get_scene_output_dir

    assert (get_scene_output_dir(scene_id) / "refined_dsm.npy").is_file()


def test_refine_unknown_scene_404(client):
    response = client.post(
        "/api/v1/scenes/scene_000000000000/refine",
        json={"bbox": {"x_min": 0, "y_min": 0, "x_max": 10, "y_max": 10}},
    )
    assert response.status_code == 404


def test_refine_bbox_out_of_bounds(client, uploaded_scene, mock_inference):
    scene_id = uploaded_scene["scene_id"]
    response = client.post(
        f"/api/v1/scenes/{scene_id}/refine",
        json={"bbox": {"x_min": 0, "y_min": 0, "x_max": 99999, "y_max": 10}},
    )
    assert response.status_code == 200
    job = client.get(f"/api/v1/jobs/{response.json()['job_id']}").json()
    assert job["status"] == "failed"
    assert job["error"]["code"] == "INVALID_INPUT"


# ---------------------------------------------------------------------------
# Terrain for non-georeferenced scenes + the 3D renderer contract
# ---------------------------------------------------------------------------

def test_terrain_non_georeferenced_scene(client, mock_inference):
    """JPG/PNG scenes have no CRS: terrain MUST still be served (relative
    mode, pixel-space bounds) — this was the '3D terrain shows nothing'
    bug."""
    import io

    from PIL import Image

    buffer = io.BytesIO()
    Image.new("RGB", (512, 384), color=(90, 120, 60)).save(buffer, format="PNG")
    buffer.seek(0)

    scene = client.post(
        "/api/v1/scenes", files={"file": ("village.png", buffer, "image/png")}
    ).json()
    job = client.post(f"/api/v1/scenes/{scene['scene_id']}/process", json={}).json()
    status = client.get(f"/api/v1/jobs/{job['job_id']}").json()["status"]
    assert status == "completed"

    response = client.get(f"/api/v1/scenes/{scene['scene_id']}/terrain")
    assert response.status_code == 200, response.text
    body = response.json()

    assert body["available"] is True
    assert body["terrain"]["elevation_mode"] == "relative"
    assert body["terrain"]["coordinate_system"]["crs"] is None
    # pixel-space bounds, honestly
    assert body["terrain"]["bounds"] == {
        "min_x": 0.0, "min_y": 0.0, "max_x": 512.0, "max_y": 384.0,
    }


def test_terrain_renderer_contract_fields(client, processed_scene):
    """TerrainCanvas destructures these top-level fields — they must exist
    and every URL must serve a real, decodable image."""
    import io

    from PIL import Image

    scene_id = processed_scene["scene"]["scene_id"]
    body = client.get(f"/api/v1/scenes/{scene_id}/terrain").json()

    for field in (
        "heightmap_url",
        "texture_url",
        "height_scale",
        "min_elevation",
        "max_elevation",
    ):
        assert field in body, f"renderer contract field missing: {field}"
    assert body["heightmap_url"] and body["texture_url"]
    assert body["height_scale"] == body["max_elevation"]

    for url in (body["heightmap_url"], body["texture_url"]):
        image = client.get(url)
        assert image.status_code == 200, url
        assert image.headers["content-type"] == "image/png"
        img = Image.open(io.BytesIO(image.content))
        img.verify()

    heightmap = Image.open(io.BytesIO(client.get(body["heightmap_url"]).content))
    assert heightmap.mode in ("L", "RGB")  # R channel carries elevation
