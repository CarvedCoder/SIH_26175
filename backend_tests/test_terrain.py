"""Terrain, measurement, and refinement tests — including the GSD-honesty
refusal for slope on non-georeferenced scenes."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

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
    # 16-bit grayscale — 256-level 8-bit heightmaps terraced visibly once
    # vertical exaggeration was applied
    assert heightmap.mode == "I;16"
    assert heightmap.size[0] <= 1024 and heightmap.size[1] <= 1024


def test_heightmap_synthetic_surface_fidelity(client, tmp_path, monkeypatch):
    """End-to-end encoder check on a synthetic DSM (Gaussian hill + sharp
    building slabs + a NaN hole, 2048 px — forces LANCZOS decimation).

    Verifies the three artefacts the old encoder produced are gone:
      1. terracing: 16-bit levels reconstruct the smooth hill, not 256 steps
      2. aliasing spikes: decimation matches a block-mean low-pass reference
      3. NaN craters: holes are neighbour-filled, not mapped to the minimum
    and that building step edges survive the pipeline.
    """

    import numpy as np
    from PIL import Image

    from backend.app.services.terrain_service import terrain_service

    rng = np.random.default_rng(42)
    h, w = 1536, 2048
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)

    dsm = 60.0 + 40.0 * np.exp(-(((xx - w * 0.3) / 300.0) ** 2 +
                                ((yy - h * 0.6) / 260.0) ** 2))
    # Two sharp building slabs (worst case for decimation filters)
    dsm[200:420, 1300:1600] = 95.0
    dsm[900:1120, 400:640] = 88.0
    # Mild sensor noise
    dsm += rng.normal(0, 0.15, dsm.shape).astype(np.float32)
    # NaN hole (e.g. water / invalid depth patch)
    dsm[700:760, 1000:1100] = np.nan
    dsm = dsm.astype(np.float32)

    # Fake scene output dir so terrain_service AND result_service find our
    # dsm.npy (each module imported get_scene_output_dir into its namespace)
    scene_id = "scene_hmtest"
    out_dir = tmp_path / scene_id
    out_dir.mkdir(parents=True)
    np.save(out_dir / "dsm.npy", dsm)
    import backend.app.services.result_service as rs_mod
    import backend.app.services.terrain_service as ts_mod
    monkeypatch.setattr(ts_mod, "get_scene_output_dir", lambda _sid: out_dir)
    monkeypatch.setattr(rs_mod, "get_scene_output_dir", lambda _sid: out_dir)

    png_path = terrain_service.get_heightmap_path(scene_id)
    hm = np.asarray(Image.open(png_path), dtype=np.float64) / 65535.0

    # Expected reference: same normalisation + block-mean decimation. The
    # NaN hole is replaced with its surrounding ground level first so the
    # block mean stays finite (the encoder neighbour-fills it likewise).
    lo, hi = float(np.nanmin(dsm)), float(np.nanmax(dsm))
    ground_norm = (60.0 - lo) / (hi - lo)
    ref = np.where(np.isnan(dsm), ground_norm, dsm)
    ref = np.clip((ref - lo) / (hi - lo), 0.0, 1.0)
    sy, sx = h // hm.shape[0], w // hm.shape[1]
    ref_small = ref[: hm.shape[0] * sy, : hm.shape[1] * sx] \
        .reshape(hm.shape[0], sy, hm.shape[1], sx).mean(axis=(1, 3))

    # 1+2. Reconstruction matches the low-passed reference closely
    rms = float(np.sqrt(np.mean((hm - ref_small) ** 2)))
    assert rms < 0.03, f"heightmap deviates from low-passed DSM: rms={rms:.4f}"

    # Building edge stays a step: across the first slab's left wall the
    # heightmap must show the full height jump (±30% for decimation blur)
    row = 300 * hm.shape[0] // h
    edge_x = 1300 * hm.shape[1] // w
    before = float(np.mean(hm[row, edge_x - 6:edge_x - 2]))
    after = float(np.mean(hm[row, edge_x + 2:edge_x + 6]))
    jump = abs(after - before)
    expected = (95.0 - 60.0) / (hi - lo)
    assert jump > 0.7 * expected, (
        f"building edge blurred away: jump={jump:.3f} expected~{expected:.3f}"
    )

    # 3. NaN hole becomes a smooth patch of its SURROUNDING terrain (which
    # here is the Gaussian flank), not a 0-level crater
    hole = hm[730 * hm.shape[0] // h, 1050 * hm.shape[1] // w]
    border_mean = float(np.nanmean(dsm[698:702, 998:1102]))
    assert hole == pytest.approx((border_mean - lo) / (hi - lo), abs=0.05)


def test_fill_invalid_no_craters():
    """NaN holes must be neighbour-filled, not written to the range minimum —
    the old encoder mapped NaN → 0 which punched crater pits into the mesh."""
    import numpy as np

    from backend.app.services.terrain_service import TerrainService

    surface = np.full((16, 16), 50.0, dtype=np.float32)
    surface[4:8, 4:8] = np.nan
    surface[0, 0] = np.nan

    filled = TerrainService._fill_invalid(surface)

    assert np.isfinite(filled).all()
    # The hole sits in a flat 50 m region — filling must return ~50, not 0
    assert float(filled[4:8, 4:8].mean()) == pytest.approx(50.0, abs=0.01)
    # Original finite values untouched
    assert float(filled[12, 12]) == pytest.approx(50.0, abs=1e-6)


# ---------------------------------------------------------------------------
# Physical world footprint (real-world terrain scale for the 3D renderer)
# ---------------------------------------------------------------------------


def test_world_scale_georeferenced_scene(client, processed_scene):
    """GeoTIFF scene (256x256 @ 0.5 m/px, EPSG:32617): the renderer contract
    must carry the real metric footprint width × GSD = 128 m."""
    scene_id = processed_scene["scene"]["scene_id"]
    body = client.get(f"/api/v1/scenes/{scene_id}/terrain").json()

    assert body["is_georeferenced_scale"] is True
    assert body["world_width_m"] == pytest.approx(128.0)
    assert body["world_depth_m"] == pytest.approx(128.0)


def test_world_scale_non_georeferenced_fallback(client, mock_inference):
    """PNG scene (512x384, no CRS): no metric scale exists, so the documented
    1 m/pixel fallback is used and is_georeferenced_scale is False — the
    frontend can label it as an assumed scale instead of pretending."""
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

    body = client.get(f"/api/v1/scenes/{scene['scene_id']}/terrain").json()
    assert body["is_georeferenced_scale"] is False
    assert body["world_width_m"] == pytest.approx(512.0)
    assert body["world_depth_m"] == pytest.approx(384.0)


def test_slope_gsd_routes_through_pixel_size_metres(client, processed_scene):
    """Geographic-CRS honesty: _pixel_size must go through
    depthwizard.geo.pixel_size_metres, never raw transform degrees."""

    scene_id = processed_scene["scene"]["scene_id"]
    body = client.post(
        f"/api/v1/scenes/{scene_id}/measure/slope",
        json={"point_a": {"x": 0, "y": 0}, "point_b": {"x": 10, "y": 10}},
    ).json()
    # projected metric CRS: 10 px * 0.5 m/px on each axis
    assert body["gsd_available"] is True
    assert body["horizontal_distance_m"] == pytest.approx(
        float(np.hypot(10 * 0.5, 10 * 0.5))
    )
