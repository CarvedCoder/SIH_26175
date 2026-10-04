"""Geometry-aware 3D building reconstruction — end-to-end API tests.

Runs the REAL processing pipeline (mocked torch inference that writes
honest artifacts: dsm.npy + building_mask.npy + georeferenced dsm.tif)
and verifies:
    * the reconstruction stage produces buildings3d.json on disk;
    * the job result payload reports the buildings3d capability;
    * the API serves metadata + the reconstruction JSON;
    * heights in the served payload equal the DSM's building height;
    * a scene WITHOUT a building mask honestly reports available=False.
"""

from __future__ import annotations

import io

import numpy as np
import pytest
from PIL import Image
from rasterio.transform import from_origin

from backend.app.core import config as config_module

BUILDING = (40, 40, 140, 90)   # x0, y0, x1, y1 (pixel)
BUILDING_HEIGHT_M = 12.5


@pytest.fixture()
def building_scene(client, tmp_path, monkeypatch):
    """A processed 256x256 GeoTIFF scene carrying a raised rectangular
    building + its building mask (as the ONNX detector would write it)."""
    import backend.app.services.processing_service as ps
    from backend_tests.conftest import _tiny_checkpoint

    ckpt = _tiny_checkpoint(tmp_path)
    monkeypatch.setenv("DW_CKPT", str(ckpt))
    config_module.get_settings.cache_clear()

    def fake_run_inference(input_path, ckpt_path, *, out_dir, write_files=True, **kwargs):
        import rasterio

        input_path = __import__("pathlib").Path(input_path)
        out_dir = __import__("pathlib").Path(out_dir)
        with rasterio.open(input_path) as ds:
            height, width = ds.height, ds.width
            crs, transform = ds.crs, ds.transform

        dsm = np.zeros((height, width), dtype=np.float32)
        x0, y0, x1, y1 = BUILDING
        dsm[y0:y1, x0:x1] = BUILDING_HEIGHT_M
        np.save(out_dir / "dsm.npy", dsm)

        with rasterio.open(
            out_dir / "dsm.tif", "w", driver="GTiff", height=height,
            width=width, count=1, dtype="float32", crs=crs,
            transform=transform, compress="deflate",
        ) as dst:
            dst.write(dsm, 1)

        mask = np.zeros((height, width), dtype=np.uint8)
        mask[y0:y1, x0:x1] = 1
        np.save(out_dir / "building_mask.npy", mask)

        Image.fromarray(np.zeros((height, width), dtype=np.uint8), mode="L").save(
            out_dir / "dsm_preview.png", format="PNG"
        )

        return {
            "ok": True,
            "stem": input_path.stem,
            "grid": {"height": 1, "width": 1, "stride": 1, "data": [0.0]},
            "rgb_png": "data:image/png;base64,",
            "stats": {"n": 1, "min": 0.0, "mean": 0.0, "median": 0.0,
                      "max": 0.0, "neg": 0},
            "anchored": False,
            "georef": {"crs": "EPSG:32617", "transform": "KNOWN"},
            "meta": {"model_tag": "mock", "device": "cpu", "dn_source": "cache",
                     "mode": "auto", "source_shape": [height, width],
                     "pixel_size_m": 0.5, "elapsed_sec": 0.0},
            "outputs": {},
        }

    monkeypatch.setattr(ps, "run_inference", fake_run_inference)

    buffer = io.BytesIO()
    import rasterio

    with rasterio.open(
        buffer, "w", driver="GTiff", height=256, width=256, count=3,
        dtype="uint8", crs="EPSG:32617",
        transform=from_origin(500000, 4000000, 0.5, 0.5),
    ) as dst:
        dst.write(np.zeros((3, 256, 256), dtype=np.uint8))
    buffer.seek(0)

    response = client.post("/api/v1/scenes",
                           files={"file": ("block.tif", buffer, "image/tiff")})
    assert response.status_code == 200, response.text
    scene_id = response.json()["scene_id"]

    response = client.post(f"/api/v1/scenes/{scene_id}/process", json={})
    assert response.status_code == 200, response.text
    job_id = response.json()["job_id"]
    for _ in range(100):
        job = client.get(f"/api/v1/jobs/{job_id}").json()
        if job["status"] in {"completed", "failed", "cancelled"}:
            break
    assert job["status"] == "completed", job
    # rebuild the env-driven Settings cache like mock_inference does, so a
    # checkpoint path captured from THIS test's tmp_path never leaks
    config_module.get_settings.cache_clear()
    return {"scene_id": scene_id, "job": job}


class TestBuildings3DEndToEnd:
    def test_reconstruction_artifacts_are_produced(self, building_scene):
        from backend.app.services.result_service import result_service

        out = result_service.get_output_dir(building_scene["scene_id"])
        assert (out / "buildings3d.json").is_file()

    def test_job_payload_reports_buildings3d(self, building_scene):
        payload = building_scene["job"].get("result") or {}
        # the job record result is sanitized but the capability flag rides
        # on the completion payload — verify via the reconstruction meta
        cap = payload.get("buildings3d")
        assert cap is not None
        assert cap["available"] is True
        assert cap["count"] >= 1
        assert "dsm" in cap["height_source"].lower()

    def test_metadata_route_reports_availability(self, client, building_scene):
        response = client.get(
            f"/api/v1/scenes/{building_scene['scene_id']}/buildings3d")
        assert response.status_code == 200
        meta = response.json()
        assert meta["available"] is True
        assert meta["count"] >= 1
        assert meta["height_source"] == "dsm.npy (predicted metric height)"
        assert meta["georeferenced"] is True

    def test_reconstruction_json_serves_footprints_and_heights(self, client, building_scene):
        response = client.get(
            f"/api/v1/scenes/{building_scene['scene_id']}/results/buildings3d")
        assert response.status_code == 200
        data = response.json()
        assert data["available"] and data["count"] >= 1

        b = data["buildings"][0]
        # the footprint matches the injected building rectangle (100x50 px)
        xs = [p[0] for p in b["footprint_px"]]
        ys = [p[1] for p in b["footprint_px"]]
        assert (max(xs) - min(xs)) == pytest.approx(100, abs=2)
        assert (max(ys) - min(ys)) == pytest.approx(50, abs=2)
        # height EXACTLY from the DSM — never a fixed default
        assert b["height_m"] == pytest.approx(BUILDING_HEIGHT_M, abs=0.05)
        # georeferenced twin present with CRS coords inside the scene extent
        assert b["footprint_crs"] is not None
        crs_x = [p[0] for p in b["footprint_crs"]]
        assert 500000 + BUILDING[0] * 0.5 <= min(crs_x) <= max(crs_x) <= 500000 + BUILDING[2] * 0.5

    def test_scene_without_building_mask_reports_unavailable(self, client, processed_scene):
        response = client.get(
            f"/api/v1/scenes/{processed_scene['scene']['scene_id']}/buildings3d")
        assert response.status_code == 200
        assert response.json()["available"] is False

    def test_missing_reconstruction_json_is_404(self, client, processed_scene):
        response = client.get(
            f"/api/v1/scenes/{processed_scene['scene']['scene_id']}/results/buildings3d")
        assert response.status_code == 404
