"""Live validation tests — a scene processed with a ground-truth reference
raster must produce populated, non-null validation metrics, with the
reference reprojected onto the prediction's exact grid (partial coverage
refused)."""

from __future__ import annotations

import io
import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def _write_reference_tif(path: Path, transform, crs, values) -> None:
    import rasterio

    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=values.shape[0],
        width=values.shape[1],
        count=1,
        dtype="float32",
        crs=crs,
        transform=transform,
    ) as dst:
        dst.write(values.astype(np.float32), 1)


def _process(client, scene_id: str) -> None:
    response = client.post(
        f"/api/v1/scenes/{scene_id}/process", json={"eager_analysis": True}
    )
    assert response.status_code == 200, response.text
    job_id = response.json()["job_id"]
    for _ in range(100):
        job = client.get(f"/api/v1/jobs/{job_id}").json()
        if job["status"] in {"completed", "failed", "cancelled"}:
            break
    assert job["status"] == "completed", job


def _output_dir(scene_id: str) -> Path:
    from backend.app.infrastructure.storage.scene_artifacts import (
        scene_artifact_store,
        scene_output_dir_key,
    )

    return scene_artifact_store().path_for(scene_output_dir_key(scene_id))


def test_validation_populated_with_same_grid_reference(
    client, uploaded_scene, mock_inference
):
    """A reference on the exact prediction grid -> reference.npy +
    validation.json + error_map.png written at processing time; the API
    reports non-null metrics."""
    from rasterio.transform import from_origin

    scene_id = uploaded_scene["scene_id"]
    from backend.app.core.paths import get_scene_raw_dir

    values = np.linspace(0.0, 5.0, 256 * 256, dtype=np.float32).reshape(
        256, 256
    )
    _write_reference_tif(
        get_scene_raw_dir(scene_id) / "reference.tif",
        from_origin(500000, 4000000, 0.5, 0.5),
        "EPSG:32617",
        values,
    )

    _process(client, scene_id)

    out = _output_dir(scene_id)
    assert (out / "reference.npy").is_file()
    assert (out / "validation.json").is_file()
    assert (out / "error_map.png").is_file()

    body = client.get(f"/api/v1/scenes/{scene_id}/validation").json()
    assert body["available"] is True
    metrics = body["metrics"]
    # mock inference writes an all-zero DSM -> error == -reference
    assert metrics["sample_count"] == 256 * 256
    assert metrics["rmse"] == pytest.approx(float(np.sqrt((values**2).mean())), rel=1e-4)
    assert metrics["mae"] == pytest.approx(float(np.abs(values).mean()), rel=1e-4)

    meta = json.loads((out / "validation.json").read_text())
    assert meta["source_reference"] == "reference.tif"
    assert meta["reprojected_to_prediction_grid"] is True
    assert meta["reference_crs"] == "EPSG:32617"

    error_map = client.get(f"/api/v1/scenes/{scene_id}/validation/error-map")
    assert error_map.status_code == 200
    image_url = error_map.json()["url"]
    assert client.get(image_url).status_code == 200


def test_validation_reprojects_onto_prediction_grid(
    client, uploaded_scene, mock_inference
):
    """A reference on a DIFFERENT grid (coarser, offset origin) that fully
    covers the prediction footprint is bilinear-reprojected before the
    comparison — metrics stay populated."""
    from rasterio.transform import from_origin

    scene_id = uploaded_scene["scene_id"]
    from backend.app.core.paths import get_scene_raw_dir

    # Prediction grid: x [500000, 500128], y [3999872, 4000000] (0.5 m px).
    # This reference is 1.0 m px with a shifted origin but still covers it.
    values = np.full((256, 256), 10.0, dtype=np.float32)
    _write_reference_tif(
        get_scene_raw_dir(scene_id) / "ref_dem.tif",
        from_origin(499999, 4000001, 1.0, 1.0),
        "EPSG:32617",
        values,
    )

    _process(client, scene_id)

    body = client.get(f"/api/v1/scenes/{scene_id}/validation").json()
    assert body["available"] is True
    metrics = body["metrics"]
    assert metrics["sample_count"] == 256 * 256
    assert metrics["rmse"] == pytest.approx(10.0, rel=1e-3)
    assert metrics["mae"] == pytest.approx(10.0, rel=1e-3)


def test_validation_refuses_partial_coverage(
    client, uploaded_scene, mock_inference
):
    """A reference covering only PART of the prediction footprint must not
    produce metrics — no artifacts are written and the API reports that no
    validation exists (never a fabricated partial comparison)."""
    from rasterio.transform import from_origin

    scene_id = uploaded_scene["scene_id"]
    from backend.app.core.paths import get_scene_raw_dir

    # Only covers the RIGHT half of the prediction grid.
    values = np.full((256, 128), 3.0, dtype=np.float32)
    _write_reference_tif(
        get_scene_raw_dir(scene_id) / "reference.tif",
        from_origin(500064, 4000000, 0.5, 0.5),
        "EPSG:32617",
        values,
    )

    _process(client, scene_id)

    out = _output_dir(scene_id)
    assert not (out / "reference.npy").exists()
    assert not (out / "validation.json").exists()

    body = client.get(f"/api/v1/scenes/{scene_id}/validation").json()
    assert body["available"] is False
    assert body["metrics"]["rmse"] is None


def test_validation_absent_without_reference(client, processed_scene):
    """No reference raster -> validation honestly unavailable."""
    scene_id = processed_scene["scene"]["scene_id"]
    body = client.get(f"/api/v1/scenes/{scene_id}/validation").json()
    assert body["available"] is False
    assert body["reference"]["available"] is False
    assert body["metrics"]["rmse"] is None
