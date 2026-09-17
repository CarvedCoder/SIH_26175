"""Shared fixtures for the backend API test suite.

The model is MOCKED for API tests (no HuggingFace download, no real torch
inference): run_inference is monkeypatched to write honest, shape-correct
result artifacts into the scene output directory, exactly like the real
pipeline would.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fastapi.testclient import TestClient

import backend.app.core.config as config_module
import backend.app.jobs.manager as manager_module


@pytest.fixture(autouse=True)
def isolated_storage(tmp_path, monkeypatch):
    """Redirect all scene storage into a per-test temp directory."""
    import backend.app.core.paths as paths

    monkeypatch.setattr(paths, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(paths, "DATA_DIR", tmp_path / "data")
    monkeypatch.setattr(paths, "RAW_DIR", tmp_path / "data" / "raw")
    monkeypatch.setattr(paths, "PROCESS_DIR", tmp_path / "data" / "process")
    monkeypatch.setattr(paths, "OUTPUT_DIR", tmp_path / "data" / "output")
    monkeypatch.setattr(
        paths, "UPLOAD_STAGING_DIR", tmp_path / "data" / "staging"
    )
    monkeypatch.setattr(
        paths, "SCENES_RAW_DIR", tmp_path / "data" / "raw" / "scenes"
    )
    monkeypatch.setattr(
        paths, "SCENES_PROCESS_DIR", tmp_path / "data" / "process" / "scenes"
    )
    monkeypatch.setattr(
        paths, "SCENES_OUTPUT_DIR", tmp_path / "data" / "output" / "scenes"
    )
    paths.ensure_directories()
    yield tmp_path


@pytest.fixture(autouse=True)
def fresh_settings(monkeypatch):
    """Per-test settings: no API key, small upload cap for limit tests."""
    monkeypatch.delenv("DW_API_KEY", raising=False)
    monkeypatch.delenv("DW_CKPT", raising=False)
    monkeypatch.setenv("DW_MAX_UPLOAD_BYTES", str(2 * 1024 * 1024))
    config_module.get_settings.cache_clear()
    yield
    config_module.get_settings.cache_clear()


@pytest.fixture()
def client():
    from backend.app.main import app

    with TestClient(app, raise_server_exceptions=False) as c:
        yield c


@pytest.fixture()
def fresh_job_manager(monkeypatch):
    """A clean JobManager (bounded defaults) injected into the module."""
    mgr = manager_module.JobManager(retention_limit=8, ttl_seconds=3600)
    monkeypatch.setattr(manager_module, "job_manager", mgr)
    # processing_service holds the job store as an INJECTED attribute —
    # patch the instance (tranche-3 DI), not the module global.
    import backend.app.services.processing_service as ps

    monkeypatch.setattr(ps.processing_service, "_job_service", mgr)
    yield mgr


def _tiny_checkpoint(tmp_path: Path) -> Path:
    """A real (untrained, tiny) CalibrationNet checkpoint — exercises the
    secure loader without any network access."""
    import torch

    from depthwizard.calibration_net import CalibrationNet

    net = CalibrationNet(in_ch=4, widths=(4, 8, 16), a0=1.0, b0=0.5)
    ckpt = {
        "model_state": net.state_dict(),
        "use_rgb": True,
        "use_dem": False,
        "use_sem": False,
        "sem_classes": 0,
        "sem_aux_head": False,
        "in_ch": 4,
        "widths": [4, 8, 16],
        "clamp_min": 0.0,
        "affine_init": {"a": 1.0, "b": 0.5},
        "loss": "l1",
        "epoch": 0,
        "val_subset_mae": 0.0,
        "splits_json": "n/a",
        "created": "2026-09-09T00:00:00+00:00",
    }
    p = tmp_path / "best.pt"
    torch.save(ckpt, p)
    return p


@pytest.fixture()
def mock_inference(tmp_path, monkeypatch):
    """Mock the inference path: verify the checkpoint with the REAL secure
    loader, then write honest result artifacts instead of running torch.

    Yields a recorder list of (input_path, kwargs) calls.
    """
    import rasterio

    import backend.app.services.processing_service as ps

    ckpt = _tiny_checkpoint(tmp_path)
    monkeypatch.setenv("DW_CKPT", str(ckpt))
    # Settings are env-driven and cached (tranche-2 centralization): a test
    # that changes the environment must rebuild the cached Settings.
    config_module.get_settings.cache_clear()

    calls: list[tuple[Path, dict]] = []

    def fake_run_inference(input_path, ckpt_path, *, out_dir, write_files=True, **kwargs):
        input_path = Path(input_path)
        out_dir = Path(out_dir)
        calls.append((input_path, kwargs))

        with rasterio.open(input_path) as ds:
            height, width = ds.height, ds.width
            crs = ds.crs
            transform = ds.transform

        dsm = np.zeros((height, width), dtype=np.float32)
        np.save(out_dir / "dsm.npy", dsm)

        # mirror the real pipeline: dsm.tif ONLY when georeferenced
        if crs is not None:
            with rasterio.open(
                out_dir / "dsm.tif",
                "w",
                driver="GTiff",
                height=height,
                width=width,
                count=1,
                dtype="float32",
                crs=crs,
                transform=transform,
                compress="deflate",
            ) as dst:
                dst.write(dsm, 1)

        from PIL import Image

        Image.fromarray(np.zeros((height, width), dtype=np.uint8), mode="L").save(
            out_dir / "dsm_preview.png", format="PNG"
        )

        return {
            "ok": True,
            "stem": input_path.stem,
            "grid": {"height": 1, "width": 1, "stride": 1, "data": [0.0]},
            "rgb_png": "data:image/png;base64,",
            "stats": {"n": 1, "min": 0.0, "mean": 0.0, "median": 0.0, "max": 0.0, "neg": 0},
            "anchored": False,
            "georef": {"crs": "UNKNOWN", "transform": "UNKNOWN"},
            "meta": {"model_tag": "mock", "device": "cpu", "dn_source": "cache",
                     "mode": "auto", "source_shape": [height, width],
                     "pixel_size_m": None, "elapsed_sec": 0.0},
            "outputs": {},
        }

    monkeypatch.setattr(ps, "run_inference", fake_run_inference)
    yield calls
    config_module.get_settings.cache_clear()


@pytest.fixture()
def uploaded_scene(client, tmp_path):
    """Create a real scene via the API: a tiny valid GeoTIFF upload."""
    import io

    import rasterio
    from rasterio.transform import from_origin

    buffer = io.BytesIO()
    with rasterio.open(
        buffer,
        "w",
        driver="GTiff",
        height=256,
        width=256,
        count=3,
        dtype="uint8",
        crs="EPSG:32617",
        transform=from_origin(500000, 4000000, 0.5, 0.5),
    ) as dst:
        dst.write(np.zeros((3, 256, 256), dtype=np.uint8))
    buffer.seek(0)

    response = client.post(
        "/api/v1/scenes",
        files={"file": ("ortho.tif", buffer, "image/tiff")},
    )
    assert response.status_code == 200, response.text
    return response.json()


@pytest.fixture()
def processed_scene(client, uploaded_scene, mock_inference):
    """A scene that has a completed processing job behind it."""
    scene_id = uploaded_scene["scene_id"]
    response = client.post(f"/api/v1/scenes/{scene_id}/process", json={})
    assert response.status_code == 200, response.text
    job_id = response.json()["job_id"]

    for _ in range(100):
        job = client.get(f"/api/v1/jobs/{job_id}").json()
        if job["status"] in {"completed", "failed", "cancelled"}:
            break
    assert job["status"] == "completed", job
    return {"scene": uploaded_scene, "job_id": job_id}
