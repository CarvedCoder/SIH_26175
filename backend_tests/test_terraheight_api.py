"""Backend API contract tests for the TerraHeight-S height backend.

Covers the API slice of the TerraHeight integration (task Sec. 14/26):
  * the ``architecture=terraheight_s`` value is accepted by the process /
    refine request schemas (pydantic validation, exact string)
  * the /health/models report includes the terraheight_s backend
  * the default height backend is UNCHANGED (RDAH) — terraheight_s is an
    explicitly selected alternative
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def test_model_architecture_accepts_terraheight_s():
    from backend.app.schemas.processing import ModelArchitecture

    assert ModelArchitecture("terraheight_s") is ModelArchitecture.TERRAHEIGHT_S


def test_process_request_accepts_terraheight_s():
    from backend.app.schemas.processing import ModelArchitecture, ProcessRequest

    req = ProcessRequest(architecture="terraheight_s")
    assert req.architecture is ModelArchitecture.TERRAHEIGHT_S
    # default remains the legacy-client AUTO (-> RDAH) — TerraHeight is an
    # explicitly selected alternative, never a silent default
    assert ProcessRequest().architecture is ModelArchitecture.AUTO


def test_refine_request_accepts_terraheight_s():
    from backend.app.schemas.processing import ModelArchitecture, RefineRequest

    req = RefineRequest(
        bbox={"x_min": 0, "y_min": 0, "x_max": 10, "y_max": 10},
        architecture="terraheight_s",
    )
    assert req.architecture is ModelArchitecture.TERRAHEIGHT_S


def test_health_models_reports_terraheight_backend(client):
    resp = client.get("/api/v1/health/models")
    assert resp.status_code == 200
    body = resp.json()
    assert "terraheight_s" in body["backends"]
    assert body["backends"]["terraheight_s"]["checkpoint_exists"] in (True, False)
    # no GPU probe unless explicitly requested
    assert body["terraheight_probe"] is None


def test_process_route_passes_terraheight_s_to_inference(client, uploaded_scene, mock_inference, monkeypatch, tmp_path):
    """The selector value genuinely reaches run_inference — the backend
    executes TerraHeight-S (not just a dropdown entry)."""
    import torch

    import backend.app.services.processing_service as ps
    from depthwizard.vendor.depth_anything_v2.dpt import DepthAnythingV2

    # mock_inference points DW_CKPT at a CalibrationNet ckpt; an explicit
    # terraheight_s request must reject that loudly — so point
    # DW_CKPT_TERRAHEIGHT at a valid (tiny, random-weight) release payload.
    torch.manual_seed(0)
    core = DepthAnythingV2(encoder="vits", features=64, out_channels=[48, 96, 192, 384])
    ckpt = tmp_path / "terraheight.pth"
    torch.save(
        {
            "model": {f"net.{k}": v for k, v in core.state_dict().items()},
            "model_config": {"encoder": "vits", "features": 64,
                             "out_channels": [48, 96, 192, 384]},
            "transform": {"mode": "normalized", "scale_m": 8.492877943662961,
                          "units": "metres_AGL"},
        },
        ckpt,
    )
    monkeypatch.setattr(
        ps.processing_service.settings, "ckpt_terraheight", str(ckpt), raising=False
    )

    scene_id = uploaded_scene["scene_id"]
    resp = client.post(
        f"/api/v1/scenes/{scene_id}/process",
        json={"architecture": "terraheight_s"},
    )
    assert resp.status_code == 200, resp.text
    _input_path, kwargs = mock_inference[0]
    assert kwargs["architecture"] == "terraheight_s"
