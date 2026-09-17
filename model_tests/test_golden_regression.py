"""GOLDEN REGRESSION — the numerical pipeline's behavior contract
(refactor brief §52/53).

Pins: same inputs -> same outputs, within float tolerance, for the full
certified path EXCLUDING the Depth-Anything backbone (whose output is
frozen by the HF weights and covered by its own deterministic forward):

    fixed GeoTIFF + fixed RAW Dn .npy + fixed tiny checkpoint
        -> run_inference(cpu, live_backbone=False, dn_path=..., mode="auto")
        -> calibrated AGL / DSM / payload stats

The golden files under model_tests/fixtures/golden/ were generated from
the pre-refactor pipeline (v2_stack, 2026-09). Tranche 3 (inference.py
decomposition) may NOT change these outputs — regenerate only with an
explicit, documented numerical-intent change.

Regenerate:  pytest model_tests/test_golden_regression.py --regen-golden
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest

pytest_plugins = ()

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

GOLDEN_DIR = Path(__file__).resolve().parent / "fixtures" / "golden"
H, W = 512, 512  # single-tile path (<= TILE=1024 -> "crop" mode)


def _fixture_paths(tmp_path: Path) -> dict:
    """Deterministic synthetic scene: a seeded RGB GeoTIFF + seeded RAW Dn
    with realistic relative-depth structure (smooth blobs + edges)."""
    import rasterio

    rgb_path = tmp_path / "input.tif"
    dn_path = tmp_path / "raw_dn.npy"

    rg = np.random.default_rng(42)
    yy, xx = np.mgrid[0:H, 0:W]
    rgb = np.zeros((H, W, 3), dtype=np.uint8)
    rgb[..., 0] = (np.sin(xx / 37.0) * 60 + 140).astype(np.uint8)
    rgb[..., 1] = (np.sin(yy / 53.0) * 60 + 150).astype(np.uint8)
    rgb[..., 2] = ((xx + yy) % 97 * 2 + 40).astype(np.uint8)
    rgb[100:200, 300:420] = (rg.integers(60, 200, (100, 120, 3))).astype(np.uint8)

    raw_dn = (
        2.0
        + np.sin(xx / 91.0) * 0.6
        + np.cos(yy / 77.0) * 0.5
        + (yy > 350) * 0.8  # a step edge the calibration must respect
    ).astype(np.float32)
    raw_dn[100:200, 300:420] += 1.4  # a "building" blob

    profile = {
        "driver": "GTiff", "width": W, "height": H, "count": 3,
        "dtype": "uint8",
    }
    with rasterio.open(rgb_path, "w", **profile) as dst:
        dst.write(rgb.transpose(2, 0, 1))
    np.save(dn_path, raw_dn)
    return {"rgb": rgb_path, "dn": dn_path}


def _tiny_checkpoint(tmp_path: Path) -> Path:
    """Deterministic untrained CalibrationNet checkpoint (same recipe as
    backend_tests conftest) — exercises the REAL checkpoint loader and
    the REAL calibration forward on the fixture."""
    import torch

    from depthwizard.calibration_net import CalibrationNet

    net = CalibrationNet(in_ch=4, widths=(4, 8, 16), a0=2.076463, b0=4.110271)
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
        "affine_init": {"a": 2.076463, "b": 4.110271},
        "loss": "l1",
        "epoch": 0,
        "val_subset_mae": 0.0,
        "splits_json": "n/a",
        "created": "2026-09-17T00:00:00+00:00",
    }
    p = tmp_path / "golden_best.pt"
    torch.save(ckpt, p)
    return p


def _run_pipeline(tmp_path: Path) -> dict:
    from depthwizard.inference import run_inference

    fx = _fixture_paths(tmp_path)
    ckpt = _tiny_checkpoint(tmp_path)
    out_dir = tmp_path / "out"
    payload = run_inference(
        input_path=fx["rgb"],
        ckpt_path=ckpt,
        out_dir=out_dir,
        device="cpu",
        mode="auto",
        dn_path=fx["dn"],
        cache_dir=None,
        live_backbone=False,
        anchor_dem=None,
        ground_elev=None,
        write_files=True,
    )
    agl = np.load(out_dir / "dsm.npy")
    return {"payload": payload, "agl": agl}


def _golden_signature(result: dict) -> dict:
    agl = result["agl"]
    stats = result["payload"]["stats"]
    return {
        "agl_sha": hashlib_sha(agl),
        "agl_shape": list(agl.shape),
        "stats": {k: round(float(v), 6) for k, v in stats.items()
                  if isinstance(v, (int, float))},
        "anchored": result["payload"]["anchored"],
        "dn_source": result["payload"]["meta"]["dn_source"],
        "mode": result["payload"]["meta"]["mode"],
    }


def hashlib_sha(arr: np.ndarray) -> str:
    import hashlib

    return hashlib.sha256(np.ascontiguousarray(arr, dtype=np.float32)).hexdigest()


def _load_golden() -> dict | None:
    path = GOLDEN_DIR / "golden_signature.json"
    if not path.exists():
        return None
    return json.loads(path.read_text())


def _store_golden(sig: dict) -> None:
    GOLDEN_DIR.mkdir(parents=True, exist_ok=True)
    (GOLDEN_DIR / "golden_signature.json").write_text(
        json.dumps(sig, indent=2, sort_keys=True)
    )


@pytest.fixture(scope="module")
def pipeline_result(tmp_path_factory):
    return _run_pipeline(tmp_path_factory.mktemp("golden"))


def test_golden_regression(pipeline_result):
    """Same inputs must produce the same outputs (the refactor gate)."""
    sig = _golden_signature(pipeline_result)
    golden = _load_golden()
    if os.environ.get("REGEN_GOLDEN") == "1" or golden is None:
        _store_golden(sig)
        golden = sig
    assert sig["agl_shape"] == golden["agl_shape"]
    # full array identity is the contract; allow the explicit hash mismatch
    # message to carry the diff for debugging
    assert sig["agl_sha"] == golden["agl_sha"], (
        "calibrated AGL changed vs the golden fixture — this is a NUMERICAL "
        "behavior change. If intended, regenerate with REGEN_GOLDEN=1 and "
        "document why."
    )
    for key, value in golden["stats"].items():
        got = sig["stats"][key]
        assert abs(got - value) <= max(1e-6, abs(value) * 1e-6), (
            f"stat {key} changed: {got} != {value}"
        )
    assert sig["anchored"] == golden["anchored"]
    assert sig["mode"] == golden["mode"]


def test_golden_agl_is_valid_agl(pipeline_result):
    """AGL is clamped >= 0 and finite (the clean_agl contract)."""
    agl = pipeline_result["agl"]
    assert np.isfinite(agl).all()
    assert float(agl.min()) >= 0.0
