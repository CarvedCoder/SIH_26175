"""Tests for depthwizard.inference — the shared CLI/service code path.

Covers the torch-FREE surface (geometry helpers, stats, payload, outputs)
plus, when torch is available, the full DepthWizardPredictor round trip on a
synthetic checkpoint. This is where webapp/CLI drift would be caught.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from depthwizard.inference import (build_scene_payload, center_crop_to_tile,
                                   compute_stats, downsample_grid,
                                   downsample_stride, resize_to_tile,
                                   tile_bounds)


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def test_resize_to_tile_2d_and_3d():
    g = np.random.default_rng(0).random((64, 32)).astype(np.float32)
    out = resize_to_tile(g, 128)
    assert out.shape == (128, 128) and out.dtype == np.float32
    rgb = np.zeros((10, 12, 3), dtype=np.uint8)
    out3 = resize_to_tile(rgb, 64)
    assert out3.shape == (64, 64, 3)


def test_resize_identity_when_already_tile_sized():
    g = np.ones((64, 64), dtype=np.float32)
    assert resize_to_tile(g, 64) is g


def test_center_crop_exact_and_pad():
    g = np.arange(64 * 64, dtype=np.float32).reshape(64, 64)
    c = center_crop_to_tile(g, 32)
    assert c.shape == (32, 32)
    # center of the arange grid: rows 16..48
    assert c[0, 0] == 16 * 64 + 16
    small = np.ones((20, 30), dtype=np.float32)
    padded = center_crop_to_tile(small, 32)
    assert padded.shape == (32, 32)      # reflect-padded up to tile size


def test_tile_bounds_padded_to_multiples():
    ny, nx, hp, wp = tile_bounds(1024, 1024)
    assert (ny, nx, hp, wp) == (1, 1, 1024, 1024)
    ny, nx, hp, wp = tile_bounds(1500, 700)
    assert (ny, nx) == (2, 1) and (hp, wp) == (2048, 1024)


def test_downsample_stride_and_grid():
    assert downsample_stride(1024, 1024, 512) == 2
    assert downsample_stride(500, 300, 512) == 1
    g = np.arange(64 * 64, dtype=np.float32).reshape(64, 64)
    d = downsample_grid(g, 4)
    assert d.shape == (16, 16)
    assert d[1, 1] == g[4, 4]


# ---------------------------------------------------------------------------
# Stats + payload contract (the backend<->frontend interface)
# ---------------------------------------------------------------------------

def test_compute_stats_basic():
    dsm = np.array([[0.0, 1.0], [2.0, 10.0]], dtype=np.float32)
    s = compute_stats(dsm)
    assert s["n"] == 4
    assert s["min"] == 0.0 and s["max"] == 10.0
    assert s["mean"] == pytest.approx(3.25)
    assert s["median"] == pytest.approx(1.5)
    assert s["neg"] == 0


def test_build_scene_payload_contract():
    """The webapp depends on these exact keys — breaking this test means the
    frontend contract broke."""
    rng = np.random.default_rng(1)
    dsm = rng.random((600, 800)).astype(np.float32) * 20
    rgb = rng.integers(0, 255, (600, 800, 3), dtype=np.uint8)
    payload = build_scene_payload(
        dsm, rgb, stem="X_001_001", mode="tiles", dn_source="live",
        model_tag="calib_rgb_ep24", device="cpu",
        profile={"_crs_obj": None, "_transform_obj": None},
        anchored=None, outputs={}, elapsed_sec=1.23)

    assert payload["ok"] is True
    assert payload["stem"] == "X_001_001"
    g = payload["grid"]
    assert g["height"] == 300 and g["width"] == 400      # 600x800 -> stride 2
    assert len(g["data"]) == g["height"] * g["width"]
    assert g["data"][0] == pytest.approx(dsm[0, 0], abs=1e-3)
    assert payload["rgb_png"].startswith("data:image/png;base64,")
    assert payload["anchored"] is False
    assert payload["georef"]["crs"] == "UNKNOWN"
    assert payload["georef"]["transform"] == "UNKNOWN"
    assert payload["meta"]["dn_source"] == "live"
    assert payload["meta"]["source_shape"] == [600, 800]
    # GSD honesty contract (worklog Section 4): pixel_size_m MUST be present
    # in the payload meta and MUST be null when the input carried no CRS.
    # See tests/test_payload_meta.py for the full CRS-derived-value suite.
    assert "pixel_size_m" in payload["meta"]
    assert payload["meta"]["pixel_size_m"] is None


def test_build_scene_payload_anchored_label():
    from depthwizard.anchoring import anchor_with_constant
    dsm = np.ones((4, 4), dtype=np.float32) * 5.0
    rgb = np.zeros((4, 4, 3), dtype=np.uint8)
    anchored = anchor_with_constant(dsm, 100.0)
    payload = build_scene_payload(
        dsm, rgb, stem="s", mode="crop", dn_source="cache",
        model_tag="t", device="cpu",
        profile={"_crs_obj": None, "_transform_obj": None},
        anchored=anchored, outputs={}, elapsed_sec=0.1)
    assert payload["anchored"]["label"] == "ANCHORED (not learned)"
    assert payload["anchored"]["source"].startswith("constant:")
    assert payload["stats"]["min"] == pytest.approx(5.0)


# ---------------------------------------------------------------------------
# Full predictor round trip (torch-gated)
# ---------------------------------------------------------------------------

def _make_checkpoint(tmp_path, a0=2.0, b0=4.0):
    torch = pytest.importorskip("torch")
    from depthwizard.calibration_net import CalibrationNet

    net = CalibrationNet(in_ch=1, widths=(8, 16, 32), a0=a0, b0=b0)
    ckpt = {
        "model_state": net.state_dict(),
        "use_rgb": False, "widths": [8, 16, 32], "clamp_min": 0.0,
        "affine_init": {"a": a0, "b": b0}, "loss": "l1", "epoch": 0,
        "val_subset_mae": 0.0, "splits_json": "n/a",
    }
    p = tmp_path / "best.pt"
    torch.save(ckpt, p)
    return p, a0, b0


def test_predictor_round_trip_exact_affine(tmp_path):
    """With a fresh (untrained) net the prediction is EXACTLY the affine
    baseline — the same zero-init property the training contract pins.
    Uses a smooth linear Dn so the bilinear resize round trip is exact."""
    torch = pytest.importorskip("torch")
    from depthwizard.inference import DepthWizardPredictor

    ckpt, a0, b0 = _make_checkpoint(tmp_path)
    pred = DepthWizardPredictor(ckpt, device="cpu", live_backbone=False)

    rng = np.random.default_rng(2)
    rgb = rng.integers(0, 255, (256, 256, 3), dtype=np.uint8)

    # resolve_dn without cache/live/explicit must raise honestly:
    with pytest.raises(FileNotFoundError):
        pred.resolve_dn(rgb, stem="nope")

    # explicit dn path round trip via tmp file (linear ramp: resize-exact).
    # resolve_dn returns RAW relative depth; predict() normalizes internally.
    xs = np.linspace(0.0, 1.0, 256, dtype=np.float32)
    dn = np.tile(xs[None, :], (256, 1))
    raw = dn * 10.0
    np.save(tmp_path / "nope.npy", raw)
    res = pred.resolve_dn(rgb, stem="nope", dn_path=tmp_path / "nope.npy")
    assert res.source == "explicit"

    out = pred.predict(rgb, res.raw, mode="crop")    # small image -> letterbox path
    expected = np.clip(a0 * dn + b0, 0, None)
    assert out.shape == (256, 256)
    np.testing.assert_allclose(out, expected, atol=5e-3)


def test_predictor_rejects_grid_mismatch(tmp_path):
    pytest.importorskip("torch")
    from depthwizard.inference import DepthWizardPredictor

    ckpt, _, _ = _make_checkpoint(tmp_path)
    pred = DepthWizardPredictor(ckpt, device="cpu", live_backbone=False)
    rgb = np.zeros((64, 64, 3), dtype=np.uint8)
    dn_bad = np.zeros((32, 32), dtype=np.float32)
    with pytest.raises(ValueError):
        pred.predict(rgb, dn_bad, mode="crop")
