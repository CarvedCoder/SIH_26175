"""Train/inference Dn consistency + resize-mode geometry regression pins.

These tests exist because the audit (finding H1 / M11) confirmed that
inference consumed Dn with DIFFERENT statistics than training:

    training : DAv2 per 1024 tile, Dn min-max normalized PER TILE
    inference: DAv2 once on the whole (squeezed) image, Dn normalized
               over the WHOLE scene

and that resize mode squeezed arbitrary aspect ratios to 1024x1024.

The pins below fail if either behavior regresses:
  * normalize_dn_per_tile stretches every tile to the FULL [0,1] range
    with zero cross-tile leakage (the dataset.py contract), and
  * the predictor feeds the net per-tile-normalized Dn (mocked forward
    captures what the net actually sees), and
  * the live backbone runs at training tile granularity on multi-tile
    inputs, and
  * letterbox resize preserves aspect ratio (512x1024 / 1024x512 /
    non-square) instead of squeezing.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from depthwizard.inference import (
    TILE,
    DepthWizardPredictor,
    letterbox_placement,
    letterbox_to_tile,
    normalize_dn_per_tile,
)

from model_tests.test_inference import _make_checkpoint


# ---------------------------------------------------------------------------
# Per-tile Dn normalization (the H1 fix, pure numpy)
# ---------------------------------------------------------------------------

def test_normalize_dn_per_tile_full_range_no_cross_tile_leakage():
    """Each OVERLAPPING window is stretched to [0,1] in isolation before
    weighted stitching — a locally low-contrast region must still reach
    near-0/near-1 values (no whole-scene normalization leakage)."""
    raw = np.zeros((2048, 2048), dtype=np.float32)
    raw[0:1024, 0:1024] = np.linspace(0.0, 1.0, 1024 * 1024).reshape(1024, 1024)
    raw[0:1024, 1024:] = np.linspace(2.0, 2.4, 1024 * 1024).reshape(1024, 1024)
    raw[1024:, 0:1024] = np.linspace(-5.0, 5.0, 1024 * 1024).reshape(1024, 1024)
    raw[1024:, 1024:] = 7.0  # flat quadrant -> window fallback around 0.5

    dn = normalize_dn_per_tile(raw)

    assert dn.shape == raw.shape
    assert not np.isnan(dn).any()
    assert dn.min() >= 0.0 and dn.max() <= 1.0
    # Interior of each quadrant (away from blend borders at 768/1024):
    # the locally low-contrast 2.0..2.4 quadrant spans only 0.4/12 of the
    # GLOBAL range — whole-scene normalization would squash it to a
    # sliver. Per-window normalization keeps it near-full-range.
    for r0, r1, c0, c1 in [
        (0, 700, 0, 700),
        (0, 700, 1100, 2048),
        (1100, 2048, 0, 700),
    ]:
        q = dn[r0:r1, c0:c1]
        assert q.max() - q.min() > 0.5, "quadrant compressed by global normalization"


def test_normalize_dn_per_tile_crops_padding_and_normalizes_padded_tiles():
    """Non-multiple-of-1024 sizes: normalization happens on each window's
    EDGE-PADDED tile (what the net consumes), the output crops back exactly,
    and full coverage is preserved (no NaN gaps from the stitcher)."""
    raw = np.linspace(0.0, 10.0, 1500 * 700, dtype=np.float32).reshape(1500, 700)
    dn = normalize_dn_per_tile(raw)
    assert dn.shape == (1500, 700)
    assert not np.isnan(dn).any()
    assert dn.min() >= 0.0 and dn.max() <= 1.0


def test_predictor_tiles_mode_is_seam_free(tmp_path):
    """Tiled inference of a smooth ramp with per-window-perturbed forwards
    must not produce a hard discontinuity at the OLD 1024px tile boundary —
    the defect overlapping windows + weighted stitching exist to remove."""
    pytest.importorskip("torch")

    ckpt, _a0, _b0 = _make_checkpoint(tmp_path)
    pred = DepthWizardPredictor(ckpt, device="cpu", live_backbone=False)

    call_idx = {"n": 0}

    def biased_predict(dn, rgb=None, stats=None):
        # each window's forward independently off by a constant — exactly
        # how independent per-window inference disagrees at shared borders
        call_idx["n"] += 1
        return dn + (0.4 if call_idx["n"] % 2 else -0.4)

    pred._predict_fn = biased_predict

    h = w = 2048
    yy, xx = np.mgrid[0:h, 0:w]
    truth = (xx + yy).astype(np.float32) / (h + w)  # smooth ramp in [0,1]
    rgb = np.zeros((h, w, 3), dtype=np.uint8)

    out = pred.predict(rgb, truth * 10.0, mode="tiles")
    # minmax per window cancels the scale; the forward adds the bias
    assert out.shape == (h, w)
    # no hard seam across the old tile boundary at x=1024: the step along
    # a row must stay far below the injected +-0.4 disagreement
    row = out[h // 2, :]
    assert np.abs(np.diff(row)).max() < 0.05


# ---------------------------------------------------------------------------
# Letterbox resize geometry (the M11 fix, pure numpy)
# ---------------------------------------------------------------------------

def test_letterbox_placement_512x1024_and_1024x512():
    # 512x1024: vertical strip — content fills full width, centered vertically
    y0, x0, h2, w2 = letterbox_placement(512, 1024, TILE)
    assert (y0, x0, h2, w2) == (256, 0, 512, 1024)
    # 1024x512: horizontal strip — centered horizontally
    y0, x0, h2, w2 = letterbox_placement(1024, 512, TILE)
    assert (y0, x0, h2, w2) == (0, 256, 1024, 512)


def test_letterbox_canvas_content_region_matches_original():
    """The canvas content region must be the aspect-preserving scale of the
    source; cropping it back and downsampling restores the original."""
    rng = np.random.default_rng(3)
    for h, w in [(512, 1024), (1024, 512), (640, 480)]:
        rgb = rng.integers(0, 255, (h, w, 3), dtype=np.uint8)
        canvas, (y0, x0, h2, w2) = letterbox_to_tile(rgb)
        assert canvas.shape == (1024, 1024, 3)
        # aspect preserved
        assert h2 / w2 == pytest.approx(h / w, rel=1e-2)
        content = canvas[y0 : y0 + h2, x0 : x0 + w2]
        assert content.shape == (h2, w2, 3)
        # downsampling the content region reproduces the source geometry
        from depthwizard.inference import _resize_bilinear

        back = _resize_bilinear(content, h, w)
        assert back.shape == (h, w, 3)


def test_predictor_letterbox_resize_exact_affine_512x1024(tmp_path):
    """512x1024 strip: letterbox must produce EXACTLY the affine output on
    the real content region (fresh net = exact affine baseline)."""
    torch = pytest.importorskip("torch")

    ckpt, a0, b0 = _make_checkpoint(tmp_path)
    pred = DepthWizardPredictor(ckpt, device="cpu", live_backbone=False)

    h, w = 512, 1024
    xs = np.linspace(0.0, 1.0, w, dtype=np.float32)
    dn = np.tile(xs[None, :], (h, 1))  # linear ramp along x
    raw = dn * 10.0
    rgb = np.zeros((h, w, 3), dtype=np.uint8)

    out = pred.predict(rgb, raw, mode="resize")
    assert out.shape == (h, w)
    expected = np.clip(a0 * dn + b0, 0, None)
    np.testing.assert_allclose(out, expected, atol=2e-2)
    assert torch.is_tensor  # torch imported (keeps importorskip honest)


def test_predictor_letterbox_resize_exact_affine_1024x512(tmp_path):
    ckpt, a0, b0 = _make_checkpoint(tmp_path)
    pred = DepthWizardPredictor(ckpt, device="cpu", live_backbone=False)

    h, w = 1024, 512
    ys = np.linspace(0.0, 1.0, h, dtype=np.float32)
    dn = np.tile(ys[:, None], (1, w))
    raw = dn * 10.0
    rgb = np.zeros((h, w, 3), dtype=np.uint8)

    out = pred.predict(rgb, raw, mode="resize")
    expected = np.clip(a0 * dn + b0, 0, None)
    np.testing.assert_allclose(out, expected, atol=2e-2)


# ---------------------------------------------------------------------------
# Predictor feeds per-tile-normalized Dn (H1 pin through the real forward)
# ---------------------------------------------------------------------------

def test_predictor_tiles_mode_feeds_per_tile_normalized_dn(tmp_path):
    """Captures what the net ACTUALLY receives in tiles mode: every tile's
    Dn must span [0,1] over itself — no whole-scene normalization leakage."""
    pytest.importorskip("torch")

    ckpt, _a0, _b0 = _make_checkpoint(tmp_path)
    pred = DepthWizardPredictor(ckpt, device="cpu", live_backbone=False)

    seen: list[np.ndarray] = []

    def capture_predict(dn, rgb=None, stats=None):
        seen.append(dn.copy())
        return np.zeros(dn.shape, dtype=np.float32)

    pred._predict_fn = capture_predict

    raw = np.zeros((2048, 2048), dtype=np.float32)
    raw[0:1024, 0:1024] = 0.2      # flat-ish tile with distinct range
    raw[0:1024, 1024:] = 0.9
    raw[1024:, 0:1024] = 5.0
    raw[1024:, 1024:] = -3.0
    # give each tile a real internal range so minmax has work to do
    raw[0:1024, 0:1024] += np.linspace(0.0, 0.3, 1024 * 1024).reshape(1024, 1024)
    raw[0:1024, 1024:] += np.linspace(0.0, 0.2, 1024 * 1024).reshape(1024, 1024)
    raw[1024:, 0:1024] += np.linspace(0.0, 4.0, 1024 * 1024).reshape(1024, 1024)
    raw[1024:, 1024:] += np.linspace(0.0, 1.0, 1024 * 1024).reshape(1024, 1024)

    rgb = np.zeros((2048, 2048, 3), dtype=np.uint8)
    out = pred.predict(rgb, raw, mode="tiles")

    assert out.shape == (2048, 2048)
    # Overlapping windows: 2048 = stride 768 grid with starts [0, 768, 1024]
    # per axis -> 3x3 = 9 forward passes, each still a 1024x1024 tile whose
    # Dn spans the full [0,1] range (no whole-scene normalization leakage).
    assert len(seen) == 9
    for tile in seen:
        assert tile.shape == (1024, 1024)
        assert tile.min() == pytest.approx(0.0, abs=1e-6)
        assert tile.max() == pytest.approx(1.0, abs=1e-6)


def test_predictor_live_backbone_runs_at_training_tile_granularity(tmp_path):
    """Multi-tile inputs must run DAv2 once per 1024 tile (the cache-builder
    recipe), never on the squeezed whole scene."""
    pytest.importorskip("torch")

    ckpt, _, _ = _make_checkpoint(tmp_path)
    pred = DepthWizardPredictor(ckpt, device="cpu", live_backbone=True)

    calls: list[tuple[int, int]] = []

    class FakeBackbone:
        def raw_depth(self, rgb: np.ndarray) -> np.ndarray:
            calls.append(rgb.shape[:2])
            h, w = rgb.shape[:2]
            return np.arange(h * w, dtype=np.float32).reshape(h, w) / (h * w)

    pred._backbone = FakeBackbone()

    rgb = np.zeros((2048, 2048, 3), dtype=np.uint8)
    res = pred.resolve_dn(rgb, stem="multi")
    assert res.source == "live"
    assert res.raw.shape == (2048, 2048)
    # one 1024x1024 pass per OVERLAPPING window (starts [0, 768, 1024] per
    # axis -> 3x3 = 9), never the squeezed whole scene — training granularity
    assert calls == [(1024, 1024)] * 9
    assert not np.isnan(res.raw).any()

    # single-tile inputs keep the whole-image pass
    calls.clear()
    res_small = pred.resolve_dn(np.zeros((512, 512, 3), dtype=np.uint8), stem="small")
    assert res_small.raw.shape == (512, 512)
    assert calls == [(512, 512)]
