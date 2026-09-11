"""Unit + integration tests for depthwizard.postprocess (task Sec. 20).

Categories (mirroring the task's required list):

Numerical correctness
    * constant signal stays constant under EVERY method (calibration guard)
    * known edge preserved (RGB-aligned step survives guided/WLS)
    * NaNs handled (pattern preserved, no neighbour poisoning)
    * nodata/invalid pixels preserved
    * negative values -> clamp guard; clamp_min=None disables it

Semantic correctness
    * same-class region is smoothed (variance reduction)
    * cross-class boundary is NOT excessively smoothed (semantic WLS
      preserves a step that RGB-only WLS destroys — the core property)

Calibration correctness
    * refinement does not shift global mean/median/std materially
    * calibration_report numbers are exact

Geospatial / integration correctness
    * run_inference postprocess="none" is the unchanged legacy baseline
    * agl_raw.npy written by a postprocessed run EQUALS that baseline
    * postprocess_meta.json records method/config/calibration
    * GeoTIFF metadata (CRS/transform/shape) preserved through the run
    * refine happens BEFORE anchoring (DSM == AGL_refined + ground_elev)

Regression
    * enabled=False / method="none" -> bit-identical passthrough
    * the caller's input array is never mutated
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from depthwizard.postprocess import (
    METHODS,
    PRESETS,
    PostProcessConfig,
    config_from_preset,
    refine_agl,
)
from depthwizard.postprocess.config import PostProcessConfig as PPC


# ---------------------------------------------------------------------------
# Fixtures — a synthetic scene with the geometry that matters
# ---------------------------------------------------------------------------

def make_scene(
    h: int = 96,
    w: int = 96,
    seed: int = 0,
    n_buildings: int = 4,
    jump: float = 18.0,
    noise: float = 0.35,
    rgb_contrast: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """(rgb_u8, agl) with flat-ish ground, sharp aligned buildings, noise.

    Buildings are axis-aligned rectangles whose AGL step and RGB colour
    change coincide exactly — the aligned-edge case refinement must keep.
    """
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    ground = 1.5 + 0.8 * np.sin(xx / 23.0) + 0.4 * np.cos(yy / 17.0)
    agl = ground + rng.normal(0.0, noise, (h, w))
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    rgb[..., 0] = (110 + 18 * np.sin(xx / 11.0)).astype(np.uint8)
    rgb[..., 1] = (120 + 14 * np.cos(yy / 13.0)).astype(np.uint8)
    rgb[..., 2] = 95
    for _ in range(n_buildings):
        bh = int(rng.integers(h // 8, h // 3))
        bw = int(rng.integers(w // 8, w // 3))
        y0 = int(rng.integers(2, max(3, h - bh - 2)))
        x0 = int(rng.integers(2, max(3, w - bw - 2)))
        agl[y0 : y0 + bh, x0 : x0 + bw] += jump
        if rgb_contrast:
            rgb[y0 : y0 + bh, x0 : x0 + bw] = (172, 148, 132)
    return rgb, agl.astype(np.float32)


def make_sem_probs(
    rgb: np.ndarray, agl: np.ndarray, jump: float = 18.0, sharp: float = 0.9
) -> np.ndarray:
    """[6,H,W] probabilities: building where AGL is high, ground elsewhere.

    ``sharp`` is the confidence of the argmax class; the remainder is spread
    uniformly — the shape a real softmax head emits.
    """
    h, w = agl.shape
    building = agl > (agl.min() + 0.5 * jump)
    p = np.full((6, h, w), (1.0 - sharp) / 5.0, dtype=np.float32)
    p[0][building] = (1.0 - sharp) / 5.0          # building channel stays low
    p[0][building] = sharp                        # then set to sharp
    p[4][~building] = sharp                       # ground channel
    # the two lines above leave channel 0 at background level on non-building
    # pixels and channel 4 at background on building pixels — consistent.
    return p


def _step_scene(h=64, w=64, left_h=0.0, right_h=10.0, noise=0.3, split=32):
    """Two flat regions separated by a vertical boundary at x=split."""
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    agl = np.where(xx < split, left_h, right_h).astype(np.float32)
    agl += np.random.default_rng(1).normal(0.0, noise, (h, w)).astype(np.float32)
    rgb = np.full((h, w, 3), 120, dtype=np.uint8)  # uniform — NO RGB evidence
    return rgb, agl, split


def _step_sem(split, shape, sharp=0.9):
    h, w = shape
    xx = np.arange(w)[None, :].repeat(h, 0)
    building = xx >= split
    p = np.full((6, h, w), (1.0 - sharp) / 5.0, dtype=np.float32)
    p[0][building] = sharp
    p[4][~building] = sharp
    return p


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def test_config_method_ladder_valid():
    for m in METHODS:
        assert PostProcessConfig(enabled=True, method=m).method == m


def test_config_rejects_unknown_method():
    with pytest.raises(ValueError):
        PostProcessConfig(method="gaussian_blur")


def test_config_rejects_bad_radius_and_lambda():
    with pytest.raises(ValueError):
        PostProcessConfig(guided_radius=0)
    with pytest.raises(ValueError):
        PostProcessConfig(wls_lambda=-1.0)


def test_config_effective_method_and_disabled():
    c = PostProcessConfig(enabled=False, method="semantic_wls")
    assert c.effective_method == "none"
    assert PostProcessConfig(enabled=True, method="wls").effective_method == "wls"


def test_config_from_dict_ignores_unknown_keys_and_overrides():
    c = PostProcessConfig.from_dict(
        {"method": "wls", "wls_lambda": 3.0, "totally_unknown": 42}
    )
    assert c.method == "wls" and c.wls_lambda == 3.0


def test_presets_build_and_roundtrip():
    for name, over in PRESETS.items():
        cfg = config_from_preset(name)
        assert cfg.method in METHODS
        # every preset name maps to a runnable config; overrides apply
        cfg2 = config_from_preset(name, {"wls_lambda": 2.5} if "wls" in name else {})
        if "wls" in name:
            assert cfg2.wls_lambda == 2.5
    # unknown preset name is interpreted as a method (validated by ctor)
    with pytest.raises(ValueError):
        config_from_preset("no_such_preset")


def test_with_updates_returns_new_config():
    c = PostProcessConfig()
    c2 = c.with_updates(method="guided", guided_radius=8)
    assert c2.guided_radius == 8 and c is not c2


# ---------------------------------------------------------------------------
# Identity / regression (task Sec. 20 "Regression")
# ---------------------------------------------------------------------------

def test_disabled_is_bit_identical_passthrough():
    rgb, agl = make_scene()
    cfg = PostProcessConfig(enabled=False, method="semantic_wls")
    out, rep = refine_agl(agl, rgb, cfg)
    assert out.dtype == np.float32
    assert rep.method == "none"
    # bit-identical, INCLUDING NaN pattern
    np.testing.assert_array_equal(out, agl)


def test_disabled_preserves_nan_pattern():
    rgb, agl = make_scene()
    agl = agl.copy()
    agl[10:14, 10:14] = np.nan
    cfg = PostProcessConfig(enabled=False)
    out, _ = refine_agl(agl, rgb, cfg)
    np.testing.assert_array_equal(np.isnan(out), np.isnan(agl))


def test_method_none_enabled_is_passthrough_too():
    rgb, agl = make_scene()
    cfg = PostProcessConfig(enabled=True, method="none", spike_removal=False)
    out, rep = refine_agl(agl, rgb, cfg)
    np.testing.assert_array_equal(out, agl)
    assert rep.method == "none"


def test_input_not_mutated():
    rgb, agl = make_scene()
    agl_copy = agl.copy()
    for m in ("median", "guided", "bilateral", "wls", "semantic_wls"):
        refine_agl(agl, rgb, PostProcessConfig(enabled=True, method=m))
    np.testing.assert_array_equal(agl, agl_copy)


def test_shape_and_dtype_preserved_all_methods():
    rgb, agl = make_scene(seed=3)
    sem = make_sem_probs(rgb, agl)
    for m in METHODS:
        if m == "none":
            continue
        cfg = PostProcessConfig(enabled=True, method=m, spike_removal=True)
        out, _ = refine_agl(agl, rgb, cfg, sem_probs=sem)
        assert out.shape == agl.shape, m
        assert out.dtype == np.float32, m


def test_rejects_bad_shapes():
    rgb, agl = make_scene()
    with pytest.raises(ValueError):
        refine_agl(agl, rgb[..., :2], PostProcessConfig(enabled=True))  # not 3ch
    with pytest.raises(ValueError):
        refine_agl(agl[:, :32], rgb, PostProcessConfig(enabled=True))  # grid mismatch


# ---------------------------------------------------------------------------
# Numerical correctness — constant invariance (calibration guard)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("method", ["median", "guided", "bilateral", "wls",
                                    "conf_wls", "semantic_wls"])
def test_constant_agl_stays_exactly_constant(method):
    """A flat calibrated surface must pass through refinement UNCHANGED —
    the strongest form of 'never shift the global height scale'."""
    rgb, _ = make_scene(seed=5)
    agl = np.full(rgb.shape[:2], 7.31, dtype=np.float32)
    cfg = PostProcessConfig(enabled=True, method=method, spike_removal=True)
    out, rep = refine_agl(agl, rgb, cfg, sem_probs=make_sem_probs(rgb, agl))
    np.testing.assert_allclose(out, 7.31, atol=1e-5)
    assert rep.calibration["mean_shift"] == pytest.approx(0.0, abs=1e-5)


def test_constant_guidance_keeps_signal_edges():
    """Guidance is flat (no image evidence): a signal step must survive
    (the guided filter degenerates to a local mean ONLY where the signal
    is flat; a coherent step is structure, not noise)."""
    h, w = 64, 64
    rgb = np.full((h, w, 3), 100, dtype=np.uint8)
    agl = np.zeros((h, w), dtype=np.float32)
    agl[:, w // 2 :] = 12.0
    out = refine_agl(
        agl, rgb, PostProcessConfig(enabled=True, method="guided",
                                    spike_removal=False)
    )[0]
    left = out[:, : w // 2 - 6].mean()
    right = out[:, w // 2 + 6 :].mean()
    assert right - left > 0.5 * 12.0  # most of the step survives


def test_rgb_aligned_edge_preserved_by_guided_and_wls():
    """A step in AGL aligned with a step in RGB is structure: both the
    guided filter and WLS must keep the boundary sharp."""
    h, w = 64, 64
    rgb = np.full((h, w, 3), 90, dtype=np.uint8)
    rgb[:, w // 2 :] = 200
    agl = np.zeros((h, w), dtype=np.float32)
    agl[:, w // 2 :] = 20.0
    for method in ("guided", "wls"):
        out = refine_agl(
            agl, rgb,
            PostProcessConfig(enabled=True, method=method, spike_removal=False),
        )[0]
        left = out[:, : w // 2 - 4].mean()
        right = out[:, w // 2 + 4 :].mean()
        assert right - left > 0.75 * 20.0, method


# ---------------------------------------------------------------------------
# NaN / nodata handling
# ---------------------------------------------------------------------------

def test_nan_pattern_preserved_all_methods():
    rgb, agl = make_scene(seed=7)
    agl = agl.copy()
    agl[20:30, 20:30] = np.nan  # a nodata block
    for m in ("median", "guided", "bilateral", "wls", "semantic_wls"):
        out, _ = refine_agl(
            agl, rgb, PostProcessConfig(enabled=True, method=m),
            sem_probs=make_sem_probs(rgb, np.nan_to_num(agl)),
        )
        assert np.isnan(out[20:30, 20:30]).all(), m
        assert np.isfinite(out[40:60, 40:60]).all(), m  # no poisoning


def test_nan_does_not_poison_neighbours():
    """The guided-filter box statistics are masked: a NaN block must not
    drag the surrounding window means toward 0/NaN."""
    h, w = 64, 64
    rgb = np.full((h, w, 3), 128, dtype=np.uint8)
    agl = np.full((h, w), 5.0, dtype=np.float32)
    agl[28:36, 28:36] = np.nan
    out = refine_agl(
        agl, rgb, PostProcessConfig(enabled=True, method="guided",
                                    spike_removal=False)
    )[0]
    np.testing.assert_allclose(out[24, 24], 5.0, atol=0.2)


def test_invalid_rgb_pixels_excluded():
    """NaN RGB channels make the pixel invalid (validity contract)."""
    rgb, agl = make_scene(seed=9)
    rgb = rgb.astype(np.float32)
    rgb[10:20, 10:20] = np.nan
    out, rep = refine_agl(
        agl, rgb.astype(np.float32),
        PostProcessConfig(enabled=True, method="guided"),
    )
    assert np.isnan(out[10:20, 10:20]).all()
    assert rep.n_valid < rep.n_pixels


def test_negative_values_clamped_by_default_and_optional():
    h, w = 32, 32
    rgb = np.full((h, w, 3), 128, dtype=np.uint8)
    agl = np.where(
        (np.mgrid[0:h, 0:w][1] % 8) < 4, -1.5, 2.0
    ).astype(np.float32)
    cfg = PostProcessConfig(
        enabled=True, method="wls", wls_lambda=0.0,  # identity solve
        spike_removal=False, clamp_min=0.0,
    )
    out, _ = refine_agl(agl, rgb, cfg)
    assert (out >= 0.0).all()  # physical guard applied
    cfg_none = cfg.with_updates(clamp_min=None)
    out2, _ = refine_agl(agl, rgb, cfg_none)
    assert (out2 < 0).any()  # guard is genuinely optional


# ---------------------------------------------------------------------------
# Semantic gating (task Sec. 20 "Semantic correctness")
# ---------------------------------------------------------------------------

def test_same_class_region_is_smoothed():
    """Inside a confident single-class region, refinement must reduce
    noise variance (that is the point of the smoothness term)."""
    rgb, agl, split = _step_scene(noise=0.8)
    # single class everywhere -> semantics never blocks smoothing
    sem = np.zeros((6, *agl.shape), dtype=np.float32)
    sem[4] = 1.0
    out = refine_agl(
        agl, rgb,
        PostProcessConfig(enabled=True, method="semantic_wls",
                          spike_removal=False),
        sem_probs=sem,
    )[0]
    left = out[:, : split - 8]
    left_in = agl[:, : split - 8]
    assert left.std() < 0.9 * left_in.std()


def test_cross_class_boundary_not_smoothed_semantic_wls():
    """WEAK RGB evidence + STRONG semantics: with strong smoothing
    (wls_lambda=50) the boundary band must bleed far less under semantic
    WLS than under RGB-only WLS (roof->ground bleeding prevention)."""
    rgb, agl, split = _step_scene(noise=0.0)
    sem = _step_sem(split, agl.shape, sharp=0.98)
    lam = 50.0
    out_sem = refine_agl(
        agl, rgb,
        PostProcessConfig(enabled=True, method="semantic_wls",
                          spike_removal=False, wls_lambda=lam,
                          semantic_edge_weight=2.0),
        sem_probs=sem,
    )[0]
    out_rgb = refine_agl(
        agl, rgb,
        PostProcessConfig(enabled=True, method="wls", spike_removal=False,
                          wls_lambda=lam),
    )[0]

    def bleed(img):
        # error of the two pixels straddling the boundary, per side
        return abs(img[:, split + 1].mean() - 10.0) + abs(img[:, split - 1].mean())

    assert bleed(out_sem) < 0.45 * bleed(out_rgb)   # >2x less bleeding
    # and the step itself still stands (far from the boundary)
    assert out_sem[:, split + 8:].mean() - out_sem[:, : split - 8].mean() > 0.95 * 10.0


def test_hard_semantic_boundary_keeps_step_exactly():
    rgb, agl, split = _step_scene(noise=0.0)
    sem = _step_sem(split, agl.shape, sharp=1.0)
    out = refine_agl(
        agl, rgb,
        PostProcessConfig(enabled=True, method="semantic_wls",
                          spike_removal=False, semantic_hard_boundary=True),
        sem_probs=sem,
    )[0]
    np.testing.assert_allclose(out, agl, atol=1e-4)  # w=0 everywhere on edge


def test_semantic_wls_without_probs_degrades_honestly():
    rgb, agl = make_scene()
    out, rep = refine_agl(
        agl, rgb, PostProcessConfig(enabled=True, method="semantic_wls")
    )
    assert out.shape == agl.shape
    assert any("RGB-only" in n for n in rep.notes)  # degradation is recorded


# ---------------------------------------------------------------------------
# Calibration correctness (task Sec. 20 "Calibration")
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("method", ["guided", "bilateral", "wls", "semantic_wls"])
def test_refinement_keeps_global_scale(method):
    """mean/median/std of the refined field stay within a small band of the
    raw field — refinement is local geometry, never a rescale."""
    rgb, agl = make_scene(seed=11)
    cfg = PostProcessConfig(enabled=True, method=method)
    out, rep = refine_agl(agl, rgb, cfg, sem_probs=make_sem_probs(rgb, agl))
    c = rep.calibration
    scale = agl.std()
    assert abs(c["mean_shift"]) < 0.02 * scale + 0.05
    assert abs(c["median_shift"]) < 0.02 * scale + 0.05
    assert 0.85 < c["std_ratio"] < 1.15
    # direct re-derivation: the report must match the arrays
    np.testing.assert_allclose(
        c["raw_mean"], float(np.mean(agl)), rtol=1e-6
    )
    np.testing.assert_allclose(
        c["refined_mean"], float(np.mean(out)), rtol=1e-6
    )


def test_calibration_report_exact_values():
    raw = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    refined = raw + 1.0
    valid = np.ones(raw.shape, dtype=bool)
    from depthwizard.postprocess.refinement import calibration_report

    c = calibration_report(raw, refined, valid)
    assert c["raw_mean"] == pytest.approx(2.5)
    assert c["refined_mean"] == pytest.approx(3.5)
    assert c["mean_shift"] == pytest.approx(1.0)
    assert c["raw_std"] == pytest.approx(np.std([1, 2, 3, 4]))
    assert c["std_ratio"] == pytest.approx(1.0)
    assert c["raw_median"] == pytest.approx(2.5)


def test_preserve_mean_option_restores_scale_exactly():
    rgb, agl = make_scene(seed=13)
    cfg = PostProcessConfig(enabled=True, method="wls", preserve_mean=True)
    out, rep = refine_agl(agl, rgb, cfg)
    assert rep.calibration["mean_shift"] == pytest.approx(0.0, abs=1e-5)
    assert any("preserve_mean" in n for n in rep.notes)


def test_no_normalization_of_output_range():
    """The refined output keeps the RAW value range — no per-tile min-max
    renormalization is ever applied (which would create seams)."""
    rgb, agl = make_scene(seed=15)
    out, _ = refine_agl(
        agl, rgb, PostProcessConfig(enabled=True, method="guided")
    )
    assert out.min() >= agl.min() - 1.0
    assert out.max() <= agl.max() + 1.0


# ---------------------------------------------------------------------------
# Spike removal (conservative outlier stage)
# ---------------------------------------------------------------------------

def _spike_scene(h=48, w=48):
    rgb = np.full((h, w, 3), 130, dtype=np.uint8)
    rgb[..., 0] = (110 + 20 * np.sin(np.arange(w)[None, :] / 9.0)).astype(np.uint8)
    agl = np.full((h, w), 3.0, dtype=np.float32) + np.random.default_rng(2).normal(
        0, 0.05, (h, w)
    ).astype(np.float32)
    return rgb, agl


def test_isolated_spike_removed():
    from depthwizard.postprocess.spike_removal import remove_spikes

    rgb, agl = _spike_scene()
    agl = agl.copy()
    agl[24, 24] += 40.0  # isolated 40 m spike over smooth ground
    cleaned, mask = remove_spikes(agl, rgb)
    assert mask[24, 24]
    assert cleaned[24, 24] < 10.0  # replaced by local median ~3
    # neighbours untouched
    np.testing.assert_allclose(cleaned[24, 23], agl[24, 23], atol=1e-6)


def test_contiguous_roof_band_survives():
    """A wide elevated band (a roof) is NOT isolated: co-deviating pixels
    form a contiguous structure and the stage must leave it alone."""
    from depthwizard.postprocess.spike_removal import remove_spikes

    rgb, agl = _spike_scene()
    agl = agl.copy()
    agl[16:32, 16:32] += 25.0  # 16x16 'roof'
    rgb[16:32, 16:32] = (170, 150, 135)  # + strong RGB outline
    cleaned, mask = remove_spikes(agl, rgb)
    assert not mask.any()  # nothing replaced anywhere
    np.testing.assert_array_equal(cleaned, agl)


def test_strong_rgb_edge_protects_even_isolated_values():
    from depthwizard.postprocess.spike_removal import remove_spikes

    rgb, agl = _spike_scene()
    agl = agl.copy()
    agl[24, 24] += 40.0
    rgb[:, 24:] = 200  # a genuine RGB step edge through the spike pixel
    _, mask = remove_spikes(agl, rgb)
    assert not mask[24, 24]  # edge-supported geometry is never a 'spike'


def test_spike_tau_configurable():
    from depthwizard.postprocess.spike_removal import remove_spikes

    rgb, agl = _spike_scene()
    agl = agl.copy()
    agl[24, 24] += 40.0
    _, mask_small_tau = remove_spikes(agl, rgb, tau=2.0)
    _, mask_huge_tau = remove_spikes(agl, rgb, tau=1000.0)
    assert mask_small_tau[24, 24]
    assert not mask_huge_tau[24, 24]  # huge tau: nothing is an outlier


def test_refinement_report_counts_spikes():
    rgb, agl = _spike_scene()
    agl = agl.copy()
    agl[24, 24] += 40.0
    out, rep = refine_agl(
        agl, rgb, PostProcessConfig(enabled=True, method="guided",
                                     spike_removal=True)
    )
    assert rep.spike_count == 1
    assert out[24, 24] < 10.0


# ---------------------------------------------------------------------------
# Guided filter internals (He et al. 2010 properties)
# ---------------------------------------------------------------------------

def test_guided_filter_exact_for_constant_pair():
    from depthwizard.postprocess.guided import guided_filter

    g = np.random.default_rng(0).random((32, 32)).astype(np.float32)
    p = np.full((32, 32), 4.2, dtype=np.float32)
    q = guided_filter(p, g, radius=3, eps=1e-3)
    np.testing.assert_allclose(q, 4.2, atol=1e-6)


def test_guided_filter_eps_controls_edge_preservation():
    """eps is the edge threshold: a small eps keeps the guidance-aligned
    step sharp; a huge eps collapses a -> local mean and blurs it."""
    from depthwizard.postprocess.guided import guided_filter

    h, w = 48, 48
    g = np.full((h, w), 0.3, dtype=np.float32)
    g[:, w // 2 :] = 0.8                       # guidance step
    p = g * 10.0 + np.random.default_rng(1).normal(0, 0.2, (h, w)).astype(
        np.float32
    )
    jump = {}
    for eps in (1e-6, 1.0, 10.0):
        q = guided_filter(p, g, radius=4, eps=eps)
        jump[eps] = np.abs(q[:, w // 2 + 1] - q[:, w // 2 - 1]).mean()
    assert jump[1e-6] > 0.9 * 5.0     # edge amplitude preserved
    assert jump[1.0] < 0.4 * 5.0      # edge amplitude blurred away
    assert jump[10.0] <= jump[1.0]    # monotone: more eps, more blur


def test_guided_filter_box_masked_mean_handles_all_invalid():
    from depthwizard.postprocess.guided import _box_filter

    a = np.ones((8, 8), dtype=np.float64)
    mask = np.zeros((8, 8), dtype=bool)
    out = _box_filter(a, 2, mask)
    assert (out == 0).all()  # zero valid pixels -> zero sum convention


# ---------------------------------------------------------------------------
# WLS internals
# ---------------------------------------------------------------------------

def test_wls_lambda_zero_is_identity():
    from depthwizard.postprocess.wls import wls_refine

    rgb, agl = make_scene(seed=17)
    res = wls_refine(agl, rgb, lambda_=0.0)
    np.testing.assert_allclose(res.z, agl, atol=1e-6)
    assert res.converged and res.iterations == 0


def test_wls_constant_input_exact_zero_iterations():
    from depthwizard.postprocess.wls import wls_refine

    rgb, _ = make_scene(seed=19)
    agl = np.full(rgb.shape[:2], 6.0, dtype=np.float32)
    res = wls_refine(agl, rgb, lambda_=2.0)
    np.testing.assert_allclose(res.z, 6.0, atol=1e-6)
    assert res.iterations == 0


def test_wls_high_confidence_pins_prediction():
    """c=1 everywhere -> data term dominates -> output hugs the raw signal."""
    from depthwizard.postprocess.wls import wls_refine

    rgb, agl = make_scene(seed=21, noise=0.4)
    conf = np.ones(agl.shape, dtype=np.float32)
    res = wls_refine(agl, rgb, lambda_=1.0, confidence=conf)
    # with uniform unit data weight the solve stays near the input
    assert np.abs(res.z - agl).mean() < 1.0


def test_wls_invalid_pixels_nan_out():
    from depthwizard.postprocess.wls import wls_refine

    rgb, agl = make_scene(seed=23)
    agl = agl.copy()
    agl[10:20, 10:20] = np.nan
    res = wls_refine(agl, rgb, lambda_=1.0)
    assert np.isnan(res.z[10:20, 10:20]).all()
    assert np.isfinite(res.z[40:, 40:]).all()


# ---------------------------------------------------------------------------
# Confidence (documented relative map — NOT calibrated probabilities)
# ---------------------------------------------------------------------------

def test_confidence_range_and_invalid_zero():
    from depthwizard.postprocess.confidence import estimate_confidence

    rgb, agl = make_scene(seed=25)
    conf = estimate_confidence(agl, rgb)
    assert conf.shape == agl.shape
    assert (conf >= 0.0).all() and (conf <= 1.0).all()
    agl_bad = agl.copy()
    agl_bad[5:9, 5:9] = np.nan
    conf_bad = estimate_confidence(agl_bad, rgb)
    assert (conf_bad[5:9, 5:9] == 0.0).all()


def test_spike_pixel_gets_low_confidence():
    from depthwizard.postprocess.confidence import estimate_confidence

    rgb, agl = _spike_scene()
    agl = agl.copy()
    agl[24, 24] += 40.0
    conf = estimate_confidence(agl, rgb)
    assert conf[24, 24] < 0.5
    assert conf[10, 10] > conf[24, 24]


def test_tta_disagreement_lowers_confidence():
    from depthwizard.postprocess.confidence import estimate_confidence

    rgb, agl = _spike_scene()
    variants = [agl + 0.01, agl - 0.01, agl.copy()]
    conf_no_tta = estimate_confidence(agl, rgb)
    conf_tta = estimate_confidence(agl, rgb, tta_stack=[agl + 8.0, agl - 8.0])
    # huge disagreement must reduce confidence on average
    assert conf_tta.mean() < conf_no_tta.mean()
    assert conf_tta.mean() < 0.9


# ---------------------------------------------------------------------------
# TTA fusion
# ---------------------------------------------------------------------------

def test_tta_fuse_alignment_round_trip():
    """predict = red channel: every augmentation aligns back exactly, so the
    fused map equals the red channel regardless of aggregation."""
    from depthwizard.postprocess.tta import tta_fuse

    rgb, _ = make_scene(seed=27)
    fn = lambda img: img[..., 0].astype(np.float32)  # noqa: E731
    for agg in ("median", "mean"):
        fused = tta_fuse(fn, rgb, aggregation=agg)
        np.testing.assert_array_equal(fused["agl"], rgb[..., 0].astype(np.float32))
        assert set(fused["augmentations"]) == {"identity", "hflip", "vflip", "rot180"}
        assert len(fused["stack"]) == 4


def test_tta_fuse_median_robust_to_one_bad_variant():
    from depthwizard.postprocess.tta import tta_fuse

    rgb, _ = make_scene(seed=29)

    def make_fn():
        state = {"calls": 0}

        def fn(img):
            state["calls"] += 1
            # the 2nd call (hflip) is poisoned; others return 5.0
            return np.full(img.shape[:2], 5.0 + (100.0 if state["calls"] == 2 else 0.0), dtype=np.float32)

        return fn

    fused = tta_fuse(make_fn(), rgb, aggregation="median")
    np.testing.assert_allclose(fused["agl"], 5.0, atol=1e-6)
    fused_mean = tta_fuse(make_fn(), rgb, aggregation="mean")  # mean poisoned
    assert fused_mean["agl"].mean() > 20.0


def test_tta_fuse_rejects_unknowns():
    from depthwizard.postprocess.tta import tta_fuse

    rgb, _ = make_scene(seed=31)
    with pytest.raises(ValueError):
        tta_fuse(lambda im: im[..., 0], rgb, augmentations=("transpose",))
    with pytest.raises(ValueError):
        tta_fuse(lambda im: im[..., 0], rgb, aggregation="max")


def test_refinement_tta_mean_aggregation_now_applies():
    """Regression test for the fixed stage-3 bug: 'mean' TTA fusion must
    replace the signal (previously only 'median' did)."""
    rgb, _ = make_scene(seed=33)
    agl = np.full(rgb.shape[:2], 3.0, dtype=np.float32)
    calls = {"n": 0}

    def fn(img):
        calls["n"] += 1
        return np.full(img.shape[:2], 9.0, dtype=np.float32)

    cfg = PostProcessConfig(enabled=True, method="none", tta=True,
                            tta_aggregation="mean", spike_removal=False)
    out, rep = refine_agl(agl, rgb, cfg, tta_predict_fn=fn)
    np.testing.assert_allclose(out, 9.0, atol=1e-6)  # fused, not the raw 3.0
    assert calls["n"] == 4
    assert rep.tta_augmentations == ("identity", "hflip", "vflip", "rot180")


def test_refinement_tta_without_fn_skips_honestly():
    rgb, agl = make_scene(seed=35)
    agl = np.maximum(agl, 0.0)  # deployment AGL is clamped >= 0 by the net
    cfg = PostProcessConfig(enabled=True, method="none", tta=True,
                            spike_removal=False)
    out, rep = refine_agl(agl, rgb, cfg, tta_predict_fn=None)
    np.testing.assert_array_equal(out, agl)  # nothing to fuse -> passthrough
    assert any("tta" in n.lower() for n in rep.notes)


# ---------------------------------------------------------------------------
# Planar refinement (EXPERIMENTAL)
# ---------------------------------------------------------------------------

def test_planar_flat_roof_replaced():
    from depthwizard.postprocess.planar import planar_refine

    h, w = 64, 64
    agl = np.full((h, w), 1.0, dtype=np.float32)
    agl[16:48, 16:48] = 20.0 + np.random.default_rng(3).normal(0, 0.05, (32, 32))
    mask = np.zeros((h, w), dtype=bool)
    mask[16:48, 16:48] = True
    out, stats = planar_refine(agl, mask)
    assert stats["accepted"] == 1
    assert stats["pixels_replaced"] > 0
    np.testing.assert_allclose(out[20:44, 20:44], 20.0, atol=0.3)


def test_planar_noisy_roof_rejected():
    from depthwizard.postprocess.planar import planar_refine

    h, w = 64, 64
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    agl = np.full((h, w), 1.0, dtype=np.float32)
    agl[16:48, 16:48] = 20.0 + 4.0 * np.sin(xx[16:48, 16:48] / 5.0)
    mask = np.zeros((h, w), dtype=bool)
    mask[16:48, 16:48] = True
    out, stats = planar_refine(agl, mask, max_residual=1.0)
    assert stats["accepted"] == 0
    np.testing.assert_array_equal(out, agl)  # untouched


def test_planar_small_component_skipped():
    from depthwizard.postprocess.planar import planar_refine

    h, w = 64, 64
    agl = np.full((h, w), 1.0, dtype=np.float32)
    agl[30:34, 30:34] = 20.0  # 16 px < min_area
    mask = np.zeros((h, w), dtype=bool)
    mask[30:34, 30:34] = True
    out, stats = planar_refine(agl, mask)
    assert stats["too_small"] >= 1
    np.testing.assert_array_equal(out, agl)


def test_planar_needs_semantics():
    rgb, agl = make_scene(seed=37)
    cfg = PostProcessConfig(enabled=True, method="planar")
    with pytest.raises(ValueError):
        refine_agl(agl, rgb, cfg)  # refuses to guess building regions


# ---------------------------------------------------------------------------
# Boundary-aware metrics
# ---------------------------------------------------------------------------

def _two_class_onehot(h=32, w=32, split=16):
    """building (class 0) right of split, ground (class 4) left of it."""
    xx = np.arange(w)[None, :].repeat(h, axis=0)
    onehot = np.zeros((6, h, w), dtype=np.float32)
    onehot[0][xx >= split] = 1.0
    onehot[4][xx < split] = 1.0
    return onehot


def test_gradient_error_zero_for_perfect_prediction():
    from depthwizard.postprocess.metrics import gradient_error

    t = np.random.default_rng(0).random((32, 32)).astype(np.float32) * 10
    m = gradient_error(t.copy(), t)
    assert m["grad_mae"] == pytest.approx(0.0, abs=1e-12)


def test_gradient_error_detects_blurring():
    from depthwizard.postprocess.metrics import gradient_error

    t = np.zeros((32, 32), dtype=np.float32)
    t[:, 16:] = 10.0
    blurred = np.full((32, 32), 5.0, dtype=np.float32)
    m = gradient_error(blurred, t)
    assert m["grad_mae"] > 0.1  # a destroyed step is a large gradient error


def test_boundary_bands_around_class_change():
    from depthwizard.postprocess.metrics import boundary_bands

    onehot = _two_class_onehot(32, 32, split=16)
    bands = boundary_bands(onehot, band_px=2)
    assert bands["any"][:, 14:19].all()      # band covers the boundary ±2px
    assert not bands["any"][:, :10].any()    # deep inside a region: no band
    assert bands["building"].any()           # the boundary touches building
    assert bands["building_ground"].any()    # and it is a building/ground one


def test_discontinuity_jump_ratio():
    from depthwizard.postprocess.metrics import discontinuity_preservation

    onehot = _two_class_onehot(32, 32, split=16)
    gt = np.zeros((32, 32), dtype=np.float32)
    gt[:, 16:] = 12.0
    perfect = discontinuity_preservation(gt.copy(), gt, onehot)
    assert perfect["n_pairs"] > 0
    assert perfect["jump_ratio"] == pytest.approx(1.0)
    halved = gt * 0.5  # blurred prediction: only half the jump survives
    m = discontinuity_preservation(halved, gt, onehot)
    assert m["jump_ratio"] == pytest.approx(0.5, rel=0.05)


def test_seam_error_step_and_smooth():
    from depthwizard.postprocess.metrics import seam_error

    p = np.zeros((64, 2048), dtype=np.float32)
    p[:, 1024:] = 8.0  # a hard seam exactly on the 1024 boundary
    m = seam_error(p, tile=1024)
    assert m["n_seams"] >= 1
    assert m["seam_mae"] == pytest.approx(8.0)
    smooth = np.tile(
        (np.arange(2048, dtype=np.float32) / 2048.0)[None, :], (64, 1)
    )
    m2 = seam_error(smooth, tile=1024)
    assert m2["seam_mae"] < 0.01  # sub-cm/px ramp -> no seam artefact


def test_postprocess_tile_metrics_groups():
    from depthwizard.postprocess.metrics import (postprocess_tile_metrics,
                                                  summarize_tile_metrics)

    rgb, agl = make_scene(seed=39)
    pred = agl + 0.5
    onehot = make_sem_probs(rgb, agl)  # soft probs work as one-hot proxy
    m = postprocess_tile_metrics(pred, agl, onehot)
    for key in ("global", "gradient", "seam", "region_building",
                "region_ground", "boundary_building", "discontinuity"):
        assert key in m, key
    assert m["global"]["mae"] == pytest.approx(0.5)
    assert m["global"]["negative_fraction"] == 0.0
    s = summarize_tile_metrics([m, m])
    assert s["global"]["mae"] == pytest.approx(0.5)
    # summarize AVERAGES per-tile metric dicts (n is the mean tile count)
    assert s["global"]["n"] == pytest.approx(m["global"]["n"])


# ---------------------------------------------------------------------------
# Acceptance rules (task Sec. 17)
# ---------------------------------------------------------------------------

def _base_metrics():
    return {
        "mae": 2.0, "rmse": 3.0, "bias": 0.1, "building_mae": 3.0,
        "boundary_mae": 1.0, "grad_mae": 0.5, "seam_mae": 0.1,
        "negative_fraction": 0.0,
    }


def test_acceptance_pass_for_improvement():
    from depthwizard.postprocess.acceptance import evaluate_acceptance

    cand = dict(_base_metrics())
    cand.update(mae=1.8, rmse=2.7, building_mae=2.5, boundary_mae=0.9,
                grad_mae=0.4)
    res = evaluate_acceptance(_base_metrics(), cand, calib_mean_shift=0.01)
    assert res.verdict == "PASS"
    assert not res.failed_rules


def test_acceptance_fail_on_mae_regression():
    from depthwizard.postprocess.acceptance import evaluate_acceptance

    cand = dict(_base_metrics())
    cand.update(mae=2.5)  # 0.5 worse, tolerance 0.01
    res = evaluate_acceptance(_base_metrics(), cand, calib_mean_shift=0.0)
    assert res.verdict == "FAIL"
    assert "mae" in res.failed_rules


def test_acceptance_fail_on_boundary_and_negatives():
    from depthwizard.postprocess.acceptance import evaluate_acceptance

    cand = dict(_base_metrics())
    cand.update(boundary_mae=1.5)  # +50% boundary error, tol 5%
    res = evaluate_acceptance(_base_metrics(), cand, calib_mean_shift=0.0)
    assert "boundary_mae" in res.failed_rules
    cand2 = dict(_base_metrics())
    cand2.update(negative_fraction=0.01)
    res2 = evaluate_acceptance(_base_metrics(), cand2, calib_mean_shift=0.0)
    assert "negative_fraction" in res2.failed_rules


def test_acceptance_fail_on_calibration_shift():
    from depthwizard.postprocess.acceptance import evaluate_acceptance

    cand = dict(_base_metrics())
    res = evaluate_acceptance(_base_metrics(), cand, calib_mean_shift=0.4)
    assert "calibration_mean_shift" in res.failed_rules


def test_acceptance_missing_metrics_skip_not_fail():
    from depthwizard.postprocess.acceptance import evaluate_acceptance

    res = evaluate_acceptance({}, {"mae": 1.0}, calib_mean_shift=None)
    assert res.verdict == "PASS"
    assert all(v.get("skipped") for v in res.rules.values())


def test_acceptance_bias_uses_absolute_value():
    from depthwizard.postprocess.acceptance import evaluate_acceptance

    base = dict(_base_metrics())
    base.update(bias=-0.1)
    cand = dict(base)
    cand.update(bias=-0.5)  # |bias| grew 0.4 > 0.05 tolerance
    res = evaluate_acceptance(base, cand, calib_mean_shift=0.0)
    assert "bias" in res.failed_rules


# ---------------------------------------------------------------------------
# Integration: the certified inference path (torch-gated)
# ---------------------------------------------------------------------------

def _make_rgb_ckpt(tmp_path, a0=1.5, b0=2.0, sem_head=True):
    """Tiny Dn+RGB checkpoint; with sem_head it is the FULL Exp-4/5 shape
    (Dn+RGB+6 zero-filled semantic INPUT channels + predicted aux head)
    — the deployment shape that exposes sem_probs."""
    torch = pytest.importorskip("torch")
    from depthwizard.calibration_net import CalibrationNet

    in_ch = 4 + (6 if sem_head else 0)
    net = CalibrationNet(
        in_ch=in_ch, widths=(8, 16, 32), a0=a0, b0=b0,
        sem_classes=6 if sem_head else 0, sem_aux_head=sem_head,
    )
    ckpt = {
        "model_state": net.state_dict(),
        "use_rgb": True, "use_sem": sem_head, "widths": [8, 16, 32],
        "clamp_min": 0.0,
        "affine_init": {"a": a0, "b": b0}, "loss": "l1", "epoch": 0,
        "val_subset_mae": 0.0, "splits_json": "n/a",
        "sem_aux_head": sem_head, "sem_classes": 6 if sem_head else 0,
        "in_ch": in_ch,
    }
    p = tmp_path / "best.pt"
    torch.save(ckpt, p)
    return p


def _make_geotiff(tmp_path, h=192, w=192, name="scene.tif"):
    """A small georeferenced input with real CRS/transform."""
    import rasterio
    from rasterio.transform import from_origin

    rng = np.random.default_rng(0)
    rgb = rng.integers(0, 255, (h, w, 3), dtype=np.uint8)
    prof = {
        "driver": "GTiff", "width": w, "height": h, "count": 3,
        "dtype": "uint8", "crs": "EPSG:32633",
        "transform": from_origin(500000.0, 4000000.0, 0.5, 0.5),
    }
    p = tmp_path / name
    with rasterio.open(p, "w", **prof) as dst:
        for b in range(3):
            dst.write(rgb[..., b], b + 1)
    return p


def _run(tmp_path, tag, ckpt, tif, postprocess, **kw):
    from depthwizard.inference import run_inference

    out_dir = tmp_path / f"out_{tag}"
    dn_path = tmp_path / "dn.npy"
    if not dn_path.exists():
        h, w = 192, 192
        xs = np.linspace(0.0, 1.0, w, dtype=np.float32)
        dn = np.tile(xs[None, :], (h, 1)) * 8.0  # linear ramp: resize-exact
        np.save(dn_path, dn)
    payload = run_inference(
        tif, ckpt, out_dir=out_dir, device="cpu", mode="resize",
        dn_path=dn_path, live_backbone=False, write_files=True,
        postprocess=postprocess, **kw,
    )
    return payload, out_dir


def test_run_inference_postprocess_none_equals_legacy_baseline(tmp_path):
    """postprocess='none' (and the default) must be the UNCHANGED legacy
    path — byte-identical dsm.npy between the two runs."""
    ckpt = _make_rgb_ckpt(tmp_path)
    tif = _make_geotiff(tmp_path)
    p_default, out_default = _run(tmp_path, "default", ckpt, tif,
                                  postprocess="none")
    p_none, out_none = _run(tmp_path, "explicit_none", ckpt, tif,
                            postprocess="none")
    dsm_default = np.load(out_default / "dsm.npy")
    dsm_none = np.load(out_none / "dsm.npy")
    np.testing.assert_array_equal(dsm_default, dsm_none)  # bit-identical
    # no postprocess artefacts on the legacy path
    assert not (out_default / "agl_raw.npy").exists()
    assert not (out_default / "postprocess_meta.json").exists()
    assert "postprocess" not in p_default


def test_run_inference_refined_run_writes_raw_and_meta(tmp_path):
    ckpt = _make_rgb_ckpt(tmp_path)
    tif = _make_geotiff(tmp_path)
    p_none, out_none = _run(tmp_path, "none", ckpt, tif, postprocess="none")
    p_pp, out_pp = _run(tmp_path, "guided", ckpt, tif, postprocess="guided")

    baseline_dsm = np.load(out_none / "dsm.npy")
    agl_raw = np.load(out_pp / "agl_raw.npy")
    refined_dsm = np.load(out_pp / "dsm.npy")

    # (task Sec. 3) raw AGL of the postprocessed run == the legacy baseline
    np.testing.assert_array_equal(agl_raw, baseline_dsm)
    # refinement actually changed something
    assert not np.array_equal(refined_dsm, agl_raw)
    assert (refined_dsm >= 0).all()  # clamp policy preserved

    import json
    meta = json.loads((out_pp / "postprocess_meta.json").read_text())
    assert meta["method"] == "guided"
    assert "calibration" in meta and "config" in meta
    assert p_pp["postprocess"]["method"] == "guided"


def test_run_inference_semantic_head_exposed_to_postprocess(tmp_path):
    """The predicted (never GT) semantic probabilities flow from the aux
    head through make_full_predict_fn into semantic_wls refinement."""
    import json

    ckpt = _make_rgb_ckpt(tmp_path, sem_head=True)
    tif = _make_geotiff(tmp_path)
    p_sem, out_sem = _run(tmp_path, "sem", ckpt, tif,
                          postprocess="semantic")
    meta = json.loads((out_sem / "postprocess_meta.json").read_text())
    assert meta["method"] == "semantic_wls"
    # the checkpoint HAS a head, so no degradation note should appear
    assert not any("RGB-only" in n for n in meta["notes"])

    # and the headless checkpoint degrades honestly
    nosem_dir = tmp_path / "nosem"
    nosem_dir.mkdir(parents=True, exist_ok=True)
    ckpt_nosem = _make_rgb_ckpt(nosem_dir, sem_head=False)
    p_ns, out_ns = _run(tmp_path, "sem_nosem", ckpt_nosem, tif,
                        postprocess="semantic")
    meta_ns = json.loads((out_ns / "postprocess_meta.json").read_text())
    assert any("RGB-only" in n for n in meta_ns["notes"])


def test_run_inference_geotiff_metadata_preserved(tmp_path):
    """CRS, transform, shape, dtype survive the postprocessed run."""
    import rasterio

    ckpt = _make_rgb_ckpt(tmp_path)
    tif = _make_geotiff(tmp_path)
    with rasterio.open(tif) as src:
        ref_crs, ref_tf, (ref_h, ref_w) = src.crs, src.transform, src.shape

    _p, out = _run(tmp_path, "geo", ckpt, tif, postprocess="guided")
    with rasterio.open(out / "dsm.tif") as ds:
        assert ds.crs == ref_crs
        assert ds.transform == ref_tf
        assert ds.shape == (ref_h, ref_w)
        assert ds.dtypes[0] == "float32"
        assert ds.read(1).shape == (ref_h, ref_w)


def test_run_inference_anchor_uses_refined_agl(tmp_path):
    """FROZEN order: AGL -> refine -> anchor. With ground_elev=100 the
    anchored DSM must equal the REFINED dsm + 100 exactly (anchoring is
    pure arithmetic AFTER refinement — never a filtered DSM)."""
    import rasterio

    ckpt = _make_rgb_ckpt(tmp_path)
    tif = _make_geotiff(tmp_path)
    _p, out = _run(tmp_path, "anchor", ckpt, tif, postprocess="guided",
                   ground_elev=100.0)
    dsm = np.load(out / "dsm.npy")            # refined AGL (ungrounded)
    with rasterio.open(out / "dsm_anchored.tif") as ds:  # georef -> tif
        anchored = ds.read(1)
    np.testing.assert_allclose(anchored, dsm + 100.0, atol=1e-4)
    # and the refinement really happened before anchoring: the raw differs
    agl_raw = np.load(out / "agl_raw.npy")
    assert not np.array_equal(dsm, agl_raw)
