"""Post-processing orchestrator — the ONE entry point everything calls.

Pipeline (fixed order, each stage optional/configurable):

    AGL_raw (metric metres, calibrated by CalibrationNet — NEVER rescaled)
      1. validity mask          finite AGL + finite RGB
      2. spike removal          conservative median/MAD isolated outliers
      3. optional TTA fusion    caller-supplied ensemble replaces the signal
      4. core refinement        guided | bilateral | wls | conf_wls |
                                semantic_wls | median | planar | none
      5. calibration guard      report mean/median/std shift raw->refined;
                                optional exact mean restoration
      6. physical guard         clamp to [clamp_min, inf) — the SAME clamp
                                policy the CalibrationNet itself applies

    returns (agl_refined, PostProcessReport)

Downstream (inference.py — unchanged order):
    DSM = agl_refined + DEM        [ANCHORED (not learned)]

The function is PURE: arrays in, arrays out, no file IO, no global state,
deterministic for a given config. NaN pixels stay NaN (the anchoring and
export layers handle them the same way they handle raw NaNs).

Confidence semantics: ``confidence`` (when supplied) is a RELATIVE weight
map in [0,1] built by confidence.estimate_confidence (or caller-supplied
TTA agreement). It is NOT a calibrated probability and is only used to
modulate the WLS data term (see confidence.py for the exact recipe).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional

import numpy as np

from .config import PostProcessConfig
from .guided import guided_filter, rgb_to_luma
from .wls import wls_refine


@dataclass
class PostProcessReport:
    """Everything the caller needs to audit a refinement run."""

    method: str  # effective method ('none' when disabled)
    config: Dict[str, Any]
    elapsed_sec: float
    n_pixels: int
    n_valid: int
    n_changed: int  # |refined - raw| > 1e-4 among valid pixels
    calibration: Dict[str, float]  # mean/median/std raw vs refined + deltas
    spike_count: int = 0
    planar_stats: Dict[str, Any] = field(default_factory=dict)
    shape_stats: Dict[str, Any] = field(default_factory=dict)
    wls_iterations: int = 0
    wls_converged: bool = True
    tta_augmentations: tuple = ()
    notes: list = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "method": self.method,
            "config": self.config,
            "elapsed_sec": round(self.elapsed_sec, 4),
            "n_pixels": self.n_pixels,
            "n_valid": self.n_valid,
            "n_changed": self.n_changed,
            "calibration": self.calibration,
            "spike_count": self.spike_count,
            "planar_stats": self.planar_stats,
            "shape_stats": self.shape_stats,
            "wls_iterations": self.wls_iterations,
            "wls_converged": self.wls_converged,
            "tta_augmentations": list(self.tta_augmentations),
            "notes": self.notes,
        }


def calibration_report(
    raw: np.ndarray, refined: np.ndarray, valid: np.ndarray
) -> Dict[str, float]:
    """Global-scale audit: mean/median/std of raw vs refined (valid pixels).

    The absolute calibration contract: a compliant refinement moves the
    MEAN by a negligible amount (the data term pins it). This report is
    attached to every run so scale drift is always measurable, never silent.
    """
    r = raw[valid].astype(np.float64)
    f = refined[valid].astype(np.float64)
    return {
        "raw_mean": float(r.mean()),
        "refined_mean": float(f.mean()),
        "raw_median": float(np.median(r)),
        "refined_median": float(np.median(f)),
        "raw_std": float(r.std()),
        "refined_std": float(f.std()),
        "mean_shift": float(f.mean() - r.mean()),
        "median_shift": float(np.median(f) - np.median(r)),
        "std_ratio": float(f.std() / r.std()) if r.std() > 0 else 1.0,
    }


def refine_agl(
    agl: np.ndarray,
    rgb_u8: np.ndarray,
    config: PostProcessConfig,
    sem_probs: Optional[np.ndarray] = None,
    confidence: Optional[np.ndarray] = None,
    tta_predict_fn: Optional[Callable] = None,
    gsd: Optional[tuple] = None,
) -> tuple[np.ndarray, PostProcessReport]:
    """Refine a calibrated AGL tile. See module docstring for the stages.

    agl            [H,W] float32 metres (CalibrationNet output; NaN allowed)
    rgb_u8         [H,W,3] uint8 (the SAME image the model saw)
    sem_probs      [K,H,W] float in [0,1] PREDICTED class probabilities
                   (model auxiliary head or external model — never GT)
    confidence     [H,W] float in [0,1] relative confidence (optional;
                   built here from stability/edge signals when a
                   confidence-aware method is selected but none supplied)
    tta_predict_fn callable(rgb_u8_variant) -> AGL [H,W]; enables the TTA
                   stage when config.tta is True.
    gsd            (dx, dy) ground sample distance in metres/pixel from the
                   input raster's georeference. Drives the physical slope gate
                   in spike detection (never assumes 1 px = 1 m). Falls back
                   to (1, 1) for non-georeferenced input, which is reported.
    """
    t0 = time.perf_counter()
    agl = np.asarray(agl, dtype=np.float32)
    rgb = np.asarray(rgb_u8)
    if rgb.ndim != 3 or rgb.shape[2] < 3:
        raise ValueError(f"rgb_u8 must be [H,W,3], got {rgb.shape}")
    if agl.shape != rgb.shape[:2]:
        raise ValueError(f"agl {agl.shape} != rgb {rgb.shape[:2]} grid")

    spatial_wanted = config.enabled and config.effective_method != "none"
    if not spatial_wanted and not config.tta:
        report = PostProcessReport(
            method="none",
            config=config.to_dict(),
            elapsed_sec=time.perf_counter() - t0,
            n_pixels=agl.size,
            n_valid=int(np.isfinite(agl).sum()),
            n_changed=0,
            calibration={},
            notes=["postprocess disabled — raw AGL passed through unchanged"],
        )
        return agl.copy(), report

    valid = np.isfinite(agl) & np.isfinite(rgb[..., :3]).all(axis=2)

    # Effective ground sample distance for physical slope gating. Never
    # silently pretend 1 px = 1 m: the fallback is explicit and reported.
    if gsd is not None and len(gsd) == 2 and gsd[0] > 0 and gsd[1] > 0:
        gsd_eff = (float(gsd[0]), float(gsd[1]))
    else:
        gsd_eff = (1.0, 1.0)

    raw = agl.copy()
    signal = agl.astype(np.float64)
    notes: list[str] = []
    spike_mask = np.zeros(agl.shape, dtype=bool)
    planar_stats: Dict[str, Any] = {}
    wls_iters, wls_conv = 0, True
    tta_augs: tuple = ()

    # ---- stage 2: conservative spike removal -----------------------------
    if config.spike_removal:
        from .spike_removal import remove_spikes

        signal32, spike_mask = remove_spikes(
            signal.astype(np.float32),
            rgb,
            radius=config.spike_radius,
            tau=config.spike_tau,
            min_isolation=config.spike_min_isolation,
            max_rgb_grad=config.spike_max_rgb_grad,
            min_spike_height=config.spike_min_height,
            max_component_size=config.spike_max_component_size,
            max_local_slope=config.spike_max_local_slope,
            valid=valid,
            gsd=gsd_eff,
        )
        signal = signal32.astype(np.float64)
        notes.append(f"spike_removal: {int(spike_mask.sum())} pixels replaced")

    # ---- stage 3: TTA fusion ----------------------------------------------
    tta_stack: list[np.ndarray] = []
    if config.tta:
        if tta_predict_fn is None:
            notes.append("tta requested but no tta_predict_fn — TTA skipped")
        else:
            from .tta import tta_fuse

            fused = tta_fuse(
                tta_predict_fn,
                rgb,
                aggregation=config.tta_aggregation,
            )
            # BOTH aggregations replace the signal (bug fix: previously only
            # 'median' did, silently ignoring a 'mean' TTA fusion). NaN keeps
            # the input's invalid pattern exactly.
            ok = np.isfinite(fused["agl"]) & valid
            signal = np.where(ok, fused["agl"], np.nan).astype(np.float64)
            # non-identity aligned predictions feed the confidence estimator
            # (TTA disagreement is the strongest deployment-available
            # instability signal — see confidence.py signal 3)
            tta_stack = list(fused["stack"])[1:]
            notes.append(
                f"tta: fused {fused['augmentations']} ({config.tta_aggregation})"
            )
            tta_augs = tuple(fused["augmentations"])

    # ---- stage 4: core refinement ------------------------------------------
    method = config.effective_method
    luma = rgb_to_luma(rgb)

    if not spatial_wanted:
        # TTA-only configuration: fused signal IS the output
        out = signal
    elif method == "median":
        from .spike_removal import _shifted_med_mad

        med, _mad = _shifted_med_mad(signal, 1, valid)
        out = np.where(valid, np.nan_to_num(med, nan=0.0), np.nan)

    elif method == "guided":
        out = guided_filter(
            signal.astype(np.float32), luma,
            radius=config.guided_radius, eps=config.guided_eps, valid=valid,
        ).astype(np.float64)

    elif method == "bilateral":
        from .bilateral import joint_bilateral_filter

        out = joint_bilateral_filter(
            signal.astype(np.float32), rgb,
            radius=config.bilateral_radius,
            sigma_color=config.bilateral_sigma_color,
            sigma_space=config.bilateral_sigma_space,
            valid=valid,
        ).astype(np.float64)

    elif method in ("wls", "conf_wls", "semantic_wls", "full"):
        # "full" = semantic + confidence + WLS core (+ planar below when
        # config.planar) — the composed V8 pipeline of the ablation ladder.
        conf = None
        wants_conf = method in ("conf_wls", "full") or (
            method == "semantic_wls" and confidence is not None
        )
        if wants_conf:
            if confidence is not None and config.confidence_weight > 0:
                conf = np.clip(
                    np.asarray(confidence, dtype=np.float64), 0.0, 1.0
                ) ** config.confidence_weight
                conf = np.maximum(conf, config.confidence_floor)
                conf = np.where(valid, conf, 0.0)
                notes.append(
                    "confidence: caller-supplied relative map (NOT calibrated "
                    "probabilities) modulating the WLS data term"
                )
            elif method in ("conf_wls", "full"):
                from .confidence import estimate_confidence

                conf_map = estimate_confidence(
                    signal.astype(np.float32), rgb, valid=valid,
                    tta_stack=tta_stack or None,
                )
                conf = (
                    np.clip(conf_map.astype(np.float64), 0.0, 1.0)
                    ** config.confidence_weight
                )
                conf = np.maximum(conf, config.confidence_floor)
                conf = np.where(valid, conf, 0.0)
                notes.append(
                    "confidence: constructed from local stability + edge "
                    "consistency (relative weights, NOT calibrated "
                    "probabilities) — see depthwizard/postprocess/confidence.py"
                )

        sem = sem_probs if method in ("semantic_wls", "full") else None
        if method in ("semantic_wls", "full") and sem is None:
            notes.append(
                f"{method} selected but no sem_probs supplied — running "
                "RGB-only WLS (honest degradation, semantics NOT fabricated)"
            )

        # guided warm start: cheap approximation halves CG iterations
        warm = guided_filter(
            signal.astype(np.float32), luma,
            radius=config.guided_radius, eps=config.guided_eps, valid=valid,
        )
        res = wls_refine(
            signal.astype(np.float32),
            rgb,
            lambda_=config.wls_lambda,
            sigma_rgb=config.wls_sigma_rgb,
            confidence=conf,
            sem_probs=sem,
            semantic_edge_weight=(
                config.semantic_edge_weight
                if method in ("semantic_wls", "full") else 0.0
            ),
            semantic_hard_boundary=config.semantic_hard_boundary,
            tol=config.wls_tol,
            max_iter=config.wls_max_iter,
            x0=warm,
            valid=valid,
        )
        out = res.z.astype(np.float64)
        wls_iters, wls_conv = res.iterations, res.converged

        # optional EXPERIMENTAL planar stage rides on top of "full" (and on
        # plain "planar" below) — predicted building regions only, never GT.
        if method == "full" and config.planar:
            if sem_probs is not None:
                from .planar import planar_refine
                from .semantic import building_mask_from_probs

                bmask = building_mask_from_probs(sem_probs)
                out, planar_stats = planar_refine(
                    out.astype(np.float32), bmask,
                    min_area=config.planar_min_area,
                    max_residual=config.planar_max_residual,
                    min_inlier_frac=config.planar_min_inlier_frac,
                    inlier_tol=config.planar_inlier_tol,
                    valid=valid,
                )
                out = out.astype(np.float64)
                notes.append(
                    "planar: EXPERIMENTAL robust plane replacement applied "
                    "(inside predicted building components only)"
                )
            else:
                notes.append(
                    "planar requested within 'full' but no sem_probs — "
                    "planar stage skipped (no fabricated building regions)"
                )

    elif method == "planar":
        if sem_probs is None:
            raise ValueError(
                "planar refinement needs sem_probs (predicted building "
                "probabilities) — refusing to guess building regions"
            )
        from .planar import planar_refine
        from .semantic import building_mask_from_probs

        bmask = building_mask_from_probs(sem_probs)
        out, planar_stats = planar_refine(
            signal.astype(np.float32), bmask,
            min_area=config.planar_min_area,
            max_residual=config.planar_max_residual,
            min_inlier_frac=config.planar_min_inlier_frac,
            inlier_tol=config.planar_inlier_tol,
            valid=valid,
        )
        out = out.astype(np.float64)
        notes.append("planar: EXPERIMENTAL robust plane replacement applied")

    else:  # pragma: no cover — config validates the method name
        raise ValueError(f"unhandled method '{method}'")

    # ---- stage 4.6: semantic structural shaping (rendering quality) --------
    # Predicted-building plateaus + predicted-vegetation canopy domes.
    # Skipped with an explicit note when no semantics are available — the
    # stage never guesses regions.
    shape_stats: Dict[str, Any] = {}
    if config.semantic_shape and spatial_wanted:
        if sem_probs is not None:
            from .semantic import building_mask_from_probs
            from .shaping import flatten_buildings, shape_tree_canopies, vegetation_mask_from_probs

            bmask = building_mask_from_probs(sem_probs)
            out, b_stats = flatten_buildings(
                out.astype(np.float32), bmask, valid=valid,
                min_area=config.shape_min_building_area,
                max_residual=config.shape_building_max_residual,
                min_inlier_frac=config.shape_building_min_inlier_frac,
            )
            vmask = vegetation_mask_from_probs(sem_probs)
            out, v_stats = shape_tree_canopies(
                out.astype(np.float32), vmask, valid=valid,
                min_area=config.shape_min_veg_area,
                min_tree_height=config.shape_min_tree_height,
                peak_window=config.shape_peak_window,
                peak_percentile=config.shape_peak_percentile,
                max_crowns=config.shape_max_crowns,
            )
            shape_stats = {**b_stats, **v_stats}
            n_shaped = (
                b_stats["building_pixels_replaced"]
                + v_stats["veg_pixels_replaced"]
            )
            if n_shaped > 0:
                notes.append(
                    "semantic_shape: buildings flattened to plateau planes "
                    f"({b_stats['building_flattened']}/"
                    f"{b_stats['building_components']} footprints), "
                    f"vegetation domed to canopy shapes "
                    f"({v_stats['veg_crowns_domed']}/{v_stats['veg_crowns']} "
                    "crowns)"
                )
            else:
                notes.append(
                    "semantic_shape: no region passed the shape guards — "
                    "heights left as refined"
                )
        else:
            notes.append(
                "semantic_shape enabled but no sem_probs — skipped (no "
                "fabricated building/vegetation regions)"
            )

    # ---- stage 4.7: post-refinement spike cleanup --------------------------
    if config.spike_removal and config.spike_post_refine and spatial_wanted:
        from .spike_removal import remove_spikes

        out_cleaned, post_spike_mask = remove_spikes(
            out.astype(np.float32),
            rgb,
            radius=config.spike_radius,
            tau=config.spike_tau,
            min_isolation=config.spike_min_isolation,
            max_rgb_grad=config.spike_max_rgb_grad,
            min_spike_height=config.spike_min_height,
            max_component_size=config.spike_max_component_size,
            max_local_slope=config.spike_max_local_slope,
            valid=valid,
            gsd=gsd_eff,
        )
        n_post = int(post_spike_mask.sum())
        if n_post > 0:
            out = np.where(valid, out_cleaned, np.nan).astype(np.float64)
            spike_mask |= post_spike_mask
            notes.append(f"post_spike_cleanup: {n_post} residual spikes repaired")

    # ---- stage 5: calibration guard ----------------------------------------
    refined = np.where(valid, out, np.nan).astype(np.float32)
    calib = calibration_report(raw, refined, valid)
    if config.preserve_mean and valid.any():
        # mean_shift = refined_mean - raw_mean, so restoring the raw mean
        # means SUBTRACTING it (a historic sign bug here doubled the shift
        # instead of cancelling it — pinned by unit test).
        shift = calib["mean_shift"]
        refined[valid] = (refined[valid] - shift).astype(np.float32)
        calib = calibration_report(raw, refined, valid)
        notes.append(f"preserve_mean: restored global mean (shift {shift:+.4f} m)")

    # ---- stage 6: physical guard (the model's own clamp policy) ------------
    if config.clamp_min is not None:
        refined = np.maximum(refined, np.float32(config.clamp_min))
    refined = np.where(valid, refined, np.nan).astype(np.float32)

    changed = valid & (np.abs(refined.astype(np.float64) - raw.astype(np.float64)) > 1e-4)
    report = PostProcessReport(
        method=method,
        config=config.to_dict(),
        elapsed_sec=time.perf_counter() - t0,
        n_pixels=agl.size,
        n_valid=int(valid.sum()),
        n_changed=int(changed.sum()),
        calibration=calib,
        spike_count=int(spike_mask.sum()),
        planar_stats=planar_stats,
        shape_stats=shape_stats,
        wls_iterations=wls_iters,
        wls_converged=wls_conv,
        tta_augmentations=tta_augs,
        notes=notes
        + (
            [f"gsd: {gsd_eff[0]:.3g} x {gsd_eff[1]:.3g} m/px (georeferenced)"]
            if (gsd is not None and len(gsd) == 2 and gsd[0] > 0 and gsd[1] > 0)
            else ["gsd: not georeferenced — slope gate used 1 px = 1 m fallback"]
        ),
    )
    return refined, report
