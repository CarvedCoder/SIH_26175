"""DEM prior utilities for the Method-D calibration variant (Dn+RGB+DEM).

This module is the learned half of Track-2 anchoring. ``anchoring.py``
adds a non-learned ground datum (AGL + DEM -> DSM) at inference time;
this module prepares the DEM that flows into the CalibrationNet's
5-channel input (Dn+RGB+DEM) at training and inference time.

Honesty rules encoded here (frozen by tests in ``tests/test_demprior.py``):

  * REAL DEMs flow through ``resample_dem_to_tile`` (imported verbatim from
    ``anchoring.py`` — single source of truth for the bilinear-reproject,
    CRS-gate, full-coverage-or-raise behavior). We never duplicate that
    logic; any divergence would silently misalign the DEM with the AGL.
  * ``synth_dem_from_agl`` is a TRAINING-TIME ABSTRACTION ONLY. It exists
    because DFC2019 Track-1 training tiles ship WITHOUT a CRS, so real
    SRTM/DTM alignment against them is impossible in the training loop. The
    function synthesises a "ground" prior by aggressively low-passing the
    AGL itself — every output, every worklog line, every consumer label
    MUST tag this fact with the literal string ``SYNTHETIC-DEM-PROXY``.
    Calling this function and presenting the result as if it were real
    SRTM would be a fabrication, not an approximation.
  * The acceptance gate for the DEM-variant challenger is pre-registered
    in worklog Section 3 (Decision log) BEFORE any training is run: the
    challenger must beat val MAE 2.494 / val RMSE 4.365 by >=3% on BOTH
    metrics on the val split before it earns its single test-split shot
    (worklog Section 3, rule 4 — one test shot per model, ever). If only
    SYNTHETIC-DEM-PROXY training data is available, any resulting win is a
    MECHANISM RESULT, not a citable "DEM conditioning works" claim.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

# Re-export the SINGLE source of truth for bilinear DEM->tile reprojection.
# This is the same function ``anchoring.anchor_with_dem`` calls; importing
# it here lets demprior consumers depend on one implementation, not two.
# (See worklog Section 4: "GSD honesty regression" — the rule that there is
# ONE place where CRS-unit-to-metres logic lives applies here too.)
from .anchoring import resample_dem_to_tile  # noqa: F401  (public re-export)

# The literal tag every output line and consumer label must carry when
# the DEM it used was synthesised from AGL rather than read from a real
# DEM file. Surfaced as a module constant so tests can grep for it.
SYNTHETIC_DEM_TAG = "SYNTHETIC-DEM-PROXY"


def synth_dem_from_agl(
    agl: np.ndarray, sigma_m: float = 8.0, *, gsd_m: Optional[float] = None
) -> np.ndarray:
    """Synthesise a "ground" prior from an AGL field, for training-time
    ablation ONLY when real SRTM/DTM alignment is not available.

    The DFC2019 Track-1 training tiles have NO CRS — the GSD honesty
    regression (worklog Section 4) means we cannot honestly align a real
    DEM against them, so the training loop uses this proxy instead. The
    proxy is a heavy Gaussian low-pass of the AGL: buildings (which
    contribute the high-frequency "above ground" energy) are suppressed,
    leaving a smooth approximation of the underlying terrain.

    CAUTION — this is a *training-time proxy*, not a measurement:

      * It is NOT real SRTM, NOT a DTM, NOT a measurement of the ground.
      * Every consumer of a DEM produced by this function MUST tag its
        outputs with the literal ``SYNTHETIC-DEM-PROXY`` (import
        ``SYNTHETIC_DEM_TAG`` and include it in any print/log/payload
        label). See ``tests/test_demprior.py`` for the contract.
      * A val/test win by a model trained on this proxy is a MECHANISM
        result (the architecture can consume a DEM conditioning channel),
        NOT a citable "DEM conditioning works" claim. A real-DEM
        validation run is the separate, data-dependent next step.

    Args:
        agl:    [H, W] float32, metres above ground level (raw or
                clean_agl — both work; we low-pass either way).
        sigma_m: Gaussian sigma in METRES of ground distance, not pixels.
                Default 8 m matches a typical 0.5-1 m GSD ortho at
                ~10-15 pixel sigma, which suppresses building footprints
                while preserving broad terrain undulation. Smaller sigma
                leaks building energy into the prior; larger flattens
                real terrain features.
        gsd_m:  Ground-sample distance per pixel, in metres. REQUIRED
                when sigma_m is to be converted to a pixel-space kernel
                size. When ``None``, the function assumes a 1 m GSD
                (sigma_m == sigma_px) — this assumption MUST be reported
                by the caller via the SYNTHETIC-DEM-PROXY tag, because
                silently assuming 1 m GSD on a non-georeferenced input
                is the exact bug the worklog Section 4 incident pins.

    Returns:
        float32 [H, W] "ground" prior, same shape as ``agl``. Always finite
        for finite input (NaN inputs propagate — do not feed NaN AGL).
    """
    if agl.ndim != 2:
        raise ValueError(
            f"synth_dem_from_agl expects a 2-D AGL array, got shape {agl.shape}"
        )
    if sigma_m <= 0:
        raise ValueError(f"sigma_m must be > 0, got {sigma_m}")

    # Convert sigma from metres to pixels. If GSD is unknown we fall back
    # to 1 m / px (sigma_m == sigma_px) — the caller MUST tag the output
    # SYNTHETIC-DEM-PROXY so this assumption is never presented as fact.
    sigma_px = float(sigma_m) / float(gsd_m) if gsd_m and gsd_m > 0 else float(sigma_m)

    # Truncate the kernel at 3 sigma (Gaussian is ~99.7% inside). For an
    # 8 m default on a 1 m GSD, that's a 49x49 kernel — cheap enough on
    # CPU, exact on the small synthetic arrays tests use.
    radius = max(1, int(np.ceil(3 * sigma_px)))
    ks = 2 * radius + 1
    if ks > max(agl.shape) * 2:
        # Pathological: the requested sigma is larger than the whole tile.
        # Returning the AGL mean is mathematically the limit of the
        # Gaussian as sigma -> infinity; this keeps the function honest
        # about its heavy-low-pass character rather than silently
        # zero-padding.
        return np.full_like(agl, float(np.nanmean(agl)), dtype=np.float32)

    # Separable 1-D Gaussian (mathematically equivalent to the 2-D kernel,
    # ~K x faster — matters when training tiles are 1024^2).
    coords = np.arange(ks, dtype=np.float64) - radius
    g1d = np.exp(-(coords**2) / (2 * sigma_px**2))
    s = float(g1d.sum())
    if s > 0:
        g1d /= s
    g1d = g1d.astype(np.float32)

    # Convolve along both axes via reflect padding (matches the dataset's
    # joint-transform edge mode, so the proxy is consistent with how AGL
    # itself is cropped/augmented).
    from scipy.ndimage import convolve1d

    out = convolve1d(agl.astype(np.float32), g1d, axis=0, mode="reflect")
    out = convolve1d(out, g1d, axis=1, mode="reflect")
    return out.astype(np.float32)
