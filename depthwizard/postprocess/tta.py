"""Test-time augmentation: flip/rotate ensemble with robust fusion.

Augmentations (configurable): identity, horizontal flip, vertical flip,
180-degree rotation. For each variant the FULL prediction path runs on the
transformed RGB (the caller supplies ``predict_agl_fn`` — the same
certified path as identity inference, so the backbone + CalibrationNet
both see a coherently transformed image), and the predicted AGL is mapped
back to the original orientation before fusion.

Fusion:
    "median"  per-pixel median across variants (default — robust to one
              bad variant; recommended)
    "mean"    per-pixel mean (slightly smoother; sensitive to outliers)

Returns {'agl', 'spread', 'augmentations', 'stack'}: ``stack`` is the list
of ORIENTATION-ALIGNED per-variant predictions (identity first) — callers
feed the non-identity entries to confidence.estimate_confidence as the
TTA-disagreement signal, and ``spread`` is the per-pixel std over the stack
(doubles as a diagnostic map).

Honesty note: variants that cannot recompute the RAW Dn backbone for the
flipped image (e.g. a cached-Dn-only deployment) may flip the cached Dn
instead — valid only up to the backbone's flip-equivariance error, and the
caller MUST set ``approximate_dn=True`` so the run's metadata records it.
"""

from __future__ import annotations

import numpy as np

AUGMENTATIONS = ("identity", "hflip", "vflip", "rot180")


def _apply(img: np.ndarray, aug: str) -> np.ndarray:
    if aug == "identity":
        return img
    if aug == "hflip":
        return img[:, ::-1]
    if aug == "vflip":
        return img[::-1]
    if aug == "rot180":
        return img[::-1, ::-1]
    raise ValueError(f"unknown augmentation '{aug}'")


def tta_fuse(
    predict_agl_fn,
    rgb_u8: np.ndarray,
    augmentations: tuple[str, ...] = ("hflip", "vflip", "rot180"),
    aggregation: str = "median",
    include_identity: bool = True,
) -> dict:
    """Run the augmented ensemble -> {'agl','spread','augmentations','stack'}.

    predict_agl_fn(rgb_u8_transformed) -> AGL [H,W] float32 on the SAME
    grid (the caller guarantees grid-preserving prediction).
    """
    augs: list[str] = (
        (["identity"] if include_identity else []) + list(augmentations)
    )
    for aug in augs:
        if aug not in AUGMENTATIONS:
            raise ValueError(f"unknown augmentation '{aug}'")

    preds = []
    for aug in augs:
        p = np.asarray(predict_agl_fn(_apply(rgb_u8, aug)), dtype=np.float32)
        preds.append(_apply(p, aug))  # inverse of hflip/vflip/rot180 is itself

    stack = np.stack(preds)
    if aggregation == "median":
        with np.errstate(all="ignore"):
            agl = np.nanmedian(stack, axis=0)
    elif aggregation == "mean":
        agl = np.nanmean(stack, axis=0)
    else:
        raise ValueError(f"unknown aggregation '{aggregation}' (median|mean)")

    with np.errstate(all="ignore"):
        spread = np.nanstd(stack, axis=0)
    spread = np.where(np.isfinite(agl), np.nan_to_num(spread), np.nan)
    return {
        "agl": agl.astype(np.float32),
        "spread": spread.astype(np.float32),
        "augmentations": augs,
        "stack": [p.astype(np.float32) for p in preds],
    }
