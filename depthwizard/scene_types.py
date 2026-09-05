"""Heuristic scene-type classification for tile-level stratification (Phase 5).

Purpose: the evaluation spec requires per-scene-type reporting
(urban / sparse-suburban / forest / hilly-high-relief / flat). GAMUS covers
three DIFFERENT US cities (DC / NYC / PHL), so scene-type breakdown is how
cross-domain behavior becomes visible beyond city-level means.

HONESTY NOTE: these labels are ANALYSIS HEURISTICS, not ground truth. The
thresholds below are documented, deterministic, and derived only from
VERIFIED inputs (project-class shares from the dataset legends + AGL
relief). They exist to stratify evaluation — never as claims about what a
tile "is".

Decision order (first match wins — documented, deterministic):
    1. building_share >= 0.15          -> urban
    2. vegetation_share >= 0.50        -> forest
    3. relief (AGL p95 - p50) >= 15 m  -> hilly-high-relief
    4. building_share >= 0.02          -> sparse-suburban
    5. otherwise                       -> flat

Relief uses p95-p50 (robust to a few tall outliers while still capturing
genuine terrain/building mass variation). Thresholds are round numbers —
tuning them changes only the STRATIFICATION labels, not any underlying
metric; document any change here.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np

SCENE_TYPES: Tuple[str, ...] = ("urban", "sparse-suburban", "forest",
                                "hilly-high-relief", "flat")

# Documented thresholds (see module docstring).
URBAN_BUILDING_SHARE = 0.15
FOREST_VEGETATION_SHARE = 0.50
HILLY_RELIEF_M = 15.0
SPARSE_BUILDING_SHARE = 0.02


def classify_scene(agl: np.ndarray, cls: np.ndarray, dataset: str) -> str:
    """Classify ONE tile from its AGL field + RAW class map.

    agl     [H,W] float — clean AGL (metres)
    cls     [H,W] int   — RAW dataset class ids (mapped through the VERIFIED
                          legend via datasets.semantics)
    dataset "dfc2019" | "gamus"
    """
    from depthwizard.datasets.semantics import (PROJECT_CLASS_TO_INDEX,
                                                semantic_layers)
    onehot, _ignore, _unmapped = semantic_layers(cls, dataset)
    total = onehot.sum() + 0.0
    if total <= 0:
        return "flat"                       # no labeled pixels -> degenerate
    building_share = float(onehot[PROJECT_CLASS_TO_INDEX["building"]].sum()) / total
    vegetation_share = float(onehot[PROJECT_CLASS_TO_INDEX["vegetation"]].sum()) / total
    a = np.asarray(agl, dtype=np.float64)
    valid = np.isfinite(a)
    if valid.sum() > 10:
        p50, p95 = np.percentile(a[valid], [50, 95])
        relief = float(p95 - p50)
    else:
        relief = 0.0

    if building_share >= URBAN_BUILDING_SHARE:
        return "urban"
    if vegetation_share >= FOREST_VEGETATION_SHARE:
        return "forest"
    if relief >= HILLY_RELIEF_M:
        return "hilly-high-relief"
    if building_share >= SPARSE_BUILDING_SHARE:
        return "sparse-suburban"
    return "flat"


def stratify_by_scene_type(per_tile_metrics: List[Dict],
                           scene_types: List[str]) -> Dict[str, Dict[str, float]]:
    """Mean of per-tile metric dicts grouped by scene-type label.

    Returns {scene_type: {metric_name_mean...}} (only types with >= 1 tile;
    NaN-safe per metric via np.nanmean semantics).
    """
    groups: Dict[str, List[Dict]] = defaultdict(list)
    for m, st in zip(per_tile_metrics, scene_types):
        groups[st].append(m)
    out: Dict[str, Dict[str, float]] = {}
    for st, lst in groups.items():
        keys = [k for k in lst[0]
                if isinstance(lst[0][k], (int, float)) and k != "n"]
        out[st] = {f"{k}_mean": float(np.nanmean([d[k] for d in lst]))
                   for k in keys}
        out[st]["n_tiles"] = len(lst)
    return out
