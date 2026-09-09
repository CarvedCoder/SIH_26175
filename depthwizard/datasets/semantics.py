"""Semantic class normalization — VERIFIED legends -> project classes.

Project-level classes (6, canonical ONE-HOT order — frozen):
    0 building, 1 vegetation, 2 road, 3 water, 4 ground, 5 other
    (+ an explicit IGNORE mask — ignore is NOT a one-hot channel).

ANTI-FABRICATION CONTRACT (binding):
    * Raw dataset class ids are preserved in the sample's ``cls`` layer and
      in ``meta``; the mapping below is the ONLY place raw ids are assigned
      meanings, and only for ids documented in OFFICIAL sources.
    * A raw id NOT present in the verified legend maps to IGNORE and is
      reported in ``meta["sem_unmapped_ids"]`` — never silently guessed.
    * Do not add or "fix" legends here without a citable source.

VERIFIED SOURCES (per-file, checked 2026-09):
    DFC2019 / US3D Track-1 (one legend for all tracks, LAS codes):
        https://github.com/pubgeo/dfc2019/blob/master/data/README.md
        (confirmed by track1/track1-metrics.py and track1/unets/params.py)
        2=Ground, 5=Trees, 6=Buildings, 9=Water,
        17=Bridge/elevated road, 65=Unlabeled (void).
        Official protocol EXCLUDES 65 from evaluation -> we map it to IGNORE.
    GAMUS (RSI-MMSegmentation README class table; paper arXiv:2305.14914):
        https://github.com/EarthNets/RSI-MMSegmentation
        0=others (background), 1=ground, 2=low vegetation, 3=buildings,
        4=water, 5=road, 6=tree.
        NOTE: GAMUS 0 ("others") is a REAL class -> project class "other".
        GAMUS has no documented void class; nothing maps to IGNORE.

Mapping decisions (documented, arguable, NOT invented):
    DFC 17 "Bridge/elevated road" -> "road"  (semantically an elevated road;
        the closest project class — no separate bridge channel exists).
    GAMUS 2 "low vegetation" and 6 "tree"    -> "vegetation" (merged).
    DFC 65 void                               -> IGNORE (official protocol).
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Project-level class system (the ONE-HOT channel order — frozen)
# ---------------------------------------------------------------------------

PROJECT_CLASSES: Tuple[str, ...] = (
    "building", "vegetation", "road", "water", "ground", "other",
)
NUM_PROJECT_CLASSES: int = len(PROJECT_CLASSES)          # = 6
PROJECT_CLASS_TO_INDEX: Dict[str, int] = {
    name: i for i, name in enumerate(PROJECT_CLASSES)
}

# Sentinel for "not a valid project class" in id maps (int-valued maps only).
IGNORE: int = -1

# ---------------------------------------------------------------------------
# Verified raw legends (source of truth — see module docstring)
# ---------------------------------------------------------------------------

RAW_LEGENDS: Dict[str, Dict[int, str]] = {
    "dfc2019": {
        2: "Ground", 5: "Trees", 6: "Buildings", 9: "Water",
        17: "Bridge/elevated road", 65: "Unlabeled (void)",
    },
    "gamus": {
        0: "others (background)", 1: "ground", 2: "low vegetation",
        3: "buildings", 4: "water", 5: "road", 6: "tree",
    },
}

# raw id -> project class NAME, or None -> IGNORE.
PROJECT_MAPS: Dict[str, Dict[int, Optional[str]]] = {
    "dfc2019": {
        2: "ground", 5: "vegetation", 6: "building", 9: "water",
        17: "road", 65: None,              # void -> IGNORE (official protocol)
    },
    "gamus": {
        0: "other", 1: "ground", 2: "vegetation", 3: "building",
        4: "water", 5: "road", 6: "vegetation",
    },
}


def class_to_project(dataset: str, raw_id: int) -> Optional[str]:
    """Map one raw id to a project class name; None -> IGNORE.

    Raises KeyError for an unknown DATASET (a code bug), but returns None
    (-> IGNORE) for an unknown raw ID within a known dataset — unknown data
    must degrade to ignore + reporting, never crash a training loop and
    never be guessed.
    """
    if dataset not in PROJECT_MAPS:
        raise KeyError(
            f"unknown dataset '{dataset}' — legends exist for: "
            f"{sorted(PROJECT_MAPS)}. Add the VERIFIED legend (with source) "
            "in depthwizard/datasets/semantics.py first; do not guess.")
    return PROJECT_MAPS[dataset].get(int(raw_id))          # None -> IGNORE


def semantic_layers(cls: np.ndarray,
                    dataset: str) -> Tuple[np.ndarray, np.ndarray, List[int]]:
    """Raw class map -> (one-hot [K,H,W] float32, ignore [H,W] bool,
    unmapped_ids list).

    * one-hot channel order = PROJECT_CLASSES (frozen).
    * Ignore pixels (unmapped raw ids or legend-mapped void, e.g. DFC 65)
      have an ALL-ZERO one-hot column AND ignore=True — the mask is the
      explicit authority, the zero-vector is the numeric consequence.
    * ``unmapped_ids`` lists raw ids absent from the verified legend
      (sorted, deduplicated) so callers can surface them in meta — the
      anti-fabrication audit trail.
    """
    if dataset not in PROJECT_MAPS:
        raise KeyError(f"unknown dataset '{dataset}' (see class_to_project)")
    cls = np.asarray(cls)
    h, w = cls.shape[:2]

    lut = np.full((256,), IGNORE, dtype=np.int64)          # ids are uint8-ish
    for raw_id, name in PROJECT_MAPS[dataset].items():
        lut[int(raw_id)] = (PROJECT_CLASS_TO_INDEX[name]
                            if name is not None else IGNORE)

    ids = cls.astype(np.int64)                             # float CLS -> int
    in_range = (ids >= 0) & (ids <= 255)
    proj = np.where(in_range, lut[np.clip(ids, 0, 255)], IGNORE)
    ignore = (proj == IGNORE)                              # includes out-of-range

    known = set(PROJECT_MAPS[dataset])
    unmapped = sorted({int(v) for v in np.unique(ids)
                       if int(v) not in known})

    onehot = np.zeros((NUM_PROJECT_CLASSES, h, w), dtype=np.float32)
    for k in range(NUM_PROJECT_CLASSES):
        onehot[k] = ((proj == k) & ~ignore).astype(np.float32)
    return onehot, ignore, unmapped


def legend_report(dataset: str) -> Dict:
    """Human/machine-readable legend + mapping (for inspect output & docs)."""
    if dataset not in RAW_LEGENDS:
        raise KeyError(f"unknown dataset '{dataset}'")
    return {
        "dataset": dataset,
        "raw_legend": {str(k): v for k, v in RAW_LEGENDS[dataset].items()},
        "project_mapping": {
            str(raw): (name if name is not None else "IGNORE")
            for raw, name in PROJECT_MAPS[dataset].items()
        },
        "project_classes": list(PROJECT_CLASSES),
        "note": "Raw ids are preserved in sample['cls'] and meta; this "
                "mapping is the only place meanings are assigned.",
    }
