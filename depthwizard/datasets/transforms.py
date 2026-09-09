"""Joint spatial transforms shared by EVERY dataset adapter.

SINGLE SOURCE OF TRUTH for random-crop / flip / rot90 semantics (moved from
``depthwizard.dataset`` in the GAMUS integration, Phase 1 — the code is
byte-identical; ``DFC2019Dataset`` now delegates here so DFC and GAMUS can
never drift apart on spatial semantics).

Contract (frozen by model_tests/test_dataset.py and
model_tests/test_datasets_interface.py):
    * ONE random window / ONE transform choice is sampled and applied to ALL
      layers — there is no code path where rgb and agl can drift apart,
      including the dem / sem_onehot / sem_ignore layers added later.
    * ``layers`` is a dict of arrays sharing the same leading [H, W] grid
      (rgb is [H, W, 3]; every other layer is [H, W] or [K, H, W]).
    * ``layers["rgb"]`` must exist — it is the reference grid.

Import-cycle note: ``depthwizard.dataset`` imports this module LAZILY (inside
methods) because ``depthwizard.datasets`` package init imports modules that
import ``depthwizard.dataset`` back. Keep this module free of any
``depthwizard.dataset`` import.
"""

from __future__ import annotations

import random
from typing import Dict, Tuple

import numpy as np


def joint_crop(
    layers: Dict[str, np.ndarray], rng: random.Random, crop_size: int | None
) -> Tuple[int, int]:
    """One shared window for every layer, or identity when crop is None.

    Returns (y0, x0) of the applied window (0, 0 when no crop was taken).
    Behavior preserved exactly from the pre-GAMUS DFC2019Dataset (including
    the no-op slice when the tile is smaller than the requested crop).
    """
    c = crop_size
    ref = layers["rgb"]
    h, w = ref.shape[:2]
    if c is None or (h <= c and w <= c):
        for name in layers:
            layers[name] = layers[name][: c or h, : c or w] if c else layers[name]
        return 0, 0
    y0 = rng.randint(0, h - c)
    x0 = rng.randint(0, w - c)
    for name in layers:  # <- single window, all layers
        layers[name] = layers[name][y0 : y0 + c, x0 : x0 + c]
    return y0, x0


def joint_flip_rot(
    layers: Dict[str, np.ndarray], rng: random.Random
) -> Tuple[int, bool, bool]:
    """One shared (rot90 k, hflip, vflip) for every layer.

    Returns (k, do_h, do_v). Applied only when the caller enables augment.
    """
    k = rng.randint(0, 3)
    do_h = rng.random() < 0.5
    do_v = rng.random() < 0.5
    for name, arr in layers.items():
        a = np.rot90(arr, k=k)
        if do_h:
            a = np.flip(a, axis=1)
        if do_v:
            a = np.flip(a, axis=0)
        layers[name] = np.ascontiguousarray(a)
    return k, do_h, do_v
