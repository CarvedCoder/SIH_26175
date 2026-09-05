"""DFC2019 adapter — the existing frozen dataset + project semantics.

Thin subclass of ``depthwizard.dataset.DFC2019Dataset`` (which keeps its
frozen, test-pinned behavior untouched). The adapter adds ONLY:

    * ``meta["dataset"] = "dfc2019"`` and ``meta["sample_id"]`` (also added
      to the parent in Phase 1 — additive meta keys, zero behavior change);
    * ONE-HOT semantic layers (``sem_onehot`` [6,H,W] float32 +
      ``sem_ignore`` [1,H,W] bool) computed from the RAW ``cls`` ids through
      the VERIFIED DFC2019 LAS legend (see datasets/semantics.py).

Per-pixel class mapping COMMUTES with the joint crop/flip/rot90 transform
(one-hot of a cropped map == crop of a one-hot map), so computing the
one-hot AFTER the parent's joint transform is EXACTLY equivalent to
computing it inside the loop — no alignment risk, no frozen-path edit.

Keys are ALWAYS present in the returned sample (sem layers None when
``load_semantics=False``) so torch default_collate stays consistent.
"""

from __future__ import annotations

from typing import List

import numpy as np
import torch

from ..dataset import DFC2019Config, DFC2019Dataset
from .semantics import semantic_layers
from ..geo import TilePaths


class DFC2019Adapter(DFC2019Dataset):
    dataset_name = "dfc2019"
    sem_legend = "dfc2019"

    def __init__(self, tiles: List[TilePaths], config: DFC2019Config,
                 load_semantics: bool = True):
        super().__init__(tiles, config)
        self.load_semantics = bool(load_semantics)

    def __getitem__(self, idx: int) -> dict:
        s = super().__getitem__(idx)
        if self.load_semantics:
            cls_np = s["cls"][0].numpy()
            onehot, ignore, unmapped = semantic_layers(cls_np, self.sem_legend)
            s["sem_onehot"] = torch.from_numpy(onehot)       # [6,H,W] f32
            s["sem_ignore"] = torch.from_numpy(
                np.ascontiguousarray(ignore))[None, ...]     # [1,H,W] bool
            s["meta"]["sem_legend"] = self.sem_legend
            if unmapped:
                s["meta"]["sem_unmapped_ids"] = unmapped
        else:
            s["sem_onehot"] = None
            s["sem_ignore"] = None
        return s


def discover_and_split_adapter(config: DFC2019Config, splits_json,
                               load_semantics: bool = True):
    """discover_and_split, but returning DFC2019Adapter datasets.

    Reuses the frozen tile/split logic from depthwizard.dataset (single
    source of truth for DFC discovery) — no duplication.
    """
    from ..dataset import discover_and_split
    out = discover_and_split(config, splits_json,
                             dataset_class=DFC2019Adapter,
                             load_semantics=load_semantics)
    return out
