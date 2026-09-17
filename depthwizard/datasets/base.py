"""BaseDepthDataset — the shared multi-dataset sample pipeline.

Every NEW adapter (GAMUS, mixed, future datasets) subclasses this class and
implements ONLY ``_load_arrays`` (data-source specific I/O). The shared
pipeline below then guarantees, for every dataset, the SAME:

    * joint crop / flip / rot90 (ONE window, ONE transform, ALL layers) via
      depthwizard.datasets.transforms — the same functions the frozen
      DFC2019Dataset delegates to;
    * tensor conventions identical to the frozen DFC2019 contract
      (ImageNet-normalized rgb, clean_agl, raw int64 cls, [1,H,W] dn/dem);
    * semantic layer production: ONE-HOT over PROJECT_CLASSES (6 channels)
      + explicit sem_ignore mask, computed from RAW ids through the VERIFIED
      legend (depthwizard.datasets.semantics) — raw ids stay in ``cls``/meta.

Sample contract (superset of, and byte-compatible with, the frozen DFC2019
contract in depthwizard/dataset.py — keys ALWAYS present, values None when
the layer is absent so torch default_collate stays consistent):

    rgb         [3,H,W]  float32   ImageNet-normalized
    agl         [1,H,W]  float32   clean_agl (>= clamp_min), metres
    cls         [1,H,W]  int64     RAW dataset class ids (legend in meta)
    dn          [1,H,W]  float32 | None   min-max normalized relative depth
    dem         [1,H,W]  float32 | None   DEM prior (real or SYNTHETIC tag)
    sem_onehot  [K,H,W]  float32 | None  K = 6 project classes (one-hot)
    sem_ignore  [1,H,W]  bool     | None  explicit semantic ignore mask
    meta        dict     {stem, h, w, y0, x0, rot90, flip_h, flip_v,
                         dem_tag, dataset, sample_id, sem_legend, ...}

GAMUS-specific honesty fields (set by the GAMUS adapter via ``_load_arrays``
meta): ``height_semantics`` ("nDSM/AGL semantics; units undocumented —
assumed metres") and ``georef`` ("none — non-georeferenced tiles").
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any, ClassVar, Dict, List, Optional

import numpy as np
import torch
from torch.utils.data import Dataset

from .semantics import semantic_layers
from .transforms import joint_crop, joint_flip_rot
from ..dataset import IMAGENET_MEAN, IMAGENET_STD  # same constants
from ..normalize import clean_agl


@dataclass
class AdapterConfig:
    """Shared config for BaseDepthDataset adapters.

    DFC2019 keeps its own frozen ``DFC2019Config``; adapters duck-type on
    these fields (crop_size / augment / clamp_agl_min / load_depth /
    load_semantics / seed / _rng). Dataset-source fields live on the
    adapter's own config subclass (see gamus.GAMUSConfig).
    """

    crop_size: Optional[int] = None  # None = full tile
    augment: bool = False
    clamp_agl_min: float = 0.0
    load_depth: bool = True  # False -> dn None (pre-cache runs)
    load_semantics: bool = True  # False -> no sem layers
    seed: int = 42
    _rng: Optional[random.Random] = field(default=None, repr=False, compare=False)

    def __post_init__(self):
        self._rng = random.Random(self.seed)


def collate_dict_none_safe(batch):
    """default_collate + None pass-through + heterogeneous-meta tolerance.

    Three historical behaviors restored / added (torch >= 2.13 removed the
    None branch; mixed datasets made meta heterogeneous):

    1. all-None values under a key -> None (torch < 2.13 behavior; the
       optional layers dn/dem/sem_onehot/sem_ignore use it);
    2. ``meta`` -> the LIST of per-sample dicts, NOT collated: meta is
       bookkeeping whose keys legitimately differ per dataset (GAMUS carries
       height_semantics/units/split; DFC carries dem_tag...) — tensor
       collation of bookkeeping is meaningless and crashes;
    3. a key present in only SOME samples -> None for the missing ones;
       all-None -> None (mixed datasets whose sources produce different
       optional layers).

    A MIX of None and tensors (same key, same dataset) still raises — an
    inconsistent dataset is a bug, not something to paper over. Results are
    identical to old-torch default_collate wherever the old behavior was
    defined.
    """
    from torch.utils.data import default_collate

    first = batch[0]
    if not isinstance(first, dict):
        return default_collate(batch)
    out = {}
    keys: list = []
    for b in batch:
        for k in b:
            if k not in keys:
                keys.append(k)
    for k in keys:
        if k == "meta":
            out[k] = [b.get("meta") for b in batch]  # bookkeeping list
            continue
        vals = [b.get(k) for b in batch]
        if all(v is None for v in vals):
            out[k] = None
        else:
            out[k] = collate_dict_none_safe(vals)
    return out


class BaseDepthDataset(Dataset):
    """Shared __getitem__ pipeline; subclasses implement ``_load_arrays``.

    Subclass contract for ``_load_arrays(idx)`` (all returns numpy):
        rgb  uint8  [H,W,3]     required  (band order as stored, R,G,B)
        agl  float32 [H,W]      required  raw values incl. small negatives
        cls  int-ish [H,W]      required  RAW dataset ids, unmodified
        dn   float32 [H,W]      optional  ALREADY min-max normalized
                                        (via depthwizard.normalize.minmax_
                                         normalize — the single source)
        dem  float32 [H,W]      optional  DEM prior on the tile grid
        meta dict               required  MUST contain sample_id (+ stem);
                                        may carry source-specific honesty
                                        fields (height_semantics, split, ...)
        dem_tag str | None      optional  "dem:<file>" | "SYNTHETIC-DEM-PROXY"
    Grid mismatches between layers MUST raise in the adapter (never
    broadcast) — the same rule as geo.read_tile.
    """

    dataset_name: ClassVar[str] = "base"
    sem_legend: ClassVar[Optional[str]] = None  # semantics.py dataset key

    def __init__(self, samples: List[Any], cfg: AdapterConfig):
        if len(samples) == 0:
            raise ValueError(
                "Empty sample list — check discovery/manifest "
                "before constructing a dataset."
            )
        self.samples = samples
        self.cfg = cfg

    def __len__(self) -> int:
        return len(self.samples)

    # ------------------------------------------------------------------
    # subclass hooks
    # ------------------------------------------------------------------
    def _load_arrays(self, idx: int) -> Dict[str, Any]:
        raise NotImplementedError

    # ------------------------------------------------------------------
    # shared pipeline
    # ------------------------------------------------------------------
    def __getitem__(self, idx: int) -> Dict:
        arrs = self._load_arrays(idx)
        rgb, agl, cls = arrs["rgb"], arrs["agl"], arrs["cls"]
        meta_in = arrs.get("meta", {}) or {}

        # The joint transform operates ONLY on [H,W] and [H,W,3] layers —
        # the exact surface the frozen DFC pipeline was written for. The
        # ONE-HOT semantic layers are computed AFTER the transform from the
        # already-transformed cls: per-pixel class mapping COMMUTES with
        # crop/flip/rot90, so this is exactly equivalent to transforming the
        # one-hot itself (pinned by test_joint_transform_alignment_sem) —
        # and it avoids np.rot90 axis ambiguity on channel-first arrays.
        layers: Dict[str, np.ndarray] = {"rgb": rgb, "agl": agl, "cls": cls}
        for k in ("dn", "dem", "confidence"):
            if arrs.get(k) is not None:

                layers[k] = arrs[k]

        y0 = x0 = 0
        k_, do_h, do_v = 0, False, False
        rng = self.cfg._rng
        if rng is None:
            rng = random.Random(self.cfg.seed)
        if self.cfg.crop_size is not None:
            y0, x0 = joint_crop(layers, rng, self.cfg.crop_size)
        if self.cfg.augment:
            k_, do_h, do_v = joint_flip_rot(layers, rng)

        # ---- tensorize (identical conventions to the frozen DFC path) ----
        rgb_t = (
            torch.from_numpy(
                ((layers["rgb"].astype(np.float32) / 255.0) - IMAGENET_MEAN)
                / IMAGENET_STD
            )
            .permute(2, 0, 1)
            .contiguous()
        )  # [3,H,W]
        agl_t = torch.from_numpy(clean_agl(layers["agl"], self.cfg.clamp_agl_min))[
            None, ...
        ]  # [1,H,W]
        cls_t = torch.from_numpy(np.ascontiguousarray(layers["cls"]).astype(np.int64))[
            None, ...
        ]  # [1,H,W] raw ids
        dn_t = (
            torch.from_numpy(np.ascontiguousarray(layers["dn"]).astype(np.float32))[
                None, ...
            ]
            if "dn" in layers
            else None
        )
        dem_t = (
            torch.from_numpy(np.ascontiguousarray(layers["dem"]).astype(np.float32))[
                None, ...
            ]
            if "dem" in layers
            else None
        )

        # ---- semantic layers from the TRANSFORMED raw cls ----
        sem_t = ign_t = None
        if self.cfg.load_semantics and self.sem_legend is not None:
            onehot, ignore, unmapped = semantic_layers(
                cls_t[0].numpy(), self.sem_legend
            )
            sem_t = torch.from_numpy(onehot)  # [K,H,W] f32
            ign_t = torch.from_numpy(np.ascontiguousarray(ignore))[
                None, ...
            ]  # [1,H,W] bool
            meta_in["sem_legend"] = self.sem_legend
            if unmapped:
                meta_in["sem_unmapped_ids"] = unmapped

        sid = meta_in.get("sample_id") or meta_in.get("stem")
        meta = dict(meta_in)  # adapter extras
        meta.update(
            {
                "stem": sid,  # DFC-compat key
                "h": agl_t.shape[1],
                "w": agl_t.shape[2],
                "y0": y0,
                "x0": x0,
                "rot90": k_,
                "flip_h": do_h,
                "flip_v": do_v,
                "dem_tag": arrs.get("dem_tag"),
                "dataset": self.dataset_name,
                "sample_id": sid,
            }
        )
        return {
            "rgb": rgb_t,
            "agl": agl_t,
            "cls": cls_t,
            "dn": dn_t,
            "dem": dem_t,
            "sem_onehot": sem_t,
            "sem_ignore": ign_t,
            "meta": meta,
        }
