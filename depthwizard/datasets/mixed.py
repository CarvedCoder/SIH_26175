"""Mixed-dataset training support (Phase 3) — GATED by design.

USER CONSTRAINT (binding, encoded below): "Do not mix GAMUS and DFC2019
until per-dataset statistics and normalization are verified."
``build_mixed_datasets`` therefore REFUSES to construct a mixed dataset
unless the config points to a stats artifact that

    * was produced by `python model.py stats` (per-dataset AGL/Dn statistics),
    * covers EVERY source dataset in the mixed config,
    * carries "verified": true (a human checked it — the stats command
      writes false and a reviewer flips it after inspecting).

This is a process gate, not a math gate: the code path exists so mixed
training can be enabled by config the moment verification is done, with
zero further code changes and a full audit trail.

Weighting: per-source ``weight`` (default 1.0, i.e. sample-count
proportional). Weights are surfaced in meta and the manifest — they are
EXPLICIT knobs, documented as non-optimized (plan risk R6: dataset scale
imbalance ~5,004 GAMUS vs ~1,400 DFC tiles).
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

from torch.utils.data import Dataset

from ..geo import load_json


class MixedDataset(Dataset):
    """Concatenation of per-dataset adapters with provenance.

    A plain concat (global index -> underlying dataset). For weighted
    sampling, the training loop uses ``.per_sample_weights()`` to build a
    torch WeightedRandomSampler — weights live HERE so the sampling policy
    travels with the data, not scattered in training scripts.
    """

    dataset_name = "mixed"

    def __init__(self, datasets: List[Dataset], weights: List[float]):
        if not datasets:
            raise ValueError("MixedDataset needs at least one source dataset")
        if len(weights) != len(datasets):
            raise ValueError("one weight per source dataset required")
        self.datasets = list(datasets)
        self.weights = [float(w) for w in weights]
        self._offsets = []
        total = 0
        for d in self.datasets:
            total += len(d)
            self._offsets.append(total)
        if total == 0:
            raise ValueError("all source datasets are empty")

    def __len__(self) -> int:
        return self._offsets[-1]

    def source_index(self, idx: int) -> int:
        for i, off in enumerate(self._offsets):
            if idx < off:
                return i
        raise IndexError(idx)

    def __getitem__(self, idx: int) -> dict:
        i = self.source_index(idx)
        local = idx - (self._offsets[i - 1] if i else 0)
        s = self.datasets[i][local]
        s["meta"]["mixed_source"] = self.datasets[i].dataset_name
        s["meta"]["mixed_source_weight"] = self.weights[i]
        return s

    def per_sample_weights(self) -> List[float]:
        """One weight per global sample (for WeightedRandomSampler)."""
        out: List[float] = []
        for d, w in zip(self.datasets, self.weights):
            out.extend([w] * len(d))
        return out

    @property
    def source_names(self) -> List[str]:
        return [d.dataset_name for d in self.datasets]


def _verify_stats_gate(mcfg: dict, source_names: List[str]) -> dict:
    """Enforce the no-mixing-before-verification rule. Returns the stats."""
    stats_path = mcfg.get("verified_stats")
    if not stats_path:
        raise ValueError(
            "MIXED DATASETS ARE GATED: per-dataset statistics and "
            "normalization must be verified BEFORE mixing GAMUS with "
            "DFC2019 (binding user constraint). Run `python model.py stats` "
            "for both datasets, review the report, set 'verified: true' in "
            "it, then point dataset.verified_stats at that JSON file.")
    p = Path(stats_path)
    if not p.exists():
        raise FileNotFoundError(f"verified_stats artifact not found: {p}")
    stats = load_json(p)
    if not stats.get("verified", False):
        raise ValueError(
            f"{p}: stats artifact exists but 'verified' is not true — a "
            "human must review the per-dataset statistics before mixing. "
            "(Constraint: do not mix GAMUS and DFC2019 until per-dataset "
            "statistics and normalization are verified.)")
    covered = set(stats.get("datasets", {}).keys())
    missing = [n for n in source_names if n not in covered]
    if missing:
        raise ValueError(
            f"stats artifact {p} does not cover dataset(s) {missing} — "
            f"covered: {sorted(covered)}. Re-run `model.py stats` for them.")
    return stats


def build_mixed_datasets(mcfg: dict, *, crop_size=None, augment: bool = False,
                          load_depth: bool = True, depth_cache_dir=None,
                          clamp_agl_min: float = 0.0,
                          seed: int = 42) -> Dict[str, "MixedDataset"]:
    """{split: MixedDataset} from a mixed dataset config.

    Config:
        name: mixed
        verified_stats: outputs/stats/dataset_stats.json   # REQUIRED + true
        sources:
          - name: dfc2019
            weight: 1.0
            ...per-dataset fields (rgb_dir/truth_dir/splits_json or
            source/local_root/manifest/...)
    Per-dataset splits are aligned by split NAME (train/val/test); a source
    missing a split is skipped for that split (surface, not crash).
    """
    sources = mcfg.get("sources") or []
    if not sources:
        raise ValueError("mixed config requires a non-empty 'sources' list")
    names = [s.get("name") for s in sources]
    if len(set(names)) != len(names):
        raise ValueError(f"duplicate source names in mixed config: {names}")
    _verify_stats_gate(mcfg, names)

    from .factory import build_datasets

    per_split_datasets: Dict[str, list] = {}
    weights: Dict[str, list] = {}
    for src in sources:
        sub_cfg = {"dataset": dict(src), "paths": {}}
        dss = build_datasets(sub_cfg, crop_size=crop_size, augment=augment,
                             load_depth=load_depth,
                             depth_cache_dir=depth_cache_dir,
                             clamp_agl_min=clamp_agl_min, seed=seed)
        for split, ds in dss.items():
            per_split_datasets.setdefault(split, []).append(ds)
            weights.setdefault(split, []).append(float(src.get("weight", 1.0)))

    out = {}
    for split, dss in per_split_datasets.items():
        if not dss:
            continue
        out[split] = MixedDataset(dss, weights[split])
    if not out:
        raise ValueError("mixed config produced no splits")
    return out
