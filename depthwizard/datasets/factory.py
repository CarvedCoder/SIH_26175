"""Dataset factory — ONE config schema building any dataset adapter.

Config schema (the ``dataset:`` YAML section; every field optional —
defaults reproduce the pre-GAMUS DFC2019 path):

    dataset:
      name: dfc2019 | gamus | mixed        # default dfc2019
      load_semantics: true                 # one-hot + ignore mask layers

      # --- DFC2019 (paths may also come from the top-level paths: block,
      #     which stays fully supported — zero behavior change) ---
      rgb_dir: ...
      truth_dir: ...
      splits_json: ...

      # --- GAMUS ---
      source: hf | local                   # default hf (raw HDF5 lazy DL)
      backend: hdf5 | dataset4eo           # default hdf5 (see gamus.py)
      local_root: ...                      # required when source: local
      repo_id: earthflow/GAMUS
      manifest: ...                        # offline split index
      save_manifest: ...                   # write index after listing
      hf_cache_dir: ...
      limit: 0                             # per-split cap (deterministic)

    # --- mixed (Phase 3; GATED until per-dataset stats are verified) ---
    dataset:
      name: mixed
      sources:
        - name: dfc2019  {weight: 1.0, ...}
        - name: gamus    {weight: 1.0, ...}

Shared train/eval knobs (crop_size, augment, clamp_agl_min, load_depth,
seed) are taken from the caller (e.g. the train command's own config) and
passed through — the factory only owns DATASET identity and sources.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Mapping, Optional

from .base import AdapterConfig


def build_dfc2019(dataset_cfg: dict, paths: dict, *, crop_size=None,
                  augment: bool = False, load_depth: bool = True,
                  depth_cache_dir=None, clamp_agl_min: float = 0.0,
                  seed: int = 42) -> Mapping[str, "object"]:
    """{train,val,test} DFC2019Adapter datasets (frozen split logic reused)."""
    from ..dataset import DFC2019Config
    from .dfc2019 import DFC2019Adapter, discover_and_split_adapter

    rgb_dir = dataset_cfg.get("rgb_dir") or paths.get("rgb_dir")
    truth_dir = dataset_cfg.get("truth_dir") or paths.get("truth_dir")
    splits_json = dataset_cfg.get("splits_json") or paths.get("splits_json")
    if not (rgb_dir and truth_dir and splits_json):
        raise ValueError(
            "DFC2019 requires rgb_dir, truth_dir and splits_json (dataset: "
            "section or top-level paths:).")
    cfg = DFC2019Config(
        rgb_dir=Path(rgb_dir), truth_dir=Path(truth_dir),
        depth_cache_dir=Path(depth_cache_dir) if depth_cache_dir else None,
        load_depth=load_depth, crop_size=crop_size, augment=augment,
        clamp_agl_min=clamp_agl_min, seed=seed)
    return discover_and_split_adapter(
        cfg, Path(splits_json),
        load_semantics=bool(dataset_cfg.get("load_semantics", True)))


def build_gamus(dataset_cfg: dict, *, crop_size=None, augment: bool = False,
                load_depth: bool = True, depth_cache_dir=None,
                clamp_agl_min: float = 0.0, seed: int = 42
                ) -> Mapping[str, "object"]:
    """{split: GAMUSDataset} — official splits, verbatim."""
    from .gamus import GAMUSConfig, build_gamus_datasets

    cfg = GAMUSConfig(
        source=dataset_cfg.get("source", "hf"),
        backend=dataset_cfg.get("backend", "hdf5"),
        local_root=(Path(dataset_cfg["local_root"])
                    if dataset_cfg.get("local_root") else None),
        repo_id=dataset_cfg.get("repo_id", "earthflow/GAMUS"),
        manifest=(Path(dataset_cfg["manifest"])
                  if dataset_cfg.get("manifest") else None),
        save_manifest=(Path(dataset_cfg["save_manifest"])
                       if dataset_cfg.get("save_manifest") else None),
        hf_cache_dir=(Path(dataset_cfg["hf_cache_dir"])
                      if dataset_cfg.get("hf_cache_dir") else None),
        limit=int(dataset_cfg.get("limit", 0)),
        depth_cache_dir=(Path(depth_cache_dir) if depth_cache_dir else None),
        splits=tuple(dataset_cfg.get("splits", ("train", "val", "test"))),
        crop_size=crop_size, augment=augment, load_depth=load_depth,
        load_semantics=bool(dataset_cfg.get("load_semantics", True)),
        clamp_agl_min=clamp_agl_min, seed=seed)
    return build_gamus_datasets(cfg)


def build_datasets(cfg: dict, *, crop_size=None, augment: bool = False,
                   load_depth: bool = True, depth_cache_dir=None,
                   clamp_agl_min: float = 0.0,
                   seed: int = 42) -> Mapping[str, "object"]:
    """Build {train,val,test} datasets from a full YAML config dict.

    Supports:
      * legacy configs (no ``dataset:`` section) -> DFC2019 from ``paths:``;
      * dataset: name: dfc2019 | gamus (mixed arrives in Phase 3, gated).
    """
    paths = cfg.get("paths", {}) or {}
    dcfg = cfg.get("dataset", {}) or {}
    name = dcfg.get("name", "dfc2019")

    if name == "dfc2019":
        return build_dfc2019(dcfg, paths, crop_size=crop_size,
                             augment=augment, load_depth=load_depth,
                             depth_cache_dir=depth_cache_dir,
                             clamp_agl_min=clamp_agl_min, seed=seed)
    if name == "gamus":
        return build_gamus(dcfg, crop_size=crop_size, augment=augment,
                           load_depth=load_depth,
                           depth_cache_dir=depth_cache_dir,
                           clamp_agl_min=clamp_agl_min, seed=seed)
    if name == "mixed":
        from .mixed import build_mixed_datasets
        return build_mixed_datasets(dcfg, crop_size=crop_size,
                                    augment=augment, load_depth=load_depth,
                                    depth_cache_dir=depth_cache_dir,
                                    clamp_agl_min=clamp_agl_min, seed=seed)
    raise ValueError(
        f"unknown dataset name '{name}' — expected dfc2019 | gamus | mixed.")


def build_dataset(cfg: dict, split: str, **kwargs):
    """Convenience: build ONE split's dataset (see build_datasets)."""
    ds = build_datasets(cfg, **kwargs)
    if split not in ds:
        raise KeyError(f"split '{split}' not produced by the dataset config "
                       f"(got {sorted(ds)})")
    return ds[split]
