"""GAMUS dataset adapter (Phase 2 of the GAMUS integration).

Sources (VERIFIED — see depthwizard/datasets/semantics.py provenance rules):
    HF dataset card  : https://huggingface.co/datasets/earthflow/GAMUS
                       8,724 tiles (train 5,004 / val 859 / test 2,861),
                       cities DC/NYC/PHL, 1024x1024, GSD 0.33 m,
                       CC-BY-4.0, NO CRS anywhere (HDF5 carries no attrs).
    Official loader  : EarthNets/RSI-MMSegmentation gamus_dataset.py (HDF5,
                       key "image" per file, sorted listdir).
    Paper            : arXiv:2305.14914 (height modality = nDSM semantics).

Raw HDF5 layout (this adapter's PRIMARY source — keeps float32 heights):
    images/{split}/{ID}_RGB.h5    key "image"  (1024,1024,3) uint8 HWC
    heights/{split}/{ID}_AGL.h5   key "image"  (1024,1024)   float32
    classes/{split}/{ID}_CLS.h5   key "image"  (1024,1024)   uint8 OR float32
                                                       (inconsistent — cast)

Three backends, in order of preference:
    source="hf"     (DEFAULT) lazy per-file ``hf_hub_download`` into a local
                    cache — only the tiles actually needed are downloaded;
                    no 80 GB materialization. Official splits verbatim.
    source="local"  offline / pre-downloaded subset: a directory following
                    the raw layout above (works with tools/make_fake_gamus
                    fixtures — network-free tests).
    backend="dataset4eo"  OPTIONAL, NOT the default and NOT implemented yet:
                    the EarthNets_GAMUS litData/Dataset4EO streaming
                    re-packaging. Deliberately deferred because it stores
                    height as FLOAT16 (metric-regression precision loss) and
                    carries CC-BY-NC-ND-4.0 (stricter than the raw release's
                    CC-BY-4.0). The config knob exists so the backend can be
                    added without breaking configs; enabling it today raises
                    NotImplementedError with this rationale.

Honesty fields every sample carries in ``meta``:
    height_semantics : "nDSM/AGL (GAMUS paper: height modality represents
                       nDSM data)"
    units            : "undocumented — ASSUMED metres (values consistent
                       with metres); flagged, not fabricated"
    gsd_m            : 0.33 (documented on the HF card)
    georef           : "none" (never fabricate CRS -> relative-DSM mode,
                       same as DFC2019 Track-1)
    split            : official split name
    source           : "hf" | "local"

Official splits are used VERBATIM (scene-level by construction of the
GAMUS release). Deterministic sample order: sorted sample ids; ``limit``
caps each split's list after sorting (reproducible budgets).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from .base import AdapterConfig, BaseDepthDataset
from ..geo import depth_npy_candidates, dump_json
from ..normalize import minmax_normalize

GAMUS_REPO_ID = "earthflow/GAMUS"
GAMUS_SPLITS = ("train", "val", "test")
_H5_KEY = "image"

# Honesty constants (sources in module docstring).
GAMUS_HEIGHT_SEMANTICS = ("nDSM/AGL (GAMUS paper: height modality "
                          "represents nDSM data)")
GAMUS_UNITS_NOTE = "undocumented — ASSUMED metres (values consistent with metres)"
GAMUS_GSD_M = 0.33


@dataclass
class GAMUSConfig(AdapterConfig):
    """GAMUS adapter configuration.

    source: "hf" (default, lazy per-file download) | "local" (offline dir).
    backend: "hdf5" (default, this module) | "dataset4eo" (optional,
    not implemented — see module docstring for the deferral rationale).
    """
    source: str = "hf"
    backend: str = "hdf5"
    local_root: Optional[Path] = None            # required when source="local"
    repo_id: str = GAMUS_REPO_ID
    manifest: Optional[Path] = None              # splits manifest (offline idx)
    save_manifest: Optional[Path] = None         # write manifest after listing
    hf_cache_dir: Optional[Path] = None          # local download cache root
    limit: int = 0                               # per-split cap (0 = all)
    depth_cache_dir: Optional[Path] = None       # model-tag dir; gamus/ inside
    splits: tuple = GAMUS_SPLITS                 # which splits to index

    def __post_init__(self):
        super().__post_init__()
        if self.source not in ("hf", "local"):
            raise ValueError(f"source must be 'hf' or 'local', got {self.source!r}")
        if self.backend not in ("hdf5", "dataset4eo"):
            raise ValueError(f"backend must be 'hdf5' or 'dataset4eo', "
                             f"got {self.backend!r}")
        if self.backend == "dataset4eo":
            raise NotImplementedError(
                "Dataset4EO/EarthNets_GAMUS streaming backend is deliberately "
                "NOT the default and not yet implemented: it stores height as "
                "float16 (metric-regression precision loss) and carries "
                "CC-BY-NC-ND-4.0 vs the raw release's CC-BY-4.0. Use the "
                "default raw-HDF5 backend (source='hf' or 'local').")
        if self.source == "local":
            if self.local_root is None:
                raise ValueError("source='local' requires local_root")
            self.local_root = Path(self.local_root)
        if self.manifest is not None:
            self.manifest = Path(self.manifest)
        if self.save_manifest is not None:
            self.save_manifest = Path(self.save_manifest)
        if self.depth_cache_dir is not None:
            self.depth_cache_dir = Path(self.depth_cache_dir)


@dataclass
class GAMUSSample:
    """One GAMUS tile: split + sample id (e.g. 'DC_01_25')."""
    sample_id: str
    split: str

    @property
    def rgb_rel(self) -> str:
        return f"images/{self.split}/{self.sample_id}_RGB.h5"

    @property
    def agl_rel(self) -> str:
        return f"heights/{self.split}/{self.sample_id}_AGL.h5"

    @property
    def cls_rel(self) -> str:
        return f"classes/{self.split}/{self.sample_id}_CLS.h5"


# ---------------------------------------------------------------------------
# HDF5 primitives (official RSI-MMSegmentation mechanism: f["image"][()])
# ---------------------------------------------------------------------------

def read_gamus_h5(path: Path) -> np.ndarray:
    """Read one GAMUS .h5 file (key 'image'), h5py imported lazily."""
    import h5py
    with h5py.File(path, "r") as f:
        if _H5_KEY not in f:
            raise KeyError(f"{path}: expected HDF5 key '{_H5_KEY}' "
                           "(official GAMUS layout) — got keys "
                           f"{list(f.keys())}")
        return f[_H5_KEY][()]


def _hf_download(cfg: GAMUSConfig, rel_path: str) -> Path:
    """Lazy per-file download from the GAMUS repo into a local cache."""
    from huggingface_hub import hf_hub_download
    return Path(hf_hub_download(
        repo_id=cfg.repo_id, filename=rel_path, repo_type="dataset",
        cache_dir=str(cfg.hf_cache_dir) if cfg.hf_cache_dir else None))


def _resolve_sample_files(cfg: GAMUSConfig, s: GAMUSSample) -> Dict[str, Path]:
    """Local paths of the (rgb, agl, cls) triple for one sample."""
    if cfg.source == "local":
        base = cfg.local_root
        return {"rgb": base / s.rgb_rel, "agl": base / s.agl_rel,
                "cls": base / s.cls_rel}
    return {"rgb": _hf_download(cfg, s.rgb_rel),
            "agl": _hf_download(cfg, s.agl_rel),
            "cls": _hf_download(cfg, s.cls_rel)}


# ---------------------------------------------------------------------------
# Sample discovery (local walk or manifest or HF repo listing)
# ---------------------------------------------------------------------------

def _stems_from_dir(d: Path) -> List[str]:
    """Sorted sample ids from a directory of {ID}_RGB.h5 files."""
    if not d.is_dir():
        return []
    out = []
    for p in sorted(d.iterdir()):
        m = re.match(r"^(.+)_RGB\.h5$", p.name)
        if m:
            out.append(m.group(1))
    return out


def _check_triples(cfg: GAMUSConfig, per_split: Dict[str, List[str]]
                   ) -> List[str]:
    """Triple completeness check (local mode) — problems reported, never
    silently dropped (same philosophy as geo.discover_tiles)."""
    problems: List[str] = []
    if cfg.source != "local":
        return problems
    for split, ids in per_split.items():
        for sid in ids:
            s = GAMUSSample(sid, split)
            for kind, rel in (("rgb", s.rgb_rel), ("agl", s.agl_rel),
                              ("cls", s.cls_rel)):
                if not (cfg.local_root / rel).exists():
                    problems.append(f"GAMUS {split}/{sid}: missing {kind} "
                                    f"({rel})")
    return problems


def list_gamus_samples(cfg: GAMUSConfig
                       ) -> tuple[Dict[str, List[GAMUSSample]], List[str]]:
    """{split: [GAMUSSample, ...]} in deterministic (sorted) order.

    Order of resolution:
      1. manifest file (offline index; written by `model.py splits
         --dataset gamus` or save_manifest) — no network needed;
      2. source=local: walk local_root/images/{split}/;
      3. source=hf: live repo listing (HfApi) — needs network once.
    """
    if cfg.manifest is not None and cfg.manifest.exists():
        from ..geo import load_json
        payload = load_json(cfg.manifest)
        per_split = {k: [GAMUSSample(sid, k) for sid in v]
                     for k, v in payload["splits"].items()
                     if k in cfg.splits}
        problems = _check_triples(cfg, {k: [s.sample_id for s in v]
                                        for k, v in per_split.items()})
        return per_split, problems

    if cfg.source == "local":
        per_split_ids = {}
        for split in cfg.splits:
            ids = _stems_from_dir(cfg.local_root / "images" / split)
            per_split_ids[split] = ids
    else:
        from huggingface_hub import HfApi
        files = HfApi().list_repo_files(cfg.repo_id, repo_type="dataset")
        per_split_ids = {split: [] for split in cfg.splits}
        for f in files:
            parts = f.split("/")
            if len(parts) == 3 and parts[0] == "images" and \
                    parts[1] in per_split_ids:
                m = re.match(r"^(.+)_RGB\.h5$", parts[2])
                if m:
                    per_split_ids[parts[1]].append(m.group(1))
        for split in per_split_ids:
            per_split_ids[split].sort()

    if cfg.limit and cfg.limit > 0:
        for split in per_split_ids:
            per_split_ids[split] = per_split_ids[split][:cfg.limit]

    if cfg.save_manifest is not None:
        dump_json({
            "created": datetime.now(timezone.utc).isoformat(),
            "generator": "depthwizard-gamus-splits v1 (official splits, "
                         "verbatim, sorted ids)",
            "source": cfg.source, "repo_id": cfg.repo_id,
            "official": True,
            "counts": {k: len(v) for k, v in per_split_ids.items()},
            "limit": cfg.limit,
            "splits": per_split_ids,
        }, cfg.save_manifest)

    per_split = {k: [GAMUSSample(sid, k) for sid in ids]
                 for k, ids in per_split_ids.items()}
    problems = _check_triples(cfg, per_split_ids)
    return per_split, problems


# ---------------------------------------------------------------------------
# The dataset
# ---------------------------------------------------------------------------

class GAMUSDataset(BaseDepthDataset):
    dataset_name = "gamus"
    sem_legend = "gamus"

    def __init__(self, samples: List[GAMUSSample], cfg: GAMUSConfig):
        super().__init__(samples, cfg)

    # ------------------------------------------------------------------
    def _load_depth(self, sid: str, h: int, w: int) -> Optional[np.ndarray]:
        if not self.cfg.load_depth or self.cfg.depth_cache_dir is None:
            return None
        candidates = depth_npy_candidates(self.cfg.depth_cache_dir,
                                          "gamus", sid)
        f = next((p for p in candidates if p.exists()), None)
        if f is None:
            raise FileNotFoundError(
                f"GAMUS depth cache miss for '{sid}': none of {candidates}. "
                "Run `python model.py depth --dataset gamus ...` first, or "
                "set load_depth=False.")
        raw = np.load(f)
        if raw.shape != (h, w):
            raise ValueError(
                f"GAMUS depth cache for '{sid}' has shape {raw.shape}, tile "
                f"grid is {(h, w)} — cache and tiles are out of sync. Delete "
                "the stale .npy and re-run the depth command.")
        return minmax_normalize(raw)     # single source of truth (normalize.py)

    # ------------------------------------------------------------------
    def _load_arrays(self, idx: int) -> Dict[str, object]:
        s: GAMUSSample = self.samples[idx]
        paths = _resolve_sample_files(self.cfg, s)

        rgb = read_gamus_h5(paths["rgb"])
        agl = read_gamus_h5(paths["agl"])
        cls = read_gamus_h5(paths["cls"])

        # Official layout: RGB (1024,1024,3) HWC uint8; AGL (1024,1024)
        # float32; CLS (1024,1024) uint8 OR float32 (verified inconsistency)
        # -> defensive casts, NEVER reinterpretation.
        rgb = np.ascontiguousarray(rgb)
        if rgb.ndim != 3 or rgb.shape[2] < 3:
            raise ValueError(
                f"GAMUS {s.sample_id}: RGB .h5 shape {rgb.shape} — expected "
                "(H, W, 3) per the official layout.")
        rgb = rgb[:, :, :3].astype(np.uint8)
        agl = np.asarray(agl, dtype=np.float32)
        if agl.ndim == 3:                        # (1,H,W) defensive
            agl = agl[0]
        # CLS dtype inconsistency: uint8 or float32 -> int32 ids either way.
        cls = np.asarray(cls)
        if cls.ndim == 3:                        # (1,H,W) defensive
            cls = cls[0]
        cls = cls.astype(np.int32)

        h, w = rgb.shape[:2]
        if agl.shape != (h, w) or cls.shape != (h, w):
            raise ValueError(
                f"GAMUS {s.sample_id}: grid mismatch rgb{(h, w)} "
                f"agl{agl.shape} cls{cls.shape} — files NOT pixel-aligned; "
                "do not proceed.")

        dn = self._load_depth(s.sample_id, h, w)

        meta = {
            "sample_id": s.sample_id, "stem": s.sample_id,
            "split": s.split,
            "source": self.cfg.source,
            "height_semantics": GAMUS_HEIGHT_SEMANTICS,
            "units": GAMUS_UNITS_NOTE,
            "gsd_m": GAMUS_GSD_M,
            "georef": "none",
        }
        return {"rgb": rgb, "agl": agl, "cls": cls, "dn": dn, "meta": meta,
                "dem_tag": None}


def build_gamus_datasets(cfg: GAMUSConfig
                         ) -> Dict[str, GAMUSDataset]:
    """{split: GAMUSDataset} for the configured splits (official, verbatim)."""
    per_split, problems = list_gamus_samples(cfg)
    if problems:
        print("[gamus] discovery problems (fix or ignore if expected):")
        for p in problems:
            print("  -", p)
    out = {}
    for split, samples in per_split.items():
        if not samples:
            continue
        out[split] = GAMUSDataset(samples, cfg)
    if not out:
        raise ValueError(
            "GAMUS discovery produced no samples — check source/local_root/"
            f"manifest (source={cfg.source}, splits={cfg.splits}).")
    return out
