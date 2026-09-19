"""Reusable PyTorch dataset for DFC2019/US3D Track-1 style triples.

Contract (frozen — later phases and the FastAPI backend build on it):
    sample = {
        "rgb": float32 [3,H,W]  ImageNet-normalized (matches DAv2 preprocessing)
        "agl": float32 [1,H,W]  metres above ground, clamped >= 0 (clean_agl)
        "cls": int64   [1,H,W]  RAW class ids (meanings not yet verified)
        "dn":  float32 [1,H,W]  min-max normalized relative depth  (optional)
        "dem": float32 [1,H,W]  DEM prior for Method-D ablation      (optional;
                                when present, MUST be jointly cropped/flipped
                                with the other layers — see below)
        "meta": {stem, h, w, y0, x0, flipped, rot90,
                 dem_tag ("dem:<file>" | "SYNTHETIC-DEM-PROXY" | None)}
    }

Alignment guarantee: ANY spatial op (random crop, flip, rot90) samples ONE
window / ONE transform choice and applies it to ALL layers. There is no code
path where rgb and agl can drift apart — including the new ``dem`` layer,
which is folded into the SAME joint transform (see
``tests/test_dataset.py::test_crop_alignment_across_layers_with_dem``).

Two granularities, one class:
    crop_size=None          -> full tile (e.g. 1024x1024)  — evaluation
    crop_size=512, augment  -> random sub-tile + joint augs — Phase-2 training
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
from torch.utils.data import Dataset

from .geo import TilePaths, read_tile
from .normalize import clean_agl, minmax_normalize, minmax_normalize_with_stats

# ImageNet stats — same constants Depth Anything V2's preprocessing uses.
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


@dataclass
class DFC2019Config:
    rgb_dir: Path
    truth_dir: Path
    depth_cache_dir: Optional[Path] = None  # dir with {stem}.npy raw DAv2 outputs
    load_depth: bool = True  # False -> dn omitted (pre-cache runs)
    crop_size: Optional[int] = None  # None = full tile
    augment: bool = False
    clamp_agl_min: float = 0.0
    seed: int = 42
    # DEM prior (Method-D ablation; see depthwizard/demprior.py). When
    # ``dem_dir`` is set, the dataset loads ``{stem}.tif`` from it and
    # folds the resulting array into the SAME joint crop/flip/rot90 transform
    # as rgb/agl/cls/dn — see test_crop_alignment_across_layers_with_dem.
    # When ``dem_dir`` is None AND ``synth_dem_fallback`` is True, the
    # dataset synthesises a DEM from the AGL via demprior.synth_dem_from_agl
    # and tags every sample with the literal SYNTHETIC-DEM-PROXY tag —
    # the training-time ablation contract documented in demprior.py.
    # When both are unset (default), no dem layer is produced — the
    # dataset behaves exactly as it did before this hook was added.
    dem_dir: Optional[Path] = None
    synth_dem_fallback: bool = False
    synth_dem_sigma_m: float = 8.0
    # Per-tile GSD (metres/pixel) for the synthetic-DEM sigma conversion.
    # When None, demprior assumes 1 m / px AND the SYNTHETIC-DEM-PROXY tag
    # must travel with the sample (which the dataset does automatically via
    # meta.dem_tag). The honesty rule (worklog Section 4) forbids silently
    # assuming a GSD on Track-1 imagery; surfacing the tag is the contract.
    synth_dem_gsd_m: Optional[float] = None
    _rng: random.Random | None = field(default=None, repr=False, compare=False)

    def __post_init__(self):
        self.rgb_dir = Path(self.rgb_dir)
        self.truth_dir = Path(self.truth_dir)
        self.depth_cache_dir = (
            Path(self.depth_cache_dir) if self.depth_cache_dir else None
        )
        self.dem_dir = Path(self.dem_dir) if self.dem_dir else None
        self._rng = random.Random(self.seed)


@dataclass
class DepthSample:
    """Typed view of one sample — used by tests; dicts flow through training."""

    rgb: torch.Tensor
    agl: torch.Tensor
    cls: torch.Tensor
    dn: Optional[torch.Tensor]
    meta: Dict


class DFC2019Dataset(Dataset):
    def __init__(self, tiles: List[TilePaths], config: DFC2019Config):
        if len(tiles) == 0:
            raise ValueError(
                "Empty tile list — check discovery/splits before constructing."
            )
        self.tiles = tiles
        self.cfg = config
        if config.crop_size is not None and config.augment:
            # keep per-epoch determinism: rng shared, seeded once
            pass

    def __len__(self) -> int:
        return len(self.tiles)

    # ------------------------------------------------------------------
    def _load_depth_with_stats(
        self, stem: str, h: int, w: int
    ) -> tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """(normalized Dn, RAW-tile stats [4]) — FiLM conditioning (Exp 1)."""
        if not self.cfg.load_depth or self.cfg.depth_cache_dir is None:
            return None, None
        # Phase 1 (GAMUS integration): namespaced layout first
        # (<model_tag>/dfc2019/{stem}.npy), legacy flat layout
        # (<model_tag>/{stem}.npy) second — existing caches keep working.
        from .geo import depth_npy_candidates as _cand

        candidates = _cand(self.cfg.depth_cache_dir, "dfc2019", stem)
        f = next((p for p in candidates if p.exists()), None)
        if f is None:
            raise FileNotFoundError(
                f"Depth cache miss for '{stem}': none of {candidates}. Run "
                "`python model.py depth` first, or set load_depth=False."
            )
        raw = np.load(f)
        if raw.shape != (h, w):
            raise ValueError(
                f"Depth cache for '{stem}' has shape {raw.shape}, tile grid is "
                f"{(h, w)} — cache and rasters are out of sync. Delete the "
                "stale .npy and re-run the depth command."
            )
        return minmax_normalize_with_stats(raw)

    def _load_depth(self, stem: str, h: int, w: int) -> Optional[np.ndarray]:
        return self._load_depth_with_stats(stem, h, w)[0]

    # ------------------------------------------------------------------
    def _resolve_dem(
        self, stem: str, agl: np.ndarray, h: int, w: int
    ) -> tuple[Optional[np.ndarray], Optional[str]]:
        """Return (dem[H,W] float32 | None, tag | None).

        Three paths, all folded through the SAME joint transform later:

          * Real DEM on disk at ``cfg.dem_dir / f"{stem}.tif"`` -> load,
            tag ``"dem:{filename}"``. CRS alignment vs the (CRS-less,
            Track-1) tile is NOT checked here — the worklog Section 4 GSD
            honesty rule applies: a Track-1 tile cannot be honestly
            aligned to a real DEM. If your training tiles ARE
            georeferenced, validate alignment once outside this loop.
          * No real DEM and ``cfg.synth_dem_fallback=True`` -> synthesise
            via demprior.synth_dem_from_agl, tag with the literal
            ``SYNTHETIC-DEM-PROXY`` string (the contract every output /
            worklog line MUST carry in this case).
          * Neither (default) -> ``(None, None)``; the dataset produces no
            dem layer, identical to its pre-Method-D behavior.
        """
        if self.cfg.dem_dir is None and not self.cfg.synth_dem_fallback:
            return None, None

        if self.cfg.dem_dir is not None:
            f = self.cfg.dem_dir / f"{stem}.tif"
            if not f.exists():
                raise FileNotFoundError(
                    f"DEM cache miss for '{stem}': {f} not found. The "
                    "Method-D ablation requires a per-tile DEM file at "
                    "{stem}.tif under dem_dir. Either populate the cache, "
                    "switch to synth_dem_fallback=True (and tag your "
                    "outputs SYNTHETIC-DEM-PROXY), or unset dem_dir."
                )
            # rasterio single-band read; CRS/transform not required here
            # (the DEM is used as a conditioning channel, not for
            # anchoring). The alignment-vs-tile check happens upstream.
            import rasterio

            with rasterio.open(f) as ds:
                if ds.count != 1:
                    raise ValueError(
                        f"DEM '{f.name}' has {ds.count} bands; expected 1 "
                        "(single-band elevation raster)."
                    )
                dem = ds.read(1).astype(np.float32)
            if dem.shape != (h, w):
                raise ValueError(
                    f"DEM for '{stem}' has shape {dem.shape}, tile grid is "
                    f"{(h, w)} — DEM and tile are out of sync. Pre-resample "
                    "the DEM to the tile grid (demprior.resample_dem_to_tile) "
                    "before caching, or rebuild the cache."
                )
            return dem, f"dem:{f.name}"

        # Synthetic fallback. The SYNTHETIC-DEM-PROXY tag is non-negotiable;
        # it travels through the sample meta and (transitively) into every
        # consumer's output/worklog line per demprior.py's contract.
        from .demprior import SYNTHETIC_DEM_TAG, synth_dem_from_agl

        dem = synth_dem_from_agl(
            agl, sigma_m=self.cfg.synth_dem_sigma_m, gsd_m=self.cfg.synth_dem_gsd_m
        )
        return dem, SYNTHETIC_DEM_TAG

    # ------------------------------------------------------------------
    def _joint_crop(self, layers: Dict[str, np.ndarray], rng: random.Random):
        """One shared window for every layer, or identity when crop is None.

        Delegates to depthwizard.datasets.transforms.joint_crop (single
        source of truth shared with the new adapters) — behavior identical
        to the pre-GAMUS implementation (test_dataset.py pins it).
        """
        from .datasets.transforms import joint_crop

        return joint_crop(layers, rng, self.cfg.crop_size)

    def _joint_flip_rot(self, layers: Dict[str, np.ndarray], rng: random.Random):
        """One shared (rot90 k, hflip, vflip) for every layer.

        Delegates to depthwizard.datasets.transforms.joint_flip_rot —
        behavior identical to the pre-GAMUS implementation.
        """
        from .datasets.transforms import joint_flip_rot

        return joint_flip_rot(layers, rng)

    # ------------------------------------------------------------------
    def __getitem__(self, idx: int) -> Dict:
        tile = self.tiles[idx]
        data = read_tile(tile)  # raises on grid mismatch
        rgb, agl, cls = data["rgb"], data["agl"], data["cls"]
        h, w = rgb.shape[:2]

        dn, dn_stats = self._load_depth_with_stats(tile.stem, h, w)
        # Method-D ablation: optionally resolve a DEM prior (real on disk
        # or SYNTHETIC-DEM-PROXY). The DEM is folded into the SAME joint
        # crop/flip/rot90 transform as the other layers — one window,
        # one transform, no per-layer drift (see
        # test_crop_alignment_across_layers_with_dem).
        dem, dem_tag = self._resolve_dem(tile.stem, agl, h, w)

        layers: Dict[str, np.ndarray] = {"rgb": rgb, "agl": agl, "cls": cls}
        if dn is not None:
            layers["dn"] = dn
        if dem is not None:
            layers["dem"] = dem

        y0 = x0 = 0
        k, do_h, do_v = 0, False, False
        rng = self.cfg._rng
        assert rng is not None
        if self.cfg.crop_size is not None:
            y0, x0 = self._joint_crop(layers, rng)
        if self.cfg.augment:
            k, do_h, do_v = self._joint_flip_rot(layers, rng)

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
        cls_t = torch.from_numpy(np.ascontiguousarray(layers["cls"]))[None, ...].long()
        dn_t = (
            torch.from_numpy(np.ascontiguousarray(layers["dn"]))[None, ...]
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

        meta = {
            "stem": tile.stem,
            "h": agl_t.shape[1],
            "w": agl_t.shape[2],
            "y0": y0,
            "x0": x0,
            "rot90": k,
            "flip_h": do_h,
            "flip_v": do_v,
            "dem_tag": dem_tag,
            # Phase 1 (GAMUS integration): dataset provenance — additive keys.
            "dataset": "dfc2019",
            "sample_id": tile.stem,
        }
        dn_stats_t = (
            torch.from_numpy(np.asarray(dn_stats, dtype=np.float32))
            if dn_stats is not None
            else None
        )
        return {
            "rgb": rgb_t,
            "agl": agl_t,
            "cls": cls_t,
            "dn": dn_t,
            "dn_stats": dn_stats_t,
            "dem": dem_t,
            "meta": meta,
        }


# ---------------------------------------------------------------------------
def discover_and_split(
    config: DFC2019Config,
    splits_json: Path | str,
    dataset_class=DFC2019Dataset,
    **adapter_kwargs,
):
    """Convenience: discover tiles, load frozen splits.json, build the three
    dataset objects. Used by 04/05 scripts and (later) the training loop.

    ``dataset_class`` (Phase 1, additive): defaults to the frozen
    DFC2019Dataset; the datasets.dfc2019 adapter passes DFC2019Adapter so
    existing call sites gain semantics without any duplication.
    """
    from .geo import discover_tiles, load_json

    tiles, problems = discover_tiles(config.rgb_dir, config.truth_dir)
    if problems:
        print("[discover_and_split] discovery problems (see inspect step):")
        for p in problems:
            print("  -", p)
    by_stem = {t.stem: t for t in tiles}
    payload = load_json(splits_json)
    splits = payload["splits"] if "splits" in payload else payload  # wrapper-aware
    out = {}
    for split in ("train", "val", "test"):
        stems = splits[split]
        missing = [s for s in stems if s not in by_stem]
        if missing:
            raise KeyError(
                f"splits.json references tiles not found on disk: {missing[:5]}"
            )
        cfg = DFC2019Config(
            rgb_dir=config.rgb_dir,
            truth_dir=config.truth_dir,
            depth_cache_dir=config.depth_cache_dir,
            load_depth=config.load_depth,
            crop_size=config.crop_size,
            augment=config.augment,
            clamp_agl_min=config.clamp_agl_min,
            dem_dir=config.dem_dir,
            synth_dem_fallback=config.synth_dem_fallback,
            synth_dem_sigma_m=config.synth_dem_sigma_m,
            synth_dem_gsd_m=config.synth_dem_gsd_m,
            seed=config.seed + hash(split) % 1000,
        )
        out[split] = dataset_class([by_stem[s] for s in stems], cfg, **adapter_kwargs)
    return out


def build_datasets(**kwargs) -> Dict[str, DFC2019Dataset]:
    """Thin wrapper kept for API stability across phases."""
    cfg = DFC2019Config(
        **{k: v for k, v in kwargs.items() if k in DFC2019Config.__dataclass_fields__}
    )
    splits_json = kwargs.get("splits_json")
    if splits_json is None:
        raise ValueError("build_datasets requires splits_json=<path>")
    return discover_and_split(cfg, splits_json)
