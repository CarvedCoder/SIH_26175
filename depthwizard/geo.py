"""Geospatial I/O helpers built on rasterio.

Rules encoded here (do not violate them elsewhere):
  * A ``.tif`` extension proves nothing about georeferencing. We always read
    the profile and report CRS/transform explicitly.
  * DFC2019 Track-1 files ship WITHOUT CRS (identity transform). Pairing is
    therefore done by filename stem + grid-shape equality, and alignment is
    confirmed visually with quicklook composites, not assumed.
  * Nothing in this module invents CRS, bounds, or class meanings.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    import rasterio
except ImportError as e:  # pragma: no cover
    raise ImportError(
        "rasterio is required for all geospatial I/O. "
        "Install with: pip install rasterio"
    ) from e


# ---------------------------------------------------------------------------
# Tile naming / discovery
# ---------------------------------------------------------------------------

# Matches e.g. JAX_004_006, OMA_012_003, tile_07_11 ...
STEM_RE = re.compile(r"^(?P<city>[A-Za-z]+)_(?P<row>\d+)_(?P<col>\d+)$")

RGB_SUFFIXES = ("_RGB.tif", "_RGB.tiff", "_RGB.TIF")
AGL_SUFFIXES = ("_AGL.tif", "_AGL.tiff", "_AGL.TIF")
CLS_SUFFIXES = ("_CLS.tif", "_CLS.tiff", "_CLS.TIF")

from typing import TypedDict


class TileMeta(TypedDict):
    stem: str
    height: int
    width: int
    rgb_crs: str | None
    agl_crs: str | None
    cls_crs: str | None
    rgb_transform: list[float] | None
    agl_transform: list[float] | None


class TileData(TypedDict):
    rgb: np.ndarray
    agl: np.ndarray
    cls: np.ndarray
    meta: TileMeta


@dataclass
class TilePaths:
    """Absolute paths of the aligned (rgb, agl, cls) triple for one tile."""

    stem: str
    rgb: Path
    agl: Path
    cls: Path

    def to_dict(self) -> Dict[str, str]:
        return {
            "stem": self.stem,
            "rgb": str(self.rgb),
            "agl": str(self.agl),
            "cls": str(self.cls),
        }


def parse_stem(stem: str) -> Tuple[str, int, int]:
    """'JAX_004_006' -> ('JAX', 4, 6). Raises ValueError on unknown naming."""
    m = STEM_RE.match(stem)
    if m is None:
        raise ValueError(
            f"Tile stem '{stem}' does not match <city>_<row>_<col>. "
            "If the real dataset uses another naming scheme, extend STEM_RE "
            "in depthwizard/geo.py — do NOT silently fall back to random splits."
        )
    return m.group("city"), int(m.group("row")), int(m.group("col"))


def _index_dir(directory: Path, suffixes) -> Dict[str, Path]:
    """Map file stem -> path for files ending with one of `suffixes`."""
    if not directory.exists():
        raise FileNotFoundError(f"Directory not found: {directory}")
    out: Dict[str, Path] = {}
    for p in sorted(directory.rglob("*")):
        if not p.is_file():
            continue
        for suf in suffixes:
            if p.name.endswith(suf):
                out[p.name[: -len(suf)]] = p
                break
    return out


def discover_tiles(
    rgb_dir: Path | str, truth_dir: Path | str
) -> Tuple[List[TilePaths], List[str]]:
    """Discover RGB/AGL/CLS triples by filename stem.

    Returns (tiles, problems). ``problems`` is a list of human-readable
    issues: orphan files present on only one side. We never silently drop
    orphans — silently dropping data is how alignment bugs are born.
    """
    rgb_dir, truth_dir = Path(rgb_dir), Path(truth_dir)
    rgb_idx = _index_dir(rgb_dir, RGB_SUFFIXES)
    agl_idx = _index_dir(truth_dir, AGL_SUFFIXES)
    cls_idx = _index_dir(truth_dir, CLS_SUFFIXES)

    problems: List[str] = []
    for stem in sorted(set(rgb_idx) - set(agl_idx) - set(cls_idx)):
        problems.append(f"RGB without truth pair: {stem}")
    for stem in sorted((set(agl_idx) | set(cls_idx)) - set(rgb_idx)):
        problems.append(f"Truth without RGB pair: {stem}")

    common = sorted(set(rgb_idx) & set(agl_idx) & set(cls_idx))
    tiles = [
        TilePaths(stem=s, rgb=rgb_idx[s], agl=agl_idx[s], cls=cls_idx[s])
        for s in common
    ]
    return tiles, problems


# ---------------------------------------------------------------------------
# Reading
# ---------------------------------------------------------------------------


def is_georeferenced(profile: dict) -> Tuple[bool, str]:
    """True iff the raster carries a CRS (identity transform alone does not
    georeference an image). Returns (flag, human-readable reason)."""
    crs = profile.get("crs", None)
    if crs is not None:
        return True, f"CRS present: {crs}"
    transform = profile.get("transform", None)
    identity = rasterio.Affine.identity()
    if transform is not None and transform != identity:
        return True, "No CRS but non-identity transform (rare, inspect manually)"
    return False, "No CRS, identity transform -> pixel coordinates only"


def pixel_size_metres(crs, transform) -> Optional[Tuple[float, float]]:
    """Ground-sample distance per pixel, in metres — or None when unknown.

    The SINGLE source of truth for "what is one source pixel worth in metres"
    across the inference payload, the viewer's mesh vertical scale, the slope
    HUD math, and the DEM-resample step in :mod:`depthwizard.demprior`.

    Returns ``(pixel_w_m, pixel_h_m)`` where:

      * ``pixel_w_m`` is the size of one pixel in the *x* (easting / longitude)
        direction, always positive, always in metres;
      * ``pixel_h_m`` is the size in the *y* (northing / latitude) direction,
        always positive, always in metres.

    Honesty contract (frozen by tests in ``tests/test_payload_meta.py``):

      * ``crs is None`` -> ``None``. This is the *only* honest null case.
        DFC2019 Track-1 imagery ships without a CRS — every consumer MUST treat
        ``None`` as "we do not know what one pixel is worth", and refuse to
        compute slope / vertical scale / DEM alignment from it. Fabricating a
        default value here is the silent-wrong-number bug class (see worklog
        Section 4, "GSD honesty regression").
      * Projected CRS with metre units (EPSG:32617, EPSG:3857...) -> ``(|a|, |e|)``
        directly. Hand-verified on the EPSG:32617 + 30 m transform used by
        ``tests/test_anchoring.py`` and ``tests/test_payload_meta.py``.
      * Projected CRS with non-metre linear units (US survey foot, etc.) ->
        multiplied by the CRS ``linear_units_factor``. Verified never to fall
        back to 1.0 silently when the factor is unavailable — that path returns
        ``None`` so the unknown is reported rather than guessed.
      * Geographic CRS (degrees) -> a local-latitude approximation at the
        raster's y-origin (``transform.f``):

            1° lat  ≈ 111_320 m   (constant; WGS84 meridian length)
            1° lon  ≈ 111_320 * cos(lat) m

        This is honest about being an approximation. It is correct to <0.5%
        within ±30° of the equator and <2% poleward of ±60°. Slope and mesh
        scale derived from it are likewise approximation-bounded; consumers
        that need higher precision should reproject to a metric CRS first.
      * Any CRS whose unit machinery cannot be queried -> ``None`` (reported
        unknown, never silently guessed as 1.0).

    The function never raises for CRS/transform shape issues — those are
    upstream contract violations caught by ``read_image``/``read_raster``.
    """
    if crs is None or transform is None:
        return None
    try:
        pa = abs(float(transform.a))
        pe = abs(float(transform.e))
    except (AttributeError, TypeError):
        return None
    if pa == 0.0 or pe == 0.0:
        return None

    try:
        is_geog = bool(getattr(crs, "is_geographic", False))
    except Exception:
        is_geog = False
    try:
        is_proj = bool(getattr(crs, "is_projected", False))
    except Exception:
        is_proj = False

    if is_geog and not is_proj:
        # Degrees → metres via the local-latitude approximation described above.
        try:
            center_lat = float(transform.f) if transform.f is not None else 0.0
        except (AttributeError, TypeError):
            center_lat = 0.0
        cos_lat = max(0.0, math.cos(math.radians(center_lat)))
        m_per_deg_lat = 111_320.0
        m_per_deg_lon = 111_320.0 * cos_lat
        return (pa * m_per_deg_lon, pe * m_per_deg_lat)

    # Projected CRS — read the linear-units factor from the CRS itself.
    factor = None
    try:
        luf = crs.linear_units_factor  # (unit_name, factor_to_metre)
        if luf and len(luf) >= 2 and luf[1]:
            factor = float(luf[1])
    except Exception:
        factor = None
    if factor is None:
        # We do NOT fall back to 1.0 — that would silently mislabel a CRS
        # whose unit machinery rasterio did not expose. Report unknown.
        return None
    return (pa * factor, pe * factor)


def read_raster(path: Path | str) -> Tuple[np.ndarray, dict]:
    """Read a raster. Returns (array[C,H,W], profile). No resampling, no
    warping — read exactly what is on disk."""
    with rasterio.open(path) as src:
        arr = src.read()
        profile = src.profile.copy()
        profile["crs_str"] = str(src.crs) if src.crs else None
    return arr, profile


def read_tile(paths: TilePaths) -> TileData:
    rgb, rgb_prof = read_raster(paths.rgb)
    agl, agl_prof = read_raster(paths.agl)
    cls, cls_prof = read_raster(paths.cls)

    if rgb.shape[0] < 3:
        raise ValueError(
            f"{paths.stem}: RGB raster has {rgb.shape[0]} bands, expected >=3"
        )

    rgb = rgb[:3].transpose(1, 2, 0)
    agl = agl[0] if agl.ndim == 3 else agl
    cls = cls[0] if cls.ndim == 3 else cls

    h, w = rgb.shape[:2]

    if agl.shape != (h, w) or cls.shape != (h, w):
        raise ValueError(
            f"{paths.stem}: grid mismatch "
            f"rgb{rgb.shape[:2]} agl{agl.shape} cls{cls.shape} "
            f"— files are NOT pixel-aligned; do not proceed."
        )

    return {
        "rgb": np.ascontiguousarray(rgb, dtype=np.uint8),
        "agl": np.ascontiguousarray(agl, dtype=np.float32),
        "cls": np.ascontiguousarray(cls, dtype=np.int32),
        "meta": {
            "stem": paths.stem,
            "height": int(h),
            "width": int(w),
            "rgb_crs": rgb_prof.get("crs_str"),
            "agl_crs": agl_prof.get("crs_str"),
            "cls_crs": cls_prof.get("crs_str"),
            "rgb_transform": (
                list(rgb_prof["transform"])[:6] if rgb_prof.get("transform") else None
            ),
            "agl_transform": (
                list(agl_prof["transform"])[:6] if agl_prof.get("transform") else None
            ),
        },
    }


# ---------------------------------------------------------------------------
# Quicklooks (visual alignment verification)
# ---------------------------------------------------------------------------


def save_quicklook(
    tile: TileData, out_png: Path, cls_ids: Optional[List[int]] = None
) -> None:
    """3-panel PNG: RGB | AGL (magma) | CLS (tab20). The human eye is the
    final alignment check: rooftops in AGL must sit on rooftops in RGB."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import colors as mcolors

    rgb, agl, cls = tile["rgb"], tile["agl"], tile["cls"]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.6), constrained_layout=True)
    axes[0].imshow(rgb)
    axes[0].set_title(f"{tile['meta']['stem']} RGB")
    im1 = axes[1].imshow(agl, cmap="magma")
    axes[1].set_title(f"AGL  min={agl.min():.2f}  max={agl.max():.2f}")
    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.02)
    if cls_ids is not None and len(cls_ids) > 0:
        cmap = mcolors.ListedColormap(
            plt.cm.tab20(np.linspace(0, 1, max(len(cls_ids), 2)))
        )
        bounds = list(cls_ids) + [max(cls_ids) + 1]
        norm = mcolors.BoundaryNorm(bounds, cmap.N)
        axes[2].imshow(cls, cmap=cmap, norm=norm, interpolation="nearest")
    else:
        axes[2].imshow(cls, cmap="tab20", interpolation="nearest")
    axes[2].set_title("CLS (raw ids — legend per report)")
    for ax in axes:
        ax.set_xticks([]), ax.set_yticks([])
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=130)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Writing (used from Phase 8 onward, provided now so tests cover it)
# ---------------------------------------------------------------------------


def write_tif_like(
    path: Path | str,
    array: np.ndarray,
    like_profile: dict,
    nodata: Optional[float] = None,
    dtype: str = "float32",
) -> None:
    """Write `array` [C,H,W] or [H,W] reusing the grid/CRS of `like_profile`.
    For DFC2019 Track-1 (no CRS) the output is likewise non-georeferenced —
    we never fabricate coordinates."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if array.ndim == 2:
        array = array[None, ...]
    prof = {
        "driver": "GTiff",
        "width": array.shape[2],
        "height": array.shape[1],
        "count": array.shape[0],
        "dtype": dtype,
        "crs": like_profile.get("crs"),
        "transform": like_profile.get("transform"),
        "nodata": nodata,
    }
    with rasterio.open(path, "w", **prof) as dst:
        dst.write(array.astype(dtype))


def load_json(path: Path | str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def dump_json(obj: dict, path: Path | str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=False)


def resolve_cache_dir(root, subdir: str | None = None) -> "Path":
    """Resolve the depth-cache directory across all layouts.

    Layouts (Phase 1+ of the GAMUS integration):
      <out_dir>/<tag>/*.npy                 legacy flat (pre-GAMUS)
      <out_dir>/<tag>/<dataset>/*.npy       NEW namespaced (dfc2019 | gamus)
    This helper returns the MODEL-TAGGED directory in every case; it accepts
    <out_dir> (auto-resolving the tag when unambiguous) or the tagged dir
    itself, fails loudly when ambiguous — never mix depth outputs across
    backbones.

    Usage: resolve_cache_dir(paths["depth_cache_dir"], args.cache_subdir)
    """
    from pathlib import Path as _P

    root = _P(root)
    if not root.exists():
        raise FileNotFoundError(f"depth cache root not found: {root}")
    if subdir:
        d = root / subdir
        if not d.is_dir():
            raise FileNotFoundError(f"--cache-subdir '{subdir}' not found under {root}")
        return d
    if list(root.glob("*.npy")):
        return root  # legacy flat model dir
    subs = sorted(d for d in root.iterdir() if d.is_dir())
    model_subs = [d for d in subs if d.name not in DATASET_NAMESPACES]
    if subs and not model_subs:
        # children are dataset namespaces -> root IS the model-tag dir
        return root
    if len(model_subs) == 1:
        return model_subs[0]
    if not subs:
        raise FileNotFoundError(f"no *.npy under {root} — run the depth command")
    raise FileNotFoundError(
        f"multiple model caches under {root}: {[d.name for d in model_subs]} — "
        "pass cache_subdir explicitly. Never mix depths across backbones."
    )


# ---------------------------------------------------------------------------
# Depth-cache entry paths (multi-dataset, Phase 1 of the GAMUS integration)
# ---------------------------------------------------------------------------

# Dataset namespace directory names (also used by resolve_cache_dir to tell
# "dataset namespace under a model dir" apart from "model tag under out_dir").
DATASET_NAMESPACES = ("dfc2019", "gamus", "mixed")

# Datasets that may exist under a pre-GAMUS (flat) cache layout.
_LEGACY_FLAT_DATASETS = ("dfc2019",)


def depth_npy_candidates(model_cache_dir, dataset: str, sample_id: str) -> List[Path]:
    """Candidate .npy paths for one depth-cache entry, NEW namespaced layout
    first, LEGACY flat layout second (dfc2019 only).

    NEW (Phase 1+):   <model_tag>/<dataset>/{sample_id}.npy
    LEGACY (pre-GAMUS): <model_tag>/{sample_id}.npy           (dfc2019 only)

    ``model_cache_dir`` is the MODEL-TAGGED directory (what
    resolve_cache_dir returns). Same model+tile ⇒ same content in both
    layouts; when both exist the first (namespaced) wins — writers refresh
    both only via an explicit --overwrite.
    """
    root = Path(model_cache_dir)
    cands = [root / dataset / f"{sample_id}.npy"]
    if dataset in _LEGACY_FLAT_DATASETS:
        cands.append(root / f"{sample_id}.npy")
    return cands
