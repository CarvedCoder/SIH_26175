"""Track-2 anchoring: AGL + ground elevation -> ABSOLUTE DSM.

Doctrine (frozen, from the campaign worklog):
    AGL is NOT an absolute DSM.  DSM = DTM + AGL.  Ground anchoring is an
    INDEPENDENT, NON-LEARNED arithmetic addition performed at inference
    time.  Nothing here trains, finetunes, or "improves" anything — it
    merely adds a terrain datum to a predicted height field, and every
    output/ledger line it touches is labeled:

        ANCHORED (not learned)

Rules encoded:
  * ``anchor_with_dem`` requires BOTH rasters to carry a CRS. DFC2019
    Track-1 imagery has none — if the image is not georeferenced we raise
    (extent is UNKNOWN; fabricating an alignment would be silent garbage).
  * The DEM is resampled to the IMAGE's grid with bilinear
    ``rasterio.warp.reproject``; extent/CRS mismatch raises, never warns.
  * ``anchor_with_constant`` adds a scalar ground elevation supplied by the
    user (e.g. a known local datum for a demo site).
  * Unknowns stay UNKNOWN: if CRS/units cannot be verified they are
    reported as UNKNOWN strings, never guessed.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import numpy as np

ANCHORED_LABEL = "ANCHORED (not learned)"


@dataclass
class AnchorResult:
    dsm: np.ndarray                 # absolute DSM, float32 [H,W], metres
    source: str                     # "dem:<path>" | "constant:<value>"
    dem_stats: Optional[Dict[str, float]] = None   # min/mean/max of resampled DEM

    @property
    def label(self) -> str:
        return ANCHORED_LABEL


def _require_crs(profile: dict, what: str) -> None:
    """Hard gate: anchoring only works on georeferenced inputs."""
    crs = profile.get("crs")
    if crs is None:
        raise ValueError(
            f"{what} has NO CRS — absolute anchoring is impossible. "
            "The image extent is UNKNOWN; aligning a DEM by guesswork would "
            "produce silently-wrong elevations. Re-run with a georeferenced "
            "GeoTIFF (Track 2) or use --ground-elev for a constant datum.")


def resample_dem_to_tile(dem_path: Path | str, tile_profile: dict) -> np.ndarray:
    """Bilinear-resample a DEM raster onto the image tile's exact grid.

    ``tile_profile`` must be a rasterio profile (dict with crs, transform,
    width, height). Raises ValueError on CRS absence or extent mismatch —
    partial overlaps are NOT tolerated (an anchored DSM is only meaningful
    when the DEM fully covers the image footprint).
    """
    import rasterio
    from rasterio.warp import Resampling, reproject

    dem_path = Path(dem_path)
    if not dem_path.exists():
        raise FileNotFoundError(f"DEM not found: {dem_path}")
    _require_crs(tile_profile, "target image")

    with rasterio.open(dem_path) as dem:
        _require_crs(dem.profile, f"DEM '{dem_path.name}'")
        dst = np.full((tile_profile["height"], tile_profile["width"]),
                      np.nan, dtype=np.float32)
        try:
            reproject(
                source=rasterio.band(dem, 1),
                destination=dst,
                src_transform=dem.transform,
                src_crs=dem.crs,
                src_nodata=dem.nodata,
                dst_transform=tile_profile["transform"],
                dst_crs=tile_profile["crs"],
                dst_nodata=np.nan,
                resampling=Resampling.bilinear,
            )
        except Exception as e:  # rasterio raises bare ValueError families
            raise ValueError(
                f"DEM->tile reprojection failed for '{dem_path.name}': {e}") from e

    if np.isnan(dst).any():
        raise ValueError(
            f"DEM '{dem_path.name}' does not fully cover the image footprint "
            f"({np.isnan(dst).mean():.1%} missing pixels) — refusing to anchor "
            "on a partial overlap.")
    return dst


def anchor_with_dem(agl: np.ndarray, dem_path: Path | str,
                    tile_profile: dict) -> AnchorResult:
    """DSM = AGL + resampled(DTM/DEM). The addition is exact and unlearned."""
    dem = resample_dem_to_tile(dem_path, tile_profile)
    if dem.shape != agl.shape:
        raise ValueError(
            f"resampled DEM shape {dem.shape} != AGL shape {agl.shape} "
            "(grid contract violated — this is a bug, not a data issue)")
    dsm = (agl.astype(np.float32) + dem).astype(np.float32)
    return AnchorResult(
        dsm=dsm,
        source=f"dem:{Path(dem_path).name}",
        dem_stats={"min": float(dem.min()), "mean": float(dem.mean()),
                   "max": float(dem.max())},
    )


def anchor_with_constant(agl: np.ndarray, ground_elev: float) -> AnchorResult:
    """DSM = AGL + constant datum (user-supplied, e.g. local ground level)."""
    if not np.isfinite(ground_elev):
        raise ValueError(f"ground_elev must be finite, got {ground_elev}")
    dsm = (agl.astype(np.float32) + float(ground_elev)).astype(np.float32)
    return AnchorResult(dsm=dsm, source=f"constant:{ground_elev:.3f}")


def anchor(agl: np.ndarray,
           dem_path: Optional[Path | str],
           ground_elev: Optional[float],
           tile_profile: Optional[dict] = None) -> Optional[AnchorResult]:
    """Dispatch: DEM wins, then constant, then None (relative DSM only)."""
    if dem_path is not None:
        if tile_profile is None:
            raise ValueError("--anchor-dem requires the input image's raster "
                             "profile (georeferenced raster expected).")
        return anchor_with_dem(agl, dem_path, tile_profile)
    if ground_elev is not None:
        return anchor_with_constant(agl, ground_elev)
    return None
