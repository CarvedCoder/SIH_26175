"""Absolute DSM construction + machine-readable provenance (Parts B/C).

    RGB GeoTIFF -> height model -> AGL prediction
                                  |
    reference DEM ----align------ +
                                  v
    DSM_absolute = DEM_reference + AGL_prediction

Contract (frozen by model_tests/test_absolute_dsm.py):

  * Input CRS, affine transform and dimensions are preserved EXACTLY.
  * The DEM is reprojected onto the IMAGE grid (bilinear) with a hard
    full-coverage gate — partial coverage raises
    DWStatusError(DEM_COVERAGE_INSUFFICIENT), never silent clipping.
  * The result is called "absolute_dsm" ONLY when an external elevation
    reference (a real DEM) was used. A user-supplied constant datum is
    "anchored_constant_dsm" (honest: datum asserted by the user, not an
    external reference). No reference at all -> "relative_height".
  * Provenance records what is known; unknowns are None — never invented.
  * Non-georeferenced inputs (PNG/JPG): DEM anchoring is impossible
    (CRS_REQUIRED); constant anchoring stays allowed (arithmetic, no
    grid alignment involved); without anchoring the product is a
    relative height map and nothing pretends otherwise.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import numpy as np

from .anchoring import ANCHORED_LABEL, AnchorResult, resample_dem_to_tile
from .statuses import CRS_REQUIRED, DWStatusError

# output_type values (Part D / Part O — the relative vs absolute frontier)
OUTPUT_RELATIVE_HEIGHT = "relative_height"
OUTPUT_ABSOLUTE_DSM = "absolute_dsm"
OUTPUT_ANCHORED_CONSTANT_DSM = "anchored_constant_dsm"

ABSOLUTE_DSM_FORMULA = "DSM_absolute = DEM_reference + AGL_prediction"
DEFAULT_RESAMPLING = "bilinear"


@dataclass
class AbsoluteDSMResult:
    dsm: np.ndarray  # [H,W] float32 metres
    output_type: str  # one of OUTPUT_*
    absolute_reference_available: bool
    anchored: Optional[AnchorResult]  # kept for the existing payload contract
    provenance: Dict  # machine-readable, JSON-serializable (unknown -> None)

    @property
    def label(self) -> str:
        return ANCHORED_LABEL if self.anchored is not None else "RELATIVE (AGL)"


def scene_gsd(scene_profile: Dict) -> Optional[float]:
    """Nominal scene GSD in metres (max of x/y pixel size), None if unknown."""
    from .geo import pixel_size_metres

    ps = pixel_size_metres(scene_profile.get("_crs_obj"), scene_profile.get("_transform_obj"))
    if ps is None:
        return None
    return float(max(abs(ps[0]), abs(ps[1])))


def build_absolute_dsm(
    agl: np.ndarray,
    scene_profile: Dict,
    dem_result,  # dem_provider.DEMResult | None
    ground_elev: Optional[float] = None,
    *,
    height_model: Optional[str] = None,
    height_model_version: Optional[str] = None,
    calibration_enabled: Optional[bool] = None,
    calibration_model: Optional[str] = None,
    height_semantics: str = "AGL",
    resampling: str = DEFAULT_RESAMPLING,
) -> AbsoluteDSMResult:
    """Combine an AGL prediction with an elevation reference (if any).

    ``scene_profile`` is the read_image() profile (carrying _crs_obj /
    _transform_obj). ``dem_result`` is a dem_provider.DEMResult or None.
    Exactly one of dem_result / ground_elev may anchor; both None keeps
    the relative product. Raises DWStatusError on alignment failure.
    """
    from .geo import pixel_size_metres

    crs = scene_profile.get("_crs_obj")
    transform = scene_profile.get("_transform_obj")
    georef = crs is not None

    if dem_result is not None:
        if not georef:
            raise DWStatusError(
                CRS_REQUIRED,
                "DEM anchoring requires a georeferenced input — the scene "
                "footprint is UNKNOWN without a CRS and no CRS is invented "
                "here. Use a GeoTIFF, or --ground-elev for a constant datum.",
            )
        tile_profile = {
            "crs": crs,
            "transform": transform,
            "height": agl.shape[0],
            "width": agl.shape[1],
        }
        try:
            dem_grid = resample_dem_to_tile(dem_result.mosaic_path, tile_profile)
        except ValueError as exc:
            # anchoring raises ValueError on partial coverage / CRS absence;
            # re-raise with the machine-readable code attached.
            raise DWStatusError(
                "DEM_COVERAGE_INSUFFICIENT",
                str(exc),
            ) from exc
        if dem_grid.shape != agl.shape:  # belt+braces: no broadcasting, ever
            raise DWStatusError(
                "GRID_MISMATCH",
                f"resampled DEM shape {dem_grid.shape} != AGL shape "
                f"{agl.shape} — grid contract violated",
            )
        dsm = (agl.astype(np.float32) + dem_grid).astype(np.float32)
        anchored = AnchorResult(
            dsm=dsm,
            source=f"dem:{Path(dem_result.mosaic_path).name}",
            dem_stats={
                "min": float(dem_grid.min()),
                "mean": float(dem_grid.mean()),
                "max": float(dem_grid.max()),
            },
        )
        provenance = _provenance(
            dem_result=dem_result,
            scene_crs=str(crs),
            scene_gsd_m=scene_gsd(scene_profile),
            resampling=resampling,
            height_model=height_model,
            height_model_version=height_model_version,
            calibration_enabled=calibration_enabled,
            calibration_model=calibration_model,
            height_semantics=height_semantics,
            scene_bounds=_bounds_of(crs, transform, agl.shape),
            scene_transform=list(transform)[:6] if transform is not None else None,
            scene_shape=[int(agl.shape[0]), int(agl.shape[1])],
            dem_grid_stats=anchored.dem_stats,
        )
        return AbsoluteDSMResult(
            dsm=dsm,
            output_type=OUTPUT_ABSOLUTE_DSM,
            absolute_reference_available=True,
            anchored=anchored,
            provenance=provenance,
        )

    if ground_elev is not None:
        if not np.isfinite(ground_elev):
            raise DWStatusError(
                "INVALID_GEOREFERENCE",
                f"ground_elev must be finite, got {ground_elev}",
            )
        from .anchoring import anchor_with_constant

        anchored = anchor_with_constant(agl, float(ground_elev))
        provenance = _provenance(
            dem_result=None,
            scene_crs=str(crs) if georef else None,
            scene_gsd_m=scene_gsd(scene_profile),
            resampling=None,
            height_model=height_model,
            height_model_version=height_model_version,
            calibration_enabled=calibration_enabled,
            calibration_model=calibration_model,
            height_semantics=height_semantics,
            scene_bounds=_bounds_of(crs, transform, agl.shape),
            scene_transform=list(transform)[:6] if transform is not None else None,
            scene_shape=[int(agl.shape[0]), int(agl.shape[1])],
            constant_datum_m=float(ground_elev),
        )
        return AbsoluteDSMResult(
            dsm=anchored.dsm,
            output_type=OUTPUT_ANCHORED_CONSTANT_DSM,
            # a user-asserted constant is NOT an externally referenced DEM:
            absolute_reference_available=False,
            anchored=anchored,
            provenance=provenance,
        )

    provenance = _provenance(
        dem_result=None,
        scene_crs=str(crs) if georef else None,
        scene_gsd_m=scene_gsd(scene_profile),
        resampling=None,
        height_model=height_model,
        height_model_version=height_model_version,
        calibration_enabled=calibration_enabled,
        calibration_model=calibration_model,
        height_semantics=height_semantics,
        scene_bounds=_bounds_of(crs, transform, agl.shape),
        scene_transform=list(transform)[:6] if transform is not None else None,
        scene_shape=[int(agl.shape[0]), int(agl.shape[1])],
    )
    return AbsoluteDSMResult(
        dsm=agl.astype(np.float32),
        output_type=OUTPUT_RELATIVE_HEIGHT,
        absolute_reference_available=False,
        anchored=None,
        provenance=provenance,
    )


def _bounds_of(crs, transform, shape) -> Optional[list]:
    """(west, south, east, north) of the scene, or None when unprojectable."""
    if crs is None or transform is None:
        return None
    try:
        from rasterio.transform import array_bounds

        w, s, e, n = array_bounds(shape[0], shape[1], transform)
        return [float(w), float(s), float(e), float(n)]
    except Exception:
        return None


def _provenance(
    *,
    dem_result,
    scene_crs: Optional[str],
    scene_gsd_m: Optional[float],
    resampling: Optional[str],
    height_model: Optional[str],
    height_model_version: Optional[str],
    calibration_enabled: Optional[bool],
    calibration_model: Optional[str],
    height_semantics: str,
    scene_bounds: Optional[list],
    scene_transform: Optional[list],
    scene_shape: list,
    dem_grid_stats: Optional[Dict] = None,
    constant_datum_m: Optional[float] = None,
) -> Dict:
    """The Part-C provenance block. Unknowns are None, never invented."""
    if dem_result is not None:
        dem_block = dem_result.provenance()
        alignment_block = {
            "resampling": resampling,
            "dem_grid_stats": dem_grid_stats,
            "formula": ABSOLUTE_DSM_FORMULA,
        }
        dem_crs = dem_block.get("dem_crs")
        if dem_crs is None:
            dem_block["dem_crs"] = None
        if dem_result.crs is not None and scene_crs is not None:
            alignment_block["scene_dem_crs_equal"] = (dem_crs == scene_crs)
        else:
            alignment_block["scene_dem_crs_equal"] = None
        vertical_reference = dem_block.get("dem_vertical_reference")
    else:
        dem_block = {
            "dem_source": None,
            "dem_identifier": None,
            "dem_crs": None,
            "dem_resolution_m": None,
            "dem_vertical_reference": None,
            "dem_provider": None,
            "dem_tiles": [],
            "dem_acquisition_timestamp_utc": None,
            "dem_synthetic": False,
        }
        alignment_block = {
            "resampling": None,
            "dem_grid_stats": None,
            "formula": None,
            "scene_dem_crs_equal": None,
        }
        vertical_reference = None
    if constant_datum_m is not None:
        alignment_block["constant_datum_m"] = constant_datum_m
        alignment_block["formula"] = "DSM = constant_ground_elev + AGL_prediction"

    return {
        "height_model": height_model,
        "height_model_version": height_model_version,
        "calibration_enabled": calibration_enabled,
        "calibration_model": calibration_model,
        "height_units": "meters",
        "height_semantics": height_semantics,
        **dem_block,
        "scene_crs": scene_crs,
        "scene_gsd_m": scene_gsd_m,
        "scene_bounds": scene_bounds,
        "scene_transform": scene_transform,
        "scene_shape": scene_shape,
        "alignment": alignment_block,
        "vertical_reference": vertical_reference,
    }


def write_provenance(provenance: Dict, out_dir: Path) -> Path:
    """Persist absolute_dsm_provenance.json (best-effort provenance file)."""
    import json

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    p = out_dir / "absolute_dsm_provenance.json"
    with open(p, "w", encoding="utf-8") as f:
        json.dump(provenance, f, indent=2)
    return p


def payload_output_fields(result: AbsoluteDSMResult) -> Dict:
    """The payload['meta'] additions exposing Part D to API/UI consumers."""
    return {
        "output_type": result.output_type,
        "absolute_reference_available": result.absolute_reference_available,
        "absolute_dsm_available": result.output_type == OUTPUT_ABSOLUTE_DSM,
        "provenance": result.provenance,
    }


def maybe_acquire_reference_dem(
    anchor_dem: Optional[Path | str],
    dem_provider: Optional[str],
    dem_cache_dir: Optional[Path | str],
    scene_profile: Dict,
    scene_shape,
    *,
    allow_mock: bool = False,
):
    """Resolve the reference DEM for a scene (or None — stay relative).

    ``anchor_dem`` (manual path) wins; otherwise the named automatic
    provider runs. Auto-retrieval failures of the DEM_UNAVAILABLE kind
    are printed LOUDLY and degrade to the relative product (Part A rule
    11: never fake data, never fail the whole workflow) — every other
    status (e.g. CRS_REQUIRED on a non-georeferenced input) propagates.

    Returns (DEMResult | None, dem_provider_requested: str | None).
    """
    from .dem_provider import resolve_reference_dem
    from .statuses import DEM_UNAVAILABLE

    if anchor_dem is None and dem_provider in (None, "", "none"):
        return None, None

    crs = scene_profile.get("_crs_obj")
    transform = scene_profile.get("_transform_obj")
    bounds = _bounds_of(crs, transform, scene_shape)
    gsd = scene_gsd(scene_profile)
    try:
        result = resolve_reference_dem(
            anchor_dem,
            None if anchor_dem is not None else dem_provider,
            crs,
            bounds if bounds is not None else (0.0, 0.0, 0.0, 0.0),
            dem_cache_dir or (Path("outputs") / "dem_cache"),
            scene_gsd_m=gsd,
            allow_mock=allow_mock,
        )
    except DWStatusError as exc:
        if exc.code == DEM_UNAVAILABLE and bounds is not None:
            # explicit, loud degradation to the relative product — the
            # absolute claim is silently DROPPED, never faked.
            print(
                f"[warn] [{exc.code}] {exc.detail}\n"
                "[warn] continuing with the RELATIVE (AGL) product — "
                "output will be labeled relative_height, NOT absolute DSM."
            )
            return None, dem_provider
        raise
    if result is not None:
        print(
            f"[dem] reference DEM resolved: provider={result.provider} "
            f"source={result.dem_source} crs={result.crs} "
            f"resolution={result.resolution_m} "
            f"vertical_reference={result.vertical_reference or 'unknown'}"
        )
    return result, dem_provider
