"""GeoTIFF mosaic stage: adjacent survey rasters -> one contiguous raster.

Why this module exists
-----------------------
Survey datasets sometimes arrive as spatially adjacent GeoTIFFs
(``survey_01.tif``, ``survey_02.tif``, ...) that together cover one area.
The scene pipeline historically accepted exactly ONE input raster, so such
datasets could not be processed as a unit. This module adds the missing
stage:

    MULTIPLE GEOTIFFS -> CRS VALIDATION -> RESOLUTION VALIDATION
    -> MERGE (rasterio.merge) -> NODATA HANDLING -> SINGLE RASTER

Honesty rules (mirroring depthwizard.geo):
    * every input must be genuinely georeferenced (a real CRS) — pixel-space
      PNG/JPG images are rejected rather than mosaicked by raw pixel
      coordinates;
    * all inputs must share one CRS and one ground resolution — inputs are
      NEVER silently resized/resampled to line up (a resolution mismatch is
      an error, not something to paper over);
    * inputs must actually touch or overlap (an adjacency check rejects
      scattered tiles that would produce a mosaic full of nodata gaps);
    * nodata: where sources disagree about coverage, the merge fills with
      the documented output nodata value and the profile reports it.

Relationship to src/tile_dem.py
--------------------------------
The standalone CLI prototype tiles ONE raster into windowed reads; it is
not a mosaic stage. depthwizard.geo.discover_tiles pairs training
(RGB/AGL/CLS) dataset triples, not survey inputs. Neither serves this
purpose, so this module is genuinely new functionality — it deliberately
builds on rasterio.merge (already a rasterio capability, no new dependency)
instead of reimplementing alignment.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import rasterio
from rasterio.crs import CRS
from rasterio.merge import merge as _rio_merge
from rasterio.transform import Affine

#: Relative tolerance when comparing pixel sizes across inputs (floats in
#: geotransforms are never bit-identical across exports; 1e-6 is tight
#: enough that a genuinely different resolution cannot slip through).
_RESOLUTION_RTOL = 1e-6


class MosaicError(ValueError):
    """Raised when inputs cannot be mosaicked into one contiguous raster.

    Subclasses ValueError so callers that catch ValueError (e.g. API
    validation layers) handle it uniformly.
    """


@dataclass(frozen=True)
class MosaicResult:
    """One merged, contiguous raster plus the profile needed to write it.

    ``data`` is [C, H, W] in the inputs' common dtype; ``profile`` is a
    complete rasterio writing profile (GTiff, crs, transform, nodata) for
    the merged grid; ``sources`` records each input's shape/CRS/bounds for
    logging and honest reporting.
    """

    data: np.ndarray
    profile: dict
    sources: list[dict]


def _validate_inputs(datasets: list[rasterio.DatasetReader], names: list[str]) -> None:
    """Shared CRS / resolution / band / adjacency validation.

    Raises MosaicError with an explicit message on the first violated
    precondition — every failure mode the merge step cannot decide for us.
    """
    if len(datasets) < 2:
        raise MosaicError(
            "mosaic needs at least 2 input rasters (a single file needs no mosaic)"
        )

    first = datasets[0]
    for i, ds in enumerate(datasets):
        name = names[i]
        if ds.crs is None:
            raise MosaicError(
                f"'{name}' has no CRS — pixel-space images (PNG/JPG) cannot be "
                "mosaicked; only genuinely georeferenced rasters can"
            )
        if i > 0 and not _same_crs(ds.crs, first.crs):
            raise MosaicError(
                f"CRS mismatch: '{name}' is {ds.crs.to_string()} but "
                f"'{names[0]}' is {first.crs.to_string()} — reproject the "
                "inputs to a common CRS first (never merged blind)"
            )
        if ds.count != first.count:
            raise MosaicError(
                f"band count mismatch: '{name}' has {ds.count} bands, "
                f"'{names[0]}' has {first.count}"
            )

        if i > 0 and not _same_resolution(ds.transform, first.transform):
            raise MosaicError(
                f"resolution mismatch: '{name}' is "
                f"({abs(ds.transform.a):.6g}, {abs(ds.transform.e):.6g}) "
                f"per pixel but '{names[0]}' is "
                f"({abs(first.transform.a):.6g}, {abs(first.transform.e):.6g}) "
                "— inputs are never silently resized to line up"
            )

    _validate_adjacency(datasets, names)


def _same_crs(a: CRS, b: CRS) -> bool:
    """CRS equality robust to formatting differences (EPSG codes vs WKT)."""
    return bool(a == b or a.equals(b))


def _same_resolution(t1: Affine, t2: Affine) -> bool:
    """True when two transforms describe the same ground resolution."""
    return bool(
        np.isclose(abs(t1.a), abs(t2.a), rtol=_RESOLUTION_RTOL, atol=0)
        and np.isclose(abs(t1.e), abs(t2.e), rtol=_RESOLUTION_RTOL, atol=0)
    )


def _bounds_overlap(a, b) -> bool:
    """True when two rasterio BoundingBoxes overlap OR share an exact edge
    (edge-touching tiles form a contiguous mosaic; only real gaps reject)."""
    return not (
        a.right < b.left
        or b.right < a.left
        or a.top < b.bottom
        or b.top < a.bottom
    )


def _validate_adjacency(datasets: list[rasterio.DatasetReader], names: list[str]) -> None:
    """Every input must touch or overlap at least one other input.

    Scattered tiles (no shared edges anywhere) would merge into a raster
    dominated by fabricated nodata gaps — rejected instead of merged.
    """
    for i, ds in enumerate(datasets):
        if not any(
            _bounds_overlap(ds.bounds, other.bounds)
            for j, other in enumerate(datasets)
            if j != i
        ):
            raise MosaicError(
                f"'{names[i]}' does not touch or overlap any other input — "
                "scattered tiles would produce a mosaic full of nodata gaps"
            )


def mosaic_rasters(paths: list[Path | str]) -> MosaicResult:
    """Merge spatially adjacent GeoTIFFs into one contiguous raster.

    Returns a :class:`MosaicResult`; the merged grid spans the union of the
    inputs' bounds at the (validated, shared) input resolution. Where a
    source does not cover the grid, the output carries ``nodata`` and the
    profile reports it — coverage gaps are honest, never interpolated.
    """
    paths = [Path(p) for p in paths]
    names = [p.name for p in paths]

    datasets: list[rasterio.DatasetReader] = []
    try:
        for p in paths:
            if not p.is_file():
                raise MosaicError(f"input raster not found: '{p.name}'")
            try:
                datasets.append(rasterio.open(p))
            except Exception as exc:
                raise MosaicError(
                    f"'{p.name}' could not be opened as a raster ({exc})"
                ) from exc

        _validate_inputs(datasets, names)

        first = datasets[0]
        merged, transform = _rio_merge(datasets)
        nodata = first.nodata
        if nodata is None:
            # Integer rasters default to 0 fill; floats document an explicit
            # -9999.0 output nodata so uncovered pixels are detectable.
            nodata = (
                np.iinfo(first.dtypes[0]).max
                if np.issubdtype(np.dtype(first.dtypes[0]), np.integer)
                else -9999.0
            )

        profile = {
            "driver": "GTiff",
            "height": merged.shape[1],
            "width": merged.shape[2],
            "count": merged.shape[0],
            "dtype": first.dtypes[0],
            "crs": first.crs,
            "transform": transform,
            "nodata": nodata,
        }
        sources = [
            {
                "name": names[i],
                "width": ds.width,
                "height": ds.height,
                "crs": ds.crs.to_string(),
                "bounds": [ds.bounds.left, ds.bounds.bottom, ds.bounds.right, ds.bounds.top],
            }
            for i, ds in enumerate(datasets)
        ]
        return MosaicResult(data=merged.astype(first.dtypes[0], copy=False), profile=profile, sources=sources)
    finally:
        for ds in datasets:
            ds.close()


def write_mosaic(result: MosaicResult, out_path: Path) -> Path:
    """Write a :class:`MosaicResult` to disk as a GeoTIFF (CRS/transform
    preserved — the merged raster is a first-class georeferenced input)."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(out_path, "w", **result.profile) as dst:
        dst.write(result.data)
    return out_path
