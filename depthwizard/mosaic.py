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
from rasterio.transform import Affine, from_origin

#: Relative tolerance when comparing pixel sizes across inputs (floats in
#: geotransforms are never bit-identical across exports; 1e-6 is tight
#: enough that a genuinely different resolution cannot slip through).
_RESOLUTION_RTOL = 1e-6

#: Feathering (overlap blending): each source's contribution is weighted by
#: a Gaussian-blurred footprint mask, so inside an overlap band the blend
#: ramps smoothly from one source's values to the other's instead of the
#: hard "last tile wins" cut rasterio.merge applies by default — that hard
#: cut showed as a visible seam line in the projected RGB texture whenever
#: adjacent survey tiles' radiometry differed slightly. Sigma is a fraction
#: of the source's smaller pixel dimension, clamped to sane absolute bounds.
_FEATHER_SIGMA_FRACTION = 0.05
_FEATHER_SIGMA_MIN_PX = 8.0
_FEATHER_SIGMA_MAX_PX = 96.0


def _box_blur(a: np.ndarray, radius: int) -> np.ndarray:
    """Edge-padded 2-D box blur (separable, O(N) via cumsum), 3 passes —
    the standard box-blur approximation of a Gaussian with sigma≈radius."""
    out = a.astype(np.float64)
    for _ in range(3):
        for axis in (0, 1):
            n = out.shape[axis]
            if n == 0 or radius <= 0:
                continue
            ap = np.concatenate(
                [np.repeat(out.take([0], axis=axis), radius, axis=axis),
                 out,
                 np.repeat(out.take([-1], axis=axis), radius, axis=axis)],
                axis=axis,
            )
            c = np.cumsum(ap, axis=axis)
            zeros_shape = list(c.shape)
            zeros_shape[axis] = 1
            c = np.concatenate([np.zeros(zeros_shape), c], axis=axis)
            win = 2 * radius + 1
            out = (np.take(c, np.arange(win, win + n), axis=axis)
                   - np.take(c, np.arange(0, n), axis=axis)) / win
    return out


def _gaussian_feather(mask: np.ndarray, sigma: float) -> np.ndarray:
    """Gaussian-blur a boolean footprint mask (float32, same shape).

    The mask is zero-padded by 3σ before blurring: an edge-replicated
    blur of a mask that fills its own array (the common all-valid
    footprint) would never ramp at the source's edge — and the edge ramp
    is exactly the feather.
    """
    pad = int(np.ceil(3.0 * sigma))
    padded = np.zeros(
        (mask.shape[0] + 2 * pad, mask.shape[1] + 2 * pad), dtype=np.float64
    )
    padded[pad : pad + mask.shape[0], pad : pad + mask.shape[1]] = mask
    blurred = _box_blur(padded, max(1, int(round(sigma))))
    return blurred[pad : pad + mask.shape[0], pad : pad + mask.shape[1]].astype(np.float32)


def _feathered_merge(
    datasets: list[rasterio.DatasetReader],
    height: int,
    width: int,
    transform: Affine,
    nodata: float,
) -> np.ndarray:
    """Blend all sources onto the merged (height, width) grid, feathered.

    Each source contributes ``value * weight`` where ``weight`` is its
    Gaussian-feathered footprint mask (1.0 in the interior, ramping to 0
    over ~2σ at the source's own edge). The final pixel value is the
    weighted mean over contributing sources:

        out = Σ(source * weight) / Σ(weight)

    Consequences that matter:
      * single-source regions normalise back to the EXACT source value
        (w·v / w = v) — outer mosaic borders and non-overlapping edges
        are pixel-perfect, never darkened;
      * overlap bands ramp smoothly between the two sources — no seam line;
      * uncovered pixels keep zero total weight and carry ``nodata`` —
        coverage gaps stay explicit, never interpolated.
    """
    count = datasets[0].count
    acc = np.zeros((count, height, width), dtype=np.float64)
    wsum = np.zeros((height, width), dtype=np.float64)

    for ds in datasets:
        # Pixel placement of this source on the merged grid (the sources
        # share one validated resolution, so bounds → pixel offsets are exact
        # up to sub-pixel rounding of the geotransform).
        col0, row0 = ~transform * (ds.bounds.left, ds.bounds.top)
        c0, r0 = int(round(col0)), int(round(row0))
        h, w = ds.height, ds.width

        src = ds.read().astype(np.float64)
        if nodata is not None:
            valid = np.all(src != nodata, axis=0).astype(np.float32)
        else:
            valid = np.ones((h, w), dtype=np.float32)

        sigma = float(
            np.clip(
                min(h, w) * _FEATHER_SIGMA_FRACTION,
                _FEATHER_SIGMA_MIN_PX,
                _FEATHER_SIGMA_MAX_PX,
            )
        )
        # Weight from the FULL footprint (zero-padded blur), so a source
        # extending past the merged grid doesn't get a spurious ramp at
        # the grid border. The blurred step sits at 0.5 exactly ON the
        # footprint edge — remapping (blur − 0.5)·2 makes the weight zero
        # AT the edge and continuous on both sides (a plain blur×valid
        # product would jump to ~0.5 just inside the edge — a hard step).
        weight_full = np.clip(
            (_gaussian_feather(valid > 0, sigma) - 0.5) * 2.0, 0.0, 1.0
        ) * valid

        # Copy the grid-intersecting sub-window.
        sr, sc = max(0, -r0), max(0, -c0)
        tr, tc = max(0, r0), max(0, c0)
        hh = min(h - sr, height - tr)
        ww = min(w - sc, width - tc)
        if hh <= 0 or ww <= 0:
            continue

        weight = weight_full[sr : sr + hh, sc : sc + ww]
        wsum[tr : tr + hh, tc : tc + ww] += weight
        acc[:, tr : tr + hh, tc : tc + ww] += (
            src[:, sr : sr + hh, sc : sc + ww] * weight[None, :, :]
        )

    out = np.full((count, height, width), nodata, dtype=np.float64)
    covered = wsum > 0
    for k in range(count):
        out[k][covered] = acc[k][covered] / wsum[covered]
    return out


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
    inputs' bounds at the (validated, shared) input resolution. Overlap
    bands are FEATHERED (edge-distance weighted blend, see
    :func:`_feathered_merge`) instead of rasterio.merge's hard last-wins
    cut, so the joined part of the mosaic shows a smooth transition rather
    than a seam. Where a source does not cover the grid, the output
    carries ``nodata`` and the profile reports it — coverage gaps are
    honest, never interpolated.
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
        nodata = first.nodata
        if nodata is None:
            # Integer rasters default to 0 fill; floats document an explicit
            # -9999.0 output nodata so uncovered pixels are detectable.
            nodata = (
                np.iinfo(first.dtypes[0]).max
                if np.issubdtype(np.dtype(first.dtypes[0]), np.integer)
                else -9999.0
            )

        # Union grid: shared resolution (validated) × union of bounds.
        res = abs(first.transform.a)
        left = min(ds.bounds.left for ds in datasets)
        right = max(ds.bounds.right for ds in datasets)
        top = max(ds.bounds.top for ds in datasets)
        bottom = min(ds.bounds.bottom for ds in datasets)
        width = max(1, int(round((right - left) / res)))
        height = max(1, int(round((top - bottom) / res)))
        transform = from_origin(left, top, res, res)

        merged = _feathered_merge(datasets, height, width, transform, nodata)

        # Cast back to the input dtype (round for integers — the feathered
        # mean is fractional and a truncating cast would bias it dark).
        if np.issubdtype(np.dtype(first.dtypes[0]), np.integer):
            merged = np.rint(merged)
        merged = merged.astype(first.dtypes[0])

        profile = {
            "driver": "GTiff",
            "height": height,
            "width": width,
            "count": first.count,
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
        return MosaicResult(data=merged, profile=profile, sources=sources)
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
