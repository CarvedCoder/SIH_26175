"""Overlapping tile windows + weighted stitching for the inference pipeline.

Why this module exists
-----------------------
``depthwizard/inference.py``'s ``mode == "tiles"`` path (and its two
siblings — the per-tile live-backbone pass and the per-tile Dn
normalization) previously walked a NON-overlapping grid
(``tile_bounds``: stride == tile_size) and wrote each tile's prediction
with a hard slice assignment (``out[y:y+TILE, x:x+TILE] = pred``). Two
independently-inferred tiles therefore never agreed at their shared
border, producing visible seams on any input larger than one tile
(>1024px on either side).

This module provides the three pieces a seam-free tiled pipeline needs:

    TilingConfig          single documented dataclass for every tiling
                           knob (mirrors depthwizard/postprocess/config.py's
                           PostProcessConfig convention — one dataclass,
                           one place, every field commented).
    iter_tile_windows()   full-coverage OVERLAPPING window generator
                           (stride = tile_size - overlap). Windows never
                           read past the source raster; only a source
                           smaller than tile_size needs padding (handled
                           the same way inference.py already pads
                           elsewhere: edge-replication, never invented
                           content).
    OverlapStitcher        accumulates weighted per-tile predictions into
                           one full-resolution array. Predictions near a
                           tile's center get a higher blend weight than
                           predictions near its border (smooth cosine
                           ramp over the overlap zone), NaN / masked-out
                           pixels contribute zero weight (never poison a
                           neighboring tile's valid prediction), and the
                           final division by the accumulated weight sum
                           renormalizes automatically at image borders
                           where only one tile ever contributes.

Relationship to src/tile_dem.py
--------------------------------
``src/tile_dem.py`` (Stage 5 of the standalone ``main.py`` CLI prototype)
already implements correct overlapping-window *start offsets* for
windowed GeoTIFF reads, and its own docstring explicitly says stitching
is out of scope for that module. The stride/edge-snapping algorithm
below is the same shape (deliberately — it is the correct, tested
approach), reimplemented here for in-memory numpy tiles (RGB + Dn
arrays) rather than on-disk rasterio windows, and paired with the
weighted stitcher that module never built. The two modules are not
merged because they serve different pipelines (see architecture report,
Section 1) with different I/O shapes.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

#: Training/inference tile size (kept in sync with inference.TILE).
DEFAULT_TILE_SIZE = 1024

#: Overlap between adjacent inference windows, in pixels. Large enough
#: that the cosine blend ramp (see blend_weight_window) has room to fully
#: transition from 0 -> 1 before the tile interior begins.
DEFAULT_OVERLAP = 256

#: Weight floor so no pixel is ever assigned exactly zero blend weight
#: (a hard zero would make that pixel's contribution vanish even when it
#: is the ONLY tile covering that location, e.g. at a scene's outer
#: border — see OverlapStitcher docstring).
_MIN_WEIGHT = 1e-3


@dataclass(frozen=True)
class TilingConfig:
    """All knobs for overlapping-window inference tiling (units in comments).

    Every tunable that influences the tiled inference path lives here —
    same rule ``depthwizard/postprocess/config.py`` documents for
    PostProcessConfig: no tiling magic numbers scattered through
    inference.py.
    """

    #: Edge length of one square inference window, in pixels. Must match
    #: the CalibrationNet's trained input resolution (depthwizard.inference.TILE)
    #: for crop/resize modes to stay valid; only the "tiles" mode consumes
    #: this dataclass, so it is independent of TILE at the type level but
    #: should be constructed with the same value in practice.
    tile_size: int = DEFAULT_TILE_SIZE

    #: Overlap between spatially adjacent windows, in pixels. 0 disables
    #: blending entirely (windows touch but never overlap — degenerates
    #: to the old non-overlapping behavior, kept as a valid config for
    #: tests/back-compat, not the default).
    overlap: int = DEFAULT_OVERLAP

    #: Blend-weight profile shape. "cosine": smooth half-cosine (Hann-style)
    #: ramp across the overlap zone — the "Gaussian/cosine-style weighting"
    #: the task calls for; chosen over a true Gaussian because a cosine
    #: ramp reaches exactly 1.0 at the tile interior and a documented
    #: floor at the tile edge, which is easier to reason about at exact
    #: overlap boundaries than a Gaussian's unbounded tail. "none": uniform
    #: weight 1.0 everywhere (no tapering) — use only for tests that need
    #: to reproduce the pre-overlap, non-blended numeric contract.
    blend: str = "cosine"

    def __post_init__(self) -> None:
        if self.tile_size <= 0:
            raise ValueError(f"tile_size must be positive, got {self.tile_size}")
        if self.overlap < 0:
            raise ValueError(f"overlap must be >= 0, got {self.overlap}")
        if self.overlap >= self.tile_size:
            raise ValueError(
                f"overlap ({self.overlap}) must be smaller than tile_size "
                f"({self.tile_size}) — otherwise windows never advance."
            )
        if self.blend not in ("cosine", "none"):
            raise ValueError(f"unknown blend '{self.blend}' (cosine|none)")

    @property
    def stride(self) -> int:
        """Pixel step between consecutive window starts along one axis."""
        return max(self.tile_size - self.overlap, 1)


@dataclass(frozen=True)
class TileWindow:
    """One inference window's placement within the full-resolution grid."""

    row_off: int  # top-left row in the SOURCE (unpadded) array
    col_off: int  # top-left col in the SOURCE (unpadded) array
    row_index: int  # grid position, for logging/diagnostics
    col_index: int
    valid_height: int  # rows actually available from the source at this offset
    valid_width: int  # cols actually available from the source at this offset
    tile_size: int  # window edge length this TileWindow was generated for

    @property
    def needs_padding(self) -> bool:
        """True when the source doesn't fully cover this window (only
        possible when height/width < tile_size — see iter_tile_windows)."""
        return self.valid_height < self.tile_size or self.valid_width < self.tile_size


def _axis_starts(dimension: int, tile_size: int, overlap: int) -> list[int]:
    """Full-coverage overlapping start offsets along one axis.

    Mirrors src/tile_dem.py's ``_tile_starts`` (see module docstring):
    stride = tile_size - overlap, starting at 0, advancing until the next
    start would run past the raster; the final start is then SNAPPED to
    ``dimension - tile_size`` so the last window's far edge exactly
    touches the raster boundary (no gap, no reading past the edge). When
    the whole axis is smaller than tile_size, a single start at 0 is
    returned — that window is padded up to tile_size by the caller.
    """
    if dimension <= tile_size:
        return [0]
    stride = max(tile_size - overlap, 1)
    max_start = dimension - tile_size
    starts = list(range(0, max_start + 1, stride))
    if starts[-1] != max_start:
        starts.append(max_start)
    return starts


def iter_tile_windows(
    height: int, width: int, cfg: TilingConfig = TilingConfig()
) -> list[TileWindow]:
    """Full-coverage overlapping windows over a (height, width) array.

    Every window is ``cfg.tile_size`` x ``cfg.tile_size``. When
    ``height``/``width`` >= ``tile_size``, every window is read entirely
    from within the source (``row_off + tile_size <= height`` always
    holds) — no padding, no synthetic border between real tiles. When an
    axis is SMALLER than tile_size, the single window for that axis has
    ``valid_height``/``valid_width`` < ``tile_size``; the caller pads the
    source up to tile_size (edge-replication, consistent with every
    other padding site in inference.py) before running inference, and
    OverlapStitcher only accumulates the ``valid_height``/``valid_width``
    region back into the output.
    """
    if height <= 0 or width <= 0:
        raise ValueError(f"height/width must be positive, got ({height}, {width})")

    row_starts = _axis_starts(height, cfg.tile_size, cfg.overlap)
    col_starts = _axis_starts(width, cfg.tile_size, cfg.overlap)

    windows: list[TileWindow] = []
    for ri, row_off in enumerate(row_starts):
        valid_h = min(cfg.tile_size, height - row_off)
        for ci, col_off in enumerate(col_starts):
            valid_w = min(cfg.tile_size, width - col_off)
            windows.append(
                TileWindow(
                    row_off=row_off,
                    col_off=col_off,
                    row_index=ri,
                    col_index=ci,
                    valid_height=valid_h,
                    valid_width=valid_w,
                    tile_size=cfg.tile_size,
                )
            )
    return windows


def _cosine_ramp_1d(length: int, taper: int) -> np.ndarray:
    """1-D weight profile: 0->1 half-cosine ramp over ``taper`` px at each
    end, 1.0 in the interior. Floors at ``_MIN_WEIGHT`` so no sample is
    ever assigned exactly zero weight (see OverlapStitcher)."""
    w = np.ones(length, dtype=np.float32)
    if taper <= 0:
        return w
    taper = min(taper, length // 2) if length > 1 else 0
    if taper <= 0:
        return w
    ramp = 0.5 - 0.5 * np.cos(np.linspace(0.0, np.pi, taper, dtype=np.float32))
    ramp = np.maximum(ramp, _MIN_WEIGHT).astype(np.float32)
    w[:taper] = ramp
    w[length - taper :] = ramp[::-1]
    return w


def blend_weight_window(tile_size: int, overlap: int, blend: str = "cosine") -> np.ndarray:
    """(tile_size, tile_size) float32 blend weights: 1.0 at the tile's
    center, tapering toward the border over ``overlap`` px so a pixel
    predicted near two tiles' shared boundary is trusted less than one
    predicted near either tile's center (Blueprint: "predictions near
    the center of an inference tile should have higher confidence than
    predictions near the tile border").

    ``blend="none"`` returns a uniform weight of 1.0 (no tapering) —
    used only to reproduce the pre-overlap, non-blended numeric contract
    in tests.
    """
    if blend == "none":
        return np.ones((tile_size, tile_size), dtype=np.float32)
    if blend != "cosine":
        raise ValueError(f"unknown blend '{blend}' (cosine|none)")
    row_w = _cosine_ramp_1d(tile_size, overlap)
    col_w = _cosine_ramp_1d(tile_size, overlap)
    return np.outer(row_w, col_w).astype(np.float32)


class OverlapStitcher:
    """Accumulates weighted, overlapping tile predictions into one
    full-resolution array.

        weighted_prediction_sum += prediction * weight
        weight_sum              += weight
        final_output = weighted_prediction_sum / weight_sum

    NaN-safety and nodata-safety: any pixel that is NaN in the prediction,
    or excluded by an optional per-tile ``valid_mask``, contributes ZERO
    weight at that pixel — it is neither summed into the numerator nor
    the denominator, so it can never poison a neighboring tile's valid
    prediction at the same location, and it never turns a valid
    neighbor's average into NaN. A pixel covered by zero valid tiles
    (all contributing tiles NaN/masked there) is reported as NaN in the
    final output — an honest "no prediction available" rather than a
    fabricated value.
    """

    def __init__(self, height: int, width: int, cfg: TilingConfig = TilingConfig()):
        self.height = height
        self.width = width
        self.cfg = cfg
        self._weight_template = blend_weight_window(cfg.tile_size, cfg.overlap, cfg.blend)
        self._sum = np.zeros((height, width), dtype=np.float64)
        self._wsum = np.zeros((height, width), dtype=np.float64)

    def add_tile(
        self,
        window: TileWindow,
        prediction: np.ndarray,
        valid_mask: np.ndarray | None = None,
    ) -> None:
        """Accumulate one tile's prediction.

        ``prediction`` must be ``(cfg.tile_size, cfg.tile_size)`` (the
        model's raw tile output, including any padded region — this
        method crops to ``window.valid_height/valid_width`` and to the
        output bounds itself, so callers never need to pre-crop).
        ``valid_mask`` (optional), same shape, True where the prediction
        is trustworthy (e.g. not a GeoTIFF nodata pixel); pixels outside
        the mask get zero weight exactly like NaN predictions.
        """
        if prediction.shape != (self.cfg.tile_size, self.cfg.tile_size):
            raise ValueError(
                f"prediction shape {prediction.shape} != tile_size "
                f"{(self.cfg.tile_size, self.cfg.tile_size)}"
            )
        vh = min(window.valid_height, self.height - window.row_off)
        vw = min(window.valid_width, self.width - window.col_off)
        if vh <= 0 or vw <= 0:
            return

        y0, x0 = window.row_off, window.col_off
        pred = prediction[:vh, :vw]
        weight = self._weight_template[:vh, :vw].copy()

        finite = np.isfinite(pred)
        if valid_mask is not None:
            finite &= valid_mask[:vh, :vw]
        weight = np.where(finite, weight, 0.0)
        pred_safe = np.where(finite, pred, 0.0).astype(np.float64)

        self._sum[y0 : y0 + vh, x0 : x0 + vw] += pred_safe * weight
        self._wsum[y0 : y0 + vh, x0 : x0 + vw] += weight

    def finalize(self) -> np.ndarray:
        """Full-resolution [height, width] float32 stitched output.
        Pixels covered by zero valid tiles are NaN (never fabricated)."""
        out = np.full((self.height, self.width), np.nan, dtype=np.float32)
        covered = self._wsum > 0
        out[covered] = (self._sum[covered] / self._wsum[covered]).astype(np.float32)
        return out