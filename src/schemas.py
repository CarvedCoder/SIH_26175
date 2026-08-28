"""Shared data contracts passed between pipeline stages.

Every stage function returns one of these dataclasses instead of printing
its results and exiting. main.py reads the returned object and passes the
relevant field(s) into the next stage, per the pipeline's input/output
contract.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional


@dataclass
class GeoTiffMetadata:
    """Result of inspect_geotiff(). Read-only description of the source file."""

    path: Path
    width: int
    height: int
    band_count: int
    dtypes: tuple[str, ...]
    crs: Optional[str]
    transform: tuple[float, float, float, float, float, float]
    bounds: tuple[float, float, float, float]
    resolution: tuple[float, float]
    nodata: Optional[float]
    total_pixels: int
    nodata_pixel_count: int
    nan_pixel_count: int
    inf_pixel_count: int
    valid_pixel_count: int
    elevation_min: Optional[float]
    elevation_max: Optional[float]
    elevation_mean: Optional[float]


@dataclass
class CleaningResult:
    """Result of clean_dem()."""

    output_path: Path
    invalid_pixel_count: int
    nan_pixel_count: int
    inf_pixel_count: int
    nodata_pixel_count: int
    total_pixels: int
    valid_pixel_count: int
    nodata_value_used: float
    notes: tuple[str, ...] = field(default_factory=tuple)


@dataclass
class NormalizationParams:
    """Recorded parameters needed to reproduce or invert normalization."""

    method: str
    source_min: float
    source_max: float
    nodata_value: float
    mean: Optional[float] = None
    std: Optional[float] = None


@dataclass
class PreprocessResult:
    """Result of preprocess_dem()."""

    output_path: Path
    params: NormalizationParams
    normalized_min: float
    normalized_max: float
    valid_pixel_count: int


@dataclass
class ValidationResult:
    """Result of visualize_dem()."""

    figure_path: Optional[Path]
    valid_pixel_count: int
    invalid_pixel_count: int
    data_min: float
    data_max: float
    data_mean: float


@dataclass
class TileInfo:
    """Metadata for one generated tile, sufficient for future stitching.

    Coordinates are all in the *source* raster's pixel grid (post any
    reflect-padding is tracked separately via `padding`), so a future
    stitching module can place every tile back into a full-scene mosaic
    without needing to parse filenames.
    """

    tile_path: Path
    row_index: int
    col_index: int
    row_offset: int
    col_offset: int
    width: int
    height: int
    valid_width: int
    valid_height: int
    padding: tuple[int, int, int, int]  # (top, bottom, left, right) pixels
    pad_mode_used: str
    transform: tuple[float, float, float, float, float, float]
    source_bounds: tuple[float, float, float, float]
    overlap: int


@dataclass
class TilingResult:
    """Result of tile_dem()."""

    output_dir: Path
    tile_size: int
    overlap: int
    tiles: list[TileInfo]
    grid_rows: int
    grid_cols: int


@dataclass
class TileVerificationEntry:
    tile_path: Path
    ok: bool
    reason: Optional[str]


@dataclass
class VerificationResult:
    """Result of verify_tiles() (and reused for cleaned/preprocessed checks)."""

    all_ok: bool
    checked_count: int
    failed_count: int
    entries: list[TileVerificationEntry]


@dataclass
class ModelInput:
    """Result of prepare_model_input(). Not model inference - just the
    final, shape-and-dtype-correct tensor plus the metadata needed to
    trace it back to its source tile."""

    array: Any  # numpy.ndarray, shape (batch, channel, H, W), float32
    shape: tuple[int, int, int, int]
    dtype: str
    source_tile: Path
    normalization_params: Optional[NormalizationParams]


@dataclass
class StageResult:
    """Generic wrapper used by main.py to track per-stage success/failure."""

    stage_name: str
    ok: bool
    payload: Any = None
    error: Optional[str] = None


@dataclass
class PipelineResult:
    """Final result returned by running the whole pipeline."""

    input_path: Path
    metadata: Optional[GeoTiffMetadata] = None
    cleaning: Optional[CleaningResult] = None
    preprocessing: Optional[PreprocessResult] = None
    validation: Optional[ValidationResult] = None
    tiling: Optional[TilingResult] = None
    verification: Optional[VerificationResult] = None
    model_inputs: Optional[list[ModelInput]] = None
    success: bool = False
    failed_stage: Optional[str] = None
    error_message: Optional[str] = None
