"""
Central configuration for the DepthWizard GeoTIFF preprocessing pipeline.

Every tunable constant used by more than one module lives here so the
pipeline never needs a code change to adjust tiling, normalization, or
directory layout.

Provenance of the key constants (see project root README.md, "Blueprint vs.
paper vs. implemented" section, for the full discussion):

- TILE_SIZE: [BLUEPRINT] Technical Blueprint Section 9 ("Training Strategy")
  fixes training/inference resolution at 512x512, and Section 11
  ("Large Image / Tiling Pipeline") explicitly repeats 512x512 as the tiling
  size ("matching your training resolution"). This is a Blueprint
  requirement, not a value copied from the IM2ELEVATION paper (which trains
  on 500x500 patches - see NOTE below).

- OVERLAP: [BLUEPRINT] Section 11 point 3 specifies "Overlap of 64-128 px
  (12-25%) between adjacent tiles". 64 px is chosen as the default because
  it is the lower (cheaper) bound of that explicit range and matches the
  worked example given elsewhere in the same document (TILE_SIZE=512,
  OVERLAP=64). It is NOT taken from IM2ELEVATION, which uses 250 px overlap
  on 500 px patches (50%) for a different purpose (training-set expansion,
  not seamless-mosaic stitching) - the instructions are explicit that the
  two documents must not be silently merged, and the Blueprint is the
  primary spec for this project.

- PAD_MODE: [BLUEPRINT] Section 11 point 5: "Padding at scene borders
  (reflect-pad) so edge tiles get full-size input even at the image
  boundary."

- Gaussian-weighted blending (Section 11 point 4) is a STITCHING-time
  concern (reconstructing a seamless mosaic from overlapping tile
  predictions). Model inference and stitching are explicitly out of scope
  for the current task, so it is not implemented here. The tiling module
  does, however, record everything (grid position, pixel offsets, padding)
  a future stitching module would need - see src/schemas.py: TileInfo.

- Normalization target (float32, [0, 1] min-max): this is a pipeline
  requirement carried over from the original prototype/task description,
  not a Blueprint-mandated value (the Blueprint discusses normalization
  only in the context of the training loss, e.g. scale-and-shift-invariant
  loss on relative depth - Section 10 - which is a model-training concern,
  not a preprocessing-stage concern). Because no document mandates a fixed
  global elevation range, normalization here is per-source min-max with
  the parameters explicitly recorded (see src/preprocess_dem.py) so it is
  reproducible and reversible, and the method is swappable via
  NORMALIZATION_METHOD without touching pipeline code.
"""

from __future__ import annotations

from pathlib import Path

# ---------------------------------------------------------------------------
# Directory layout
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent

DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESS_DIR = DATA_DIR / "process"
TILES_DIR = PROCESS_DIR / "tiles"
OUTPUT_DIR = DATA_DIR / "output"

# ---------------------------------------------------------------------------
# Tiling configuration [BLUEPRINT Section 11]
# ---------------------------------------------------------------------------

#: Tile edge length in pixels. Blueprint Sections 9 & 11.
TILE_SIZE: int = 512

#: Overlap between adjacent tiles, in pixels. Blueprint Section 11 point 3
#: recommends 64-128 px (12-25% of TILE_SIZE); 64 is the documented default.
OVERLAP: int = 64

#: Padding strategy for tiles that would otherwise run past the raster
#: edge (non-divisible dimensions, or a raster smaller than TILE_SIZE).
#: Blueprint Section 11 point 5 specifies reflect-padding. If a window is
#: too small for numpy's reflect mode (pad width >= data extent along that
#: axis - only possible for pathologically tiny rasters), the pipeline
#: falls back to edge-replication padding and records this deviation in
#: the tile's metadata; this fallback is an engineering safeguard, not a
#: Blueprint-specified behavior.
PAD_MODE: str = "reflect"
PAD_MODE_FALLBACK: str = "edge"

# ---------------------------------------------------------------------------
# Normalization configuration
# ---------------------------------------------------------------------------

#: "minmax" rescales valid elevation values to [0, 1] in float32.
#: "standardize" (z-score) is also implemented for future use.
NORMALIZATION_METHOD: str = "minmax"

#: Output dtype for normalized elevation rasters and model input tensors.
NORMALIZED_DTYPE: str = "float32"

# ---------------------------------------------------------------------------
# Model input configuration
# ---------------------------------------------------------------------------

#: Derived, never hardcoded elsewhere: batch x channel x height x width.
#: With TILE_SIZE = 512 this yields (1, 1, 512, 512), per Blueprint Section
#: 9's 512x512 training tiles and the batch/channel-first convention
#: Section 15/16 assume for the (future) model interface.
MODEL_INPUT_CHANNELS: int = 1
MODEL_INPUT_BATCH: int = 1


def model_input_shape(tile_size: int = TILE_SIZE) -> tuple[int, int, int, int]:
    """Return the (batch, channel, height, width) shape for a single tile."""
    return (MODEL_INPUT_BATCH, MODEL_INPUT_CHANNELS, tile_size, tile_size)


# ---------------------------------------------------------------------------
# Supported input formats [current scope vs. future scope]
# ---------------------------------------------------------------------------

#: Formats the pipeline can process today. JPG/JPEG/PNG are deliberately
#: not implemented yet (per the task scope), but the input layer
#: (src/raster_io.py) is structured so adding them later is a matter of
#: adding a loader function and an entry here - not a pipeline rewrite.
SUPPORTED_EXTENSIONS: tuple[str, ...] = (".tif", ".tiff")
FUTURE_EXTENSIONS: tuple[str, ...] = (".jpg", ".jpeg", ".png")


def ensure_directories() -> None:
    """Create the data directory tree if it does not already exist."""
    for directory in (RAW_DIR, PROCESS_DIR, TILES_DIR, OUTPUT_DIR):
        directory.mkdir(parents=True, exist_ok=True)
