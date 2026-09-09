#!/usr/bin/env python3
"""DepthWizard (SIH26175) - GeoTIFF preprocessing pipeline entry point.

Usage
-----
Interactive (default):

    python main.py

Non-interactive:

    python main.py --input data/raw/286_5630_dem.tif

This is the ONLY script a user needs to run. It orchestrates all seven
pipeline stages and passes each stage's return value into the next, per
each stage's input/output contract (see src/schemas.py). No stage is a
standalone script - every stage lives in src/ as an importable function
that returns a dataclass instead of printing and exiting.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import config
from src.clean_dem import clean_dem
from src.create_model_input import prepare_model_inputs
from src.exceptions import PipelineError
from src.inspect_geotiff import inspect_geotiff
from src.preprocess_dem import preprocess_dem
from src.raster_io import detect_format
from src.schemas import PipelineResult
from src.tile_dem import tile_dem
from src.verification import verify_tiles
from src.visualize_dem import visualize_dem

STAGE_COUNT = 7


def _banner() -> None:
    print("=" * 42)
    print("      GeoTIFF Processing Pipeline")
    print("=" * 42)


def _prompt_for_path() -> Path:
    raw = input("\nEnter DEM path:\n> ").strip().strip('"')
    return Path(raw)


def _stage_header(index: int, label: str) -> None:
    print(f"[{index}/{STAGE_COUNT}] {label}...", end=" ", flush=True)


def run_pipeline(input_path: Path) -> PipelineResult:
    """Run all seven stages against input_path, stage by stage.

    Returns a PipelineResult that is always populated with whatever
    stages completed successfully, even if a later stage failed - so a
    caller (or a future GUI/API) can inspect partial progress.
    """
    result = PipelineResult(input_path=input_path)
    config.ensure_directories()

    # Stage 0: format detection / input validation (not one of the 7
    # numbered stages, but must happen before Stage 1 can run at all).
    try:
        detect_format(input_path)
    except PipelineError as exc:
        result.failed_stage = "input validation"
        result.error_message = str(exc)
        print(f"\nInput validation FAILED\n\nReason:\n{exc}")
        return result

    try:
        # [1/7] Inspect
        _stage_header(1, "Inspecting GeoTIFF")
        result.metadata = inspect_geotiff(input_path)
        print("OK")
        print(
            f"       {result.metadata.width}x{result.metadata.height}, "
            f"{result.metadata.band_count} band(s), CRS={result.metadata.crs}, "
            f"valid pixels={result.metadata.valid_pixel_count}/{result.metadata.total_pixels}"
        )

        # [2/7] Clean
        _stage_header(2, "Cleaning DEM")
        result.cleaning = clean_dem(input_path, config.PROCESS_DIR)
        print("OK")
        print(
            f"       invalid pixels handled: {result.cleaning.invalid_pixel_count} "
            f"(nan={result.cleaning.nan_pixel_count}, inf={result.cleaning.inf_pixel_count}, "
            f"nodata={result.cleaning.nodata_pixel_count})"
        )
        for note in result.cleaning.notes:
            print(f"       note: {note}")

        # [3/7] Preprocess / normalize
        _stage_header(3, "Preprocessing DEM")
        result.preprocessing = preprocess_dem(result.cleaning.output_path, config.PROCESS_DIR)
        print("OK")
        print(
            f"       method={result.preprocessing.params.method}, "
            f"range=[{result.preprocessing.normalized_min:.4f}, "
            f"{result.preprocessing.normalized_max:.4f}]"
        )

        # [4/7] Visualize / validate
        _stage_header(4, "Visualizing / validating DEM")
        result.validation = visualize_dem(
            result.preprocessing.output_path, output_dir=config.OUTPUT_DIR
        )
        print("OK")
        if result.validation.figure_path:
            print(f"       figure saved: {result.validation.figure_path}")

        # [5/7] Tile
        _stage_header(5, "Creating tiles")
        result.tiling = tile_dem(result.preprocessing.output_path, config.TILES_DIR)
        print("OK")
        print(
            f"       {len(result.tiling.tiles)} tiles "
            f"({result.tiling.grid_rows}x{result.tiling.grid_cols} grid), "
            f"tile_size={result.tiling.tile_size}, overlap={result.tiling.overlap}"
        )

        # [6/7] Verify tiles
        _stage_header(6, "Verifying tiles")
        result.verification = verify_tiles(result.tiling.tiles, tile_size=result.tiling.tile_size)
        if not result.verification.all_ok:
            failed = [e for e in result.verification.entries if not e.ok]
            raise PipelineError(
                f"{len(failed)}/{result.verification.checked_count} tiles failed "
                f"verification. First failure: {failed[0].tile_path} - {failed[0].reason}"
            )
        print("OK")
        print(f"       {result.verification.checked_count} tiles verified")

        # [7/7] Model input
        _stage_header(7, "Preparing model input")
        tile_paths = [t.tile_path for t in result.tiling.tiles]
        result.model_inputs = prepare_model_inputs(
            tile_paths,
            normalization_params=result.preprocessing.params,
            tile_size=result.tiling.tile_size,
        )
        print("OK")
        print(
            f"       {len(result.model_inputs)} tensors, "
            f"shape={result.model_inputs[0].shape}, dtype={result.model_inputs[0].dtype}"
        )

        result.success = True

    except PipelineError as exc:
        stage_names = {
            1: "Inspecting GeoTIFF",
            2: "Cleaning DEM",
            3: "Preprocessing DEM",
            4: "Visualizing / validating DEM",
            5: "Creating tiles",
            6: "Verifying tiles",
            7: "Preparing model input",
        }
    if result.metadata is None:
             failed_stage_name = stage_names[1]
    elif result.cleaning is None:
             failed_stage_name = stage_names[2]
    elif result.preprocessing is None:
             failed_stage_name = stage_names[3]
    elif result.validation is None:
             failed_stage_name = stage_names[4]
    elif result.tiling is None:
             failed_stage_name = stage_names[5]
    elif result.verification is None or not result.verification.all_ok:
             failed_stage_name = stage_names[6]
    else:
             failed_stage_name = stage_names[7]
    print("FAILED")
    print(f"\nReason:\n{exc}")
    result.failed_stage = failed_stage_name
    result.error_message = str(exc)

    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="DepthWizard GeoTIFF preprocessing pipeline")
    parser.add_argument(
        "--input",
        type=str,
        default=None,
        help="Path to a GeoTIFF. If omitted, the pipeline prompts interactively.",
    )
    args = parser.parse_args()

    _banner()

    if args.input:
        input_path = Path(args.input)
    else:
        input_path = _prompt_for_path()

    print("\nStarting pipeline...\n")
    result = run_pipeline(input_path)

    print()
    if result.success:
        print("Pipeline completed successfully.")
        return 0
    else:
        print(f"Pipeline stopped at stage: {result.failed_stage}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
