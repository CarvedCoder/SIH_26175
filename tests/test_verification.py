from pathlib import Path

from src.clean_dem import clean_dem
from src.preprocess_dem import preprocess_dem
from src.tile_dem import tile_dem
from src.verification import verify_cleaned, verify_preprocessed, verify_tiles


def test_verify_cleaned_ok(tmp_geotiff: Path, tmp_path: Path):
    cleaning_result = clean_dem(tmp_geotiff, tmp_path / "process")
    verification = verify_cleaned(cleaning_result.output_path)
    assert verification.all_ok


def test_verify_preprocessed_ok(tmp_geotiff: Path, tmp_path: Path):
    cleaning_result = clean_dem(tmp_geotiff, tmp_path / "process")
    preprocess_result = preprocess_dem(cleaning_result.output_path, tmp_path / "process")
    verification = verify_preprocessed(preprocess_result.output_path)
    assert verification.all_ok


def test_verify_tiles_all_ok(tmp_geotiff_non_divisible: Path, tmp_path: Path):
    tiling_result = tile_dem(tmp_geotiff_non_divisible, tmp_path / "tiles", tile_size=512, overlap=64)
    verification = verify_tiles(tiling_result.tiles, tile_size=512)
    assert verification.all_ok
    assert verification.checked_count == len(tiling_result.tiles)
    assert verification.failed_count == 0


def test_verify_tiles_detects_shape_mismatch(tmp_geotiff_non_divisible: Path, tmp_path: Path):
    tiling_result = tile_dem(tmp_geotiff_non_divisible, tmp_path / "tiles", tile_size=512, overlap=64)
    # Verify against a tile_size the tiles do NOT actually have.
    verification = verify_tiles(tiling_result.tiles, tile_size=256)
    assert not verification.all_ok
    assert verification.failed_count == len(tiling_result.tiles)
