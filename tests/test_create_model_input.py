from pathlib import Path

import pytest

import config
from src.create_model_input import prepare_model_input, prepare_model_inputs
from src.exceptions import ModelInputError
from src.tile_dem import tile_dem


def test_model_input_shape_matches_config(tmp_geotiff_exact: Path, tmp_path: Path):
    tiling_result = tile_dem(tmp_geotiff_exact, tmp_path / "tiles", tile_size=512, overlap=0)
    model_input = prepare_model_input(tiling_result.tiles[0].tile_path, tile_size=512)

    assert model_input.shape == (1, 1, 512, 512)
    assert model_input.dtype == "float32"
    assert model_input.array.shape == config.model_input_shape(512)


def test_model_input_not_hardcoded_to_256(tmp_geotiff_exact: Path, tmp_path: Path):
    tiling_result = tile_dem(tmp_geotiff_exact, tmp_path / "tiles", tile_size=512, overlap=0)
    model_input = prepare_model_input(tiling_result.tiles[0].tile_path, tile_size=512)
    assert model_input.shape[-1] != 256
    assert model_input.shape[-1] == 512


def test_model_input_shape_mismatch_raises(tmp_geotiff_exact: Path, tmp_path: Path):
    tiling_result = tile_dem(tmp_geotiff_exact, tmp_path / "tiles", tile_size=512, overlap=0)
    with pytest.raises(ModelInputError):
        # Ask for a tile_size that doesn't match what's actually on disk.
        prepare_model_input(tiling_result.tiles[0].tile_path, tile_size=256)


def test_prepare_model_inputs_handles_all_tiles_dynamically(
    tmp_geotiff_non_divisible: Path, tmp_path: Path
):
    tiling_result = tile_dem(tmp_geotiff_non_divisible, tmp_path / "tiles", tile_size=512, overlap=64)
    tile_paths = [t.tile_path for t in tiling_result.tiles]
    model_inputs = prepare_model_inputs(tile_paths, tile_size=512)
    assert len(model_inputs) == len(tile_paths)
    assert all(mi.shape == (1, 1, 512, 512) for mi in model_inputs)
