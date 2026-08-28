"""Stage 7: Model input preparation.

Converts a normalized tile GeoTIFF into a (batch, channel, height, width)
float32 tensor. Dimensions are derived entirely from config.py
(config.model_input_shape()) rather than hardcoded, so changing
config.TILE_SIZE automatically changes the produced shape - with
TILE_SIZE = 512 (the Blueprint-mandated default) this yields
(1, 1, 512, 512), not the prototype's hardcoded (1, 1, 256, 256).

This does NOT run model inference. It creates the clean interface
`prepare_model_input(tile_path)` that a future inference stage can call
directly, per the task's explicit scope (model inference is out of scope
unless the supplied documents required a placeholder - they instead
describe the eventual model's tensor convention, batch x channel x height
x width, which is what this stage targets).
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import rasterio

import config
from src.exceptions import ModelInputError
from src.schemas import ModelInput, NormalizationParams


def prepare_model_input(
    tile_path: Path,
    normalization_params: Optional[NormalizationParams] = None,
    tile_size: int = config.TILE_SIZE,
) -> ModelInput:
    """Convert one normalized tile into a model-ready tensor.

    Parameters
    ----------
    tile_path:
        Path to a normalized tile GeoTIFF (output of tile_dem(), itself
        built from preprocess_dem()'s output).
    normalization_params:
        The NormalizationParams used upstream (from PreprocessResult),
        carried through purely as provenance metadata on the returned
        ModelInput - not re-applied here.
    tile_size:
        Expected tile edge length; defaults to config.TILE_SIZE. Used only
        to validate the tile's actual shape matches expectations.

    Raises
    ------
    ModelInputError
        If the tile cannot be read or its shape does not match
        (tile_size, tile_size).
    """
    try:
        with rasterio.open(tile_path) as dataset:
            data = dataset.read(1)
    except rasterio.errors.RasterioIOError as exc:
        raise ModelInputError(f"Could not open tile {tile_path}: {exc}") from exc

    if data.shape != (tile_size, tile_size):
        raise ModelInputError(
            f"Tile {tile_path} has shape {data.shape}, expected "
            f"({tile_size}, {tile_size}). Was it generated with a different "
            f"TILE_SIZE than the current config?"
        )

    array = data.astype(np.dtype(config.NORMALIZED_DTYPE))
    array = np.expand_dims(array, axis=0)  # channel
    array = np.expand_dims(array, axis=0)  # batch

    expected_shape = config.model_input_shape(tile_size)
    if array.shape != expected_shape:
        raise ModelInputError(
            f"Prepared tensor shape {array.shape} does not match configured "
            f"model input shape {expected_shape}."
        )

    return ModelInput(
        array=array,
        shape=array.shape,
        dtype=str(array.dtype),
        source_tile=tile_path,
        normalization_params=normalization_params,
    )


def prepare_model_inputs(
    tile_paths: list[Path],
    normalization_params: Optional[NormalizationParams] = None,
    tile_size: int = config.TILE_SIZE,
) -> list[ModelInput]:
    """Prepare model input tensors for every tile (dynamic tile count)."""
    return [
        prepare_model_input(path, normalization_params=normalization_params, tile_size=tile_size)
        for path in tile_paths
    ]
