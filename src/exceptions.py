"""Custom exceptions for the GeoTIFF pipeline.

Each pipeline stage raises a specific, descriptive exception rather than
letting a stage fail silently or pass invalid data downstream. main.py
catches PipelineError (the common base) at each stage boundary and reports
which stage failed and why.
"""

from __future__ import annotations


class PipelineError(Exception):
    """Base class for all pipeline-raised errors."""


class InputPathError(PipelineError):
    """The input path does not exist, is not a file, or is unreadable."""


class UnsupportedFormatError(PipelineError):
    """The input file extension is not currently supported."""


class InvalidGeoTiffError(PipelineError):
    """The file could not be opened/parsed as a valid GeoTIFF."""


class MissingCRSError(PipelineError):
    """The GeoTIFF has no coordinate reference system defined."""


class NoValidDataError(PipelineError):
    """A raster (or a tile of one) contains no usable, non-nodata pixels."""


class NormalizationError(PipelineError):
    """Normalization could not be performed (e.g. zero valid-data range)."""


class TilingError(PipelineError):
    """Tile generation failed."""


class TileVerificationError(PipelineError):
    """One or more generated tiles failed verification."""


class ModelInputError(PipelineError):
    """Model-input tensor preparation failed."""


class OutputDirectoryError(PipelineError):
    """An output directory could not be created or written to."""
