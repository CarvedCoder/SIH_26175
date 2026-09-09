"""Stage 4: Visual validation.

The prototype's visualize_normalized.py called plt.show() and did nothing
else, which blocks execution until a human closes the window - unusable in
an automated pipeline. This stage instead:

- always computes and returns validation statistics (so the pipeline can
  run headless / unattended), and
- optionally saves a PNG figure to disk (default), and
- optionally also calls plt.show() interactively, only if the caller
  explicitly asks for it (show=True) - never by default.

Uses the non-interactive "Agg" backend for saving so this works in
environments with no display (servers, CI, this container).
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np
import rasterio  # noqa: E402

from src.raster_io import build_invalid_mask
from src.schemas import ValidationResult


def visualize_dem(
    input_path: Path,
    output_dir: Optional[Path] = None,
    show: bool = False,
    title: Optional[str] = None,
) -> ValidationResult:
    """Compute validation statistics for a raster and optionally save a figure.

    Parameters
    ----------
    input_path:
        Raster to validate (typically the normalized DEM, but works on any
        single-band raster - cleaned, normalized, or a tile).
    output_dir:
        If given, a PNG figure is saved here as "<stem>_validation.png".
        If None, no figure is saved (statistics are still returned).
    show:
        If True, also opens an interactive matplotlib window
        (plt.show()). Off by default so the pipeline never blocks.
    title:
        Optional custom plot title.
    """
    with rasterio.open(input_path) as dataset:
        data = dataset.read(1)
        nodata = dataset.nodata

    _, _, _, invalid_mask = build_invalid_mask(data, nodata)
    valid_mask = ~invalid_mask
    valid_pixel_count = int(np.sum(valid_mask))
    invalid_pixel_count = int(np.sum(invalid_mask))

    if valid_pixel_count > 0:
        valid_values = data[valid_mask]
        data_min = float(np.min(valid_values))
        data_max = float(np.max(valid_values))
        data_mean = float(np.mean(valid_values))
    else:
        data_min = data_max = data_mean = float("nan")

    figure_path: Optional[Path] = None
    if output_dir is not None or show:
        if show:
            try:
                plt.switch_backend("TkAgg")
            except Exception as exc:
                raise RuntimeError(
                    "show=True requires an interactive matplotlib backend (e.g. TkAgg). "
                    "Install Tk support or call visualize_dem(show=False)."
                ) from exc

        display_data = np.where(valid_mask, data, np.nan).astype(np.float64)

        fig, ax = plt.subplots(figsize=(10, 8))
        im = ax.imshow(display_data)
        fig.colorbar(im, ax=ax, label="Elevation (normalized or raw, per source raster)")
        ax.set_title(title or f"DEM validation - {input_path.name}")
        ax.set_xlabel("Pixel Column")
        ax.set_ylabel("Pixel Row")

        if output_dir is not None:
            output_dir.mkdir(parents=True, exist_ok=True)
            figure_path = output_dir / f"{input_path.stem}_validation.png"
            fig.savefig(figure_path, dpi=150, bbox_inches="tight")

        if show:
            plt.show()

        plt.close(fig)

    return ValidationResult(
        figure_path=figure_path,
        valid_pixel_count=valid_pixel_count,
        invalid_pixel_count=invalid_pixel_count,
        data_min=data_min,
        data_max=data_max,
        data_mean=data_mean,
    )
