"""Diagnostic visualization generator for DSM and postprocessing quality evaluation.

Generates:
1. Elevation heatmap (normalized colormap)
2. Local residual/error map (|refined - raw|)
3. Gradient magnitude map
4. Slope map (degrees)
5. Outlier/spike mask
6. Comparative elevation and gradient histograms
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .diagnostics import compute_terrain_gradients


def generate_diagnostic_visualizations(
    raw: np.ndarray,
    refined: np.ndarray,
    out_dir: Path | str,
    spike_mask: np.ndarray | None = None,
    gsd: Tuple[float, float] = (1.0, 1.0),
    prefix: str = "diagnostic",
) -> dict[str, str]:
    """Generate all required visual diagnostic maps and histograms as PNGs."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    generated: dict[str, str] = {}

    valid = np.isfinite(raw) & np.isfinite(refined)

    # 1. Elevation Heatmap (Side-by-Side: Raw vs Refined)
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    vmin = min(float(np.nanmin(raw)), float(np.nanmin(refined)))
    vmax = max(float(np.nanmax(raw)), float(np.nanmax(refined)))

    im0 = axes[0].imshow(raw, cmap="terrain", vmin=vmin, vmax=vmax)
    axes[0].set_title("Pre-Postprocessing Elevation (m)")
    axes[0].axis("off")
    plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

    im1 = axes[1].imshow(refined, cmap="terrain", vmin=vmin, vmax=vmax)
    axes[1].set_title("Postprocessed DSM Elevation (m)")
    axes[1].axis("off")
    plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    plt.tight_layout()
    p1 = out_dir / f"{prefix}_elevation_comparison.png"
    fig.savefig(p1, dpi=150)
    plt.close(fig)
    generated["elevation_comparison"] = str(p1)

    # 2. Local Residual / Absolute Error Map
    diff = np.where(valid, np.abs(refined.astype(np.float64) - raw.astype(np.float64)), 0.0)
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(diff, cmap="magma", vmin=0.0, vmax=max(1.0, float(np.percentile(diff[valid], 99.5))))
    ax.set_title("Local Elevation Delta / Residual Map |Refined - Raw| (m)")
    ax.axis("off")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    p2 = out_dir / f"{prefix}_residual_map.png"
    fig.savefig(p2, dpi=150)
    plt.close(fig)
    generated["residual_map"] = str(p2)

    # 3. Gradient Magnitude & Slope Maps
    _, _, slope_raw = compute_terrain_gradients(raw, gsd)
    _, _, slope_ref = compute_terrain_gradients(refined, gsd)

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    s_max = max(60.0, float(np.percentile(slope_ref[valid], 99.0)))
    im0 = axes[0].imshow(slope_raw, cmap="inferno", vmin=0.0, vmax=s_max)
    axes[0].set_title("Raw Slope Map (degrees)")
    axes[0].axis("off")
    plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

    im1 = axes[1].imshow(slope_ref, cmap="inferno", vmin=0.0, vmax=s_max)
    axes[1].set_title("Postprocessed Slope Map (degrees)")
    axes[1].axis("off")
    plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    plt.tight_layout()
    p3 = out_dir / f"{prefix}_slope_comparison.png"
    fig.savefig(p3, dpi=150)
    plt.close(fig)
    generated["slope_comparison"] = str(p3)

    # 4. Spike / Outlier Mask
    if spike_mask is not None:
        fig, ax = plt.subplots(figsize=(8, 7))
        ax.imshow(spike_mask, cmap="Reds", interpolation="nearest")
        ax.set_title(f"Detected Spike Outlier Mask ({int(np.sum(spike_mask))} px)")
        ax.axis("off")
        plt.tight_layout()
        p4 = out_dir / f"{prefix}_spike_mask.png"
        fig.savefig(p4, dpi=150)
        plt.close(fig)
        generated["spike_mask"] = str(p4)

    # 5. Histograms: Elevation & Gradients
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].hist(raw[valid].ravel(), bins=60, alpha=0.6, label="Raw Elevation", density=True, color="blue")
    axes[0].hist(refined[valid].ravel(), bins=60, alpha=0.6, label="Refined Elevation", density=True, color="green")
    axes[0].set_title("Histogram of Elevation Values")
    axes[0].set_xlabel("Elevation (m)")
    axes[0].set_ylabel("Density")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].hist(slope_raw[valid].ravel(), bins=60, alpha=0.6, label="Raw Slope", density=True, color="orange")
    axes[1].hist(slope_ref[valid].ravel(), bins=60, alpha=0.6, label="Refined Slope", density=True, color="purple")
    axes[1].set_title("Histogram of Terrain Slope")
    axes[1].set_xlabel("Slope (degrees)")
    axes[1].set_ylabel("Density")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    p5 = out_dir / f"{prefix}_histograms.png"
    fig.savefig(p5, dpi=150)
    plt.close(fig)
    generated["histograms"] = str(p5)

    return generated
