"""Scene outputs + payload assembly (tranche 3c — pure moves from
depthwizard.inference; names are re-exported there so every existing
import path keeps working).

Contains the artifact writer (GeoTIFF/NPY/preview) and the
backend/frontend scene payload contract (webapp/src/lib/dw.ts mirrors
these types). The golden regression test pins the numerical behavior of
the DSM products these functions produce.
"""

from __future__ import annotations

import base64
import io
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from rasterio import CRS, Affine

from ..anchoring import ANCHORED_LABEL, AnchorResult

MAX_GRID_SIDE = 512  # webapp mesh grid cap (stride-downsampled)


def georef_state(profile: dict) -> tuple[bool, CRS | None, Affine | None]:
    """(is_georeferenced, crs, transform) — CRS None means pixel-space only."""
    crs = profile.get("_crs_obj")
    tf = profile.get("_transform_obj")
    return (crs is not None, crs, tf)


# ---------------------------------------------------------------------------
# Dn resolution: explicit file -> cache -> live backbone
# ---------------------------------------------------------------------------


def downsample_stride(h: int, w: int, max_side: int = MAX_GRID_SIDE) -> int:
    """Integer stride keeping both dims <= max_side (>=1)."""
    return max(1, int(np.ceil(max(h, w) / max_side)))


def downsample_grid(grid: np.ndarray, stride: int) -> np.ndarray:
    """Stride-subsample [H,W] -> [ceil(H/s), ceil(W/s)] (mesh-friendly, cheap)."""
    if stride <= 1:
        return grid
    return grid[::stride, ::stride]


def compute_stats(dsm: np.ndarray) -> dict[str, float]:
    """The [stats] line of the infer path — descriptive, NOT citable metrics."""
    d = np.asarray(dsm, dtype=np.float64)
    return {
        "n": int(d.size),
        "min": float(d.min()),
        "mean": float(d.mean()),
        "median": float(np.median(d)),
        "max": float(d.max()),
        "neg": int((d < 0).sum()),
    }


def rgb_png_data_url(rgb_u8: np.ndarray, max_side: int = MAX_GRID_SIDE) -> str:
    """uint8 [H,W,3] -> 'data:image/png;base64,...' (downsampled for the mesh)."""
    from PIL import Image

    h, w = rgb_u8.shape[:2]
    s = downsample_stride(h, w, max_side)
    if s > 1:
        rgb_u8 = rgb_u8[::s, ::s]
    buf = io.BytesIO()
    Image.fromarray(rgb_u8).save(buf, format="PNG", optimize=True)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def save_preview_png(dsm: np.ndarray, path: Path, title: str) -> None:
    """Terrain-colormapped preview (matplotlib, Agg)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 6), constrained_layout=True)
    im = ax.imshow(dsm, cmap="terrain")
    ax.set_title(title, fontsize=10)
    ax.set_axis_off()
    fig.colorbar(im, ax=ax, label="elevation (m)")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def write_outputs(
    out_dir: Path,
    dsm: np.ndarray,
    profile: dict,
    anchored: AnchorResult | None,
    preview_title: str,
    agl_raw: np.ndarray | None = None,
    postprocess_meta: dict | None = None,
) -> dict[str, str | None]:
    """Write dsm.npy (+ dsm.tif when georeferenced) (+ anchored DSM).

    When post-processing ran, ``agl_raw`` (the untouched CalibrationNet
    output) is ALSO written to agl_raw.npy — the raw signal is never
    overwritten — and ``postprocess_meta`` lands in postprocess_meta.json.

    Returns relative path strings for the payload's ``outputs`` block.
    """
    import json

    import rasterio

    out_dir.mkdir(parents=True, exist_ok=True)
    outputs: dict[str, str | None] = {}

    if agl_raw is not None:
        np.save(out_dir / "agl_raw.npy", agl_raw.astype(np.float32))
        outputs["agl_raw_npy"] = str(out_dir / "agl_raw.npy")
        if postprocess_meta is not None:
            with open(out_dir / "postprocess_meta.json", "w", encoding="utf-8") as f:
                json.dump(postprocess_meta, f, indent=2)
            outputs["postprocess_meta"] = str(out_dir / "postprocess_meta.json")

    np.save(out_dir / "dsm.npy", dsm.astype(np.float32))
    outputs["dsm_npy"] = str(out_dir / "dsm.npy")

    georef, crs, tf = georef_state(profile)
    if georef:
        with rasterio.open(
            out_dir / "dsm.tif",
            "w",
            driver="GTiff",
            height=dsm.shape[0],
            width=dsm.shape[1],
            count=1,
            dtype="float32",
            crs=crs,
            transform=tf,
            compress="deflate",
        ) as dst:
            dst.write(dsm.astype(np.float32), 1)
        outputs["dsm_tif"] = str(out_dir / "dsm.tif")
    else:
        outputs["dsm_tif"] = None

    if anchored is not None:
        p = out_dir / "dsm_anchored.tif" if georef else out_dir / "dsm_anchored.npy"
        if georef:
            with rasterio.open(
                p,
                "w",
                driver="GTiff",
                height=anchored.dsm.shape[0],
                width=anchored.dsm.shape[1],
                count=1,
                dtype="float32",
                crs=crs,
                transform=tf,
                compress="deflate",
            ) as dst:
                dst.write(anchored.dsm.astype(np.float32), 1)
        else:
            np.save(p, anchored.dsm.astype(np.float32))
        outputs["dsm_anchored"] = str(p)
    else:
        outputs["dsm_anchored"] = None

    save_preview_png(
        anchored.dsm if anchored is not None else dsm,
        out_dir / "dsm_preview.png",
        preview_title,
    )
    outputs["preview_png"] = str(out_dir / "dsm_preview.png")
    return outputs


def build_scene_payload(
    dsm: np.ndarray,
    rgb_u8: np.ndarray,
    *,
    stem: str,
    mode: str,
    dn_source: str,
    model_tag: str,
    device: str,
    profile: dict,
    anchored: AnchorResult | None,
    outputs: dict[str, str | None],
    elapsed_sec: float,
) -> dict:
    """JSON-serializable scene description for the Three.js viewer.

    Contract (webapp/src/lib/dw.ts mirrors these types):
        grid.data       row-major flattened [height*width] floats (metres)
        rgb_png         data-URL PNG, downsampled to the SAME grid footprint
        stats           descriptive only — never citable metrics
        anchored        false | {label: 'ANCHORED (not learned)', source: ...}
        georef.crs      string; "UNKNOWN" when the input carried no CRS
        meta.pixel_size_m   [float, float] metres-per-source-pixel [x, y]
                            or null when CRS is absent (Track-1 honest null;
                            viewers MUST branch on null and refuse metric
                            claims — see worklog Section 4)
    """
    georef, crs, tf = georef_state(profile)
    stride = downsample_stride(dsm.shape[0], dsm.shape[1])
    grid = downsample_grid(dsm, stride)
    stats = compute_stats(dsm)
    transform_repr = list(tf)[:6] if tf is not None else "UNKNOWN"

    # GSD honesty (worklog Section 4: "GSD honesty regression"): pixel_size_m is
    # the real ground-sample distance per source pixel, in metres, derived from
    # the raster's CRS+transform. Honest ``None`` for non-georeferenced Track-1
    # inputs — never a fallback guess. Consumers (Viewer3D, demprior) MUST
    # branch on null and refuse metric claims when it is absent.
    from ..geo import pixel_size_metres

    pixel_size_m = pixel_size_metres(crs, tf)
    pixel_size_m_json = (
        [round(float(v), 6) for v in pixel_size_m] if pixel_size_m is not None else None
    )

    payload = {
        "ok": True,
        "stem": stem,
        "grid": {
            "height": int(grid.shape[0]),
            "width": int(grid.shape[1]),
            "stride": int(stride),
            "data": [round(float(v), 3) for v in grid.ravel()],
        },
        "rgb_png": rgb_png_data_url(rgb_u8),
        "stats": stats,
        "anchored": (
            {"label": ANCHORED_LABEL, "source": anchored.source}
            if anchored is not None
            else False
        ),
        "georef": {
            "crs": str(crs) if crs is not None else "UNKNOWN",
            "transform": transform_repr,
        },
        "meta": {
            "model_tag": model_tag,
            "device": device,
            "dn_source": dn_source,
            "mode": mode,
            "source_shape": [int(dsm.shape[0]), int(dsm.shape[1])],
            "pixel_size_m": pixel_size_m_json,
            "elapsed_sec": round(elapsed_sec, 2),
        },
        "outputs": outputs,
    }
    return payload


def write_semantic_outputs(
    out_dir: Path,
    sem_probs: np.ndarray,
    checkpoint_name: str = "unknown",
    model_name: str = "CalibrationNet_aux",
) -> dict[str, str | None]:
    """Persist semantic segmentation artifacts (once per scene).

    Writes:
        semantic_labels.npy     uint8  [H,W]   — argmax class IDs (0–5)
        semantic_probs.npy      float16 [6,H,W] — compact probability tensor
        semantic_confidence.npy float16 [H,W]   — max probability per pixel
        semantic_map.png        RGBA            — class-colored visualization
        semantic_meta.json      JSON            — model, confidence stats, legend

    The float16 format halves storage vs float32 while keeping >3 decimal
    digits of precision — more than enough for classification and routing.
    The canonical machine-readable data is the .npy files; the PNG is
    for human consumption only.

    Returns dict of artifact output paths for the payload.
    """
    import json

    from PIL import Image

    from ..semantic_segmenter import (
        prediction_from_probs,
        render_semantic_map,
        semantic_metadata,
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    outputs: dict[str, str | None] = {}

    prediction = prediction_from_probs(sem_probs)

    # Machine-readable artifacts (the authority for Route Assist and Inspect)
    labels_path = out_dir / "semantic_labels.npy"
    np.save(labels_path, prediction.labels)
    outputs["semantic_labels"] = str(labels_path)

    probs_path = out_dir / "semantic_probs.npy"
    np.save(probs_path, prediction.probs.astype(np.float16))
    outputs["semantic_probs"] = str(probs_path)

    conf_path = out_dir / "semantic_confidence.npy"
    np.save(conf_path, prediction.confidence.astype(np.float16))
    outputs["semantic_confidence"] = str(conf_path)

    # Visualization artifact (class-colored RGBA PNG)
    map_rgba = render_semantic_map(prediction)
    map_path = out_dir / "semantic_map.png"
    Image.fromarray(map_rgba, mode="RGBA").save(map_path, format="PNG")
    outputs["semantic_map"] = str(map_path)

    # Metadata (model provenance, confidence stats, legend)
    meta = semantic_metadata(prediction, checkpoint_name, model_name)
    meta_path = out_dir / "semantic_meta.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    outputs["semantic_meta"] = str(meta_path)

    return outputs

