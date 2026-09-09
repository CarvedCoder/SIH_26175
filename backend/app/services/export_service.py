from __future__ import annotations

from pathlib import Path


from backend.app.core.paths import get_scene_output_dir


class ExportFileNotFound(FileNotFoundError):
    """Raised when a requested scene export does not exist."""


# ---------------------------------------------------------------------------
# File lookup
# ---------------------------------------------------------------------------

_EXPORT_CANDIDATES: dict[str, list[str]] = {
    "depth": [
        "depth.tif",
        "depth.png",
        "depth.npy",
        "depth.npz",
    ],
    "dsm": [
        "dsm.tif",
        "dsm.tiff",
        "dsm.npy",
        "dsm.png",
    ],
    "terrain": [
        "terrain.tif",
        "terrain.tiff",
        "terrain.png",
        "heightmap.png",
        "heightmap.npy",
    ],
    "validation": [
        "validation.json",
        "validation_metrics.json",
        "error_map.png",
        "validation_error_map.png",
    ],
}


def _find_export_file(
    scene_id: str,
    export_type: str,
) -> Path:
    """
    Find an exported artifact for a scene.

    The function only returns files that actually exist.
    """

    output_dir = get_scene_output_dir(scene_id)

    if not output_dir.exists():
        raise ExportFileNotFound(
            f"No output directory exists for scene '{scene_id}'."
        )

    candidates = _EXPORT_CANDIDATES.get(export_type)

    if candidates is None:
        raise ValueError(
            f"Unsupported export type: {export_type}"
        )

    for filename in candidates:
        path = output_dir / filename

        if path.is_file():
            return path

    raise ExportFileNotFound(
        f"No {export_type} export exists for scene '{scene_id}'."
    )


# ---------------------------------------------------------------------------
# Public export functions
# ---------------------------------------------------------------------------

def get_depth_export(scene_id: str) -> Path:
    """Return the generated depth artifact."""
    return _find_export_file(scene_id, "depth")


def get_dsm_export(scene_id: str) -> Path:
    """Return the generated DSM artifact."""
    return _find_export_file(scene_id, "dsm")


def get_terrain_export(scene_id: str) -> Path:
    """Return the generated terrain artifact."""
    return _find_export_file(scene_id, "terrain")


def get_validation_export(scene_id: str) -> Path:
    """Return the generated validation artifact."""
    return _find_export_file(scene_id, "validation")


def get_export(scene_id: str, export_type: str) -> Path:
    """
    Generic export lookup.

    Supported values:

        depth
        dsm
        terrain
        validation
    """

    return _find_export_file(
        scene_id,
        export_type,
    )