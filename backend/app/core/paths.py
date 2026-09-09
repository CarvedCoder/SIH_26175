"""Filesystem layout + safe path handling for scene storage.

Layout (all relative to PROJECT_ROOT):
    data/raw/scenes/<scene_id>/       uploaded GeoTIFF inputs
    data/process/scenes/<scene_id>/   processing intermediates
    data/output/scenes/<scene_id>/    DSM products + previews

Security invariants enforced here:
    * scene/job ids are validated against a strict format BEFORE any path
      is built from them (defense in depth against path traversal, in case
      a future route accepts raw ids);
    * resolved paths must stay inside their storage root (symlink escape
      guard used by scene deletion).
"""

from __future__ import annotations

import os
import re
import shutil
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]

SCENE_ID_PATTERN = re.compile(r"^scene_[0-9a-f]{12}$")
JOB_ID_PATTERN = re.compile(r"^job_[0-9a-f]{12}$")

# ---------------------------------------------------------------------------
# Data directories
# ---------------------------------------------------------------------------

DATA_DIR = PROJECT_ROOT / "data"

RAW_DIR = DATA_DIR / "raw"
PROCESS_DIR = DATA_DIR / "process"
OUTPUT_DIR = DATA_DIR / "output"

# Scratch space for upload validation (files are validated here BEFORE
# being atomically moved into a scene's permanent raw directory).
UPLOAD_STAGING_DIR = DATA_DIR / "staging"


# ---------------------------------------------------------------------------
# Scene directories
# ---------------------------------------------------------------------------

SCENES_RAW_DIR = RAW_DIR / "scenes"
SCENES_PROCESS_DIR = PROCESS_DIR / "scenes"
SCENES_OUTPUT_DIR = OUTPUT_DIR / "scenes"


def valid_scene_id(scene_id: str) -> bool:
    return bool(SCENE_ID_PATTERN.match(scene_id))


def valid_job_id(job_id: str) -> bool:
    return bool(JOB_ID_PATTERN.match(job_id))


def get_scene_raw_dir(scene_id: str) -> Path:
    """Return the raw/input directory for a scene."""
    return SCENES_RAW_DIR / scene_id


def get_scene_process_dir(scene_id: str) -> Path:
    """Return the processing directory for a scene."""
    return SCENES_PROCESS_DIR / scene_id


def get_scene_output_dir(scene_id: str) -> Path:
    """Return the output directory for a scene."""
    return SCENES_OUTPUT_DIR / scene_id


def get_scene_input_path(scene_id: str, filename: str) -> Path:
    """Return the path for an uploaded scene input file."""
    return get_scene_raw_dir(scene_id) / filename


def is_within(child: Path, parent: Path) -> bool:
    """True when ``child`` resolves inside ``parent`` (symlink-safe)."""
    try:
        child.resolve().relative_to(parent.resolve())
        return True
    except ValueError:
        return False


def atomic_move(source: Path, destination: Path) -> None:
    """Move a validated file into place on the same filesystem.

    os.replace is atomic when both paths live on one device — staging lives
    under DATA_DIR with the scenes, so uploads never appear half-written in
    their final location.
    """
    destination.parent.mkdir(parents=True, exist_ok=True)
    os.replace(source, destination)


def delete_scene_directories(scene_id: str) -> None:
    """Remove every stored artifact for a scene, symlink-safe.

    Raises ValueError if any directory resolves OUTSIDE the corresponding
    storage root (a planted symlink must not escalate into deleting host
    files). Idempotent: missing directories are a no-op.
    """
    for directory, root in (
        (get_scene_raw_dir(scene_id), SCENES_RAW_DIR),
        (get_scene_process_dir(scene_id), SCENES_PROCESS_DIR),
        (get_scene_output_dir(scene_id), SCENES_OUTPUT_DIR),
    ):
        if directory.is_symlink() or not is_within(directory, root):
            raise ValueError(
                f"refusing to delete '{directory}': resolves outside the storage root"
            )
        if directory.exists():
            shutil.rmtree(directory)


# ---------------------------------------------------------------------------
# Directory initialization
# ---------------------------------------------------------------------------

def ensure_directories() -> None:
    """Create all directories required by the backend."""

    directories = [
        DATA_DIR,
        RAW_DIR,
        PROCESS_DIR,
        OUTPUT_DIR,
        SCENES_RAW_DIR,
        SCENES_PROCESS_DIR,
        SCENES_OUTPUT_DIR,
        UPLOAD_STAGING_DIR,
    ]

    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)
