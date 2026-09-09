from __future__ import annotations

from pathlib import Path


# ---------------------------------------------------------------------------
# Project directories
# ---------------------------------------------------------------------------

# backend/app/core/paths.py
#                         ↑
# parents[0] = core
# parents[1] = app
# parents[2] = backend
# parents[3] = SIH_main
PROJECT_ROOT = Path(__file__).resolve().parents[3]


# ---------------------------------------------------------------------------
# Data directories
# ---------------------------------------------------------------------------

DATA_DIR = PROJECT_ROOT / "data"

RAW_DIR = DATA_DIR / "raw"
PROCESS_DIR = DATA_DIR / "process"
OUTPUT_DIR = DATA_DIR / "output"


# ---------------------------------------------------------------------------
# Scene directories
# ---------------------------------------------------------------------------

SCENES_RAW_DIR = RAW_DIR / "scenes"
SCENES_PROCESS_DIR = PROCESS_DIR / "scenes"
SCENES_OUTPUT_DIR = OUTPUT_DIR / "scenes"


# ---------------------------------------------------------------------------
# Scene-specific paths
# ---------------------------------------------------------------------------

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
    ]

    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)