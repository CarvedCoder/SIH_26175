"""FileSceneRepository — persistence for scene records.

A scene is a directory of artifacts (the raw input raster + intermediates
+ outputs, laid out by core/paths.py) plus ONE metadata document
(``scene.json``) that makes the scene self-describing: exact input
filename, raster inspection metadata, creation time.

The repository owns the RECORD; it does not do raster content validation
(application layer) or HTTP streaming (API layer). Layout/security
invariants stay in core/paths.py (id validation, symlink guards) — the
repository composes them.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from backend.app.core.paths import (
    get_scene_raw_dir,
    valid_scene_id,
)

SCENE_METADATA_NAME = "scene.json"

# Designated upload names in deterministic check order (the upload stores
# the validated raster as input.<original extension>; PNG/JPG are
# first-class inputs alongside GeoTIFF).
DESIGNATED_INPUTS = ("input.tif", "input.tiff", "input.png", "input.jpg", "input.jpeg")
RASTER_SUFFIXES = {".tif", ".tiff", ".png", ".jpg", ".jpeg"}


def designated_input_name(extension: str) -> str:
    """Exact filename the validated upload is stored under — the
    determinism contract of the processing service (input.<ext>)."""
    return f"input{extension}"


class FileSceneRepository:
    """Scene records over the per-scene raw directories."""

    # -- record I/O ----------------------------------------------------------

    def load_record(self, scene_id: str) -> dict[str, Any] | None:
        """The stored scene.json document, or None when absent/corrupt."""
        if not valid_scene_id(scene_id):
            return None
        path = get_scene_raw_dir(scene_id) / SCENE_METADATA_NAME
        if not path.is_file():
            return None
        try:
            with path.open("r", encoding="utf-8") as f:
                data = json.load(f)
            return data if isinstance(data, dict) else None
        except (OSError, json.JSONDecodeError):
            return None

    def write_record(
        self, scene_id: str, original_filename: str, metadata: dict[str, Any]
    ) -> None:
        """Persist the scene record (metadata values may be enums —
        they are unwrapped via .value when present)."""
        payload = {
            "filename": original_filename,
            "created_at": datetime_now_iso(),
            "metadata": {
                key: (value.value if hasattr(value, "value") else value)
                for key, value in metadata.items()
            },
        }
        path = get_scene_raw_dir(scene_id) / SCENE_METADATA_NAME
        with path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

    # -- scene identity / inputs ------------------------------------------------

    def exists(self, scene_id: str) -> bool:
        return valid_scene_id(scene_id) and get_scene_raw_dir(scene_id).is_dir()

    def raw_dir(self, scene_id: str) -> Path:
        return get_scene_raw_dir(scene_id)

    def find_designated_input(self, scene_id: str) -> Path | None:
        """The scene's designated input raster by exact known filename —
        never by iteration order (determinism contract, audit M9)."""
        input_dir = get_scene_raw_dir(scene_id)
        if not input_dir.exists():
            return None
        for name in DESIGNATED_INPUTS:
            candidate = input_dir / name
            if candidate.is_file():
                return candidate
        return None

    def list_scene_ids(self) -> list[str]:
        """All valid scene ids, most recent record first."""
        from backend.app.core.paths import SCENES_RAW_DIR

        if not SCENES_RAW_DIR.is_dir():
            return []
        ids = [
            d.name
            for d in SCENES_RAW_DIR.iterdir()
            if d.is_dir() and valid_scene_id(d.name)
        ]
        ids.sort(
            key=lambda sid: (self.load_record(sid) or {}).get("created_at") or sid,
            reverse=True,
        )
        return ids

    def replace_input(self, scene_id: str, staged_path: Path) -> None:
        """Atomically move a validated staged raster into the scene's raw
        dir under the designated input.tif name (mosaic commit path)."""
        scene_dir = get_scene_raw_dir(scene_id)
        for old in scene_dir.glob("input.*"):
            old.unlink()
        staged_path.replace(scene_dir / designated_input_name(".tif"))


def datetime_now_iso() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat()
