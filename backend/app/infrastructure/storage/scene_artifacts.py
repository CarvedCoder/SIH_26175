"""Scene artifact addressing through the ArtifactStore seam (tranche 3b).

Business logic must not construct raw filesystem paths for persistent
artifacts (refactor brief §7/§24). This module is the single place that
maps SCENE ARTIFACT KEYS to storage:

    key layout (under the store root = the data dir):
        raw/scenes/<scene_id>/input.<ext>        uploaded inputs
        process/scenes/<scene_id>/...            intermediates (incl. jobs)
        output/scenes/<scene_id>/dsm.npy ...     pipeline products

The keys are exactly the on-disk relative layout, so the LocalArtifactStore
serves the SAME files the path-convention code did — behavior identical,
addressing now canonical and traversal-validated. Swapping in an
S3/MinIO ArtifactStore later changes this module, not its consumers.

Until the S3 store exists, consumers receive local Paths (needed by
rasterio and FastAPI FileResponse) via ``store.path_for(key)``.
"""

from __future__ import annotations

from typing import Dict

from backend.app.domain.protocols import ArtifactStore
from backend.app.infrastructure.storage.local_artifact_store import (
    LocalArtifactStore,
)


def scene_artifact_store() -> ArtifactStore:
    """The artifact store rooted at the configured data dir.

    The root is resolved DYNAMICALLY (never imported at module load) so
    test fixtures that redirect the data root are honored. Constructed
    per call — cheap, holds no state, config-derived and disposable.
    """
    from backend.app.core import paths as paths_module

    return LocalArtifactStore(paths_module.DATA_DIR)


# ---------------------------------------------------------------------------
# Key builders (the ONLY place artifact key structure is defined)
# ---------------------------------------------------------------------------


def scene_output_dir_key(scene_id: str) -> str:
    return f"output/scenes/{scene_id}"


def scene_output_key(scene_id: str, name: str) -> str:
    return f"output/scenes/{scene_id}/{name}"


def scene_process_dir_key(scene_id: str) -> str:
    return f"process/scenes/{scene_id}"


def scene_raw_key(scene_id: str, name: str) -> str:
    return f"raw/scenes/{scene_id}/{name}"


# Known pipeline products (name -> artifact key suffix), shared by the
# result service. The names are the API-facing product vocabulary.
RESULT_ARTIFACT_KEYS: Dict[str, str] = {
    "depth": "dsm.npy",
    "dsm": "dsm.tif",
    "dsm_anchored": "dsm_anchored.tif",
    "preview": "dsm_preview.png",
}
