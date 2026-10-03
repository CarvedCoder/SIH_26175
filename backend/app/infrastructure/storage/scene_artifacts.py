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
S3/RustFS ArtifactStore later changes this module, not its consumers.

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
#
# CONTRACT NOTE — depth vs dsm vs preview (do not blur these):
#   "depth"   = the predicted metric surface array (dsm.npy): the model's
#               monocular-depth-derived height field in metres. There is
#               no separate "raw depth" product; AGL/agl_raw.npy is the
#               un-anchored model signal and is NOT served as depth.
#   "dsm"     = the georeferenced GeoTIFF twin of the same surface
#               (only exists for scenes with a CRS).
#   "preview" = dsm_preview.png — a Matplotlib diagnostic figure
#               (axes/colourmap/colourbar) for HUMAN download only. It
#               must NEVER be served as an interactive WebGL layer
#               texture; viewer layers use the Pillow-generated
#               greyscale data PNGs (dsm_layer.png / depth_layer.png,
#               materialised on demand by terrain_service).
RESULT_ARTIFACT_KEYS: Dict[str, str] = {
    "depth": "dsm.npy",
    "dsm": "dsm.tif",
    "dsm_anchored": "dsm_anchored.tif",
    "preview": "dsm_preview.png",
    "semantic_labels": "semantic_labels.npy",
    "semantic_probs": "semantic_probs.npy",
    "semantic_confidence": "semantic_confidence.npy",
    "semantic_map": "semantic_map.png",
    "semantic_meta": "semantic_meta.json",
    # Disaster assessment artifacts (building detection + damage)
    "buildings_geojson": "buildings.geojson",
    "building_mask": "building_mask.npy",
    "building_confidence": "building_confidence.npy",
    "buildings_preview": "buildings_preview.png",
    "buildings_meta": "buildings_meta.json",
    "damage_geojson": "damage_buildings.geojson",
    "damage_labels": "damage_labels.npy",
    "damage_confidence": "damage_confidence.npy",
    "damage_preview": "damage_preview.png",
    "damage_meta": "damage_meta.json",
    # Geometry-aware 3D building reconstruction (footprints + DSM heights)
    "buildings3d": "buildings3d.json",
    "buildings3d_preview": "buildings3d_preview.png",
    # Region refinement product (isolated crop re-inference)
    "refined_dsm": "refined_dsm.npy",
    # TerraHeight-S backend (AGL product + provenance; .npy twin exists for
    # non-georeferenced scenes where no .tif is written)
    "terraheight_agl": "terraheight_agl.tif",
    "terraheight_agl_npy": "terraheight_agl.npy",
    "terraheight_preview": "terraheight_preview.png",
    "terraheight_meta": "terraheight_meta.json",
}

