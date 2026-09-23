"""Semantic segmentation API routes.

GET /api/v1/scenes/{id}/semantic              metadata + availability
GET /api/v1/scenes/{id}/results/semantic      semantic_map.png (visualization)
GET /api/v1/scenes/{id}/results/semantic-labels    class-ID array for GPU texture
GET /api/v1/scenes/{id}/results/semantic-confidence  confidence array
"""

from __future__ import annotations

import json
from pathlib import Path

from fastapi import APIRouter
from fastapi.responses import FileResponse

from backend.app.api.routes._deps import require_scene
from backend.app.core.errors import AppError
from backend.app.schemas.semantic import SemanticLegend, SemanticMetaResponse

router = APIRouter(
    prefix="/api/v1/scenes",
    tags=["Semantic"],
)


def _semantic_files(scene_id: str) -> dict[str, Path]:
    """Return existing semantic artifact paths for a scene."""
    from backend.app.services.result_service import result_service

    files = result_service.get_result_files(scene_id)
    return {
        k: v for k, v in files.items()
        if k.startswith("semantic_")
    }


@router.get("/{scene_id}/semantic", response_model=SemanticMetaResponse)
async def semantic_metadata(scene_id: str):
    """Semantic segmentation metadata and availability for a scene."""
    require_scene(scene_id)
    sem_files = _semantic_files(scene_id)

    if not sem_files.get("semantic_labels"):
        return SemanticMetaResponse(
            scene_id=scene_id,
            available=False,
        )

    from backend.app.services.terrain_service import terrain_service

    meta_path = sem_files.get("semantic_meta")
    meta = {}
    if meta_path and meta_path.exists():
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)

    legend_dict = meta.get("legend", {})
    legend = None
    if legend_dict and all(
        k in legend_dict
        for k in ("building", "vegetation", "road", "water", "ground", "other")
    ):
        legend = SemanticLegend(**legend_dict)

    base = f"/api/v1/scenes/{scene_id}/results"
    return SemanticMetaResponse(
        scene_id=scene_id,
        available=True,
        url=terrain_service._artifact_url(
            scene_id, "semantic_map.png", f"{base}/semantic"
        ),
        labels_url=terrain_service._artifact_url(
            scene_id, "semantic_labels.npy", f"{base}/semantic-labels"
        ),
        confidence_url=terrain_service._artifact_url(
            scene_id, "semantic_confidence.npy", f"{base}/semantic-confidence"
        ),
        legend=legend,
        model=meta.get("model"),
        checkpoint=meta.get("checkpoint"),
        mean_confidence=meta.get("mean_confidence"),
        low_confidence_fraction=meta.get("low_confidence_fraction"),
        class_fractions=meta.get("class_fractions"),
    )


@router.get("/{scene_id}/results/semantic")
async def semantic_layer(scene_id: str):
    """Semantic segmentation visualization PNG for the terrain viewer."""
    require_scene(scene_id)
    sem_files = _semantic_files(scene_id)
    path = sem_files.get("semantic_map")
    if path is None or not path.exists():
        raise AppError(
            status_code=404,
            code="SEMANTIC_NOT_AVAILABLE",
            message="Semantic segmentation is not available for this scene.",
            details={"scene_id": scene_id},
            recoverable=True,
        )
    return _serve_artifact(scene_id, path)


@router.get("/{scene_id}/results/semantic-labels")
async def semantic_labels(scene_id: str):
    """Semantic class-ID array (uint8 .npy) for GPU texture / browser use."""
    require_scene(scene_id)
    sem_files = _semantic_files(scene_id)
    path = sem_files.get("semantic_labels")
    if path is None or not path.exists():
        raise AppError(
            status_code=404,
            code="SEMANTIC_NOT_AVAILABLE",
            message="Semantic labels are not available for this scene.",
            details={"scene_id": scene_id},
            recoverable=True,
        )
    return _serve_artifact(scene_id, path)


@router.get("/{scene_id}/results/semantic-confidence")
async def semantic_confidence(scene_id: str):
    """Per-pixel semantic confidence array (float16 .npy)."""
    require_scene(scene_id)
    sem_files = _semantic_files(scene_id)
    path = sem_files.get("semantic_confidence")
    if path is None or not path.exists():
        raise AppError(
            status_code=404,
            code="SEMANTIC_NOT_AVAILABLE",
            message="Semantic confidence data is not available for this scene.",
            details={"scene_id": scene_id},
            recoverable=True,
        )
    return _serve_artifact(scene_id, path)


def _serve_artifact(scene_id: str, path: Path):
    """Serve an artifact via presigned URL or direct file response."""
    from backend.app.storage.service import storage_service

    if storage_service.is_object_store:
        url = storage_service.presign_artifact(scene_id, path)
        if url is not None:
            from fastapi.responses import RedirectResponse
            return RedirectResponse(url=url, status_code=307)
    return FileResponse(path=path, filename=path.name)
