"""Route-assist routes (jury round 1 + 2).

POST /api/v1/scenes/{id}/route/assess   fleet verdicts + recommended path
GET  /api/v1/scenes/{id}/route/heatmap  mini heat map URL + risk stats
GET  /api/v1/scenes/{id}/results/passability  heat map PNG (viewer layer)

All routes enforce scene ownership via the shared require_scene guard.
"""

from __future__ import annotations

from fastapi import APIRouter, Query
from fastapi.responses import FileResponse

from backend.app.api.routes._deps import require_scene
from backend.app.core.errors import AppError
from backend.app.schemas.route import (
    HeatmapResponse,
    RouteAssessRequest,
    RouteAssessResponse,
    VehicleAssessment,
)
from backend.app.services.route_service import VEHICLE_PROFILES, route_service

router = APIRouter(
    prefix="/api/v1/scenes",
    tags=["Route Assist"],
)


def _require_results(scene_id: str) -> None:
    try:
        route_service.load_dsm(scene_id)
    except FileNotFoundError as exc:
        raise AppError(
            status_code=404,
            code="RESULTS_NOT_FOUND",
            message="No DSM results exist for this scene yet — process it first.",
            details={"scene_id": scene_id},
            recoverable=True,
        ) from exc


@router.post("/{scene_id}/route/assess", response_model=RouteAssessResponse)
async def assess_route(scene_id: str, request: RouteAssessRequest):
    """Assess whether emergency vehicles can move from start to end.

    Returns one verdict per requested vehicle profile (CAN_GO / CAUTION /
    CANNOT_GO) with the recommended path, blocking reasons, and — when the
    destination is on unreachable ground — the nearest reachable detour
    point. Geometry-based analysis only (see the response disclaimer)."""
    require_scene(scene_id)
    _require_results(scene_id)

    unknown = [v for v in request.vehicles if v not in VEHICLE_PROFILES]
    if unknown:
        raise AppError(
            status_code=400,
            code="INVALID_VEHICLE",
            message=(
                "Unknown vehicle profile(s): "
                + ", ".join(unknown)
                + f". Available: {', '.join(sorted(VEHICLE_PROFILES))}."
            ),
            recoverable=True,
        )

    payload = route_service.assess(
        scene_id,
        (request.start.x, request.start.y),
        (request.end.x, request.end.y),
        request.vehicles,
    )
    payload["vehicles"] = [
        VehicleAssessment(**v) for v in payload["vehicles"]
    ]
    return RouteAssessResponse(**payload)


@router.get("/{scene_id}/route/heatmap", response_model=HeatmapResponse)
async def route_heatmap(
    scene_id: str,
    vehicle: str = Query("fire_truck"),
):
    """Mini heat map for rapid visual recognition (jury round 2): traffic-
    light passability map + the fraction of the scene that is blocked /
    caution for the selected vehicle."""
    require_scene(scene_id)
    _require_results(scene_id)
    if vehicle not in VEHICLE_PROFILES:
        raise AppError(
            status_code=400,
            code="INVALID_VEHICLE",
            message=f"Unknown vehicle profile: {vehicle}.",
            recoverable=True,
        )

    try:
        path = route_service.heatmap_path(scene_id, vehicle)
        stats = route_service.heatmap_stats(scene_id, vehicle)
    except (FileNotFoundError, ValueError) as exc:
        raise AppError(
            status_code=404,
            code="RESULTS_NOT_FOUND",
            message="Heat map could not be generated for this scene.",
            details={"scene_id": scene_id, "reason": str(exc)},
            recoverable=True,
        ) from exc

    from backend.app.services.terrain_service import terrain_service

    url = terrain_service._artifact_url(
        scene_id, path.name, f"/api/v1/scenes/{scene_id}/results/passability"
    )
    return HeatmapResponse(scene_id=scene_id, url=url, **stats)


@router.get("/{scene_id}/results/passability")
async def passability_layer(
    scene_id: str,
    vehicle: str = Query("fire_truck"),
):
    """The passability heat map as a viewer layer texture (traffic-light
    PNG, RGB colormap so no shader remapping is needed)."""
    require_scene(scene_id)
    _require_results(scene_id)
    if vehicle not in VEHICLE_PROFILES:
        raise AppError(
            status_code=400,
            code="INVALID_VEHICLE",
            message=f"Unknown vehicle profile: {vehicle}.",
            recoverable=True,
        )

    try:
        path = route_service.heatmap_path(scene_id, vehicle)
    except (FileNotFoundError, ValueError) as exc:
        raise AppError(
            status_code=404,
            code="RESULT_NOT_FOUND",
            message="The passability layer is not available for this scene.",
            details={"scene_id": scene_id, "reason": str(exc)},
            recoverable=True,
        ) from exc

    from backend.app.core.auth import current_user
    from backend.app.storage.service import storage_service

    if storage_service.is_object_store:
        url = storage_service.presign_artifact(scene_id, path)
        if url is not None:
            from fastapi.responses import RedirectResponse

            return RedirectResponse(url=url, status_code=307)
    return FileResponse(path=path, filename=path.name)


@router.get("/{scene_id}/results/route-risk")
async def route_risk_layer(
    scene_id: str,
    vehicle: str = Query("fire_truck"),
):
    """The route-risk heat map (geometry + 6-class semantics) as a viewer layer texture."""
    require_scene(scene_id)
    _require_results(scene_id)
    if vehicle not in VEHICLE_PROFILES:
        raise AppError(
            status_code=400,
            code="INVALID_VEHICLE",
            message=f"Unknown vehicle profile: {vehicle}.",
            recoverable=True,
        )

    try:
        path = route_service.route_risk_heatmap_path(scene_id, vehicle)
    except (FileNotFoundError, ValueError) as exc:
        raise AppError(
            status_code=404,
            code="RESULT_NOT_FOUND",
            message="The route-risk layer is not available for this scene.",
            details={"scene_id": scene_id, "reason": str(exc)},
            recoverable=True,
        ) from exc

    from backend.app.storage.service import storage_service

    if storage_service.is_object_store:
        url = storage_service.presign_artifact(scene_id, path)
        if url is not None:
            from fastapi.responses import RedirectResponse

            return RedirectResponse(url=url, status_code=307)
    return FileResponse(path=path, filename=path.name)

