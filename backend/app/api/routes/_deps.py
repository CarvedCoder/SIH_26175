"""Shared route dependencies — the single home for cross-route guards.

Previously these lived in routes/scenes.py and other route modules
imported from it (route-to-route imports: an anti-pattern). The guard
logic itself is in the application layer; this module is the FastAPI-side
alias so route modules have one clean import.
"""

from __future__ import annotations

from backend.app.application.scenes.service import scene_service
from backend.app.core.auth import ensure_owner


def require_scene(scene_id: str) -> None:
    """Existence + ownership guard for every scene-scoped route (jobs,
    terrain, results, export, reference, validation all use this).

    Raises the typed 404/400 unless the scene exists, then 403 unless the
    verified request user owns it. Ownership lives in the scene's SQL row
    (the auth-era record); a raw directory without a row is a
    pre-migration/local scene and stays accessible (local development)."""
    scene_service.require_scene(scene_id)
    from backend.app.db.database import session_scope
    from backend.app.db.models import SceneRow

    with session_scope() as session:
        row = session.get(SceneRow, scene_id)
        owner_id = row.owner_id if row is not None else None
    if owner_id is not None:
        ensure_owner(owner_id, f"scene:{scene_id}")
