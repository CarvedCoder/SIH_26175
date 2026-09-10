from __future__ import annotations

from pydantic import BaseModel


class ReferenceResponse(BaseModel):
    """Reference elevation metadata for a scene (GET /scenes/{id}/reference).

    Honest by construction: when no reference DEM exists for the scene,
    ``available`` is False and every URL field is null — never fabricated.
    """

    scene_id: str
    available: bool = False
    name: str | None = None
    download_url: str | None = None
    visualization_url: str | None = None
    crs: str | None = None
    units: str | None = None
