"""Semantic scene API schemas."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class SemanticLegend(BaseModel):
    building: str
    vegetation: str
    road: str
    water: str
    ground: str
    other: str


class SemanticMetaResponse(BaseModel):
    """Response for GET /api/v1/scenes/{id}/semantic — metadata + availability."""

    scene_id: str
    classes: list[str] = Field(
        default_factory=lambda: [
            "building", "vegetation", "road", "water", "ground", "other"
        ],
    )
    available: bool
    url: str | None = None
    labels_url: str | None = None
    confidence_url: str | None = None
    legend: SemanticLegend | None = None
    model: str | None = None
    checkpoint: str | None = None
    mean_confidence: float | None = None
    low_confidence_fraction: float | None = None
    class_fractions: dict[str, float] | None = None
