"""Canonical data types for the disaster assessment pipeline.

These types are the internal representation — independent of the
6-class semantic segmentation taxonomy and independent of the
API response schemas. Damage classes are a SEPARATE concept from
semantic classes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Optional


# Canonical damage class labels — order matches the damage model's
# 4-class logit output (index 0–3).
DAMAGE_CLASSES: tuple[str, ...] = (
    "no-damage",
    "minor-damage",
    "major-damage",
    "destroyed",
)

DAMAGE_CLASS_TO_INDEX: dict[str, int] = {
    cls: i for i, cls in enumerate(DAMAGE_CLASSES)
}


@dataclass
class BuildingDetection:
    """A single detected building footprint."""

    building_id: str                           # deterministic, e.g. "B001"
    polygon: list[tuple[float, float]]         # list of (x, y) coords
    area_px: float                             # area in pixel² units
    centroid_x: float                          # centroid x (pixel or geo)
    centroid_y: float                          # centroid y (pixel or geo)
    confidence: float                          # max probability within footprint
    area_m2: float | None = None               # area in m² (None if non-georef)
    bbox: tuple[int, int, int, int] | None = None  # (x_min, y_min, x_max, y_max) pixel bbox
    # Where this footprint came from: "detector" (building localization
    # model) or "damage_map" (recovered from the damage model's own
    # destroyed-class output — the detector cannot see destroyed/rubble
    # structures). Damage-map footprints may merge several structures.
    source: str = "detector"


@dataclass
class DamageAssessment:
    """Damage assessment for a single building."""

    building_id: str
    damage_class: Literal[
        "no-damage",
        "minor-damage",
        "major-damage",
        "destroyed",
    ]
    confidence: float | None                   # None if model doesn't expose it
    probabilities: dict[str, float] | None     # per-class probabilities
    review_required: bool | None               # None if not determinable
    mode: Literal["post_only", "pre_post"]


@dataclass
class DisasterResult:
    """Complete disaster assessment result for a scene."""

    buildings: list[BuildingDetection] = field(default_factory=list)
    damages: list[DamageAssessment] = field(default_factory=list)
    mode: Literal["post_only", "pre_post"] = "post_only"
    building_model: str = ""
    damage_model: str = ""
    georeferenced: bool = False
    building_mask: object = None   # np.ndarray [H, W] uint8 — set after detection
    building_confidence: object = None  # np.ndarray [H, W] float32
    damage_labels: object = None   # np.ndarray [H, W] uint8
    damage_confidence_map: object = None  # np.ndarray [H, W] float32

    @property
    def building_count(self) -> int:
        return len(self.buildings)

    @property
    def damage_counts(self) -> dict[str, int]:
        counts = {cls: 0 for cls in DAMAGE_CLASSES}
        for d in self.damages:
            counts[d.damage_class] = counts.get(d.damage_class, 0) + 1
        return counts

    @property
    def review_count(self) -> int:
        return sum(1 for d in self.damages if d.review_required)

    @property
    def mean_confidence(self) -> float | None:
        confidences = [d.confidence for d in self.damages if d.confidence is not None]
        if not confidences:
            return None
        return sum(confidences) / len(confidences)
