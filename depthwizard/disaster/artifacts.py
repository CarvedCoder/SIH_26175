"""Artifact persistence for disaster assessment results.

Writes:
    buildings.geojson       — building footprints with damage annotations
    building_mask.npy       — binary building mask [H, W] uint8
    building_confidence.npy — per-pixel building probability [H, W] float32
    buildings_preview.png   — visualization of detected buildings
    buildings_meta.json     — building detection metadata

    damage_buildings.geojson — buildings with damage class annotations
    damage_labels.npy       — per-pixel damage labels [H, W] uint8
    damage_confidence.npy   — per-pixel damage confidence [H, W] float32
    damage_preview.png      — color-coded damage visualization
    damage_meta.json        — damage assessment metadata

All artifacts follow the existing scene artifact store conventions.
File paths are never leaked to the frontend.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from .types import (
    BuildingDetection,
    DamageAssessment,
    DisasterResult,
    DAMAGE_CLASSES,
)


# Damage class colors (RGBA 0-255) for preview rendering
DAMAGE_COLORS = {
    "no-damage":    (76, 175, 80, 180),    # green
    "minor-damage": (255, 193, 7, 180),    # amber
    "major-damage": (255, 87, 34, 180),    # deep orange
    "destroyed":    (211, 47, 47, 200),    # red
}

DAMAGE_COLORS_HEX = {
    "no-damage":    "#4CAF50",
    "minor-damage": "#FFC107",
    "major-damage": "#FF5722",
    "destroyed":    "#D32F2F",
}

BUILDING_OUTLINE_COLOR = (52, 152, 219, 200)  # blue


def write_building_artifacts(
    output_dir: Path,
    buildings: list[BuildingDetection],
    building_mask: np.ndarray,
    building_confidence: np.ndarray,
    rgb: np.ndarray,
    *,
    georeferenced: bool = False,
    crs_string: str | None = None,
    transform: Any = None,
    mode: str = "post_only",
) -> dict[str, str]:
    """Write building detection artifacts to the output directory."""
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs: dict[str, str] = {}

    # GeoJSON
    geojson = _buildings_to_geojson(
        buildings,
        georeferenced=georeferenced,
        crs_string=crs_string,
        transform=transform,
    )
    geojson_path = output_dir / "buildings.geojson"
    with open(geojson_path, "w", encoding="utf-8") as f:
        json.dump(geojson, f, indent=2)
    outputs["buildings_geojson"] = str(geojson_path)

    # Mask
    np.save(output_dir / "building_mask.npy", building_mask)
    outputs["building_mask"] = str(output_dir / "building_mask.npy")

    # Confidence
    np.save(output_dir / "building_confidence.npy", building_confidence)
    outputs["building_confidence"] = str(output_dir / "building_confidence.npy")

    # Preview
    preview = _render_buildings_preview(rgb, buildings, building_mask)
    preview_path = output_dir / "buildings_preview.png"
    Image.fromarray(preview).save(preview_path)
    outputs["buildings_preview"] = str(preview_path)

    # Metadata
    meta = {
        "available": True,
        "count": len(buildings),
        "mode": mode,
        "georeferenced": georeferenced,
        "coordinate_space": "geographic" if georeferenced else "pixel_space",
        "crs": crs_string,
        "building_ids": [b.building_id for b in buildings],
    }
    meta_path = output_dir / "buildings_meta.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    outputs["buildings_meta"] = str(meta_path)

    return outputs


def write_damage_artifacts(
    output_dir: Path,
    result: DisasterResult,
    rgb: np.ndarray,
    building_mask: np.ndarray,
    damage_labels: np.ndarray,
    damage_confidence: np.ndarray,
    *,
    georeferenced: bool = False,
    crs_string: str | None = None,
    transform: Any = None,
) -> dict[str, str]:
    """Write damage assessment artifacts to the output directory."""
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs: dict[str, str] = {}

    # Damage GeoJSON (buildings with damage annotations)
    geojson = _damage_to_geojson(
        result.buildings,
        result.damages,
        georeferenced=georeferenced,
        crs_string=crs_string,
        transform=transform,
    )
    geojson_path = output_dir / "damage_buildings.geojson"
    with open(geojson_path, "w", encoding="utf-8") as f:
        json.dump(geojson, f, indent=2)
    outputs["damage_geojson"] = str(geojson_path)

    # Damage labels
    np.save(output_dir / "damage_labels.npy", damage_labels)
    outputs["damage_labels"] = str(output_dir / "damage_labels.npy")

    # Damage confidence
    np.save(output_dir / "damage_confidence.npy", damage_confidence)
    outputs["damage_confidence"] = str(output_dir / "damage_confidence.npy")

    # Damage preview
    preview = _render_damage_preview(rgb, building_mask, damage_labels)
    preview_path = output_dir / "damage_preview.png"
    Image.fromarray(preview).save(preview_path)
    outputs["damage_preview"] = str(preview_path)

    # Damage metadata
    meta = _damage_metadata(result, georeferenced=georeferenced)
    meta_path = output_dir / "damage_meta.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    outputs["damage_meta"] = str(meta_path)

    return outputs


def _buildings_to_geojson(
    buildings: list[BuildingDetection],
    *,
    georeferenced: bool = False,
    crs_string: str | None = None,
    transform: Any = None,
) -> dict:
    """Convert buildings to GeoJSON FeatureCollection."""
    features = []
    for b in buildings:
        # Transform pixel coords to geographic if georeferenced
        if georeferenced and transform is not None:
            coords = [
                _pixel_to_geo(x, y, transform)
                for x, y in b.polygon
            ]
            centroid = _pixel_to_geo(b.centroid_x, b.centroid_y, transform)
            # Keep the raster pixel-space rings alongside the geographic
            # geometry so pixel-based consumers (terrain-viewer clicks)
            # can do containment without knowing the CRS/transform.
            pixel_geometry = [b.polygon]
        else:
            coords = b.polygon
            centroid = (b.centroid_x, b.centroid_y)
            pixel_geometry = None

        feature = {
            "type": "Feature",
            "geometry": {
                "type": "Polygon",
                "coordinates": [coords],
            },
            "properties": {
                "building_id": b.building_id,
                "confidence": round(b.confidence, 4),
                "area_px": round(b.area_px, 1),
                "area_m2": round(b.area_m2, 1) if b.area_m2 is not None else None,
                "centroid": list(centroid),
                "coordinate_space": "geographic" if georeferenced else "pixel_space",
                "detection_source": b.source,
            },
        }
        if pixel_geometry is not None:
            feature["properties"]["pixel_geometry"] = pixel_geometry
        features.append(feature)

    return {
        "type": "FeatureCollection",
        "crs": (
            {"type": "name", "properties": {"name": crs_string}}
            if crs_string
            else None
        ),
        "features": features,
    }


def _damage_to_geojson(
    buildings: list[BuildingDetection],
    assessments: list[DamageAssessment],
    *,
    georeferenced: bool = False,
    crs_string: str | None = None,
    transform: Any = None,
) -> dict:
    """Convert buildings + damage to GeoJSON FeatureCollection."""
    assessment_map = {a.building_id: a for a in assessments}
    features = []

    for b in buildings:
        assessment = assessment_map.get(b.building_id)

        if georeferenced and transform is not None:
            coords = [
                _pixel_to_geo(x, y, transform)
                for x, y in b.polygon
            ]
            pixel_geometry = [b.polygon]
        else:
            coords = b.polygon
            pixel_geometry = None

        properties: dict[str, Any] = {
            "building_id": b.building_id,
            "confidence": round(b.confidence, 4),
            "area_px": round(b.area_px, 1),
            "area_m2": round(b.area_m2, 1) if b.area_m2 is not None else None,
            "coordinate_space": "geographic" if georeferenced else "pixel_space",
            "detection_source": b.source,
        }
        if pixel_geometry is not None:
            properties["pixel_geometry"] = pixel_geometry

        if assessment is not None:
            properties["damage_class"] = assessment.damage_class
            properties["damage_confidence"] = (
                round(assessment.confidence, 4)
                if assessment.confidence is not None
                else None
            )
            properties["damage_review"] = assessment.review_required
            properties["damage_mode"] = assessment.mode
            if assessment.probabilities:
                properties["damage_probabilities"] = {
                    k: round(v, 4) for k, v in assessment.probabilities.items()
                }

        features.append({
            "type": "Feature",
            "geometry": {
                "type": "Polygon",
                "coordinates": [coords],
            },
            "properties": properties,
        })

    return {
        "type": "FeatureCollection",
        "crs": (
            {"type": "name", "properties": {"name": crs_string}}
            if crs_string
            else None
        ),
        "features": features,
    }


def _damage_metadata(
    result: DisasterResult,
    *,
    georeferenced: bool = False,
) -> dict:
    """Create damage_meta.json content from actual results."""
    return {
        "available": True,
        "mode": result.mode,
        "building_model": result.building_model,
        "damage_model": result.damage_model,
        "building_count": result.building_count,
        "damage_counts": result.damage_counts,
        "review_count": result.review_count,
        "mean_confidence": (
            round(result.mean_confidence, 4)
            if result.mean_confidence is not None
            else None
        ),
        "recovered_destroyed_areas": sum(
            1 for b in result.buildings if b.source == "damage_map"
        ),
        "georeferenced": georeferenced,
        "damage_classes": list(DAMAGE_CLASSES),
        "colors": DAMAGE_COLORS_HEX,
    }


def _pixel_to_geo(
    px: float, py: float, transform: Any
) -> tuple[float, float]:
    """Convert pixel coordinates to geographic using rasterio transform."""
    geo_x = transform.c + px * transform.a + py * transform.b
    geo_y = transform.f + px * transform.d + py * transform.e
    return (geo_x, geo_y)


def _render_buildings_preview(
    rgb: np.ndarray,
    buildings: list[BuildingDetection],
    building_mask: np.ndarray,
) -> np.ndarray:
    """Render building outlines on top of the RGB image."""
    h, w = rgb.shape[:2]
    preview = rgb.copy()

    # Semi-transparent building overlay
    overlay = np.zeros((h, w, 4), dtype=np.uint8)
    building_pixels = building_mask > 0
    overlay[building_pixels] = BUILDING_OUTLINE_COLOR

    # Blend
    alpha = overlay[:, :, 3:4].astype(np.float32) / 255.0
    blended = (
        preview.astype(np.float32) * (1.0 - alpha)
        + overlay[:, :, :3].astype(np.float32) * alpha
    ).astype(np.uint8)

    return blended


def _render_damage_preview(
    rgb: np.ndarray,
    building_mask: np.ndarray,
    damage_labels: np.ndarray,
) -> np.ndarray:
    """Render damage-colored buildings on top of the RGB image."""
    h, w = rgb.shape[:2]
    preview = rgb.copy()
    overlay = np.zeros((h, w, 4), dtype=np.uint8)

    for cls_idx, cls_name in enumerate(DAMAGE_CLASSES):
        label_val = cls_idx + 1  # 1-based in damage_labels
        mask = damage_labels == label_val
        if mask.any():
            color = DAMAGE_COLORS[cls_name]
            overlay[mask] = color

    # Blend
    alpha = overlay[:, :, 3:4].astype(np.float32) / 255.0
    blended = (
        preview.astype(np.float32) * (1.0 - alpha)
        + overlay[:, :, :3].astype(np.float32) * alpha
    ).astype(np.uint8)

    return blended
