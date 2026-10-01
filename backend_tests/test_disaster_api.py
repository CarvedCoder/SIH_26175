"""API endpoint tests for building detection and damage assessment layers.

Covers:
- GET /api/v1/scenes/{id}/buildings (availability + metadata)
- GET /api/v1/scenes/{id}/damage (availability + metadata)
- GET /api/v1/scenes/{id}/results/buildings-geojson and /buildings/geojson
- GET /api/v1/scenes/{id}/results/buildings-preview and /buildings/preview
- GET /api/v1/scenes/{id}/results/damage-geojson and /damage/geojson
- GET /api/v1/scenes/{id}/results/damage-preview and /damage/preview
- GET /api/v1/scenes/{id}/buildings/mask
- GET /api/v1/scenes/{id}/damage/labels
- POST /api/v1/scenes/{id}/route damage integration
"""

from __future__ import annotations

import io
import json
from pathlib import Path

import numpy as np
import pytest

from depthwizard.disaster.types import (
    BuildingDetection,
    DamageAssessment,
    DisasterResult,
)
from depthwizard.disaster.artifacts import (
    write_building_artifacts,
    write_damage_artifacts,
)


@pytest.fixture()
def disaster_scene(client, processed_scene):
    """Augment a processed scene with synthetic disaster assessment artifacts."""
    scene_id = processed_scene["scene"]["scene_id"]
    from backend.app.core.paths import SCENES_OUTPUT_DIR

    scene_dir = SCENES_OUTPUT_DIR / scene_id

    # Create synthetic buildings
    buildings = [
        BuildingDetection(
            building_id="B001",
            polygon=[(20, 20), (50, 20), (50, 50), (20, 50)],
            centroid_x=35.0,
            centroid_y=35.0,
            area_px=900.0,
            confidence=0.92,
            bbox=(20, 20, 50, 50),
        ),
        BuildingDetection(
            building_id="B002",
            polygon=[(100, 100), (140, 100), (140, 140), (100, 140)],
            centroid_x=120.0,
            centroid_y=120.0,
            area_px=1600.0,
            confidence=0.88,
            bbox=(100, 100, 140, 140),
        ),
    ]

    mask = np.zeros((256, 256), dtype=np.uint8)
    mask[20:50, 20:50] = 1
    mask[100:140, 100:140] = 1
    conf = mask.astype(np.float32) * 0.90
    rgb = np.full((256, 256, 3), 120, dtype=np.uint8)

    write_building_artifacts(scene_dir, buildings, mask, conf, rgb)

    # Damage assessments
    damages = [
        DamageAssessment(
            building_id="B001",
            damage_class="destroyed",
            confidence=0.95,
            probabilities={"no-damage": 0.01, "minor-damage": 0.02, "major-damage": 0.02, "destroyed": 0.95},
            review_required=False,
            mode="post_only",
        ),
        DamageAssessment(
            building_id="B002",
            damage_class="minor-damage",
            confidence=0.75,
            probabilities={"no-damage": 0.20, "minor-damage": 0.75, "major-damage": 0.05, "destroyed": 0.00},
            review_required=True,
            mode="post_only",
        ),
    ]

    result = DisasterResult(mode="post_only", buildings=buildings, damages=damages)
    dmg_labels = np.zeros((256, 256), dtype=np.uint8)
    dmg_labels[20:50, 20:50] = 4    # destroyed (class index 3 -> 4)
    dmg_labels[100:140, 100:140] = 2  # minor-damage (class index 1 -> 2)
    dmg_conf = conf

    write_damage_artifacts(scene_dir, result, rgb, mask, dmg_labels, dmg_conf)

    return {"scene_id": scene_id, "scene_dir": scene_dir}


def test_disaster_unknown_scene_404(client):
    res_bld = client.get("/api/v1/scenes/scene_000000000000/buildings")
    assert res_bld.status_code == 404
    res_dmg = client.get("/api/v1/scenes/scene_000000000000/damage")
    assert res_dmg.status_code == 404


def test_disaster_metadata_unavailable_scene(client, processed_scene):
    scene_id = processed_scene["scene"]["scene_id"]

    res_bld = client.get(f"/api/v1/scenes/{scene_id}/buildings")
    assert res_bld.status_code == 200
    bld_data = res_bld.json()
    assert bld_data["available"] is False
    assert bld_data["count"] == 0

    res_dmg = client.get(f"/api/v1/scenes/{scene_id}/damage")
    assert res_dmg.status_code == 200
    dmg_data = res_dmg.json()
    assert dmg_data["available"] is False
    assert dmg_data["building_count"] == 0


def test_disaster_artifacts_unavailable_404(client, processed_scene):
    scene_id = processed_scene["scene"]["scene_id"]

    res = client.get(f"/api/v1/scenes/{scene_id}/results/buildings-geojson")
    assert res.status_code == 404
    assert res.json()["error"]["code"] == "BUILDINGS_NOT_AVAILABLE"

    res = client.get(f"/api/v1/scenes/{scene_id}/results/damage-geojson")
    assert res.status_code == 404
    assert res.json()["error"]["code"] == "DAMAGE_NOT_AVAILABLE"


def test_disaster_metadata_available_scene(client, disaster_scene):
    scene_id = disaster_scene["scene_id"]

    res_bld = client.get(f"/api/v1/scenes/{scene_id}/buildings")
    assert res_bld.status_code == 200
    bld_data = res_bld.json()
    assert bld_data["available"] is True
    assert bld_data["count"] == 2
    assert "buildings-preview" in bld_data["url"]
    assert "buildings-geojson" in bld_data["geojson_url"]

    res_dmg = client.get(f"/api/v1/scenes/{scene_id}/damage")
    assert res_dmg.status_code == 200
    dmg_data = res_dmg.json()
    assert dmg_data["available"] is True
    assert dmg_data["building_count"] == 2
    assert dmg_data["mode"] == "post_only"
    assert dmg_data["damage_counts"]["destroyed"] == 1
    assert dmg_data["damage_counts"]["minor-damage"] == 1
    assert dmg_data["review_count"] == 1
    assert dmg_data["url"] == dmg_data["preview_url"]


def test_disaster_artifacts_endpoints(client, disaster_scene):
    scene_id = disaster_scene["scene_id"]

    # Buildings GeoJSON
    res = client.get(f"/api/v1/scenes/{scene_id}/results/buildings-geojson")
    assert res.status_code == 200
    geojson = res.json()
    assert geojson["type"] == "FeatureCollection"
    assert len(geojson["features"]) == 2
    assert geojson["features"][0]["properties"]["building_id"] == "B001"

    # Alias /buildings/geojson
    res_alias = client.get(f"/api/v1/scenes/{scene_id}/buildings/geojson")
    assert res_alias.status_code == 200
    assert res_alias.json()["type"] == "FeatureCollection"

    # Damage GeoJSON
    res_dmg_geojson = client.get(f"/api/v1/scenes/{scene_id}/results/damage-geojson")
    assert res_dmg_geojson.status_code == 200
    dmg_geojson = res_dmg_geojson.json()
    assert dmg_geojson["type"] == "FeatureCollection"
    assert dmg_geojson["features"][0]["properties"]["damage_class"] == "destroyed"
    assert dmg_geojson["features"][0]["properties"]["damage_mode"] == "post_only"

    # Previews
    res_bld_prev = client.get(f"/api/v1/scenes/{scene_id}/results/buildings-preview")
    assert res_bld_prev.status_code == 200
    assert "image/png" in res_bld_prev.headers["content-type"]

    res_dmg_prev = client.get(f"/api/v1/scenes/{scene_id}/results/damage-preview")
    assert res_dmg_prev.status_code == 200
    assert "image/png" in res_dmg_prev.headers["content-type"]

    # Rasters
    res_mask = client.get(f"/api/v1/scenes/{scene_id}/buildings/mask")
    assert res_mask.status_code == 200
    mask_arr = np.load(io.BytesIO(res_mask.content))
    assert mask_arr.shape == (256, 256)

    res_labels = client.get(f"/api/v1/scenes/{scene_id}/damage/labels")
    assert res_labels.status_code == 200
    labels_arr = np.load(io.BytesIO(res_labels.content))
    assert labels_arr.shape == (256, 256)
    assert (labels_arr == 4).any()  # destroyed class present


def test_route_assist_incorporates_damage(client, disaster_scene):
    scene_id = disaster_scene["scene_id"]

    payload = {
        "start": {"x": 5, "y": 5},
        "end": {"x": 200, "y": 200},
        "vehicles": ["fire_truck", "rescue_atv"],
    }
    res = client.post(f"/api/v1/scenes/{scene_id}/route", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert data["damage_available"] is True
    for v in data["vehicles"]:
        assert "damage_aware" in v
        assert v["damage_aware"] is True
        assert "damage_risk_fraction" in v
