"""Unit and integration tests for semantic-aware routing in route_service.

Verifies:
1. Probability volume downsampling preserves probability distribution and sums.
2. Semantic cost weighting (road preference, vehicle-specific vegetation penalties).
3. Hard blocking for buildings and water bodies even on flat terrain.
4. Path diversion around semantic obstacles (buildings) onto roads.
5. Honest fallback when semantic artifacts are absent.
6. Combined route-risk heatmap generation.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from backend.app.services.route_service import (
    SemanticConfig,
    VEHICLE_PROFILES,
    _block_downsample_probs,
    _combined_cost,
    _semantic_cost,
    classify_grid,
    route_service,
)
from depthwizard.datasets.semantics import NUM_PROJECT_CLASSES


@pytest.fixture()
def scene_env(monkeypatch, tmp_path):
    """Point backend paths at temporary directories."""
    import backend.app.core.paths as paths

    monkeypatch.setattr(paths, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(paths, "DATA_DIR", tmp_path / "data")
    monkeypatch.setattr(
        paths, "SCENES_OUTPUT_DIR", tmp_path / "data" / "output" / "scenes"
    )
    paths.ensure_directories()
    yield tmp_path


def install_scene_data(
    monkeypatch,
    scene_id: str,
    dsm: np.ndarray,
    sem_probs: np.ndarray | None = None,
    gsd: float | None = 0.5,
):
    """Stub the route_service's DSM, semantics, and GSD accessors."""
    def fake_load_dsm(_self, sid):
        assert sid == scene_id
        return dsm

    def fake_load_semantics(_self, sid):
        assert sid == scene_id
        if sem_probs is None:
            return None, None, None
        labels = np.argmax(sem_probs, axis=0).astype(np.uint8)
        conf = np.max(sem_probs, axis=0).astype(np.float32)
        return sem_probs, labels, conf

    def fake_gsd(_self, sid):
        return gsd

    monkeypatch.setattr(route_service, "load_dsm", fake_load_dsm.__get__(route_service))
    monkeypatch.setattr(route_service, "load_semantics", fake_load_semantics.__get__(route_service))
    monkeypatch.setattr(route_service, "gsd_for", fake_gsd.__get__(route_service))


# ---------------------------------------------------------------------------
# Unit tests on helper functions
# ---------------------------------------------------------------------------

def test_block_downsample_probs():
    # Shape [6, 16, 16] with class 2 (road) dominant
    probs = np.zeros((NUM_PROJECT_CLASSES, 16, 16), dtype=np.float32)
    probs[2, :, :] = 1.0

    down = _block_downsample_probs(probs, factor=4)
    assert down.shape == (6, 4, 4)
    assert np.allclose(down[2, :, :], 1.0)
    assert np.allclose(down.sum(axis=0), 1.0)


def test_semantic_cost_road_preference():
    # Pure road [class 2 = 1.0] vs pure ground [class 4 = 1.0]
    p_road = np.zeros((6, 1, 1), dtype=np.float32)
    p_road[2, 0, 0] = 1.0
    labels_road = np.array([[2]], dtype=np.uint8)
    conf_road = np.array([[1.0]], dtype=np.float32)

    p_ground = np.zeros((6, 1, 1), dtype=np.float32)
    p_ground[4, 0, 0] = 1.0
    labels_ground = np.array([[4]], dtype=np.uint8)
    conf_ground = np.array([[1.0]], dtype=np.float32)

    fire_truck = VEHICLE_PROFILES["fire_truck"]
    cfg = SemanticConfig(enabled=True)

    cost_road, blocked_road = _semantic_cost(p_road, labels_road, conf_road, fire_truck, cfg)
    cost_ground, blocked_ground = _semantic_cost(p_ground, labels_ground, conf_ground, fire_truck, cfg)

    # Road should be significantly cheaper than ground
    assert cost_road[0, 0] < cost_ground[0, 0]
    assert abs(cost_road[0, 0] - 0.65) < 1e-4
    assert abs(cost_ground[0, 0] - 1.0) < 1e-4
    assert not blocked_road[0, 0]
    assert not blocked_ground[0, 0]


def test_semantic_cost_vegetation_penalty_by_vehicle():
    p_veg = np.zeros((6, 1, 1), dtype=np.float32)
    p_veg[1, 0, 0] = 1.0
    labels_veg = np.array([[1]], dtype=np.uint8)
    conf_veg = np.array([[1.0]], dtype=np.float32)

    fire_truck = VEHICLE_PROFILES["fire_truck"]
    rescue_atv = VEHICLE_PROFILES["rescue_atv"]
    cfg = SemanticConfig(enabled=True)

    cost_fire, _ = _semantic_cost(p_veg, labels_veg, conf_veg, fire_truck, cfg)
    cost_atv, _ = _semantic_cost(p_veg, labels_veg, conf_veg, rescue_atv, cfg)

    # Fire truck should be penalized much more heavily in vegetation than a rescue ATV
    assert cost_fire[0, 0] > cost_atv[0, 0]
    assert cost_fire[0, 0] >= 5.0
    assert cost_atv[0, 0] <= 3.0


def test_semantic_cost_building_and_water_hard_block():
    p_bldg = np.zeros((6, 1, 1), dtype=np.float32)
    p_bldg[0, 0, 0] = 0.95  # high-confidence building > 0.90
    p_bldg[4, 0, 0] = 0.05
    labels_bldg = np.array([[0]], dtype=np.uint8)
    conf_bldg = np.array([[0.95]], dtype=np.float32)

    p_water = np.zeros((6, 1, 1), dtype=np.float32)
    p_water[3, 0, 0] = 0.92  # high-confidence water > 0.90
    p_water[4, 0, 0] = 0.08
    labels_water = np.array([[3]], dtype=np.uint8)
    conf_water = np.array([[0.92]], dtype=np.float32)

    fire_truck = VEHICLE_PROFILES["fire_truck"]
    cfg = SemanticConfig(enabled=True, hard_block_threshold=0.90)

    _, blocked_bldg = _semantic_cost(p_bldg, labels_bldg, conf_bldg, fire_truck, cfg)
    _, blocked_water = _semantic_cost(p_water, labels_water, conf_water, fire_truck, cfg)

    assert blocked_bldg[0, 0]
    assert blocked_water[0, 0]


def test_combined_cost_geometry_and_semantics():
    p_road = np.zeros((6, 4, 4), dtype=np.float32)
    p_road[2, :, :] = 1.0
    labels_road = np.full((4, 4), 2, dtype=np.uint8)
    conf_road = np.ones((4, 4), dtype=np.float32)

    p_bldg = np.zeros((6, 4, 4), dtype=np.float32)
    p_bldg[0, :, :] = 0.95
    labels_bldg = np.zeros((4, 4), dtype=np.uint8)
    conf_bldg = np.full((4, 4), 0.95, dtype=np.float32)

    fire_truck = VEHICLE_PROFILES["fire_truck"]
    cfg = SemanticConfig(enabled=True, hard_block_threshold=0.90)

    # Flat terrain -> all passable
    dsm_flat = np.full((4, 4), 10.0, dtype=np.float32)
    grid_flat = classify_grid(dsm_flat, 0.5, fire_truck)

    cost_road, blocked_road = _combined_cost(
        grid_flat, p_road, labels_road, conf_road, cfg
    )
    cost_bldg, blocked_bldg = _combined_cost(
        grid_flat, p_bldg, labels_bldg, conf_bldg, cfg
    )

    # Road on flat ground is cheaper than default (1.0) and unblocked
    assert cost_road[0, 0] < 1.0
    assert not blocked_road[0, 0]

    # Building on flat ground is hard blocked
    assert blocked_bldg[0, 0]


# ---------------------------------------------------------------------------
# Pathfinding integration tests
# ---------------------------------------------------------------------------

def test_semantic_aware_routing_diverts_around_building(scene_env, monkeypatch):
    """Flat terrain with a building in the direct line of sight.

    Start: (10, 10), Goal: (10, 110)
    Direct path would cross y=40..70.
    A building is placed at y=40..70, x=0..60.
    A road is placed to the right at x=75..85.
    The pathfinder must route around the building through the road or ground.
    """
    size = 128
    dsm = np.full((size, size), 10.0, dtype=np.float32)  # Perfectly flat

    # Default all terrain to ground (class 4)
    probs = np.zeros((6, size, size), dtype=np.float32)
    probs[4, :, :] = 1.0

    # Place building block across y=40..70, x=0..70
    probs[4, 40:70, :70] = 0.0
    probs[0, 40:70, :70] = 0.95  # Building

    # Place road strip at x=75..85 along full height
    probs[4, :, 75:85] = 0.0
    probs[2, :, 75:85] = 0.95  # Road

    scene_id = "scene_sem_001"
    install_scene_data(monkeypatch, scene_id, dsm, sem_probs=probs)

    result = route_service.assess(
        scene_id, (10, 10), (10, 110), ["fire_truck"]
    )
    assert result["semantic_available"] is True
    vehicle = result["vehicles"][0]
    assert vehicle["verdict"] in {"CAN_GO", "CAUTION"}
    assert vehicle["semantic_aware"] is True
    assert vehicle["building_fraction"] == 0.0  # None of the path crosses the building!
    assert vehicle["road_fraction"] > 0.0       # Took the road detour

    # Path coordinates must detour to x >= 70
    xs = [p["x"] for p in vehicle["path"]]
    assert max(xs) >= 70, f"Expected path to detour around building, max x was {max(xs)}"


def test_semantic_fallback_when_unavailable(scene_env, monkeypatch):
    """When a scene has no semantic outputs, routing proceeds honestly on DSM."""
    size = 128
    dsm = np.full((size, size), 10.0, dtype=np.float32)
    scene_id = "scene_no_sem"
    install_scene_data(monkeypatch, scene_id, dsm, sem_probs=None)

    result = route_service.assess(
        scene_id, (10, 10), (10, 110), ["fire_truck"]
    )
    assert result["semantic_available"] is False
    vehicle = result["vehicles"][0]
    assert vehicle["verdict"] == "CAN_GO"
    assert vehicle["semantic_aware"] is False
    assert vehicle["road_fraction"] is None


def test_route_risk_heatmap(scene_env, monkeypatch, tmp_path):
    """Verify route risk heatmap generation and RGBA output."""
    size = 64
    dsm = np.full((size, size), 10.0, dtype=np.float32)
    probs = np.zeros((6, size, size), dtype=np.float32)
    probs[4, :, :] = 1.0        # ground
    probs[0, :32, :32] = 0.95   # building in top-left

    scene_id = "scene_risk_map"
    scene_dir = tmp_path / "data" / "output" / "scenes" / scene_id
    scene_dir.mkdir(parents=True, exist_ok=True)

    install_scene_data(monkeypatch, scene_id, dsm, sem_probs=probs)

    heatmap_path = route_service.route_risk_heatmap_path(scene_id, "fire_truck")
    assert heatmap_path.is_file()

    from PIL import Image
    img = Image.open(heatmap_path)
    assert img.mode == "RGBA"
    assert img.size == (size, size)
