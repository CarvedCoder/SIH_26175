"""Route-assist service + endpoint tests (jury round 1/2 module).

Covers: vehicle profiles, verdicts on synthetic terrain (flat / ramp /
wall), hierarchical pathfinding correctness on LARGE rasters, island
reachability with detour suggestion, heat map generation + layer route,
and the honest pixel-units degradation for non-georeferenced scenes.
"""

from __future__ import annotations

import numpy as np
import pytest

from backend.app.services.route_service import (
    CLASS_GRID_CAP,
    VEHICLE_PROFILES,
    classify_grid,
    find_path,
    reachability,
    route_service,
)


# ---------------------------------------------------------------------------
# Synthetic terrain builders
# ---------------------------------------------------------------------------

def flat_scene(size: int = 256, base: float = 10.0) -> np.ndarray:
    return np.full((size, size), base, dtype=np.float32)


def ramp_scene(size: int = 256, total_rise: float = 20.0) -> np.ndarray:
    """Gentle planar ramp — traversable when the grade stays in limit."""
    rows = np.linspace(0.0, total_rise, size, dtype=np.float32)
    return np.broadcast_to(rows[:, None], (size, size)).copy()


def wall_scene(size: int = 256, wall_at: int | None = None, height: float = 6.0) -> np.ndarray:
    """Flat terrain cut by a vertical wall — impassable for every profile."""
    wall_at = size // 2 if wall_at is None else wall_at
    scene = flat_scene(size)
    scene[wall_at:, :] += height
    return scene


def island_scene(size: int = 256) -> np.ndarray:
    """Goal enclosed by a moat (deep trench) — disconnected for vehicles."""
    scene = flat_scene(size)
    moat = 4.0
    band = slice(size // 2 - 8, size // 2 + 8)
    scene[band, :] -= moat  # trench: step height exceeds every profile
    return scene


@pytest.fixture()
def scene_env(monkeypatch, tmp_path):
    """Point result_service at synthetic DSM files, no API needed."""
    import backend.app.core.paths as paths

    monkeypatch.setattr(paths, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(paths, "DATA_DIR", tmp_path / "data")
    monkeypatch.setattr(
        paths, "SCENES_OUTPUT_DIR", tmp_path / "data" / "output" / "scenes"
    )
    paths.ensure_directories()
    yield tmp_path


def install_dsm(monkeypatch, scene_id: str, dsm: np.ndarray, gsd: float | None = 0.5):
    """Stub the service's DSM/GSD accessors with synthetic data."""

    def fake_load(_self, sid):
        assert sid == scene_id
        return dsm

    def fake_gsd(_self, sid):
        return gsd

    monkeypatch.setattr(route_service, "load_dsm", fake_load.__get__(route_service))
    monkeypatch.setattr(route_service, "gsd_for", fake_gsd.__get__(route_service))


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------

def test_flat_terrain_is_free_for_all_vehicles():
    for profile in VEHICLE_PROFILES.values():
        grid = classify_grid(flat_scene(), 0.5, profile)
        assert grid.classes.max() == 0, profile.key


def test_steep_ramp_blocks_fire_truck_but_not_atv():
    # 20 m rise over 256 px at 0.5 m/px = 128 m run -> ~8.9 deg.
    dsm = ramp_scene(256, total_rise=40.0)  # ~17.7 deg
    fire = classify_grid(dsm, 0.5, VEHICLE_PROFILES["fire_truck"])
    atv = classify_grid(dsm, 0.5, VEHICLE_PROFILES["rescue_atv"])
    assert fire.classes.max() == 2  # blocked
    assert atv.classes.max() <= 1  # at most caution


def test_wall_is_blocked_for_every_vehicle():
    for profile in VEHICLE_PROFILES.values():
        grid = classify_grid(wall_scene(), 0.5, profile)
        assert grid.classes.max() == 2


# ---------------------------------------------------------------------------
# Pathfinding (incl. the large-raster hierarchical case)
# ---------------------------------------------------------------------------

def test_path_found_on_flat_scene_and_has_stats(scene_env, monkeypatch):
    scene_id = "scene_000000000001"
    install_dsm(monkeypatch, scene_id, flat_scene(512))
    result = route_service.assess(
        scene_id, (10, 10), (500, 500), ["fire_truck", "ambulance"]
    )
    assert result["georeferenced"] is True
    for vehicle in result["vehicles"]:
        assert vehicle["verdict"] == "CAN_GO", vehicle
        assert vehicle["path"] and len(vehicle["path"]) > 5
        assert vehicle["max_slope_on_path_deg"] < VEHICLE_PROFILES[
            vehicle["vehicle"]
        ].max_slope_deg
        assert vehicle["estimated_travel_seconds"] > 0
        assert vehicle["path_length_m"] > 0


def test_wall_crossing_is_cannot_go(scene_env, monkeypatch):
    scene_id = "scene_000000000002"
    install_dsm(monkeypatch, scene_id, wall_scene(512))
    result = route_service.assess(scene_id, (10, 10), (10, 500), ["fire_truck"])
    vehicle = result["vehicles"][0]
    assert vehicle["verdict"] == "CANNOT_GO"
    assert vehicle["path"] is None
    assert vehicle["reasons"]


def test_detour_going_around_the_wall(scene_env, monkeypatch):
    """A wall with a ramped gap: the path must exist and route through it.

    The gap is a smooth ramp (6 m rise over 80 px at 0.5 m/px ≈ 8.5°,
    under the fire truck's 10° limit) so the two height levels connect —
    a flat gap at the lower height would be honestly classified as a
    separate compartment."""
    size = 512
    dsm = wall_scene(size, wall_at=size // 2, height=6.0)
    cols = slice(220, 292)
    rows = np.arange(216, 296)
    ramp = 10.0 + 6.0 * (rows - 216) / 80.0
    dsm[216:296, cols] = ramp[:, None]
    dsm[:216, cols] = 10.0
    dsm[296:, cols] = 16.0
    scene_id = "scene_000000000003"
    install_dsm(monkeypatch, scene_id, dsm)
    # start above the wall, end below it — the only way through is the ramp
    result = route_service.assess(scene_id, (10, 10), (10, size - 10), ["fire_truck"])
    vehicle = result["vehicles"][0]
    assert vehicle["verdict"] in {"CAN_GO", "CAUTION"}
    # every route must cross the wall line through the ramp's column band
    xs = [p["x"] for p in vehicle["path"]]
    assert min(xs) <= 292 and max(xs) >= 220


def test_hierarchical_search_handles_oversized_raster(scene_env, monkeypatch):
    """A 4000x4000 DSM (larger than CLASS_GRID_CAP) still assesses fast,
    proving the hierarchical pathfinder never explores the full grid."""
    size = 4000
    dsm = ramp_scene(size, total_rise=30.0)  # gentle, globally passable
    scene_id = "scene_000000000004"
    install_dsm(monkeypatch, scene_id, dsm)
    result = route_service.assess(scene_id, (0, 0), (size - 1, size - 1), ["suv_4x4"])
    vehicle = result["vehicles"][0]
    assert vehicle["verdict"] in {"CAN_GO", "CAUTION"}
    assert vehicle["path"]


def test_island_destination_suggests_detour(scene_env, monkeypatch):
    scene_id = "scene_000000000005"
    install_dsm(monkeypatch, scene_id, island_scene(512))
    result = route_service.assess(scene_id, (10, 10), (10, 500), ["fire_truck"])
    vehicle = result["vehicles"][0]
    assert vehicle["verdict"] == "CANNOT_GO"
    assert vehicle.get("detour_pixel") is not None
    detour = vehicle["detour_pixel"]
    # the detour must be on the START side of the moat
    assert detour["y"] < 512 // 2 - 8


def test_pixel_units_for_ungeoreferenced_scene(scene_env, monkeypatch):
    scene_id = "scene_000000000006"
    install_dsm(monkeypatch, scene_id, flat_scene(256), gsd=None)
    result = route_service.assess(scene_id, (10, 10), (200, 200), ["fire_truck"])
    assert result["georeferenced"] is False
    assert result["units"] == "pixels"
    vehicle = result["vehicles"][0]
    assert vehicle["verdict"] == "CAN_GO"
    assert vehicle["path_length_m"] is None
    assert vehicle["estimated_travel_seconds"] is None


# ---------------------------------------------------------------------------
# Heat map
# ---------------------------------------------------------------------------

def test_heatmap_generation_and_stats(scene_env, monkeypatch):
    from PIL import Image

    scene_id = "scene_000000000007"
    install_dsm(monkeypatch, scene_id, wall_scene(256))
    path = route_service.heatmap_path(scene_id, "fire_truck")
    assert path.is_file()
    img = Image.open(path)
    assert img.mode == "RGBA"

    stats = route_service.heatmap_stats(scene_id, "fire_truck")
    assert stats["blocked_pct"] > 0
    assert abs(
        stats["blocked_pct"] + stats["caution_pct"] + stats["free_pct"] - 100.0
    ) < 0.5


# ---------------------------------------------------------------------------
# API routes
# ---------------------------------------------------------------------------

def test_route_endpoints_require_processed_scene(client, uploaded_scene):
    scene_id = uploaded_scene["scene_id"]
    body = {"start": {"x": 0, "y": 0}, "end": {"x": 100, "y": 100}}
    assert (
        client.post(f"/api/v1/scenes/{scene_id}/route/assess", json=body).status_code
        == 404
    )
    assert (
        client.get(f"/api/v1/scenes/{scene_id}/route/heatmap").status_code == 404
    )


def test_route_assess_endpoint_full_flow(client, processed_scene):
    scene_id = processed_scene["scene"]["scene_id"]
    body = {
        "start": {"x": 0, "y": 0},
        "end": {"x": 255, "y": 255},
        "vehicles": ["fire_truck", "ambulance"],
    }
    response = client.post(f"/api/v1/scenes/{scene_id}/route/assess", json=body)
    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["georeferenced"] is True
    assert len(payload["vehicles"]) == 2
    for vehicle in payload["vehicles"]:
        assert vehicle["verdict"] in {"CAN_GO", "CAUTION", "CANNOT_GO"}
        assert vehicle["reasons"]
        assert "disclaimer" in payload and "NOT visible" in payload["disclaimer"]


def test_route_heatmap_endpoint(client, processed_scene):
    scene_id = processed_scene["scene"]["scene_id"]
    response = client.get(
        f"/api/v1/scenes/{scene_id}/route/heatmap?vehicle=fire_truck"
    )
    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["url"]
    assert "blocked_pct" in payload

    # the layer texture route serves a real PNG
    layer = client.get(
        f"/api/v1/scenes/{scene_id}/results/passability?vehicle=fire_truck"
    )
    assert layer.status_code == 200
    assert layer.headers["content-type"].startswith("image/png")


def test_unknown_vehicle_400(client, processed_scene):
    scene_id = processed_scene["scene"]["scene_id"]
    body = {
        "start": {"x": 0, "y": 0},
        "end": {"x": 10, "y": 10},
        "vehicles": ["mechWarrior"],
    }
    response = client.post(f"/api/v1/scenes/{scene_id}/route/assess", json=body)
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "INVALID_VEHICLE"


# ── rescue chopper: aerial landing-zone assessment ─────────────────────────

def test_chopper_lands_on_flat_terrain(scene_env, monkeypatch):
    install_dsm(monkeypatch, "s", flat_scene(), gsd=0.5)
    res = route_service.assess("s", (10, 10), (190, 190), ["rescue_chopper"])
    ch = res["vehicles"][0]
    assert ch["verdict"] in ("CAN_GO", "CAUTION")
    assert ch["landing_zone"] is not None
    assert ch["path"] and len(ch["path"]) >= 2  # straight flight line


def test_chopper_denied_on_rough_destination(scene_env, monkeypatch):
    rng = np.random.default_rng(7)
    rough = flat_scene() + rng.normal(0, 2.0, flat_scene().shape).astype(np.float32)
    install_dsm(monkeypatch, "s", rough.astype(np.float32), gsd=0.5)
    res = route_service.assess("s", (10, 10), (190, 190), ["rescue_chopper"])
    ch = res["vehicles"][0]
    assert ch["verdict"] == "CANNOT_GO"
    assert ch["landing_zone"] is None
    assert ch["path"] is None


def test_chopper_lz_search_radius_respects_scale(scene_env, monkeypatch):
    # Flat scene but destination far from any special terrain: LZ should be
    # within the declared search radius of the destination pixel.
    install_dsm(monkeypatch, "s", flat_scene(), gsd=0.5)
    res = route_service.assess("s", (5, 5), (250, 250), ["rescue_chopper"])
    ch = res["vehicles"][0]
    lz = ch["landing_zone"]
    assert lz is not None
    dx = abs(lz["pixel"]["x"] - 250) / 2  # grid factor 2 for 256px/1024cap
    dy = abs(lz["pixel"]["y"] - 250) / 2
    assert max(dx, dy) <= 26  # radius 25m / 0.5m-cell * factor 2 = 25 cells
