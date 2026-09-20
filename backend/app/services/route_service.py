"""Route-assist service — geometry-based vehicle passability analysis.

Jury round 1/2 module: turns the predicted DSM into actionable movement
decisions for emergency vehicles. HONEST SCOPE: this is a GEOMETRIC
analysis (slope, step height, roughness) — the DSM carries no semantics,
so fences, wires, water and traffic are invisible to it. Every response
says so.

Large-raster design:
    * the DSM (potentially 10k+) is downsampled by block-mean to a
      CLASSIFICATION grid capped at ~1024 cells on the long side —
      memory-bounded regardless of source size;
    * pathfinding is HIERARCHICAL: an A* pass on a coarse (≤256) grid
      finds the global corridor, then a second A* refines the path on the
      full classification grid RESTRICTED to that corridor — the fine
      search never explores the whole map;
    * reachability is answered with connected-component labeling (not
      path failure): when the destination is on a disconnected "island",
      the response says so and proposes the nearest reachable point as a
      detour target instead of a bare "no path".

All artifacts (heat map PNGs) are generated on demand and cached in the
scene output directory, following the terrain_service layer pattern.
"""

from __future__ import annotations

import heapq
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from backend.app.core.logging import logger
from backend.app.core.paths import get_scene_output_dir

# ── vehicle profiles ────────────────────────────────────────────────────────
# Conservative civil-defense style thresholds (slope in degrees, step and
# roughness in metres, vmax in km/h). Used for BOTH classification and the
# travel-time speed model.


@dataclass(frozen=True)
class VehicleProfile:
    key: str
    label: str
    max_slope_deg: float
    max_step_m: float
    max_roughness_m: float
    vmax_kmh: float


VEHICLE_PROFILES: dict[str, VehicleProfile] = {
    "fire_truck": VehicleProfile("fire_truck", "Fire truck", 10.0, 0.35, 0.30, 40.0),
    "ambulance": VehicleProfile("ambulance", "Ambulance", 8.0, 0.25, 0.20, 50.0),
    "rescue_atv": VehicleProfile("rescue_atv", "Rescue ATV", 20.0, 0.60, 0.60, 40.0),
    "suv_4x4": VehicleProfile("suv_4x4", "4x4 SUV", 16.0, 0.50, 0.45, 60.0),
}

CLASS_GRID_CAP = 1024          # classification grid long-side cap
COARSE_GRID_CAP = 256          # hierarchical level-1 cap
CORRIDOR_MARGIN_CELLS = 8      # fine-search corridor half-width
HEATMAP_PNG_CAP = 2048


# ── grid construction (large-raster safe) ──────────────────────────────────

def _block_mean(array: np.ndarray, factor: int) -> np.ndarray:
    """Downsample by integer block mean, trimming a ragged remainder."""
    if factor <= 1:
        return array.astype(np.float32, copy=False)
    h = (array.shape[0] // factor) * factor
    w = (array.shape[1] // factor) * factor
    trimmed = array[:h, :w]
    return trimmed.reshape(h // factor, factor, w // factor, factor).mean(axis=(1, 3))


@dataclass
class PassabilityGrid:
    """The classified cost surface for one scene + vehicle."""

    vehicle: VehicleProfile
    slope_deg: np.ndarray      # [H, W] per-cell slope
    step_m: np.ndarray         # max-min elevation in 3x3 neighbourhood
    roughness_m: np.ndarray    # std-dev of residuals in 5x5 neighbourhood
    classes: np.ndarray        # int8: 0 free, 1 caution, 2 blocked
    factor: int                # source-DSM px per grid cell
    gsd_m: float | None        # ground sample distance (None = pixel units)

    @property
    def passable(self) -> np.ndarray:
        return self.classes < 2


def classify_grid(
    dsm: np.ndarray, gsd_m: float | None, vehicle: VehicleProfile
) -> PassabilityGrid:
    """Build the passability classification grid from a DSM array.

    ``gsd_m`` is the metric ground sample distance; None means the scene
    is not georeferenced, in which case slope/step/roughness are computed
    in PIXEL-depth units and honestly reported as unitless."""
    factor = max(1, math.ceil(max(dsm.shape) / CLASS_GRID_CAP))
    grid = _block_mean(dsm, factor)
    cell = (gsd_m * factor) if gsd_m is not None else float(factor)

    gy, gx = np.gradient(grid, cell)
    slope_deg = np.degrees(np.arctan(np.hypot(gx, gy)))

    if gsd_m is not None:
        step = _window_max(grid) - _window_min(grid)
        rough = _window_std(grid)
    else:
        # Pixel-depth units: normalize step/roughness thresholds by the
        # profile's metre thresholds are meaningless — use slope-only
        # classification with a generous depth-ratio guard.
        step = np.zeros_like(grid)
        rough = np.zeros_like(grid)

    classes = np.zeros(grid.shape, dtype=np.int8)
    limit_slope = vehicle.max_slope_deg
    limit_step = vehicle.max_step_m if gsd_m is not None else math.inf
    limit_rough = vehicle.max_roughness_m if gsd_m is not None else math.inf

    over = (slope_deg > limit_slope) | (step > limit_step) | (rough > limit_rough)
    near = (
        (slope_deg > 0.75 * limit_slope)
        | (step > 0.75 * limit_step)
        | (rough > 0.75 * limit_rough)
    )
    classes[near] = 1
    classes[over] = 2

    return PassabilityGrid(vehicle, slope_deg, step, rough, classes, factor, gsd_m)


def _window_max(a: np.ndarray) -> np.ndarray:
    from scipy.ndimage import maximum_filter

    return maximum_filter(a, size=3, mode="nearest")


def _window_min(a: np.ndarray) -> np.ndarray:
    from scipy.ndimage import minimum_filter

    return minimum_filter(a, size=3, mode="nearest")


def _window_std(a: np.ndarray) -> np.ndarray:
    from scipy.ndimage import uniform_filter

    mean = uniform_filter(a, size=5, mode="nearest")
    mean_sq = uniform_filter(a * a, size=5, mode="nearest")
    return np.sqrt(np.maximum(mean_sq - mean * mean, 0.0))


# ── hierarchical A* ────────────────────────────────────────────────────────

_SQRT2 = math.sqrt(2.0)


def _astar(
    cost: np.ndarray, blocked: np.ndarray, start: tuple[int, int], goal: tuple[int, int]
) -> list[tuple[int, int]] | None:
    """8-connected A* on a cost grid. ``cost`` multiplies edge length;
    ``blocked`` cells are impassable. Returns the path EXCLUDING start."""
    h, w = cost.shape
    sx, sy = start
    gx, gy = goal
    if blocked[sx, sy] or blocked[gx, gy]:
        return None

    # Heuristic: euclidean distance scaled by the minimum edge cost.
    def heur(x: int, y: int) -> float:
        return math.hypot(x - gx, y - gy)

    open_heap: list[tuple[float, float, int, int]] = [(heur(sx, sy), 0.0, sx, sy)]
    g_score = {start: 0.0}
    came: dict[tuple[int, int], tuple[int, int]] = {}
    closed: set[tuple[int, int]] = set()

    while open_heap:
        _, g, x, y = heapq.heappop(open_heap)
        if (x, y) == goal:
            path = [(x, y)]
            while path[-1] != start:
                path.append(came[path[-1]])
            path.reverse()
            return path
        if (x, y) in closed:
            continue
        closed.add((x, y))
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                if dx == 0 and dy == 0:
                    continue
                nx, ny = x + dx, y + dy
                if not (0 <= nx < h and 0 <= ny < w) or blocked[nx, ny]:
                    continue
                if (nx, ny) in closed:
                    continue
                step_len = _SQRT2 if (dx and dy) else 1.0
                ng = g + step_len * cost[nx, ny]
                if ng < g_score.get((nx, ny), math.inf):
                    g_score[(nx, ny)] = ng
                    came[(nx, ny)] = (x, y)
                    heapq.heappush(open_heap, (ng + heur(nx, ny), ng, nx, ny))
    return None


def _soft_cost(grid: PassabilityGrid) -> np.ndarray:
    """Pathfinding edge cost: keeps vehicles away from near-limit terrain
    without forbidding it (free ≈ 1, steep/caution ≫ 1)."""
    v = grid.vehicle
    slope_ratio = grid.slope_deg / max(v.max_slope_deg, 1e-6)
    step_ratio = grid.step_m / max(v.max_step_m, 1e-6)
    base = 1.0 + 4.0 * slope_ratio**2 + 3.0 * step_ratio**2
    base = np.where(grid.classes == 1, base * 1.5, base)
    return np.where(grid.classes >= 2, 1e9, base)


def find_path(
    grid: PassabilityGrid,
    start: tuple[int, int],
    goal: tuple[int, int],
) -> list[tuple[int, int]] | None:
    """Hierarchical A*: coarse global corridor, then fine corridor search.

    Level 1 downsample (≤ COARSE_GRID_CAP) finds the global shape of the
    route; level 2 runs on the FULL classification grid restricted to a
    dilated corridor around it — the fine search explores only a ribbon,
    never the whole map (this is what keeps 1024² grids fast)."""
    cost = _soft_cost(grid)
    blocked = ~grid.passable
    h, w = cost.shape

    # Level 2 first attempt: direct fine search only on small maps.
    if max(h, w) <= COARSE_GRID_CAP:
        return _astar(cost, blocked, start, goal)

    f2 = math.ceil(max(h, w) / COARSE_GRID_CAP)
    coarse_cost = _block_mean(cost, f2)
    coarse_blocked = _block_mean(blocked.astype(np.float32), f2) > 0.4
    cs = (start[0] // f2, start[1] // f2)
    cg = (goal[0] // f2, goal[1] // f2)
    coarse_path = _astar(coarse_cost, coarse_blocked, cs, cg)
    if coarse_path is None:
        return None

    # Corridor mask around the coarse path (matplotlib-free dilation via
    # scipy's binary dilation on the path cells).
    from scipy.ndimage import binary_dilation

    mask = np.zeros_like(blocked, dtype=bool)
    for x, y in coarse_path:
        x0, x1 = max(0, x * f2 - f2), min(h, x * f2 + 2 * f2)
        y0, y1 = max(0, y * f2 - f2), min(w, y * f2 + 2 * f2)
        mask[x0:x1, y0:y1] = True
    mask = binary_dilation(mask, iterations=CORRIDOR_MARGIN_CELLS)

    # Corridor-restricted fine search: everything outside is "blocked".
    fine_blocked = blocked | ~mask
    return _astar(cost, fine_blocked, start, goal)


# ── reachability (island detection) ───────────────────────────────────────

def reachability(
    grid: PassabilityGrid, start: tuple[int, int]
) -> tuple[np.ndarray | None, int]:
    """Connected-component label of the passable mask (0 = unreachable
    background). Returns (labels, label id of the start component)."""
    from scipy.ndimage import label

    labels, _n = label(grid.passable)
    return labels, int(labels[start])


def nearest_reachable(
    labels: np.ndarray, component: int, target: tuple[int, int]
) -> tuple[int, int] | None:
    """Closest cell of ``component`` to ``target`` (detour suggestion)."""
    ys, xs = np.nonzero(labels == component)
    if xs.size == 0:
        return None
    d2 = (xs - target[1]) ** 2 + (ys - target[0]) ** 2
    i = int(np.argmin(d2))
    return int(ys[i]), int(xs[i])


# ── assessment orchestration ───────────────────────────────────────────────

class RouteService:
    """Per-scene vehicle passability assessment + heat map generation."""

    def load_dsm(self, scene_id: str) -> np.ndarray:
        from backend.app.services.result_service import result_service

        files = result_service.get_result_files(scene_id)
        depth = files.get("depth")
        if depth is None:
            raise FileNotFoundError(f"DSM result not found for scene '{scene_id}'.")
        return np.load(depth, mmap_mode="r")

    def gsd_for(self, scene_id: str) -> float | None:
        """Metric GSD from the DSM GeoTIFF; None for pixel-space scenes."""
        from backend.app.services.result_service import result_service

        dsm_path = result_service.get_result_files(scene_id).get("dsm")
        if dsm_path is None or dsm_path.suffix != ".tif":
            return None
        import rasterio

        with rasterio.open(dsm_path) as ds:
            if ds.crs is None:
                return None
            return float(abs(ds.transform.a))

    # -- public assessment -------------------------------------------------

    def assess(
        self,
        scene_id: str,
        start_xy: tuple[int, int],
        goal_xy: tuple[int, int],
        vehicle_keys: list[str],
    ) -> dict[str, Any]:
        dsm = self.load_dsm(scene_id)
        h, w = int(dsm.shape[0]), int(dsm.shape[1])
        gsd = self.gsd_for(scene_id)

        def clamp(px: int, py: int) -> tuple[int, int]:
            return (
                min(max(int(round(py)), 0), h - 1),
                min(max(int(round(px)), 0), w - 1),
            )

        start = clamp(*start_xy)
        goal = clamp(*goal_xy)

        response: dict[str, Any] = {
            "scene_id": scene_id,
            "start_pixel": {"x": start[1], "y": start[0]},
            "end_pixel": {"x": goal[1], "y": goal[0]},
            "units": "meters" if gsd is not None else "pixels",
            "georeferenced": gsd is not None,
            "disclaimer": (
                "Geometry-based passability estimate from the predicted DSM "
                "(slope, step height, roughness). Fences, wires, water and "
                "traffic are NOT visible to this analysis."
            ),
            "vehicles": [],
        }

        geo = self._geo_transform(scene_id)
        if geo is not None:
            response["path_crs"] = geo["crs"]

        for key in vehicle_keys:
            profile = VEHICLE_PROFILES.get(key)
            if profile is None:
                continue
            response["vehicles"].append(
                self._assess_vehicle(scene_id, dsm, gsd, profile, start, goal, geo)
            )
        return response

    def _assess_vehicle(
        self, scene_id, dsm, gsd, profile, start, goal, geo
    ) -> dict[str, Any]:
        grid = classify_grid(dsm, gsd, profile)
        f = grid.factor
        gstart = (start[0] // f, start[1] // f)
        ggoal = (goal[0] // f, goal[1] // f)

        result: dict[str, Any] = {
            "vehicle": profile.key,
            "vehicle_label": profile.label,
            "limits": {
                "max_slope_deg": profile.max_slope_deg,
                "max_step_m": profile.max_step_m,
                "max_roughness_m": profile.max_roughness_m,
            },
        }

        # Start/goal themselves on blocked terrain: honest immediate no.
        for role, cell in (("start", gstart), ("end", ggoal)):
            if grid.classes[cell] == 2:
                result.update(
                    verdict="CANNOT_GO",
                    reasons=[f"The {role} point is on terrain this vehicle "
                             f"cannot stand on (blocked cell)."],
                    path=None,
                )
                return result

        labels, component = reachability(grid, gstart)
        if labels[ggoal] != component:
            detour = nearest_reachable(labels, component, ggoal)
            reasons = [
                "The destination is on ground disconnected from the start "
                "by passable terrain for this vehicle (blocked by slope, "
                "steps or roughness)."
            ]
            if detour is not None:
                reasons.append(
                    "Nearest reachable point to the destination suggested "
                    "as a detour target."
                )
                result["detour_pixel"] = {
                    "x": detour[1] * f, "y": detour[0] * f,
                }
            result.update(verdict="CANNOT_GO", reasons=reasons, path=None)
            return result

        grid_path = find_path(grid, gstart, ggoal)
        if grid_path is None:
            result.update(
                verdict="CANNOT_GO",
                reasons=["No passable route found for this vehicle."],
                path=None,
            )
            return result

        # Path statistics on the classification grid.
        path_arr = np.asarray(grid_path)
        slope_on = grid.slope_deg[path_arr[:, 0], path_arr[:, 1]]
        step_on = grid.step_m[path_arr[:, 0], path_arr[:, 1]]
        class_on = grid.classes[path_arr[:, 0], path_arr[:, 1]]
        caution_frac = float((class_on == 1).mean())

        cell_m = (gsd * f) if gsd is not None else float(f)
        seg = np.hypot(np.diff(path_arr[:, 0]), np.diff(path_arr[:, 1]))
        length_m = float(seg.sum() * cell_m)

        # Travel time: speed degrades with slope; integrate along the path.
        vmax_ms = profile.vmax_kmh / 3.6
        speed = vmax_ms * np.maximum(
            0.15, 1.0 - slope_on / max(profile.max_slope_deg, 1e-6)
        )
        with np.errstate(divide="ignore"):
            time_s = float((seg / np.maximum(speed[:-1], 0.1)).sum())

        verdict = "CAN_GO"
        reasons: list[str] = []
        if caution_frac > 0.30:
            verdict = "CAUTION"
            reasons.append(
                f"{caution_frac * 100:.0f}% of the route crosses near-limit "
                "terrain (caution cells)."
            )
        if slope_on.max() > 0.9 * profile.max_slope_deg:
            verdict = "CAUTION" if verdict == "CAN_GO" else verdict
            reasons.append(
                f"Maximum slope on route ({slope_on.max():.1f}°) is within "
                "10% of this vehicle's limit."
            )

        pixel_path = [
            {"x": int(y * f), "y": int(x * f)} for x, y in grid_path
        ]
        result.update(
            verdict=verdict,
            reasons=reasons or ["Terrain along the route is within this vehicle's limits."],
            path=pixel_path,
            path_length_m=length_m if gsd is not None else None,
            path_length_px=length_m if gsd is None else None,
            max_slope_on_path_deg=round(float(slope_on.max()), 2),
            max_step_on_path_m=(
                round(float(step_on.max()), 3) if gsd is not None else None
            ),
            caution_fraction=round(caution_frac, 3),
            estimated_travel_seconds=round(time_s, 1) if gsd is not None else None,
        )
        if geo is not None:
            result["path_geojson"] = {
                "type": "Feature",
                "geometry": {
                    "type": "LineString",
                    "coordinates": [
                        geo["pixel_to_world"](p["x"], p["y"]) for p in pixel_path
                    ],
                },
                "properties": {"vehicle": profile.key, "verdict": verdict},
            }
        return result

    # -- geo helpers ---------------------------------------------------------

    def _geo_transform(self, scene_id: str) -> dict | None:
        """pixel→world mapping from the DSM GeoTIFF (None when pixel-space)."""
        import rasterio

        from backend.app.services.result_service import result_service

        dsm_path = result_service.get_result_files(scene_id).get("dsm")
        if dsm_path is None or dsm_path.suffix != ".tif":
            return None
        with rasterio.open(dsm_path) as ds:
            if ds.crs is None:
                return None

            def pixel_to_world(x: float, y: float) -> list[float]:
                wx, wy = ds.transform * (x + 0.5, y + 0.5)
                return [float(wx), float(wy)]

            return {"crs": ds.crs.to_string(), "pixel_to_world": pixel_to_world}

    # -- heat map ------------------------------------------------------------

    def heatmap_path(self, scene_id: str, vehicle_key: str) -> Path:
        """Generate (idempotently, cached) the passability heat map PNG."""
        profile = VEHICLE_PROFILES.get(vehicle_key)
        if profile is None:
            raise ValueError(f"Unknown vehicle profile: {vehicle_key}")

        out = get_scene_output_dir(scene_id) / f"passability_{vehicle_key}.png"
        dsm = self.load_dsm(scene_id)
        gsd = self.gsd_for(scene_id)

        grid = classify_grid(dsm, gsd, profile)
        rgba = np.zeros((*grid.classes.shape, 4), dtype=np.uint8)
        # Traffic-light palette: green free, amber caution, red blocked.
        palette = {
            0: (46, 204, 113, 255),
            1: (241, 196, 15, 255),
            2: (231, 76, 60, 255),
        }
        for cls, color in palette.items():
            rgba[grid.classes == cls] = color

        from PIL import Image

        image = Image.fromarray(rgba, mode="RGBA")
        if max(image.size) > HEATMAP_PNG_CAP:
            scale = HEATMAP_PNG_CAP / max(image.size)
            image = image.resize(
                (int(image.width * scale), int(image.height * scale)),
                Image.NEAREST,
            )
        out.parent.mkdir(parents=True, exist_ok=True)
        image.save(out, format="PNG")
        return out

    def heatmap_stats(self, scene_id: str, vehicle_key: str) -> dict[str, Any]:
        profile = VEHICLE_PROFILES.get(vehicle_key)
        if profile is None:
            raise ValueError(f"Unknown vehicle profile: {vehicle_key}")
        dsm = self.load_dsm(scene_id)
        grid = classify_grid(dsm, self.gsd_for(scene_id), profile)
        total = grid.classes.size
        blocked = int((grid.classes == 2).sum())
        caution = int((grid.classes == 1).sum())
        return {
            "vehicle": vehicle_key,
            "blocked_pct": round(blocked / total * 100, 1),
            "caution_pct": round(caution / total * 100, 1),
            "free_pct": round((total - blocked - caution) / total * 100, 1),
            "georeferenced": grid.gsd_m is not None,
        }


route_service = RouteService()
