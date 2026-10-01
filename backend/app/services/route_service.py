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
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from backend.app.core.config import settings
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
    is_aerial: bool = False


VEHICLE_PROFILES: dict[str, VehicleProfile] = {
    "fire_truck": VehicleProfile("fire_truck", "Fire truck", 10.0, 0.35, 0.30, 40.0),
    "ambulance": VehicleProfile("ambulance", "Ambulance", 8.0, 0.25, 0.20, 50.0),
    "rescue_atv": VehicleProfile("rescue_atv", "Rescue ATV", 20.0, 0.60, 0.60, 40.0),
    "suv_4x4": VehicleProfile("suv_4x4", "4x4 SUV", 16.0, 0.50, 0.45, 60.0),
    # Aerial: no ground route is planned. The assessment instead searches
    # for a LANDING ZONE near the destination (flat, open, low roughness)
    # and reports a straight flight line; denial means "no viable LZ".
    "rescue_chopper": VehicleProfile(
        "rescue_chopper", "Rescue Chopper", 6.0, 0.30, 0.35, 140.0, is_aerial=True
    ),
}

# ── semantic configuration & vehicle weights ──────────────────────────────
@dataclass(frozen=True)
class SemanticConfig:
    enabled: bool = True
    confidence_threshold: float = 0.70
    hard_block_threshold: float = 0.90
    weights: dict[str, float] = field(default_factory=lambda: {
        "building": 1000.0,  # effectively blocked
        "water": 1000.0,     # effectively blocked
        "road": 0.65,        # preference multiplier
        "vegetation": 4.0,   # baseline penalty
        "ground": 1.0,       # baseline
        "other": 1.5,        # uncertainty penalty
    })

@dataclass(frozen=True)
class DamageConfig:
    enabled: bool = True
    destroyed_cost: float = 50.0
    major_cost: float = 20.0
    minor_cost: float = 3.0


VEGETATION_PENALTIES: dict[str, float] = {
    "fire_truck": 8.0,
    "ambulance": 8.0,
    "rescue_atv": 2.5,
    "suv_4x4": 5.0,
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


def _block_downsample_probs(probs: np.ndarray, factor: int) -> np.ndarray:
    """Downsample [6, H, W] probabilities by block-averaging (NOT class IDs)."""
    if factor <= 1:
        return probs.astype(np.float32, copy=False)
    k, h_orig, w_orig = probs.shape
    h = (h_orig // factor) * factor
    w = (w_orig // factor) * factor
    trimmed = probs[:, :h, :w]
    return trimmed.reshape(k, h // factor, factor, w // factor, factor).mean(axis=(2, 4))


def _semantic_cost(
    sem_probs: np.ndarray,  # [6, H, W]
    sem_labels: np.ndarray, # [H, W]
    sem_conf: np.ndarray,   # [H, W]
    vehicle: VehicleProfile,
    config: SemanticConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """Returns (semantic_cost_multiplier [H,W], semantic_blocked [H,W] bool)."""
    veg_weight = VEGETATION_PENALTIES.get(
        vehicle.key, config.weights.get("vegetation", 4.0)
    )
    weights = np.array(
        [
            config.weights.get("building", 1000.0),
            veg_weight,
            config.weights.get("road", 0.65),
            config.weights.get("water", 1000.0),
            config.weights.get("ground", 1.0),
            config.weights.get("other", 1.5),
        ],
        dtype=np.float32,
    )
    raw_sem_cost = np.tensordot(weights, sem_probs, axes=(0, 0))
    other_cost = config.weights.get("other", 1.5)
    sem_cost = sem_conf * raw_sem_cost + (1.0 - sem_conf) * other_cost

    bldg_prob = sem_probs[0]
    water_prob = sem_probs[3]
    hard_blocked = (bldg_prob >= config.hard_block_threshold) | (
        water_prob >= config.hard_block_threshold
    )
    return sem_cost, hard_blocked


def _damage_cost(
    damage_labels: np.ndarray,  # [H, W] uint8 (0=background, 1=no-damage, 2=minor, 3=major, 4=destroyed)
    config: DamageConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """Returns (damage_cost_multiplier [H,W], damage_hazard_blocked [H,W] bool).

    Damage classes:
        1 = no-damage (multiplier 1.0)
        2 = minor-damage (multiplier config.minor_cost)
        3 = major-damage (multiplier config.major_cost)
        4 = destroyed (multiplier config.destroyed_cost, rubble hazard)
    """
    mult = np.ones(damage_labels.shape, dtype=np.float32)
    mult[damage_labels == 2] = config.minor_cost
    mult[damage_labels == 3] = config.major_cost
    mult[damage_labels == 4] = config.destroyed_cost
    hazard_blocked = damage_labels == 4
    return mult, hazard_blocked


def _combined_cost(
    grid: PassabilityGrid,
    sem_probs: np.ndarray | None = None,
    sem_labels: np.ndarray | None = None,
    sem_conf: np.ndarray | None = None,
    sem_config: SemanticConfig | None = None,
    damage_labels: np.ndarray | None = None,
    damage_config: DamageConfig | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Combines geometric passability cost with semantic and damage cost/blocking."""
    geo_cost = _soft_cost(grid)
    total_cost = geo_cost
    total_blocked = ~grid.passable

    if sem_probs is not None and sem_config is not None and sem_config.enabled:
        sem_cost, sem_blocked = _semantic_cost(
            sem_probs, sem_labels, sem_conf, grid.vehicle, sem_config
        )
        total_cost = total_cost * sem_cost
        total_blocked = total_blocked | sem_blocked

    if damage_labels is not None and damage_config is not None and damage_config.enabled:
        dmg_cost, dmg_blocked = _damage_cost(damage_labels, damage_config)
        total_cost = total_cost * dmg_cost
        total_blocked = total_blocked | dmg_blocked

    return total_cost, total_blocked


def find_path(
    grid: PassabilityGrid,
    start: tuple[int, int],
    goal: tuple[int, int],
    sem_probs: np.ndarray | None = None,
    sem_labels: np.ndarray | None = None,
    sem_conf: np.ndarray | None = None,
    sem_config: SemanticConfig | None = None,
    damage_labels: np.ndarray | None = None,
    damage_config: DamageConfig | None = None,
) -> list[tuple[int, int]] | None:
    """Hierarchical A*: coarse global corridor, then fine corridor search.

    When semantic or damage data is provided, incorporates cost weights and
    hard blocking (buildings, water, destroyed rubble) in both levels."""
    cost, blocked = _combined_cost(
        grid, sem_probs, sem_labels, sem_conf, sem_config,
        damage_labels=damage_labels, damage_config=damage_config,
    )
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
        # Fall back to fine search if coarse disconnected
        return _astar(cost, blocked, start, goal)

    from scipy.ndimage import binary_dilation

    mask = np.zeros_like(blocked, dtype=bool)
    for x, y in coarse_path:
        x0, x1 = max(0, x * f2 - f2), min(h, x * f2 + 2 * f2)
        y0, y1 = max(0, y * f2 - f2), min(w, y * f2 + 2 * f2)
        mask[x0:x1, y0:y1] = True
    mask = binary_dilation(mask, iterations=CORRIDOR_MARGIN_CELLS)

    fine_blocked = blocked | ~mask
    fine_path = _astar(cost, fine_blocked, start, goal)
    if fine_path is not None:
        return fine_path
    return _astar(cost, blocked, start, goal)


# ── reachability (island detection) ───────────────────────────────────────

def reachability(
    grid: PassabilityGrid,
    start: tuple[int, int],
    passable_mask: np.ndarray | None = None,
) -> tuple[np.ndarray | None, int]:
    """Connected-component label of the passable mask (0 = unreachable
    background). Returns (labels, label id of the start component)."""
    from scipy.ndimage import label

    mask = passable_mask if passable_mask is not None else grid.passable
    labels, _n = label(mask)
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


def nearest_standable(
    grid: PassabilityGrid,
    target: tuple[int, int],
    radius: int,
    blocked_mask: np.ndarray | None = None,
) -> tuple[int, int] | None:
    """Closest non-blocked cell to ``target`` within ``radius`` (Chebyshev
    ring search on the classification grid). None when everything nearby
    is blocked for this vehicle."""
    th, tw = grid.classes.shape
    ty, tx = target
    r0 = min(radius, max(th, tw))
    for r in range(1, r0 + 1):
        best = None
        best_d2 = None
        y0, y1 = max(0, ty - r), min(th - 1, ty + r)
        x0, x1 = max(0, tx - r), min(tw - 1, tx + r)
        for y in range(y0, y1 + 1):
            for x in range(x0, x1 + 1):
                # ring-only after the first pass keeps it cheap
                if r > 1 and max(abs(y - ty), abs(x - tx)) != r:
                    continue
                is_blocked = (
                    blocked_mask[y, x] if blocked_mask is not None
                    else (grid.classes[y, x] == 2)
                )
                if is_blocked:
                    continue
                d2 = (y - ty) ** 2 + (x - tx) ** 2
                if best_d2 is None or d2 < best_d2:
                    best_d2, best = d2, (y, x)
        if best is not None:
            return best
    return None


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

    def load_semantics(
        self, scene_id: str
    ) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None]:
        """Load semantic artifacts (probs, labels, confidence) if available.
        Returns (probs [6,H,W], labels [H,W], conf [H,W]) or (None, None, None).
        """
        from backend.app.services.result_service import result_service

        files = result_service.get_result_files(scene_id)
        probs_file = files.get("semantic_probs")
        labels_file = files.get("semantic_labels")
        conf_file = files.get("semantic_confidence")

        if probs_file is None or not probs_file.exists():
            return None, None, None
        try:
            probs = np.load(probs_file, mmap_mode="r")
            labels = (
                np.load(labels_file, mmap_mode="r")
                if labels_file and labels_file.exists()
                else np.argmax(probs, axis=0).astype(np.uint8)
            )
            conf = (
                np.load(conf_file, mmap_mode="r")
                if conf_file and conf_file.exists()
                else np.max(probs, axis=0).astype(np.float32)
            )
            return probs, labels, conf
        except Exception as e:
            logger.warning("Failed to load semantics for scene %s: %s", scene_id, e)
            return None, None, None

    def load_damage(
        self, scene_id: str
    ) -> tuple[np.ndarray | None, np.ndarray | None]:
        """Load damage artifacts (labels, confidence) if available.
        Returns (damage_labels [H,W], damage_conf [H,W]) or (None, None).
        """
        from backend.app.services.result_service import result_service

        files = result_service.get_result_files(scene_id)
        labels_file = files.get("damage_labels")
        conf_file = files.get("damage_confidence")

        if labels_file is None or not labels_file.exists():
            return None, None
        try:
            labels = np.load(labels_file, mmap_mode="r")
            conf = (
                np.load(conf_file, mmap_mode="r")
                if conf_file and conf_file.exists()
                else None
            )
            return labels, conf
        except Exception as e:
            logger.warning("Failed to load damage for scene %s: %s", scene_id, e)
            return None, None

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

        sem_config = SemanticConfig(
            enabled=settings.semantic_enabled and settings.route_semantic_enabled,
            confidence_threshold=settings.semantic_confidence_threshold,
            hard_block_threshold=settings.semantic_hard_block_threshold,
        )
        sem_probs_full, sem_labels_full, sem_conf_full = self.load_semantics(scene_id)
        has_semantics = sem_probs_full is not None and sem_config.enabled

        dmg_config = DamageConfig(
            enabled=getattr(settings, "disaster_enabled", True) and getattr(settings, "route_damage_enabled", True),
            destroyed_cost=getattr(settings, "route_damage_destroyed_cost", 50.0),
            major_cost=getattr(settings, "route_damage_major_cost", 20.0),
            minor_cost=getattr(settings, "route_damage_minor_cost", 3.0),
        )
        damage_labels_full, damage_conf_full = self.load_damage(scene_id)
        has_damage = damage_labels_full is not None and dmg_config.enabled

        def clamp(px: int, py: int) -> tuple[int, int]:
            return (
                min(max(int(round(py)), 0), h - 1),
                min(max(int(round(px)), 0), w - 1),
            )

        start = clamp(*start_xy)
        goal = clamp(*goal_xy)

        disclaimer = (
            "Geometry, semantic, and structural damage passability estimate from the predicted DSM "
            "(slope, step height, roughness, 6-class semantics, and disaster damage). Roads are "
            "preferred; buildings, water bodies, and damaged rubble are avoided."
            if (has_semantics or has_damage)
            else (
                "Geometry-based passability estimate from the predicted DSM "
                "(slope, step height, roughness). Fences, wires, water and "
                "traffic are NOT visible to this analysis."
            )
        )

        response: dict[str, Any] = {
            "scene_id": scene_id,
            "start_pixel": {"x": start[1], "y": start[0]},
            "end_pixel": {"x": goal[1], "y": goal[0]},
            "units": "meters" if gsd is not None else "pixels",
            "georeferenced": gsd is not None,
            "semantic_available": has_semantics,
            "damage_available": has_damage,
            "disclaimer": disclaimer,
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
                self._assess_vehicle(
                    scene_id,
                    dsm,
                    gsd,
                    profile,
                    start,
                    goal,
                    geo,
                    sem_probs_full=sem_probs_full,
                    sem_labels_full=sem_labels_full,
                    sem_conf_full=sem_conf_full,
                    sem_config=sem_config,
                    damage_labels_full=damage_labels_full,
                    damage_conf_full=damage_conf_full,
                    damage_config=dmg_config,
                )
            )
        return response

    def _assess_vehicle(
        self,
        scene_id,
        dsm,
        gsd,
        profile,
        start,
        goal,
        geo,
        sem_probs_full: np.ndarray | None = None,
        sem_labels_full: np.ndarray | None = None,
        sem_conf_full: np.ndarray | None = None,
        sem_config: SemanticConfig | None = None,
        damage_labels_full: np.ndarray | None = None,
        damage_conf_full: np.ndarray | None = None,
        damage_config: DamageConfig | None = None,
    ) -> dict[str, Any]:
        if profile.is_aerial:
            return self._assess_chopper(
                dsm,
                gsd,
                profile,
                start,
                goal,
                geo,
                sem_probs_full=sem_probs_full,
                sem_labels_full=sem_labels_full,
                sem_conf_full=sem_conf_full,
                sem_config=sem_config,
                damage_labels_full=damage_labels_full,
                damage_conf_full=damage_conf_full,
                damage_config=damage_config,
            )
        grid = classify_grid(dsm, gsd, profile)
        f = grid.factor
        gstart = (start[0] // f, start[1] // f)
        ggoal = (goal[0] // f, goal[1] // f)

        # Downsample semantics to match grid resolution
        sem_probs = None
        sem_labels = None
        sem_conf = None
        sem_blocked = None
        if sem_probs_full is not None and sem_config is not None and sem_config.enabled:
            sem_probs = _block_downsample_probs(sem_probs_full, f)
            sem_labels = np.argmax(sem_probs, axis=0).astype(np.uint8)
            sem_conf = np.max(sem_probs, axis=0).astype(np.float32)
            _, sem_blocked = _semantic_cost(
                sem_probs, sem_labels, sem_conf, profile, sem_config
            )

        # Downsample damage to match grid resolution
        damage_labels = None
        damage_conf = None
        dmg_blocked = None
        if damage_labels_full is not None and damage_config is not None and damage_config.enabled:
            damage_labels = damage_labels_full[::f, ::f][:grid.classes.shape[0], :grid.classes.shape[1]]
            if damage_conf_full is not None:
                damage_conf = damage_conf_full[::f, ::f][:grid.classes.shape[0], :grid.classes.shape[1]]
            _, dmg_blocked = _damage_cost(damage_labels, damage_config)

        total_blocked = ~grid.passable
        if sem_blocked is not None:
            total_blocked = total_blocked | sem_blocked
        if dmg_blocked is not None:
            total_blocked = total_blocked | dmg_blocked
        total_passable = ~total_blocked

        result: dict[str, Any] = {
            "vehicle": profile.key,
            "vehicle_label": profile.label,
            "limits": {
                "max_slope_deg": profile.max_slope_deg,
                "max_step_m": profile.max_step_m,
                "max_roughness_m": profile.max_roughness_m,
            },
        }

        # Picks that land on blocked or disconnected ground are snapped to
        # the nearest standable / reachable cell (within a bounded radius)
        # instead of refusing outright.
        snap_radius_cells = 64  # ~1/8 of the typical grid per axis
        snapped: dict[str, bool] = {}

        if total_blocked[gstart]:
            alt = nearest_standable(grid, gstart, snap_radius_cells, blocked_mask=total_blocked)
            if alt is None:
                result.update(
                    verdict="CANNOT_GO",
                    reasons=[
                        "No standable ground for this vehicle anywhere near the start point."
                    ],
                    path=None,
                )
                return result
            gstart = alt
            snapped["start"] = True

        labels, component = reachability(grid, gstart, passable_mask=total_passable)
        if labels[ggoal] != component:
            detour = nearest_reachable(labels, component, ggoal)
            if detour is None:
                result.update(
                    verdict="CANNOT_GO",
                    reasons=[
                        "No passable ground reachable from the start for this vehicle."
                    ],
                    path=None,
                )
                return result
            dy = math.hypot(detour[0] - ggoal[0], detour[1] - ggoal[1])
            if dy > snap_radius_cells:
                result["detour_pixel"] = {
                    "x": detour[1] * f,
                    "y": detour[0] * f,
                }
                result.update(
                    verdict="CANNOT_GO",
                    reasons=[
                        "The destination is too far from any ground reachable by this vehicle to route to.",
                        "Nearest reachable point to the destination suggested as a detour target.",
                    ],
                    path=None,
                )
                return result
            ggoal = detour
            snapped["end"] = True

        grid_path = find_path(
            grid,
            gstart,
            ggoal,
            sem_probs=sem_probs,
            sem_labels=sem_labels,
            sem_conf=sem_conf,
            sem_config=sem_config,
            damage_labels=damage_labels,
            damage_config=damage_config,
        )
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

        # Travel time: speed degrades with slope and surface
        vmax_ms = profile.vmax_kmh / 3.6
        speed_factor = 1.0 - slope_on / max(profile.max_slope_deg, 1e-6)
        if sem_labels is not None:
            path_labels = sem_labels[path_arr[:, 0], path_arr[:, 1]]
            # Roads permit slightly higher smooth travel; brush slows down
            surface_mult = np.ones_like(speed_factor)
            surface_mult[path_labels == 2] = 1.1   # Road
            surface_mult[path_labels == 1] = 0.6   # Vegetation
            surface_mult[path_labels == 4] = 0.9   # Ground
            speed = vmax_ms * np.maximum(0.15, speed_factor * surface_mult)
        else:
            speed = vmax_ms * np.maximum(0.15, speed_factor)

        with np.errstate(divide="ignore"):
            time_s = float((seg / np.maximum(speed[:-1], 0.1)).sum())

        verdict = "CAN_GO"
        reasons: list[str] = []
        if snapped.get("start"):
            reasons.append(
                "Start point snapped to the nearest standable ground for this vehicle."
            )
        if snapped.get("end"):
            reasons.append(
                "Destination snapped to the nearest reachable ground for this vehicle."
            )

        # Semantic stats on path
        road_frac = None
        bldg_frac = None
        water_frac = None
        veg_frac = None
        ground_frac = None
        other_frac = None
        sem_risk_frac = None

        if sem_labels is not None:
            path_labels = sem_labels[path_arr[:, 0], path_arr[:, 1]]
            total_pts = max(len(path_labels), 1)
            road_frac = float((path_labels == 2).sum()) / total_pts
            veg_frac = float((path_labels == 1).sum()) / total_pts
            bldg_frac = float((path_labels == 0).sum()) / total_pts
            water_frac = float((path_labels == 3).sum()) / total_pts
            ground_frac = float((path_labels == 4).sum()) / total_pts
            other_frac = float((path_labels == 5).sum()) / total_pts
            sem_risk_frac = float(((path_labels == 0) | (path_labels == 3)).sum()) / total_pts

            if road_frac > 0.60:
                reasons.append(
                    f"Preferred road corridor selected ({road_frac * 100:.0f}% road coverage)."
                )
            elif road_frac > 0.20:
                reasons.append(
                    f"Route utilizes available road sections ({road_frac * 100:.0f}%)."
                )
            if veg_frac > 0.30:
                reasons.append(
                    f"Route traverses vegetated terrain ({veg_frac * 100:.0f}% canopy/brush)."
                )
            if bldg_frac > 0 or water_frac > 0:
                reasons.append(
                    "Route skirts or touches building/water boundaries."
                )

        # Damage stats on path
        damage_risk_frac = None
        if damage_labels is not None:
            path_dmg = damage_labels[path_arr[:, 0], path_arr[:, 1]]
            damaged_pts = ((path_dmg >= 2) & (path_dmg <= 4)).sum()
            damage_risk_frac = float(damaged_pts) / max(len(path_dmg), 1)
            destroyed_pts = (path_dmg == 4).sum()
            if destroyed_pts > 0:
                reasons.append("Route navigates near rubble from destroyed structures.")
            elif damage_risk_frac > 0.05:
                reasons.append(f"Route navigates damaged building zones ({damage_risk_frac * 100:.0f}% exposure).")

        if caution_frac > 0.30:
            verdict = "CAUTION"
            reasons.append(
                f"{caution_frac * 100:.0f}% of the route crosses near-limit terrain (caution cells)."
            )
        if slope_on.max() > 0.9 * profile.max_slope_deg:
            verdict = "CAUTION" if verdict == "CAN_GO" else verdict
            reasons.append(
                f"Maximum slope on route ({slope_on.max():.1f}°) is within 10% of this vehicle's limit."
            )

        if snapped:
            result["snapped"] = snapped
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
            road_fraction=round(road_frac, 3) if road_frac is not None else None,
            building_fraction=round(bldg_frac, 3) if bldg_frac is not None else None,
            water_fraction=round(water_frac, 3) if water_frac is not None else None,
            vegetation_fraction=round(veg_frac, 3) if veg_frac is not None else None,
            ground_fraction=round(ground_frac, 3) if ground_frac is not None else None,
            other_fraction=round(other_frac, 3) if other_frac is not None else None,
            semantic_risk_fraction=round(sem_risk_frac, 3) if sem_risk_frac is not None else None,
            semantic_aware=(sem_labels is not None),
            damage_aware=(damage_labels is not None),
            damage_risk_fraction=round(damage_risk_frac, 3) if damage_risk_frac is not None else None,
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

    # -- aerial assessment (rescue chopper) ----------------------------------

    LZ_RADIUS_M = 25.0          # search radius around the destination
    LZ_MIN_PATCH_CELLS = 3      # ≥3×3 free cells ≈ a pad the chopper fits on

    @staticmethod
    def _straight_line(
        start: tuple[int, int], goal: tuple[int, int]
    ) -> list[tuple[int, int]]:
        """Grid cells along the start→goal segment (Bresenham-style)."""
        x0, y0 = start
        x1, y1 = goal
        n = max(abs(x1 - x0), abs(y1 - y0), 1)
        return [
            (int(round(x0 + (x1 - x0) * i / n)),
             int(round(y0 + (y1 - y0) * i / n)))
            for i in range(n + 1)
        ]

    def _assess_chopper(
        self,
        dsm,
        gsd,
        profile,
        start,
        goal,
        geo,
        sem_probs_full: np.ndarray | None = None,
        sem_labels_full: np.ndarray | None = None,
        sem_conf_full: np.ndarray | None = None,
        sem_config: SemanticConfig | None = None,
        damage_labels_full: np.ndarray | None = None,
        damage_conf_full: np.ndarray | None = None,
        damage_config: DamageConfig | None = None,
    ) -> dict[str, Any]:
        """Aerial verdict: fly start→destination directly, but only if a
        viable LANDING ZONE exists near the destination. No LZ ⇒ denial."""
        grid = classify_grid(dsm, gsd, profile)
        f = grid.factor
        cell_m = (gsd * f) if gsd is not None else float(f)

        result: dict[str, Any] = {
            "vehicle": profile.key,
            "vehicle_label": profile.label,
            "limits": {
                "max_slope_deg": profile.max_slope_deg,
                "max_step_m": profile.max_step_m,
                "max_roughness_m": profile.max_roughness_m,
            },
        }

        ggoal = (goal[0] // f, goal[1] // f)
        radius_cells = max(1, int(self.LZ_RADIUS_M / max(cell_m, 1e-6)))

        # Landing patch = free cell whose (LZ_MIN_PATCH_CELLS)² neighbourhood
        # is entirely free, not building/water, and free of structural damage.
        from scipy.ndimage import uniform_filter

        free_mask = grid.classes == 0
        if sem_probs_full is not None and sem_config is not None and sem_config.enabled:
            sem_probs = _block_downsample_probs(sem_probs_full, f)
            free_mask = free_mask & (sem_probs[0] < 0.5) & (sem_probs[3] < 0.5)

        if damage_labels_full is not None and damage_config is not None and damage_config.enabled:
            damage_labels = damage_labels_full[::f, ::f][:grid.classes.shape[0], :grid.classes.shape[1]]
            # Avoid damaged/collapsed building zones (classes 2, 3, 4)
            free_mask = free_mask & (damage_labels < 2)

        free = free_mask.astype(np.float32)
        patch = uniform_filter(free, size=self.LZ_MIN_PATCH_CELLS) >= 0.999

        h, w = patch.shape
        x0 = max(0, ggoal[0] - radius_cells)
        x1 = min(h, ggoal[0] + radius_cells + 1)
        y0 = max(0, ggoal[1] - radius_cells)
        y1 = min(w, ggoal[1] + radius_cells + 1)
        window = patch[x0:x1, y0:y1]

        if not window.any():
            result.update(
                verdict="CANNOT_GO",
                reasons=[
                    "No viable landing zone within "
                    f"{self.LZ_RADIUS_M:.0f}{'m' if gsd is not None else 'px'} "
                    "of the destination — terrain is too steep, stepped, rough, or "
                    "obstructed by buildings/water. Chopper request denied."
                ],
                path=None,
                landing_zone=None,
            )
            return result

        # Best patch cell: flattest first, then closest to the destination.
        ys, xs = np.nonzero(window)
        xs_abs = xs + y0
        ys_abs = ys + x0
        slope_c = grid.slope_deg[ys_abs, xs_abs]
        rough_c = grid.roughness_m[ys_abs, xs_abs]
        dist_c = np.hypot(ys_abs - ggoal[0], xs_abs - ggoal[1])
        score = (
            4.0 * (slope_c / max(profile.max_slope_deg, 1e-6)) ** 2
            + 2.0 * (rough_c / max(profile.max_roughness_m, 1e-6)) ** 2
            + dist_c / max(radius_cells, 1)
        )
        best = int(np.argmin(score))
        lz = (int(ys_abs[best]), int(xs_abs[best]))
        lz_slope = float(grid.slope_deg[lz])
        lz_step = float(grid.step_m[lz])

        lz_radius_m = radius_cells * cell_m
        verdict = "CAN_GO"
        reasons: list[str] = [
            f"Landing zone found {dist_c[best] * cell_m:.0f}"
            f"{'m' if gsd is not None else 'px'} from the destination "
            f"(slope {lz_slope:.1f}°, step {lz_step:.2f}"
            f"{'m' if gsd is not None else 'px'})."
        ]
        if lz_slope > 0.75 * profile.max_slope_deg:
            verdict = "CAUTION"
            reasons.append(
                f"Landing zone slope ({lz_slope:.1f}°) is within 25% of the "
                "touchdown limit — approach with care."
            )

        # Flight line: straight start → landing zone, no ground pathfinding.
        flight = self._straight_line((start[0] // f, start[1] // f), lz)
        pixel_path = [{"x": int(y * f), "y": int(x * f)} for x, y in flight]
        seg = np.hypot(*np.diff(np.asarray(flight), axis=0).T) if len(flight) > 1 else np.array([])
        length_m = float(seg.sum() * cell_m) if seg.size else 0.0

        result.update(
            verdict=verdict,
            reasons=reasons,
            path=pixel_path,
            landing_zone={
                "pixel": {"x": int(lz[1] * f), "y": int(lz[0] * f)},
                "slope_deg": round(lz_slope, 2),
                "distance_to_goal_m": (
                    round(float(dist_c[best] * cell_m), 1)
                    if gsd is not None
                    else None
                ),
                "distance_to_goal_px": (
                    round(float(dist_c[best] * cell_m), 1)
                    if gsd is None
                    else None
                ),
                "search_radius_m": (
                    round(lz_radius_m, 1) if gsd is not None else round(lz_radius_m, 1)
                ),
            },
            path_length_m=length_m if gsd is not None else None,
            path_length_px=length_m if gsd is None else None,
            max_slope_on_path_deg=round(lz_slope, 2),
            max_step_on_path_m=(
                round(lz_step, 3) if gsd is not None else None
            ),
            caution_fraction=0.0,
            estimated_travel_seconds=(
                round(length_m / (profile.vmax_kmh / 3.6), 1)
                if gsd is not None
                else None
            ),
            semantic_aware=(sem_probs_full is not None),
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
                "properties": {
                    "vehicle": profile.key,
                    "verdict": verdict,
                    "kind": "flight_line",
                },
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

    def route_risk_heatmap_path(self, scene_id: str, vehicle_key: str) -> Path:
        """Generate (idempotently, cached) a combined geometry + semantics
        route-risk heat map PNG."""
        profile = VEHICLE_PROFILES.get(vehicle_key)
        if profile is None:
            raise ValueError(f"Unknown vehicle profile: {vehicle_key}")

        out = get_scene_output_dir(scene_id) / f"route_risk_{vehicle_key}.png"
        dsm = self.load_dsm(scene_id)
        gsd = self.gsd_for(scene_id)
        grid = classify_grid(dsm, gsd, profile)
        f = grid.factor

        sem_probs_full, sem_labels_full, sem_conf_full = self.load_semantics(scene_id)

        rgba = np.zeros((*grid.classes.shape, 4), dtype=np.uint8)

        geo_blocked = grid.classes == 2
        geo_caution = grid.classes == 1
        geo_free = grid.classes == 0

        if sem_probs_full is not None:
            sem_probs = _block_downsample_probs(sem_probs_full, f)
            sem_labels = np.argmax(sem_probs, axis=0).astype(np.uint8)
            bldg_or_water = (sem_probs[0] >= 0.70) | (sem_probs[3] >= 0.70)

            # 1. Blocked: geometric blocked OR building OR water (Red)
            is_blocked = geo_blocked | bldg_or_water
            rgba[is_blocked] = (231, 76, 60, 255)

            # 2. Road on passable geometry -> Optimal corridor (Emerald)
            is_road = (sem_labels == 2) & ~is_blocked
            rgba[is_road] = (39, 174, 96, 255)

            # 3. Ground on free geometry -> Passable (Green)
            is_ground = (sem_labels == 4) & geo_free & ~is_blocked & ~is_road
            rgba[is_ground] = (46, 204, 113, 255)

            # 4. Vegetation -> Orange
            is_veg = (sem_labels == 1) & ~is_blocked & ~is_road
            rgba[is_veg] = (230, 126, 34, 255)

            # 5. Geometry caution (remaining) -> Yellow
            is_caution = geo_caution & ~is_blocked & ~is_road & ~is_veg
            rgba[is_caution] = (241, 196, 15, 255)

            # 6. Unassigned free -> Muted grey/tan
            unassigned = rgba[:, :, 3] == 0
            rgba[unassigned] = (149, 165, 166, 255)
        else:
            palette = {
                0: (46, 204, 113, 255),
                1: (241, 196, 15, 255),
                2: (231, 76, 60, 255),
            }
            for cls, color in palette.items():
                rgba[grid.classes == cls] = color

        # Overlay damage hazards if available
        damage_labels_full, _ = self.load_damage(scene_id)
        if damage_labels_full is not None and getattr(settings, "route_damage_enabled", True):
            damage_labels = damage_labels_full[::f, ::f][:grid.classes.shape[0], :grid.classes.shape[1]]
            # Destroyed structures / debris: Dark Red (211, 47, 47, 255)
            rgba[damage_labels == 4] = (211, 47, 47, 255)
            # Major damage: Deep Orange (255, 87, 34, 255)
            rgba[damage_labels == 3] = (255, 87, 34, 255)

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


route_service = RouteService()
