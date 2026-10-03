"""Geometry-aware 3D building reconstruction.

Builds on EXISTING DepthWizard outputs only:
    * building candidates: the disaster ONNX building mask when present,
      else the semantic segmentation's `building` class (labels == 0);
    * heights: the predicted metric DSM (`dsm.npy`, AGL metres above the
      local ground) — never an invented or normalized height;
    * georeference: CRS + affine transform of the prediction grid.

Pipeline (per the geometry-aware reconstruction spec):
    building mask
      -> geometric regularization (closing/opening, hole fill, component
         filtering)                       [scipy.ndimage]
      -> boundary extraction              [Moore contour tracing]
      -> adaptive primitive fitting       [shapely + IoU scoring:
         rectangle / square / circle / ellipse / semicircle / stadium /
         composite rectangles / polygon]
      -> composite decomposition          [oriented max-inscribed-rectangle
         recursion for L-shaped / multi-part footprints]
      -> robust per-building height       [ring-ground + MAD-trimmed median
         of the predicted DSM inside the footprint; optional piecewise
         upper levels for multi-height complexes]
      -> confidence score                 [mask confidence, boundary quality,
         fit error, height consistency] — low confidence preserves more of
         the original footprint instead of aggressively regularizing
      -> watertight prism extrusion       [ear-cut triangulation; used by
         tests and export; the web viewer extrudes the same footprints
         with THREE.ExtrudeGeometry]

ANTI-FABRICATION CONTRACT:
    * Heights ALWAYS come from the predicted DSM. No fixed heights, no
      normalization to an arbitrary range.
    * Edge/gradient information only REFINES boundaries of regions the
      semantic model already identified — never the primary detector.
    * Non-georeferenced scenes keep pixel-space footprints and honestly
      report `georeferenced: false`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage
from shapely.geometry import Polygon


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class Building3DConfig:
    """Tunable thresholds for footprint extraction, fitting, heights.

    All pixel-area/tolerance values are in source-raster pixels; heights
    are in the DSM's metres. Defaults are deliberately conservative —
    every knob is overridable from backend Settings (DW_BUILDINGS3D_*).
    """

    # -- mask regularization -------------------------------------------------
    min_area_px: int = 100           # drop components smaller than this
    close_iterations: int = 2        # morphological closing passes (3x3)
    open_iterations: int = 1         # morphological opening passes (3x3)

    # -- footprint simplification ---------------------------------------------
    # Base simplify tolerance in px; scaled by sqrt(area/10000) so large
    # roofs tolerate more smoothing, and by (0.5 + 0.5*confidence) so low
    # confidence preserves more of the original boundary.
    simplify_tol_px: float = 1.5
    max_vertices: int = 64           # per-ring cap (curved structures)
    arc_segments: int = 16           # segments per 180° arc for stadium/semi

    # -- primitive acceptance --------------------------------------------------
    rect_iou: float = 0.80           # oriented-rectangle acceptance
    circle_circularity: float = 0.85 # 4*pi*A/P^2 threshold for circle
    stadium_iou: float = 0.88        # capsule acceptance
    ellipse_iou: float = 0.88        # ellipse acceptance
    min_confidence_for_primitive: float = 0.45  # below: keep polygon footprint

    # -- composite decomposition ------------------------------------------------
    max_parts: int = 4               # max inscribed rectangles per building
    min_part_frac: float = 0.08      # part must cover >=8% of component area

    # -- heights ------------------------------------------------------------------
    ground_percentile: int = 20      # ring percentile used as ground/base
    height_mad_trim: float = 3.0     # MAD multiplier for outlier rejection
    piecewise_gap_m: float = 2.0     # min cluster separation for a 2nd level
    piecewise_min_frac: float = 0.15 # smaller level must hold >=15% of pixels

    # -- confidence weighting ---------------------------------------------------
    w_mask: float = 0.35
    w_boundary: float = 0.25
    w_fit: float = 0.20
    w_height: float = 0.20

    # -- limits -----------------------------------------------------------------
    max_buildings: int = 60          # hard cap on reconstructed buildings


# ---------------------------------------------------------------------------
# Small raster helpers
# ---------------------------------------------------------------------------


def _rasterize_polygon(poly: Polygon, width: int, height: int) -> np.ndarray:
    """Rasterize a shapely polygon (in pixel coordinates) to a bool mask.

    Accepts MultiPolygon inputs (rasterizes each part)."""
    img = Image.new("L", (width, height), 0)
    drawer = ImageDraw.Draw(img)
    parts = list(getattr(poly, "geoms", [poly])) if hasattr(poly, "geoms") else [poly]
    for part in parts:
        exterior = [(float(x), float(y)) for x, y in part.exterior.coords]
        if len(exterior) >= 3:
            drawer.polygon(exterior, outline=1, fill=1)
        for hole in part.interiors:
            hole_pts = [(float(x), float(y)) for x, y in hole.coords]
            if len(hole_pts) >= 3:
                drawer.polygon(hole_pts, outline=0, fill=0)
    return np.asarray(img, dtype=bool)


def _trace_boundary(mask: np.ndarray) -> list[tuple[int, int]] | None:
    """Moore-neighbour boundary trace of the largest region in a binary mask.

    Returns an ordered (x, y) pixel ring, or None when the mask is empty.
    The input must contain exactly one connected region (callers label
    first); holes were already filled during regularization.
    """
    ys, xs = np.nonzero(mask)
    if ys.size == 0:
        return None

    h, w = mask.shape
    start = (int(xs[0]), int(ys[0]))  # row-major first pixel — westernmost
    # 8 directions, clockwise, starting West
    dirs = [(-1, 0), (-1, -1), (0, -1), (1, -1), (1, 0), (1, 1), (0, 1), (-1, 1)]

    def inside(x: int, y: int) -> bool:
        return 0 <= x < w and 0 <= y < h and mask[y, x]

    contour = [start]
    # backtrack: the neighbour we came from (pixel West of start)
    prev_dir = 0  # index of direction pointing at the backtrack pixel
    cur = start
    for _ in range(mask.size * 4):  # hard safety bound
        found = False
        for i in range(1, 9):
            d = (prev_dir + i) % 8
            nx, ny = cur[0] + dirs[d][0], cur[1] + dirs[d][1]
            if inside(nx, ny):
                # backtrack for the next step points back at `cur`
                prev_dir = (d + 4) % 8
                cur = (nx, ny)
                contour.append(cur)
                found = True
                break
        if not found:
            break  # isolated pixel
        # Jacob's stopping criterion: entered the start pixel the same way
        if cur == start and len(contour) > 2:
            break
    # drop the duplicated closing point if present
    if len(contour) > 1 and contour[-1] == contour[0]:
        contour.pop()
    return contour if len(contour) >= 3 else None


def _polygon_iou(poly_a: Polygon, poly_b: Polygon, width: int, height: int,
                 bbox: tuple[int, int, int, int] | None = None) -> float:
    """IoU of two polygons via rasterization (optionally cropped to a bbox)."""
    if bbox is not None:
        x0, y0, x1, y1 = bbox
        poly_a = _shift_poly(poly_a, -x0, -y0)
        poly_b = _shift_poly(poly_b, -x0, -y0)
        width, height = x1 - x0, y1 - y0
    if width <= 0 or height <= 0:
        return 0.0
    a = _rasterize_polygon(poly_a, width, height)
    b = _rasterize_polygon(poly_b, width, height)
    union = np.logical_or(a, b).sum()
    if union == 0:
        return 0.0
    return float(np.logical_and(a, b).sum() / union)


def _shift_poly(poly: Polygon, dx: float, dy: float) -> Polygon:
    from shapely.affinity import translate

    return translate(poly, dx, dy)


def _minrect_frame(poly: Polygon) -> tuple[float, float, float, float, float, float]:
    """Oriented minimum-rectangle frame of a polygon.

    Returns (cx, cy, length, width, cos, sin) where (cos, sin) rotate
    pixel-space into the rectangle's frame (long axis = local X).
    """
    rect = poly.minimum_rotated_rectangle
    coords = list(rect.exterior.coords)[:-1]
    # edge 0-1 and 1-2: pick the longer as the long axis
    (x0, y0), (x1, y1), (x2, y2) = coords[0], coords[1], coords[2]
    d01 = math.hypot(x1 - x0, y1 - y0)
    d12 = math.hypot(x2 - x1, y2 - y1)
    if d01 >= d12:
        ax, ay = x1 - x0, y1 - y0
        length, width = d01, d12
    else:
        ax, ay = x2 - x1, y2 - y1
        length, width = d12, d01
    norm = math.hypot(ax, ay) or 1.0
    cos_t, sin_t = ax / norm, ay / norm
    cx, cy = poly.centroid.x, poly.centroid.y
    return cx, cy, length, width, cos_t, sin_t


def _rotate_coords(x: np.ndarray, y: np.ndarray, cx: float, cy: float,
                   cos_t: float, sin_t: float) -> tuple[np.ndarray, np.ndarray]:
    """Rotate pixel coords into the min-rect frame (long axis = local X)."""
    dx, dy = x - cx, y - cy
    return dx * cos_t + dy * sin_t, -dx * sin_t + dy * cos_t


def _unrotate_coords(u: np.ndarray, v: np.ndarray, cx: float, cy: float,
                     cos_t: float, sin_t: float) -> tuple[np.ndarray, np.ndarray]:
    """Inverse of _rotate_coords."""
    x = cx + u * cos_t - v * sin_t
    y = cy + u * sin_t + v * cos_t
    return x, y


# ---------------------------------------------------------------------------
# Mask regularization
# ---------------------------------------------------------------------------


def regularize_mask(mask: np.ndarray, cfg: Building3DConfig) -> np.ndarray:
    """Noisy semantic mask -> clean building mask.

    closing -> opening -> hole filling -> small-component removal.
    Vegetation contamination and broken boundaries are the semantic
    model's problem to flag (confidence); here we only remove
    pixel-level noise while preserving the true footprint.
    """
    clean = mask.astype(bool)
    if not clean.any():
        return clean
    structure = np.ones((3, 3), dtype=bool)
    if cfg.close_iterations > 0:
        clean = ndimage.binary_closing(clean, structure=structure,
                                       iterations=cfg.close_iterations)
    if cfg.open_iterations > 0:
        clean = ndimage.binary_opening(clean, structure=structure,
                                       iterations=cfg.open_iterations)
    clean = ndimage.binary_fill_holes(clean)
    if cfg.min_area_px > 1:
        labels, n = ndimage.label(clean)
        if n:
            sizes = ndimage.sum_labels(np.ones_like(clean, dtype=np.int32),
                                       labels, index=np.arange(1, n + 1))
            keep = np.flatnonzero(sizes >= cfg.min_area_px) + 1
            clean = np.isin(labels, keep)
    return clean


# ---------------------------------------------------------------------------
# Primitive fitting
# ---------------------------------------------------------------------------


def _arc_points(cx: float, cy: float, r_x: float, r_y: float,
                a0: float, a1: float, segments: int) -> list[tuple[float, float]]:
    angles = np.linspace(a0, a1, max(2, segments) + 1)
    return [(cx + r_x * math.cos(a), cy + r_y * math.sin(a)) for a in angles]


def _frame_ring_to_pixel(pts_frame: list[tuple[float, float]], cx: float, cy: float,
                         cos_t: float, sin_t: float) -> list[tuple[float, float]]:
    """Rotate a frame-space ring into pixel space (vectorized)."""
    if not pts_frame:
        return []
    u = np.array([p[0] for p in pts_frame])
    v = np.array([p[1] for p in pts_frame])
    wx, wy = _unrotate_coords(u, v, cx, cy, cos_t, sin_t)
    return [(float(x), float(y)) for x, y in zip(wx, wy)]


def _candidate_polygons(poly: Polygon, cfg: Building3DConfig,
                        include_polygon: bool = True) -> list[tuple[str, Polygon]]:
    """Build the analytic candidate set for one simplified footprint."""
    area = max(poly.area, 1e-6)
    cx, cy = poly.centroid.x, poly.centroid.y
    cx, cy, length, width, cos_t, sin_t = _minrect_frame(poly)
    candidates: list[tuple[str, Polygon]] = []

    # oriented rectangle (always evaluated)
    rect = poly.minimum_rotated_rectangle
    candidates.append(("rectangle", rect))
    if length > 0 and width / length > 0.8:
        candidates.append(("square", rect))

    # circle (equivalent area)
    r_circle = math.sqrt(area / math.pi)
    circle_pts = _arc_points(cx, cy, r_circle, r_circle, 0, 2 * math.pi,
                             2 * cfg.arc_segments)
    candidates.append(("circle", Polygon(circle_pts)))

    # ellipse aligned with the oriented frame (area-preserving:
    # pi * (length/2) * r_y = area)
    r_x = length / 2.0
    r_y = area / (math.pi * r_x) if r_x > 0 else width / 2.0
    ellipse_pts = _frame_ring_to_pixel(
        _arc_points(0.0, 0.0, r_x, r_y, 0, 2 * math.pi, 2 * cfg.arc_segments),
        cx, cy, cos_t, sin_t)
    if len(ellipse_pts) >= 3:
        candidates.append(("ellipse", Polygon(ellipse_pts)))

    # stadium (capsule) from the oriented frame: rectangle (length-width)
    # plus two semicircular caps of radius width/2
    if length > width > 0:
        r_cap = width / 2.0
        half_len = (length - width) / 2.0
        # cap centers along the long axis in frame coords
        c1 = (half_len, 0.0)
        c2 = (-half_len, 0.0)
        pts: list[tuple[float, float]] = []
        # arc around c1 from -90° to +90° (frame space), then around c2
        for ang in np.linspace(-math.pi / 2, math.pi / 2, cfg.arc_segments):
            pts.append((c1[0] + r_cap * math.cos(ang), r_cap * math.sin(ang)))
        for ang in np.linspace(math.pi / 2, 3 * math.pi / 2, cfg.arc_segments):
            pts.append((c2[0] + r_cap * math.cos(ang), r_cap * math.sin(ang)))
        # rotate frame -> pixel space
        wx, wy = _unrotate_coords(np.array([p[0] for p in pts]),
                                  np.array([p[1] for p in pts]),
                                  cx, cy, cos_t, sin_t)
        candidates.append(("stadium", Polygon(list(zip(wx, wy)))))

        # semicircle / half-ellipse: ellipse cut by the SHORT-axis line
        # (flat edge across the middle of the long axis), both sides
        for side in (1.0, -1.0):
            a0, a1 = (0.0, math.pi) if side > 0 else (math.pi, 2 * math.pi)
            semi_pts = _frame_ring_to_pixel(
                _arc_points(0.0, 0.0, r_x, r_y, a0, a1, cfg.arc_segments),
                cx, cy, cos_t, sin_t)
            if len(semi_pts) >= 3:
                candidates.append(("semicircle", Polygon(semi_pts)))

    # the simplified polygon itself — the curved/complex fallback
    if include_polygon:
        candidates.append(("polygon", poly))
    return candidates


def _largest_axis_aligned_rect(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    """Max-area axis-aligned rectangle inside a binary mask (histogram method).

    Returns (x0, y0, x1, y1) exclusive-right/bottom, or None.
    """
    h, w = mask.shape
    heights = np.zeros(w, dtype=int)
    best = (0.0, None)
    for row in range(h):
        row_vals = mask[row]
        heights = np.where(row_vals, heights + 1, 0)
        # largest rectangle in histogram (monotonic stack)
        stack: list[int] = []
        i = 0
        while i <= w:
            cur = int(heights[i]) if i < w else 0
            start = i
            while stack and stack[-1][1] >= cur:
                idx, hgt = stack.pop()
                area = hgt * (i - idx)
                if area > best[0]:
                    best = (area, (idx, row - hgt + 1, i, row + 1))
                start = idx
            stack.append((start, cur))
            i += 1
    return best[1]


def decompose_into_rectangles(component_mask: np.ndarray, cfg: Building3DConfig
                              ) -> list[Polygon]:
    """Oriented rectangle decomposition for L-shaped / composite footprints.

    Works in the component's minimum-rectangle frame (so rotated L-shapes
    decompose too): repeatedly extracts the largest inscribed axis-aligned
    rectangle and subtracts it, up to ``max_parts``.
    """
    ys, xs = np.nonzero(component_mask)
    if ys.size == 0:
        return []
    poly = _polygon_from_mask(component_mask)
    if poly is None or poly.is_empty:
        return []
    cx, cy, _length, _width, cos_t, sin_t = _minrect_frame(poly)

    # rotate all component pixels into the frame, rasterize there
    rx, ry = _rotate_coords(xs.astype(np.float64), ys.astype(np.float64),
                            cx, cy, cos_t, sin_t)
    pad = 2
    u0, v0 = rx.min() - pad, ry.min() - pad
    fw = int(math.ceil(rx.max() - rx.min())) + 2 * pad
    fh = int(math.ceil(ry.max() - ry.min())) + 2 * pad
    if fw <= 0 or fh <= 0 or fw * fh > 4_000_000:
        return []
    frame_img = Image.new("L", (fw, fh), 0)
    frame_draw = ImageDraw.Draw(frame_img)
    # stamp each component pixel as a unit square (a polygon through
    # scan-ordered pixel centers zigzags and corrupts the frame mask)
    for u, v in zip(rx, ry):
        frame_draw.rectangle(
            [u - u0 - 0.5, v - v0 - 0.5, u - u0 + 0.5, v - v0 + 0.5], fill=1)
    frame_mask = np.asarray(frame_img, dtype=bool)
    frame_mask = ndimage.binary_fill_holes(frame_mask)

    total_area = float(frame_mask.sum())
    parts: list[Polygon] = []
    remaining = frame_mask
    for _ in range(cfg.max_parts):
        rect = _largest_axis_aligned_rect(remaining)
        if rect is None:
            break
        x0, y0, x1, y1 = rect
        if float(x1 - x0) * float(y1 - y0) < cfg.min_part_frac * total_area:
            break
        # inset half a pixel per side: the frame mask was stamped from
        # pixel-center unit squares, so raw rect bounds carry a 1px halo
        # (uncompensated, every composite footprint inflates ~1px outward)
        # frame-space corners -> pixel space
        corners_u = np.array([x0 + 0.5, x1 - 0.5, x1 - 0.5, x0 + 0.5],
                             dtype=np.float64) + u0
        corners_v = np.array([y0 + 0.5, y0 + 0.5, y1 - 0.5, y1 - 0.5],
                             dtype=np.float64) + v0
        wx, wy = _unrotate_coords(corners_u, corners_v, cx, cy, cos_t, sin_t)
        part = Polygon(list(zip(wx, wy)))
        if part.is_valid and part.area > 0:
            parts.append(part)
        remaining = remaining & ~_rasterize_polygon(
            Polygon([(x0, y0), (x1, y0), (x1, y1), (x0, y1)]), fw, fh)
        if not remaining.any():
            break
    if len(parts) < 2:
        return []
    return parts


def _polygon_from_mask(mask: np.ndarray) -> Polygon | None:
    ring = _trace_boundary(mask)
    if ring is None:
        return None
    poly = Polygon(ring)
    if not poly.is_valid:
        poly = poly.buffer(0)
    if poly.is_empty or poly.area <= 0:
        return None
    return poly


def fit_footprint(component_mask: np.ndarray, cfg: Building3DConfig,
                  confidence: float) -> dict[str, Any]:
    """Adaptive geometric fitting for ONE connected building region.

    Returns {footprint: Polygon, primitive: str, fit_iou: float,
             decomposition_parts: list[Polygon] | None}.
    """
    raw_poly = _polygon_from_mask(component_mask)
    if raw_poly is None:
        return {}

    area = raw_poly.area
    # topology-preserving simplify; tolerance scales with roof size and
    # DROPS with low confidence (preserve more of the original boundary)
    tol = cfg.simplify_tol_px * math.sqrt(max(area, 1.0) / 10000.0)
    tol *= (0.5 + 0.5 * confidence)
    poly = raw_poly.simplify(tol, preserve_topology=True)
    if poly.is_empty or poly.area <= 0:
        poly = raw_poly
    if len(poly.exterior.coords) > cfg.max_vertices + 1:
        # vertex cap: re-simplify harder (curved structures get more
        # segments only when the shape genuinely demands them)
        for scale in (1.5, 2.5, 4.0):
            poly = raw_poly.simplify(tol * scale, preserve_topology=True)
            if len(poly.exterior.coords) <= cfg.max_vertices + 1:
                break

    ys, xs = np.nonzero(component_mask)
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    bbox = (x0, y0, x1, y1)

    circularity = 4 * math.pi * poly.area / max(poly.length ** 2, 1e-6)

    # primitives compete on IoU; the simplified polygon itself is the
    # FALLBACK (a polygon trivially scores IoU 1.0 against itself, so it
    # must never enter the race — otherwise no primitive is ever chosen)
    best_primitive = None
    best_iou = 0.0

    for name, candidate in _candidate_polygons(poly, cfg, include_polygon=False):
        if candidate.is_empty or not candidate.is_valid:
            continue
        iou = _polygon_iou(poly, candidate, x1 - x0, y1 - y0, bbox=bbox)
        if name == "circle" and circularity < cfg.circle_circularity:
            continue
        if name in ("rectangle", "square") and iou < cfg.rect_iou:
            continue
        if name == "stadium" and iou < cfg.stadium_iou:
            continue
        if name == "ellipse" and iou < cfg.ellipse_iou:
            continue
        if name == "semicircle" and iou < cfg.stadium_iou:
            continue
        if iou > best_iou:
            best_iou, best_primitive = iou, (name, candidate)

    # low-confidence scenes keep the observed footprint — a visually clean
    # but WRONG primitive is worse than a slightly noisy honest polygon
    if confidence < cfg.min_confidence_for_primitive:
        best_primitive = None

    result: dict[str, Any] = {
        "footprint": poly,
        "primitive": "polygon",
        "fit_iou": 1.0,
        "decomposition_parts": None,
    }

    if best_primitive is not None:
        result["primitive"] = best_primitive[0]
        result["footprint"] = best_primitive[1]
        result["fit_iou"] = best_iou
        return result

    # composite attempt: oriented rectangle decomposition (L-shapes,
    # multi-wing complexes). Accept only when the union beats the plain
    # polygon representation's plausibility (union IoU vs the raw mask).
    parts = decompose_into_rectangles(component_mask, cfg)
    if parts:
        # the greedy rects can land a hair (1 rotated-frame px) apart —
        # a mitre close (grow -> union -> shrink) bridges the seam while
        # keeping rectangle corners crisp
        from shapely import union_all

        gap_bridge = 1.25
        merged = union_all(
            [p.buffer(gap_bridge, join_style=2) for p in parts]
        ).buffer(-gap_bridge, join_style=2)
        # clamp outward drift of the close against the observed boundary
        merged = merged.intersection(raw_poly)
        if merged.is_empty:
            merged = None
        # a disjoint union means the greedy rects left speckle parts —
        # a coherent building must be ONE connected footprint
        if merged.geom_type == "MultiPolygon":
            biggest = max(merged.geoms, key=lambda g: g.area)
            if biggest.area < 0.85 * merged.area:
                merged = None  # too fragmented — keep the polygon fallback
            else:
                merged = biggest
        if merged is not None and not merged.is_empty:
            iou = _polygon_iou(poly, merged, x1 - x0, y1 - y0, bbox=bbox)
            n_primary = sum(1 for p in parts if p.area > 0.2 * poly.area)
            if iou >= 0.55 and n_primary >= 2:
                result["primitive"] = "composite_rectangles"
                result["footprint"] = merged
                result["fit_iou"] = iou
                result["decomposition_parts"] = parts
    return result


# ---------------------------------------------------------------------------
# Height assignment (CRITICAL: DSM-derived only)
# ---------------------------------------------------------------------------


def robust_height(dsm: np.ndarray, inside_mask: np.ndarray, ring_mask: np.ndarray,
                  cfg: Building3DConfig) -> dict[str, Any]:
    """Representative building height from the predicted metric DSM.

    ground/base = a low ring percentile OUTSIDE the footprint (robust to
    the building's own bump); height = MAD-trimmed median of
    (DSM - base) INSIDE the eroded footprint. Never a fixed value.
    """
    inside_vals = dsm[inside_mask]
    inside_vals = inside_vals[np.isfinite(inside_vals)]
    ring_vals = dsm[ring_mask]
    ring_vals = ring_vals[np.isfinite(ring_vals)]

    if inside_vals.size == 0:
        return {}
    base = float(np.percentile(ring_vals, cfg.ground_percentile)) \
        if ring_vals.size else 0.0
    rel = inside_vals - base

    median = float(np.median(rel))
    mad = float(np.median(np.abs(rel - median))) * 1.4826
    if mad > 1e-6:
        keep = np.abs(rel - median) <= cfg.height_mad_trim * mad
        trimmed = rel[keep]
    else:
        trimmed = rel
    height = float(np.median(trimmed)) if trimmed.size else median

    return {
        "base_elevation_m": round(base, 3),
        "height_m": round(height, 3),
        "median_m": round(median, 3),
        "mad_m": round(mad, 3),
        "p10_m": round(float(np.percentile(rel, 10)), 3),
        "p90_m": round(float(np.percentile(rel, 90)), 3),
        "n_samples": int(rel.size),
    }


def piecewise_levels(dsm: np.ndarray, inside_mask: np.ndarray, base: float,
                     height_m: float, cfg: Building3DConfig) -> list[dict[str, Any]]:
    """Detect multi-height complexes -> upper-level footprint polygons.

    A 1-D 2-means split on inside-heights; a second level is emitted only
    when the cluster gap is significant (metres AND pixel share).
    """
    vals = dsm[inside_mask] - base
    vals = vals[np.isfinite(vals)]
    if vals.size < 30:
        return []
    lo, hi = float(vals.min()), float(vals.max())
    if hi - lo < cfg.piecewise_gap_m:
        return []
    # 2-means
    c_lo, c_hi = lo + 0.25 * (hi - lo), lo + 0.75 * (hi - lo)
    for _ in range(24):
        assign = np.abs(vals - c_lo) <= np.abs(vals - c_hi)
        if assign.all() or (~assign).all():
            return []
        new_lo, new_hi = float(vals[assign].mean()), float(vals[~assign].mean())
        if abs(new_lo - c_lo) < 1e-3 and abs(new_hi - c_hi) < 1e-3:
            break
        c_lo, c_hi = new_lo, new_hi
    if c_hi - c_lo < cfg.piecewise_gap_m:
        return []
    upper_frac = float((~assign).mean())
    if upper_frac < cfg.piecewise_min_frac:
        return []
    threshold = 0.5 * (c_lo + c_hi)
    upper_mask = inside_mask & np.isfinite(dsm) & ((dsm - base) > threshold)
    if upper_mask.sum() < cfg.min_area_px:
        return []

    levels: list[dict[str, Any]] = []
    labels, n = ndimage.label(upper_mask)
    if n:
        sizes = ndimage.sum_labels(np.ones_like(upper_mask, dtype=np.int32),
                                   labels, index=np.arange(1, n + 1))
        order = np.argsort(sizes)[::-1][:2]
        for idx in order:
            if sizes[idx] < cfg.min_area_px:
                continue
            comp = labels == (idx + 1)
            core = ndimage.binary_erosion(comp, iterations=1) if comp.sum() > 12 else comp
            if not core.any():
                core = comp
            level_vals = (dsm[core] - base)
            level_vals = level_vals[np.isfinite(level_vals)]
            if level_vals.size == 0:
                continue
            lvl_height = float(np.median(level_vals))
            if lvl_height <= height_m + 0.5 * cfg.piecewise_gap_m:
                continue
            poly = _polygon_from_mask(comp)
            if poly is None:
                continue
            poly = poly.simplify(cfg.simplify_tol_px, preserve_topology=True)
            poly = _as_single_polygon(poly) or _polygon_from_mask(comp)
            if poly is None:
                continue
            levels.append({
                "polygon": poly,
                "height_m": round(lvl_height, 3),
                "pixel_count": int(comp.sum()),
            })
    return levels


# ---------------------------------------------------------------------------
# Watertight prism extrusion (validation/export)
# ---------------------------------------------------------------------------


def _earcut(polygon: Polygon) -> list[tuple[int, int, int]]:
    """Ear-clipping triangulation of a polygon with holes.

    Returns vertex-index triples indexing the combined ring vertex list.
    """
    exterior = list(polygon.exterior.coords)[:-1]
    rings = [exterior]
    index_base = len(exterior)
    holes_flat: list[tuple[float, float]] = []
    hole_ranges: list[tuple[int, int]] = []
    for hole in polygon.interiors:
        hole_coords = list(hole.coords)[:-1]
        hole_ranges.append((index_base, index_base + len(hole_coords)))
        holes_flat.extend(hole_coords)
        index_base += len(hole_coords)
    vertices = exterior + holes_flat

    # build the joined loop (exterior + bridges through holes) — for the
    # small vertex counts here, a simple ear clip on a polygon-with-holes
    # is done by bridging each hole to the exterior with a zero-width seam
    loop = list(range(len(exterior)))
    for start, end in hole_ranges:
        hole_idx = list(range(start, end))
        # bridge: closest exterior vertex to the hole's first vertex
        hx, hy = vertices[start]
        best_i, best_d = 0, float("inf")
        for i in loop:
            vx, vy = vertices[i]
            d = (vx - hx) ** 2 + (vy - hy) ** 2
            if d < best_d:
                best_d, best_i = d, i
        pos = loop.index(best_i)
        loop[pos + 1:pos + 1] = hole_idx + [best_i]

    def cross(o: int, a: int, b: int) -> float:
        (xo, yo), (xa, ya), (xb, yb) = vertices[o], vertices[a], vertices[b]
        return (xa - xo) * (yb - yo) - (ya - yo) * (xb - xo)

    def is_ear(prev: int, cur: int, nxt: int, ring: list[int]) -> bool:
        if cross(prev, cur, nxt) <= 1e-12:
            return False
        # no other vertex inside the ear
        for v in ring:
            if v in (prev, cur, nxt):
                continue
            # barycentric test
            ax, ay = vertices[prev]
            bx, by = vertices[cur]
            cx, cy = vertices[nxt]
            px, py = vertices[v]
            denom = ((by - cy) * (ax - cx) + (cx - bx) * (ay - cy))
            if abs(denom) < 1e-12:
                continue
            l1 = ((by - cy) * (px - cx) + (cx - bx) * (py - cy)) / denom
            l2 = ((cy - ay) * (px - cx) + (ax - cx) * (py - cy)) / denom
            l3 = 1.0 - l1 - l2
            if l1 > -1e-9 and l2 > -1e-9 and l3 > -1e-9:
                return False
        return True

    faces: list[tuple[int, int, int]] = []
    ring = list(loop)
    guard = 0
    while len(ring) > 3 and guard < 10000:
        guard += 1
        n = len(ring)
        ear_found = False
        for i in range(n):
            prev, cur, nxt = ring[i - 1], ring[i], ring[(i + 1) % n]
            if is_ear(prev, cur, nxt, ring):
                faces.append((prev, cur, nxt))
                ring.pop(i)
                ear_found = True
                break
        if not ear_found:
            # degenerate ring — clip anyway to keep the mesh closed
            faces.append((ring[0], ring[1], ring[2]))
            ring.pop(1)
    if len(ring) == 3:
        faces.append((ring[0], ring[1], ring[2]))
    return faces, vertices


def extrude_prism(footprint: Polygon, base_elev: float, top_elev: float
                  ) -> dict[str, Any]:
    """Watertight prism from a 2D footprint: top + bottom + side walls.

    The footprint polygon is in pixel coordinates; callers map vertices
    to world/CRS space afterwards (the topology is invariant to that
    transform). No internal geometry: the top/bottom triangulation spans
    the polygon exactly once, walls connect corresponding ring vertices.
    """
    faces_2d, vertices_2d = _earcut(footprint)
    n = len(vertices_2d)
    verts: list[tuple[float, float, float]] = []
    for (x, y) in vertices_2d:
        verts.append((float(x), float(y), float(base_elev)))
    for (x, y) in vertices_2d:
        verts.append((float(x), float(y), float(top_elev)))
    faces: list[tuple[int, int, int]] = []
    for (a, b, c) in faces_2d:
        faces.append((a, b, c))            # base (down-facing)
        faces.append((n + a, n + c, n + b))  # top (up-facing, reversed)
    # side walls — exterior ring and hole rings handled uniformly by
    # walking each ring of the ORIGINAL polygon (seam vertices excluded)
    def wall_ring(ring: list[tuple[float, float]], offset: int) -> None:
        m = len(ring)
        for i in range(m):
            j = (i + 1) % m
            # find vertex indices (they are unique per ring by position)
            faces.append((offset + i, offset + j, n + j))
            faces.append((offset + i, n + j, n + i))
    exterior = list(footprint.exterior.coords)[:-1]
    wall_ring(exterior, 0)
    # NOTE: hole rings would need their own vertex blocks; this module's
    # footprints are hole-free by construction (holes are filled during
    # regularization; composite parts are unions without holes).
    return {
        "vertices": [(round(x, 3), round(y, 3), round(z, 3)) for x, y, z in verts],
        "faces": faces,
    }


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


#: canonical damage class order (depthwizard.disaster.types.DAMAGE_CLASSES)
DAMAGE_CLASS_NAMES = ("no-damage", "minor-damage", "major-damage", "destroyed")

#: share of footprint pixels that must carry a label for classification
_DAMAGE_MIN_SHARE = 0.25


def _classify_damage(footprint_mask, damage_labels, damage_confidence):
    """Majority-vote damage classification for ONE footprint.

    Returns {damage_class, damage_confidence, pixel_share} or None when no
    damage raster exists or too few footprint pixels carry a label.
    Absence is reported honestly, never folded into a default class.
    """
    if damage_labels is None:
        return None
    # damage_labels encoding (depthwizard.disaster.damage_assessor):
    # 0 = background/unassessed, 1..4 = DAMAGE_CLASSES index + 1
    vals = damage_labels[footprint_mask]
    vals = vals[np.isfinite(vals)]
    labelled = vals[vals > 0]
    share = float(labelled.size) / float(vals.size) if vals.size else 0.0
    if labelled.size == 0 or share < _DAMAGE_MIN_SHARE:
        return None
    counts = np.bincount(labelled.astype(np.int64), minlength=5)
    winner = int(np.argmax(counts))
    if winner < 1 or winner > len(DAMAGE_CLASS_NAMES):
        return None
    conf = 1.0
    if damage_confidence is not None:
        mask_vals = damage_labels[footprint_mask]
        cvals = damage_confidence[footprint_mask]
        keep = np.isfinite(cvals) & (mask_vals == winner)
        if keep.any():
            conf = float(np.mean(cvals[keep]))
    return {
        "damage_class": DAMAGE_CLASS_NAMES[winner - 1],
        "damage_confidence": round(conf, 3),
        "pixel_share": round(share, 3),
    }


def _as_single_polygon(geom):
    """Return the largest polygon of a (Multi)polygon — level footprints
    must be single rings for the JSON contract and the viewer."""
    if geom is None or geom.is_empty:
        return None
    if geom.geom_type == "Polygon":
        return geom
    if hasattr(geom, "geoms"):
        biggest = max(geom.geoms, key=lambda g: g.area)
        return biggest if biggest.geom_type == "Polygon" else None
    return None


def fuse_building_candidates(dsm, primary_mask, *, rgb=None, ground_dsm=None,
                             config=None):
    """Recall-fusion detection pass (the 'detect EVERYTHING' layer).

    The primary semantic/ONNX mask is the PRECISION source; this pass adds
    RECALL: every region standing above the local ground that the primary
    mask does not cover becomes a candidate — missed buildings, trees,
    vehicles, containers. Candidates come from height discontinuities
    (DSM − local ground), which is exactly where structure edges live;
    an optional RGB frame adds a green-dominance check for vegetation.

    Returns (extra_mask, objects):
        extra_mask  bool [H,W] — additional building-candidate pixels
                    (clusters large enough to be structures)
        objects     list of {x_px, y_px, height_m, ground_elevation_m,
                    kind: 'tree'|'object', pixel_count} — small compact
                    clusters (canopies, vehicles, containers)
    """
    cfg = config or Building3DConfig()
    h, w = dsm.shape[:2]
    pm = primary_mask.astype(bool)
    if pm.shape != (h, w):
        raise ValueError("primary mask grid != DSM grid")

    # local ground (same recipe as the stage's inpaint)
    if ground_dsm is None:
        med = float(np.nanmedian(dsm))
        ground = ndimage.median_filter(np.nan_to_num(dsm, nan=med), size=64)
    else:
        ground = ground_dsm
    rel = dsm - ground

    structure = np.ones((3, 3), bool)
    cand = (
        np.isfinite(rel)
        & (rel > 1.2)
        & ~ndimage.binary_dilation(pm, structure=structure, iterations=2)
    )
    cand = ndimage.binary_opening(cand, structure=structure)
    labels, n = ndimage.label(cand)
    extra = np.zeros((h, w), dtype=bool)
    objects = []
    if not n:
        return extra, objects

    sizes = ndimage.sum_labels(np.ones_like(cand, np.int32), labels,
                               np.arange(1, n + 1))

    # green-dominance raster (when RGB exists) for tree vs object
    green = None
    if rgb is not None and rgb.shape[:2] == (h, w):
        r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
        green = (g > r * 1.05) & (g > b * 1.05) & (g > 40)

    order = np.argsort(sizes)[::-1]
    for idx in order:
        comp = labels == (idx + 1)
        area = int(sizes[idx])
        vals = rel[comp]
        vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            continue
        rel_h = float(np.median(vals))
        if area >= cfg.min_area_px and rel_h > 1.5:
            extra |= comp                      # missed building/structure
        elif area >= 25 and 0.8 <= rel_h <= 12.0:
            # small compact object: tree (canopy) or vehicle/container
            core = ndimage.binary_erosion(comp, structure=structure)
            use = core if core.sum() >= 12 else comp
            ys, xs = np.nonzero(use)
            cy, cx = int(ys.mean()), int(xs.mean())
            kind = "object"
            if green is not None:
                gshare = float(green[comp].mean())
                kind = "tree" if gshare > 0.25 else "object"
            else:
                # no RGB evidence: compact elevated clusters in residential
                # aerial scenes are overwhelmingly tree canopies
                kind = "tree" if rel_h > 1.6 else "object"
            objects.append({
                "x_px": round(float(xs.mean()), 1),
                "y_px": round(float(ys.mean()), 1),
                "height_m": round(rel_h, 2),
                "ground_elevation_m": round(float(ground[cy, cx]), 2),
                "kind": kind,
                "pixel_count": area,
            })
    return extra, objects


def reconstruct_buildings_3d(
    building_mask: np.ndarray,
    dsm: np.ndarray,
    *,
    confidence_raster: np.ndarray | None = None,
    mask_source: str = "semantic_building_class",
    crs: str | None = None,
    transform: Any = None,
    config: Building3DConfig | None = None,
    damage_labels: np.ndarray | None = None,
    damage_confidence: np.ndarray | None = None,
    source_ref_mask: np.ndarray | None = None,
) -> dict[str, Any]:
    """Full geometry-aware reconstruction -> JSON-serializable dict.

    The returned structure is written as ``buildings3d.json`` and consumed
    by the web viewer (BuildingsLayer). Every height field is in the DSM's
    metres; footprints are in prediction-grid pixel coordinates with an
    optional CRS-coordinate twin when the scene is georeferenced.
    """
    cfg = config or Building3DConfig()
    h, w = dsm.shape[:2]
    if building_mask.shape != (h, w):
        raise ValueError(
            f"building mask {building_mask.shape} != DSM grid {(h, w)} — "
            "refusing to reconstruct on misaligned rasters"
        )

    clean = regularize_mask(building_mask, cfg)
    if not clean.any():
        return {"available": False, "reason": "no building candidates", "buildings": []}

    labels, n_components = ndimage.label(clean)
    sizes = ndimage.sum_labels(np.ones_like(clean, dtype=np.int32), labels,
                               index=np.arange(1, n_components + 1))
    order = np.argsort(sizes)[::-1][:cfg.max_buildings]

    georeferenced = transform is not None

    def to_crs(ring_px: list[tuple[float, float]]) -> list[list[float]] | None:
        if not georeferenced:
            return None
        cols = np.array([p[0] for p in ring_px]) + 0.5
        rows = np.array([p[1] for p in ring_px]) + 0.5
        xs = transform.c + cols * transform.a + rows * transform.b
        ys = transform.f + cols * transform.d + rows * transform.e
        return [[round(float(x), 3), round(float(y), 3)] for x, y in zip(xs, ys)]

    buildings: list[dict[str, Any]] = []
    structure = np.ones((3, 3), dtype=bool)
    for comp_id in order:
        comp_id = int(comp_id) + 1
        comp = labels == comp_id
        if comp.sum() < cfg.min_area_px:
            continue

        # per-component confidence: mean candidate-mask confidence inside
        if confidence_raster is not None:
            conf_vals = confidence_raster[comp]
            conf_vals = conf_vals[np.isfinite(conf_vals)]
            comp_conf = float(np.mean(conf_vals)) if conf_vals.size else 0.5
        else:
            comp_conf = 0.8  # hard mask, no per-pixel confidence available

        fit = fit_footprint(comp, cfg, comp_conf)
        if not fit:
            continue
        footprint: Polygon = fit["footprint"]

        # height sampling inside the ERODED footprint (avoid edge bleed
        # from the building's own walls and from mask fringes)
        footprint_mask = _rasterize_polygon(footprint, w, h) & comp
        core = ndimage.binary_erosion(footprint_mask, structure=structure,
                                      iterations=1)
        if core.sum() < max(8, cfg.min_area_px // 4):
            core = footprint_mask

        # ring: dilated footprint minus the footprint itself
        dilated = ndimage.binary_dilation(footprint_mask, structure=structure,
                                          iterations=max(2, cfg.close_iterations + 1))
        ring = dilated & ~footprint_mask

        height_stats = robust_height(dsm, core, ring, cfg)
        if not height_stats or height_stats["height_m"] <= 0.05:
            continue  # not a raised structure — honest skip, never a fake height

        levels = piecewise_levels(
            dsm, core, height_stats["base_elevation_m"],
            height_stats["height_m"], cfg,
        )

        # -- disaster damage classification (when the damage model ran) -----
        # Majority vote of the damage model's per-pixel labels inside the
        # footprint. Heights stay 100% DSM-derived regardless of class: a
        # destroyed structure's debris height is the model's honest
        # estimate, never replaced by an assumed value.
        damage_entry = _classify_damage(
            footprint_mask, damage_labels, damage_confidence
        )

        # -- confidence -----------------------------------------------------
        boundary_iou = _polygon_iou(
            _polygon_from_mask(comp) or footprint, footprint, w, h)
        height_consistency = 1.0 - min(
            1.0, height_stats["mad_m"] / max(height_stats["height_m"], 0.1) / 0.5)
        confidence = float(
            cfg.w_mask * comp_conf
            + cfg.w_boundary * boundary_iou
            + cfg.w_fit * fit["fit_iou"]
            + cfg.w_height * max(0.0, height_consistency)
        )

        footprint = _as_single_polygon(footprint) or footprint
        ext_px = [[round(float(x), 2), round(float(y), 2)]
                  for x, y in footprint.exterior.coords[:-1]]
        # provenance: did the PRIMARY detector see this footprint, or is
        # it a recall-fusion candidate (DSM elevation evidence only)?
        if source_ref_mask is not None:
            overlap = float((comp & source_ref_mask).sum()) / max(int(comp.sum()), 1)
            comp_source = mask_source if overlap > 0.3 else "dsm_edge_fusion"
        else:
            comp_source = mask_source

        building: dict[str, Any] = {
            "id": len(buildings) + 1,
            "mask_source": comp_source,
            "primitive": fit["primitive"],
            "confidence": round(confidence, 3),
            "confidence_parts": {
                "mask": round(comp_conf, 3),
                "boundary": round(boundary_iou, 3),
                "fit": round(fit["fit_iou"], 3),
                "height_consistency": round(max(0.0, height_consistency), 3),
            },
            "fit_iou": round(fit["fit_iou"], 3),
            "height_m": height_stats["height_m"],
            "base_elevation_m": height_stats["base_elevation_m"],
            "top_elevation_m": round(
                height_stats["base_elevation_m"] + height_stats["height_m"], 3),
            "height_stats": height_stats,
            "footprint_px": ext_px,
            "footprint_crs": to_crs(ext_px),
            "footprint_area_m2": round(float(footprint.area), 2),
            "vertex_count": len(ext_px),
        }
        if damage_entry is not None:
            building["damage_class"] = damage_entry["damage_class"]
            building["damage_confidence"] = damage_entry["damage_confidence"]
            building["damage_pixel_share"] = damage_entry["pixel_share"]

        if levels:
            building["levels"] = [
                {
                    "polygon_px": [[round(float(x), 2), round(float(y), 2)]
                                   for x, y in lvl["polygon"].exterior.coords[:-1]],
                    "height_m": lvl["height_m"],
                    "pixel_count": lvl["pixel_count"],
                }
                for lvl in levels
            ]
        buildings.append(building)

    damage_classified = sum(1 for b in buildings if "damage_class" in b)
    return {
        "available": len(buildings) > 0,
        "count": len(buildings),
        "mask_source": mask_source,
        "height_source": "dsm.npy (predicted metric height)",
        "damage_classified": damage_classified,
        "damage_classes": sorted({b["damage_class"] for b in buildings
                                  if "damage_class" in b}),
        "units": "m",
        "georeferenced": georeferenced,
        "crs": crs,
        "grid": {"width": int(w), "height": int(h)},
        "buildings": buildings,
    }


# ---------------------------------------------------------------------------
# Preview rendering
# ---------------------------------------------------------------------------


def render_buildings3d_preview(rgb: np.ndarray, reconstruction: dict[str, Any]) -> np.ndarray:
    """RGB + footprint overlay colored by confidence (visual QA artifact)."""
    from PIL import Image as PILImage

    # damage classes override confidence colors in disaster scenes
    DAMAGE_COLORS = {
        "no-damage":     (76, 175, 80, 220),
        "minor-damage":  (255, 193, 7, 220),
        "major-damage":  (255, 87, 34, 220),
        "destroyed":     (211, 47, 47, 220),
    }

    base = PILImage.fromarray(rgb.astype(np.uint8)).convert("RGBA")
    overlay = PILImage.new("RGBA", base.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    for b in reconstruction.get("buildings", []):
        pts = [tuple(p) for p in b["footprint_px"]]
        if len(pts) < 3:
            continue
        if b.get("damage_class") in DAMAGE_COLORS:
            color = DAMAGE_COLORS[b["damage_class"]]
        else:
            conf = b.get("confidence", 0.0)
            if conf >= 0.7:
                color = (52, 211, 153, 220)     # green — high confidence
            elif conf >= 0.45:
                color = (56, 189, 248, 220)     # cyan — medium
            else:
                color = (251, 191, 36, 220)     # amber — low
        draw.polygon(pts, outline=color, width=2)
        label = b["primitive"] + f" {b['height_m']:.1f}m"
        if b.get("damage_class"):
            label += f" {b['damage_class']}"
        draw.text((pts[0][0] + 3, pts[0][1] + 3), label, fill=color)
    out = PILImage.alpha_composite(base, overlay)
    return np.asarray(out)
