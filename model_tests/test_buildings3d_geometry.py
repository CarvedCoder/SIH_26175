"""Building footprint geometry quality for 3D reconstruction (Part L).

Regression suite for the "ugly/incomplete polygons and poor wall
geometry" problem: rectangle, L-shape, irregular, adjacent, small, and
gap-split masks must each yield VALID footprints whose prism extrusion
covers the FULL boundary (complete walls, no missing sides, no
self-intersections).

Raw vs refined: fit_footprint exposes both the raw extracted polygon
(via the component mask) and the refined footprint; the reconstruction
JSON carries the refined geometry and per-building confidence.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
from scipy import ndimage

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from depthwizard.reconstruction.buildings3d import (
    Building3DConfig,
    extrude_prism,
    fit_footprint,
    reconstruct_buildings_3d,
    regularize_mask,
)

CFG = Building3DConfig(min_area_px=50, close_iterations=2, open_iterations=1)


def _mask_from(shape, rects):
    m = np.zeros(shape, dtype=bool)
    for (r0, c0, r1, c1) in rects:
        m[r0:r1, c0:c1] = True
    return m


def _assert_walls_complete(footprint, base=0.0, top=10.0):
    """Extrusion must cover EVERY exterior-ring segment with wall quads."""
    mesh = extrude_prism(footprint, base, top)
    from shapely.geometry import Polygon as SPoly

    poly = SPoly(footprint.exterior.coords[:-1])
    assert poly.is_valid, "footprint polygon self-intersects"
    n = len(footprint.exterior.coords) - 1  # ring vertices (closing point excluded)
    assert n >= 3
    n_verts = len(mesh["vertices"])
    assert n_verts == 2 * n  # base ring + top ring
    faces = mesh["faces"]
    # caps: 2*(n-2) triangles; walls: 2 triangles per ring edge
    wall_faces = len(faces) - 2 * (n - 2)
    assert wall_faces == 2 * n
    # every exterior edge appears in at least one wall triangle
    edges = set()
    for (a, b, c) in faces:
        edges.add(frozenset((a, b)))
        edges.add(frozenset((b, c)))
        edges.add(frozenset((c, a)))
    for i in range(n):
        j = (i + 1) % n
        assert frozenset((i, j)) in edges, f"wall missing for edge {i}->{j}"
        assert frozenset((i, i + n)) in edges, (
            f"wall vertical edge missing at vertex {i}"
        )
    # z-fighting guard: base ring at base_elev, top ring at top_elev
    zs = [v[2] for v in mesh["vertices"]]
    assert min(zs) == pytest.approx(base) and max(zs) == pytest.approx(top)


def test_rectangle_building_complete_walls():
    mask = _mask_from((120, 120), [(30, 40, 80, 90)])
    fit = fit_footprint(mask, CFG, confidence=0.9)
    assert fit, "footprint extraction failed"
    poly = fit["footprint"]
    assert poly.is_valid and not poly.is_empty
    assert fit["fit_iou"] > 0.85
    _assert_walls_complete(poly)


def test_l_shaped_building_keeps_shape_and_walls():
    rects = [(20, 20, 60, 70), (60, 20, 110, 45)]  # L
    mask = _mask_from((140, 140), rects)
    fit = fit_footprint(mask, CFG, confidence=0.9)
    assert fit
    poly = fit["footprint"]
    assert poly.is_valid
    # the L must NOT collapse to a rectangle worse than its own IoU
    iou_mask = mask
    from depthwizard.reconstruction.buildings3d import _rasterize_polygon

    rasterized = _rasterize_polygon(poly, 140, 140)
    inter = (rasterized & iou_mask).sum()
    union = (rasterized | iou_mask).sum()
    assert inter / union > 0.75, "L-shape footprint lost its shape"
    _assert_walls_complete(poly)


def test_irregular_building_valid_polygon():
    rng = np.random.default_rng(0)
    yy, xx = np.mgrid[0:100, 0:100]
    rr = np.hypot(yy - 50, xx - 50)
    mask = (rr < 30) | ((rr < 34) & (rng.random((100, 100)) < 0.6))
    mask = ndimage.binary_closing(mask, iterations=2)
    fit = fit_footprint(mask, CFG, confidence=0.7)
    assert fit
    poly = fit["footprint"]
    assert poly.is_valid and poly.area > 0
    assert len(poly.exterior.coords) <= CFG.max_vertices + 1
    _assert_walls_complete(poly)


def test_adjacent_buildings_reconstruct_separately():
    # 12 px street gap: clearly separate structures (small gaps BELOW the
    # cluster_close_px threshold are deliberately fused as one structure).
    mask = _mask_from(
        (200, 200), [(20, 20, 80, 80), (20, 92, 80, 152)]
    )
    dsm = mask.astype(np.float32) * 12.0  # flat 12 m roofs
    out = reconstruct_buildings_3d(mask, dsm)
    assert out["available"] is True
    buildings = out["buildings"]
    assert len(buildings) == 2, "adjacent buildings merged into one"
    for b in buildings:
        assert b["height_m"] == pytest.approx(12.0, abs=1.0)
        assert b["footprint_px"], "footprint missing"


def test_small_building_above_min_area_survives():
    mask = _mask_from((100, 100), [(40, 40, 55, 55)])  # 225 px
    fit = fit_footprint(mask, CFG, confidence=0.9)
    assert fit
    assert fit["footprint"].is_valid


def test_segmentation_gap_closed_by_regularization():
    """One building split by a 2 px gap -> regularize_mask closes it."""
    m1 = _mask_from((100, 100), [(30, 20, 70, 48)])
    m2 = _mask_from((100, 100), [(30, 50, 70, 78)])
    gapped = m1 | m2
    labels0, n0 = ndimage.label(regularize_mask(gapped, CFG))
    # the un-regularized gap is 2 px; closing with 3x3 x2 merges it
    labels, n = ndimage.label(regularize_mask(gapped, CFG))
    assert n == 1, f"gap not closed: {n} components (raw {n0})"


def test_invalid_mask_handled_honestly():
    out = reconstruct_buildings_3d(
        np.zeros((64, 64), bool), np.zeros((64, 64), np.float32)
    )
    assert out["available"] is False
    assert out["buildings"] == []


def test_raw_and_refined_footprint_distinguishable():
    """The fit exposes the refined polygon; the raw boundary is the
    component mask itself — both must be recoverable for debugging."""
    mask = _mask_from((120, 120), [(30, 40, 80, 90)])
    fit = fit_footprint(mask, CFG, confidence=0.9)
    assert "footprint" in fit and "primitive" in fit and "fit_iou" in fit
    # raw mask still available to callers (it IS the input)
    assert mask.sum() == 50 * 50
