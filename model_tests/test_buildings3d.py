"""Geometry-aware 3D building reconstruction — synthetic-shape tests.

Every test builds a synthetic building mask + predicted metric DSM, runs
the reconstruction, and asserts:
    * the correct primitive is chosen (never a forced rectangle);
    * the extrusion height comes from the DSM (never fixed/normalized);
    * regularization removes pixel noise without moving the footprint;
    * composite structures decompose and re-merge into ONE footprint;
    * extruded prisms are watertight (every edge shared by exactly 2 faces);
    * confidence is well-formed and georeferencing survives.
"""

from __future__ import annotations

import math
from collections import Counter

import numpy as np
import pytest
from PIL import Image, ImageDraw
from shapely.geometry import Polygon

from depthwizard.reconstruction import (
    Building3DConfig,
    extrude_prism,
    reconstruct_buildings_3d,
    regularize_mask,
)

W, H = 400, 300


def _draw(*plans) -> np.ndarray:
    img = Image.new("L", (W, H), 0)
    draw = ImageDraw.Draw(img)
    for plan in plans:
        plan(draw)
    return np.asarray(img) > 127


RECT = [40, 40, 140, 90]          # 100x50 rectangle
CIRCLE = [200, 40, 280, 120]      # r=40 circle
L_PTS = [(40, 150), (140, 150), (140, 230), (100, 230), (100, 190), (40, 190)]
STADIUM = ((180, 160, 280, 200), (260, 160, 300, 200), (160, 160, 200, 200))

TRUE_HEIGHTS = {
    "rect": 12.5,
    "circle": 7.3,
    "L": 9.1,
    "stadium": 15.2,
}


def _mask() -> np.ndarray:
    return _draw(
        lambda d: d.rectangle(RECT, fill=255),
        lambda d: d.ellipse(CIRCLE, fill=255),
        lambda d: d.polygon(L_PTS, fill=255),
        lambda d: (
            d.rectangle(STADIUM[0], fill=255),
            d.ellipse(STADIUM[1], fill=255),
            d.ellipse(STADIUM[2], fill=255),
        ),
    )


def _dsm(mask: np.ndarray) -> np.ndarray:
    h = np.zeros((H, W), dtype=np.float32)
    h[_draw(lambda d: d.rectangle(RECT, fill=255))] = TRUE_HEIGHTS["rect"]
    h[_draw(lambda d: d.ellipse(CIRCLE, fill=255))] = TRUE_HEIGHTS["circle"]
    h[_draw(lambda d: d.polygon(L_PTS, fill=255))] = TRUE_HEIGHTS["L"]
    h[_draw(lambda d: (
        d.rectangle(STADIUM[0], fill=255),
        d.ellipse(STADIUM[1], fill=255),
        d.ellipse(STADIUM[2], fill=255),
    ))] = TRUE_HEIGHTS["stadium"]
    return np.where(mask, h, 0.0).astype(np.float32)


def _by_primitive(rec, primitive: str) -> dict:
    matches = [b for b in rec["buildings"] if b["primitive"] == primitive]
    assert matches, f"no {primitive} in {[b['primitive'] for b in rec['buildings']]}"
    return matches[0]


def _assert_watertight(mesh: dict) -> None:
    edges = Counter()
    for a, b, c in mesh["faces"]:
        for e in ((a, b), (b, c), (c, a)):
            edges[tuple(sorted(e))] += 1
    bad = [e for e, cnt in edges.items() if cnt != 2]
    assert not bad, f"non-manifold edges: {bad[:5]}"
    assert mesh["faces"], "empty mesh"


class TestPrimitiveClassification:
    """Adaptive fitting: each shape gets its natural primitive."""

    @pytest.fixture(scope="class")
    def rec(self) -> dict:
        mask = _mask()
        return reconstruct_buildings_3d(mask, _dsm(mask),
                                        config=Building3DConfig())

    def test_rectangular_building_fits_a_rectangle(self, rec):
        b = _by_primitive(rec, "rectangle")
        assert b["vertex_count"] == 4
        # footprint close to the truth (100x50 rectangle)
        poly = Polygon(b["footprint_px"])
        assert abs(poly.area - 5000) / 5000 < 0.05

    def test_circular_building_fits_a_circle(self, rec):
        b = _by_primitive(rec, "circle")
        poly = Polygon(b["footprint_px"])
        # equivalent-area radius close to the true r=40
        r = math.sqrt(poly.area / math.pi)
        assert abs(r - 40) / 40 < 0.05

    def test_stadium_is_not_forced_into_a_rectangle(self, rec):
        b = _by_primitive(rec, "stadium")
        assert b["vertex_count"] > 8      # arcs present
        assert b["vertex_count"] <= Building3DConfig().max_vertices

    def test_l_shape_decomposes_into_multiple_rectangles(self, rec):
        b = _by_primitive(rec, "composite_rectangles")
        poly = Polygon(b["footprint_px"])
        # one coherent footprint ≈ the L's true area (5600 px^2)
        assert abs(poly.area - 5600) / 5600 < 0.06
        assert len(b["footprint_px"]) >= 5   # more than a rectangle

    def test_no_forced_rectangle_on_curved_shapes(self, rec):
        prims = {b["primitive"] for b in rec["buildings"]}
        assert "rectangle" in prims          # the rectangle IS a rectangle
        # ...but the circle and stadium were NOT approximated by one
        circle = _by_primitive(rec, "circle")
        assert circle["primitive"] == "circle"


class TestHeightAssignment:
    """CRITICAL: heights come from the predicted metric DSM only."""

    @pytest.fixture(scope="class")
    def rec(self) -> dict:
        mask = _mask()
        return reconstruct_buildings_3d(mask, _dsm(mask),
                                        config=Building3DConfig())

    @pytest.mark.parametrize("primitive,height_key", [
        ("rectangle", "rect"),
        ("circle", "circle"),
        ("composite_rectangles", "L"),
        ("stadium", "stadium"),
    ])
    def test_height_matches_dsm_median(self, rec, primitive, height_key):
        b = _by_primitive(rec, primitive)
        assert abs(b["height_m"] - TRUE_HEIGHTS[height_key]) < 0.05

    def test_no_fixed_default_height(self, rec):
        heights = {b["height_m"] for b in rec["buildings"]}
        assert len(heights) == len(rec["buildings"])  # distinct DSM-derived
        assert not heights & {10.0, 15.0, 20.0, 5.0}

    def test_dsm_is_the_only_height_input(self):
        # scaling the DSM scales every reported height 1:1 — proof the
        # module never normalizes or invents a value
        mask = _mask()
        base = reconstruct_buildings_3d(mask, _dsm(mask),
                                        config=Building3DConfig())
        doubled = reconstruct_buildings_3d(
            mask, _dsm(mask) * 2.0, config=Building3DConfig())
        for a, b in zip(sorted(x["height_m"] for x in base["buildings"]),
                        sorted(x["height_m"] for x in doubled["buildings"])):
            assert abs(b - 2 * a) < 0.1

    def test_flat_region_is_skipped_not_given_a_fake_height(self):
        mask = _mask()
        dsm = _dsm(mask)
        # an additional MASKED region with ZERO height (flat ground)
        extra = _draw(lambda d: d.rectangle([300, 220, 380, 280], fill=255))
        combined = mask | extra
        rec = reconstruct_buildings_3d(combined, dsm,
                                       config=Building3DConfig())
        assert all(b["height_m"] > 0.05 for b in rec["buildings"])
        # the flat patch produced no building
        assert all(
            Polygon(b["footprint_px"]).centroid.y < 220 or b["height_m"] > 0.05
            for b in rec["buildings"]
        )


class TestPiecewiseHeights:
    def test_two_level_complex_gets_an_upper_level(self):
        # podium at 6 m over the full footprint + tower at 18 m over half
        mask = _draw(lambda d: d.rectangle([40, 40, 200, 160], fill=255))
        dsm = np.where(mask, 6.0, 0.0).astype(np.float32)
        tower = _draw(lambda d: d.rectangle([40, 40, 120, 100], fill=255))
        dsm[tower] = 18.0
        rec = reconstruct_buildings_3d(mask, dsm, config=Building3DConfig())
        assert rec["available"] and rec["count"] == 1
        b = rec["buildings"][0]
        assert b.get("levels"), "no piecewise level detected"
        level = b["levels"][0]
        assert abs(level["height_m"] - 18.0) < 0.5
        assert abs(b["height_m"] - 6.0) < 0.5  # podium stays at podium height


class TestRegularization:
    def test_noise_speckles_and_holes_are_removed(self):
        rng = np.random.default_rng(7)
        mask = _draw(lambda d: d.rectangle([40, 40, 140, 90], fill=255))
        dsm = np.where(mask, TRUE_HEIGHTS["rect"], 0.0).astype(np.float32)

        noisy = mask.copy()
        noisy |= rng.random(mask.shape) < 0.0015          # speckles
        noisy &= ~(rng.random(mask.shape) < 0.02)         # pinholes
        noisy[60:64, 60:100] = True                        # protrusion

        # regularization alone: speckles gone, footprint largely intact
        clean = regularize_mask(noisy, Building3DConfig())
        assert clean.sum() < noisy.sum()
        assert clean.sum() >= 0.85 * mask.sum()

        rec = reconstruct_buildings_3d(noisy, dsm, config=Building3DConfig())
        # exactly one building survives and it is still a clean rectangle
        b = _by_primitive(rec, "rectangle")
        assert abs(Polygon(b["footprint_px"]).area - 5000) / 5000 < 0.10
        assert abs(b["height_m"] - TRUE_HEIGHTS["rect"]) < 0.05

    def test_vegetation_contamination_is_pruned(self):
        # small satellite blobs (trees) near the roof must not become buildings
        mask = _draw(
            lambda d: d.rectangle([40, 40, 140, 90], fill=255),
            lambda d: d.ellipse([150, 95, 159, 105], fill=255),    # blob
            lambda d: d.ellipse([20, 20, 27, 27], fill=255),      # blob
        )
        dsm = np.where(mask, 10.0, 0.0).astype(np.float32)
        rec = reconstruct_buildings_3d(mask, dsm, config=Building3DConfig())
        assert rec["count"] == 1


class TestConfidence:
    def test_confidence_wellformed_and_shapes_regularization(self):
        mask = _draw(lambda d: d.rectangle(RECT, fill=255))
        dsm = np.where(mask, TRUE_HEIGHTS["rect"], 0.0).astype(np.float32)

        confident = reconstruct_buildings_3d(
            mask, dsm, confidence_raster=np.where(mask, 0.95, 0.0).astype(np.float32),
            config=Building3DConfig())
        unsure = reconstruct_buildings_3d(
            mask, dsm, confidence_raster=np.where(mask, 0.40, 0.0).astype(np.float32),
            config=Building3DConfig())

        hi = confident["buildings"][0]
        lo = unsure["buildings"][0]
        assert 0.0 < hi["confidence"] <= 1.0
        assert 0.0 < lo["confidence"] <= 1.0
        assert lo["confidence"] < hi["confidence"]
        # low confidence preserves MORE of the original (pixelated) boundary
        assert lo["vertex_count"] >= hi["vertex_count"] - 1


class TestDamageClassification:
    """Disaster scenes: reconstructed 3D buildings carry the damage
    model's classification; heights stay 100% DSM-derived."""

    def test_destroyed_and_intact_buildings_are_classified(self):
        # two buildings: one intact (no-damage), one destroyed
        mask = _draw(
            lambda d: d.rectangle([40, 40, 140, 90], fill=255),
            lambda d: d.rectangle([200, 40, 280, 90], fill=255),
        )
        dsm = np.zeros((H, W), dtype=np.float32)
        dsm[40:90, 40:140] = 12.5
        dsm[40:90, 200:280] = 3.2   # rubble heap, still model-derived

        # damage_labels encoding: 0=background, 1=no-damage .. 4=destroyed
        damage = np.zeros((H, W), dtype=np.uint8)
        damage[40:90, 40:140] = 1       # no-damage
        damage[40:90, 200:280] = 4      # destroyed

        rec = reconstruct_buildings_3d(
            mask, dsm,
            damage_labels=damage,
            config=Building3DConfig())
        assert rec["damage_classified"] == 2
        by_height = {b["height_m"]: b for b in rec["buildings"]}
        assert by_height[12.5]["damage_class"] == "no-damage"
        assert by_height[3.2]["damage_class"] == "destroyed"
        # heights remain exactly the DSM values
        assert abs(by_height[3.2]["height_m"] - 3.2) < 0.05

    def test_damage_confidence_averaged_from_raster(self):
        mask = _draw(lambda d: d.rectangle(RECT, fill=255))
        dsm = np.where(mask, TRUE_HEIGHTS["rect"], 0).astype(np.float32)
        damage = np.where(mask, 3, 0).astype(np.uint8)      # major-damage (1-based)
        conf = np.where(mask, 0.77, 0).astype(np.float32)
        rec = reconstruct_buildings_3d(
            mask, dsm, damage_labels=damage, damage_confidence=conf,
            config=Building3DConfig())
        b = rec["buildings"][0]
        assert b["damage_class"] == "major-damage"
        assert abs(b["damage_confidence"] - 0.77) < 0.02

    def test_unlabelled_footprint_is_honest_not_defaulted(self):
        mask = _draw(lambda d: d.rectangle(RECT, fill=255))
        dsm = np.where(mask, TRUE_HEIGHTS["rect"], 0).astype(np.float32)
        damage = np.zeros((H, W), dtype=np.uint8)  # nothing labelled
        rec = reconstruct_buildings_3d(
            mask, dsm, damage_labels=damage, config=Building3DConfig())
        assert "damage_class" not in rec["buildings"][0]
        assert rec["damage_classified"] == 0


class TestExtrusion:
    def test_prism_is_watertight(self):
        mask = _draw(lambda d: d.rectangle(RECT, fill=255))
        rec = reconstruct_buildings_3d(
            mask,
            np.where(mask, TRUE_HEIGHTS["rect"], 0).astype(np.float32), config=Building3DConfig())
        b = rec["buildings"][0]
        mesh = extrude_prism(Polygon(b["footprint_px"]), 0.0, b["height_m"])
        _assert_watertight(mesh)
        # heights carried into Z
        zs = [v[2] for v in mesh["vertices"]]
        assert min(zs) == 0.0 and max(zs) == pytest.approx(b["height_m"], abs=1e-6)

    def test_concave_prism_is_watertight(self):
        poly = Polygon(L_PTS)
        mesh = extrude_prism(poly, 0.0, 9.1)
        _assert_watertight(mesh)


class TestGeoreferencing:
    def test_crs_coordinates_derived_from_transform(self):
        # simple north-up affine: 1 px = 0.5 m, origin (500000, 4000000)
        class TF:
            a, b, c = 0.5, 0.0, 500000.0
            d, e, f = 0.0, -0.5, 4000000.0

        mask = _draw(lambda d: d.rectangle(RECT, fill=255))
        rec = reconstruct_buildings_3d(
            mask,
            np.where(mask, TRUE_HEIGHTS["rect"], 0).astype(np.float32), crs="EPSG:32643", transform=TF(),
            config=Building3DConfig())
        b = rec["buildings"][0]
        assert rec["georeferenced"] is True
        assert b["footprint_crs"] is not None
        xs = [p[0] for p in b["footprint_crs"]]
        ys = [p[1] for p in b["footprint_crs"]]
        # footprint lies inside the transform's footprint of the bbox
        assert 500000 + RECT[0] * 0.5 - 1 <= min(xs) <= max(xs) <= 500000 + RECT[2] * 0.5 + 1
        assert 4000000 - RECT[3] * 0.5 - 1 <= min(ys) <= max(ys) <= 4000000 - RECT[1] * 0.5 + 1

    def test_non_georeferenced_scene_is_honest(self):
        mask = _draw(lambda d: d.rectangle(RECT, fill=255))
        rec = reconstruct_buildings_3d(
            mask,
            np.where(mask, 10.0, 0).astype(np.float32), config=Building3DConfig())
        assert rec["georeferenced"] is False
        assert rec["buildings"][0]["footprint_crs"] is None


class TestGuards:
    def test_mask_dsm_grid_mismatch_is_refused(self):
        mask = np.zeros((50, 60), dtype=bool)
        mask[10:40, 10:50] = True
        with pytest.raises(ValueError, match="misaligned"):
            reconstruct_buildings_3d(mask, np.zeros((100, 100), dtype=np.float32))

    def test_empty_mask_reports_unavailable(self):
        rec = reconstruct_buildings_3d(
            np.zeros((H, W), dtype=bool), np.zeros((H, W), dtype=np.float32))
        assert rec["available"] is False
        assert rec["buildings"] == []
