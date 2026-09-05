"""Tests for the Phase-5 metrics extensions + scene-type classifier.

Covers slope_error (GSD honesty), building_metrics, project-class
stratification, and the documented scene-type decision order.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from depthwizard.datasets.semantics import PROJECT_CLASSES, semantic_layers
from depthwizard.metrics import (building_metrics, slope_error,
                                 stratified_by_project_class)
from depthwizard.scene_types import (SCENE_TYPES, classify_scene,
                                     stratify_by_scene_type)


def _ramp(h=64, w=64):
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    return xx * 0.5, yy * 0.2


class TestSlopeError:
    def test_refuses_without_gsd(self):
        pred, target = _ramp()
        assert slope_error(pred, target, gsd_m=None) is None
        assert slope_error(pred, target, gsd_m=0.0) is None

    def test_zero_error_for_identical(self):
        pred, _ = _ramp()
        r = slope_error(pred, pred, gsd_m=0.33)
        assert r["slope_mae_deg"] == pytest.approx(0.0, abs=1e-9)

    def test_flat_vs_ramp_error(self):
        h, w = 64, 64
        flat = np.zeros((h, w), np.float32)
        ramp = np.tile(np.linspace(0, 10, w, dtype=np.float32), (h, 1))
        r = slope_error(flat, ramp, gsd_m=0.33)
        assert r["slope_mae_deg"] > 1.0            # ramp slope ~ atan(10/(0.33*63))
        assert np.isfinite(r["slope_rmse_deg"])

    def test_border_excluded(self):
        """Central differences are undefined at the 1-px border — the mask
        excludes it (n == (h-2)*(w-2) for a full-valid target)."""
        h, w = 16, 16
        pred = np.zeros((h, w), np.float32)
        r = slope_error(pred, pred, gsd_m=1.0)
        assert r["n"] == (h - 2) * (w - 2)


class TestBuildingMetrics:
    def test_restricts_to_building_pixels(self):
        h, w = 32, 32
        pred = np.zeros((h, w), np.float32)
        target = np.zeros((h, w), np.float32)
        target[:8, :8] = 10.0                     # building region
        pred[:8, :8] = 12.0                       # +2 m error on buildings
        mask = np.zeros((h, w), bool)
        mask[:8, :8] = True
        m = building_metrics(pred, target, mask)
        assert m["n"] == 64
        assert m["mae"] == pytest.approx(2.0)
        assert m["rmse"] == pytest.approx(2.0)

    def test_from_verified_dfc_legend(self):
        """The building mask derives from the VERIFIED DFC legend (6)."""
        cls = np.full((8, 8), 2, np.int32)        # ground
        cls[:2, :2] = 6                           # buildings
        onehot, _ig, _ = semantic_layers(cls, "dfc2019")
        assert building_metrics(np.zeros((8, 8)), np.zeros((8, 8)),
                                onehot[0] > 0.5)["n"] == 4


class TestProjectClassStratification:
    def test_gamus_all_classes_covered(self):
        cls = np.array([[0, 1, 3], [4, 5, 6]], np.uint8)
        onehot, _ig, _ = semantic_layers(cls, "gamus")
        out = stratified_by_project_class(np.zeros((2, 3)),
                                          np.zeros((2, 3)), onehot)
        # all six project classes appear in this fixture
        assert set(out) == set(PROJECT_CLASSES)
        for name, m in out.items():
            assert m["n"] == 1


class TestSceneTypes:
    def test_all_types_present_in_catalog(self):
        assert set(SCENE_TYPES) == {"urban", "sparse-suburban", "forest",
                                    "hilly-high-relief", "flat"}

    def test_urban(self):
        cls = np.full((16, 16), 3, np.uint8)      # gamus buildings -> >=0.15
        agl = np.zeros((16, 16), np.float32)
        assert classify_scene(agl, cls, "gamus") == "urban"

    def test_forest(self):
        cls = np.full((16, 16), 6, np.uint8)      # gamus tree -> vegetation
        agl = np.zeros((16, 16), np.float32)
        assert classify_scene(agl, cls, "gamus") == "forest"

    def test_flat(self):
        cls = np.full((16, 16), 1, np.uint8)      # gamus ground
        agl = np.zeros((16, 16), np.float32)      # no relief
        assert classify_scene(agl, cls, "gamus") == "flat"

    def test_hilly_high_relief(self):
        h = 64
        cls = np.full((h, h), 1, np.uint8)        # ground, few buildings
        yy, _ = np.mgrid[0:h, 0:h]
        agl = (yy * 1.0).astype(np.float32)       # 0..63 m relief
        assert classify_scene(agl, cls, "gamus") == "hilly-high-relief"

    def test_sparse_suburban(self):
        cls = np.full((16, 16), 1, np.uint8)
        cls[:2, :2] = 3                           # 4/256 = 1.5% buildings?
        agl = np.zeros((16, 16), np.float32)
        # 4/256 = 0.0156 < 0.02 -> flat; make it 8 buildings -> 0.031
        cls[:3, :3] = 3
        assert classify_scene(agl, cls, "gamus") == "sparse-suburban"

    def test_stratify_grouping(self):
        per_tile = [{"mae": 1.0, "rmse": 2.0}, {"mae": 3.0, "rmse": 4.0}]
        types = ["urban", "urban"]
        tab = stratify_by_scene_type(per_tile, types)
        assert tab["urban"]["mae_mean"] == pytest.approx(2.0)
        assert tab["urban"]["n_tiles"] == 2
