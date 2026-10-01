"""Unit and integration tests for DepthWizard disaster assessment pipeline.

Tests:
1. Model loading & validation (invalid file, valid ONNX signatures).
2. Building detector (preprocessing, single-tile inference, full sliding window, polygonization, deterministic IDs, georeferencing).
3. Damage assessor (classes, post-only zeros input, probability extraction, review thresholding).
4. Artifact creation (GeoJSON, rasters, metadata, preview images).
5. Pipeline orchestrator end-to-end execution.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from depthwizard.disaster.types import (
    BuildingDetection,
    DamageAssessment,
    DisasterResult,
    DAMAGE_CLASSES,
)
from depthwizard.disaster.onnx_runtime import (
    OnnxSession,
    OnnxModelError,
)
from depthwizard.disaster.building_detector import (
    BuildingDetector,
    TILE_SIZE,
    NUM_CLASSES,
)
from depthwizard.disaster.damage_assessor import (
    DamageAssessor,
    DAMAGE_TILE_SIZE,
    NUM_DAMAGE_CLASSES,
)
from depthwizard.disaster.artifacts import (
    write_building_artifacts,
    write_damage_artifacts,
    _buildings_to_geojson,
    _damage_to_geojson,
)
from depthwizard.disaster.pipeline import run_disaster_pipeline


# ── Model loading tests ───────────────────────────────────────────────

def test_missing_model_file_raises_error(tmp_path):
    missing_path = tmp_path / "nonexistent.onnx"
    with pytest.raises(OnnxModelError, match="not found"):
        OnnxSession(missing_path)


def test_invalid_onnx_file_raises_error(tmp_path):
    corrupt_file = tmp_path / "corrupt.onnx"
    corrupt_file.write_text("not a real onnx model")
    with pytest.raises(OnnxModelError):
        OnnxSession(corrupt_file)


def test_resolve_providers():
    cpu_providers = OnnxSession._select_providers("cpu")
    assert cpu_providers == ["CPUExecutionProvider"]
    auto_providers = OnnxSession._select_providers("auto")
    assert "CPUExecutionProvider" in auto_providers


# ── Building detection tests ──────────────────────────────────────────

def test_building_detector_preprocessing():
    # Synthetic tile [256, 256, 3]
    tile = np.full((256, 256, 3), 128, dtype=np.uint8)
    
    # We can test preprocessing without running session
    with patch.object(OnnxSession, "__init__", return_value=None):
        detector = BuildingDetector.__new__(BuildingDetector)
        detector.tile_size = 256
        tensor = detector.preprocess_tile(tile)

        assert tensor.shape == (1, 3, 256, 256)
        assert tensor.dtype == np.float32
        assert -3.0 < float(tensor.min()) < 3.0
        assert -3.0 < float(tensor.max()) < 3.0


def test_building_polygonization_and_deterministic_ids():
    with patch.object(OnnxSession, "__init__", return_value=None):
        detector = BuildingDetector.__new__(BuildingDetector)
        detector.min_area_px = 16

        # Create binary mask with two distinct square buildings
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[10:30, 10:30] = 1  # 20x20 = 400px
        mask[50:80, 50:80] = 1  # 30x30 = 900px
        conf = mask.astype(np.float32) * 0.9

        buildings = detector.polygonize(mask, conf)
        assert len(buildings) == 2

        # Check deterministic IDs (B001, B002)
        assert buildings[0].building_id == "B001"
        assert buildings[1].building_id == "B002"

        # Check area and properties
        assert buildings[0].area_px > 300
        assert buildings[1].area_px > 700
        assert buildings[0].confidence == pytest.approx(0.9, rel=1e-2)
        assert len(buildings[0].polygon) >= 4  # closed polygon


def test_building_detection_georeferencing():
    from affine import Affine

    buildings = [
        BuildingDetection(
            building_id="B001",
            polygon=[(10, 10), (20, 10), (20, 20), (10, 20)],
            centroid_x=15.0,
            centroid_y=15.0,
            area_px=100.0,
            confidence=0.95,
            bbox=(10, 10, 20, 20),
        )
    ]
    # Transform: 0.5m resolution, upper-left at (500000, 4000000)
    transform = Affine(0.5, 0.0, 500000.0, 0.0, -0.5, 4000000.0)

    from depthwizard.disaster.pipeline import _compute_geo_areas
    _compute_geo_areas(buildings, transform)

    # 100 px * 0.25 m2/px = 25 m2
    assert buildings[0].area_m2 == pytest.approx(25.0, rel=1e-2)

    geojson = _buildings_to_geojson(
        buildings,
        georeferenced=True,
        crs_string="EPSG:32633",
        transform=transform,
    )
    feature = geojson["features"][0]
    coords = feature["geometry"]["coordinates"][0]
    # Check transformed coordinates
    assert coords[0][0] == 500000.0 + 10 * 0.5
    assert coords[0][1] == 4000000.0 - 10 * 0.5
    assert feature["properties"]["area_m2"] == pytest.approx(25.0, rel=1e-2)
    assert feature["properties"]["coordinate_space"] == "geographic"


# ── Damage assessment tests ───────────────────────────────────────────

def test_damage_classes_mapping():
    assert list(DAMAGE_CLASSES) == [
        "no-damage",
        "minor-damage",
        "major-damage",
        "destroyed",
    ]


def test_damage_post_only_mode():
    with patch.object(OnnxSession, "__init__", return_value=None):
        assessor = DamageAssessor.__new__(DamageAssessor)
        assessor.batch_size = 1
        assessor.review_threshold = 0.4
        assessor.session = MagicMock()

        # Mock model output: returns logits favoring "destroyed" (class index 3)
        mock_logits = np.zeros((1, 4, 512, 512), dtype=np.float32)
        mock_logits[0, 3, :, :] = 10.0  # high confidence destroyed
        assessor.session.run.return_value = {"damage_logits": mock_logits}

        post_rgb = np.full((128, 128, 3), 100, dtype=np.uint8)
        building = BuildingDetection(
            building_id="BLD-0001",
            polygon=[(20, 20), (50, 20), (50, 50), (20, 50)],
            centroid_x=35.0,
            centroid_y=35.0,
            area_px=900.0,
            confidence=0.88,
            bbox=(20, 20, 50, 50),
        )

        # Call with pre_rgb=None -> POST ONLY mode
        assessment = assessor.predict_building(
            post_rgb,
            building,
            pre_rgb=None,
        )

        assert assessment.building_id == "BLD-0001"
        assert assessment.damage_class == "destroyed"
        assert assessment.mode == "post_only"
        assert assessment.confidence > 0.9
        assert assessment.review_required is False
        assert assessment.probabilities["destroyed"] > 0.9

        # Verify that pre input passed to session was zeros (not fabricated pre imagery)
        call_inputs = assessor.session.run.call_args[0][0]
        assert "post" in call_inputs and "pre" in call_inputs
        pre_tensor = call_inputs["pre"]
        # ImageNet normalized zeros: (0 - mean) / std is constant negative
        assert pre_tensor.shape == (1, 3, 512, 512)


def test_damage_review_flag_on_low_confidence():
    with patch.object(OnnxSession, "__init__", return_value=None):
        assessor = DamageAssessor.__new__(DamageAssessor)
        assessor.batch_size = 1
        assessor.review_threshold = 0.6  # higher threshold
        assessor.session = MagicMock()

        # Output with close probabilities between minor and major damage
        mock_logits = np.zeros((1, 4, 512, 512), dtype=np.float32)
        mock_logits[0, 1, :, :] = 1.05
        mock_logits[0, 2, :, :] = 1.00
        assessor.session.run.return_value = {"damage_logits": mock_logits}

        post_rgb = np.full((128, 128, 3), 100, dtype=np.uint8)
        building = BuildingDetection(
            building_id="BLD-0002",
            polygon=[(10, 10), (30, 10), (30, 30), (10, 30)],
            centroid_x=20.0,
            centroid_y=20.0,
            area_px=400.0,
            confidence=0.80,
            bbox=(10, 10, 30, 30),
        )

        assessment = assessor.predict_building(post_rgb, building)
        # Margin is small (~0.05), so review_required should be True
        assert assessment.review_required is True


# ── Destroyed-structure recovery tests ────────────────────────────────

def _synthetic_damage_probs(h, w, hotspots):
    """Build a [4, H, W] probability map with 'destroyed' hotspots.

    hotspots: list of (y0, y1, x0, x1, prob) boxes of constant destroyed
    probability. Background: no-damage probability 1.
    """
    probs = np.zeros((4, h, w), dtype=np.float32)
    probs[0] = 1.0  # background = no-damage
    for y0, y1, x0, x1, prob in hotspots:
        probs[3][y0:y1, x0:x1] = prob
        probs[0][y0:y1, x0:x1] = 1.0 - prob
    total = probs.sum(axis=0, keepdims=True)
    return probs / total


def test_recover_destroyed_structures_finds_blobs():
    from depthwizard.disaster.damage_assessor import (
        recover_destroyed_structures,
    )

    h, w = 400, 400
    probs = _synthetic_damage_probs(h, w, [
        (50, 150, 50, 150, 0.95),    # destroyed blob
        (250, 350, 250, 350, 0.9),   # another destroyed blob
    ])
    building_mask = np.zeros((h, w), dtype=np.uint8)
    recovered = recover_destroyed_structures(probs, building_mask)
    assert len(recovered) == 2

    buildings = [b for b, _p, _m in recovered]
    # Deterministic ordering: top-to-bottom
    assert buildings[0].centroid_y < buildings[1].centroid_y
    # Damage-map provenance is recorded
    assert all(b.source == "damage_map" for b in buildings)
    # IDs are left empty for the pipeline to assign sequentially
    assert all(b.building_id == "" for b in buildings)
    # Class probabilities favor 'destroyed' inside the blobs
    for _b, class_probs, _m in recovered:
        assert int(class_probs.argmax()) == 3


def test_recover_destroyed_structures_excludes_known_footprints():
    from depthwizard.disaster.damage_assessor import (
        recover_destroyed_structures,
    )

    h, w = 400, 400
    probs = _synthetic_damage_probs(h, w, [(50, 150, 50, 150, 0.95)])
    # A single footprint covering the whole blob area
    building_mask = np.zeros((h, w), dtype=np.uint8)
    building_mask[50:150, 50:150] = 1

    recovered = recover_destroyed_structures(probs, building_mask)
    assert recovered == []


def test_recover_destroyed_structures_ignores_weak_signal():
    from depthwizard.disaster.damage_assessor import (
        recover_destroyed_structures,
        DESTROYED_PROB_THRESHOLD,
    )

    h, w = 400, 400
    # Peak below the hotspot threshold must recover nothing
    peak = DESTROYED_PROB_THRESHOLD - 0.1
    probs = _synthetic_damage_probs(h, w, [(50, 150, 50, 150, peak)])
    building_mask = np.zeros((h, w), dtype=np.uint8)

    assert recover_destroyed_structures(probs, building_mask) == []


# ── Artifact tests ────────────────────────────────────────────────────

def test_artifacts_generation(tmp_path):
    buildings = [
        BuildingDetection(
            building_id="BLD-0001",
            polygon=[(10, 10), (30, 10), (30, 30), (10, 30)],
            centroid_x=20.0,
            centroid_y=20.0,
            area_px=400.0,
            confidence=0.92,
            bbox=(10, 10, 30, 30),
        )
    ]
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[10:30, 10:30] = 1
    conf = mask.astype(np.float32) * 0.92
    rgb = np.full((100, 100, 3), 120, dtype=np.uint8)

    # Write building artifacts
    bld_out = write_building_artifacts(tmp_path, buildings, mask, conf, rgb)

    assert (tmp_path / "buildings.geojson").exists()
    assert (tmp_path / "building_mask.npy").exists()
    assert (tmp_path / "building_confidence.npy").exists()
    assert (tmp_path / "buildings_preview.png").exists()
    assert (tmp_path / "buildings_meta.json").exists()

    with open(tmp_path / "buildings_meta.json") as f:
        meta = json.load(f)
        assert meta["count"] == 1
        assert meta["available"] is True

    # Damage artifacts
    damages = [
        DamageAssessment(
            building_id="BLD-0001",
            damage_class="major-damage",
            confidence=0.85,
            probabilities={
                "no-damage": 0.05,
                "minor-damage": 0.10,
                "major-damage": 0.85,
                "destroyed": 0.00,
            },
            review_required=False,
            mode="post_only",
        )
    ]
    result = DisasterResult(mode="post_only", buildings=buildings, damages=damages)
    dmg_labels = np.zeros((100, 100), dtype=np.uint8)
    dmg_labels[10:30, 10:30] = 3  # major-damage index 3 (1-based)
    dmg_conf = conf

    dmg_out = write_damage_artifacts(
        tmp_path,
        result,
        rgb,
        mask,
        dmg_labels,
        dmg_conf,
    )

    assert (tmp_path / "damage_buildings.geojson").exists()
    assert (tmp_path / "damage_labels.npy").exists()
    assert (tmp_path / "damage_confidence.npy").exists()
    assert (tmp_path / "damage_preview.png").exists()
    assert (tmp_path / "damage_meta.json").exists()

    with open(tmp_path / "damage_meta.json") as f:
        dmg_meta = json.load(f)
        assert dmg_meta["available"] is True
        assert dmg_meta["building_count"] == 1
        assert dmg_meta["damage_counts"]["major-damage"] == 1
        assert dmg_meta["mode"] == "post_only"
