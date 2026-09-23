"""Unit tests for semantic mapping: segmenter abstraction, predictions,
rendering, and output artifact serialization.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from depthwizard.datasets.semantics import NUM_PROJECT_CLASSES, PROJECT_CLASSES
from depthwizard.pipeline.scene_outputs import write_semantic_outputs
from depthwizard.semantic_segmenter import (
    SEMANTIC_COLORS,
    SEMANTIC_COLORS_HEX,
    SemanticPrediction,
    prediction_from_probs,
    render_semantic_map,
    semantic_metadata,
)


def _make_synthetic_probs(h: int = 32, w: int = 32) -> np.ndarray:
    """Generate synthetic softmax probabilities [6, H, W]."""
    # Create logits where each quadrant favors a different class
    # 0: building, 1: vegetation, 2: road, 3: water, 4: ground, 5: other
    logits = np.zeros((NUM_PROJECT_CLASSES, h, w), dtype=np.float32)
    half_h, half_w = h // 2, w // 2

    logits[0, :half_h, :half_w] = 5.0    # Top-left: building
    logits[1, :half_h, half_w:] = 5.0    # Top-right: vegetation
    logits[2, half_h:, :half_w] = 5.0    # Bottom-left: road
    logits[3, half_h:, half_w:] = 5.0    # Bottom-right: water

    # Softmax
    exp = np.exp(logits - logits.max(axis=0, keepdims=True))
    probs = exp / exp.sum(axis=0, keepdims=True)
    return probs.astype(np.float32)


def test_prediction_from_probs_valid():
    probs = _make_synthetic_probs(32, 32)
    pred = prediction_from_probs(probs)

    assert pred.height == 32
    assert pred.width == 32
    assert pred.num_classes == 6
    pred.validate()

    # Check quadrant classes
    assert pred.labels[0, 0] == 0       # building
    assert pred.labels[0, 31] == 1      # vegetation
    assert pred.labels[31, 0] == 2      # road
    assert pred.labels[31, 31] == 3     # water

    # Confidence should be high in all quadrants (> 0.8)
    assert pred.confidence.min() > 0.8
    assert pred.mean_confidence() > 0.8
    assert pred.low_confidence_fraction(0.5) == 0.0


def test_prediction_invalid_shape():
    with pytest.raises(ValueError, match="Expected probs shape"):
        prediction_from_probs(np.zeros((5, 10, 10), dtype=np.float32))

    with pytest.raises(ValueError, match="Expected probs shape"):
        prediction_from_probs(np.zeros((6, 10), dtype=np.float32))


def test_prediction_class_fractions():
    probs = _make_synthetic_probs(32, 32)
    pred = prediction_from_probs(probs)

    # 4 quadrants -> approximately 0.25 each for classes 0, 1, 2, 3
    for cid in range(4):
        frac = pred.class_fraction(cid)
        assert 0.20 <= frac <= 0.30

    assert pred.class_fraction(4) == 0.0  # ground
    assert pred.class_fraction(5) == 0.0  # other


def test_render_semantic_map():
    probs = _make_synthetic_probs(32, 32)
    pred = prediction_from_probs(probs)
    rgba = render_semantic_map(pred)

    assert rgba.shape == (32, 32, 4)
    assert rgba.dtype == np.uint8

    # Top-left (building) should match SEMANTIC_COLORS['building']
    expected_rgb = SEMANTIC_COLORS["building"]
    np.testing.assert_array_equal(rgba[0, 0, :3], expected_rgb)

    # Bottom-left (road) should match SEMANTIC_COLORS['road']
    expected_road = SEMANTIC_COLORS["road"]
    np.testing.assert_array_equal(rgba[31, 0, :3], expected_road)

    # Alpha channel should be high (confidence is high)
    assert rgba[:, :, 3].min() >= 153  # >= 60% opacity


def test_semantic_metadata():
    probs = _make_synthetic_probs(32, 32)
    pred = prediction_from_probs(probs)
    meta = semantic_metadata(pred, checkpoint_name="best.pt", model_name="CalibrationNet_aux")

    assert meta["classes"] == 6
    assert meta["checkpoint"] == "best.pt"
    assert meta["model"] == "CalibrationNet_aux"
    assert "class_fractions" in meta
    assert "legend" in meta
    assert meta["legend"]["building"] == SEMANTIC_COLORS_HEX["building"]


def test_write_semantic_outputs(tmp_path):
    out_dir = tmp_path / "scene_output"
    probs = _make_synthetic_probs(32, 32)

    outputs = write_semantic_outputs(
        out_dir=out_dir,
        sem_probs=probs,
        checkpoint_name="test_ckpt.pt",
        model_name="CalibrationNet_aux",
    )

    # Verify all expected artifacts are written
    assert "semantic_labels" in outputs
    assert "semantic_probs" in outputs
    assert "semantic_confidence" in outputs
    assert "semantic_map" in outputs
    assert "semantic_meta" in outputs

    for key, path_str in outputs.items():
        assert path_str is not None
        assert Path(path_str).is_file(), f"{key} file does not exist"

    # Verify labels content and dtype
    labels = np.load(outputs["semantic_labels"])
    assert labels.dtype == np.uint8
    assert labels.shape == (32, 32)

    # Verify probs compactness (float16)
    saved_probs = np.load(outputs["semantic_probs"])
    assert saved_probs.dtype == np.float16
    assert saved_probs.shape == (6, 32, 32)

    # Verify metadata JSON content
    with open(outputs["semantic_meta"], "r", encoding="utf-8") as f:
        meta_json = json.load(f)
    assert meta_json["checkpoint"] == "test_ckpt.pt"
    assert meta_json["classes"] == 6
