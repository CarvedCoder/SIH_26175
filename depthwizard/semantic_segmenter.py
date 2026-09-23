"""Semantic segmenter abstraction — the ONE entry point for predicted semantics.

Wraps the CalibrationNet auxiliary semantic head (sem_aux_head checkpoints)
to provide a clean segmentation interface for Inspect Mode and Route Assist.

Both consumers share the SAME semantic prediction — generated once per scene,
persisted as artifacts, then loaded by each system. This module is only
responsible for producing the predictions; artifact I/O lives in
``pipeline.scene_outputs``.

Project-level classes (frozen, from datasets.semantics.PROJECT_CLASSES):
    0 = building
    1 = vegetation
    2 = road
    3 = water
    4 = ground
    5 = other

ANTI-FABRICATION CONTRACT:
    * Only the model's own softmax probabilities are used — never GT masks.
    * When the checkpoint has no auxiliary head, ``predict()`` returns None
      and every downstream system degrades honestly.
    * Probabilities are preserved (never immediately reduced to hard labels).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

from .datasets.semantics import NUM_PROJECT_CLASSES, PROJECT_CLASSES

# Canonical semantic class color palette (RGB 0-255) — stable across scenes.
SEMANTIC_COLORS: Dict[str, tuple] = {
    "building":   (231,  76,  60),   # #E74C3C — warm red
    "vegetation": ( 46, 204, 113),   # #2ECC71 — green
    "road":       (155, 155, 155),   # #9B9B9B — neutral grey
    "water":      ( 52, 152, 219),   # #3498DB — blue
    "ground":     (210, 180, 140),   # #D2B48C — tan
    "other":      (149, 130, 168),   # #9582A8 — muted purple
}

SEMANTIC_COLORS_HEX: Dict[str, str] = {
    name: "#{:02X}{:02X}{:02X}".format(*rgb)
    for name, rgb in SEMANTIC_COLORS.items()
}


@dataclass
class SemanticPrediction:
    """Result of semantic segmentation for one scene/tile.

    probs       [K, H, W] float32 in [0, 1] — softmax class probabilities
    labels      [H, W] uint8 — argmax class IDs (0–5)
    confidence  [H, W] float32 in [0, 1] — max probability per pixel
    """

    probs: np.ndarray       # [K, H, W] float32
    labels: np.ndarray      # [H, W] uint8
    confidence: np.ndarray  # [H, W] float32

    @property
    def num_classes(self) -> int:
        return self.probs.shape[0]

    @property
    def height(self) -> int:
        return self.labels.shape[0]

    @property
    def width(self) -> int:
        return self.labels.shape[1]

    def class_mask(self, class_id: int) -> np.ndarray:
        """Boolean mask for a specific class."""
        return self.labels == class_id

    def class_fraction(self, class_id: int) -> float:
        """Fraction of pixels assigned to a class."""
        return float((self.labels == class_id).mean())

    def mean_confidence(self) -> float:
        return float(self.confidence.mean())

    def low_confidence_fraction(self, threshold: float = 0.7) -> float:
        """Fraction of pixels with confidence below threshold."""
        return float((self.confidence < threshold).mean())

    def validate(self) -> None:
        """Sanity checks on the prediction."""
        assert self.probs.ndim == 3, f"probs must be [K,H,W], got {self.probs.shape}"
        assert self.probs.shape[0] == NUM_PROJECT_CLASSES, (
            f"expected {NUM_PROJECT_CLASSES} classes, got {self.probs.shape[0]}"
        )
        assert self.labels.shape == self.probs.shape[1:], (
            f"labels shape {self.labels.shape} != probs spatial {self.probs.shape[1:]}"
        )
        assert self.confidence.shape == self.labels.shape, (
            f"confidence shape {self.confidence.shape} != labels {self.labels.shape}"
        )
        # Probabilities should approximately sum to 1 (bilinear resize may
        # break the simplex slightly — acceptable for smoothness gating).
        sums = self.probs.sum(axis=0)
        assert np.allclose(sums, 1.0, atol=0.05), (
            f"probs don't sum to 1: min={sums.min():.3f}, max={sums.max():.3f}"
        )
        # Labels must be argmax of probs.
        expected = self.probs.argmax(axis=0).astype(np.uint8)
        assert np.array_equal(self.labels, expected), "labels != argmax(probs)"


def prediction_from_probs(probs: np.ndarray) -> SemanticPrediction:
    """Build a SemanticPrediction from raw softmax probabilities [K,H,W]."""
    probs = np.asarray(probs, dtype=np.float32)
    if probs.ndim != 3 or probs.shape[0] != NUM_PROJECT_CLASSES:
        raise ValueError(
            f"Expected probs shape [{NUM_PROJECT_CLASSES},H,W], got {probs.shape}"
        )
    # Ensure valid probabilities (softmax output should already be, but
    # bilinear resize during tiling may introduce small violations).
    probs = np.clip(probs, 0.0, 1.0)
    # Re-normalize each pixel to ensure the simplex constraint holds.
    sums = probs.sum(axis=0, keepdims=True)
    sums = np.maximum(sums, 1e-8)
    probs = probs / sums

    labels = probs.argmax(axis=0).astype(np.uint8)
    confidence = probs.max(axis=0).astype(np.float32)

    return SemanticPrediction(
        probs=probs,
        labels=labels,
        confidence=confidence,
    )


def render_semantic_map(prediction: SemanticPrediction) -> np.ndarray:
    """Render a class-colored RGBA visualization [H,W,4] uint8.

    Each pixel is colored by its argmax class, with alpha proportional to
    confidence (fully opaque at confidence >= 0.8, 60% at minimum).
    """
    h, w = prediction.labels.shape
    rgba = np.zeros((h, w, 4), dtype=np.uint8)

    for class_id, class_name in enumerate(PROJECT_CLASSES):
        mask = prediction.labels == class_id
        if not mask.any():
            continue
        r, g, b = SEMANTIC_COLORS[class_name]
        rgba[mask, 0] = r
        rgba[mask, 1] = g
        rgba[mask, 2] = b

    # Alpha: scale confidence to [153, 255] (60%–100%)
    alpha = np.clip(prediction.confidence * 255.0, 153.0, 255.0).astype(np.uint8)
    rgba[:, :, 3] = alpha

    return rgba


def semantic_metadata(
    prediction: SemanticPrediction,
    checkpoint_name: str = "unknown",
    model_name: str = "CalibrationNet_aux",
) -> Dict[str, Any]:
    """Scene-level semantic metadata for the API and diagnostics."""
    class_fractions = {}
    for class_id, class_name in enumerate(PROJECT_CLASSES):
        class_fractions[class_name] = round(prediction.class_fraction(class_id), 4)

    return {
        "model": model_name,
        "checkpoint": checkpoint_name,
        "classes": NUM_PROJECT_CLASSES,
        "class_list": list(PROJECT_CLASSES),
        "class_fractions": class_fractions,
        "mean_confidence": round(prediction.mean_confidence(), 4),
        "low_confidence_fraction": round(prediction.low_confidence_fraction(0.7), 4),
        "legend": SEMANTIC_COLORS_HEX,
    }
