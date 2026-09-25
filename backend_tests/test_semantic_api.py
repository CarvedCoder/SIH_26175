"""API endpoint tests for semantic segmentation and route-risk layers.

Covers:
- GET /api/v1/scenes/{id}/semantic
- GET /api/v1/scenes/{id}/results/semantic
- GET /api/v1/scenes/{id}/results/semantic-labels
- GET /api/v1/scenes/{id}/results/semantic-confidence
- GET /api/v1/scenes/{id}/results/route-risk
"""

from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import pytest

from depthwizard.pipeline.scene_outputs import write_semantic_outputs


@pytest.fixture()
def semantic_scene(client, processed_scene):
    """Augment a processed scene with synthetic semantic artifacts."""
    scene_id = processed_scene["scene"]["scene_id"]
    from backend.app.core.paths import SCENES_OUTPUT_DIR

    scene_dir = SCENES_OUTPUT_DIR / scene_id

    # Create 256x256 synthetic probabilities (matching the DSM dimensions)
    probs = np.zeros((6, 256, 256), dtype=np.float32)
    probs[2, :128, :] = 0.9   # Road on top
    probs[4, 128:, :] = 0.9   # Ground on bottom
    # Normalize
    sums = probs.sum(axis=0, keepdims=True)
    probs = probs / sums

    write_semantic_outputs(
        out_dir=scene_dir,
        sem_probs=probs,
        checkpoint_name="test_aux.pt",
        model_name="CalibrationNet_aux",
    )

    return {"scene_id": scene_id, "scene_dir": scene_dir, "probs": probs}


def test_semantic_unknown_scene_404(client):
    response = client.get("/api/v1/scenes/scene_000000000000/semantic")
    assert response.status_code == 404


def test_semantic_metadata_unavailable_scene(client, processed_scene):
    scene_id = processed_scene["scene"]["scene_id"]
    response = client.get(f"/api/v1/scenes/{scene_id}/semantic")
    assert response.status_code == 200
    data = response.json()
    assert data["scene_id"] == scene_id
    assert data["available"] is False
    assert data["url"] is None


def test_semantic_metadata_available_scene(client, semantic_scene):
    scene_id = semantic_scene["scene_id"]
    response = client.get(f"/api/v1/scenes/{scene_id}/semantic")
    assert response.status_code == 200
    data = response.json()

    assert data["scene_id"] == scene_id
    assert data["available"] is True
    assert data["url"] == f"/api/v1/scenes/{scene_id}/results/semantic"
    assert data["labels_url"] == f"/api/v1/scenes/{scene_id}/results/semantic-labels"
    assert data["confidence_url"] == f"/api/v1/scenes/{scene_id}/results/semantic-confidence"
    assert data["checkpoint"] == "test_aux.pt"
    assert data["legend"]["road"] == "#9B9B9B"
    assert "class_fractions" in data
    assert data["class_fractions"]["road"] > 0.4


def test_semantic_layer_png(client, semantic_scene):
    scene_id = semantic_scene["scene_id"]
    response = client.get(f"/api/v1/scenes/{scene_id}/results/semantic")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("image/png")
    assert len(response.content) > 100


def test_semantic_labels_npy(client, semantic_scene):
    scene_id = semantic_scene["scene_id"]
    response = client.get(f"/api/v1/scenes/{scene_id}/results/semantic-labels")
    assert response.status_code == 200

    labels = np.load(io.BytesIO(response.content))
    assert labels.dtype == np.uint8
    assert labels.shape == (256, 256)
    assert (labels[:128, :] == 2).all()  # road


def test_semantic_confidence_npy(client, semantic_scene):
    scene_id = semantic_scene["scene_id"]
    response = client.get(f"/api/v1/scenes/{scene_id}/results/semantic-confidence")
    assert response.status_code == 200

    conf = np.load(io.BytesIO(response.content))
    assert conf.dtype == np.float16
    assert conf.shape == (256, 256)
    assert conf.min() > 0.8


def test_semantic_not_available_error(client, processed_scene):
    scene_id = processed_scene["scene"]["scene_id"]
    response = client.get(f"/api/v1/scenes/{scene_id}/results/semantic")
    assert response.status_code == 404
    data = response.json()
    assert data["error"]["code"] == "SEMANTIC_NOT_AVAILABLE"


def test_route_risk_endpoint(client, semantic_scene):
    scene_id = semantic_scene["scene_id"]
    response = client.get(f"/api/v1/scenes/{scene_id}/results/route-risk?vehicle=fire_truck")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("image/png")
    assert len(response.content) > 100
