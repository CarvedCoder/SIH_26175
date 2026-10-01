"""HOTOSM earthquake damage assessment model.

Model signature (verified from model.onnx):
    INPUT:  post float32 [1, 3, 512, 512]   — post-disaster RGB
    INPUT:  pre  float32 [1, 3, 512, 512]   — pre-disaster RGB
    OUTPUT: damage_logits float32 [1, 4, 512, 512]  — 4 damage classes

Damage classes (index → label):
    0 = no-damage
    1 = minor-damage
    2 = major-damage
    3 = destroyed

POST-ONLY MODE:
    When no pre-disaster image is available, the pre input is filled with
    zeros. This is explicitly a post-only assessment — the API and UI
    report this honestly as "post_only" mode.

    DO NOT generate fake pre-disaster imagery.

BUILDING-LEVEL ASSESSMENT:
    For each detected building:
    1. Crop the post (and optional pre) image around the building bbox
    2. Apply the building footprint as context
    3. Run inference on the 512×512 crop
    4. Aggregate damage prediction within the building footprint
    5. Assign damage class to the building
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Callable, Literal

import numpy as np

from .onnx_runtime import OnnxSession, OnnxModelError
from .types import (
    BuildingDetection,
    DamageAssessment,
    DAMAGE_CLASSES,
)

# Model constants (from actual ONNX signature)
DAMAGE_TILE_SIZE = 512
NUM_DAMAGE_CLASSES = 4

# ImageNet normalization
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# Context padding ratio for building crops
CONTEXT_PADDING_RATIO = 0.25
MIN_CROP_SIZE = 64


class DamageAssessor:
    """Assesses building-level damage using the HOTOSM damage model."""

    def __init__(
        self,
        model_path: str | Path,
        *,
        device: str = "auto",
        batch_size: int = 1,
        review_threshold: float = 0.4,
    ) -> None:
        self.session = OnnxSession(model_path, device=device)
        self.batch_size = batch_size
        self.review_threshold = review_threshold

        # Validate signature
        sig = self.session.signature
        input_names = {s.name for s in sig.inputs}
        if "post" not in input_names or "pre" not in input_names:
            raise OnnxModelError(
                f"Damage model expected inputs 'post' and 'pre', got: "
                f"{[s.name for s in sig.inputs]}"
            )
        output_names = {s.name for s in sig.outputs}
        if "damage_logits" not in output_names:
            raise OnnxModelError(
                f"Damage model expected output 'damage_logits', got: "
                f"{[s.name for s in sig.outputs]}"
            )

    def preprocess(self, tile_rgb: np.ndarray) -> np.ndarray:
        """Preprocess a [H, W, 3] uint8 tile for the damage model.

        Returns [1, 3, 512, 512] float32 (ImageNet-normalized).
        """
        from PIL import Image

        # Resize to model input size
        pil_img = Image.fromarray(tile_rgb)
        pil_img = pil_img.resize(
            (DAMAGE_TILE_SIZE, DAMAGE_TILE_SIZE), Image.BILINEAR
        )
        tile = np.array(pil_img, dtype=np.float32) / 255.0
        tile = (tile - IMAGENET_MEAN) / IMAGENET_STD
        tile = tile.transpose(2, 0, 1)  # HWC → CHW
        return tile[np.newaxis, ...]     # add batch dim

    def predict_building(
        self,
        post_rgb: np.ndarray,
        building: BuildingDetection,
        *,
        pre_rgb: np.ndarray | None = None,
        building_mask: np.ndarray | None = None,
    ) -> DamageAssessment:
        """Assess damage for a single building.

        Args:
            post_rgb: full post-disaster image [H, W, 3] uint8
            building: detected building with bbox
            pre_rgb: optional pre-disaster image [H, W, 3] uint8
            building_mask: optional full-scene building mask [H, W]

        Returns:
            DamageAssessment for this building
        """
        h, w = post_rgb.shape[:2]
        mode: Literal["post_only", "pre_post"] = (
            "pre_post" if pre_rgb is not None else "post_only"
        )

        # Extract building crop with context padding
        bbox = building.bbox
        if bbox is None:
            # Fallback: compute from centroid
            cx, cy = int(building.centroid_x), int(building.centroid_y)
            half = DAMAGE_TILE_SIZE // 4
            bbox = (
                max(0, cx - half),
                max(0, cy - half),
                min(w, cx + half),
                min(h, cy + half),
            )

        x_min, y_min, x_max, y_max = bbox
        bw, bh = x_max - x_min, y_max - y_min

        # Add context padding
        pad_x = max(int(bw * CONTEXT_PADDING_RATIO), MIN_CROP_SIZE // 4)
        pad_y = max(int(bh * CONTEXT_PADDING_RATIO), MIN_CROP_SIZE // 4)

        crop_x_min = max(0, x_min - pad_x)
        crop_y_min = max(0, y_min - pad_y)
        crop_x_max = min(w, x_max + pad_x)
        crop_y_max = min(h, y_max + pad_y)

        # Ensure minimum crop size
        crop_w = crop_x_max - crop_x_min
        crop_h = crop_y_max - crop_y_min
        if crop_w < MIN_CROP_SIZE:
            expand = (MIN_CROP_SIZE - crop_w) // 2 + 1
            crop_x_min = max(0, crop_x_min - expand)
            crop_x_max = min(w, crop_x_max + expand)
        if crop_h < MIN_CROP_SIZE:
            expand = (MIN_CROP_SIZE - crop_h) // 2 + 1
            crop_y_min = max(0, crop_y_min - expand)
            crop_y_max = min(h, crop_y_max + expand)

        # Extract crops
        post_crop = post_rgb[crop_y_min:crop_y_max, crop_x_min:crop_x_max]

        if pre_rgb is not None:
            pre_crop = pre_rgb[crop_y_min:crop_y_max, crop_x_min:crop_x_max]
        else:
            # Post-only: use zeros for pre-disaster input
            pre_crop = np.zeros_like(post_crop)

        # Preprocess
        post_tensor = self.preprocess(post_crop)
        pre_tensor = self.preprocess(pre_crop)

        # Run inference
        outputs = self.session.run({
            "post": post_tensor,
            "pre": pre_tensor,
        })
        logits = outputs["damage_logits"]  # [1, 4, 512, 512]
        logits = logits[0]  # [4, 512, 512]

        # Softmax
        exp_logits = np.exp(logits - logits.max(axis=0, keepdims=True))
        probs = exp_logits / exp_logits.sum(axis=0, keepdims=True)

        # If we have a building mask, aggregate within the footprint
        if building_mask is not None:
            bm_crop = building_mask[crop_y_min:crop_y_max, crop_x_min:crop_x_max]
            # Resize mask to model output size
            from PIL import Image
            bm_pil = Image.fromarray(bm_crop * 255)
            bm_pil = bm_pil.resize(
                (DAMAGE_TILE_SIZE, DAMAGE_TILE_SIZE), Image.NEAREST
            )
            bm_resized = np.array(bm_pil) > 127

            if bm_resized.any():
                # Average probabilities within building footprint
                class_probs = np.array([
                    float(probs[c][bm_resized].mean())
                    for c in range(NUM_DAMAGE_CLASSES)
                ])
            else:
                # Fallback: use center region
                center = DAMAGE_TILE_SIZE // 4
                region = probs[
                    :,
                    center:DAMAGE_TILE_SIZE - center,
                    center:DAMAGE_TILE_SIZE - center,
                ]
                class_probs = region.mean(axis=(1, 2))
        else:
            # Use center region as building approximate
            center = DAMAGE_TILE_SIZE // 4
            region = probs[
                :,
                center:DAMAGE_TILE_SIZE - center,
                center:DAMAGE_TILE_SIZE - center,
            ]
            class_probs = region.mean(axis=(1, 2))

        # Normalize
        class_probs = class_probs / (class_probs.sum() + 1e-8)

        # Determine damage class
        predicted_class_idx = int(np.argmax(class_probs))
        damage_class = DAMAGE_CLASSES[predicted_class_idx]
        confidence = float(class_probs[predicted_class_idx])

        # Review recommendation: if confidence is below threshold or
        # the top two classes are close
        sorted_probs = np.sort(class_probs)[::-1]
        margin = sorted_probs[0] - sorted_probs[1] if len(sorted_probs) > 1 else 1.0
        review_required = bool(
            confidence < self.review_threshold or margin < 0.15
        )

        probabilities = {
            cls: float(class_probs[i])
            for i, cls in enumerate(DAMAGE_CLASSES)
        }

        return DamageAssessment(
            building_id=building.building_id,
            damage_class=damage_class,
            confidence=confidence,
            probabilities=probabilities,
            review_required=review_required,
            mode=mode,
        )

    def assess_buildings(
        self,
        post_rgb: np.ndarray,
        buildings: list[BuildingDetection],
        *,
        pre_rgb: np.ndarray | None = None,
        building_mask: np.ndarray | None = None,
        should_cancel: Callable[[], bool] | None = None,
    ) -> list[DamageAssessment]:
        """Assess damage for all detected buildings.

        Processes buildings sequentially (batch_size=1 default for
        memory safety on 6GB GPU). Each building gets its own
        crop-and-predict cycle.
        """
        assessments: list[DamageAssessment] = []

        for i, building in enumerate(buildings):
            if should_cancel and should_cancel():
                raise RuntimeError("Damage assessment cancelled")

            assessment = self.predict_building(
                post_rgb,
                building,
                pre_rgb=pre_rgb,
                building_mask=building_mask,
            )
            assessments.append(assessment)

        return assessments

    def map_damage_probability(
        self,
        post_rgb: np.ndarray,
        *,
        pre_rgb: np.ndarray | None = None,
        should_cancel: Callable[[], bool] | None = None,
    ) -> np.ndarray:
        """Run the damage model tiled over the FULL scene.

        The building model cannot see destroyed/rubble structures (it is
        trained on intact footprints), so structures that are completely
        destroyed never receive a footprint. This pass produces a
        per-pixel damage probability map for the whole scene, which the
        recovery step uses to find destroyed structures the detector
        missed.

        Returns:
            probs: [4, H, W] float32 per-class damage probabilities,
                   averaged over overlapping tiles.
        """
        h, w = post_rgb.shape[:2]
        tile = DAMAGE_TILE_SIZE
        stride = tile // 2

        prob_sum = np.zeros((NUM_DAMAGE_CLASSES, h, w), dtype=np.float64)
        count = np.zeros((h, w), dtype=np.float64)

        y_positions = list(range(0, max(1, h - tile + 1), stride))
        if y_positions[-1] + tile < h:
            y_positions.append(h - tile)
        x_positions = list(range(0, max(1, w - tile + 1), stride))
        if x_positions[-1] + tile < w:
            x_positions.append(w - tile)

        pre_full = pre_rgb if pre_rgb is not None else None

        for y0 in y_positions:
            for x0 in x_positions:
                if should_cancel and should_cancel():
                    raise RuntimeError("Damage mapping cancelled")
                y1, x1 = min(y0 + tile, h), min(x0 + tile, w)
                post_crop = post_rgb[y0:y1, x0:x1]
                pre_crop = (
                    pre_full[y0:y1, x0:x1]
                    if pre_full is not None
                    else np.zeros_like(post_crop)
                )
                probs = self._predict_probs(post_crop, pre_crop)  # [4, th, tw]
                prob_sum[:, y0:y1, x0:x1] += probs
                count[y0:y1, x0:x1] += 1.0

        return (prob_sum / np.maximum(count, 1.0)[None]).astype(np.float32)

    def _predict_probs(
        self, post_crop: np.ndarray, pre_crop: np.ndarray
    ) -> np.ndarray:
        """Run one forward pass on a crop and return [4, H, W] probs."""
        post_tensor = self.preprocess(post_crop)
        pre_tensor = self.preprocess(pre_crop)
        outputs = self.session.run({"post": post_tensor, "pre": pre_tensor})
        logits = outputs["damage_logits"][0]  # [4, 512, 512]
        exp_logits = np.exp(logits - logits.max(axis=0, keepdims=True))
        probs = exp_logits / exp_logits.sum(axis=0, keepdims=True)
        th, tw = post_crop.shape[:2]
        return probs[:, :th, :tw]

    def create_damage_rasters(
        self,
        assessments: list[DamageAssessment],
        buildings: list[BuildingDetection],
        building_mask: np.ndarray,
        height: int,
        width: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Create damage label and confidence rasters from assessments.

        Returns:
            damage_labels: [H, W] uint8 — 0=background, 1-4=damage classes
            damage_confidence: [H, W] float32 — per-pixel confidence
        """
        from scipy import ndimage

        damage_labels = np.zeros((height, width), dtype=np.uint8)
        damage_confidence = np.zeros((height, width), dtype=np.float32)

        # Build assessment lookup
        assessment_map = {a.building_id: a for a in assessments}

        # Label connected components in building mask
        labeled, n_features = ndimage.label(building_mask)

        # Match buildings to labeled components by centroid proximity
        for building in buildings:
            assessment = assessment_map.get(building.building_id)
            if assessment is None:
                continue

            cy, cx = int(building.centroid_y), int(building.centroid_x)
            cy = min(max(cy, 0), height - 1)
            cx = min(max(cx, 0), width - 1)

            component_id = labeled[cy, cx]
            if component_id == 0:
                # Try nearby pixels
                for dy in range(-3, 4):
                    for dx in range(-3, 4):
                        ny, nx = cy + dy, cx + dx
                        if 0 <= ny < height and 0 <= nx < width:
                            if labeled[ny, nx] > 0:
                                component_id = labeled[ny, nx]
                                break
                    if component_id > 0:
                        break

            if component_id > 0:
                component_mask = labeled == component_id
                # Damage class index (1-based: 1=no-damage, 4=destroyed)
                cls_idx = DAMAGE_CLASSES.index(assessment.damage_class) + 1
                damage_labels[component_mask] = cls_idx
                if assessment.confidence is not None:
                    damage_confidence[component_mask] = assessment.confidence

        return damage_labels, damage_confidence


# ── Destroyed-structure recovery ─────────────────────────────────────
# The building localization model is trained on intact footprints and
# produces near-zero probability on rubble (verified: max 0.17 over a
# debris field vs 0.86 on intact roofs). Completely destroyed structures
# therefore never get a footprint. The damage model DOES fire strongly
# on rubble (destroyed-class mean 0.80 over debris), so structures the
# detector missed can be recovered from its per-pixel output.

# Per-pixel destroyed probability for a region to count as destroyed area.
# Tuned on xBD-style tornado imagery: below ~0.75 the smooth probability
# field spills over streets and yards between destroyed blocks.
DESTROYED_PROB_THRESHOLD = 0.75
# Probability required at a split seed (local maximum) inside a hotspot
DESTROYED_SEED_PROBABILITY = 0.85
# Local-maximum filter size for splitting merged debris bands into
# structure-sized regions (px; typical building pitch in these scenes
# is 40-80 px)
RECOVERY_SEED_SPACING = 25
# Smallest region considered a structure remains
RECOVERY_MIN_AREA = 400


def recover_destroyed_structures(
    damage_probs: np.ndarray,
    building_mask: np.ndarray,
    *,
    prob_threshold: float = DESTROYED_PROB_THRESHOLD,
    seed_probability: float = DESTROYED_SEED_PROBABILITY,
    seed_spacing: int = RECOVERY_SEED_SPACING,
    min_area: int = RECOVERY_MIN_AREA,
) -> list[tuple[BuildingDetection, np.ndarray]]:
    """Recover destroyed structures missed by the building detector.

    Takes the full-scene damage probability map, finds regions where the
    model predicts 'destroyed' OUTSIDE the detected building footprints,
    and splits them into structure-sized polygons at the local maxima of
    the destroyed probability.

    The resulting footprints come from the damage model, not from
    geometric building detection: each polygon may merge several
    collapsed structures, so callers must mark them for review.

    Args:
        damage_probs: [4, H, W] per-class damage probabilities
        building_mask: [H, W] uint8 detected building footprints
        prob_threshold: minimum destroyed probability for a hotspot
        seed_probability: minimum destroyed probability at a split seed
        seed_spacing: local-maximum filter size for splitting (px)
        min_area: smallest recovered region (px)

    Returns:
        list of (BuildingDetection, class_probabilities, region_mask)
        triples, sorted top-to-bottom/left-to-right for deterministic
        IDs. Probabilities are the per-class means over the region
        footprint; region_mask is the [H, W] boolean footprint.
    """
    from scipy import ndimage
    from .building_detector import _mask_to_polygon

    destroyed = damage_probs[3]
    h, w = destroyed.shape

    # Hotspot: model says destroyed, outside already-detected footprints
    hot = (destroyed >= prob_threshold) & (building_mask == 0)
    if not hot.any():
        return []

    # Clean speckle. NOTE: no binary_fill_holes here — the holes between
    # destroyed blocks are streets and yards, not destroyed structures;
    # filling them paints whole street grids as damage.
    hot = ndimage.binary_opening(hot, iterations=2)
    if not hot.any():
        return []

    # Split merged debris bands: seeds at well-separated local maxima
    local_max = ndimage.maximum_filter(destroyed, size=seed_spacing)
    seeds = (destroyed >= seed_probability) & (destroyed == local_max) & hot
    seed_labels, n_seeds = ndimage.label(seeds)
    if n_seeds == 0:
        return []

    # Partition each hotspot pixel between its nearest seed (scipy-only
    # watershed substitute via distance transform nearest-seed indices)
    _, (iy, ix) = ndimage.distance_transform_edt(
        seed_labels == 0, return_indices=True
    )
    nearest_seed = seed_labels[iy, ix]
    regions = np.where(hot, nearest_seed, 0)

    results: list[tuple[BuildingDetection, np.ndarray, np.ndarray]] = []
    for region_id in range(1, n_seeds + 1):
        region_mask = regions == region_id
        area_px = int(region_mask.sum())
        if area_px < min_area:
            continue

        ys, xs = np.nonzero(region_mask)
        y0, y1 = int(ys.min()), int(ys.max()) + 1
        x0, x1 = int(xs.min()), int(xs.max()) + 1

        # Class probabilities aggregated over the region footprint
        class_probs = damage_probs[:, region_mask].mean(axis=1)
        class_probs = class_probs / (class_probs.sum() + 1e-8)

        building = BuildingDetection(
            building_id="",  # assigned by the pipeline (continues sequence)
            polygon=_mask_to_polygon(region_mask[y0:y1, x0:x1], offset=(x0, y0)),
            area_px=float(area_px),
            area_m2=None,
            centroid_x=float(xs.mean()),
            centroid_y=float(ys.mean()),
            confidence=float(class_probs[3]),  # destroyed probability
            bbox=(x0, y0, x1, y1),
            source="damage_map",
        )
        results.append((building, class_probs, region_mask))

    # Deterministic ordering for stable IDs: top-to-bottom, then left
    results.sort(key=lambda item: (item[0].centroid_y, item[0].centroid_x))
    return results
