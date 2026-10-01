"""HOTOSM DINOv3 building footprint detector.

Model signature (verified from local_model.onnx):
    INPUT:  image  float32 [1, 3, 256, 256]
    OUTPUT: logits float32 [1, 3, 256, 256]

Class semantics (verified empirically against aerial imagery — bright
regions of each probability channel align with the corresponding features):
    0 = building footprint
    1 = road / paved surface
    2 = background

Pipeline:
    RGB image → sliding-window 256×256 tiles → model inference →
    stitch probability map → threshold → morphological cleanup →
    connected components → vectorized polygons → BuildingDetection
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np

from .onnx_runtime import OnnxSession, OnnxModelError
from .types import BuildingDetection

# Model constants (from actual ONNX signature inspection)
TILE_SIZE = 256
NUM_CLASSES = 3
BUILDING_CLASS = 0

# ImageNet normalization (standard for vision models)
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


class BuildingDetector:
    """Detects building footprints from RGB imagery using the HOTOSM
    DINOv3 building ONNX model."""

    def __init__(
        self,
        model_path: str | Path,
        *,
        device: str = "auto",
        threshold: float = 0.5,
        tile_size: int = TILE_SIZE,
        tile_stride: int | None = None,
        min_area_px: int = 25,
    ) -> None:
        self.session = OnnxSession(model_path, device=device)
        self.threshold = threshold
        self.tile_size = tile_size
        self.tile_stride = tile_stride or max(tile_size // 2, 128)
        self.min_area_px = min_area_px

        # Validate signature
        sig = self.session.signature
        if len(sig.inputs) != 1 or sig.inputs[0].name != "image":
            raise OnnxModelError(
                f"Building model expected input 'image', got: "
                f"{[s.name for s in sig.inputs]}"
            )

    def preprocess_tile(self, tile_rgb: np.ndarray) -> np.ndarray:
        """Preprocess a [H, W, 3] uint8 tile for the model.

        Returns [1, 3, H, W] float32 (ImageNet-normalized).
        """
        tile = tile_rgb.astype(np.float32) / 255.0
        tile = (tile - IMAGENET_MEAN) / IMAGENET_STD
        tile = tile.transpose(2, 0, 1)  # HWC → CHW
        return tile[np.newaxis, ...]     # add batch dim

    def predict_tile(self, tile_rgb: np.ndarray) -> np.ndarray:
        """Run inference on a single [H, W, 3] uint8 tile.

        Returns [H, W] float32 building probability map.
        """
        h, w = tile_rgb.shape[:2]

        # Pad to tile_size if needed
        pad_h = max(0, self.tile_size - h)
        pad_w = max(0, self.tile_size - w)
        if pad_h > 0 or pad_w > 0:
            tile_rgb = np.pad(
                tile_rgb,
                ((0, pad_h), (0, pad_w), (0, 0)),
                mode="reflect",
            )

        # Resize to model input size if different
        if tile_rgb.shape[0] != self.tile_size or tile_rgb.shape[1] != self.tile_size:
            from PIL import Image
            pil_img = Image.fromarray(tile_rgb)
            pil_img = pil_img.resize(
                (self.tile_size, self.tile_size), Image.BILINEAR
            )
            tile_rgb = np.array(pil_img)

        inp = self.preprocess_tile(tile_rgb)
        outputs = self.session.run({"image": inp})
        logits = outputs["logits"]  # [1, 3, 256, 256]

        # Softmax over class dimension
        logits = logits[0]  # [3, 256, 256]
        exp_logits = np.exp(logits - logits.max(axis=0, keepdims=True))
        probs = exp_logits / exp_logits.sum(axis=0, keepdims=True)

        # Building probability (class 0 = building footprint)
        building_prob = probs[BUILDING_CLASS]
        building_prob = np.clip(building_prob, 0.0, 1.0)

        # Crop back to original size
        building_prob = building_prob[:h, :w]
        return building_prob.astype(np.float32)

    def predict_full(
        self,
        rgb: np.ndarray,
        *,
        should_cancel: Callable[[], bool] | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Run sliding-window building detection on full image.

        Args:
            rgb: [H, W, 3] uint8 image
            should_cancel: optional cooperative cancel hook

        Returns:
            building_mask: [H, W] uint8 (0/1 binary mask)
            building_confidence: [H, W] float32 probability map
        """
        h, w = rgb.shape[:2]
        prob_sum = np.zeros((h, w), dtype=np.float64)
        count = np.zeros((h, w), dtype=np.float64)

        stride = self.tile_stride
        tile_sz = self.tile_size

        # Generate tile positions
        y_positions = list(range(0, max(1, h - tile_sz + 1), stride))
        if y_positions[-1] + tile_sz < h:
            y_positions.append(h - tile_sz)
        x_positions = list(range(0, max(1, w - tile_sz + 1), stride))
        if x_positions[-1] + tile_sz < w:
            x_positions.append(w - tile_sz)

        for y0 in y_positions:
            for x0 in x_positions:
                if should_cancel and should_cancel():
                    raise RuntimeError("Building detection cancelled")

                y1 = min(y0 + tile_sz, h)
                x1 = min(x0 + tile_sz, w)
                tile = rgb[y0:y1, x0:x1]

                prob = self.predict_tile(tile)

                # Resize prob back if tile was smaller
                actual_h, actual_w = y1 - y0, x1 - x0
                if prob.shape[0] != actual_h or prob.shape[1] != actual_w:
                    from PIL import Image
                    prob_img = Image.fromarray(
                        (prob * 255).astype(np.uint8)
                    )
                    prob_img = prob_img.resize(
                        (actual_w, actual_h), Image.BILINEAR
                    )
                    prob = np.array(prob_img).astype(np.float32) / 255.0

                prob_sum[y0:y1, x0:x1] += prob
                count[y0:y1, x0:x1] += 1.0

        # Average overlapping predictions
        count = np.maximum(count, 1.0)
        building_confidence = (prob_sum / count).astype(np.float32)

        # Threshold
        building_mask = (building_confidence >= self.threshold).astype(np.uint8)

        # Morphological cleanup
        building_mask = self._morphological_cleanup(building_mask)

        return building_mask, building_confidence

    @staticmethod
    def _morphological_cleanup(mask: np.ndarray) -> np.ndarray:
        """Remove noise and fill small holes using morphological operations."""
        from scipy import ndimage

        # Remove small isolated regions (vectorized via bincount lookup)
        labeled, n_features = ndimage.label(mask)
        if n_features:
            sizes = np.bincount(labeled.ravel())
            keep = sizes >= 25  # minimum building size in pixels
            keep[0] = False     # background stays background
            mask = keep[labeled].astype(np.uint8)

        # Fill small holes within buildings
        filled = ndimage.binary_fill_holes(mask)
        return filled.astype(np.uint8)

    def detect(
        self,
        rgb: np.ndarray,
        *,
        should_cancel: Callable[[], bool] | None = None,
    ) -> tuple[list[BuildingDetection], np.ndarray, np.ndarray]:
        """Full building detection pipeline.

        Returns:
            buildings: list of BuildingDetection objects
            building_mask: [H, W] uint8 binary mask
            building_confidence: [H, W] float32 probability map
        """
        mask, confidence = self.predict_full(
            rgb, should_cancel=should_cancel
        )

        buildings = _extract_buildings(mask, confidence, min_area=self.min_area_px)

        return buildings, mask, confidence

    def polygonize(
        self, mask: np.ndarray, confidence: np.ndarray
    ) -> list[BuildingDetection]:
        """Convert a building mask to vectorized building polygons."""
        return _extract_buildings(mask, confidence, min_area=self.min_area_px)


def _extract_buildings(
    mask: np.ndarray,
    confidence: np.ndarray,
    min_area: int = 25,
) -> list[BuildingDetection]:
    """Extract individual building detections from a binary mask.

    Uses connected component labeling to identify individual buildings,
    then traces each component's contour for polygon representation.
    """
    from scipy import ndimage

    labeled, n_features = ndimage.label(mask)
    buildings: list[BuildingDetection] = []
    slices = ndimage.find_objects(labeled)

    for i, sl in enumerate(slices, start=1):
        if sl is None:
            continue

        component_crop = labeled[sl] == i
        area_px = float(component_crop.sum())

        if area_px < min_area:
            continue

        y0, x0 = sl[0].start, sl[1].start

        # Centroid (in full-image coordinates)
        ys, xs = np.nonzero(component_crop)
        centroid_y = float(ys.mean()) + y0
        centroid_x = float(xs.mean()) + x0

        # Confidence = max probability within the building footprint
        building_conf = float(confidence[sl][component_crop].max())

        # Extract precise contour polygon from the component mask
        polygon = _mask_to_polygon(component_crop, offset=(x0, y0))

        building_id = f"B{len(buildings) + 1:03d}"

        buildings.append(BuildingDetection(
            building_id=building_id,
            polygon=polygon,
            area_px=area_px,
            area_m2=None,  # set later if georeferenced
            centroid_x=centroid_x,
            centroid_y=centroid_y,
            confidence=building_conf,
            bbox=(int(x0), int(y0), int(sl[1].stop), int(sl[0].stop)),
        ))

    return buildings


def _ring_area(ring: list) -> float:
    """Shoelace area of a coordinate ring."""
    area = 0.0
    n = len(ring)
    for i in range(n):
        x1, y1 = ring[i][0], ring[i][1]
        x2, y2 = ring[(i + 1) % n][0], ring[(i + 1) % n][1]
        area += x1 * y2 - x2 * y1
    return abs(area) / 2.0


def _simplify_ring(
    ring: list[tuple[float, float]], max_points: int = 256
) -> list[tuple[float, float]]:
    """Reduce ring complexity while preserving shape.

    Uses shapely's Douglas-Peucker simplification when available;
    otherwise decimates uniformly. Keeps the ring closed.
    """
    if len(ring) <= max_points:
        return list(ring)
    try:
        from shapely.geometry import LineString

        simplified = LineString(ring).simplify(1.0, preserve_topology=True)
        result = [(float(x), float(y)) for x, y in simplified.coords]
    except ImportError:
        step = max(1, len(ring) // max_points)
        result = list(ring[::step])
    if result and result[0] != result[-1]:
        result.append(result[0])
    return result


def _mask_to_polygon(
    mask: np.ndarray,
    offset: tuple[int, int] = (0, 0),
) -> list[tuple[float, float]]:
    """Convert a binary crop mask to a polygon in full-image pixel coordinates.

    Uses rasterio.features.shapes for accurate contour tracing — the ring
    follows pixel-corner coordinates, so it can be concave and
    non-self-intersecting (unlike angle-sorted boundary points).

    Args:
        mask: boolean mask crop containing a single connected component
        offset: (x, y) origin of the crop within the full image

    Returns a closed exterior ring [(x, y), ...], or [] if empty.
    """
    try:
        from rasterio import features
    except ImportError:
        # Fallback: bounding box
        r, c = np.nonzero(mask)
        if len(r) == 0:
            return []
        y_min, y_max = int(r.min()), int(r.max())
        x_min, x_max = int(c.min()), int(c.max())
        return [
            (float(x_min + offset[0]), float(y_min + offset[1])),
            (float(x_max + offset[0]), float(y_min + offset[1])),
            (float(x_max + offset[0]), float(y_max + offset[1])),
            (float(x_min + offset[0]), float(y_max + offset[1])),
            (float(x_min + offset[0]), float(y_min + offset[1])),
        ]

    best_ring: list | None = None
    best_area = 0.0
    for geom, _val in features.shapes(
        mask.astype(np.uint8), mask=mask.astype(bool), connectivity=4
    ):
        ring = geom["coordinates"][0]
        area = _ring_area(ring)
        if area > best_area:
            best_area = area
            best_ring = ring

    if best_ring is None:
        return []

    ox, oy = offset
    polygon = _simplify_ring([
        (float(x) + ox, float(y) + oy) for x, y in best_ring
    ])
    return polygon
