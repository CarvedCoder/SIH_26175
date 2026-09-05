"""Live Depth Anything V2 backbone — the cache-miss fallback.

The training pipeline uses a precomputed raw-depth cache (one .npy per tile).
But the webapp must predict on ARBITRARY uploaded images for which no cache
exists. This module wraps the exact same HuggingFace forward pass used to
BUILD the cache, so a live Dn is numerically identical in kind to a cached
one (same model, same 518 bicubic input, same ImageNet normalization, same
bilinear upsample back to native resolution).

Recipe (frozen — must match the ``depth`` command exactly):
    input  : uint8 [H,W,3] -> bicubic resize 518x518 -> /255 -> ImageNet norm
    output : predicted_depth -> bilinear upsample to (H, W), float32, RAW
             (no sigmoid, no min-max — normalization happens ONLY in
              depthwizard.normalize.minmax_normalize at consume time)

Offline honesty: the first call downloads weights from HuggingFace
(ViT-Base ~390 MB, ViT-Large ~1.3 GB). If the host is offline and no cache
exists, predict() raises — we never fabricate a depth substitute.

Default alignment (Phase 0.2): the module default is
``depth-anything/Depth-Anything-V2-Base-hf`` (ViT-B) — the SAME variant the
``depth`` command used to build the training cache. The pre-GAMUS default was
V2-Large-hf, which silently mismatched cached vs live Dn distributions
(risk R9 in the integration plan). Callers can still override per-config.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from PIL import Image

INPUT_SIZE = 518  # 14 * 37 — multiple of the ViT patch size (do not change)

# Same constants the cache builder and dataset use.
_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


class DepthAnythingBackbone:
    """Lazy-loading singleton-style wrapper around the HF depth model."""

    def __init__(self, model_id: str = "depth-anything/Depth-Anything-V2-Base-hf",
                 device: str = "cpu", fp16: bool = False):
        self.model_id = model_id
        self.device = device
        self.fp16 = fp16 and device.startswith("cuda")
        self._model = None

    # ------------------------------------------------------------------
    @property
    def loaded(self) -> bool:
        return self._model is not None

    def load(self):
        """Download + load the model (idempotent)."""
        if self._model is not None:
            return self
        import torch
        from transformers import AutoModelForDepthEstimation

        self._torch = torch
        self._model = AutoModelForDepthEstimation.from_pretrained(self.model_id)
        self._model.to(self.device).eval()
        if self.fp16:
            self._model.half()
        return self

    # ------------------------------------------------------------------
    def preprocess(self, rgb_u8: np.ndarray):
        """uint8 [H,W,3] -> float tensor [1,3,518,518] on self.device."""
        torch = self._torch
        im = Image.fromarray(rgb_u8).resize(
            (INPUT_SIZE, INPUT_SIZE), Image.BICUBIC)
        x = (np.asarray(im, dtype=np.float32) / 255.0 - _MEAN) / _STD
        t = torch.from_numpy(x).permute(2, 0, 1)[None]
        return t.to(self.device)

    def raw_depth(self, rgb_u8: np.ndarray) -> np.ndarray:
        """Raw relative depth [H,W] float32 at the input's native resolution.

        Larger value = closer to sensor (DAv2 convention). NOT metric —
        the calibration net maps it to metres.
        """
        if self._model is None:
            self.load()
        torch = self._torch
        h, w = rgb_u8.shape[:2]
        with torch.inference_mode():
            x = self.preprocess(rgb_u8)
            if self.fp16:
                x = x.half()
            pred = self._model(pixel_values=x).predicted_depth
            pred = pred[0].float()[None, None]          # -> [1,1,h',w']
            pred = torch.nn.functional.interpolate(
                pred, size=(h, w), mode="bilinear", align_corners=False)
        return pred[0, 0].cpu().numpy().astype(np.float32)


# ----------------------------------------------------------------------
# Module-level default instance (models are expensive; share one per process)
# ----------------------------------------------------------------------
_DEFAULT: Optional[DepthAnythingBackbone] = None


def get_backbone(model_id: str = "depth-anything/Depth-Anything-V2-Base-hf",
                 device: str = "cpu") -> DepthAnythingBackbone:
    """Process-wide shared backbone instance (loads on first raw_depth call)."""
    global _DEFAULT
    if _DEFAULT is None or _DEFAULT.model_id != model_id:
        _DEFAULT = DepthAnythingBackbone(model_id=model_id, device=device)
    return _DEFAULT
