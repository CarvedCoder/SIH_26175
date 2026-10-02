"""Vendored official Depth Anything V2 implementation (DepthWizard TerraHeight).

Source of truth:
    https://github.com/DepthAnything/Depth-Anything-V2 @ main
    (fetched 2026-10-02; files dpt.py, dinov2.py, dinov2_layers/*, util/*)

The TerraHeight-S checkpoint (benfox6515/TerraHeight-S, Apache-2.0) is a
native DepthAnythingV2 .pth in the OFFICIAL implementation's parameter
format (wrapper keys ``net.pretrained.*`` / ``net.depth_head.*``) — it is
NOT loadable into the HuggingFace ``transformers`` DAv2 classes. This
vendored copy is the only implementation that rebuilds it exactly.

Deviations from upstream (documented, behavior-preserving for this repo):
  * ``dpt.py`` / ``util/transform.py``: ``import cv2`` is deferred/optional —
    cv2 is only used by ``DepthAnythingV2.infer_image``/``image2tensor``
    and ``Resize``, none of which DepthWizard's tiled inference path calls.
  * ``util/transform.py``: ``Resize``'s ``image_interpolation_method``
    default is resolved lazily for the same reason.

License: Apache-2.0 (see LICENSE). DINOv2 layer code is
(c) Meta Platforms, Inc. — Apache-2.0 (header comment preserved).
"""
