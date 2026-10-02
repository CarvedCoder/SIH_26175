# Vendored: Depth Anything V2 (official implementation)

- Upstream: https://github.com/DepthAnything/Depth-Anything-V2 (`depth_anything_v2/`)
- Fetched: 2026-10-02 (upstream `main`)
- License: Apache-2.0 (see `LICENSE`); `dinov2_layers/*` and `dinov2.py` are
  (c) Meta Platforms, Inc., Apache-2.0.
- Used by: `depthwizard/terraheight.py` to rebuild the released TerraHeight-S
  checkpoint (https://huggingface.co/benfox6515/TerraHeight-S), which is a
  native DepthAnythingV2 state dict (`net.pretrained.*` + `net.depth_head.*`).
- Local deviations: cv2 import made optional (see package `__init__.py`
  docstring). Everything else is upstream code, unmodified.

To re-vendor at a newer pin: replace `dpt.py`, `dinov2.py`,
`dinov2_layers/`, `util/blocks.py`, `util/transform.py` with the upstream
files, re-apply the cv2 guards above, and update this notice + the pin date.
