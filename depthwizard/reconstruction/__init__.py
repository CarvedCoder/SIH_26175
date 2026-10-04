"""Geometry-aware 3D building reconstruction package.

Public API: :mod:`depthwizard.reconstruction.buildings3d`
(reconstruction orchestrator, config, watertight prism extrusion).
"""

from .buildings3d import (
    Building3DConfig,
    fuse_building_candidates,
    extrude_prism,
    reconstruct_buildings_3d,
    regularize_mask,
    render_buildings3d_preview,
)

__all__ = [
    "Building3DConfig",
    "fuse_building_candidates",
    "extrude_prism",
    "reconstruct_buildings_3d",
    "regularize_mask",
    "render_buildings3d_preview",
]
