"""depthwizard.postprocess — research-grade AGL refinement (post-CalibrationNet).

Public API:
    PostProcessConfig        every tunable (config.py — single source)
    PRESETS / config_from_preset
    refine_agl               the orchestrator (pure, deterministic)
    PostProcessReport        audit record for one run
    postprocess_tile_metrics boundary-aware evaluation metrics

Pipeline contract (see refinement.py): refinement runs on calibrated AGL
in metres, BEFORE DEM anchoring; it never rescales the signal; NaN pixels
are preserved; ``enabled=False`` is an exact passthrough.
"""

from .config import METHODS, PRESETS, PostProcessConfig, config_from_preset
from .refinement import PostProcessReport, calibration_report, refine_agl

__all__ = [
    "METHODS",
    "PRESETS",
    "PostProcessConfig",
    "PostProcessReport",
    "calibration_report",
    "config_from_preset",
    "refine_agl",
]
