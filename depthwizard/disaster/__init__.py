"""DepthWizard disaster assessment sub-package.

Provides ONNX-based building localization (HOTOSM DINOv3) and
earthquake damage assessment (HOTOSM damage model) as optional
scene processing stages.

Architecture:
    onnx_runtime.py     — reusable ONNX inference session wrapper
    building_detector.py — DINOv3 building footprint extraction +
                           raster-to-polygon contour conversion
    damage_assessor.py   — building-level damage classification +
                           destroyed-structure recovery from the damage map
    graph_fp32.py        — internal-fp16 -> fp32 graph restore (GPU fix)
    artifacts.py         — artifact persistence (GeoJSON, NPY, PNG, JSON)
    types.py             — canonical data types (BuildingDetection, DamageAssessment)
    inspect_models.py    — CLI diagnostic for model I/O signatures
"""
