"""Disaster assessment pipeline orchestrator.

Coordinates building detection → damage assessment → artifact writing
as an optional stage in the existing scene processing pipeline.

This module is the ONLY entry point for disaster processing — both the
web service and CLI use this function. It handles:
    * Model resolution from settings
    * Graceful degradation when models are unavailable
    * Partial failure reporting
    * GPU memory management (sequential, not concurrent)
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Callable

import numpy as np

from .types import DisasterResult, DAMAGE_CLASSES


class DisasterPipelineError(Exception):
    """Non-fatal disaster pipeline error — main processing continues."""


def run_disaster_pipeline(
    rgb: np.ndarray,
    output_dir: Path,
    *,
    pre_rgb: np.ndarray | None = None,
    building_model_path: str | None = None,
    damage_model_path: str | None = None,
    device: str = "auto",
    threshold: float = 0.5,
    tile_size: int = 256,
    tile_stride: int | None = None,
    batch_size: int = 1,
    georeferenced: bool = False,
    crs_string: str | None = None,
    transform: Any = None,
    should_cancel: Callable[[], bool] | None = None,
    building_detection_enabled: bool = True,
    damage_enabled: bool = True,
    recover_destroyed: bool = True,
) -> dict[str, Any]:
    """Run the complete disaster assessment pipeline.

    This function is designed to be called from process_scene() after
    depth/semantic inference, and to NEVER crash the main pipeline.
    All errors are caught and reported as partial failures.

    Returns a dict with:
        buildings_available: bool
        damage_available: bool
        building_count: int
        damage_counts: dict
        mode: str
        errors: list[str]
        artifacts: dict[str, str]
    """
    t0 = time.perf_counter()
    result = DisasterResult(
        mode="pre_post" if pre_rgb is not None else "post_only",
        georeferenced=georeferenced,
    )
    errors: list[str] = []
    artifacts: dict[str, str] = {}
    buildings_available = False
    damage_available = False
    recovered_count = 0

    # ── Stage 1: Building detection ──────────────────────────────────
    building_mask = None
    building_confidence = None

    if building_detection_enabled and building_model_path:
        try:
            from .building_detector import BuildingDetector

            print("[disaster] Loading building detection model...")
            detector = BuildingDetector(
                building_model_path,
                device=device,
                threshold=threshold,
                tile_size=tile_size,
                tile_stride=tile_stride,
            )
            result.building_model = str(
                Path(building_model_path).name
            )

            print("[disaster] Running building detection...")
            buildings, building_mask, building_confidence = detector.detect(
                rgb, should_cancel=should_cancel,
            )
            result.buildings = buildings
            result.building_mask = building_mask
            result.building_confidence = building_confidence

            print(
                f"[disaster] Detected {len(buildings)} buildings "
                f"({time.perf_counter() - t0:.1f}s)"
            )

            # Compute area_m2 if georeferenced
            if georeferenced and transform is not None:
                _compute_geo_areas(buildings, transform)

            # Write building artifacts
            from .artifacts import write_building_artifacts

            bld_artifacts = write_building_artifacts(
                output_dir,
                buildings,
                building_mask,
                building_confidence,
                rgb,
                georeferenced=georeferenced,
                crs_string=crs_string,
                transform=transform,
            )
            artifacts.update(bld_artifacts)
            buildings_available = True

            # Release detector GPU memory
            del detector

        except Exception as exc:
            error_msg = f"Building detection failed: {exc}"
            print(f"[disaster] {error_msg}")
            errors.append(error_msg)

    elif building_detection_enabled and not building_model_path:
        errors.append(
            "Building detection enabled but model path not configured "
            "(set DW_BUILDING_MODEL_ONNX)"
        )

    # ── Stage 2: Damage assessment ───────────────────────────────────
    if (
        damage_enabled
        and damage_model_path
        and buildings_available
        and result.buildings
        and building_mask is not None
    ):
        try:
            t1 = time.perf_counter()
            from .damage_assessor import DamageAssessor

            print("[disaster] Loading damage assessment model...")
            assessor = DamageAssessor(
                damage_model_path,
                device=device,
                batch_size=batch_size,
            )
            result.damage_model = str(
                Path(damage_model_path).name
            )

            print(
                f"[disaster] Assessing damage for "
                f"{len(result.buildings)} buildings "
                f"(mode={result.mode})..."
            )
            assessments = assessor.assess_buildings(
                rgb,
                result.buildings,
                pre_rgb=pre_rgb,
                building_mask=building_mask,
                should_cancel=should_cancel,
            )
            result.damages = assessments

            # ── Recover destroyed structures the detector missed ─────
            # The building model cannot see rubble (trained on intact
            # footprints), so completely destroyed structures would be
            # invisible. The damage model's own per-pixel 'destroyed'
            # output fires strongly on debris — use it to recover them.
            combined_mask = building_mask
            recovered_count = 0
            if recover_destroyed:
                try:
                    from .damage_assessor import (
                        recover_destroyed_structures,
                    )
                    from .types import DamageAssessment

                    print("[disaster] Mapping scene-wide damage for "
                          "destroyed-structure recovery...")
                    damage_probs = assessor.map_damage_probability(
                        rgb,
                        pre_rgb=pre_rgb,
                        should_cancel=should_cancel,
                    )
                    recovered = recover_destroyed_structures(
                        damage_probs, building_mask
                    )
                    rec_mask = np.zeros_like(building_mask, dtype=np.uint8)
                    for building, class_probs, region_mask in recovered:
                        building.building_id = (
                            f"B{len(result.buildings) + 1:03d}"
                        )
                        result.buildings.append(building)
                        rec_mask[region_mask] = 1
                        top = int(class_probs.argmax())
                        result.damages.append(DamageAssessment(
                            building_id=building.building_id,
                            damage_class=DAMAGE_CLASSES[top],
                            confidence=float(class_probs[top]),
                            probabilities={
                                cls: float(class_probs[i])
                                for i, cls in enumerate(DAMAGE_CLASSES)
                            },
                            review_required=True,  # no geometric footprint
                            mode=result.mode,
                        ))
                    recovered_count = len(recovered)
                    if recovered_count:
                        combined_mask = np.where(
                            (building_mask > 0) | (rec_mask > 0), 1, 0
                        ).astype(np.uint8)
                        print(
                            f"[disaster] Recovered {recovered_count} "
                            "destroyed structure area(s) from damage map"
                        )
                except Exception as exc:
                    error_msg = f"Destroyed-structure recovery failed: {exc}"
                    print(f"[disaster] {error_msg}")
                    errors.append(error_msg)

            # Create damage rasters
            damage_labels, damage_confidence = assessor.create_damage_rasters(
                assessments,
                result.buildings,
                combined_mask,
                rgb.shape[0],
                rgb.shape[1],
            )
            result.damage_labels = damage_labels
            result.damage_confidence_map = damage_confidence

            print(
                f"[disaster] Damage assessment complete "
                f"({time.perf_counter() - t1:.1f}s)"
            )
            for cls in DAMAGE_CLASSES:
                count = result.damage_counts[cls]
                print(f"[disaster]   {cls}: {count}")

            # Write damage artifacts
            from .artifacts import write_damage_artifacts

            dmg_artifacts = write_damage_artifacts(
                output_dir,
                result,
                rgb,
                combined_mask,
                damage_labels,
                damage_confidence,
                georeferenced=georeferenced,
                crs_string=crs_string,
                transform=transform,
            )
            artifacts.update(dmg_artifacts)
            damage_available = True

            # Release assessor GPU memory
            del assessor

        except Exception as exc:
            error_msg = f"Damage assessment failed: {exc}"
            print(f"[disaster] {error_msg}")
            errors.append(error_msg)

    elif damage_enabled and not damage_model_path:
        errors.append(
            "Damage assessment enabled but model path not configured "
            "(set DW_DAMAGE_MODEL_ONNX)"
        )
    elif damage_enabled and not buildings_available:
        errors.append(
            "Damage assessment skipped: no buildings detected"
        )

    elapsed = time.perf_counter() - t0
    print(f"[disaster] Pipeline complete ({elapsed:.1f}s)")

    return {
        "buildings_available": buildings_available,
        "damage_available": damage_available,
        "building_count": result.building_count,
        "damage_counts": result.damage_counts,
        "mode": result.mode,
        "review_count": result.review_count,
        "mean_confidence": result.mean_confidence,
        "recovered_destroyed_areas": recovered_count,
        "georeferenced": georeferenced,
        "errors": errors,
        "artifacts": artifacts,
        "elapsed_sec": elapsed,
    }


def _compute_geo_areas(
    buildings: list,
    transform: Any,
) -> None:
    """Compute area_m2 for buildings using the raster's geotransform."""
    # Approximate pixel area in m² from the affine transform
    # |a| * |e| gives the pixel area (ignoring rotation/skew)
    pixel_area_m2 = abs(transform.a * transform.e)

    for b in buildings:
        b.area_m2 = b.area_px * pixel_area_m2
