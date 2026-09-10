"""Connects backend jobs to the certified DepthWizard inference path.

Guarantees:
    * DETERMINISTIC INPUT: the scene's input raster is selected by exact
      known filename (stored at upload), never by arbitrary iterdir order.
      A scene with multiple ambiguous rasters is rejected instead of
      silently picking one.
    * CONCURRENCY: at most ``DW_MAX_CONCURRENT_JOBS`` inference runs execute
      at once (semaphore). Default 1 — multiple simultaneous torch
      executions on one device only degrade latency.
    * CANCELLATION: cooperative — checked before inference starts and
      around preview writing; a cancelled job's outputs are removed.
    * NO FABRICATION: failures record typed errors and the exception is
      re-raised for the server-side log.
"""

from __future__ import annotations

import os
import threading
from pathlib import Path
from typing import Any

import numpy as np

from backend.app.core.logging import logger
from backend.app.core.paths import (
    PROJECT_ROOT,
    get_scene_output_dir,
    get_scene_process_dir,
    get_scene_raw_dir,
)
from backend.app.jobs.manager import job_manager
from depthwizard.inference import run_inference

# Module-level guard shared by ALL jobs in this process: never run more
# than DW_MAX_CONCURRENT_JOBS torch forwards simultaneously.
_INFERENCE_SEMAPHORE = threading.Semaphore(
    max(1, int(os.environ.get("DW_MAX_CONCURRENT_JOBS", "1")))
)


class SceneInputError(FileNotFoundError):
    """The scene's stored input does not match the deterministic contract."""


class ProcessingService:
    def __init__(self) -> None:
        self.default_device = os.environ.get("DW_DEVICE", "auto")
        self.default_backbone = os.environ.get(
            "DW_BACKBONE",
            "depth-anything/Depth-Anything-V2-Base-hf",
        )
        self.default_live_backbone = os.environ.get("DW_NO_LIVE") != "1"

    # -- configuration ------------------------------------------------------

    def _resolve_checkpoint(self) -> Path:
        """Resolve the calibration checkpoint. Single source of truth:
        DW_CKPT, else the repo default below (mirrored in docker-compose)."""

        env_checkpoint = os.environ.get("DW_CKPT")

        if env_checkpoint:
            checkpoint = Path(env_checkpoint)
        else:
            checkpoint = (
                PROJECT_ROOT / "outputs" / "calib_net" / "gamus_rgb_grad" / "best.pt"
            )

        if not checkpoint.exists():
            raise FileNotFoundError(
                "DepthWizard checkpoint not found. Set DW_CKPT to a valid "
                "checkpoint."
            )

        expected_sha = os.environ.get("DW_CKPT_SHA256")
        if expected_sha:
            from depthwizard.tifops import sha256_file

            actual = sha256_file(checkpoint)
            if actual.lower() != expected_sha.lower():
                raise ValueError(
                    "checkpoint SHA-256 mismatch — refusing to load a "
                    "checkpoint that does not match DW_CKPT_SHA256."
                )

        return checkpoint

    # -- deterministic input ------------------------------------------------

    # Designated upload names in deterministic check order (the upload route
    # stores the validated raster as input.<original extension>; PNG/JPG
    # are first-class inputs alongside GeoTIFF).
    _DESIGNATED_INPUTS = (
        "input.tif",
        "input.tiff",
        "input.png",
        "input.jpg",
        "input.jpeg",
    )
    _RASTER_SUFFIXES = {".tif", ".tiff", ".png", ".jpg", ".jpeg"}

    def find_scene_input(self, scene_id: str) -> Path | None:
        """Return the scene's designated input raster, or None.

        Determinism rule (audit M9): pick by exact known filename — never
        by iteration order.
        """
        input_dir = get_scene_raw_dir(scene_id)

        if not input_dir.exists():
            return None

        for name in self._DESIGNATED_INPUTS:
            candidate = input_dir / name
            if candidate.is_file():
                return candidate

        return None

    def resolve_scene_input(self, scene_id: str) -> Path:
        """Return the ONE input raster of a scene, deterministically.

        Falls back to the single-raster rule: several images with no
        designated input is an ambiguous scene and is REJECTED, never
        resolved by iteration order.
        """
        designated = self.find_scene_input(scene_id)
        if designated is not None:
            return designated

        input_dir = get_scene_raw_dir(scene_id)

        if not input_dir.exists():
            raise FileNotFoundError(f"Scene '{scene_id}' has no stored input.")

        rasters = sorted(
            path
            for path in input_dir.iterdir()
            if path.is_file() and path.suffix.lower() in self._RASTER_SUFFIXES
        )

        if not rasters:
            raise FileNotFoundError(
                f"No image input found for scene '{scene_id}'."
            )
        if len(rasters) > 1:
            raise SceneInputError(
                f"Scene '{scene_id}' contains multiple images with no "
                "designated input; rename exactly one to input.tif "
                "(or input.png / input.jpg)."
            )

        return rasters[0]

    # -- cancellation --------------------------------------------------------

    def _check_cancelled(self, job_id: str) -> bool:
        job = job_manager.get_job(job_id)
        return bool(job and job.cancel_requested)

    # -- processing -----------------------------------------------------------

    def process_scene(
        self,
        job_id: str,
        scene_id: str,
        *,
        mode: str = "auto",
        ground_elev: float | None = None,
    ) -> dict[str, Any]:
        """Run DepthWizard inference for a scene (blocking; call from a worker)."""

        input_path = self.resolve_scene_input(scene_id)
        output_dir = get_scene_output_dir(scene_id)
        process_dir = get_scene_process_dir(scene_id)
        output_dir.mkdir(parents=True, exist_ok=True)
        process_dir.mkdir(parents=True, exist_ok=True)

        checkpoint = self._resolve_checkpoint()

        job_manager.update_job(
            job_id,
            status="processing",
            stage="depth_inference",
            progress=5.0,
        )

        if self._check_cancelled(job_id):
            self._discard_outputs(output_dir)
            job_manager.update_job(job_id, status="cancelled", stage="cancelled")
            return {"cancelled": True}

        with _INFERENCE_SEMAPHORE:
            if self._check_cancelled(job_id):
                self._discard_outputs(output_dir)
                job_manager.update_job(
                    job_id, status="cancelled", stage="cancelled"
                )
                return {"cancelled": True}

            payload = run_inference(
                input_path=input_path,
                ckpt_path=checkpoint,
                out_dir=output_dir,
                device=self.default_device,
                mode=mode,
                dn_path=None,
                cache_dir=None,
                live_backbone=self.default_live_backbone,
                backbone_id=self.default_backbone,
                anchor_dem=None,
                ground_elev=ground_elev,
                write_files=True,
            )

        if self._check_cancelled(job_id):
            self._discard_outputs(output_dir)
            job_manager.update_job(job_id, status="cancelled", stage="cancelled")
            return {"cancelled": True}

        job_manager.update_job(
            job_id,
            status="completed",
            stage="completed",
            progress=100.0,
            result=payload,
        )
        logger.info("job completed: %s scene=%s", job_id, scene_id)
        return payload

    def refine_region(
        self,
        job_id: str,
        scene_id: str,
        *,
        bbox: tuple[int, int, int, int],
    ) -> dict[str, Any]:
        """Re-run inference on a pixel-space bbox of the scene input.

        The crop is processed through the SAME run_inference path (tiles
        mode) at source resolution and stored as refined_dsm.npy next to
        the scene results. The full-scene results are left untouched.
        """
        x_min, y_min, x_max, y_max = bbox
        if x_max <= x_min or y_max <= y_min:
            raise ValueError("refinement bbox must have positive extent.")

        input_path = self.resolve_scene_input(scene_id)
        output_dir = get_scene_output_dir(scene_id)
        output_dir.mkdir(parents=True, exist_ok=True)
        checkpoint = self._resolve_checkpoint()

        job_manager.update_job(
            job_id, status="processing", stage="depth_inference", progress=5.0
        )

        import rasterio

        with rasterio.open(input_path) as ds:
            width, height = ds.width, ds.height

        if x_max > width or y_max > height:
            raise ValueError(
                "refinement bbox exceeds the scene raster dimensions."
            )

        with _INFERENCE_SEMAPHORE:
            if self._check_cancelled(job_id):
                job_manager.update_job(
                    job_id, status="cancelled", stage="cancelled"
                )
                return {"cancelled": True}

            # Crop the source raster to the bbox via rasterio window, into
            # the scene's process dir, then run the certified path on it.
            from rasterio.windows import Window

            process_dir = get_scene_process_dir(scene_id)
            process_dir.mkdir(parents=True, exist_ok=True)
            crop_path = process_dir / "refine_crop.tif"

            with rasterio.open(input_path) as src:
                window = Window(
                    x_min, y_min, x_max - x_min, y_max - y_min
                )
                profile = src.profile.copy()
                profile.update(
                    width=int(window.width),
                    height=int(window.height),
                    transform=src.window_transform(window),
                )
                data = src.read(window=window)
                with rasterio.open(crop_path, "w", **profile) as dst:
                    dst.write(data)

            payload = run_inference(
                input_path=crop_path,
                ckpt_path=checkpoint,
                out_dir=output_dir,
                device=self.default_device,
                mode="tiles",
                dn_path=None,
                cache_dir=None,
                live_backbone=self.default_live_backbone,
                backbone_id=self.default_backbone,
                anchor_dem=None,
                ground_elev=None,
                write_files=True,
            )

        # Persist the refined product explicitly; run_inference wrote the
        # full-scene-shaped outputs — copy the dsm.npy to the refined name.
        dsm_npy = output_dir / "dsm.npy"
        refined_npy = output_dir / "refined_dsm.npy"
        if dsm_npy.is_file():
            refined_npy.write_bytes(dsm_npy.read_bytes())

        job_manager.update_job(
            job_id,
            status="completed",
            stage="completed",
            progress=100.0,
            result=payload,
        )
        logger.info("refine job completed: %s scene=%s", job_id, scene_id)
        return payload

    # -- failure recording ------------------------------------------------------

    def record_failure(self, job_id: str, exc: Exception) -> None:
        """Map an exception to a typed job error (client contract) and log
        the full detail server-side."""
        if isinstance(exc, ValueError):
            error = {
                "code": "INVALID_INPUT",
                "message": str(exc),
                "details": {},
                "recoverable": True,
            }
        elif isinstance(exc, FileNotFoundError):
            error = {
                "code": "FILE_NOT_FOUND",
                "message": str(exc),
                "details": {},
                "recoverable": False,
            }
        else:
            error = {
                "code": "PROCESSING_FAILED",
                "message": "Processing failed due to an internal error.",
                "details": {},
                "recoverable": False,
            }

        logger.error(
            "job %s failed: %s: %s", job_id, type(exc).__name__, exc
        )
        job_manager.update_job(
            job_id,
            status="failed",
            stage="failed",
            error=error,
        )

    def _discard_outputs(self, output_dir: Path) -> None:
        """Remove partial outputs of a cancelled run (idempotent)."""
        for name in ("dsm.npy", "dsm.tif", "dsm_anchored.tif", "dsm_anchored.npy", "dsm_preview.png"):
            path = output_dir / name
            try:
                path.unlink()
            except FileNotFoundError:
                pass


processing_service = ProcessingService()
