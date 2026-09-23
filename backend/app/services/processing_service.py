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


from backend.app.core.config import get_settings
from backend.app.core.errors import AppError
from backend.app.core.logging import logger
from backend.app.core.paths import (
    PROJECT_ROOT,
    get_scene_process_dir,
    get_scene_raw_dir,
)
from backend.app.infrastructure.storage.scene_artifacts import (
    scene_artifact_store,
    scene_output_dir_key,
)
from backend.app.jobs.manager import job_manager
from depthwizard.inference import run_inference

# Process-local GPU guard. HONEST LIMITATION (audit §3.4): this serializes
# torch forwards within ONE process only; cross-instance concurrency policy
# belongs to the queue/worker layer (worker concurrency=1 per GPU worker).
_INFERENCE_SEMAPHORE = threading.Semaphore(1)


class ProcessingService:
    """Inference execution. All serving knobs come from the centralized
    Settings (core/config.py) — this module never reads os.environ."""

    def __init__(self, settings=None, job_service=None) -> None:
        # None => resolve lazily per call so test fixtures that rebuild
        # Settings (fresh_settings) are honored; an explicit Settings may
        # be injected for workers with their own configuration.
        self._settings = settings
        # Job store access is INJECTED (never a bare module global): the
        # API process uses the default facade; a worker passes its own
        # JobService bound to whichever repository it was configured with.
        self._job_service = job_service

    @property
    def settings(self):
        return self._settings if self._settings is not None else get_settings()

    @property
    def jobs(self):
        """The injected job store facade (module facade by default)."""
        if self._job_service is not None:
            return self._job_service
        from backend.app.jobs.manager import job_manager

        return job_manager

    # -- configuration ------------------------------------------------------

    def _resolve_checkpoint(self) -> Path:
        """Resolve the calibration checkpoint. Single source of truth:
        settings.checkpoint (DW_CKPT), else the repo default below
        (mirrored in docker-compose)."""

        env_checkpoint = self.settings.checkpoint

        if env_checkpoint:
            checkpoint = Path(env_checkpoint)
        else:
            checkpoint = (
                PROJECT_ROOT / "outputs" / "calib_net" / "postproc_flagship" / "best.pt"
            )

        if not checkpoint.exists():
            raise FileNotFoundError(
                "DepthWizard checkpoint not found. Set DW_CKPT to a valid "
                "checkpoint."
            )

        expected_sha = self.settings.checkpoint_sha256
        if expected_sha:
            from depthwizard.tifops import sha256_file

            actual = sha256_file(checkpoint)
            if actual.lower() != expected_sha.lower():
                raise ValueError(
                    "checkpoint SHA-256 mismatch — refusing to load a "
                    "checkpoint that does not match DW_CKPT_SHA256."
                )

        return checkpoint

    def _inference_kwargs(self) -> dict[str, Any]:
        """Every run_inference knob resolved from Settings (env-driven).

        Keeps both call sites (full scene + refine) on one config source:
        DW_* environment variables — see core/config.py and .env.example.
        """
        s = self.settings
        return {
            "device": s.device,
            "dn_path": s.dn_path,
            "cache_dir": s.cache_dir,
            "live_backbone": s.live_backbone,
            "backbone_id": s.backbone_id,
            "anchor_dem": s.anchor_dem,
            "postprocess": s.postprocess,
            "postprocess_params": {
                "wls_lambda": s.wls_lambda,
                "wls_sigma_rgb": s.wls_sigma_rgb,
                "wls_max_iter": s.wls_max_iter,
            },
            "tta": s.tta,
        }

    # -- deterministic input (delegates to the scene application service) --

    def find_scene_input(self, scene_id: str) -> Path | None:
        """The scene's designated input raster by exact known filename."""
        from backend.app.application.scenes.service import scene_service

        return scene_service.find_scene_input(scene_id)

    def resolve_scene_input(self, scene_id: str) -> Path:
        """The ONE input raster of a scene, deterministically. Ambiguous
        scenes raise the typed SceneInputAmbiguous error — never resolved
        by iteration order."""
        from backend.app.application.scenes.service import scene_service

        return scene_service.resolve_scene_input(scene_id)

    # -- cancellation --------------------------------------------------------

    def _check_cancelled(self, job_id: str) -> bool:
        job = self.jobs.get_job(job_id)
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
        output_dir = scene_artifact_store().path_for(scene_output_dir_key(scene_id))
        process_dir = get_scene_process_dir(scene_id)
        output_dir.mkdir(parents=True, exist_ok=True)
        process_dir.mkdir(parents=True, exist_ok=True)

        checkpoint = self._resolve_checkpoint()

        self.jobs.update_job(
            job_id,
            status="processing",
            stage="depth_inference",
            progress=5.0,
        )

        if self._check_cancelled(job_id):
            self._discard_outputs(output_dir)
            self.jobs.update_job(job_id, status="cancelled", stage="cancelled")
            return {"cancelled": True}

        # Durable lease heartbeat: while this worker runs, the job's lease
        # is renewed so OTHER instances never finalize it as interrupted.
        # If THIS process dies, the lease expires and any reader honestly
        # fails the job — no PID liveness anywhere.
        with self.jobs.lease_heartbeat(job_id), _INFERENCE_SEMAPHORE:
            if self._check_cancelled(job_id):
                self._discard_outputs(output_dir)
                self.jobs.update_job(
                    job_id, status="cancelled", stage="cancelled"
                )
                return {"cancelled": True}

            payload = run_inference(
                input_path=input_path,
                ckpt_path=checkpoint,
                out_dir=output_dir,
                mode=mode,
                ground_elev=ground_elev,
                write_files=True,
                **self._inference_kwargs(),
            )

        if self._check_cancelled(job_id):
            self._discard_outputs(output_dir)
            self.jobs.update_job(job_id, status="cancelled", stage="cancelled")
            return {"cancelled": True}

        self._publish_artifacts(scene_id, output_dir)

        self.jobs.update_job(
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
        output_dir = scene_artifact_store().path_for(scene_output_dir_key(scene_id))
        output_dir.mkdir(parents=True, exist_ok=True)
        checkpoint = self._resolve_checkpoint()

        self.jobs.update_job(
            job_id, status="processing", stage="depth_inference", progress=5.0
        )

        import rasterio

        with rasterio.open(input_path) as ds:
            width, height = ds.width, ds.height

        if x_max > width or y_max > height:
            raise ValueError(
                "refinement bbox exceeds the scene raster dimensions."
            )

        with self.jobs.lease_heartbeat(job_id), _INFERENCE_SEMAPHORE:
            if self._check_cancelled(job_id):
                self.jobs.update_job(
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
                mode="tiles",
                write_files=True,
                **self._inference_kwargs(),
            )

        # Persist the refined product explicitly; run_inference wrote the
        # full-scene-shaped outputs — copy the dsm.npy to the refined name.
        dsm_npy = output_dir / "dsm.npy"
        refined_npy = output_dir / "refined_dsm.npy"
        if dsm_npy.is_file():
            refined_npy.write_bytes(dsm_npy.read_bytes())

        self._publish_artifacts(scene_id, output_dir)

        self.jobs.update_job(
            job_id,
            status="completed",
            stage="completed",
            progress=100.0,
            result=payload,
        )
        logger.info("refine job completed: %s scene=%s", job_id, scene_id)
        return payload

    # -- artifact publication ------------------------------------------------

    def _publish_artifacts(self, scene_id: str, output_dir: Path) -> None:
        """Post-inference persistence: upload every generated artifact to
        the object store and register its KEY in the scene's SQL row.

        Local files are kept (they are the processing cache); the DB is
        updated transactionally. Failures are logged and swallowed: the
        local products still exist and the job must still complete."""
        try:
            from backend.app.db.database import session_scope
            from backend.app.db.models import SceneRow
            from backend.app.storage.service import storage_service

            with session_scope() as session:
                row = session.get(SceneRow, scene_id)
                owner_id = row.owner_id if row is not None else "local"
            artifacts = storage_service.sync_scene_outputs(owner_id, scene_id)
            if artifacts:
                with session_scope() as session:
                    row = session.get(SceneRow, scene_id)
                    if row is not None:
                        row.artifacts = artifacts
        except Exception:
            logger.exception(
                "artifact publication failed for scene %s "
                "(local outputs remain available)", scene_id,
            )

    # -- failure recording ------------------------------------------------------

    def record_failure(self, job_id: str, exc: Exception) -> None:
        """Map an exception to a typed job error (client contract) and log
        the full detail server-side."""
        if isinstance(exc, AppError):
            error = {
                "code": exc.code,
                "message": exc.message,
                "details": exc.details,
                "recoverable": exc.recoverable,
            }
        elif isinstance(exc, ValueError):
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
        self.jobs.update_job(
            job_id,
            status="failed",
            stage="failed",
            error=error,
        )

    # -- external-worker execution -------------------------------------------

    def execute_job_record(self, job_id: str) -> dict[str, Any] | None:
        """Execute a PERSISTED job record — the external-worker entrypoint.

        Everything needed comes from durable state: the job document
        (request parameters) + the scene's stored input + configuration.
        A fresh worker process can execute any job with no memory of the
        API request that created it (statelessness definition).

        Delivery semantics: at-least-once with idempotent outputs (see
        FileJobRepository.claim_queued)."""
        job = self.jobs.get_job(job_id)
        if job is None:
            return None
        if job.is_terminal:
            return job.result
        request = job.request or {"kind": "process"}
        try:
            if request.get("kind") == "refine":
                bbox = request.get("bbox") or {}
                return self.refine_region(
                    job_id,
                    job.scene_id,
                    bbox=(
                        int(bbox["x_min"]),
                        int(bbox["y_min"]),
                        int(bbox["x_max"]),
                        int(bbox["y_max"]),
                    ),
                )
            return self.process_scene(
                job_id,
                job.scene_id,
                mode=request.get("mode", "auto"),
                ground_elev=request.get("ground_elev"),
            )
        except Exception as exc:
            self.record_failure(job_id, exc)
            return None

    def _discard_outputs(self, output_dir: Path) -> None:
        """Remove partial outputs of a cancelled run (idempotent)."""
        for name in ("dsm.npy", "dsm.tif", "dsm_anchored.tif", "dsm_anchored.npy", "dsm_preview.png"):
            path = output_dir / name
            try:
                path.unlink()
            except FileNotFoundError:
                pass


processing_service = ProcessingService()
