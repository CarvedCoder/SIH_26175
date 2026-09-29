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

import json
import os
import threading
from pathlib import Path
from typing import Any

import numpy as np


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
from depthwizard.inference import InferenceCancelled, run_inference

# Reference-raster discovery conventions (see _find_reference_raster).
_REFERENCE_BASENAMES = ("reference.tif", "reference_dem.tif", "ref_dem.tif")
_REFERENCE_STEM_MARKERS = (
    "reference",
    "truth",
    "groundtruth",
    "ground_truth",
    "_gt",
)

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

    def _resolve_checkpoint(
        self, architecture: str = "auto", *, allow_download: bool = False
    ) -> tuple[Path, str]:
        """Resolve the checkpoint and CONCRETE architecture for the
        requested height-model backend -> (checkpoint, "rdah"|"calibration_net").

        rdah: the released pretrained Track1 checkpoint. With
        ``allow_download`` (inference paths only — NEVER health probes) a
        missing checkpoint is auto-downloaded + MD5-verified on first use
        (depthwizard/rdah.py). DW_CKPT_RDAH (or the generic DW_CKPT) may
        point at a different RDAH checkpoint — e.g. a fine-tuned one; an
        override that is NOT an RDAH checkpoint is a loud error, never a
        silent swap.
        calibration_net: DW_CKPT_CALIB, else DW_CKPT, else the repo default
        below (mirrored in docker-compose), else the tiny tracked flagship
        at the repo root.
        auto (legacy clients and old job records): follow the checkpoint —
        the generic DW_CKPT's detected architecture when set, else the RDAH
        default.
        """

        env_checkpoint = self.settings.checkpoint

        if architecture == "auto":
            if env_checkpoint and Path(env_checkpoint).exists():
                from depthwizard.tifops import detect_architecture

                architecture = detect_architecture(env_checkpoint)
            else:
                architecture = "rdah"

        if architecture == "rdah":
            env_checkpoint = self.settings.ckpt_rdah or env_checkpoint
            if env_checkpoint:
                env_path = Path(env_checkpoint)
                if not env_path.exists():
                    raise FileNotFoundError(
                        "the configured RDAH checkpoint override points at "
                        f"a missing file: {env_path}"
                    )
                from depthwizard.tifops import detect_architecture

                if detect_architecture(str(env_path)) != "rdah":
                    raise ValueError(
                        "the configured RDAH checkpoint override "
                        f"({env_path}) is not an RDAH checkpoint — check "
                        "DW_CKPT_RDAH / DW_CKPT."
                    )
                return self._verify_checkpoint_sha(env_path), "rdah"
            default = (
                PROJECT_ROOT / "checkpoints" / "rdah" / "rdah_track1_best_model.pth"
            )
            if allow_download:
                from depthwizard.rdah import ensure_rdah_checkpoint

                return Path(ensure_rdah_checkpoint(str(default))), "rdah"
            if not default.exists():
                raise FileNotFoundError(
                    "RDAH checkpoint not present yet — it is downloaded and "
                    "MD5-verified automatically on the first processing job "
                    "(depthwizard/rdah.py), or pre-fetch it with "
                    "`python model.py infer`."
                )
            return default, "rdah"

        env_checkpoint = self.settings.ckpt_calib or self.settings.checkpoint
        if env_checkpoint:
            checkpoint = Path(env_checkpoint)
        else:
            checkpoint = (
                PROJECT_ROOT / "outputs" / "calib_net" / "postproc_flagship" / "best.pt"
            )
            if not checkpoint.exists():
                # Tiny tracked fallback (repo root) so calibration_net stays
                # servable on a fresh clone with no trained outputs.
                checkpoint = PROJECT_ROOT / "best.pt"

        if not checkpoint.exists():
            raise FileNotFoundError(
                "DepthWizard checkpoint not found. Set DW_CKPT to a valid "
                "checkpoint."
            )

        return self._verify_checkpoint_sha(checkpoint), architecture

    def _verify_checkpoint_sha(self, checkpoint: Path) -> Path:
        """Enforce DW_CKPT_SHA256 when set (defense against a tampered or
        swapped serving checkpoint)."""

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

    # -- live validation (reference-grounded metrics) ------------------------

    def _find_reference_raster(
        self, scene_id: str, input_path: Path
    ) -> Path | None:
        """The scene's ground-truth reference raster, if one exists.

        Two deterministic conventions, checked in order:
          * exact filenames: reference.tif / reference_dem.tif / ref_dem.tif
            next to the scene input;
          * a single sibling raster whose stem marks it as truth/gt
            (DFC2019/GAMUS-style *-truth tiles).
        Several marked siblings is AMBIGUOUS — refuse rather than guess
        (the same determinism contract as resolve_scene_input).
        """
        raw_dir = get_scene_raw_dir(scene_id)
        if not raw_dir.exists():
            return None
        input_resolved = input_path.resolve()
        siblings = sorted(
            p
            for p in raw_dir.iterdir()
            if p.is_file()
            and p.suffix.lower() in {".tif", ".tiff"}
            and p.resolve() != input_resolved
        )
        exact = [p for p in siblings if p.name.lower() in _REFERENCE_BASENAMES]
        if exact:
            return exact[0]
        marked = [
            p
            for p in siblings
            if any(marker in p.stem.lower() for marker in _REFERENCE_STEM_MARKERS)
        ]
        if len(marked) > 1:
            logger.warning(
                "scene %s has %d candidate reference rasters — refusing to "
                "pick one for validation",
                scene_id,
                len(marked),
            )
            return None
        return marked[0] if marked else None

    def _reproject_reference_to_grid(
        self,
        ref_path: Path,
        crs: Any,
        transform: Any,
        width: int,
        height: int,
    ) -> np.ndarray:
        """Bilinear-reproject the reference raster onto the prediction's
        exact grid. Mirrors depthwizard.anchoring.resample_dem_to_tile:
        CRS required on BOTH sides, bilinear resampling, and PARTIAL
        COVERAGE REFUSED (a partially covered comparison would silently
        average fabricated NaN-free errors)."""
        import rasterio
        from rasterio.warp import Resampling, reproject

        with rasterio.open(ref_path) as ref:
            if ref.crs is None:
                raise ValueError(
                    f"reference '{ref_path.name}' has no CRS — refusing to "
                    "compare grids"
                )
            if crs is None:
                raise ValueError(
                    "prediction grid has no CRS — reference reprojection "
                    "is undefined"
                )
            dst = np.full((height, width), np.nan, dtype=np.float32)
            try:
                reproject(
                    source=rasterio.band(ref, 1),
                    destination=dst,
                    src_transform=ref.transform,
                    src_crs=ref.crs,
                    src_nodata=ref.nodata,
                    dst_transform=transform,
                    dst_crs=crs,
                    dst_nodata=np.nan,
                    resampling=Resampling.bilinear,
                )
            except Exception as exc:
                raise ValueError(
                    f"reference->prediction reprojection failed for "
                    f"'{ref_path.name}': {exc}"
                ) from exc

        if np.isnan(dst).any():
            raise ValueError(
                f"reference '{ref_path.name}' does not fully cover the "
                f"prediction footprint ({np.isnan(dst).mean():.1%} missing "
                "pixels) — refusing partial-coverage validation."
            )
        return dst

    def _write_validation_artifacts(
        self, scene_id: str, input_path: Path, output_dir: Path
    ) -> None:
        """If the scene carries a ground-truth reference raster, write
        reference.npy (ON the prediction grid), validation.json (masked-
        difference metrics from depthwizard.metrics.height_metrics) and
        error_map.png into the scene output dir.

        A validation failure NEVER fails the job: it is logged and the
        artifacts stay absent — get_validation then reports honestly that
        no validation exists (no fabricated metrics anywhere).
        """
        ref_path = self._find_reference_raster(scene_id, input_path)
        if ref_path is None:
            return

        import rasterio
        from depthwizard.metrics import height_metrics
        from depthwizard.pipeline.scene_outputs import save_preview_png

        pred_path = output_dir / "dsm.npy"
        if not pred_path.exists():
            return
        prediction = np.load(pred_path)
        height, width = prediction.shape

        tif_path = output_dir / "dsm.tif"
        if tif_path.exists():
            with rasterio.open(tif_path) as ds:
                crs, transform = ds.crs, ds.transform
        else:
            crs = transform = None

        try:
            if crs is not None:
                reference = self._reproject_reference_to_grid(
                    ref_path, crs, transform, width, height
                )
                reprojected = True
                ref_crs: str | None = str(crs)
            else:
                # Non-georeferenced prediction: only a pixel-registered
                # (CRS-less) same-grid reference can be compared honestly.
                with rasterio.open(ref_path) as ref:
                    if ref.crs is not None:
                        raise ValueError(
                            "prediction has no CRS but the reference is "
                            "georeferenced — the grids cannot be aligned"
                        )
                    reference = ref.read(1).astype(np.float32)
                if reference.shape != prediction.shape:
                    raise ValueError(
                        f"reference shape {reference.shape} != prediction "
                        f"shape {prediction.shape} and no grid to "
                        "reproject onto"
                    )
                reprojected = False
                ref_crs = None

            metrics = height_metrics(prediction, reference)
            if metrics["n"] == 0:
                raise ValueError(
                    "reference has no finite pixels overlapping the "
                    "prediction"
                )

            np.save(output_dir / "reference.npy", reference.astype(np.float32))

            validation_meta = {
                "source_reference": ref_path.name,
                "reference_crs": ref_crs,
                "reference_units": "meters",
                "reprojected_to_prediction_grid": reprojected,
                "metrics": {
                    "sample_count": int(metrics["n"]),
                    "rmse": metrics["rmse"],
                    "mae": metrics["mae"],
                    "median_abs_error": metrics["medae"],
                    "bias": metrics["bias"],
                    "correlation": metrics["pearson_r"],
                },
            }
            with open(output_dir / "validation.json", "w", encoding="utf-8") as f:
                json.dump(validation_meta, f, indent=2)

            error = (
                prediction.astype(np.float64) - reference.astype(np.float64)
            ).astype(np.float32)
            save_preview_png(error, output_dir / "error_map.png", "DSM error (m)")

            logger.info(
                "validation artifacts written for scene %s (n=%d, mae=%.3f)",
                scene_id,
                metrics["n"],
                metrics["mae"],
            )
        except Exception as exc:
            logger.warning(
                "validation skipped for scene %s: %s", scene_id, exc
            )

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

    _PROGRESS_MIN_INTERVAL = 0.8  # s — persistence cadence for progress beats
    _PROGRESS_MIN_STEP = 1.0  # pct — always report a jump this large

    def _progress_reporter(self, job_id: str):
        """Build a throttled on_progress callback for run_inference.

        Stage keys mirror the inference pipeline (preprocessing,
        depth_estimation, refinement, dsm_generation, validation); every
        stage change is persisted immediately, percentage-only updates at
        most every ~0.8 s. Job rows are durable storage, not a log — so
        the callback WRITES state, never raises.
        """
        state = {"stage": None, "pct": 0.0, "t": 0.0}

        def report(stage: str, pct: float) -> None:
            import time as _time

            pct = max(0.0, min(float(pct), 99.0))
            now = _time.monotonic()
            stage_changed = stage != state["stage"]
            due = now - state["t"] >= self._PROGRESS_MIN_INTERVAL
            jumped = pct - state["pct"] >= self._PROGRESS_MIN_STEP
            if not (stage_changed or due or jumped):
                return
            state.update(stage=stage, pct=pct, t=now)
            try:
                self.jobs.update_job(
                    job_id,
                    status="processing",
                    stage=stage,
                    progress=round(pct, 1),
                )
            except Exception:  # noqa: BLE001 — reporting must not kill a job
                logger.exception("progress update failed for job %s", job_id)

        return report

    # -- processing -----------------------------------------------------------

    def process_scene(
        self,
        job_id: str,
        scene_id: str,
        *,
        mode: str = "auto",
        architecture: str = "auto",
        ground_elev: float | None = None,
    ) -> dict[str, Any]:
        """Run DepthWizard inference for a scene (blocking; call from a worker)."""

        input_path = self.resolve_scene_input(scene_id)
        output_dir = scene_artifact_store().path_for(scene_output_dir_key(scene_id))
        process_dir = get_scene_process_dir(scene_id)
        output_dir.mkdir(parents=True, exist_ok=True)
        process_dir.mkdir(parents=True, exist_ok=True)

        checkpoint, architecture = self._resolve_checkpoint(architecture, allow_download=True)

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

            try:
                payload = run_inference(
                    input_path=input_path,
                    ckpt_path=checkpoint,
                    out_dir=output_dir,
                    mode=mode,
                    architecture=architecture,
                    ground_elev=ground_elev,
                    write_files=True,
                    should_cancel=lambda: self._check_cancelled(job_id),
                    on_progress=self._progress_reporter(job_id),
                    **self._inference_kwargs(),
                )
            except InferenceCancelled:
                # Cooperative cancel fired inside the pipeline (between tiles
                # or at a stage boundary) — drop partial outputs, mark the
                # job cancelled. The next poll transitions the UI back.
                self._discard_outputs(output_dir)
                self.jobs.update_job(
                    job_id, status="cancelled", stage="cancelled"
                )
                return {"cancelled": True}

        if self._check_cancelled(job_id):
            self._discard_outputs(output_dir)
            self.jobs.update_job(job_id, status="cancelled", stage="cancelled")
            return {"cancelled": True}

        # Live validation: if the scene carries a ground-truth reference
        # raster, produce reference.npy / validation.json / error_map.png
        # NOW — get_validation only reports what actually exists on disk.
        self._write_validation_artifacts(scene_id, input_path, output_dir)

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
        architecture: str = "auto",
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
        checkpoint, architecture = self._resolve_checkpoint(architecture, allow_download=True)

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

            try:
                payload = run_inference(
                    input_path=crop_path,
                    ckpt_path=checkpoint,
                    out_dir=output_dir,
                    mode="tiles",
                    architecture=architecture,
                    write_files=True,
                    should_cancel=lambda: self._check_cancelled(job_id),
                    on_progress=self._progress_reporter(job_id),
                    **self._inference_kwargs(),
                )
            except InferenceCancelled:
                self.jobs.update_job(
                    job_id, status="cancelled", stage="cancelled"
                )
                return {"cancelled": True}

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
                    architecture=request.get("architecture", "auto"),
                )
            return self.process_scene(
                job_id,
                job.scene_id,
                mode=request.get("mode", "auto"),
                architecture=request.get("architecture", "auto"),
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
