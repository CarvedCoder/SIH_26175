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
        requested height-model backend ->
        (checkpoint, "rdah"|"calibration_net"|"terraheight_s").

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
        terraheight_s: DW_CKPT_TERRAHEIGHT, else the repo-relative
        candidates in depthwizard.terraheight.DEFAULT_CHECKPOINT_CANDIDATES
        (models/terraheight/best_model.pth, then repo-root best_model.pth).
        NEVER auto-downloaded — a missing external checkpoint is a loud
        FileNotFoundError naming the env var to set.
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

        if architecture == "terraheight_s":
            env_checkpoint = self.settings.ckpt_terraheight or env_checkpoint
            if env_checkpoint:
                env_path = Path(env_checkpoint)
                if not env_path.exists():
                    raise FileNotFoundError(
                        "the configured TerraHeight-S checkpoint override "
                        f"points at a missing file: {env_path} (check "
                        "DW_CKPT_TERRAHEIGHT / DW_CKPT)"
                    )
                from depthwizard.tifops import detect_architecture

                if detect_architecture(str(env_path)) != "terraheight_s":
                    raise ValueError(
                        "the configured TerraHeight-S checkpoint override "
                        f"({env_path}) is not a TerraHeight-S release "
                        "checkpoint — check DW_CKPT_TERRAHEIGHT / DW_CKPT."
                    )
                return self._verify_checkpoint_sha(env_path), "terraheight_s"
            from depthwizard.terraheight import default_checkpoint_path

            return default_checkpoint_path(PROJECT_ROOT), "terraheight_s"

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
            # TerraHeight tiled-inference knobs (used only when the resolved
            # backend is terraheight_s; run_inference ignores them otherwise)
            "terraheight_tile_size": s.terraheight_tile_size,
            "terraheight_overlap": max(
                s.terraheight_tile_size - s.terraheight_tile_stride, 0
            ),
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

        # ── Disaster assessment (optional, non-blocking) ──────────────
        # Runs AFTER depth/semantic inference. If the disaster models
        # are unavailable or fail, the existing depth pipeline results
        # are preserved intact. Partial failure is logged honestly.
        disaster_result = self._run_disaster_stage(
            job_id, scene_id, input_path, output_dir
        )
        if disaster_result:
            payload["disaster"] = disaster_result

        self._publish_artifacts(scene_id, output_dir)

        self.jobs.update_job(
            job_id,
            status="completed",
            stage="completed",
            progress=100.0,
            result=self._sanitize_payload_paths(payload),
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

            # Run the crop through the standard inference path into an
            # ISOLATED directory — run_inference always writes dsm.npy/
            # dsm.tif/dsm_preview.png, and pointing it at output_dir would
            # silently overwrite the full-scene depth products.
            refine_dir = process_dir / "refine"
            refine_dir.mkdir(parents=True, exist_ok=True)
            try:
                payload = run_inference(
                    input_path=crop_path,
                    ckpt_path=checkpoint,
                    out_dir=refine_dir,
                    mode="tiles",
                    architecture=architecture,
                    write_files=True,
                    should_cancel=lambda: self._check_cancelled(job_id),
                    **self._inference_kwargs(),
                )
            except InferenceCancelled:
                self.jobs.update_job(
                    job_id, status="cancelled", stage="cancelled"
                )
                return {"cancelled": True}

        # Persist the refined product under its own name; the full-scene
        # dsm.npy/dsm.tif in output_dir are never touched.
        dsm_npy = refine_dir / "dsm.npy"
        refined_npy = output_dir / "refined_dsm.npy"
        if dsm_npy.is_file():
            refined_npy.write_bytes(dsm_npy.read_bytes())
        else:
            raise RuntimeError(
                "refinement inference produced no dsm.npy — nothing stored."
            )
        import shutil

        shutil.rmtree(refine_dir, ignore_errors=True)
        payload = {
            "refined": True,
            "bbox": list(bbox),
            "outputs": {"refined_dsm": "refined_dsm.npy"},
        }

        self._publish_artifacts(scene_id, output_dir)

        self.jobs.update_job(
            job_id,
            status="completed",
            stage="completed",
            progress=100.0,
            result=self._sanitize_payload_paths(payload),
        )
        logger.info("refine job completed: %s scene=%s", job_id, scene_id)
        return payload

    @staticmethod
    def _sanitize_payload_paths(payload: dict) -> dict:
        """Strip absolute filesystem paths from a job result payload.

        Job results are served verbatim by /api/v1/jobs/{id}; the artifact
        FILENAMES are useful to clients, the server paths are not.
        """
        import copy as _copy

        # Read at call time so a redirected project root is respected
        from backend.app.core import paths as _paths

        root = str(_paths.PROJECT_ROOT)

        def _clean(value):
            if isinstance(value, dict):
                return {
                    k: (Path(v).name if isinstance(v, str) and v.startswith(root) else _clean(v))
                    for k, v in value.items()
                }
            if isinstance(value, list):
                return [
                    (Path(v).name if isinstance(v, str) and v.startswith(root) else _clean(v))
                    for v in value
                ]
            return value

        return _clean(_copy.deepcopy(payload)) if isinstance(payload, dict) else payload

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

    # -- disaster assessment integration ------------------------------------

    def _resolve_disaster_model_path(self, env_path: str | None, fallbacks: tuple[str, ...]) -> str | None:
        """Resolve an ONNX model path from settings, falling back to known
        locations relative to PROJECT_ROOT."""
        if env_path:
            p = Path(env_path)
            if p.exists():
                return str(p)
            # Try relative to PROJECT_ROOT
            rel = PROJECT_ROOT / env_path
            if rel.exists():
                return str(rel)
            logger.warning("configured disaster model path not found: %s", env_path)
            return None
        # Try fallback paths
        for fb in fallbacks:
            p = PROJECT_ROOT / fb
            if p.exists():
                return str(p)
        return None

    def _run_disaster_stage(
        self,
        job_id: str,
        scene_id: str,
        input_path: Path,
        output_dir: Path,
    ) -> dict | None:
        """Run disaster assessment as an optional post-inference stage.

        NEVER crashes the main pipeline — all errors are caught and
        returned as partial failure metadata.
        """
        s = self.settings
        if not s.disaster_enabled:
            return None

        try:
            self.jobs.update_job(
                job_id, stage="disaster_assessment", progress=85.0
            )

            # Resolve model paths
            building_model = self._resolve_disaster_model_path(
                s.building_model_onnx,
                ("local_model.onnx", "models/building/model.onnx"),
            )
            damage_model = self._resolve_disaster_model_path(
                s.damage_model_onnx,
                ("model.onnx", "models/damage/model.onnx"),
            )

            if not building_model and not damage_model:
                logger.info(
                    "disaster assessment skipped for scene %s: "
                    "no ONNX models found", scene_id
                )
                return None

            # Read the source RGB image
            from depthwizard.inference import read_image
            from depthwizard.pipeline.scene_outputs import georef_state

            rgb, profile = read_image(input_path)
            georef, crs, tf = georef_state(profile)

            from depthwizard.disaster.pipeline import run_disaster_pipeline

            result = run_disaster_pipeline(
                rgb,
                output_dir,
                building_model_path=building_model,
                damage_model_path=damage_model,
                device=s.disaster_device,
                threshold=s.building_threshold,
                tile_size=s.disaster_tile_size,
                tile_stride=s.disaster_tile_stride,
                batch_size=s.disaster_batch_size,
                georeferenced=georef,
                crs_string=str(crs) if crs else None,
                transform=tf,
                should_cancel=lambda: self._check_cancelled(job_id),
                building_detection_enabled=s.building_detection_enabled,
                damage_enabled=s.damage_enabled,
                recover_destroyed=s.disaster_recover_destroyed,
            )

            self.jobs.update_job(
                job_id, stage="disaster_complete", progress=95.0
            )

            return result

        except Exception as exc:
            logger.warning(
                "disaster assessment failed for scene %s: %s",
                scene_id, exc,
            )
            return {
                "buildings_available": False,
                "damage_available": False,
                "errors": [f"Disaster pipeline error: {exc}"],
            }

    def _discard_outputs(self, output_dir: Path) -> None:
        """Remove partial outputs of a cancelled run (idempotent).

        Covers every artifact the pipeline can have produced before a
        cancellation checkpoint — otherwise a cancelled scene could still
        report scene_has_results=True from orphan files.
        """
        names = [
            "dsm.npy", "dsm.tif", "dsm_anchored.tif", "dsm_anchored.npy",
            "dsm_preview.png", "agl_raw.npy", "postprocess_meta.json",
            "semantic_labels.npy", "semantic_probs.npy",
            "semantic_confidence.npy", "semantic_map.png",
            "semantic_meta.json", "refined_dsm.npy",
            "validation.json", "error_map.png", "validation_error_map.png",
            "heightmap.png",
            # TerraHeight-S backend artifacts (AGL product + provenance)
            "terraheight_agl.tif", "terraheight_agl.npy",
            "terraheight_preview.png", "terraheight_meta.json",
        ]
        names += [
            (key) for key in (
                "buildings.geojson", "building_mask.npy",
                "building_confidence.npy", "buildings_preview.png",
                "buildings_meta.json", "damage_buildings.geojson",
                "damage_labels.npy", "damage_confidence.npy",
                "damage_preview.png", "damage_meta.json",
            )
        ]
        for name in names:
            path = output_dir / name
            try:
                path.unlink()
            except FileNotFoundError:
                pass
        # derived tile/texture caches live in subdirectories
        import shutil

        for sub in ("tiles",):
            shutil.rmtree(output_dir / sub, ignore_errors=True)
        for stale in output_dir.glob("*_layer.png"):
            stale.unlink(missing_ok=True)
        (output_dir / "minimap.png").unlink(missing_ok=True)


processing_service = ProcessingService()

