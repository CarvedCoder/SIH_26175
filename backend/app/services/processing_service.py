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
import time
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
            # Automatic reference-DEM acquisition (absolute DSM Part A);
            # run_inference falls back to the relative product with an
            # explicit notice when retrieval fails.
            "dem_provider": s.dem_provider,
            "dem_cache_dir": s.dem_cache_dir,
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

    # -- live progress ---------------------------------------------------------

    #: Overall job progress band owned by tiled inference (5% -> 80%).
    _INFERENCE_PROGRESS_START = 5.0
    _INFERENCE_PROGRESS_SPAN = 75.0
    #: Minimum seconds between persisted progress writes (polls read every 2.5s).
    _PROGRESS_EMIT_INTERVAL = 0.5

    def _progress_reporter(self, job_id: str):
        """Map inference tile callbacks -> durable job progress.

        ``run_inference`` reports (done, total) after every tiled forward
        (backbone Dn windows, prediction tiles, TerraHeight AGL tiles).
        That fraction is mapped onto the 5%–80% band, clamped monotonic
        (the backbone pass and the prediction pass each restart at
        done=1), and persisted at most twice a second. No fabricated
        percentages: every emitted value comes from real completed tiles.
        """
        state = {
            "last_emit": 0.0,
            "last_progress": self._INFERENCE_PROGRESS_START,
        }

        def report(done: int, total: int) -> None:
            if total <= 0 or done <= 0:
                return
            now = time.monotonic()
            if done < total and now - state["last_emit"] < self._PROGRESS_EMIT_INTERVAL:
                return
            state["last_emit"] = now
            frac = min(done / total, 1.0)
            progress = max(
                self._INFERENCE_PROGRESS_START + frac * self._INFERENCE_PROGRESS_SPAN,
                state["last_progress"],
            )
            state["last_progress"] = progress
            try:
                self.jobs.update_job(
                    job_id,
                    progress=round(progress, 1),
                    message=f"Tile {done} / {total}",
                )
            except Exception:  # noqa: BLE001 — progress must never kill a job
                logger.warning("progress update failed for job %s", job_id)

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
        eager_analysis: bool = False,
    ) -> dict[str, Any]:
        """Run DepthWizard inference for a scene (blocking; call from a worker).

        Lazy two-phase delivery (default): the job completes as soon as the
        elevation products exist — dsm.npy / dsm.tif / dsm_anchored /
        dsm_preview.png written by run_inference — and the slower tail
        (reference validation, disaster assessment, 3D building
        reconstruction) is DEFERRED to continue_analysis(), triggered by
        POST /scenes/{id}/analyze. ``eager_analysis=True`` restores the
        single-job behaviour (everything runs before completion).
        """

        input_path = self.resolve_scene_input(scene_id)
        output_dir = scene_artifact_store().path_for(scene_output_dir_key(scene_id))
        process_dir = get_scene_process_dir(scene_id)
        output_dir.mkdir(parents=True, exist_ok=True)
        process_dir.mkdir(parents=True, exist_ok=True)

        checkpoint, architecture = self._resolve_checkpoint(architecture, allow_download=True)

        self.jobs.update_job(
            job_id,
            status="processing",
            stage="preprocessing",
            progress=2.0,
            message="Preparing input & loading checkpoint",
        )

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
                    progress_cb=self._progress_reporter(job_id),
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

        # ── Lazy fast path: elevation products ARE the deliverable ──────
        # run_inference has written dsm.npy / dsm.tif / dsm_anchored /
        # dsm_preview.png (and the semantic artifacts). Everything below
        # this point in the eager path is the slow analysis tail; in lazy
        # mode it waits for an explicit POST /scenes/{id}/analyze.
        if not eager_analysis:
            payload["analysis_status"] = "pending"
            payload["analysis_pending"] = True
            payload["analysis_stages"] = [
                "validation", "disaster_assessment", "buildings3d",
            ]

            # Publication is cheap and keeps the object-store contract
            # intact (presigned asset URLs in the results response).
            self._publish_artifacts(scene_id, output_dir)

            self.jobs.update_job(
                job_id,
                status="completed",
                stage="dsm_ready",
                progress=100.0,
                message="Elevation products ready — full analysis available on demand",
                result=self._sanitize_payload_paths(payload),
            )
            logger.info(
                "job completed (dsm_ready, analysis deferred): %s scene=%s",
                job_id, scene_id,
            )
            return payload

        # Live validation: if the scene carries a ground-truth reference
        # raster, produce reference.npy / validation.json / error_map.png
        # NOW — get_validation only reports what actually exists on disk.
        self.jobs.update_job(
            job_id, progress=82.0, message="Writing validation artifacts"
        )
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

        buildings3d = self._run_buildings3d_stage(job_id, scene_id, output_dir)
        if buildings3d:
            payload["buildings3d"] = buildings3d

        self._publish_artifacts(scene_id, output_dir)

        payload["analysis_status"] = "complete"
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
                    progress_cb=self._progress_reporter(job_id),
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

    def continue_analysis(
        self, job_id: str, scene_id: str
    ) -> dict[str, Any]:
        """Run the analysis tail a lazy processing job deferred.

        Consumes EXISTING elevation products only (no re-inference):
        reference validation -> disaster assessment -> 3D building
        reconstruction -> artifact publication. The merged payload of the
        original processing job (meta, provenance) plus the new analysis
        blocks becomes this job's result, so GET /scenes/{id}/results
        keeps serving full provenance from the latest job record.
        """
        input_path = self.resolve_scene_input(scene_id)
        output_dir = scene_artifact_store().path_for(scene_output_dir_key(scene_id))
        if not (output_dir / "dsm.npy").is_file():
            raise FileNotFoundError(
                "scene has no elevation products yet — process it first "
                f"(scene={scene_id})"
            )

        self.jobs.update_job(
            job_id,
            status="processing",
            stage="analyzing",
            progress=5.0,
            message="Running deferred analysis",
        )

        with self.jobs.lease_heartbeat(job_id):
            # Live validation (reference.npy / validation.json / error_map).
            self.jobs.update_job(
                job_id, progress=20.0, message="Writing validation artifacts"
            )
            self._write_validation_artifacts(scene_id, input_path, output_dir)

            # Disaster assessment (optional, never fatal).
            disaster_result = self._run_disaster_stage(
                job_id, scene_id, input_path, output_dir
            )

            # Geometry-aware 3D building reconstruction.
            self.jobs.update_job(
                job_id, progress=60.0, message="Reconstructing building geometry"
            )
            buildings3d = self._run_buildings3d_stage(job_id, scene_id, output_dir)

        self._publish_artifacts(scene_id, output_dir)

        # Merge into the original processing payload so the latest job
        # record keeps serving meta/provenance through the results route.
        payload: dict[str, Any] = {}
        try:
            for job in reversed(self.jobs.list_jobs_for_scene(scene_id)):
                result = getattr(job, "result", None)
                if isinstance(result, dict) and result.get("analysis_pending"):
                    payload = dict(result)
                    break
        except Exception:  # noqa: BLE001 — merge is best-effort
            logger.warning(
                "could not merge original payload for scene %s", scene_id
            )
        if disaster_result:
            payload["disaster"] = disaster_result
        if buildings3d:
            payload["buildings3d"] = buildings3d
        payload["analysis_status"] = "complete"
        payload.pop("analysis_pending", None)

        self.jobs.update_job(
            job_id,
            status="completed",
            stage="completed",
            progress=100.0,
            result=self._sanitize_payload_paths(payload),
        )
        logger.info("analysis job completed: %s scene=%s", job_id, scene_id)
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
            if request.get("kind") == "analyze":
                return self.continue_analysis(job_id, job.scene_id)
            return self.process_scene(
                job_id,
                job.scene_id,
                mode=request.get("mode", "auto"),
                architecture=request.get("architecture", "auto"),
                ground_elev=request.get("ground_elev"),
                eager_analysis=bool(request.get("eager_analysis", False)),
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

    # -- geometry-aware 3D building reconstruction ---------------------------

    def _building3d_config(self):
        """Map Settings -> depthwizard.reconstruction Building3DConfig."""
        from depthwizard.reconstruction import Building3DConfig

        s = self.settings
        return Building3DConfig(
            min_area_px=s.buildings3d_min_area_px,
            simplify_tol_px=s.buildings3d_simplify_tol_px,
            rect_iou=s.buildings3d_rect_iou,
            circle_circularity=s.buildings3d_circle_circularity,
            stadium_iou=s.buildings3d_stadium_iou,
            max_parts=s.buildings3d_max_parts,
            max_vertices=s.buildings3d_max_vertices,
            height_mad_trim=s.buildings3d_mad_trim,
            piecewise_gap_m=s.buildings3d_piecewise_gap_m,
            min_confidence_for_primitive=s.buildings3d_min_building_conf,
            cluster_close_px=s.buildings3d_cluster_close_px,
        )

    def _run_buildings3d_stage(self, job_id: str, scene_id: str, output_dir: Path) -> dict | None:
        """Geometry-aware 3D building reconstruction (optional stage).

        Consumes EXISTING outputs only — the disaster ONNX building mask
        when available, else the semantic `building` class — plus the
        predicted metric DSM (dsm.npy). Never fails the job: any error is
        logged and the stage reports honestly unavailable.
        """
        s = self.settings
        if not s.buildings3d_enabled:
            return None

        dsm_path = output_dir / "dsm.npy"
        if not dsm_path.is_file():
            return None

        # -- building candidate: ONNX detector mask, else semantic class 0
        building_mask_path = output_dir / "building_mask.npy"
        confidence = None
        if building_mask_path.is_file():
            building_mask = np.load(building_mask_path)
            mask_source = "onnx_building_detector"
            conf_path = output_dir / "building_confidence.npy"
            if conf_path.is_file():
                confidence = np.load(conf_path).astype(np.float32)
        elif (output_dir / "semantic_labels.npy").is_file():
            labels = np.load(output_dir / "semantic_labels.npy")
            if labels.shape != dsm.shape:
                logger.warning(
                    "buildings3d skipped for job %s: semantic grid %s != "
                    "DSM grid %s", job_id, labels.shape, dsm.shape,
                )
                return None
            building_mask = labels == 0  # PROJECT_CLASSES[0] = 'building'
            mask_source = "semantic_building_class"
            probs_path = output_dir / "semantic_probs.npy"
            if probs_path.is_file():
                probs = np.load(probs_path)
                if probs.ndim == 3 and probs.shape[0] >= 1:
                    confidence = np.asarray(probs[0], dtype=np.float32)
        else:
            return None  # no building candidate exists — honest skip

        try:
            import rasterio

            from depthwizard.reconstruction import (
                reconstruct_buildings_3d,
                render_buildings3d_preview,
            )

            self.jobs.update_job(
                job_id, message="Reconstructing building geometry"
            )

            dsm = np.load(dsm_path)
            if building_mask.shape != dsm.shape:
                logger.warning(
                    "buildings3d skipped for job %s: mask grid %s != "
                    "DSM grid %s", job_id, building_mask.shape, dsm.shape,
                )
                return None

            crs = transform = None
            tif_path = output_dir / "dsm.tif"
            if tif_path.is_file():
                with rasterio.open(tif_path) as ds:
                    if ds.crs is not None:
                        crs = ds.crs.to_string()
                        transform = ds.transform

            # disaster damage rasters (when the damage model ran) travel
            # along so each 3D building carries its classified damage state
            damage_labels = damage_conf = None
            dmg_labels_path = output_dir / "damage_labels.npy"
            if dmg_labels_path.is_file():
                damage_labels = np.load(dmg_labels_path)
                dmg_conf_path = output_dir / "damage_confidence.npy"
                if dmg_conf_path.is_file():
                    damage_conf = np.load(dmg_conf_path).astype(np.float32)

            from PIL import Image as _Image, ImageDraw as _ImageDraw

            from scipy import ndimage as _nd

            # -- recall-fusion pass FIRST: catch structures the primary
            # detector missed (DSM elevation evidence + optional RGB), and
            # small compact objects (tree canopies, vehicles, containers).
            objects = []
            object_mask = np.zeros_like(building_mask, dtype=bool)
            extra_count = 0
            try:
                from depthwizard.reconstruction import fuse_building_candidates

                rgb_frame = None
                try:
                    from backend.app.application.scenes.service import (
                        scene_service,
                    )

                    _input = scene_service.find_scene_input(scene_id)
                    if _input is not None:
                        import rasterio

                        with rasterio.open(_input) as _ds:
                            _arr = _ds.read(list(range(1, min(3, _ds.count) + 1)))
                        rgb_frame = np.moveaxis(_arr, 0, -1).astype(np.float32)
                        if rgb_frame.shape[:2] != dsm.shape[:2]:
                            from PIL import Image as _Image

                            rgb_frame = np.asarray(
                                _Image.fromarray(rgb_frame.astype(np.uint8)).resize(
                                    (dsm.shape[1], dsm.shape[0]), _Image.BILINEAR
                                ),
                                dtype=np.float32,
                            )
                except Exception as _rgb_err:
                    logger.info("fusion pass without RGB: %s", _rgb_err)

                extra_mask, objects, object_mask = fuse_building_candidates(
                    dsm.astype(np.float32),
                    building_mask,
                    rgb=rgb_frame,
                    config=self._building3d_config(),
                )
                extra_count = int(extra_mask.sum())
            except Exception as _fus_err:  # noqa: BLE001 — recall is additive
                logger.warning("fusion detection pass failed: %s", _fus_err)

            # -- cluster the fused mask: fragments of ONE structure (a
            # stadium ring the detector splits into segments, L-wings)
            # are merged into a single component so the reconstruction
            # emits ONE coherent 3D object per building.
            _structure = np.ones((3, 3), dtype=bool)
            fused_mask = np.logical_or(building_mask.astype(bool), extra_mask)
            clustered_mask = _nd.binary_closing(
                fused_mask, structure=_structure,
                iterations=self._building3d_config().cluster_close_px,
            )
            clustered_mask = _nd.binary_fill_holes(clustered_mask)

            reconstruction = reconstruct_buildings_3d(
                clustered_mask.astype(np.uint8),
                dsm.astype(np.float32),
                confidence_raster=confidence,
                mask_source=mask_source,
                crs=crs,
                transform=transform,
                config=self._building3d_config(),
                damage_labels=damage_labels,
                damage_confidence=damage_conf,
                source_ref_mask=building_mask.astype(bool),
            )

            # -- ground DSM: every RENDERED region (clustered buildings +
            # fusion extras + small objects) replaced by its nearest
            # surrounding ground, so the Buildings-3D mode removes exactly
            # the bumps that got 3D objects — mountains/hills keep their
            # elevation (natural terrain is never flattened).
            bmask = np.logical_or(clustered_mask, object_mask.astype(bool))
            ground_dsm = dsm.copy()
            if bmask.any():
                _, (iy, ix) = _nd.distance_transform_edt(
                    bmask, return_indices=True
                )
                ground_dsm[bmask] = dsm[iy[bmask], ix[bmask]]
                # light smooth over the inpainted regions only
                blurred = _nd.median_filter(np.nan_to_num(ground_dsm, nan=0.0), size=5)
                ground_dsm[bmask] = blurred[bmask]

            lo = float(np.nanmin(dsm)) if np.isfinite(dsm).any() else 0.0
            hi = float(np.nanmax(dsm)) if np.isfinite(dsm).any() else 1.0
            if hi - lo < 1e-6:
                hi = lo + 1.0
            gnorm = np.clip(
                (np.nan_to_num(ground_dsm, nan=lo) - lo) / (hi - lo), 0.0, 1.0
            )
            from PIL import Image as _Image

            gimg = _Image.fromarray((gnorm * 65535.0).astype(np.uint16))
            gmax = max(gimg.size)
            if gmax > 1024:
                gimg = gimg.resize(
                    (int(gimg.width * 1024 / gmax), int(gimg.height * 1024 / gmax)),
                    _Image.LANCZOS,
                )
            gimg.save(output_dir / "ground_heightmap.png")
            reconstruction["has_ground"] = True
            reconstruction["ground_range_m"] = [round(lo, 3), round(hi, 3)]

            # -- vegetation/tree candidates: green-dominant + elevated
            # pixels OUTSIDE building footprints, clustered to stand
            # centers. A classical CV heuristic (NOT a semantic model —
            # labeled honestly in the payload) so the 3D view can render
            # custom tree objects; a detection model can replace it later.
            trees = []
            try:
                from backend.app.application.scenes.service import scene_service

                input_path = scene_service.find_scene_input(scene_id)
                if input_path is not None:
                    import rasterio

                    with rasterio.open(input_path) as ds:
                        bands = min(3, ds.count)
                        arr = ds.read(list(range(1, bands + 1)))
                    rgb = np.moveaxis(arr, 0, -1).astype(np.float32)
                    if rgb.shape[:2] != dsm.shape[:2]:
                        im = _Image.fromarray(rgb.astype(np.uint8)).resize(
                            (dsm.shape[1], dsm.shape[0]), _Image.BILINEAR
                        )
                        rgb = np.asarray(im, dtype=np.float32)
                    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
                    greenish = (g > r * 1.05) & (g > b * 1.05) & (g > 40)
                    elevated = np.isfinite(dsm) & ((dsm - ground_dsm) > 1.5)
                    cand = greenish & elevated & ~bmask
                    if cand.sum() >= 200:
                        tl, tn = _nd.label(_nd.binary_opening(cand, np.ones((3, 3))))
                        tsz = _nd.sum_labels(
                            np.ones_like(cand, np.int32), tl, range(1, tn + 1)
                        )
                        order = np.argsort(tsz)[::-1][:100]
                        structure = np.ones((3, 3), bool)
                        for oi in order:
                            if tsz[oi] < 60:
                                continue
                            comp = tl == (oi + 1)
                            core = _nd.binary_erosion(comp, structure=structure)
                            use = core if core.sum() >= 20 else comp
                            ys, xs = np.nonzero(use)
                            tv = dsm[use] - ground_dsm[use]
                            tv = tv[np.isfinite(tv)]
                            if tv.size == 0 or float(np.median(tv)) <= 1.5:
                                continue
                            cy, cx = int(ys.mean()), int(xs.mean())
                            trees.append({
                                "x_px": round(float(xs.mean()), 1),
                                "y_px": round(float(ys.mean()), 1),
                                "height_m": round(float(np.median(tv)), 2),
                                "ground_elevation_m": round(float(ground_dsm[cy, cx]), 2),
                                "pixel_count": int(comp.sum()),
                            })
            except Exception as _tree_err:  # noqa: BLE001 — trees are additive
                logger.warning("tree candidate extraction failed: %s", _tree_err)
            reconstruction["trees"] = trees
            reconstruction["tree_count"] = len(trees)
            reconstruction["trees_source"] = "heuristic_greenery_height"
            reconstruction["objects"] = objects
            reconstruction["object_count"] = len(objects)
            reconstruction["fusion_extra_px"] = extra_count

            # -- clean elevated heightfield: every reconstructed building's
            # roof region is LEVELLED to its model-derived height (base +
            # robust median), so the elevated rendering shows clean flat
            # tops in the detected footprints instead of noisy per-pixel
            # spikes. Streets/ground/mountains keep the raw DSM. This is
            # the OFF-state rendering companion to the ground heightfield.
            try:
                clean_dsm = dsm.copy()
                _draw_img = _Image.new("L", (dsm.shape[1], dsm.shape[0]), 0)
                _dr = _ImageDraw.Draw(_draw_img)
                for _b in reconstruction.get("buildings", []):
                    _pts = [(float(x), float(y)) for x, y in _b.get("footprint_px", [])]
                    if len(_pts) >= 3:
                        _dr.polygon(_pts, outline=1, fill=1)
                    for _lvl in _b.get("levels", []):
                        _lp = [(float(x), float(y)) for x, y in _lvl.get("polygon_px", [])]
                        if len(_lp) >= 3:
                            _dr.polygon(_lp, outline=1, fill=1)
                bmask_clean = np.asarray(_draw_img, dtype=bool)
                if bmask_clean.any():
                    for _b in reconstruction.get("buildings", []):
                        _pts = [(float(x), float(y)) for x, y in _b.get("footprint_px", [])]
                        if len(_pts) < 3:
                            continue
                        _img = _Image.new("L", (dsm.shape[1], dsm.shape[0]), 0)
                        _ImageDraw.Draw(_img).polygon(_pts, outline=1, fill=1)
                        _m = np.asarray(_img, dtype=bool)
                        clean_dsm[_m] = _b["base_elevation_m"] + _b["height_m"]
                        for _lvl in _b.get("levels", []):
                            _lp = [(float(x), float(y)) for x, y in _lvl.get("polygon_px", [])]
                            if len(_lp) >= 3:
                                _li = _Image.new("L", (dsm.shape[1], dsm.shape[0]), 0)
                                _ImageDraw.Draw(_li).polygon(_lp, outline=1, fill=1)
                                _lm = np.asarray(_li, dtype=bool)
                                clean_dsm[_lm] = _b["base_elevation_m"] + _lvl["height_m"]
                    _lo = float(np.nanmin(dsm)) if np.isfinite(dsm).any() else 0.0
                    _hi = float(np.nanmax(dsm)) if np.isfinite(dsm).any() else 1.0
                    if _hi - _lo < 1e-6:
                        _hi = _lo + 1.0
                    _cnorm = np.clip(
                        (np.nan_to_num(clean_dsm, nan=_lo) - _lo) / (_hi - _lo),
                        0.0, 1.0,
                    )
                    _cimg = _Image.fromarray((_cnorm * 65535.0).astype(np.uint16))
                    if max(_cimg.size) > 1024:
                        _cimg = _cimg.resize(
                            (int(_cimg.width * 1024 / max(_cimg.size)),
                             int(_cimg.height * 1024 / max(_cimg.size))),
                            _Image.LANCZOS,
                        )
                    _cimg.save(output_dir / "clean_heightmap.png")
                    reconstruction["has_clean"] = True
            except Exception as _clean_err:  # noqa: BLE001 — rendering nicety
                logger.warning("clean heightfield generation failed: %s", _clean_err)

            with open(output_dir / "buildings3d.json", "w", encoding="utf-8") as f:
                json.dump(reconstruction, f)

            # preview needs the RGB — reuse the saved preview artifact when
            # present (rgb is not kept in memory on this path)
            preview_path = output_dir / "dsm_preview.png"
            if reconstruction.get("available") and preview_path.is_file():
                from PIL import Image

                rgb = np.asarray(Image.open(preview_path).convert("RGB"))
                if rgb.shape[:2] == dsm.shape[:2]:
                    overlay = render_buildings3d_preview(rgb, reconstruction)
                    Image.fromarray(overlay).save(
                        output_dir / "buildings3d_preview.png"
                    )

            logger.info(
                "buildings3d: %d structure(s) reconstructed for job %s "
                "(source=%s)", reconstruction.get("count", 0), job_id,
                mask_source,
            )
            return {
                "available": bool(reconstruction.get("available")),
                "count": int(reconstruction.get("count", 0)),
                "mask_source": mask_source,
                "height_source": reconstruction.get("height_source"),
                "georeferenced": bool(reconstruction.get("georeferenced")),
                "damage_classified": int(reconstruction.get("damage_classified", 0)),
                "damage_classes": reconstruction.get("damage_classes", []),
                "tree_count": int(reconstruction.get("tree_count", 0)),
                "has_ground": bool(reconstruction.get("has_ground")),
                "has_clean": bool(reconstruction.get("has_clean")),
            }

        except Exception as exc:
            logger.warning(
                "buildings3d reconstruction failed for job %s: %s",
                job_id, exc,
            )
            return None

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
            # Geometry-aware 3D building reconstruction artifacts
            "buildings3d.json", "buildings3d_preview.png", "ground_heightmap.png",
            "clean_heightmap.png",
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

