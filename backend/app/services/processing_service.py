from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Optional

from backend.app.core.paths import (
    get_scene_output_dir,
    get_scene_process_dir,
    get_scene_raw_dir,
)
from backend.app.jobs.manager import job_manager
from depthwizard.inference import run_inference


class ProcessingService:
    """Connects backend jobs to the existing DepthWizard inference pipeline."""

    def __init__(self) -> None:
        self.default_device = os.environ.get("DW_DEVICE", "auto")
        self.default_backbone = os.environ.get(
            "DW_BACKBONE",
            "depth-anything/Depth-Anything-V2-Base-hf",
        )
        self.default_live_backbone = os.environ.get(
            "DW_NO_LIVE"
        ) != "1"

    def _resolve_checkpoint(self) -> Path:
        """Resolve the calibration checkpoint used by DepthWizard."""

        env_checkpoint = os.environ.get("DW_CKPT")

        if env_checkpoint:
            checkpoint = Path(env_checkpoint)
        else:
            checkpoint = (
                Path(__file__).resolve().parents[3]
                / "outputs"
                / "calib_net"
                / "gamus_rgb_grad"
                / "best.pt"
            )

        if not checkpoint.exists():
            raise FileNotFoundError(
                f"DepthWizard checkpoint not found: {checkpoint}. "
                "Set DW_CKPT to a valid checkpoint."
            )

        return checkpoint

    def process_scene(
        self,
        job_id: str,
        scene_id: str,
        *,
        mode: str = "auto",
        anchor_dem: Optional[Path] = None,
        ground_elev: Optional[float] = None,
    ) -> dict[str, Any]:
        """Run DepthWizard inference for a scene."""

        input_dir = get_scene_raw_dir(scene_id)
        process_dir = get_scene_process_dir(scene_id)
        output_dir = get_scene_output_dir(scene_id)

        process_dir.mkdir(parents=True, exist_ok=True)
        output_dir.mkdir(parents=True, exist_ok=True)

        raster_files = [
            path
            for path in input_dir.iterdir()
            if path.is_file()
            and path.suffix.lower() in {".tif", ".tiff"}
        ]

        if not raster_files:
            raise FileNotFoundError(
                f"No GeoTIFF input found for scene '{scene_id}'."
            )

        input_path = raster_files[0]

        checkpoint = self._resolve_checkpoint()

        job_manager.update_job(
            job_id,
            status="processing",
            stage="loading",
        )

        try:
            job_manager.update_job(
                job_id,
                stage="inference",
            )

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
                anchor_dem=anchor_dem,
                ground_elev=ground_elev,
                write_files=True,
            )

            job_manager.update_job(
                job_id,
                status="completed",
                stage="completed",
                progress=100.0,
                result=payload,
            )

            return payload

        except ValueError as exc:
            error = {
                "code": "INVALID_INPUT",
                "message": str(exc),
                "details": {},
                "recoverable": True,
            }

            job_manager.update_job(
                job_id,
                status="failed",
                stage="failed",
                error=error,
            )

            raise

        except FileNotFoundError as exc:
            error = {
                "code": "FILE_NOT_FOUND",
                "message": str(exc),
                "details": {},
                "recoverable": False,
            }

            job_manager.update_job(
                job_id,
                status="failed",
                stage="failed",
                error=error,
            )

            raise

        except Exception as exc:
            error = {
                "code": "PROCESSING_FAILED",
                "message": str(exc),
                "details": {},
                "recoverable": False,
            }

            job_manager.update_job(
                job_id,
                status="failed",
                stage="failed",
                error=error,
            )

            raise


processing_service = ProcessingService()