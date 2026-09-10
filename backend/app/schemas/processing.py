from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, ConfigDict, Field


class ProcessMode(str, Enum):
    AUTO = "auto"
    CROP = "crop"
    RESIZE = "resize"
    TILES = "tiles"


class ProcessRequest(BaseModel):
    """Body of POST /scenes/{id}/process.

    Strict on purpose (extra="forbid"): every field is either USED by the
    pipeline or REJECTED with a 422 — nothing is silently ignored.

    Not accepted (and why), so clients don't send phantom knobs:
        model            the serving stack runs the DAv2 + CalibrationNet
                         flagship; there is no alternative model to select.
        tile_size        fixed at 1024 by the training contract (tiles are
                         min-max normalized per 1024 tile); any other value
                         would silently change the Dn statistics the net
                         was calibrated on.
        overlap          the deterministic edge-padded tiler has no overlap
                         blending to configure.
        enable_refinement
                         local refinement is a separate operation
                         (POST /scenes/{id}/refine), not an inference-time
                         toggle.
        enable_reference_calibration
                         calibration IS the model; anchoring to an absolute
                         datum is requested via ``ground_elev``.
    """

    model_config = ConfigDict(extra="forbid")

    mode: ProcessMode = ProcessMode.AUTO
    ground_elev: float | None = Field(
        default=None,
        description="Constant ground datum (metres) for absolute anchoring "
        "[ANCHORED (not learned)]. Ignored when None.",
    )


class ProcessAccepted(BaseModel):
    """Returned when a processing job was created."""

    job_id: str
    scene_id: str
    status: str = "queued"
    stage: str = "queued"
    progress: float | None = None


class BoundingBox(BaseModel):
    """Pixel-space bounding box of a region of interest."""

    x_min: int = Field(..., ge=0)
    y_min: int = Field(..., ge=0)
    x_max: int = Field(..., ge=0)
    y_max: int = Field(..., ge=0)


class RefineRequest(BaseModel):
    """Body of POST /scenes/{id}/refine — local re-processing of a bbox.

    The region is re-run through the SAME certified inference path at
    source resolution (no grid downscale); the refined product is stored
    alongside the scene results and returned through the job record.
    """

    model_config = ConfigDict(extra="forbid")

    bbox: BoundingBox
    resolution: str = Field(default="high", pattern="^(standard|high)$")


class RefineAccepted(BaseModel):
    """Returned when a refinement job was created."""

    job_id: str
    scene_id: str
    status: str = "queued"
    stage: str = "queued"
    progress: float | None = None
