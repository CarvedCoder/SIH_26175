from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, ConfigDict, Field


class ProcessMode(str, Enum):
    AUTO = "auto"
    CROP = "crop"
    RESIZE = "resize"
    TILES = "tiles"


class ModelArchitecture(str, Enum):
    """Height-model backend (RDAH integration). Both share the identical
    downstream path (tiling, anchoring, post-processing, DSM writer) —
    only the height net and its depth preprocessing differ:

        rdah             official RDAH-Net (HeightPredTransformer) with the
                         released pretrained Track1 checkpoint; depth input
                         is RAW DAv2 x 40; output is unclamped nDSM metres.
        calibration_net  the legacy Phase-2 net (H = clamp(a*Dn + b, 0),
                         Dn min-max [0,1] per tile); output clamped >= 0.
        auto             follow the checkpoint: DW_CKPT's detected
                         architecture when set, else the pretrained RDAH
                         default (the legacy-client behaviour).
    """

    RDAH = "rdah"
    CALIBRATION_NET = "calibration_net"
    AUTO = "auto"


class ProcessRequest(BaseModel):
    """Body of POST /scenes/{id}/process.

    Strict on purpose (extra="forbid"): every field is either USED by the
    pipeline or REJECTED with a 422 — nothing is silently ignored.

    Not accepted (and why), so clients don't send phantom knobs:
        model            the backend switch is ``architecture``; there is
                         no free-form model name to select.
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
    architecture: ModelArchitecture = Field(
        default=ModelArchitecture.AUTO,
        description="Height-model backend: rdah (pretrained RDAH-Net), "
        "calibration_net (legacy Phase-2 net), or auto (follow the "
        "configured checkpoint; the legacy-client default).",
    )
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
    architecture: ModelArchitecture = Field(
        default=ModelArchitecture.AUTO,
        description="Height-model backend for the re-processed region — "
        "keep it consistent with the full-scene run you are refining.",
    )


class RefineAccepted(BaseModel):
    """Returned when a refinement job was created."""

    job_id: str
    scene_id: str
    status: str = "queued"
    stage: str = "queued"
    progress: float | None = None
