"""Post-processing configuration — the single source of every tunable.

Every parameter that influences deployed refinement lives HERE (task rule:
no hard-coded experimental constants scattered through the source). The
dataclass doubles as documentation: each field's comment states its meaning,
its unit, and how it trades off.

Method ladder (``method`` field):
    "none"         identity — the raw AGL is returned untouched. The BASELINE.
    "median"       3x3 median (tiny, honest denoiser; ablation V1)
    "guided"       RGB-guided filter, He et al. ECCV 2010 (V2)
    "bilateral"    joint bilateral filter, RGB-guided (V3a)
    "wls"          edge-aware weighted least squares, RGB guidance (V3b/V4)
    "conf_wls"     confidence-weighted WLS (V4)
    "semantic_wls" semantic + RGB gated WLS (V5 / V7 — semantics from the
                   model's PREDICTED auxiliary head, never GT)
    "planar"       EXPERIMENTAL robust local plane fitting (alone)
    "full"         spike removal + confidence + semantic WLS (+ planar when
                   ``planar`` is also enabled) — the composed pipeline (V8)

Post-processing ALWAYS runs on AGL in metres, BEFORE DEM anchoring:
    DSM_anchored = refine(AGL_raw) + DEM
No normalization of the calibrated signal happens anywhere in this package.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from typing import Any, Dict

METHODS = (
    "none",
    "median",
    "guided",
    "bilateral",
    "wls",
    "conf_wls",
    "semantic_wls",
    "planar",
    "full",
)


@dataclass
class PostProcessConfig:
    """All knobs for the AGL post-processing stage (units in comments)."""

    enabled: bool = False
    # ---- core refinement method (see module docstring ladder) ------------
    method: str = "guided"

    # ---- guided filter (He et al. 2010) -----------------------------------
    guided_radius: int = 4  # box-window radius in px (window = 2r+1)
    guided_eps: float = 1e-3  # edge threshold: variance of normalized [0,1]
    #   guidance luma below which edges are KEPT. Larger -> more edges kept
    #   (less smoothing); smaller -> stronger smoothing. Units: normalized
    #   intensity^2, i.e. (1/255)^2 = 1.5e-5 per grey level.

    # ---- joint bilateral filter -------------------------------------------
    bilateral_radius: int = 5  # window radius in px
    bilateral_sigma_color: float = 0.1  # RGB distance (in [0,1]) at which a
    #   neighbour's weight drops to exp(-0.5). Small -> only very similar
    #   colours are averaged (strong edge preservation, less denoising).
    bilateral_sigma_space: float = 3.0  # spatial Gaussian sigma in px

    # ---- weighted least squares -------------------------------------------
    wls_lambda: float = 1.0  # smoothness vs data-fidelity balance
    #   (dimensionless; the data term is unit weight, so lambda is in
    #   1/m^2 of edge-weighted graph energy — see wls.py). Larger ->
    #   smoother output that can drift further from AGL_raw.
    wls_sigma_rgb: float = 0.1  # RGB edge scale: neighbour colour distance
    #   (L1 over [0,1] RGB) at which the smoothness weight falls to
    #   exp(-1). Smaller -> edges block smoothing more aggressively.
    wls_tol: float = 1e-6  # conjugate-gradient relative residual tolerance
    wls_max_iter: int = 120  # CG iteration cap (larger -> more exact solve)

    # ---- semantic gating ---------------------------------------------------
    semantic_edge_weight: float = 2.0  # exponent on the pairwise semantic
    #   similarity (Bhattacharyya coefficient between neighbouring class
    #   distributions). 0 disables semantics even if probs are supplied;
    #   1 = plain product; >1 sharpens boundary protection.
    semantic_hard_boundary: bool = False  # True -> ZERO smoothing across any
    #   neighbouring argmax-class change (brittle; only for experiments)

    # ---- confidence weighting (conf_wls / full) -----------------------------
    confidence_weight: float = 1.0  # exponent applied to the confidence map
    #   in the data term (0 disables confidence even if a map is supplied)
    confidence_floor: float = 0.15  # minimum data-term weight for a pixel
    #   with zero confidence — keeps even unreliable pixels anchored to
    #   AGL_raw so the solve can never run away.

    # ---- conservative spike removal ----------------------------------------
    spike_removal: bool = True  # median/MAD isolated-outlier stage BEFORE
    #   refinement (never touches edge-supported structures)
    spike_radius: int = 1  # local median window radius in px (3x3 default)
    spike_tau: float = 3.8  # robust z threshold: |x - median| > tau * MAD
    spike_min_isolation: float = 0.65  # fraction of boundary ring pixels that
    #   must disagree in elevation (for an isolated needle peak, surrounding
    #   boundary is lower; for a pit, surrounding boundary is higher).
    spike_max_rgb_grad: float = 0.15  # RGB L1-gradient (in [0,1] units)
    spike_min_height: float = 0.6  # minimum elevation delta (metres) for spikes
    spike_max_component_size: int = 6  # max connected-component pixel area
    #   (components larger than this are protected as genuine roofs/structures)
    spike_post_refine: bool = True  # also run spike removal on refined output
    #   to catch filter-induced edge-decoupling spikes
    spike_max_local_slope: Optional[float] = 85.0  # max plausible local slope (deg)

    # ---- test-time augmentation -------------------------------------------
    tta: bool = False  # hflip+vflip+180° ensemble (runtime ~3-4x — measure
    #   before enabling in production)
    tta_aggregation: str = "median"  # "median" (robust) | "mean"

    # ---- EXPERIMENTAL robust planar refinement ------------------------------
    planar: bool = False  # only refines PREDICTED building regions
    planar_min_area: int = 400  # minimum connected-component size (px)
    planar_max_residual: float = 1.5  # robust inlier RMSE (metres) above
    #   which the fitted plane is REJECTED for a component
    planar_min_inlier_frac: float = 0.6  # fraction of component pixels that
    #   must lie within planar_inlier_tol of the plane
    planar_inlier_tol: float = 1.0  # inlier distance to plane (metres)

    # ---- output guards ------------------------------------------------------
    clamp_min: float = 0.0  # same clamp the CalibrationNet applies to its
    #   own output (AGL >= 0 is the model's physical contract, not an
    #   arbitrary clip). Set to None to disable.
    preserve_mean: bool = False  # True -> add back (mean(raw) - mean(refined))
    #   so the global height scale is EXACTLY preserved. Default False: the
    #   data term already keeps the mean shift negligible; the shift is
    #   always measured and REPORTED (see calibration report in meta).

    def __post_init__(self) -> None:
        if self.method not in METHODS:
            raise ValueError(
                f"unknown postprocess method '{self.method}' "
                f"(choose from {METHODS})"
            )
        if self.guided_radius < 1:
            raise ValueError("guided_radius must be >= 1")
        if self.wls_lambda < 0:
            raise ValueError("wls_lambda must be >= 0")

    # ------------------------------------------------------------------
    @property
    def effective_method(self) -> str:
        """Refinement core actually used ('none' when disabled)."""
        return self.method if self.enabled else "none"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "PostProcessConfig":
        known = {k: v for k, v in d.items() if k in PostProcessConfig.__dataclass_fields__}
        return PostProcessConfig(**known)

    def with_updates(self, **kw: Any) -> "PostProcessConfig":
        return replace(self, **kw)


# ---------------------------------------------------------------------------
# CLI/API presets — one name per surface the user is offered (task Sec. 19)
# ---------------------------------------------------------------------------

PRESETS: Dict[str, Dict[str, Any]] = {
    # identity / backward-compatible default
    "none": {"enabled": False, "method": "none"},
    "median": {"enabled": True, "method": "median", "spike_removal": True},
    "guided": {"enabled": True, "method": "guided", "spike_removal": True},
    "bilateral": {"enabled": True, "method": "bilateral", "spike_removal": True},
    "wls": {"enabled": True, "method": "wls", "spike_removal": True},
    "conf": {"enabled": True, "method": "conf_wls", "spike_removal": True},
    "semantic": {
        "enabled": True,
        "method": "semantic_wls",
        "spike_removal": True,
    },
    "planar": {"enabled": True, "method": "planar", "spike_removal": False},
    "full": {
        "enabled": True,
        "method": "full",  # composed core: semantic + confidence + WLS
        "spike_removal": True,
        "confidence_weight": 1.0,
        "semantic_edge_weight": 2.0,
        "planar": False,  # opt-in extra (experimental)
    },
}


def config_from_preset(name: str, overrides: Dict[str, Any] | None = None) -> PostProcessConfig:
    """Build a config from a preset name + optional parameter overrides.

    ``overrides`` accepts any PostProcessConfig field (unknown keys raise
    through the dataclass constructor — no silent typos).
    """
    base = dict(PRESETS.get(name, {"enabled": True, "method": name}))
    if overrides:
        base.update(overrides)
    return PostProcessConfig(**base)
