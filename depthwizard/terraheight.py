"""TerraHeight-S backend for DepthWizard — adapter, preprocessing, checkpoint
loading, tiled AGL inference.

Architecture registry (see tifops.load_height_model):
    architecture = "calibration_net"  -> frozen Phase-2 CalibrationNet path
    architecture = "rdah"             -> official RDAH-Net backend (default)
    architecture = "terraheight_s"    -> THIS module (external pretrained model)

Pipeline position (DIFFERENT from RDAH/CalibrationNet — no depth cache):
    RGB -> TerraHeight-S (Depth Anything V2 Small + DPT head, trained on
          GAMUS) -> dense AGL metres
    -> (optional Track-2 anchoring) -> absolute DSM

TerraHeight is the FINAL height estimator: it never consumes the DAv2
relative-depth cache, an RDAH checkpoint, a CalibrationNet checkpoint or a
Dn input. The depth cache continues to exist for RDAH/CalibrationNet only.

--------------------------------------------------------------------------
PROVENANCE / VERIFIED FACTS (checkpoint metadata is authoritative — the
released payload embeds its own config, so nothing here is guessed):

    checkpoint : TerraHeight-S (https://huggingface.co/benfox6515/TerraHeight-S)
                 repo-root ``best_model.pth`` by default (see
                 DW_CKPT_TERRAHEIGHT / terraheight.default_checkpoint_path)
    license    : Apache-2.0 (TerraHeight repo); base weights
                 depth-anything/Depth-Anything-V2-Small
    architecture : official DepthAnythingV2 implementation — vendored
                 verbatim in depthwizard/vendor/depth_anything_v2 (the
                 checkpoint is a NATIVE DAv2 state dict, NOT loadable into
                 the HuggingFace transformers classes)
    model_config : {encoder: "vits", features: 64,
                    out_channels: [48, 96, 192, 384]}   (24,785,089 params)
    transform    : {mode: "normalized", scale_m: 8.492877943662961,
                    units: "metres_AGL", negative_training_targets:
                    "clamp_zero"}
    normalization: ImageNet mean [0.485,0.456,0.406] std [0.229,0.224,0.225]
    training crop: 630 x 630 (multiple of the 14-px ViT patch — inputs MUST
                    be a multiple of 14; the tiled path enforces this)

OUTPUT SCALE (validated empirically, task Sec. 10):
    metres_AGL = raw_model_output x scale_m, then clamped to >= 0.
    Validation on GAMUS val center crops (DC_02_26 / DC_04_23): interpreting
    the raw output directly as metres gives MAE 5.94 / 19.97 m; multiplying
    by the published scale_m gives MAE 3.02 / 4.28 m — the normalized
    transform wins decisively and matches the checkpoint's own
    ``transform.mode == "normalized"`` contract. The conversion lives ONLY
    here (TerraHeightModel.forward) — never in preprocessing helpers.

The model's own auxiliary validation metrics (embedded in the checkpoint)
are reported as PUBLISHED reference values only — local numbers come from
the ``evaluate`` command, never from this module.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Callable, Dict, Optional

import numpy as np

# ---------------------------------------------------------------------------
# Constants (single source for the TerraHeight backend)
# ---------------------------------------------------------------------------

ARCHITECTURE = "terraheight_s"
HEIGHT_TYPE = "AGL"

#: Published training crop (the checkpoint's ``crop_size`` field) — also the
#: default inference tile edge. Multiple of the ViT-14 patch size.
TERRAHEIGHT_TILE_SIZE = 630

#: Default overlap between adjacent inference windows. The checkpoint's own
#: inference config used TILE_OVERLAP = 0.25 -> 0.25 * 630 = 157 px, giving
#: stride 473 px. Memory-safe on a 6 GB GPU at batch size 1 (~330 MB peak).
TERRAHEIGHT_DEFAULT_OVERLAP = 157

#: Depth Anything V2 ViT patch size — every input edge must be a multiple.
#: 630 = 45 * 14; the tiled path edge-pads to a valid tile size.
DAV2_PATCH = 14

#: Published transform fallback (the checkpoint payload overrides this).
TERRAHEIGHT_SCALE_M = 8.492877943662961

#: Expected RGB normalization (embedded in the checkpoint; ImageNet recipe).
TERRAHEIGHT_MEAN = (0.485, 0.456, 0.406)
TERRAHEIGHT_STD = (0.229, 0.224, 0.225)

#: Small health-check probe size (multiple of 14 — never a full scene).
TERRAHEIGHT_PROBE_SIZE = 154

#: State-dict wrapper prefixes produced by the TerraHeight training wrapper.
_TERRAHEIGHT_PREFIXES = ("net.", "module.", "model.")

#: Repo-root fallback locations searched (in order) when neither
#: DW_CKPT_TERRAHEIGHT nor the conventional models/ path exists.
DEFAULT_CHECKPOINT_CANDIDATES = (
    "models/terraheight/best_model.pth",
    "best_model.pth",
)


def default_checkpoint_path(project_root: Path | str | None = None) -> Path:
    """First existing candidate checkpoint, resolved relative to the repo
    root (never an absolute machine-specific path). Raises FileNotFoundError
    with the searched locations — never downloads a replacement."""
    root = Path(project_root) if project_root else Path(__file__).resolve().parent.parent
    for rel in DEFAULT_CHECKPOINT_CANDIDATES:
        p = root / rel
        if p.exists():
            return p
    searched = ", ".join(str(root / rel) for rel in DEFAULT_CHECKPOINT_CANDIDATES)
    raise FileNotFoundError(
        "TerraHeight-S checkpoint not found — searched: " + searched +
        ". Set DW_CKPT_TERRAHEIGHT (e.g. models/terraheight/best_model.pth); "
        "the checkpoint is NOT downloaded automatically."
    )


# ---------------------------------------------------------------------------
# Checkpoint payload validation + scale conversion
# ---------------------------------------------------------------------------


def validate_terraheight_payload(ckpt: Any) -> None:
    """Fail fast on unexpected checkpoint contents (mirrors
    tifops.validate_checkpoint_payload's role for CalibrationNet).

    The released TerraHeight-S payload is a plain dict of primitives +
    tensors: {model (state dict), model_config, transform, normalization,
    config, train_statistics, validation_metrics, ...}. Anything else is
    rejected BEFORE any value is trusted."""
    from torch import Tensor

    if not isinstance(ckpt, dict):
        raise ValueError(
            f"checkpoint payload must be a dict, got {type(ckpt).__name__}"
        )
    required = ("model", "model_config", "transform")
    missing = [k for k in required if k not in ckpt]
    if missing:
        raise ValueError(
            f"checkpoint is missing required keys {missing} — not a "
            "TerraHeight-S release payload (expected 'model', "
            "'model_config', 'transform')."
        )
    state = ckpt["model"]
    if not isinstance(state, dict) or not all(
        isinstance(k, str) and isinstance(v, Tensor) for k, v in state.items()
    ):
        raise ValueError(
            "checkpoint 'model' must be a mapping of parameter name -> "
            "torch.Tensor."
        )
    mc = ckpt["model_config"]
    if not isinstance(mc, dict) or mc.get("encoder") != "vits":
        raise ValueError(
            f"checkpoint 'model_config' must be a dict with encoder='vits', "
            f"got {mc!r} — refusing to rebuild a different architecture."
        )
    tr = ckpt["transform"]
    if not isinstance(tr, dict) or tr.get("mode") != "normalized":
        raise ValueError(
            f"checkpoint 'transform.mode' must be 'normalized' (the only "
            f"conversion this adapter implements), got {tr!r}."
        )
    if tr.get("units") not in (None, "metres_AGL"):
        raise ValueError(
            f"checkpoint 'transform.units' is {tr.get('units')!r} — this "
            "adapter only emits metres AGL."
        )


def terraheight_scale_m(ckpt: dict) -> float:
    """metres-per-unit-of-raw-output from the checkpoint's own transform
    block (falls back to the published constant only when absent)."""
    scale = ckpt.get("transform", {}).get("scale_m", TERRAHEIGHT_SCALE_M)
    scale = float(scale)
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError(f"checkpoint transform.scale_m is invalid: {scale!r}")
    return scale


def _strip_prefixes(state: Dict[str, Any], prefixes=_TERRAHEIGHT_PREFIXES) -> Dict[str, Any]:
    """Drop the training-wrapper prefix so keys match DepthAnythingV2
    (``net.pretrained.*`` -> ``pretrained.*``). Handles stacked prefixes
    (``module.net.pretrained.*``) by stripping repeatedly."""
    out = {}
    for k, v in state.items():
        changed = True
        while changed:
            changed = False
            for p in prefixes:
                if k.startswith(p):
                    k = k[len(p):]
                    changed = True
                    break
        out[k] = v
    return out


# ---------------------------------------------------------------------------
# Model wrapper — the registry's "net" object for terraheight_s
# ---------------------------------------------------------------------------


def _adapter_class():
    """Build the TerraHeightModel class (deferred torch import)."""
    import torch
    import torch.nn as nn

    class TerraHeightModel(nn.Module):
        """RGB -> dense AGL metres. Same ``{"pred": [B,1,H,W]}`` forward
        contract as the RDAH/CalibrationNet adapters, with the depth-cache
        inputs REJECTED loudly (TerraHeight is itself the height estimator).

        ``rgb`` is ImageNet-normalized float32 [B,3,H,W] (see
        preprocess_rgb) — the normalization lives in ONE place per
        architecture (this module's callers), never inside forward.
        """

        def __init__(self, core, scale_m: float):
            super().__init__()
            self.core = core  # vendored official DepthAnythingV2
            self.scale_m = float(scale_m)
            # provenance surfaced by the health check / metadata writers
            self.architecture = ARCHITECTURE
            self.height_type = HEIGHT_TYPE

        @property
        def embed_dim(self):  # parity with the raw DAv2 module
            return self.core.embed_dim

        def forward(self, rgb, dn=None, dem=None, sem=None, stats=None):
            if dn is not None or dem is not None or sem is not None or stats is not None:
                raise ValueError(
                    "TerraHeight-S takes RGB ONLY — it does not consume a "
                    "depth cache (dn), DEM, semantics or stats. Route "
                    "RDAH/CalibrationNet requests through their own backend."
                )
            b, c, h, w = rgb.shape
            if c != 3:
                raise ValueError(f"TerraHeight expects 3-channel RGB, got {c}")
            if h % DAV2_PATCH or w % DAV2_PATCH:
                raise ValueError(
                    f"TerraHeight input edges must be multiples of "
                    f"{DAV2_PATCH} (ViT patch), got {h}x{w} — use the tiled "
                    "inference path (predict_agl_tiled)."
                )
            raw = self.core(rgb)  # [B,H,W], ReLU'd (>= 0) by the DAv2 head
            agl = raw * self.scale_m  # normalized transform -> metres AGL
            agl = torch.clamp(agl, min=0.0)  # published negative clamp
            return {"pred": agl[:, None]}  # [B,1,H,W]

    return TerraHeightModel


def build_terraheight_model(model_config: dict, scale_m: float):
    """Instantiate the EXACT released architecture and wrap it.

    Uses the vendored official DepthAnythingV2 implementation — never a
    substitute backbone. ``use_bn=False, use_clstoken=False`` are the
    official defaults the TerraHeight checkpoint was trained with (its
    state dict loads strict=True only with these)."""
    import torch

    from .vendor.depth_anything_v2.dpt import DepthAnythingV2

    encoder = str(model_config.get("encoder", "vits"))
    features = int(model_config.get("features", 64))
    out_channels = [int(c) for c in model_config.get("out_channels", [48, 96, 192, 384])]
    core = DepthAnythingV2(
        encoder=encoder, features=features, out_channels=out_channels,
    )
    TerraHeightModel = _adapter_class()
    return TerraHeightModel(core, scale_m)


# ---------------------------------------------------------------------------
# Preprocessing (ONE place for the TerraHeight RGB recipe)
# ---------------------------------------------------------------------------


def preprocess_rgb(rgb_u8: np.ndarray) -> np.ndarray:
    """uint8 [H,W,3] RGB -> float32 [3,H,W], /255 + ImageNet normalization
    (the checkpoint's own ``normalization`` block)."""
    rgb_f = rgb_u8[:, :, :3].astype(np.float32) / 255.0
    mean = np.asarray(TERRAHEIGHT_MEAN, dtype=np.float32).reshape(3, 1, 1)
    std = np.asarray(TERRAHEIGHT_STD, dtype=np.float32).reshape(3, 1, 1)
    return ((rgb_f.transpose(2, 0, 1) - mean) / std).astype(np.float32)


# ---------------------------------------------------------------------------
# Checkpoint loading (registry entry point)
# ---------------------------------------------------------------------------


def load_terraheight_model(ckpt_path: Path | str, device: str = "cpu") -> "object":
    """Rebuild TerraHeight-S from the released checkpoint -> tifops.LoadedModel.

    Security: the payload is loaded with ``weights_only=True`` (primitives +
    tensors) and schema-validated by validate_terraheight_payload before any
    value is used. The state dict must load strict=True — a partial or
    mutated checkpoint raises instead of silently degrading.
    """
    import torch

    from .tifops import LoadedModel, resolve_torch_device

    device = resolve_torch_device(device)
    ckpt_path = Path(ckpt_path)
    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"TerraHeight-S checkpoint not found: {ckpt_path} — set "
            f"DW_CKPT_TERRAHEIGHT or pass --checkpoint."
        )

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    validate_terraheight_payload(ckpt)
    scale_m = terraheight_scale_m(ckpt)

    net = build_terraheight_model(ckpt["model_config"], scale_m)
    state = _strip_prefixes(ckpt["model"])
    net.core.load_state_dict(state, strict=True)
    net.eval().to(device)

    model = LoadedModel(
        net=net,
        use_rgb=True,
        widths=(),
        clamp_min=0.0,
        affine_init={"a": scale_m, "b": 0.0},  # provenance only — NOT applied
        epoch=int(ckpt.get("best_epoch", 0)),
        checkpoint=ckpt_path,
        val_subset_mae=(
            float(ckpt["validation_metrics"]["all"]["mae"])
            if isinstance(ckpt.get("validation_metrics"), dict)
            and "all" in ckpt["validation_metrics"]
            else None  # PUBLISHED reference value, clearly labelled in reports
        ),
        architecture=ARCHITECTURE,
        height_type=HEIGHT_TYPE,
    )
    return model


def is_terraheight_payload(ckpt: Any) -> bool:
    """True when the (already loaded) payload is a TerraHeight release
    (used by tifops.detect_architecture — no silent guessing)."""
    return (
        isinstance(ckpt, dict)
        and isinstance(ckpt.get("model"), dict)
        and isinstance(ckpt.get("model_config"), dict)
        and isinstance(ckpt.get("transform"), dict)
        and "model_state" not in ckpt
        and "model_state_dict" not in ckpt
    )


# ---------------------------------------------------------------------------
# Predict-function factories (registry contract: predict(dn, rgb, stats))
# ---------------------------------------------------------------------------


def make_terraheight_predict_fn(model, device: str = "cpu") -> Callable:
    """predict(dn, rgb, stats) -> np.ndarray[H,W] float32 metres AGL.

    ``dn``/``stats`` must be None (TerraHeight consumes no depth cache —
    supplying one is a caller bug and raises loudly). ``rgb`` is uint8
    [H,W,3]; the tile's edges must already be multiples of 14 (the tiled
    path guarantees this)."""
    import torch

    net = model.net
    net.eval()

    @torch.no_grad()
    def predict(dn, rgb=None, stats=None) -> np.ndarray:
        if rgb is None:
            raise ValueError("TerraHeight predict() requires an RGB tile")
        if dn is not None or stats is not None:
            raise ValueError(
                "TerraHeight does not consume dn/stats — this call came "
                "through the RDAH/CalibrationNet path by mistake."
            )
        x = torch.from_numpy(preprocess_rgb(np.ascontiguousarray(rgb))[None]).to(device)
        return net(x)["pred"][0, 0].cpu().numpy().astype(np.float32)

    return predict


def make_terraheight_full_predict_fn(model, device: str = "cpu") -> Callable:
    """predict_full(dn, rgb, stats) -> {"pred", "sem_probs": None, ...}.

    TerraHeight has NO semantic head (task Sec. 20: it is a height model,
    not a semantic segmentor) — sem_probs is always None and the scene
    payload reports that honestly."""
    base = make_terraheight_predict_fn(model, device)

    def predict_full(dn, rgb=None, stats=None) -> Dict[str, Any]:
        pred = base(dn, rgb, stats)
        return {
            "pred": pred,
            "sem_probs": None,
            "sem_zero_filled": False,
            "stats_zero_filled": False,
        }

    return predict_full


# ---------------------------------------------------------------------------
# Tiled inference (seam-free, 630 training-crop windows, batch size 1)
# ---------------------------------------------------------------------------


def _edge_pad_tile(rgb_u8: np.ndarray, window, tile: int) -> np.ndarray:
    """Extract one TileWindow's RGB region, edge-padded to (tile, tile) when
    the source is smaller than the tile (edge replication — never invented
    content; same rule as inference._window_tile)."""
    vh, vw = window.valid_height, window.valid_width
    region = rgb_u8[
        window.row_off : window.row_off + vh, window.col_off : window.col_off + vw
    ]
    if vh == tile and vw == tile:
        return region
    pad2 = ((0, tile - vh), (0, tile - vw), (0, 0))
    return np.pad(region, pad2, mode="edge")


def predict_agl_tiled(
    model,
    rgb_u8: np.ndarray,
    device: str = "cpu",
    tile_size: int = TERRAHEIGHT_TILE_SIZE,
    overlap: int = TERRAHEIGHT_DEFAULT_OVERLAP,
    fp16: bool = False,
    batch_size: int = 1,
    should_cancel: Callable[[], bool] | None = None,
    progress_cb: Callable[[int, int], None] | None = None,
    log: Callable[[str], None] = print,
) -> tuple[np.ndarray, int]:
    """RGB uint8 [H,W,3] -> (AGL metres float32 [H,W], n_tiles).

    Tiled inference at the published 630x630 training crop:
      * overlapping windows (stride = tile_size - overlap) — full coverage,
        last window snapped to the raster edge (depthwizard.tiling);
      * each tile edge-padded up to the tile size when the scene is smaller;
      * per-tile forward (batch size 1 by default — verified ~330 MB peak
        VRAM per 630 tile on a 6 GB RTX 4050), cosine-blend stitched with
        OverlapStitcher so tiles never leave seams;
      * output grid == input grid, pixel-for-pixel (no stretching; the
        stitcher only ever accumulates each window's VALID region).

    ``fp16`` uses CUDA autocast for the tile forwards (halves activation
    memory; off by default — deterministic fp32 is the certified path).
    """
    import torch

    from .tifops import resolve_torch_device
    from .tiling import OverlapStitcher, TilingConfig, iter_tile_windows

    device = resolve_torch_device(device)
    if tile_size % DAV2_PATCH:
        raise ValueError(
            f"tile_size {tile_size} is not a multiple of {DAV2_PATCH} — "
            "TerraHeight cannot consume it (ViT patch constraint)."
        )
    if tile_size < DAV2_PATCH * 4:
        raise ValueError(f"tile_size {tile_size} is too small (min {DAV2_PATCH * 4})")
    if overlap < 0 or overlap >= tile_size:
        raise ValueError(f"overlap must be in [0, tile_size), got {overlap}")
    if batch_size != 1:
        raise NotImplementedError(
            "TerraHeight tiled inference is batch-1 (verified memory-safe "
            "on the 6 GB RTX 4050); larger batches are intentionally "
            "unsupported until VRAM is verified."
        )

    h, w = rgb_u8.shape[:2]
    cfg = TilingConfig(tile_size=tile_size, overlap=overlap)
    windows = iter_tile_windows(h, w, cfg)
    stitcher = OverlapStitcher(h, w, cfg)
    net = model.net
    net.eval()
    use_amp = bool(fp16 and device.startswith("cuda"))
    n_tiles = len(windows)
    log(
        f"[i] terraheight: {n_tiles} tile(s) of {tile_size} "
        f"(overlap {overlap}, stride {cfg.stride}) on {device}"
        + (" [fp16]" if use_amp else "")
    )

    done = 0
    for window in windows:
        if should_cancel is not None and should_cancel():
            from .inference import InferenceCancelled

            raise InferenceCancelled("cancelled between TerraHeight tiles")
        tile_rgb = _edge_pad_tile(rgb_u8, window, tile_size)
        x = torch.from_numpy(preprocess_rgb(np.ascontiguousarray(tile_rgb))[None]).to(
            device
        )
        with torch.no_grad(), torch.autocast(
            device_type="cuda", dtype=torch.float16, enabled=use_amp
        ):
            pred = net(x)["pred"][0, 0].float().cpu().numpy()
        if not np.isfinite(pred).all():
            raise RuntimeError(
                "TerraHeight produced non-finite output in a tile — the "
                "forward pass diverged. STOP and diff."
            )
        stitcher.add_tile(window, pred)
        done += 1
        if progress_cb is not None:
            progress_cb(done, n_tiles)
        if n_tiles > 1 and (done % 25 == 0 or done == n_tiles):
            log(f"[i] terraheight: tile {done}/{n_tiles}")

    agl = stitcher.finalize()
    if np.isnan(agl).any():
        raise RuntimeError("TerraHeight stitching left uncovered NaN pixels")
    # Published negative clamp (defensive: the DAv2 head already ReLUs).
    np.clip(agl, 0.0, None, out=agl)
    return agl, n_tiles


# ---------------------------------------------------------------------------
# Health check (small probe — never a full scene)
# ---------------------------------------------------------------------------


def terraheight_health(
    ckpt_path: Path | str | None = None,
    device: str = "cpu",
    project_root: Path | str | None = None,
) -> dict:
    """Verify checkpoint exists -> loads -> correct architecture -> one small
    forward with finite (NaN/Inf-free) output. Returns a JSON-serializable
    report; raises nothing (failures are reported in ``checks``)."""
    import torch

    report: Dict[str, Any] = {
        "architecture": ARCHITECTURE,
        "height_type": HEIGHT_TYPE,
        "checkpoint": None,
        "ok": False,
        "checks": {},
    }
    try:
        path = Path(ckpt_path) if ckpt_path else default_checkpoint_path(project_root)
        report["checkpoint"] = str(path)
        report["checks"]["checkpoint_exists"] = path.exists()
        if not path.exists():
            return report

        model = load_terraheight_model(path, device)
        report["checks"]["checkpoint_loads"] = True
        report["checks"]["architecture"] = model.architecture
        report["checks"]["parameter_count"] = int(
            sum(p.numel() for p in model.net.parameters())
        )
        report["checks"]["device"] = device

        # one small inference (154x154 = 11x14) — finite-output probe
        rng = np.random.default_rng(0)
        probe = rng.integers(0, 255, size=(TERRAHEIGHT_PROBE_SIZE, TERRAHEIGHT_PROBE_SIZE, 3), dtype=np.uint8)
        x = torch.from_numpy(preprocess_rgb(probe)[None]).to(device)
        with torch.no_grad():
            out = model.net(x)["pred"]
        out_np = out[0, 0].cpu().numpy()
        report["checks"]["probe_output_shape"] = list(out_np.shape)
        report["checks"]["output_finite"] = bool(np.isfinite(out_np).all())
        report["checks"]["output_nonnegative"] = bool((out_np >= 0).all())
        report["checks"]["output_range_m"] = [
            float(out_np.min()), float(out_np.max())
        ]
        report["ok"] = bool(
            report["checks"]["output_finite"] and report["checks"]["output_nonnegative"]
        )
    except Exception as e:  # noqa: BLE001 — health check reports, never raises
        report["checks"]["error"] = f"{type(e).__name__}: {e}"
        report["ok"] = False
    return report


# ---------------------------------------------------------------------------
# Orchestrator — one entry, used by CLI infer AND the FastAPI service
# (mirrors inference.run_inference minus the Dn/depth-cache stages)
# ---------------------------------------------------------------------------


def run_terraheight_inference(
    input_path: Path | str,
    ckpt_path: Path | str,
    *,
    out_dir: Path | str,
    device: str = "cpu",
    tile_size: int = TERRAHEIGHT_TILE_SIZE,
    overlap: int = TERRAHEIGHT_DEFAULT_OVERLAP,
    fp16: bool = False,
    anchor_dem: Path | str | None = None,
    ground_elev: float | None = None,
    write_files: bool = True,
    should_cancel: Callable[[], bool] | None = None,
    progress_cb: Callable[[int, int], None] | None = None,
) -> dict:
    """TerraHeight-S end-to-end run -> scene payload (build_scene_payload).

    Flow: read image -> tiled AGL inference (NO depth cache, NO dn) ->
    optional anchoring -> standard scene outputs (dsm.npy / dsm.tif /
    dsm_preview.png — the downstream terrain/inspection contract) PLUS
    terraheight_agl.tif / terraheight_preview.png / terraheight_meta.json
    provenance artifacts. Georeferencing is preserved exactly (CRS +
    transform of the source raster); a non-georeferenced input is reported
    honestly as pixel-space (never fabricated coordinates)."""
    import json

    import torch

    from .anchoring import ANCHORED_LABEL, anchor
    from .geo import pixel_size_metres
    from .inference import InferenceCancelled, read_image
    from .pipeline.scene_outputs import (
        build_scene_payload,
        compute_stats,
        georef_state,
        save_preview_png,
        write_outputs,
    )
    from .tifops import sha256_file

    t0 = time.perf_counter()
    input_path = Path(input_path)
    out_dir = Path(out_dir)

    model = load_terraheight_model(ckpt_path, device)
    params = int(sum(p.numel() for p in model.net.parameters()))
    scale_m = float(model.net.scale_m)

    rgb_u8, profile = read_image(input_path)
    georef, crs, tf = georef_state(profile)
    h, w = rgb_u8.shape[:2]
    print(
        f"[i] input {input_path.name}  {w}x{h}  georeferenced={georef} ({crs})"
    )
    print(f"[i] height model: TerraHeight-S ({ARCHITECTURE}) — direct AGL, no depth cache")

    def _cancelled() -> bool:
        return bool(should_cancel is not None and should_cancel())

    if _cancelled():
        raise InferenceCancelled("cancelled before inference started")

    infer_t0 = time.perf_counter()
    dsm, n_tiles = predict_agl_tiled(
        model,
        rgb_u8,
        device=device,
        tile_size=tile_size,
        overlap=overlap,
        fp16=fp16,
        should_cancel=should_cancel,
        progress_cb=progress_cb,
    )
    inference_sec = time.perf_counter() - infer_t0
    peak_vram_mb = None
    if device.startswith("cuda") and torch.cuda.is_available():
        peak_vram_mb = float(torch.cuda.max_memory_allocated()) / (1024**2)

    if _cancelled():
        raise InferenceCancelled("cancelled after AGL inference")

    stats = compute_stats(dsm)
    print(
        f"[stats] AGL (m): min {stats['min']:.2f}  mean {stats['mean']:.2f}  "
        f"median {stats['median']:.2f}  max {stats['max']:.2f}  "
        f"neg {stats['neg']}"
    )

    anchored = None
    if anchor_dem is not None or ground_elev is not None:
        tile_profile = {
            "crs": crs,
            "transform": tf,
            "height": h,
            "width": w,
        }
        anchored = anchor(dsm, anchor_dem, ground_elev, tile_profile)
        assert anchored is not None
        a_stats = compute_stats(anchored.dsm)
        print(
            f"[stats] ANCHORED DSM (m): min {a_stats['min']:.2f}  "
            f"mean {a_stats['mean']:.2f}  max {a_stats['max']:.2f}  "
            f"[{ANCHORED_LABEL} | source={anchored.source}]"
        )
    elif not georef:
        print(
            "[i] Input raster is not georeferenced — the TerraHeight result "
            "is a pixel-space AGL map. No CRS was invented."
        )

    outputs: dict[str, str | None] = {}
    meta: Dict[str, Any] = {
        "architecture": ARCHITECTURE,
        "height_type": HEIGHT_TYPE,
        "model": "TerraHeight-S",
        "backbone": "Depth Anything V2 Small (ViT-S)",
        "model_tag": model.tag,
        "checkpoint": str(ckpt_path),
        "checkpoint_sha256": sha256_file(ckpt_path),
        "parameter_count": params,
        "external_model": True,
        "trained_on": "GAMUS (published TerraHeight-S release; not trained by this project)",
        "license": "Apache-2.0 (TerraHeight); base weights Depth Anything V2 Small",
        "scale_m": scale_m,
        "output_units": "metres AGL (clamped to >= 0)",
        "input_dimensions": [h, w],
        "tile_size": tile_size,
        "tile_overlap": overlap,
        "tile_stride": tile_size - overlap,
        "batch_size": 1,
        "n_tiles": int(n_tiles),
        "device": device,
        "dtype": "float16 (autocast)" if fp16 else "float32",
        "inference_sec": round(inference_sec, 2),
        "peak_vram_mb": round(peak_vram_mb, 1) if peak_vram_mb is not None else None,
        "georeferenced": bool(georef),
        "georef_state": "georeferenced (CRS + transform preserved)" if georef else "pixel-space (no CRS on input — none fabricated)",
        "published_reference_metrics": (
            "Embedded validation metrics of the released checkpoint — "
            "PUBLISHED reference values, NOT this scene's accuracy"
        ),
    }
    if write_files:
        outputs = write_outputs(
            out_dir,
            dsm,
            profile,
            anchored,
            preview_title=f"TerraHeight-S AGL — {input_path.name}",
        )
        for k, v in outputs.items():
            if v:
                print(f"[out] {v}")

        # TerraHeight-specific provenance artifacts
        if georef:
            import rasterio

            agl_path = out_dir / "terraheight_agl.tif"
            with rasterio.open(
                agl_path,
                "w",
                driver="GTiff",
                height=h,
                width=w,
                count=1,
                dtype="float32",
                crs=crs,
                transform=tf,
                compress="deflate",
            ) as dst:
                dst.write(dsm.astype(np.float32), 1)
            outputs["terraheight_agl_tif"] = str(agl_path)
        else:
            agl_path = out_dir / "terraheight_agl.npy"
            np.save(agl_path, dsm.astype(np.float32))
            outputs["terraheight_agl_npy"] = str(agl_path)
        preview_path = out_dir / "terraheight_preview.png"
        save_preview_png(dsm, preview_path, f"TerraHeight-S AGL (m) — {input_path.name}")
        outputs["terraheight_preview"] = str(preview_path)

        # the checkpoint's own config block (architecture provenance)
        try:
            ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
            meta["model_config"] = ckpt.get("model_config")
            meta["normalization"] = ckpt.get("normalization")
            meta["training_crop"] = ckpt.get("crop_size")
            if isinstance(ckpt.get("validation_metrics"), dict):
                meta["published_validation_metrics"] = {
                    k: v for k, v in ckpt["validation_metrics"].items()
                    if k in ("all", ">1m", ">5m")
                }
        except Exception as e:  # noqa: BLE001 — provenance is best-effort
            meta["model_config"] = f"unavailable ({e})"

        meta_path = out_dir / "terraheight_meta.json"
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)
        outputs["terraheight_meta"] = str(meta_path)
        print(f"[out] {meta_path}")

    pixel_size_m = pixel_size_metres(crs, tf) if georef else None
    payload = build_scene_payload(
        dsm,
        rgb_u8,
        stem=input_path.stem,
        mode="tiles",
        dn_source="not_required (TerraHeight-S computes AGL directly from RGB)",
        model_tag=model.tag,
        device=device,
        profile=profile,
        anchored=anchored,
        outputs=outputs,
        elapsed_sec=time.perf_counter() - t0,
    )
    payload["meta"]["model_architecture"] = ARCHITECTURE
    payload["meta"]["height_type"] = HEIGHT_TYPE
    payload["meta"]["height_model_label"] = "TerraHeight-S"
    if anchored is not None:
        payload["meta"]["height_semantics"] = (
            f"{HEIGHT_TYPE} + ground datum = absolute DSM ({ANCHORED_LABEL})"
        )
    else:
        payload["meta"]["height_semantics"] = (
            f"{HEIGHT_TYPE} (above-ground height in metres, clamped >= 0 — "
            "NOT absolute terrain elevation)"
        )
    payload["terraheight"] = meta
    # Semantic layer: TerraHeight has NO semantic head (it is a height model);
    # the dedicated semantic segmentation pipeline owns land-cover semantics.
    payload["semantic"] = {
        "available": False,
        "reason": "TerraHeight-S has no semantic head — land-cover semantics "
                  "come from the dedicated semantic pipeline, not the height model",
    }
    return payload
