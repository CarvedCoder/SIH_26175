"""RDAH backend for DepthWizard — adapter, preprocessing, checkpoint loading.

Architecture registry (see tifops.load_height_model):
    architecture = "calibration_net"  -> existing CalibrationNet path (frozen)
    architecture = "rdah"             -> this module (default after integration)

Pipeline position (unchanged DepthWizard flow):
    RGB -> Depth Anything V2 (frozen) -> RAW relative depth
                                      -> RDAH (depth x40 + RGB) -> nDSM metres
                                      -> DEM anchoring -> absolute DSM

The adapter wraps the official HeightPredTransformer (depthwizard/rdah_net.py,
verbatim port) behind the SAME forward contract CalibrationNet exposes:

    out = net(dn, rgb, dem=None, sem=None, stats=None)   # -> {"pred": [B,1,H,W]}

so the training loop, the val monitor and the evaluate command run UNCHANGED
for both backends. ``dem``/``sem`` are accepted (contract compatibility) and
REJECTED with a loud error if actually supplied — RDAH has no such channels
and silently ignoring them would train a different model than reported.

--------------------------------------------------------------------------
OFFICIAL PREPROCESSING (reproduced EXACTLY; verified against the released
Track1 checkpoint — see the worklog / compatibility report):

RGB (identical to the CalibrationNet recipe — applied ONCE, by the caller):
    uint8 [H,W,3] -> /255 -> (x - ImageNet mean) / ImageNet std   [B,3,H,W]
    (official loaddata.py + paper Sec. "Input preprocessing and
    normalization"; uint16 orthos are first linearly normalized to the
    uint8 range per image — depthwizard.imgio already delivers uint8.)

DEPTH (DIFFERENT from CalibrationNet — do NOT reuse min-max [0,1]):
    raw Depth-Anything-V2 relative depth  x  RDAH_DEPTH_SCALE (default 40.0)
    -> float32 values in the ~[0, 255] range, fed DIRECTLY (NOT divided by
    255, NOT min-max normalized, NOT ImageNet-normalized).

    Derivation of the x40 constant (three independent lines of evidence):
      1. BatchNorm forensics on the released checkpoint's depth-encoder
         stem: the training-time depth input had mean ~116 and std ~24
         (ratio 0.21). Raw DAv2 ViT-B output on aerial tiles has mean
         ~2.5-2.9 and std ~0.5-0.6 (SAME ratio 0.21); per-tile min-max x255
         would give std ~42 (ratio ~0.37) — contradicting the checkpoint.
         => the input is raw DAv2 scaled by a FIXED constant ~40-46.
      2. Empirical sweep on 7 real GAMUS val tiles (DC/PHL): raw x {35..45}
         avg MAE 5.46-5.56 m vs per-tile min-max x255 6.41 m — the
         fixed-scale family clearly wins; 35-45 is flat (delta < 0.1 m).
      3. The released loader (nyu_transform.ToTensor) never divides
         rel_depth by 255 and the official files carry 0-255-range values.
    40.0 sits at the centre of the BN-consistent band and is exposed as
    ``rdah.depth_scale`` for A/B sweeps.

    DepthWizard's frozen dataset/eval contracts carry Dn as min-max
    normalized [0,1] PLUS the raw-tile stats 4-vector
    (depthwizard.normalize.dn_tile_stats = [log(min+eps), log(max+eps),
    log(range+eps), log(mean+eps)]). The RAW depth is reconstructed
    EXACTLY from those two (log/exp roundtrip error ~1e-6 relative):
        lo = exp(stats[0]) - 1e-3 ;  hi = exp(stats[1]) - 1e-3
        raw = lo + dn * (hi - lo)
    so no frozen contract changes anywhere (datasets, caches, splits).

OUTPUT (official semantics, verified):
    [B,1,H,W] nDSM height in METRES, unclamped (may go slightly negative
    on ground). The paper's "x500 fixed-point uint16" label encoding is an
    I/O detail of THEIR released dataset files; the released CHECKPOINT
    itself predicts metres (val SmoothL1 loss 1.205 is metre-consistent
    with the paper's Track1 MAE 1.54 m; parameter count 5,371,663 matches
    the paper's 5.37 M exactly). Height type is reported as "nDSM" in the
    inference payload — never labelled absolute terrain elevation.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import numpy as np

# ---------------------------------------------------------------------------
# Constants (single source for the RDAH backend)
# ---------------------------------------------------------------------------

#: raw DAv2 relative depth -> RDAH depth input = raw * RDAH_DEPTH_SCALE
RDAH_DEPTH_SCALE = 40.0

#: eps inside depthwizard.normalize.dn_tile_stats (log-stats roundtrip)
_DN_STATS_EPS = 1e-3

#: H, W must be divisible by this (BlockAttention windowing constraint)
RDAH_INPUT_DIVISIBILITY = 128

#: max H, W supported by the released checkpoints (PositionalEncoding buffer)
RDAH_MAX_INPUT = 1024

#: Released pretrained checkpoints (figshare article 31986864, MIT license,
#: "RDAH-Net checkpoints and Datasets"). The Track1 model is the appropriate
#: one for this project: trained on DFC2019 Track-1 (1024x1024, US urban
#: aerial) — the same imagery family DepthWizard's GAMUS/DFC pipeline uses.
RDAH_CHECKPOINTS = {
    "track1": {
        "filename": "rdah_track1_best_model.pth",
        "url": "https://ndownloader.figshare.com/files/63637257",
        "md5": "4fdd8769d2a05aee0ed40234aeceee09",
        "bytes": 65516400,
        "epoch": 48,
        "trained_on": "DFC2019-Track1 (1024x1024)",
    },
    "swiss": {
        "filename": "rdah_swiss_best_model.pth",
        "url": "https://ndownloader.figshare.com/files/63637254",
        "md5": "a7e8a7933d8190058d2e117e3573e3cb",
        "bytes": 65512688,
        "epoch": None,
        "trained_on": "Swiss (512x512)",
    },
    "hk": {
        "filename": "rdah_hk_best_model.pth",
        "url": "https://ndownloader.figshare.com/files/63637251",
        "md5": "e1e9334be7c005179e0dd0df14181c87",
        "bytes": 65512688,
        "epoch": None,
        "trained_on": "HK (512x512)",
    },
}

#: default checkpoint directory (relative to the repo root)
RDAH_CKPT_DIR = Path("checkpoints/rdah")

ARCHITECTURE = "rdah"
HEIGHT_TYPE = "nDSM"


# ---------------------------------------------------------------------------
# Preprocessing (the official recipes — implemented exactly once)
# ---------------------------------------------------------------------------


def rdah_depth_from_raw(raw: np.ndarray, scale: float = RDAH_DEPTH_SCALE) -> np.ndarray:
    """RAW DAv2 relative depth [H,W] -> RDAH depth input [H,W] float32.

    THE official conversion: raw * scale, values in the ~[0,255] range.
    No /255, no min-max, no ImageNet normalization — the released
    checkpoint's depth-encoder BN statistics were accumulated on exactly
    this representation (see module docstring for the derivation).
    """
    raw = np.asarray(raw, dtype=np.float32)
    return (raw * float(scale)).astype(np.float32)


def reconstruct_raw_from_stats(
    dn: np.ndarray, stats: np.ndarray, eps: float = _DN_STATS_EPS
) -> np.ndarray:
    """min-max normalized Dn [H,W] + dn_tile_stats [4] -> RAW depth [H,W].

    The inverse of depthwizard.normalize.minmax_normalize_with_stats:
        stats = [log(lo+eps), log(hi+eps), log(hi-lo+eps), log(mean+eps)]
        raw  = (exp(stats[0]) - eps) + dn * ((exp(stats[1]) - eps)
                                             - (exp(stats[0]) - eps))
    Exact up to the float32 log/exp roundtrip (~1e-6 relative). This is how
    the RDAH backend recovers the RAW DAv2 scale from the FROZEN dataset /
    eval / inference contracts without touching any of them.
    """
    lo = float(np.exp(float(stats[0])) - eps)
    hi = float(np.exp(float(stats[1])) - eps)
    dn = np.asarray(dn, dtype=np.float32)
    if hi - lo < 1e-6:  # flat tile — minmax returned 0.5 everywhere
        return np.full_like(dn, 0.5 * (lo + hi), dtype=np.float32)
    return (lo + dn * (hi - lo)).astype(np.float32)


def rdah_depth_from_normalized(
    dn: np.ndarray, stats: np.ndarray, scale: float = RDAH_DEPTH_SCALE
) -> np.ndarray:
    """(normalized Dn, dn_tile_stats) -> RDAH depth input (raw x scale).

    The single entry point used by BOTH the training adapter and the
    inference predict functions: reconstruct RAW (see
    reconstruct_raw_from_stats), then apply the official x-scale once.
    """
    return rdah_depth_from_raw(reconstruct_raw_from_stats(dn, stats), scale=scale)


# ---------------------------------------------------------------------------
# Adapter — CalibrationNet-compatible forward contract
# ---------------------------------------------------------------------------

_ADAPTER_CLS = None


def _adapter_class():
    """Build (once) the nn.Module adapter class — torch imported lazily so
    the preprocessing helpers above stay usable in torch-free contexts."""
    global _ADAPTER_CLS
    if _ADAPTER_CLS is None:
        import torch.nn as nn

        class _RDAHHeightModel(nn.Module):
            """Height-model backend: (dn, rgb, stats) -> nDSM metres.

            Mirrors CalibrationNet's forward signature so the training
            loop, ``val_subset_mae`` and the evaluate command run
            unchanged:

                out = net(dn, rgb, dem=None, sem=None, stats=None)
                out["pred"]  # [B,1,H,W] float32 nDSM metres

            Input contract (identical semantics to CalibrationNet's):
                dn    : [B,1,H,W] min-max normalized [0,1] per tile
                rgb   : [B,3,H,W] ImageNet-normalized RGB (REQUIRED — the
                        RDAH checkpoint always consumes RGB)
                stats : [B,4] dn_tile_stats of the SAME tile the dn was
                        normalized from (REQUIRED — the RAW scale is
                        reconstructed from it; missing stats is a hard
                        error, never a silent default)
                dem/sem: NOT supported by RDAH — supplying non-None raises.
            """

            architecture = ARCHITECTURE
            height_type = HEIGHT_TYPE

            def __init__(self, core=None, depth_scale: float = RDAH_DEPTH_SCALE):
                super().__init__()
                from .rdah_net import HeightPredTransformer

                self.net = core if core is not None else HeightPredTransformer()
                self.depth_scale = float(depth_scale)

            @property
            def use_rgb(self) -> bool:
                return True  # RDAH always consumes RGB

            def forward(self, dn, rgb=None, dem=None, sem=None, stats=None):
                import torch

                if dem is not None:
                    raise ValueError(
                        "RDAH backend has no DEM input channel — refusing "
                        "to silently drop it (train with architecture: "
                        "calibration_net for the Method-D DEM ablation)."
                    )
                if sem is not None:
                    raise ValueError(
                        "RDAH backend has no semantic input channels — "
                        "refusing to silently drop them (use architecture: "
                        "calibration_net with --use-sem)."
                    )
                if rgb is None:
                    raise ValueError(
                        "the RDAH checkpoint is Dn+RGB but no RGB tensor "
                        "was supplied to the forward call"
                    )
                if stats is None:
                    raise ValueError(
                        "RDAH requires the RAW-tile Dn statistics (dn_stats, "
                        "the 4-vector from depthwizard.normalize.dn_tile_stats) "
                        "to reconstruct the RAW Depth-Anything-V2 scale. The "
                        "frozen dataset/eval contracts always carry it — a "
                        "None here means a caller bypassed them; refusing to "
                        "guess a default scale silently."
                    )

                # unbatched 3-D input -> auto-batch (CalibrationNet parity)
                if dn.dim() == 3:
                    dn = dn[None]
                    rgb = rgb[None] if rgb.dim() == 3 else rgb
                    stats = stats[None] if stats.dim() == 1 else stats

                dn = dn.float()
                rgb = rgb.float()
                stats = stats.float()
                if stats.dim() == 1:  # [4] -> [B,4]
                    stats = stats[None]
                if stats.dim() == 2 and stats.shape[0] != dn.shape[0] and stats.shape[0] == 1:
                    stats = stats.expand(dn.shape[0], 4)

                # --- THE official depth preprocessing (module docstring) ---
                # reconstruct RAW from (normalized dn, log-stats), then x scale
                lo = torch.exp(stats[:, 0]) - _DN_STATS_EPS  # [B]
                hi = torch.exp(stats[:, 1]) - _DN_STATS_EPS  # [B]
                rng = (hi - lo).clamp(min=0.0)  # [B]
                raw = lo.view(-1, 1, 1) + dn[:, 0] * rng.view(-1, 1, 1)  # [B,H,W]
                # flat-tile guard (mirror of normalize.minmax_normalize's
                # 0.5-everywhere behaviour so train/infer stay consistent)
                flat = rng < 1e-6
                if bool(flat.any()):
                    mid = (0.5 * (lo + hi)).view(-1, 1, 1).expand_as(raw)
                    raw = torch.where(flat.view(-1, 1, 1), mid, raw)
                depth_repr = raw * self.depth_scale  # official: raw x scale

                pred = self.net(depth_repr[:, None].contiguous(), rgb)[:, 0]
                return {
                    "pred": pred[:, None].contiguous(),  # [B,1,H,W]
                    "height_type": HEIGHT_TYPE,
                    "model": ARCHITECTURE,
                }

        _ADAPTER_CLS = _RDAHHeightModel
    return _ADAPTER_CLS


def RDAHHeightModel(core=None, depth_scale: float = RDAH_DEPTH_SCALE):
    """Public constructor for the adapter (lazy torch import)."""
    return _adapter_class()(core=core, depth_scale=depth_scale)


# ---------------------------------------------------------------------------
# Checkpoint loading
# ---------------------------------------------------------------------------


def _strip_prefixes(state: Dict, prefixes=("module.", "net.", "model.")):
    """Normalize state-dict keys: strip DataParallel/module or wrapper
    prefixes when EVERY key carries them. Returns (clean_state, stripped)."""
    for p in prefixes:
        if state and all(k.startswith(p) for k in state):
            state = {k[len(p):]: v for k, v in state.items()}
            return state, p
    return state, None


def validate_rdah_payload(ckpt) -> None:
    """Fail fast on non-RDAH / malformed checkpoint payloads (mirrors
    tifops.validate_checkpoint_payload's security posture)."""
    if not isinstance(ckpt, dict):
        raise ValueError(
            f"RDAH checkpoint payload must be a dict, got {type(ckpt).__name__}"
        )
    if "model_state_dict" not in ckpt:
        raise ValueError(
            "RDAH checkpoint is missing 'model_state_dict' — not a "
            "HeightPredTransformer checkpoint in the official format "
            "({epoch, model_state_dict, optimizer_state_dict, loss})."
        )
    from torch import Tensor

    state = ckpt["model_state_dict"]
    if not isinstance(state, dict) or not all(
        isinstance(k, str) and isinstance(v, Tensor) for k, v in state.items()
    ):
        raise ValueError(
            "RDAH checkpoint 'model_state_dict' must be a mapping of "
            "parameter name -> torch.Tensor."
        )


def load_rdah_state_dict(net, state, source: str = "") -> None:
    """Strict load with prefix normalization + LOUD mismatch reporting.

    Never silently loads partially incompatible weights: any missing or
    unexpected key raises ValueError listing the offending keys.
    """
    state, stripped = _strip_prefixes(dict(state))
    if stripped:
        print(f"[rdah] stripped state-dict prefix '{stripped}' ({source})")
    result = net.load_state_dict(state, strict=False)
    missing, unexpected = list(result.missing_keys), list(result.unexpected_keys)
    if missing or unexpected:
        raise ValueError(
            f"Incompatible RDAH checkpoint ({source}): "
            f"{len(missing)} missing keys (e.g. {missing[:4]}) and "
            f"{len(unexpected)} unexpected keys (e.g. {unexpected[:4]}). "
            "Refusing to load partially matching weights — the architecture "
            "and the checkpoint must match exactly."
        )
    n_par = sum(p.numel() for p in net.parameters())
    print(
        f"[rdah] loaded {source or 'checkpoint'}  params={n_par:,}  "
        f"tensors={len(state)}"
    )


def ensure_rdah_checkpoint(
    path: Path | str | None = None, variant: str = "track1", verbose: bool = True
) -> Path:
    """Resolve the RDAH checkpoint path; download + MD5-verify when missing.

    Local path wins when the file exists. Otherwise the released checkpoint
    for ``variant`` is fetched from figshare into ``checkpoints/rdah/`` and
    its MD5 is verified against the published digest — a corrupt or
    mismatched download raises (never a silent partial load).
    """
    import hashlib

    path = Path(path) if path else RDAH_CKPT_DIR / RDAH_CHECKPOINTS[variant][
        "filename"
    ]
    if path.exists():
        return path

    meta = RDAH_CHECKPOINTS[variant]
    path.parent.mkdir(parents=True, exist_ok=True)
    url = meta["url"]
    if verbose:
        print(
            f"[rdah] checkpoint not found at {path} — downloading the "
            f"released {variant} model from figshare "
            f"({meta['bytes'] / 1e6:.1f} MB, MD5 {meta['md5']})"
        )
    import urllib.request

    try:
        urllib.request.urlretrieve(url, path)
    except Exception as e:  # noqa: BLE001 — report, then re-raise cleanly
        raise RuntimeError(
            f"failed to download the RDAH {variant} checkpoint from {url}: {e}. "
            "Place the file manually (figshare article 31986864) and re-run."
        ) from e

    h = hashlib.md5()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    digest = h.hexdigest()
    if digest != meta["md5"]:
        path.unlink(missing_ok=True)
        raise RuntimeError(
            f"downloaded RDAH checkpoint MD5 mismatch: got {digest}, "
            f"expected {meta['md5']} — file removed. Re-run to retry or "
            "place the file manually."
        )
    if verbose:
        print(f"[rdah] checkpoint downloaded + MD5 verified -> {path}")
    return path


def load_rdah_model(
    ckpt_path: Path | str,
    device: str = "cpu",
    depth_scale: float = RDAH_DEPTH_SCALE,
    download_variant: Optional[str] = "track1",
) -> object:
    """Build + load the RDAH adapter; returns a tifops.LoadedModel.

    Handles: CPU/GPU mapping, 'module.'-style prefixes, strict key
    validation (loud failures), checkpoint provenance printing.
    """
    import torch

    from .tifops import LoadedModel, resolve_torch_device

    device = resolve_torch_device(device)
    ckpt_path = ensure_rdah_checkpoint(
        ckpt_path, variant=download_variant or "track1"
    )
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    validate_rdah_payload(ckpt)

    from .rdah_net import HeightPredTransformer

    net_core = HeightPredTransformer()
    load_rdah_state_dict(net_core, ckpt["model_state_dict"], source=str(ckpt_path))
    net = RDAHHeightModel(core=net_core, depth_scale=depth_scale).to(device)
    net.eval()

    epoch = int(ckpt.get("epoch", -1))
    loss = ckpt.get("loss")
    # official release: 'loss' = float (val SmoothL1); DepthWizard saves:
    # 'loss' = loss NAME (str) + 'val_subset_mae' = float — read accordingly
    val_mae = ckpt.get("val_subset_mae")
    if not isinstance(val_mae, (int, float)):
        val_mae = loss if isinstance(loss, (int, float)) else None
    is_release = str(ckpt_path).startswith(str(RDAH_CKPT_DIR)) or (
        "pretrained_from" not in ckpt and "architecture" not in ckpt
    )
    provenance = "official release" if is_release else "fine-tuned checkpoint"
    loss_repr = (
        f"{loss:.4f}" if isinstance(loss, (int, float)) else str(loss or "n/a")
    )
    n_par = sum(p.numel() for p in net.parameters())
    print(
        f"[rdah] HeightPredTransformer ({provenance}: epoch {epoch}, "
        f"loss {loss_repr})  params={n_par:,}  device={device}  "
        f"depth_scale={depth_scale}"
    )
    return LoadedModel(
        net=net,
        use_rgb=True,
        widths=(32,),  # d_model — informational only
        clamp_min=None,  # RDAH does NOT clamp (may go slightly negative)
        affine_init={"a": float("nan"), "b": float("nan")},  # N/A for RDAH
        epoch=epoch,
        checkpoint=Path(ckpt_path),
        val_subset_mae=(float(val_mae) if isinstance(val_mae, (int, float)) else None),
        architecture=ARCHITECTURE,
        height_type=HEIGHT_TYPE,
        depth_scale=float(depth_scale),
    )


# ---------------------------------------------------------------------------
# Inference predict-function factories (mirror tifops.make_*_predict_fn)
# ---------------------------------------------------------------------------


def make_rdah_predict_fn(model, device: str = "cpu"):
    """predict(dn_n, rgb_u8, stats) -> np.ndarray [H,W] float32 metres.

    ``dn_n``  : [H,W] min-max normalized Dn (the frozen inference contract)
    ``rgb_u8``: [H,W,3] uint8 (ImageNet normalization applied HERE, once)
    ``stats`` : [4] dn_tile_stats of the same tile (REQUIRED)

    Same signature/semantics as tifops.make_predict_fn so
    DepthWizardPredictor serves both backends from one code path.
    """
    import torch

    from .dataset import IMAGENET_MEAN, IMAGENET_STD

    net = model.net
    net.eval()

    @torch.no_grad()
    def predict(dn, rgb, stats=None):
        if stats is None:
            raise ValueError(
                "RDAH predict requires the per-tile dn_stats 4-vector "
                "(min-max normalization discards the RAW DAv2 scale the "
                "RDAH input needs). The inference path always computes it "
                "alongside the normalized tile."
            )
        if rgb is None:
            raise ValueError(
                "the RDAH checkpoint is Dn+RGB but no RGB array was "
                "supplied to the predict function"
            )
        # RGB: official recipe, applied exactly ONCE here
        rgb_f = rgb.astype(np.float32) / 255.0
        rgb_n = (rgb_f - IMAGENET_MEAN) / IMAGENET_STD
        rgb_t = torch.from_numpy(
            np.ascontiguousarray(rgb_n.transpose(2, 0, 1))[None]
        ).to(device)
        dn_t = torch.from_numpy(
            np.ascontiguousarray(dn, dtype=np.float32)[None, None]
        ).to(device)
        stats_t = torch.from_numpy(
            np.asarray(stats, dtype=np.float32)[None]
        ).to(device)
        out = net(dn_t, rgb_t, None, None, stats_t)
        return out["pred"][0, 0].cpu().numpy().astype(np.float32)

    return predict


def make_rdah_full_predict_fn(model, device: str = "cpu"):
    """predict_full(dn_n, rgb_u8, stats) -> dict (post-processing contract).

    Keys mirror tifops.make_full_predict_fn: ``pred`` [H,W] metres,
    ``sem_probs`` (always None — RDAH has no semantic head; the
    post-processing stage degrades honestly, same as headless CalibNet
    checkpoints), ``sem_zero_filled`` False, ``stats_zero_filled`` False.
    """
    base = make_rdah_predict_fn(model, device)

    def predict_full(dn, rgb, stats=None):
        pred = base(dn, rgb, stats)
        return {
            "pred": pred,
            "sem_probs": None,
            "sem_zero_filled": False,
            "stats_zero_filled": False,
        }

    return predict_full
