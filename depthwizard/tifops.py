"""Checkpoint loading + predict-function factory (inference code path).

This module is the SINGLE place that knows how to rebuild a height model
from a checkpoint. Both the ``infer`` CLI command and the FastAPI service
import ``load_height_model`` / ``make_predict_fn`` from here, so the webapp
can never drift from the certified CLI forward pass.

Architecture registry (RDAH integration):
    "calibration_net"  the frozen Phase-2 CalibrationNet path — rebuilt by
                       ``load_calib_net`` exactly as before (this module's
                       original, byte-identical behavior).
    "rdah"             the official RDAH-Net backend — built by
                       ``depthwizard.rdah.load_rdah_model`` (pretrained
                       HeightPredTransformer + adapter).
    ``load_height_model`` selects by explicit argument or by inspecting
    the checkpoint payload (CalibrationNet checkpoints carry
    ``model_state`` + ``use_rgb``; RDAH checkpoints carry
    ``model_state_dict``) — never a silent guess when both/neither match.

Checkpoint contract, CalibrationNet (written by the ``train`` command,
do not change):
    {
      "model_state":  OrderedDict — CalibrationNet state_dict,
      "use_rgb":      bool        -> in_ch 4 | 1,
      "use_sem":      bool (Exp4/5, default False) -> semantic one-hot block,
      "sem_classes":  int  (default 0, K=6 when semantic channels present),
      "sem_aux_head": bool (default False; aux semantic logits head),
      "widths":       [w1, w2, w3],
      "clamp_min":    float | missing -> 0.0,
      "affine_init":  {"a": a0, "b": b0},
      "epoch":        int, "loss": str, "val_subset_mae": float,
      "splits_json":  str, "created": iso-timestamp,
    }
Every new field is read via ckpt.get(...) with the pre-Exp-4 default, so
OLD checkpoints rebuild bit-identically (plan risk R2: contract drift).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Optional, cast

import numpy as np


@dataclass
class LoadedModel:
    """Everything the inference path needs, plus provenance for reporting."""

    net: "object"  # CalibrationNet | RDAH adapter (torch) — kept opaque here
    use_rgb: bool
    widths: tuple
    clamp_min: float
    affine_init: Dict[str, float]
    epoch: int
    checkpoint: Path
    val_subset_mae: Optional[float]
    use_sem: bool = False  # Exp 4/5 checkpoints (additive defaults)
    sem_classes: int = 0
    sem_aux_head: bool = False
    parameterization: str = "absolute_affine"
    bounded: bool = False
    context_module: str = "none"
    fusion_mode: str = "early"
    use_uncertainty: bool = False
    film_stats: bool = False  # Exp 1 FiLM checkpoints (additive default)
    # ---- RDAH integration (additive defaults — legacy checkpoints and
    # every existing caller keep working unchanged) ----
    architecture: str = "calibration_net"  # "calibration_net" | "rdah"
    height_type: str = "AGL"  # semantic role of net output (RDAH: "nDSM")
    depth_scale: float = 40.0  # rdah only: raw DAv2 depth x scale constant

    @property
    def tag(self) -> str:
        """Human-readable model id for logs / stats panels."""
        if self.architecture == "rdah":
            return f"rdah_{Path(self.checkpoint).stem}_ep{self.epoch}"
        return f"calib_{Path(self.checkpoint).parent.name}_ep{self.epoch}"

    @property
    def needs_stats(self) -> bool:
        """True when the forward call REQUIRES the dn_stats 4-vector.

        CalibrationNet needs it only for FiLM checkpoints (film_stats);
        the RDAH backend ALWAYS needs it (the RAW DAv2 scale is
        reconstructed from the log-stats — see depthwizard/rdah.py)."""
        return self.architecture == "rdah" or self.film_stats


def resolve_torch_device(device: Optional[str]) -> str:
    """Map 'auto'/None to a concrete torch device string ('cuda' if
    available else 'cpu'). torch.load(map_location='auto') is invalid, so
    every entry point that touches a checkpoint must resolve first."""
    if device in (None, "", "auto"):
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    return device


def sha256_file(path: Path | str) -> str:
    """SHA-256 hex digest of a file (checkpoint integrity verification)."""
    import hashlib

    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def validate_checkpoint_payload(ckpt: Any) -> None:
    """Fail fast on unexpected checkpoint contents.

    The training contract (see module docstring) is a plain dict of
    primitives + a state_dict. Anything else — arbitrary objects, missing
    required keys, non-tensor model weights — is rejected BEFORE any value
    is trusted, so a planted/corrupted checkpoint cannot smuggle payloads
    past the restricted loader.
    """
    from torch import Tensor

    required = ("model_state", "use_rgb", "widths", "affine_init", "epoch")
    if not isinstance(ckpt, dict):
        raise ValueError(
            f"checkpoint payload must be a dict, got {type(ckpt).__name__}"
        )
    missing = [k for k in required if k not in ckpt]
    if missing:
        raise ValueError(
            f"checkpoint is missing required keys: {missing} — not a "
            "CalibrationNet best.pt produced by `python model.py train`."
        )
    state = ckpt["model_state"]
    if not isinstance(state, dict) or not all(
        isinstance(k, str) and isinstance(v, Tensor) for k, v in state.items()
    ):
        raise ValueError(
            "checkpoint 'model_state' must be a mapping of parameter name "
            "-> torch.Tensor."
        )
    if not isinstance(ckpt["widths"], (list, tuple)) or not all(
        isinstance(w, int) for w in ckpt["widths"]
    ):
        raise ValueError("checkpoint 'widths' must be a list of ints.")
    affine = ckpt["affine_init"]
    if not isinstance(affine, dict) or not {"a", "b"} <= set(affine):
        raise ValueError("checkpoint 'affine_init' must contain 'a' and 'b'.")


def load_calib_net(ckpt_path: Path | str, device: str = "cpu") -> LoadedModel:
    """Rebuild the flagship CalibrationNet from a training checkpoint.

    Mirrors the exact construction used by the ``evaluate`` command (09):
    in_ch from the ablation flags, widths, affine-init biases, clamp_min.
    Any new field defaults exactly like 09 does — never invent values.

    Security: the checkpoint is loaded with ``weights_only=True`` (the
    training payload is state-dict + primitives, which satisfies the
    restricted loader) and schema-validated before any value is used.
    A pickle payload or malformed checkpoint raises instead of executing.
    """
    import torch

    device = resolve_torch_device(device)
    ckpt_path = Path(ckpt_path)
    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"checkpoint not found: {ckpt_path} — train first "
            f"(`python model.py train`) or pass --checkpoint."
        )

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    validate_checkpoint_payload(ckpt)

    from .calibration_net import (
        CalibrationNet,
        derive_in_ch,
        semantic_mode_from_ckpt,
    )

    use_rgb = bool(ckpt["use_rgb"])
    use_dem = bool(ckpt.get("use_dem", False))
    use_sem = bool(ckpt.get("use_sem", False))
    sem_classes = int(ckpt.get("sem_classes", 0))
    sem_aux_head = bool(ckpt.get("sem_aux_head", False))
    # V2 architecture metadata: absolute_affine / unbounded / no-context /
    # early fusion reproduce every legacy checkpoint exactly (the defaults
    # ARE the legacy design). A v2 field present in the checkpoint must be
    # honored — never silently rebuild an incompatible architecture.
    parameterization = str(ckpt.get("parameterization", "absolute_affine"))
    bounded = bool(ckpt.get("bounded", False))
    context_module = str(ckpt.get("context_module", "none"))
    fusion_mode = str(ckpt.get("fusion_mode", "early"))
    use_uncertainty = bool(ckpt.get("use_uncertainty", False))
    # in_ch is reconstructed deterministically from the flags (legacy
    # use_rgb/use_dem mapping preserved EXACTLY for old checkpoints):
    #   in_ch=1 (Dn) | 2 (Dn+DEM) | 4 (Dn+RGB) | 5 (Dn+RGB+DEM) | +K (sem).
    # The explicit ``in_ch`` field (Method-D and Exp-4/5 checkpoints) takes
    # precedence so any future variant is handled too.
    in_ch = ckpt.get("in_ch")
    if in_ch is None:
        in_ch = derive_in_ch(
            use_rgb=use_rgb, use_sem=use_sem, use_dem=use_dem, sem_classes=sem_classes
        )
    net = CalibrationNet(
        in_ch=int(in_ch),
        widths=tuple(ckpt["widths"]),
        a0=ckpt["affine_init"]["a"],
        b0=ckpt["affine_init"]["b"],
        clamp_min=ckpt.get("clamp_min", 0.0),
        sem_classes=sem_classes,
        sem_aux_head=sem_aux_head,
        # head-only checkpoints (predicted semantics) store sem_input=False;
        # legacy ckpts omit it and default to the GT-input design (True) so
        # they rebuild bit-identically.
        semantic_mode=semantic_mode_from_ckpt(ckpt),
        parameterization=parameterization,
        bounded=bounded,
        context_module=context_module,
        fusion_mode=fusion_mode,
        use_uncertainty=use_uncertainty,
        film_stats=bool(ckpt.get("film_stats", False)),
    ).to(device)
    net.load_state_dict(ckpt["model_state"])
    net.eval()

    subset_mae = ckpt.get("val_subset_mae")
    return LoadedModel(
        net=net,
        use_rgb=use_rgb,
        widths=tuple(ckpt["widths"]),
        clamp_min=ckpt.get("clamp_min", 0.0),
        affine_init={
            "a": float(ckpt["affine_init"]["a"]),
            "b": float(ckpt["affine_init"]["b"]),
        },
        epoch=int(ckpt["epoch"]),
        checkpoint=ckpt_path,
        val_subset_mae=(float(subset_mae) if subset_mae is not None else None),
        use_sem=use_sem,
        sem_classes=sem_classes,
        sem_aux_head=sem_aux_head,
        parameterization=parameterization,
        bounded=bounded,
        context_module=context_module,
        fusion_mode=fusion_mode,
        use_uncertainty=use_uncertainty,
        film_stats=bool(ckpt.get("film_stats", False)),
    )


# ---------------------------------------------------------------------------
# Architecture registry — calibration_net | rdah
# ---------------------------------------------------------------------------


def detect_architecture(ckpt_path: Path | str) -> str:
    """Inspect a checkpoint file and return its architecture string.

    Detection rules (no silent guesses — ambiguous files raise):
      * explicit ``architecture`` field wins (DepthWizard-saved ckpts);
      * ``model_state_dict`` (official RDAH key) -> "rdah";
      * ``model_state`` (CalibrationNet key)     -> "calibration_net";
      * anything else raises — the file is not a known height-model
        checkpoint and guessing would silently load garbage.
    """
    import torch

    ckpt_path = Path(ckpt_path)
    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"checkpoint not found: {ckpt_path} — train first "
            f"(`python model.py train`) or pass --checkpoint."
        )
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    if not isinstance(ckpt, dict):
        raise ValueError(
            f"checkpoint payload must be a dict, got {type(ckpt).__name__} "
            f"({ckpt_path})"
        )
    arch = ckpt.get("architecture")
    if arch in ("calibration_net", "rdah"):
        return str(arch)
    has_rdah = "model_state_dict" in ckpt
    has_calib = "model_state" in ckpt
    if has_rdah and not has_calib:
        return "rdah"
    if has_calib and not has_rdah:
        return "calibration_net"
    raise ValueError(
        f"cannot determine the architecture of {ckpt_path}: it carries "
        f"{'both' if has_rdah and has_calib else 'neither'} "
        "model_state/model_state_dict. Pass --architecture explicitly or "
        "use a checkpoint produced by `model.py train` / the official "
        "RDAH-Net release."
    )


def load_height_model(
    ckpt_path: Path | str,
    device: str = "cpu",
    architecture: Optional[str] = None,
    depth_scale: float = 40.0,
) -> LoadedModel:
    """THE height-model entry point: registry dispatch over architectures.

    ``architecture``: "calibration_net" | "rdah" | None (None = detect
    from the checkpoint payload — see detect_architecture). Both paths
    return a LoadedModel with the same surface (net / use_rgb / tag /
    needs_stats), so DepthWizardPredictor, the service and the CLI serve
    both backends from one code path.
    """
    arch = architecture or detect_architecture(ckpt_path)
    if arch not in ("calibration_net", "rdah"):
        raise ValueError(
            f"unknown architecture {arch!r} — expected 'calibration_net' "
            "or 'rdah'."
        )
    if arch == "rdah":
        from .rdah import load_rdah_model

        return load_rdah_model(ckpt_path, device=device, depth_scale=depth_scale)
    return load_calib_net(ckpt_path, device)


def make_predict_fn(model: LoadedModel, device: str = "cpu") -> Callable:
    """Return predict(dn, rgb, stats) -> np.ndarray[H,W] float32 metres.

    ``dn``  : np.float32 [H,W] in [0,1] (minmax_normalize output)
    ``rgb`` : np.uint8 [H,W,3]  OR None when the net is dn-only
    ``stats``: dn_tile_stats [4] of the same tile — REQUIRED for the RDAH
               backend (RAW-scale reconstruction), optional for FiLM
               CalibrationNet checkpoints, ignored otherwise.

    The ImageNet normalization lives HERE (not in callers) so the recipe
    exists in exactly one inference location per architecture:
        uint8 -> /255 -> (x - mean) / std          [Phase-2 audited contract]
    (identical for both backends — the official RDAH recipe matches).
    """
    if model.architecture == "rdah":
        from .rdah import make_rdah_predict_fn

        return make_rdah_predict_fn(model, device)

    import torch

    from .dataset import IMAGENET_MEAN, IMAGENET_STD

    net = cast(Any, model.net)
    net.eval()

    @torch.no_grad()
    def predict(
        dn: np.ndarray,
        rgb: Optional[np.ndarray] = None,
        stats: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        dn_t = torch.from_numpy(
            np.ascontiguousarray(dn, dtype=np.float32)[None, None]
        ).to(device)
        rgb_t = None
        if model.use_rgb:
            if rgb is None:
                raise ValueError(
                    "this checkpoint is Dn+RGB (use_rgb=True) but no RGB array "
                    "was supplied to the predict function"
                )
            rgb_f = rgb.astype(np.float32) / 255.0
            rgb_n = (rgb_f - IMAGENET_MEAN) / IMAGENET_STD
            rgb_t = torch.from_numpy(
                np.ascontiguousarray(rgb_n.transpose(2, 0, 1))[None]
            ).to(device)
        # FiLM checkpoints (Exp 1): stats is the RAW-tile 4-vector
        # (normalize.dn_tile_stats output) at the SAME granularity the dn
        # was normalized at. Omitted -> the net zero-fills and FLAGS it
        # (out["stats_zero_filled"]) rather than silently substituting a
        # default that looks like real data.
        stats_t = None
        if model.film_stats and stats is not None:
            stats_t = torch.from_numpy(
                np.asarray(stats, dtype=np.float32)[None]
            ).to(device)
        pred = net(dn_t, rgb_t, None, None, stats_t)["pred"][0, 0].cpu().numpy()
        return pred.astype(np.float32)

    return predict


def make_full_predict_fn(model: LoadedModel, device: str = "cpu") -> Callable:
    """Return predict_full(dn, rgb, stats) -> dict with EVERYTHING the model
    emits (architecture-agnostic surface; RDAH returns sem_probs=None —
    it has no semantic head, the post-processing stage degrades honestly).

    Keys: ``pred`` (float32 [H,W] metres — identical to make_predict_fn's
    output), ``sem_probs`` (float32 [K,H,W] softmax over the PREDICTED
    auxiliary semantic head, or None when the checkpoint has no aux head),
    ``sem_zero_filled`` (bool).

    Used by the post-processing pipeline (semantic boundary gating needs
    predicted class probabilities at deployment) and by the scene payload.
    The plain ``make_predict_fn`` above is unchanged and remains the
    certified minimal path.
    """
    if model.architecture == "rdah":
        from .rdah import make_rdah_full_predict_fn

        return make_rdah_full_predict_fn(model, device)

    import torch

    from .dataset import IMAGENET_MEAN, IMAGENET_STD

    net = cast(Any, model.net)
    net.eval()

    @torch.no_grad()
    def predict_full(
        dn: np.ndarray,
        rgb: Optional[np.ndarray] = None,
        stats: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        dn_t = torch.from_numpy(
            np.ascontiguousarray(dn, dtype=np.float32)[None, None]
        ).to(device)
        rgb_t = None
        if model.use_rgb:
            if rgb is None:
                raise ValueError(
                    "this checkpoint is Dn+RGB (use_rgb=True) but no RGB array "
                    "was supplied to the predict function"
                )
            rgb_f = rgb.astype(np.float32) / 255.0
            rgb_n = (rgb_f - IMAGENET_MEAN) / IMAGENET_STD
            rgb_t = torch.from_numpy(
                np.ascontiguousarray(rgb_n.transpose(2, 0, 1))[None]
            ).to(device)
        stats_t = None
        if model.film_stats and stats is not None:
            stats_t = torch.from_numpy(
                np.asarray(stats, dtype=np.float32)[None]
            ).to(device)
        out = net(dn_t, rgb_t, None, None, stats_t)
        pred = out["pred"][0, 0].cpu().numpy().astype(np.float32)
        sem_probs = None
        if out.get("sem_logits") is not None:
            sem_probs = (
                torch.softmax(out["sem_logits"][0], dim=0).cpu().numpy().astype(np.float32)
            )
        return {
            "pred": pred,
            "sem_probs": sem_probs,
            "sem_zero_filled": bool(out.get("sem_zero_filled", False)),
            "stats_zero_filled": bool(out.get("stats_zero_filled", False)),
        }

    return predict_full
