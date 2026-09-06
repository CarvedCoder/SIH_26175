"""Checkpoint loading + predict-function factory (inference code path).

This module is the SINGLE place that knows how to rebuild a CalibrationNet
from a ``best.pt`` checkpoint. Both the ``infer`` CLI command and the FastAPI
service import ``load_calib_net`` / ``make_predict_fn`` from here, so the
webapp can never drift from the certified CLI forward pass (the same class of
drift that the Phase-2.5 smoke test guarded against).

Checkpoint contract (written by the ``train`` command, do not change):
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

    net: "object"                    # CalibrationNet (torch) — kept opaque here
    use_rgb: bool
    widths: tuple
    clamp_min: float
    affine_init: Dict[str, float]
    epoch: int
    checkpoint: Path
    val_subset_mae: Optional[float]
    use_sem: bool = False            # Exp 4/5 checkpoints (additive defaults)
    sem_classes: int = 0
    sem_aux_head: bool = False

    @property
    def tag(self) -> str:
        """Human-readable model id for logs / stats panels."""
        return f"calib_{Path(self.checkpoint).parent.name}_ep{self.epoch}"


def resolve_torch_device(device: Optional[str]) -> str:
    """Map 'auto'/None to a concrete torch device string ('cuda' if
    available else 'cpu'). torch.load(map_location='auto') is invalid, so
    every entry point that touches a checkpoint must resolve first."""
    if device in (None, "", "auto"):
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device


def load_calib_net(ckpt_path: Path | str, device: str = "cpu") -> LoadedModel:
    """Rebuild the flagship CalibrationNet from a training checkpoint.

    Mirrors the exact construction used by the ``evaluate`` command (09):
    in_ch from the ablation flags, widths, affine-init biases, clamp_min.
    Any new field defaults exactly like 09 does — never invent values.
    """
    import torch

    device = resolve_torch_device(device)
    ckpt_path = Path(ckpt_path)
    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"checkpoint not found: {ckpt_path} — train first "
            f"(`python model.py train`) or pass --checkpoint.")

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

    from .calibration_net import CalibrationNet, derive_in_ch

    use_rgb = bool(ckpt["use_rgb"])
    use_dem = bool(ckpt.get("use_dem", False))
    use_sem = bool(ckpt.get("use_sem", False))
    sem_classes = int(ckpt.get("sem_classes", 0))
    sem_aux_head = bool(ckpt.get("sem_aux_head", False))
    # in_ch is reconstructed deterministically from the flags (legacy
    # use_rgb/use_dem mapping preserved EXACTLY for old checkpoints):
    #   in_ch=1 (Dn) | 2 (Dn+DEM) | 4 (Dn+RGB) | 5 (Dn+RGB+DEM) | +K (sem).
    # The explicit ``in_ch`` field (Method-D and Exp-4/5 checkpoints) takes
    # precedence so any future variant is handled too.
    in_ch = ckpt.get("in_ch")
    if in_ch is None:
        in_ch = derive_in_ch(use_rgb=use_rgb, use_sem=use_sem,
                             use_dem=use_dem, sem_classes=sem_classes)
    net = CalibrationNet(
        in_ch=int(in_ch),
        widths=tuple(ckpt["widths"]),
        a0=ckpt["affine_init"]["a"],
        b0=ckpt["affine_init"]["b"],
        clamp_min=ckpt.get("clamp_min", 0.0),
        sem_classes=sem_classes,
        sem_aux_head=sem_aux_head,
    ).to(device)
    net.load_state_dict(ckpt["model_state"])
    net.eval()

    subset_mae = ckpt.get("val_subset_mae")
    return LoadedModel(
        net=net,
        use_rgb=use_rgb,
        widths=tuple(ckpt["widths"]),
        clamp_min=ckpt.get("clamp_min", 0.0),
        affine_init={"a": float(ckpt["affine_init"]["a"]),
                     "b": float(ckpt["affine_init"]["b"])},
        epoch=int(ckpt["epoch"]),
        checkpoint=ckpt_path,
        val_subset_mae=(float(subset_mae) if subset_mae is not None else None),
        use_sem=use_sem,
        sem_classes=sem_classes,
        sem_aux_head=sem_aux_head,
    )


def make_predict_fn(model: LoadedModel, device: str = "cpu") -> Callable:
    """Return predict(dn, rgb) -> np.ndarray[H,W] float32 metres.

    ``dn``  : np.float32 [H,W] in [0,1] (minmax_normalize output)
    ``rgb`` : np.uint8 [H,W,3]  OR None when the net is dn-only

    The ImageNet normalization lives HERE (not in callers) so the recipe
    exists in exactly one inference location:
        uint8 -> /255 -> (x - mean) / std          [Phase-2 audited contract]
    """
    import torch

    from .dataset import IMAGENET_MEAN, IMAGENET_STD

    net = cast(Any, model.net)
    net.eval()

    @torch.no_grad()
    def predict(dn: np.ndarray, rgb: Optional[np.ndarray] = None) -> np.ndarray:
        dn_t = torch.from_numpy(
            np.ascontiguousarray(dn, dtype=np.float32)[None, None]).to(device)
        rgb_t = None
        if model.use_rgb:
            if rgb is None:
                raise ValueError(
                    "this checkpoint is Dn+RGB (use_rgb=True) but no RGB array "
                    "was supplied to the predict function")
            rgb_f = rgb.astype(np.float32) / 255.0
            rgb_n = (rgb_f - IMAGENET_MEAN) / IMAGENET_STD
            rgb_t = torch.from_numpy(
                np.ascontiguousarray(rgb_n.transpose(2, 0, 1))[None]).to(device)
        pred = net(dn_t, rgb_t)["pred"][0, 0].cpu().numpy()
        return pred.astype(np.float32)

    return predict
