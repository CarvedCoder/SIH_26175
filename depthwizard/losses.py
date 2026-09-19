"""Configurable training losses for the calibration net (Phase 4 + V2).

POLICY (user spec, "minimal-first ablation"): every extra weight defaults to
0.0, in which case the composite loss is EXACTLY the pre-Phase-4 masked
L1/Huber loss (pinned by model_tests/test_losses.py::TestDefaultEquivalence).
Terms:

    main     (l1 | huber | berhu)  the height data term (l1/huber come from
                                   the frozen calibration_net implementations)
    w_grad   gradient loss  L1 between first-order image gradients of pred
                            and target — sharpens object boundaries
    w_smooth edge-aware     |grad pred| weighted by exp(-|grad rgb|):
                            smooth where RGB is flat, free at RGB edges
    w_sem    semantic CE    cross-entropy over the K=6 project classes,
                            against GT one-hot + ignore mask (requires the
                            net's auxiliary semantic head, --sem-aux-head)

V2 additions (all default OFF — enabled only via configs/v2_*.yaml):
    berhu               reverse-Huber (Laina et al. 2016): L1 below the
                        threshold c, quadratic above — penalizes large
                        (building) errors harder than forward Huber
    height_balanced     per-pixel weighting that upweights rare tall bins
    uncertainty_weight  heteroscedastic NLL on the net's log_var head
                        (Kendall & Gal 2017): 0.5*(e^2*exp(-s) + s)

Attribute note: DepthLoss accepts either the legacy LossConfig (w_grad /
w_smooth / w_sem names, pinned by tests) or the v2 LossWeightsConfig from
depthwizard.config (gradient_weight / boundary_weight / semantic_weight).
It resolves each term through both names so neither contract breaks.

Usage (train command flags -> LossConfig):
    --loss huber --w-grad 0.25 --w-smooth 0.1 --w-sem 0.2
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import torch
import torch.nn.functional as F

from depthwizard.config import LossWeightsConfig
from .calibration_net import masked_huber_loss, masked_l1_loss

# Reserved for a later additive implementation (documented, not invented).
SUPPORTED_WEIGHT_KEYS = (
    "w_grad",
    "w_slope",
    "w_smooth",
    "w_sem",
    "w_conf",
)


@dataclass
class LossConfig:
    main: str = "l1"  # "l1" | "huber" | "berhu"
    huber_delta: float = 5.0
    berhu_c: float = 0.0  # >0: fixed fraction of batch-max |err|; 0: adaptive 0.2
    w_grad: float = 0.0
    w_slope: float = 0.0
    w_smooth: float = 0.0
    w_sem: float = 0.0
    # w_conf: reserved (optional confidence weighting) — not implemented
    height_balanced: bool = False
    uncertainty_weight: float = 0.0

    @classmethod
    def from_train_cfg(cls, tcfg: dict) -> "LossConfig":
        """Build from the `train:` YAML section + CLI overrides handled by
        the caller (train command passes explicit values after merging)."""
        return cls(
            main=tcfg.get("loss", "l1"),
            huber_delta=float(tcfg.get("huber_delta", 5.0)),
            w_grad=float(tcfg.get("w_grad", 0.0)),
            w_slope=float(tcfg.get("w_slope", 0.0)),
            w_smooth=float(tcfg.get("w_smooth", 0.0)),
            w_sem=float(tcfg.get("w_sem", 0.0)),
            height_balanced=bool(tcfg.get("height_balanced", False)),
            uncertainty_weight=float(tcfg.get("uncertainty_weight", 0.0)),
        )


# ---------------------------------------------------------------------------
# Individual terms
# ---------------------------------------------------------------------------


def gradient_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """L1 between first-order finite-difference gradients (dy, dx).

    pred / target: [N,1,H,W] (channels-last dims differ for dy/dx, so the
    two directions are masked and averaged SEPARATELY — no padding, no
    shape gymnastics). Non-finite differences (LiDAR voids) are excluded.
    """
    if pred.shape != target.shape:
        raise ValueError(f"gradient_loss shape mismatch {pred.shape} vs {target.shape}")
    dy_p = pred[..., 1:, :] - pred[..., :-1, :]
    dy_t = target[..., 1:, :] - target[..., :-1, :]
    dx_p = pred[..., :, 1:] - pred[..., :, :-1]
    dx_t = target[..., :, 1:] - target[..., :, :-1]
    l_dy = (dy_p - dy_t).abs()
    l_dx = (dx_p - dx_t).abs()
    m_dy, m_dx = torch.isfinite(l_dy), torch.isfinite(l_dx)
    n = int(m_dy.sum() + m_dx.sum())
    if n == 0:
        return pred.sum() * 0.0
    total = l_dy[m_dy].sum() + l_dx[m_dx].sum()
    return total / n


def slope_angle_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    gsd_m: float = 0.33,
) -> torch.Tensor:
    """L1 loss between predicted and target slope angles in radians.

    Uses central differences, matching metrics.slope_error().
    GAMUS GSD = 0.33 m/pixel.
    """
    if gsd_m <= 0:
        raise ValueError("gsd_m must be > 0")

    # Central differences; exclude 1-pixel border just like evaluation.
    gy_p = (
        pred[..., 2:, 1:-1] - pred[..., :-2, 1:-1]
    ) / (2.0 * gsd_m)

    gx_p = (
        pred[..., 1:-1, 2:] - pred[..., 1:-1, :-2]
    ) / (2.0 * gsd_m)

    gy_t = (
        target[..., 2:, 1:-1] - target[..., :-2, 1:-1]
    ) / (2.0 * gsd_m)

    gx_t = (
        target[..., 1:-1, 2:] - target[..., 1:-1, :-2]
    ) / (2.0 * gsd_m)

    slope_p = torch.atan(
        torch.sqrt(gx_p.square() + gy_p.square() + 1e-8)
    )
    slope_t = torch.atan(
        torch.sqrt(gx_t.square() + gy_t.square() + 1e-8)
    )

    valid = torch.isfinite(slope_t) & torch.isfinite(slope_p)

    if not valid.any():
        return pred.sum() * 0.0

    return torch.abs(slope_p[valid] - slope_t[valid]).mean()


def edge_aware_smoothness(pred: torch.Tensor, rgb: torch.Tensor) -> torch.Tensor:
    """|grad pred| * exp(-|grad rgb|), averaged — smooth where RGB is flat,
    unconstrained at RGB edges (standard monocular-depth regularizer).

    pred: [N,1,H,W]; rgb: [N,3,H,W] (ImageNet-normalized or raw — only the
    RELATIVE gradient magnitude enters the exponential weight).
    """
    if pred.dim() != 4 or rgb.dim() != 4:
        raise ValueError("edge_aware_smoothness expects 4D [N,C,H,W] tensors")
    dy_p = (pred[..., 1:, :] - pred[..., :-1, :]).abs().mean(dim=1, keepdim=True)
    dx_p = (pred[..., :, 1:] - pred[..., :, :-1]).abs().mean(dim=1, keepdim=True)
    dy_r = (rgb[..., 1:, :] - rgb[..., :-1, :]).abs().mean(dim=1, keepdim=True)
    dx_r = (rgb[..., :, 1:] - rgb[..., :, :-1]).abs().mean(dim=1, keepdim=True)
    return (dy_p * torch.exp(-dy_r)).mean() + (dx_p * torch.exp(-dx_r)).mean()


def masked_semantic_ce(
    sem_logits: torch.Tensor,
    sem_target_onehot: torch.Tensor,
    sem_ignore: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Cross-entropy over the K project classes.

    sem_logits        [N,K,H,W] (aux head output, pre-softmax)
    sem_target_onehot [N,K,H,W] (GT one-hot; ignore pixels are all-zero)
    sem_ignore        [N,1,H,W] bool — excluded pixels (255 mapping below)

    Implemented as CE with ignore_index=255 by argmax-ing the one-hot.
    """
    if sem_logits.shape[0] != sem_target_onehot.shape[0]:
        raise ValueError("sem logits / target batch mismatch")
    target = sem_target_onehot.argmax(dim=1)  # [N,H,W]
    if sem_ignore is not None:
        target = target.masked_fill(sem_ignore[:, 0].bool(), 255)
    valid = target != 255
    if valid.sum() == 0:
        return sem_logits.sum() * 0.0
    return F.cross_entropy(sem_logits, target.clamp(max=255), ignore_index=255)


def masked_berhu_loss(
    pred: torch.Tensor, target: torch.Tensor, c: float = 0.0
) -> torch.Tensor:
    """Reverse Huber (BerHu, Laina et al. 2016): L1 for |e| <= c, quadratic
    above. c>0 fixes the threshold in metres; c=0 uses the common adaptive
    choice 0.2 * max|e| of the batch. Masked to finite-target pixels."""
    valid = torch.isfinite(target)
    if valid.sum() == 0:
        return pred.sum() * 0.0
    err = (pred - target).abs()[valid]
    if c <= 0:
        c = 0.2 * err.max().item()
    if c == 0:  # all errors exactly zero
        return pred.sum() * 0.0
    mask = err <= c
    l1 = err[mask]
    l2 = (err[~mask] ** 2 + c**2) / (2 * c)
    return (l1.sum() + l2.sum()) / valid.sum()


# ---------------------------------------------------------------------------
# Composite
# ---------------------------------------------------------------------------


class DepthLoss:
    """Composite loss; ``__call__`` returns the TOTAL tensor (backprop-able),
    and ``.last_parts`` carries the per-term breakdown for logging."""

    def __init__(self, cfg=None):
        self.cfg = cfg if cfg is not None else LossConfig()
        self.last_parts: Dict[str, float] = {}

    # -- weight resolution across the two config contracts ------------------
    def _w(self, v2_name: str, legacy_name: str) -> float:
        return float(getattr(self.cfg, v2_name, getattr(self.cfg, legacy_name, 0.0)) or 0.0)

    def main_term(self, pred, target) -> torch.Tensor:
        if self.cfg.main == "huber":
            return masked_huber_loss(pred, target, self.cfg.huber_delta)
        if self.cfg.main == "berhu":
            return masked_berhu_loss(pred, target, self.cfg.berhu_c)
        return masked_l1_loss(pred, target)

    def _get_balanced_weights(self, target: torch.Tensor) -> torch.Tensor:
        """Height-balanced weights: upweight rare tall bins. Kept mild so a
        few extreme pixels cannot dominate (0-10 m: 1x, 10-30 m: 2x, >30 m: 5x)."""
        valid = torch.isfinite(target)
        w = torch.ones_like(target)
        w[target > 10.0] = 2.0
        w[target > 30.0] = 5.0
        w[~valid] = 0.0
        return w

    def __call__(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        rgb: Optional[torch.Tensor] = None,
        sem_logits: Optional[torch.Tensor] = None,
        sem_target: Optional[torch.Tensor] = None,
        sem_ignore: Optional[torch.Tensor] = None,
        log_var: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        w_grad = self._w("gradient_weight", "w_grad")
        w_slope = self._w("w_slope", "w_slope")
        w_smooth = self._w("boundary_weight", "w_smooth")
        w_sem = self._w("semantic_weight", "w_sem")
        w_unc = self._w("uncertainty_weight", "uncertainty_weight")

        # main term, optionally height-balanced
        if getattr(self.cfg, "height_balanced", False):
            bw = self._get_balanced_weights(target)
            valid = torch.isfinite(target)
            if valid.sum() == 0:
                total = pred.sum() * 0.0
            else:
                err = (pred - target).abs()
                if self.cfg.main == "huber":
                    delta = self.cfg.huber_delta
                    quad = err < delta
                    per_px = torch.where(
                        quad, err, delta * (err - 0.5 * delta)
                    )
                else:
                    per_px = err
                total = (per_px * bw)[valid].mean()
        else:
            total = self.main_term(pred, target)

        # heteroscedastic NLL on the uncertainty head (L1-scale variant)
        if log_var is not None and w_unc > 0.0:
            valid = torch.isfinite(target)
            if valid.sum() > 0:
                err = (pred - target).abs()[valid]
                v = log_var.expand_as(pred)[valid]
                unc_loss = (err * torch.exp(-v) + v).mean()
                total = total + w_unc * unc_loss

        parts = {"main": float(total.detach())}
        if w_grad > 0.0:
            g = gradient_loss(pred, target)
            total = total + w_grad * g
            parts["grad"] = float(g.detach())
        if w_slope > 0.0:
            s = slope_angle_loss(pred, target, gsd_m=0.33)
            total = total + w_slope * s
            parts["slope"] = float(s.detach())
        if w_smooth > 0.0 and rgb is not None:
            s = edge_aware_smoothness(pred, rgb)
            total = total + w_smooth * s
            parts["smooth"] = float(s.detach())
        if w_sem > 0.0 and sem_logits is not None and sem_target is not None:
            c = masked_semantic_ce(sem_logits, sem_target, sem_ignore)
            total = total + w_sem * c
            parts["sem_ce"] = float(c.detach())
        self.last_parts = parts
        return total


def build_loss(cfg: Optional[LossConfig] = None) -> DepthLoss:
    return DepthLoss(cfg)
