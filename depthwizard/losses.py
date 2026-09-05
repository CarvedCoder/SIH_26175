"""Configurable training losses for the calibration net (Phase 4).

POLICY (user spec, "minimal-first ablation"): every extra weight defaults to
0.0, in which case the composite loss is EXACTLY the pre-Phase-4 masked
L1/Huber loss (pinned by test_losses.py::test_default_weights_equivalence).
Terms:

    main     (l1 | huber)   the height data term (single source: the frozen
                            calibration_net.masked_l1_loss / masked_huber_loss)
    w_grad   gradient loss  L1 between first-order image gradients of pred
                            and target — sharpens object boundaries
    w_smooth edge-aware     |grad pred| weighted by exp(-|grad rgb|):
                            smooth where RGB is flat, free at RGB edges
    w_sem    semantic CE    cross-entropy over the K=6 project classes,
                            against GT one-hot + ignore mask (requires the
                            net's auxiliary semantic head, --sem-aux-head)

Usage (train command flags -> LossConfig):
    --loss huber --w-grad 0.25 --w-smooth 0.1 --w-sem 0.2

Confidence/uncertainty weighting is deliberately NOT implemented yet
(minimal-first); the config knob is reserved and documented here so adding
it later is additive.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional

import torch
import torch.nn.functional as F

from .calibration_net import masked_huber_loss, masked_l1_loss

# Reserved for a later additive implementation (documented, not invented).
SUPPORTED_WEIGHT_KEYS = ("w_grad", "w_smooth", "w_sem", "w_conf")


@dataclass
class LossConfig:
    main: str = "l1"                 # "l1" | "huber"
    huber_delta: float = 5.0
    w_grad: float = 0.0
    w_smooth: float = 0.0
    w_sem: float = 0.0
    # w_conf: reserved (optional confidence weighting) — not implemented

    def from_train_cfg(tcfg: dict) -> "LossConfig":
        """Build from the `train:` YAML section + CLI overrides handled by
        the caller (train command passes explicit values after merging)."""
        return LossConfig(
            main=tcfg.get("loss", "l1"),
            huber_delta=float(tcfg.get("huber_delta", 5.0)),
            w_grad=float(tcfg.get("w_grad", 0.0)),
            w_smooth=float(tcfg.get("w_smooth", 0.0)),
            w_sem=float(tcfg.get("w_sem", 0.0)))


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
        raise ValueError(f"gradient_loss shape mismatch {pred.shape} vs "
                         f"{target.shape}")
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


def masked_semantic_ce(sem_logits: torch.Tensor,
                       sem_target_onehot: torch.Tensor,
                       sem_ignore: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Cross-entropy over the K project classes.

    sem_logits        [N,K,H,W] (aux head output, pre-softmax)
    sem_target_onehot [N,K,H,W] (GT one-hot; ignore pixels are all-zero)
    sem_ignore        [N,1,H,W] bool — excluded pixels (255 mapping below)

    Implemented as CE with ignore_index=255 by argmax-ing the one-hot.
    """
    if sem_logits.shape[0] != sem_target_onehot.shape[0]:
        raise ValueError("sem logits / target batch mismatch")
    target = sem_target_onehot.argmax(dim=1)                    # [N,H,W]
    if sem_ignore is not None:
        target = target.masked_fill(sem_ignore[:, 0].bool(), 255)
    valid = target != 255
    if valid.sum() == 0:
        return sem_logits.sum() * 0.0
    return F.cross_entropy(sem_logits, target.clamp(max=255), ignore_index=255)


# ---------------------------------------------------------------------------
# Composite
# ---------------------------------------------------------------------------

class DepthLoss:
    """Composite loss; ``__call__`` returns the TOTAL tensor (backprop-able),
    and ``.last_parts`` carries the per-term breakdown for logging."""

    def __init__(self, cfg: LossConfig = None):
        self.cfg = cfg or LossConfig()
        self.last_parts: Dict[str, float] = {}

    def main_term(self, pred, target) -> torch.Tensor:
        if self.cfg.main == "huber":
            return masked_huber_loss(pred, target, self.cfg.huber_delta)
        return masked_l1_loss(pred, target)

    def __call__(self, pred: torch.Tensor, target: torch.Tensor,
                 rgb: Optional[torch.Tensor] = None,
                 sem_logits: Optional[torch.Tensor] = None,
                 sem_target: Optional[torch.Tensor] = None,
                 sem_ignore: Optional[torch.Tensor] = None) -> torch.Tensor:
        total = self.main_term(pred, target)
        parts = {"main": float(total.detach())}
        if self.cfg.w_grad > 0.0:
            g = gradient_loss(pred, target)
            total = total + self.cfg.w_grad * g
            parts["grad"] = float(g.detach())
        if self.cfg.w_smooth > 0.0 and rgb is not None:
            s = edge_aware_smoothness(pred, rgb)
            total = total + self.cfg.w_smooth * s
            parts["smooth"] = float(s.detach())
        if self.cfg.w_sem > 0.0 and sem_logits is not None \
                and sem_target is not None:
            c = masked_semantic_ce(sem_logits, sem_target, sem_ignore)
            total = total + self.cfg.w_sem * c
            parts["sem_ce"] = float(c.detach())
        self.last_parts = parts
        return total


def build_loss(cfg: LossConfig = None) -> DepthLoss:
    return DepthLoss(cfg)
