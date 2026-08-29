"""Phase-2 spatial calibration network.

    H(x,y) = clamp( a(x,y) * Dn(x,y) + b(x,y),  min = clamp_min )

Why this parameterization (PS milestone: Scale Calibration):
  * The Phase-1 global affine H = a*Dn + b is the SPECIAL CASE where a and b
    are constants. The output conv is zero-weight-initialized with biases
    (a0, b0) from the frozen global_affine.json, so the network STARTS
    training EXACTLY at the baseline and can only climb from there. Ablation
    story writes itself: any val improvement is attributable to spatial
    variation of (a, b), not to architecture luck.
  * a(x, y), b(x, y) are free per-pixel fields from a small U-Net over Dn
    (+ optional RGB as ablation flag). The output stays a physically
    interpretable per-pixel affine remap of relative depth.
  * clamp(min=0): AGL truth is clamped >= 0 (clean_agl), so negative
    predictions are pure loss; the clamp lets the net express the
    median-like "ground = 0" behaviour that beat the baseline in Phase 1.

Loss: masked L1 by default (matches MAE metric and floors), Huber optional
(--loss huber). Never pure L2 — Phase 1 showed L2's upward bias is fatal on
ground-dominated data.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def _conv_block(cin: int, cout: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, padding=1, bias=False),
        nn.ReLU(inplace=True),
        nn.Conv2d(cout, cout, 3, padding=1, bias=False),
        nn.ReLU(inplace=True),
    )


class CalibrationNet(nn.Module):
    """Small U-Net: (Dn [, RGB [, DEM]]) -> per-pixel (a, b) -> H = clamp(a*Dn + b).

    Channel counts the constructor accepts:
      * ``in_ch=1``  — Dn only                       (frozen dn_only variant)
      * ``in_ch=4``  — Dn + RGB                       (frozen rgb_cos flagship)
      * ``in_ch=5``  — Dn + RGB + DEM                 (Method-D challenger; the
        DEM conditioning channel is the additive architecture change being
        ablated — its zero-weight head + a0,b0 bias init makes the variant
        start exactly at the flagship's numbers, per worklog Section 1)
      * ``in_ch=2``  — Dn + DEM (no RGB; reserved for a future ablation)

    ~0.2 M params at widths (16, 32, 64) — trains on CPU in minutes per
    epoch subset, on any GPU in seconds.
    """

    def __init__(self,
                 in_ch: int = 1,
                 widths: Tuple[int, int, int] = (16, 32, 64),
                 a0: float = 2.076463,
                 b0: float = 4.110271,
                 clamp_min: float = 0.0):
        super().__init__()
        self.clamp_min = clamp_min
        w1, w2, w3 = widths
        self.enc1 = _conv_block(in_ch, w1)
        self.enc2 = _conv_block(w1, w2)
        self.enc3 = _conv_block(w2, w3)
        self.pool = nn.MaxPool2d(2)
        self.up2 = nn.ConvTranspose2d(w3, w2, 2, stride=2)
        self.dec2 = _conv_block(2 * w2, w2)      # concat(w2 up, w2 skip) -> w2
        self.up1 = nn.ConvTranspose2d(w2, w1, 2, stride=2)
        self.dec1 = _conv_block(2 * w1, w1)      # concat(w1 up, w1 skip) -> w1
        self.head = nn.Conv2d(w1, 2, 3, padding=1)
        nn.init.zeros_(self.head.weight)
        with torch.no_grad():
            self.head.bias.copy_(torch.tensor([a0, b0], dtype=torch.float32))

    def forward(self, dn: torch.Tensor,
                rgb: Optional[torch.Tensor] = None,
                dem: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        # Accept BOTH [N,C,H,W] and unbatched [C,H,W] (dataset __getitem__
        # returns 3D; DataLoader adds the batch dim in training but direct
        # calls often forget). dim=1 cat on 3D would concat HEIGHT, not
        # channels — normalize before anything else.
        squeeze_back = dn.dim() == 3
        if squeeze_back:
            dn = dn[None]
            if rgb is not None:
                rgb = rgb[None]
            if dem is not None:
                dem = dem[None]
        # Channel order is fixed: Dn [1ch] | RGB [3ch] | DEM [1ch].
        # in_ch at construction must match the number of channels actually
        # passed: 1 (Dn only) | 4 (Dn+RGB) | 5 (Dn+RGB+DEM) | 2 (Dn+DEM).
        # The zero-weight head + a0,b0 bias init means the network starts
        # at EXACTLY the affine baseline regardless of in_ch — so the
        # Dn+RGB+DEM variant also starts at the current flagship's numbers
        # (worklog Section 1, "Affine init"; worklog Section 3, rule 4).
        parts = [dn]
        if rgb is not None:
            parts.append(rgb)
        if dem is not None:
            parts.append(dem)
        x = dn if len(parts) == 1 else torch.cat(parts, dim=1)
        H, W = x.shape[-2:]
        pad_h = (4 - H % 4) % 4                  # encoder pools twice -> pad to /4
        pad_w = (4 - W % 4) % 4
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h))
        s1 = self.enc1(x)                        # [N,w1,H,W]
        s2 = self.enc2(self.pool(s1))            # [N,w2,H/2,W/2]
        b  = self.enc3(self.pool(s2))            # [N,w3,H/4,W/4]
        y = self.up2(b)
        y = self.dec2(torch.cat([y, s2], dim=1))
        y = self.up1(y)
        y = self.dec1(torch.cat([y, s1], dim=1))
        params = self.head(y)[..., :H, :W]       # un-pad
        a = params[:, 0:1]
        b_ = params[:, 1:2]
        h = a * dn + b_
        if self.clamp_min is not None:
            h = torch.clamp(h, min=self.clamp_min)
        return {"pred": h, "a": a, "b": b_}   # always [N,1,H,W]


def masked_l1_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Mean |pred - target| over finite-target pixels. pred/target [N,1,H,W]."""
    valid = torch.isfinite(target)
    if valid.sum() == 0:
        return pred.sum() * 0.0
    return (pred - target).abs()[valid].mean()


def masked_huber_loss(pred: torch.Tensor, target: torch.Tensor,
                      delta: float = 5.0) -> torch.Tensor:
    """Huber: L1 for |e| > delta, L2 below — gentler on tall structures than
    pure L1 while keeping the ground behaviour. delta in metres."""
    valid = torch.isfinite(target)
    if valid.sum() == 0:
        return pred.sum() * 0.0
    e = (pred - target).abs()[valid]
    lin = delta * (e - 0.5 * delta)
    quad = 0.5 * e * e
    return torch.where(e > delta, lin, quad).mean()


@torch.no_grad()
def predict_full_tile(net: CalibrationNet, dn: torch.Tensor,
                      rgb: Optional[torch.Tensor] = None,
                      device: str = "cpu") -> torch.Tensor:
    """Full-tile single-tile inference. dn: [1,H,W] or [1,1,H,W] on CPU.
    Returns pred as [H,W] torch tensor on CPU."""
    net.eval()
    was_training = net.training
    x = dn.to(device)                            # forward auto-batches 3D
    r = rgb.to(device) if rgb is not None else None
    out = net(x, r)["pred"][0, 0].cpu()          # always [N,1,H,W] -> [H,W]
    if was_training:
        net.train()
    return out


def load_affine_init(path) -> Tuple[float, float]:
    """Read (a0, b0) from a global_affine.json for exact-baseline init."""
    import json
    with open(path, "r", encoding="utf-8") as f:
        bl = json.load(f)
    return float(bl["a"]), float(bl["b"])
