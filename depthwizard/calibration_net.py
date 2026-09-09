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
    (+ optional RGB / SEMANTIC one-hot / DEM ablation channels — see
    ``derive_in_ch``). The output stays a physically interpretable per-pixel
    affine remap of relative depth.
  * clamp(min=0): AGL truth is clamped >= 0 (clean_agl), so negative
    predictions are pure loss; the clamp lets the net express the
    median-like "ground = 0" behaviour that beat the baseline in Phase 1.

Channel order (FROZEN — train/eval/infer must agree):
    [ Dn (1) | RGB (3) | SEM one-hot (K) | DEM (1) ]
    The semantic block sits BETWEEN RGB and DEM. ``sem`` is the project
    one-hot [K,H,W] (K=6, datasets/semantics.py). When a checkpoint expects
    semantic channels but none are supplied at inference (no GT semantics
    exist on arbitrary images), the channels are ZERO-FILLED and the sample
    is flagged — the model stays evaluable (plan risk R8: semantics are
    privileged information during training).

Loss: masked L1 by default (matches MAE metric and floors), Huber optional
(--loss huber). Never pure L2 — Phase 1 showed L2's upward bias is fatal on
ground-dominated data. Additional configurable terms (gradient / edge-aware
    smoothness / semantic CE) live in depthwizard.losses — NOT here.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def derive_in_ch(
    use_rgb: bool = False,
    use_sem: bool = False,
    use_dem: bool = False,
    sem_classes: int = 6,
) -> int:
    """Input channel count from the ablation flags (single source).

    in_ch = 1 (Dn) + 3 (RGB) + K (semantic one-hot) + 1 (DEM)
    Legacy mappings stay exact: Dn=1, Dn+DEM=2, Dn+RGB=4, Dn+RGB+DEM=5.
    """
    return (
        1
        + (3 if use_rgb else 0)
        + (int(sem_classes) if use_sem else 0)
        + (1 if use_dem else 0)
    )


def _conv_block(cin: int, cout: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, padding=1, bias=False),
        nn.ReLU(inplace=True),
        nn.Conv2d(cout, cout, 3, padding=1, bias=False),
        nn.ReLU(inplace=True),
    )


class CalibrationNet(nn.Module):
    """Small U-Net: (Dn [, RGB [, SEM ]][, DEM]) -> per-pixel (a, b) -> H.

    Channel counts the constructor accepts (see ``derive_in_ch``):
      * ``in_ch=1``  — Dn only                       (frozen dn_only variant)
      * ``in_ch=4``  — Dn + RGB                       (frozen rgb_cos flagship)
      * ``in_ch=5``  — Dn + RGB + DEM                 (Method-D challenger)
      * ``in_ch=2``  — Dn + DEM (no RGB; reserved)
      * ``in_ch=7``  — Dn + RGB + SEM(6)              (Exp 4, semantic input)
      * ``in_ch=8``  — Dn + RGB + SEM(6) + DEM        (Exp 5, +DEM)
      * ``in_ch=1+K`` etc. — any combination via derive_in_ch

    ``sem_classes`` (default 0 = no semantic channels): the K of the one-hot
    block. When > 0 and ``sem`` is not passed to forward, the channels are
    ZERO-FILLED (flagged by the caller) so inference stays possible without
    GT semantics.

    ``sem_aux_head`` (default False): optional auxiliary conv head predicting
    K semantic logits from the decoder features — enables semantic CE
    supervision (depthwizard.losses) and predicted-semantics inference.
    OFF by default = state_dict identical to the pre-Exp-4 class (old
    checkpoints load unchanged).

    ~0.2 M params at widths (16, 32, 64) — trains on CPU in minutes per
    epoch subset, on any GPU in seconds.
    """

    def __init__(
        self,
        in_ch: int = 1,
        widths: Tuple[int, int, int] = (16, 32, 64),
        a0: float = 2.076463,
        b0: float = 4.110271,
        clamp_min: float = 0.0,
        sem_classes: int = 0,
        sem_aux_head: bool = False,
    ):
        super().__init__()
        self.clamp_min = clamp_min
        self.sem_classes = int(sem_classes)
        w1, w2, w3 = widths
        self.enc1 = _conv_block(in_ch, w1)
        self.enc2 = _conv_block(w1, w2)
        self.enc3 = _conv_block(w2, w3)
        self.pool = nn.MaxPool2d(2)
        self.up2 = nn.ConvTranspose2d(w3, w2, 2, stride=2)
        self.dec2 = _conv_block(2 * w2, w2)  # concat(w2 up, w2 skip) -> w2
        self.up1 = nn.ConvTranspose2d(w2, w1, 2, stride=2)
        self.dec1 = _conv_block(2 * w1, w1)  # concat(w1 up, w1 skip) -> w1
        self.head = nn.Conv2d(w1, 2, 3, padding=1)
        assert self.head.bias is not None
        nn.init.zeros_(self.head.weight)
        with torch.no_grad():
            self.head.bias.copy_(torch.tensor([a0, b0], dtype=torch.float32))
        self.sem_aux_head = None
        if sem_aux_head and self.sem_classes > 0:
            self.sem_aux_head = nn.Conv2d(w1, self.sem_classes, 1)

    def forward(
        self,
        dn: torch.Tensor,
        rgb: Optional[torch.Tensor] = None,
        dem: Optional[torch.Tensor] = None,
        sem: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
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
            if sem is not None:
                sem = sem[None]
        # Channel order is FIXED: Dn [1] | RGB [3] | SEM [K] | DEM [1].
        # in_ch at construction must match the number of channels actually
        # concatenated. Semantic channels expected but absent -> ZERO-FILL
        # (the model must remain evaluable without privileged GT semantics;
        # callers flag the zero-fill in their outputs — plan risk R8).
        sem_zero_filled = False
        if self.sem_classes > 0 and sem is None:
            sem = torch.zeros(
                dn.shape[0],
                self.sem_classes,
                *dn.shape[-2:],
                device=dn.device,
                dtype=dn.dtype,
            )
            sem_zero_filled = True
        parts = [dn]
        if rgb is not None:
            parts.append(rgb)
        if sem is not None:
            parts.append(sem)
        if dem is not None:
            parts.append(dem)
        x = dn if len(parts) == 1 else torch.cat(parts, dim=1)
        H, W = x.shape[-2:]
        pad_h = (4 - H % 4) % 4  # encoder pools twice -> pad to /4
        pad_w = (4 - W % 4) % 4
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h))
        s1 = self.enc1(x)  # [N,w1,H,W]
        s2 = self.enc2(self.pool(s1))  # [N,w2,H/2,W/2]
        b = self.enc3(self.pool(s2))  # [N,w3,H/4,W/4]
        y = self.up2(b)
        y = self.dec2(torch.cat([y, s2], dim=1))
        y = self.up1(y)
        y = self.dec1(torch.cat([y, s1], dim=1))
        params = self.head(y)[..., :H, :W]  # un-pad
        a = params[:, 0:1]
        b_ = params[:, 1:2]
        h = a * dn + b_
        if self.clamp_min is not None:
            h = torch.clamp(h, min=self.clamp_min)
        out = {
            "pred": h,
            "a": a,
            "b": b_,  # always [N,1,H,W]
            "sem_zero_filled": sem_zero_filled,
        }
        if self.sem_aux_head is not None:
            out["sem_logits"] = self.sem_aux_head(y[..., :H, :W])  # [N,K,H,W]
        return out


def masked_l1_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Mean |pred - target| over finite-target pixels. pred/target [N,1,H,W]."""
    valid = torch.isfinite(target)
    if valid.sum() == 0:
        return pred.sum() * 0.0
    return (pred - target).abs()[valid].mean()


def masked_huber_loss(
    pred: torch.Tensor, target: torch.Tensor, delta: float = 5.0
) -> torch.Tensor:
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
def predict_full_tile(
    net: CalibrationNet,
    dn: torch.Tensor,
    rgb: Optional[torch.Tensor] = None,
    device: str = "cpu",
) -> torch.Tensor:
    """Full-tile single-tile inference. dn: [1,H,W] or [1,1,H,W] on CPU.
    Returns pred as [H,W] torch tensor on CPU."""
    net.eval()
    was_training = net.training
    x = dn.to(device)  # forward auto-batches 3D
    r = rgb.to(device) if rgb is not None else None
    out = net(x, r)["pred"][0, 0].cpu()  # always [N,1,H,W] -> [H,W]
    if was_training:
        net.train()
    return out


def load_affine_init(path) -> Tuple[float, float]:
    """Read (a0, b0) from a global_affine.json for exact-baseline init."""
    import json

    with open(path, "r", encoding="utf-8") as f:
        bl = json.load(f)
    return float(bl["a"]), float(bl["b"])
