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



class ASPP_Lite(nn.Module):
    def __init__(self, in_c: int, out_c: int):
        super().__init__()
        branch_c = max(16, out_c // 4)
        
        self.b1 = nn.Conv2d(in_c, branch_c, kernel_size=1, bias=False)
        self.b2 = nn.Conv2d(in_c, branch_c, kernel_size=3, padding=2, dilation=2, bias=False)
        self.b3 = nn.Conv2d(in_c, branch_c, kernel_size=3, padding=4, dilation=4, bias=False)
        
        self.b4 = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_c, branch_c, kernel_size=1, bias=False)
        )
        
        self.project = nn.Sequential(
            nn.Conv2d(branch_c * 4, out_c, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h1 = self.b1(x)
        h2 = self.b2(x)
        h3 = self.b3(x)
        
        h4 = self.b4(x)
        h4 = F.interpolate(h4, size=x.shape[2:], mode='bilinear', align_corners=False)
        
        out = torch.cat([h1, h2, h3, h4], dim=1)
        return self.project(out)

class TileStatsFiLM(nn.Module):
    """FiLM conditioning from RAW-tile Dn statistics (Exp 1, Perez et al. 2018).

    A 2-layer MLP (4 -> hidden -> 2*sum(widths)) maps dn_tile_stats(raw) —
    [log min, log max, log range, log mean] of the PRE-normalization tile,
    the exact information per-tile min-max normalization discards — to
    per-channel (gamma, beta) pairs, one pair per conditioned U-Net level
    (levels 2..depth), applied to the encoder-stage outputs AND the matching
    decoder-stage outputs:
        feature = feature * gamma + beta   (broadcast spatially)

    Identity init (the invariant that preserves the exact-baseline start):
    the FINAL linear layer is zero-weight with bias [1]*w + [0]*w per level,
    so gamma=1, beta=0 exactly at initialization and the network computes
    its old function bit-for-bit until the MLP weights move.
    Params (hidden=32, widths=[32,64]): 4*32+32 + 32*192+192 = 6,496 (~6.3k).
    """

    def __init__(self, widths, hidden: int = 32):
        super().__init__()
        self.widths = tuple(int(w) for w in widths)
        out_dim = 2 * sum(self.widths)
        self.mlp = nn.Sequential(
            nn.Linear(4, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, out_dim),
        )
        final = self.mlp[-1]
        assert isinstance(final, nn.Linear)
        nn.init.zeros_(final.weight)
        with torch.no_grad():
            bias: list[float] = []
            for w in self.widths:
                bias.extend([1.0] * w + [0.0] * w)
            final.bias.copy_(torch.tensor(bias, dtype=torch.float32))

    def forward(self, stats: torch.Tensor):
        """stats [N,4] (dn_tile_stats output) -> (gammas, betas): lists of
        [N,w] tensors, one per conditioned level, ready for spatial
        broadcast."""
        out = self.mlp(stats)
        gammas, betas = [], []
        off = 0
        for w in self.widths:
            gammas.append(out[:, off : off + w])
            off += w
        for w in self.widths:
            betas.append(out[:, off : off + w])
            off += w
        return gammas, betas



class CalibrationNet(nn.Module):
    """V2 Modular CalibrationNet (Dn [, RGB [, SEM ]][, DEM]) -> H."""
    def __init__(
        self,
        in_ch: int = 1,
        widths: Tuple[int, ...] = (16, 32, 64),
        a0: float = 2.076463,
        b0: float = 4.110271,
        clamp_min: float = 0.0,
        sem_classes: int = 0,
        sem_aux_head: bool = False,
        semantic_mode: str = "input",
        parameterization: str = "absolute_affine",
        bounded: bool = False,
        max_shift: float = 10.0,
        fusion_mode: str = "early",
        context_module: str = "none",
        use_uncertainty: bool = False,
        film_stats: bool = False,
    ):
        super().__init__()
        self.clamp_min = clamp_min
        # Frozen global-affine anchor: residual parameterizations are
        # a = a0 + Δa, b = b0 + Δb, so the checkpoint's affine_init values
        # (not hardcoded constants) define the exact-baseline start.
        self.a0 = float(a0)
        self.b0 = float(b0)
        self.sem_classes = int(sem_classes)
        self.semantic_mode = semantic_mode
        self.sem_input = (self.semantic_mode == "input")
        self.parameterization = parameterization
        self.bounded = bounded
        self.max_shift = float(max_shift)
        self.fusion_mode = fusion_mode
        self.context_module = context_module
        self.use_uncertainty = use_uncertainty
        self.widths = tuple(widths)
        self.depth = len(self.widths)
        # FiLM tile-stat conditioning (Exp 1): condition enc2+ level outputs
        # on the RAW-tile Dn statistics. OFF by default = state_dict identical
        # to the pre-FiLM class (old checkpoints load unchanged). Identity
        # init preserves the exact-baseline start.
        self.film_stats = bool(film_stats)
        self.film = TileStatsFiLM(self.widths[1:]) if self.film_stats else None
        
        if self.fusion_mode == "dual_encoder":
            other_ch = in_ch - 1
            if other_ch <= 0:
                raise ValueError("dual_encoder requires >1 input channels")
            self.dn_enc = nn.ModuleList()
            self.rgb_enc = nn.ModuleList()
            
            c_dn = 1
            c_other = other_ch
            for i in range(self.depth):
                w = self.widths[i]
                self.dn_enc.append(_conv_block(c_dn, w))
                self.rgb_enc.append(_conv_block(c_other, w))
                c_dn = w
                c_other = w
                
            self.pool = nn.MaxPool2d(2)
            bottleneck_ch = self.widths[-1] * 2
            self.fusion_conv = _conv_block(bottleneck_ch, self.widths[-1])
        else:
            c_in = in_ch
            for i in range(1, self.depth + 1):
                w = self.widths[i-1]
                setattr(self, f"enc{i}", _conv_block(c_in, w))
                c_in = w
            self.pool = nn.MaxPool2d(2)

        if self.context_module == "aspp":
            self.context = ASPP_Lite(self.widths[-1], self.widths[-1])
        else:
            self.context = nn.Identity()

        for i in range(self.depth - 1, 0, -1):
            w_in = self.widths[i]
            w_out = self.widths[i-1]
            setattr(self, f"up{i}", nn.ConvTranspose2d(w_in, w_out, 2, stride=2))
            setattr(self, f"dec{i}", _conv_block(2 * w_out, w_out))
            
        head_in = self.widths[0]
        
        if self.parameterization == "absolute_affine":
            head_out = 2
        elif self.parameterization == "residual_affine":
            head_out = 2
        elif self.parameterization == "hybrid_residual":
            head_out = 3
        elif self.parameterization == "residual_depth":
            head_out = 1
        else:
            raise ValueError(f"Unknown parameterization {self.parameterization}")
            
        self.head = nn.Conv2d(head_in, head_out, 3, padding=1)
        nn.init.zeros_(self.head.weight)
        
        with torch.no_grad():
            if self.parameterization == "absolute_affine":
                self.head.bias.copy_(torch.tensor([a0, b0], dtype=torch.float32))
            else:
                self.head.bias.fill_(0.0)
                
        self.sem_aux_head = None
        if sem_aux_head and self.sem_classes > 0:
            self.sem_aux_head = nn.Conv2d(head_in, self.sem_classes, 1)

        self.joint_proj = None
        if self.semantic_mode == "joint" and self.sem_classes > 0:
            self.joint_proj = nn.Conv2d(head_in + self.sem_classes, head_in, 3, padding=1)

        self.unc_head = None

        if self.use_uncertainty:
            self.unc_head = nn.Conv2d(head_in, 1, 3, padding=1)

    def forward(
        self,
        dn: torch.Tensor,
        rgb: Optional[torch.Tensor] = None,
        dem: Optional[torch.Tensor] = None,
        sem: Optional[torch.Tensor] = None,
        stats: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        squeeze_back = dn.dim() == 3
        if squeeze_back:
            dn = dn[None]
            if rgb is not None:
                rgb = rgb[None]
            if dem is not None:
                dem = dem[None]
            if sem is not None:
                sem = sem[None]
            if stats is not None and stats.dim() == 1:
                stats = stats[None]

        sem_zero_filled = False
        if self.sem_input and self.sem_classes > 0 and sem is None:
            sem = torch.zeros(
                dn.shape[0], self.sem_classes, *dn.shape[-2:],
                device=dn.device, dtype=dn.dtype
            )
            sem_zero_filled = True

        # FiLM statistics (Exp 1): honest degradation mirrors the semantic
        # zero-fill — a missing stats vector is ZERO-FILLED and FLAGGED in
        # the output dict; every real caller (train/eval/infer) computes it
        # from the same raw tile minmax_normalize consumed, so the flag
        # should never fire in practice.
        stats_zero_filled = False
        film_apply = None
        if self.film is not None:
            if stats is None:
                stats = torch.zeros(
                    dn.shape[0], 4, device=dn.device, dtype=dn.dtype
                )
                stats_zero_filled = True
            gammas, betas = self.film(stats)

            def film_apply(level: int, feat: torch.Tensor) -> torch.Tensor:
                # level: 0-based index into the conditioned levels
                # (0 -> widths[1]); identity at init (gamma=1, beta=0).
                return (
                    feat * gammas[level][..., None, None]
                    + betas[level][..., None, None]
                )

        parts = []
        if rgb is not None: parts.append(rgb)
        if sem is not None and self.sem_input: parts.append(sem)
        if dem is not None: parts.append(dem)
        
        H, W = dn.shape[-2:]
        pad_h = (2**(self.depth-1) - H % (2**(self.depth-1))) % (2**(self.depth-1))
        pad_w = (2**(self.depth-1) - W % (2**(self.depth-1))) % (2**(self.depth-1))
        
        def do_pad(t):
            if pad_h or pad_w:
                return F.pad(t, (0, pad_w, 0, pad_h))
            return t
        
        if self.fusion_mode == "dual_encoder":
            x_dn = do_pad(dn)
            x_other = do_pad(torch.cat(parts, dim=1))
            skips_dn = []
            skips_other = []
            
            for i in range(self.depth):
                x_dn = self.dn_enc[i](x_dn)
                x_other = self.rgb_enc[i](x_other)
                if i < self.depth - 1:
                    skips_dn.append(x_dn)
                    skips_other.append(x_other)
                    x_dn = self.pool(x_dn)
                    x_other = self.pool(x_other)
            
            b = self.fusion_conv(torch.cat([x_dn, x_other], dim=1))
            skips = [skips_dn[i] + skips_other[i] for i in range(len(skips_dn))]
            if film_apply is not None:
                # skip j (0-based) has width widths[j]; condition from j=1 up
                skips = [
                    film_apply(j - 1, s) if j >= 1 else s
                    for j, s in enumerate(skips)
                ]
                b = film_apply(self.depth - 2, b)
        else:
            if len(parts) > 0:
                x = torch.cat([dn] + parts, dim=1)
            else:
                x = dn
            x = do_pad(x)

            skips = []
            for i in range(1, self.depth):
                enc = getattr(self, f"enc{i}")
                x = enc(x)
                if film_apply is not None and i >= 2:
                    # enc_i output has width widths[i-1]; conditioned levels
                    # start at widths[1] (film index i-2). enc1 (level 1) is
                    # never conditioned — matches Exp 1 exactly.
                    x = film_apply(i - 2, x)
                skips.append(x)
                x = self.pool(x)
            enc_last = getattr(self, f"enc{self.depth}")
            b = enc_last(x)
            if film_apply is not None:
                b = film_apply(self.depth - 2, b)

        b = self.context(b)
        y = b
        
        for i in range(self.depth - 1, 0, -1):
            up = getattr(self, f"up{i}")
            dec = getattr(self, f"dec{i}")
            y = up(y)
            y = dec(torch.cat([y, skips[i-1]], dim=1))
            if film_apply is not None and i >= 2:
                y = film_apply(i - 2, y)

        if self.sem_aux_head is not None:
            sem_logits = self.sem_aux_head(y[..., :H, :W])
        else:
            sem_logits = None

        if self.semantic_mode == "joint" and self.sem_classes > 0 and self.joint_proj is not None and sem_logits is not None:
            sem_probs = torch.softmax(sem_logits, dim=1)
            y_joint = torch.cat([y[..., :H, :W], sem_probs], dim=1)
            y = self.joint_proj(y_joint)
            params = self.head(y)
        else:
            params = self.head(y)[..., :H, :W]
        
        a0 = self.a0
        b0 = self.b0

        if self.parameterization == "absolute_affine":
            a = params[:, 0:1]
            b_ = params[:, 1:2]
            h = a * dn + b_
        elif self.parameterization == "residual_affine":
            da = params[:, 0:1]
            db = params[:, 1:2]
            if self.bounded:
                a = a0 + 1.0 * torch.tanh(da)
                b_ = b0 + 10.0 * torch.tanh(db)
            else:
                a = a0 + da
                b_ = b0 + db
            h = a * dn + b_
        elif self.parameterization == "hybrid_residual":
            da = params[:, 0:1]
            db = params[:, 1:2]
            dh = params[:, 2:3]
            if self.bounded:
                a = a0 + 1.0 * torch.tanh(da)
                b_ = b0 + 10.0 * torch.tanh(db)
                dh = self.max_shift * torch.tanh(dh)
            else:
                a = a0 + da
                b_ = b0 + db
            h = (a * dn + b_) + dh
        elif self.parameterization == "residual_depth":
            # Residual depth field ON TOP of the global affine anchor:
            #   H = a0*Dn + b0 + ΔH
            # so the zero-init head starts EXACTLY at the baseline. The pure
            # "H = Dn + ΔH" variant is the special case affine_init a0=1,b0=0.
            dh = params[:, 0:1]
            if self.bounded:
                dh = self.max_shift * torch.tanh(dh)
            h = self.a0 * dn + self.b0 + dh
            a = torch.full_like(dh, self.a0)
            b_ = self.b0 + dh
            
        if self.clamp_min is not None:
            h = torch.clamp(h, min=self.clamp_min)
            
        out = {
            "pred": h,
            "a": a,
            "b": b_,
            "sem_zero_filled": sem_zero_filled,
            "stats_zero_filled": stats_zero_filled,
        }
        if sem_logits is not None:
            out["sem_logits"] = sem_logits
        if self.unc_head is not None:
            # Note: if joint_proj altered y, should unc_head use it? 
            # Yes, they both can use the final y. Wait, if joint_proj was used, y is already cropped to H,W
            if self.semantic_mode == "joint":
                out["log_var"] = self.unc_head(y)
            else:
                out["log_var"] = self.unc_head(y[..., :H, :W])
            
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


def semantic_mode_from_ckpt(ckpt: dict) -> str:
    """Map checkpoint metadata to ``semantic_mode`` — legacy-aware.

    V2 checkpoints store ``semantic_mode`` directly. Exp-4/5-era checkpoints
    store only the bool ``sem_input``: True = GT-semantics input design,
    False = head-only (predicted semantics via the aux head). Older still
    omit both and default to ``"input"`` so they rebuild bit-identically.
    """
    mode = ckpt.get("semantic_mode")
    if mode is not None:
        return str(mode)
    if "sem_input" in ckpt:
        if bool(ckpt["sem_input"]):
            return "input"
        if bool(ckpt.get("sem_aux_head", False)):
            return "auxiliary"
        return "none"
    return "input"


def load_affine_init(path) -> Tuple[float, float]:
    """Read (a0, b0) from a global_affine.json for exact-baseline init."""
    import json

    with open(path, "r", encoding="utf-8") as f:
        bl = json.load(f)
    return float(bl["a"]), float(bl["b"])
