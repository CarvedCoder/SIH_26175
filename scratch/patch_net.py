import re

with open("depthwizard/calibration_net.py", "r") as f:
    content = f.read()

# The class starts at `class CalibrationNet(nn.Module):`
# and ends before `def masked_l1_loss`

parts = re.split(r'class CalibrationNet\(nn\.Module\):', content, 1)
pre_class = parts[0]
post_class_parts = re.split(r'def masked_l1_loss\(', parts[1], 1)
post_class = "def masked_l1_loss(" + post_class_parts[1]

new_class = """
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

class CalibrationNet(nn.Module):
    \"\"\"V2 Modular CalibrationNet (Dn [, RGB [, SEM ]][, DEM]) -> H.\"\"\"
    def __init__(
        self,
        in_ch: int = 1,
        widths: Tuple[int, ...] = (16, 32, 64),
        a0: float = 2.076463,
        b0: float = 4.110271,
        clamp_min: float = 0.0,
        sem_classes: int = 0,
        sem_aux_head: bool = False,
        sem_input: bool = True,
        parameterization: str = "absolute_affine",
        bounded: bool = False,
        max_shift: float = 10.0,
        fusion_mode: str = "early",
        context_module: str = "none",
        use_uncertainty: bool = False,
    ):
        super().__init__()
        self.clamp_min = clamp_min
        self.sem_classes = int(sem_classes)
        self.sem_input = bool(sem_input)
        self.parameterization = parameterization
        self.bounded = bounded
        self.max_shift = float(max_shift)
        self.fusion_mode = fusion_mode
        self.context_module = context_module
        self.use_uncertainty = use_uncertainty
        self.widths = tuple(widths)
        self.depth = len(self.widths)
        
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

        self.unc_head = None
        if self.use_uncertainty:
            self.unc_head = nn.Conv2d(head_in, 1, 3, padding=1)

    def forward(
        self,
        dn: torch.Tensor,
        rgb: Optional[torch.Tensor] = None,
        dem: Optional[torch.Tensor] = None,
        sem: Optional[torch.Tensor] = None,
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
        
        sem_zero_filled = False
        if self.sem_input and self.sem_classes > 0 and sem is None:
            sem = torch.zeros(
                dn.shape[0], self.sem_classes, *dn.shape[-2:],
                device=dn.device, dtype=dn.dtype
            )
            sem_zero_filled = True
            
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
                skips.append(x)
                x = self.pool(x)
            enc_last = getattr(self, f"enc{self.depth}")
            b = enc_last(x)

        b = self.context(b)
        y = b
        
        for i in range(self.depth - 1, 0, -1):
            up = getattr(self, f"up{i}")
            dec = getattr(self, f"dec{i}")
            y = up(y)
            y = dec(torch.cat([y, skips[i-1]], dim=1))

        params = self.head(y)[..., :H, :W]
        
        a0 = 2.076463
        b0 = 4.110271
        
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
            dh = params[:, 0:1]
            if self.bounded:
                dh = self.max_shift * torch.tanh(dh)
            h = dn + dh
            a = torch.ones_like(dh)
            b_ = h - dn
            
        if self.clamp_min is not None:
            h = torch.clamp(h, min=self.clamp_min)
            
        out = {
            "pred": h,
            "a": a,
            "b": b_,
            "sem_zero_filled": sem_zero_filled,
        }
        if self.sem_aux_head is not None:
            out["sem_logits"] = self.sem_aux_head(y[..., :H, :W])
        if self.unc_head is not None:
            out["log_var"] = self.unc_head(y[..., :H, :W])
            
        return out

"""

with open("depthwizard/calibration_net.py", "w") as f:
    f.write(pre_class + new_class + post_class)
