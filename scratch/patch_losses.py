import re

with open("depthwizard/losses.py", "r") as f:
    code = f.read()

replacement1 = """
from depthwizard.config import LossWeightsConfig
from .calibration_net import masked_huber_loss, masked_l1_loss

def masked_berhu_loss(pred: torch.Tensor, target: torch.Tensor, c: float = 0.2) -> torch.Tensor:
    valid = torch.isfinite(target)
    if valid.sum() == 0:
        return pred.sum() * 0.0
    err = (pred - target).abs()[valid]
    c = c * err.max().item() # Adaptive c based on max error in batch, or fixed if c is large. Let's use fixed if c > 0.
    if c <= 0:
        c = 0.2 * err.max().item()
    mask = err <= c
    l1 = err[mask]
    l2 = (err[~mask] ** 2 + c ** 2) / (2 * c)
    return (l1.sum() + l2.sum()) / valid.sum()
"""

# replace from imports down to `class LossConfig:`
pattern1 = re.compile(r'from \.calibration_net import masked_huber_loss, masked_l1_loss.*?class LossConfig:', re.DOTALL)
code = pattern1.sub(replacement1.strip("\n") + "\n\nclass LossConfig:", code)

# replace LossConfig
replacement2 = """
# Legacy wrapper to avoid breaking older tools
class LossConfig:
    pass

class DepthLoss:
    def __init__(self, cfg: Optional[LossWeightsConfig] = None):
        self.cfg = cfg or LossWeightsConfig()
        self.last_parts: Dict[str, float] = {}

    def main_term(self, pred, target) -> torch.Tensor:
        if self.cfg.main == "huber":
            return masked_huber_loss(pred, target, self.cfg.huber_delta)
        elif self.cfg.main == "berhu":
            return masked_berhu_loss(pred, target, self.cfg.berhu_c)
        return masked_l1_loss(pred, target)

    def _get_balanced_weights(self, target: torch.Tensor) -> torch.Tensor:
        # Simple height balancing: higher weights for rarer tall structures
        valid = torch.isfinite(target)
        if not valid.any():
            return torch.ones_like(target)
        
        vals = target[valid]
        w = torch.ones_like(target)
        # Bins: 0-10m, 10-30m, >30m
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
        # main term with optional balancing
        if self.cfg.height_balanced:
            bw = self._get_balanced_weights(target)
            valid = torch.isfinite(target)
            if valid.sum() == 0:
                total = pred.sum() * 0.0
            else:
                err = (pred - target).abs()
                if self.cfg.main == "huber":
                    delta = self.cfg.huber_delta
                    mask = err < delta
                    l1 = err[mask]
                    l2 = (err[~mask] ** 2) / (2 * delta) + delta / 2
                    total = (l1 * bw[mask]).sum() + (l2 * bw[~mask]).sum()
                    total /= valid.sum()
                else:
                    total = (err * bw)[valid].mean()
        else:
            total = self.main_term(pred, target)

        if log_var is not None and self.cfg.uncertainty_weight > 0:
            # Heteroscedastic uncertainty formulation: loss * exp(-log_var) + log_var
            valid = torch.isfinite(target)
            if valid.sum() > 0:
                err = (pred - target).abs()[valid]
                v = log_var[valid]
                unc_loss = (err * torch.exp(-v) + v).mean()
                total = total + self.cfg.uncertainty_weight * unc_loss

        parts = {"main": float(total.detach())}
        if self.cfg.gradient_weight > 0.0:
            g = gradient_loss(pred, target)
            total = total + self.cfg.gradient_weight * g
            parts["grad"] = float(g.detach())
        if getattr(self.cfg, "w_slope", 0.0) > 0.0: # fallback
            s = slope_angle_loss(pred, target, gsd_m=0.33)
            total = total + getattr(self.cfg, "w_slope", 0.0) * s
            parts["slope"] = float(s.detach())
        if self.cfg.boundary_weight > 0.0 and rgb is not None:
            s = edge_aware_smoothness(pred, rgb)
            total = total + self.cfg.boundary_weight * s
            parts["smooth"] = float(s.detach())
        if self.cfg.semantic_weight > 0.0 and sem_logits is not None and sem_target is not None:
            c = masked_semantic_ce(sem_logits, sem_target, sem_ignore)
            total = total + self.cfg.semantic_weight * c
            parts["sem_ce"] = float(c.detach())
        self.last_parts = parts
        return total
"""

pattern2 = re.compile(r'class LossConfig:.*?class DepthLoss:.*?return total', re.DOTALL)
code = pattern2.sub(replacement2.strip("\n"), code)

with open("depthwizard/losses.py", "w") as f:
    f.write(code)
