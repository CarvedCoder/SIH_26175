import re

with open("depthwizard/calibration_net.py", "r") as f:
    code = f.read()

# Change `sem_input: bool = True` to `semantic_mode: str = "input"` in __init__
code = code.replace("sem_input: bool = True", "semantic_mode: str = \"input\"")
code = code.replace("self.sem_input = bool(sem_input)", "self.semantic_mode = semantic_mode\n        self.sem_input = (self.semantic_mode == \"input\")")

# Add `joint_proj` layer if `semantic_mode == "joint"`
replacement_head = """
        self.sem_aux_head = None
        if sem_aux_head and self.sem_classes > 0:
            self.sem_aux_head = nn.Conv2d(head_in, self.sem_classes, 1)

        self.joint_proj = None
        if self.semantic_mode == "joint" and self.sem_classes > 0:
            self.joint_proj = nn.Conv2d(head_in + self.sem_classes, head_in, 3, padding=1)

        self.unc_head = None
"""
pattern_head = re.compile(r'        self\.sem_aux_head = None.*?        self\.unc_head = None', re.DOTALL)
code = pattern_head.sub(replacement_head.strip("\n") + "\n", code)

# Update the forward pass logic for `semantic_mode == "joint"`
replacement_fwd = """
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
        
        a0 = 2.076463
"""
pattern_fwd = re.compile(r'        params = self\.head\(y\)\[\.\.\., :H, :W\].*?a0 = 2\.076463', re.DOTALL)
code = pattern_fwd.sub(replacement_fwd.strip("\n") + "\n", code)

# Finally, return `sem_logits`
replacement_out = """
        out = {
            "pred": h,
            "a": a,
            "b": b_,
            "sem_zero_filled": sem_zero_filled,
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
"""
pattern_out = re.compile(r'        out = \{.*?if self\.unc_head is not None:\n            out\["log_var"\] = self\.unc_head\(y\[\.\.\., :H, :W\]\)', re.DOTALL)
code = pattern_out.sub(replacement_out.strip("\n"), code)

with open("depthwizard/calibration_net.py", "w") as f:
    f.write(code)
