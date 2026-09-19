import re

with open("depthwizard/cli/train_calibration.py", "r") as f:
    code = f.read()

replacement = """
    net = CalibrationNet(
        in_ch=in_ch,
        widths=tuple(v2_cfg.model.widths),
        a0=a0,
        b0=b0,
        clamp_min=v2_cfg.model.clamp_min,
        sem_classes=sem_classes,
        sem_aux_head=sem_aux_head,
        sem_input=use_sem,
        parameterization=v2_cfg.model.parameterization,
        bounded=v2_cfg.model.bounded,
        max_shift=10.0, # Will be configurable later
        fusion_mode=v2_cfg.fusion.mode,
        context_module=v2_cfg.model.context,
        use_uncertainty=v2_cfg.loss.uncertainty_weight > 0
    ).to(device)
"""

pattern = re.compile(r'    net = CalibrationNet\(.*?\)\.to\(device\)', re.DOTALL)
new_code = pattern.sub(replacement.strip(), code)

with open("depthwizard/cli/train_calibration.py", "w") as f:
    f.write(new_code)
