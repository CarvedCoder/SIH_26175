import re

with open("depthwizard/cli/train_calibration.py", "r") as f:
    code = f.read()

replacement = """
                    "sem_input": use_sem,
                    "parameterization": v2_cfg.model.parameterization,
                    "bounded": v2_cfg.model.bounded,
                    "fusion_mode": v2_cfg.fusion.mode,
                    "context_module": v2_cfg.model.context,
                    "use_uncertainty": v2_cfg.loss.uncertainty_weight > 0,
                    "in_ch": in_ch,
"""

pattern = re.compile(r'                    "sem_input": use_sem,.*?                    "in_ch": in_ch,', re.DOTALL)
new_code = pattern.sub(replacement.strip("\n"), code)

with open("depthwizard/cli/train_calibration.py", "w") as f:
    f.write(new_code)
