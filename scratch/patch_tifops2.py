import re

with open("depthwizard/tifops.py", "r") as f:
    code = f.read()

replacement = """
        sem_aux_head=bool(ckpt.get("sem_aux_head", False)),
        parameterization=ckpt.get("parameterization", "absolute_affine"),
        bounded=bool(ckpt.get("bounded", False)),
        context_module=ckpt.get("context_module", "none"),
        fusion_mode=ckpt.get("fusion_mode", "early"),
        use_uncertainty=bool(ckpt.get("use_uncertainty", False))
"""

pattern = re.compile(r'        sem_aux_head=bool\(ckpt\.get\("sem_aux_head", False\)\)', re.DOTALL)
code = pattern.sub(replacement.strip("\n"), code)

with open("depthwizard/tifops.py", "w") as f:
    f.write(code)
