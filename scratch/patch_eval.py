import re

with open("depthwizard/cli/eval_calibration.py", "r") as f:
    code = f.read()

replacement1 = """
        in_ch=in_ch,
        widths=tuple(ckpt["widths"]),
        a0=a0,
        b0=b0,
        clamp_min=ckpt.get("clamp_min", 0.0),
        sem_classes=int(ckpt.get("sem_classes", 0)),
        sem_aux_head=bool(ckpt.get("sem_aux_head", False)),
        sem_input=bool(ckpt.get("sem_input", True)),
        parameterization=ckpt.get("parameterization", "absolute_affine"),
        bounded=bool(ckpt.get("bounded", False)),
        fusion_mode=ckpt.get("fusion_mode", "early"),
        context_module=ckpt.get("context_module", "none"),
        use_uncertainty=bool(ckpt.get("use_uncertainty", False))
"""
pattern1 = re.compile(r'        in_ch=in_ch,.*?        max_shift=float\(ckpt\.get\("max_shift", 10\.0\)\),', re.DOTALL)
code = pattern1.sub(replacement1.strip("\n") + ",", code)

replacement2 = """
        sem_aux_head=bool(ckpt.get("sem_aux_head", False)),
        parameterization=ckpt.get("parameterization", "absolute_affine"),
        bounded=bool(ckpt.get("bounded", False)),
        context_module=ckpt.get("context_module", "none"),
        fusion_mode=ckpt.get("fusion_mode", "early"),
        use_uncertainty=bool(ckpt.get("use_uncertainty", False))
"""
pattern2 = re.compile(r'        sem_aux_head=bool\(ckpt\.get\("sem_aux_head", False\)\),.*?        max_shift=float\(ckpt\.get\("max_shift", 10\.0\)\),', re.DOTALL)
code = pattern2.sub(replacement2.strip("\n") + ",", code)

with open("depthwizard/cli/eval_calibration.py", "w") as f:
    f.write(code)
