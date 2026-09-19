import re

with open("depthwizard/tifops.py", "r") as f:
    code = f.read()

replacement1 = """
    sem_classes: int = 0
    sem_aux_head: bool = False
    parameterization: str = "absolute_affine"
    bounded: bool = False
    context_module: str = "none"
    fusion_mode: str = "early"
    use_uncertainty: bool = False
"""
pattern1 = re.compile(r'    sem_classes: int = 0.*?    sem_aux_head: bool = False', re.DOTALL)
code = pattern1.sub(replacement1.strip("\n"), code)

replacement2 = """
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
pattern2 = re.compile(r'        in_ch=in_ch,.*?        sem_input=bool\(ckpt\.get\("sem_input", True\)\),', re.DOTALL)
code = pattern2.sub(replacement2.strip("\n"), code)

with open("depthwizard/tifops.py", "w") as f:
    f.write(code)
