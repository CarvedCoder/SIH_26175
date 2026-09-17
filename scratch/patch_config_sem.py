import re

with open("depthwizard/config.py", "r") as f:
    code = f.read()

code = code.replace("bounded: bool = False", "bounded: bool = False\n    semantic_mode: str = \"input\"")

with open("depthwizard/config.py", "w") as f:
    f.write(code)

with open("depthwizard/cli/train_calibration.py", "r") as f:
    train_code = f.read()

train_code = train_code.replace("sem_input=use_sem,", "semantic_mode=v2_cfg.model.semantic_mode,")
train_code = train_code.replace("\"sem_input\": use_sem,", "\"semantic_mode\": v2_cfg.model.semantic_mode,")

with open("depthwizard/cli/train_calibration.py", "w") as f:
    f.write(train_code)

with open("depthwizard/tifops.py", "r") as f:
    tif_code = f.read()

tif_code = tif_code.replace("sem_input: bool = True", "semantic_mode: str = \"input\"")
tif_code = tif_code.replace("sem_input=bool(ckpt.get(\"sem_input\", True)),", "semantic_mode=ckpt.get(\"semantic_mode\", \"input\"),")

with open("depthwizard/tifops.py", "w") as f:
    f.write(tif_code)

with open("depthwizard/cli/eval_calibration.py", "r") as f:
    eval_code = f.read()

eval_code = eval_code.replace("sem_input=bool(ckpt.get(\"sem_input\", True)),", "semantic_mode=ckpt.get(\"semantic_mode\", \"input\"),")

with open("depthwizard/cli/eval_calibration.py", "w") as f:
    f.write(eval_code)
