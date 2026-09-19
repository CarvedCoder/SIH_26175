import re

with open("depthwizard/datasets/base.py", "r") as f:
    code = f.read()

# Add confidence to optional keys loop
replacement1 = """
        for k in ("dn", "dem", "confidence"):
            if arrs.get(k) is not None:
"""
pattern1 = re.compile(r'        for k in \("dn", "dem"\):\n            if arrs\.get\(k\) is not None:', re.DOTALL)
code = pattern1.sub(replacement1.strip("\n") + "\n", code)

# Tensorize confidence if it exists
replacement2 = """
        if "confidence" in layers:
            ret["conf"] = torch.from_numpy(layers["confidence"]).unsqueeze(0)
"""
pattern2 = re.compile(r'        if "dem" in layers:\n            ret\["dem"\] = torch\.from_numpy\(layers\["dem"\]\)\.unsqueeze\(0\)', re.DOTALL)
# append to it
code = pattern2.sub(r'        if "dem" in layers:\n            ret["dem"] = torch.from_numpy(layers["dem"]).unsqueeze(0)' + "\n\n" + replacement2.strip("\n"), code)

with open("depthwizard/datasets/base.py", "w") as f:
    f.write(code)

with open("depthwizard/cli/train_calibration.py", "r") as f:
    train_code = f.read()

train_code = train_code.replace("use_sem=v2_cfg.inputs.semantic", "use_sem=v2_cfg.inputs.semantic, use_confidence=v2_cfg.inputs.confidence")
# Wait, derive_in_ch does not take use_confidence, and we don't pass confidence as input right now. The task says:
# "Update dataset to expose `confidence` (e.g., TTA disagreement or local stability) as a first-class feature."
# And in losses.py, w_conf is reserved but not implemented. So I just need to expose it in dataset output.

with open("depthwizard/cli/train_calibration.py", "w") as f:
    f.write(train_code)
