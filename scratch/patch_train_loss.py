import re

with open("depthwizard/cli/train_calibration.py", "r") as f:
    code = f.read()

replacement1 = """
    loss_fn = DepthLoss(v2_cfg.loss)
"""
pattern1 = re.compile(r'    loss_cfg = LossConfig\(.*?\n    \)\n    loss_fn = DepthLoss\(loss_cfg\)', re.DOTALL)
code = pattern1.sub(replacement1.strip("\n"), code)

# Update where loss_fn is called to include log_var if it exists in predictions
replacement2 = """
            loss = loss_fn(
                pred=out["pred"],
                target=h_gt,
                rgb=img if use_rgb else None,
                sem_logits=out.get("sem_logits"),
                sem_target=sem_oh,
                sem_ignore=sem_ign,
                log_var=out.get("log_var")
            )
"""
pattern2 = re.compile(r'            loss = loss_fn\(\n                pred=pred,\n                target=h_gt,\n                rgb=img if use_rgb else None,\n                sem_logits=sem_logits,\n                sem_target=sem_oh,\n                sem_ignore=sem_ign,\n            \)', re.DOTALL)
code = pattern2.sub(replacement2.strip("\n"), code)

with open("depthwizard/cli/train_calibration.py", "w") as f:
    f.write(code)
