import re

with open("depthwizard/cli/train_calibration.py", "r") as f:
    code = f.read()

# We can replace the config loading block.
# Specifically, around line 235:
# cfg = load_config(args.config)
# paths, mcfg, tcfg = cfg["paths"], cfg["model"], cfg["train"]
# use_rgb = args.use_rgb or bool(mcfg.get("use_rgb", False))

replacement = """
    from depthwizard.config import CalibrationConfig
    import yaml
    with open(args.config, "r", encoding="utf-8") as f:
        raw_cfg = yaml.safe_load(f)
    v2_cfg = CalibrationConfig.from_dict(raw_cfg)
    
    # Apply CLI overrides if provided
    if args.epochs is not None: v2_cfg.train.epochs = args.epochs
    if args.batch_size is not None: v2_cfg.train.batch_size = args.batch_size
    if args.lr is not None: v2_cfg.train.lr = args.lr
    if args.loss is not None: v2_cfg.loss.main = args.loss
    if args.val_subset is not None: v2_cfg.train.val_subset = args.val_subset
    if args.use_rgb: v2_cfg.inputs.rgb = True
    if args.use_dem: v2_cfg.inputs.dem = True
    if args.use_sem: v2_cfg.inputs.semantic = True
    if args.w_grad is not None: v2_cfg.loss.gradient_weight = args.w_grad
    if args.w_smooth is not None: v2_cfg.loss.boundary_weight = args.w_smooth
    if args.w_sem is not None: v2_cfg.loss.semantic_weight = args.w_sem
    
    # Legacy fallbacks for compatibility with the script's variables
    paths = raw_cfg.get("paths", {})
    tcfg = raw_cfg.get("train", {})
    mcfg = raw_cfg.get("model", {})
    
    use_rgb = v2_cfg.inputs.rgb
    use_dem = v2_cfg.inputs.dem
    use_sem = v2_cfg.inputs.semantic
    sem_aux_head = args.sem_aux_head or bool(mcfg.get("sem_aux_head", False))
"""

# Replace the block from `cfg = load_config...` to `sem_aux_head = ...`
pattern = re.compile(r'    cfg = load_config\(args\.config\).*?sem_aux_head = args\.sem_aux_head or bool\(mcfg\.get\("sem_aux_head", False\)\)', re.DOTALL)
new_code = pattern.sub(replacement.strip(), code)

with open("depthwizard/cli/train_calibration.py", "w") as f:
    f.write(new_code)
