"""Train the spatial calibration net.

Model:  H(x,y) = clamp(a(x,y)*Dn(x,y) + b(x,y), 0)
        a, b from a ~0.2M-param U-Net over Dn (+RGB [+SEM one-hot] [+DEM]
        ablation flags). Every variant starts EXACTLY at the affine baseline
        via the zero-weight head + a0,b0 bias init (worklog Section 1).
Init:   exact global affine (zero-weight head, biases a0,b0) — training can
        only start AT the Phase-1 baseline. Loss: masked L1 (default) / Huber,
        plus OPTIONAL weighted terms (Phase 4): --w-grad (gradient loss),
        --w-smooth (edge-aware smoothness), --w-sem (semantic CE via the
        auxiliary semantic head). ALL extra weights default to 0 = the exact
        pre-Phase-4 behavior.

Dataset selection (Phase 3):
  * config WITHOUT a `dataset:` section -> the frozen DFC2019 legacy path
    (byte-identical to the pre-GAMUS command; Exp 1/2/3 preserved).
  * config WITH `dataset: {name: dfc2019|gamus|mixed, ...}` -> the
    depthwizard.datasets factory path (adapters; one-hot semantics; GAMUS
    raw-HDF5 source; mixed = WeightedRandomSampler over sources — only
    after the stats gate, see depthwizard/datasets/mixed.py).

Experiment ladder mapping:
  Exp 1  fit-baseline (affine)          Exp 4  train --use-rgb --use-sem
  Exp 2  train                          Exp 5  train --use-rgb --use-sem --use-dem
  Exp 3  train --use-rgb                Exp 6  LoRA (separate, gated on
                                             reproducible evaluation of 1-5)

Monitoring: every epoch, full-tile MAE on --val-subset val tiles (deterministic
first-K of the sorted val list). Early stop on patience. Best checkpoint by
subset MAE. FINAL numbers come from `evaluate` (full val+test) — never quote
the training monitor.

Scheduler rule (worklog incident #2, binding): CosineAnnealingLR is
instantiated ONCE (at the true base LR, T_max = total epochs) and stepped
ONCE PER EPOCH — never inside the batch loop.

Usage:
  python model.py train --config configs/phase2.yaml
  python model.py train --config configs/phase2.yaml --epochs 3 \
      --max-train-tiles 64 --val-subset 8        # smoke run
  python model.py train --config configs/phase2.yaml --use-rgb --cosine
  python model.py train --config configs/phase2.yaml --use-rgb --use-dem \
      --synth-dem --cosine                      # Method-D challenger (SYNTH)
  python model.py train --config configs/gamus.yaml --out-tag gamus_dn
  python model.py train --config configs/gamus.yaml --use-rgb --use-sem \
      --out-tag gamus_rgb_sem                   # Exp 4 on GAMUS
"""

from __future__ import annotations

import argparse
import time
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from depthwizard.calibration_net import (CalibrationNet, derive_in_ch,
                                         load_affine_init)
from depthwizard.cli.args import (add_cache_subdir_arg, add_config_arg,
                                  add_device_arg, load_config, resolve_cache,
                                  resolve_device)
from depthwizard.dataset import DFC2019Config, discover_and_split
from depthwizard.datasets.semantics import NUM_PROJECT_CLASSES
from depthwizard.geo import dump_json
from depthwizard.losses import DepthLoss, LossConfig
from depthwizard.metrics import height_metrics

NAME = "train"
HELP = "train the spatial calibration net (U-Net affine over Dn [+RGB])"


def val_subset_mae(net, ds_val, subset: int, use_rgb: bool, device: str,
                   use_dem: bool = False, use_sem: bool = False) -> float:
    """Pooled MAE (metres) over the first `subset` val tiles, full-tile.

    ``use_dem`` mirrors ``use_rgb``: when True, the val sample's ``dem``
    layer is fed to the net's third positional argument.
    ``use_sem`` (Exp 4/5): the val sample's one-hot semantic layer feeds
    the net's ``sem`` argument — validation uses the SAME input contract
    as training (GT semantics = privileged information, documented).
    """
    import torch

    with torch.no_grad():
        net.eval()
        tot_abs, tot_n = 0.0, 0
        for i in range(min(subset, len(ds_val))):
            s = ds_val[i]                               # full tile (crop_size=None)
            dn = s["dn"].to(device)
            rgb = s["rgb"].to(device) if use_rgb else None
            dem = s["dem"].to(device) if use_dem else None
            sem = (s["sem_onehot"].to(device)
                   if (use_sem and s.get("sem_onehot") is not None) else None)
            pred = net(dn, rgb, dem, sem)["pred"][0, 0].cpu().numpy()
            m = height_metrics(pred, s["agl"][0].numpy())
            tot_abs += m["mae"] * m["n"]
            tot_n += m["n"]
        net.train()
        return tot_abs / max(tot_n, 1)


def add_parser(sub: argparse._SubParsersAction) -> argparse.ArgumentParser:
    p = sub.add_parser(NAME, help=HELP, description=__doc__,
                       formatter_class=argparse.RawDescriptionHelpFormatter)
    add_config_arg(p, "configs/phase2.yaml")
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--loss", choices=("l1", "huber"), default=None)
    p.add_argument("--val-subset", type=int, default=None)
    p.add_argument("--max-train-tiles", type=int, default=0,
                   help="debug: use only first N train tiles")
    p.add_argument("--use-rgb", action="store_true")
    # Method-D challenger (Dn+RGB+DEM). Mirrors --use-rgb exactly: same loss
    # functions, same scheduler discipline, same zero-head + a0,b0 init.
    # The DEM channel is the ONLY architecture change; ablation discipline
    # (worklog Section 1) keeps every other variable fixed so any val
    # improvement is attributable to the DEM channel.
    p.add_argument("--use-dem", action="store_true",
                   help="train the in_ch=5 (Dn+RGB+DEM) Method-D variant")
    # ---- Exp 4/5: semantic one-hot input channels (Phase 4) ----
    p.add_argument("--use-sem", action="store_true",
                   help="feed the 6-channel project one-hot (sem_onehot) as "
                        "input channels (Exp 4/5). Requires a dataset "
                        "adapter producing semantics (dataset: section "
                        "config or DFC adapter).")
    p.add_argument("--sem-aux-head", action="store_true",
                   help="add the auxiliary semantic head (K logits) and "
                        "enable --w-sem supervision; also enables "
                        "predicted-semantics inference later. Default off "
                        "(minimal-first).")
    # ---- Phase 4 loss weights (ALL default 0 = exact pre-Phase-4 loss) ----
    p.add_argument("--w-grad", type=float, default=None,
                   help="weight of the gradient (boundary) loss (default 0)")
    p.add_argument("--w-smooth", type=float, default=None,
                   help="weight of the edge-aware smoothness loss (default 0)")
    p.add_argument("--w-sem", type=float, default=None,
                   help="weight of the semantic CE loss via the aux head "
                        "(default 0; requires --sem-aux-head)")
    p.add_argument("--dem-dir", type=Path, default=None,
                   help="directory of {stem}.tif single-band DEMs for the "
                        "DEM channel. Required for --use-dem unless "
                        "--synth-dem is given.")
    p.add_argument("--synth-dem", action="store_true",
                   help="when --use-dem and no --dem-dir, synthesise the DEM "
                        "from AGL via demprior.synth_dem_from_agl. Every "
                        "output / worklog line MUST carry the literal "
                        "SYNTHETIC-DEM-PROXY tag — see demprior.py.")
    p.add_argument("--synth-dem-sigma-m", type=float, default=8.0,
                   help="Gaussian sigma (metres) for the synthetic-DEM proxy")
    p.add_argument("--synth-dem-gsd-m", type=float, default=None,
                   help="GSD (metres/pixel) for sigma conversion; None "
                        "assumes 1 m/px (must be reported as SYNTHETIC-DEM-PROXY)")
    add_device_arg(p)
    add_cache_subdir_arg(p)
    p.add_argument("--workers", type=int, default=None)
    p.add_argument("--out-tag", default="", help="subdir tag, e.g. 'rgb'")
    p.add_argument("--patience", type=int, default=None)
    p.add_argument("--cosine", action="store_true", help="cosine anneal LR over --epochs")
    return p


def run(args) -> int:
    import torch
    from torch.utils.data import DataLoader

    cfg = load_config(args.config)
    paths, mcfg, tcfg = cfg["paths"], cfg["model"], cfg["train"]
    use_rgb = args.use_rgb or bool(mcfg.get("use_rgb", False))
    use_dem = args.use_dem
    use_sem = args.use_sem or bool(mcfg.get("use_sem", False))
    sem_aux_head = args.sem_aux_head or bool(mcfg.get("sem_aux_head", False))
    sem_classes = NUM_PROJECT_CLASSES if use_sem else 0
    device = resolve_device(args.device)
    torch.manual_seed(tcfg.get("seed", 42))
    np.random.seed(tcfg.get("seed", 42))

    cache_dir = resolve_cache(paths, args.cache_subdir)
    print(f"[i] depth cache: {cache_dir}")

    if not Path(paths["affine_json"]).exists():
        print(f"[error] {paths['affine_json']} missing — run `model.py fit-baseline` "
              "first (init needs a0,b0).")
        return 1
    a0, b0 = load_affine_init(paths["affine_json"])
    print(f"[i] device={device}  use_rgb={use_rgb}  use_dem={use_dem}  "
          f"use_sem={use_sem}  affine-init a0={a0:.6f} b0={b0:.6f}")

    # DEM-channel resolution (unchanged for the legacy path; guarded on the
    # factory path). The honesty contract (demprior.py) requires the proxy
    # tag to travel with every output/worklog line.
    dem_dir = args.dem_dir
    synth_dem = False
    if use_dem:
        if dem_dir is None and not args.synth_dem:
            print("[error] --use-dem requires either --dem-dir <dir> or "
                  "--synth-dem (the SYNTHETIC-DEM-PROXY fallback). Refusing "
                  "to silently fabricate a DEM channel.")
            return 1
        if dem_dir is None and args.synth_dem:
            synth_dem = True
            print("[!] SYNTHETIC-DEM-PROXY: training on a synthetic DEM "
                  "derived from AGL (demprior.synth_dem_from_agl). Any "
                  "resulting win is a MECHANISM result, NOT a citable "
                  "'DEM conditioning works' claim until validated against "
                  "a real DEM. See worklog Section 3, rule 6.")

    # ------------------------------------------------------------------
    # Dataset construction: legacy frozen path OR the multi-dataset factory
    # ------------------------------------------------------------------
    dcfg = dict(cfg.get("dataset") or {})
    dataset_name = dcfg.get("name")
    mixed_weights_note = None
    if dataset_name:
        # ---- factory path (Phase 3): adapters / GAMUS / mixed ----------
        from depthwizard.datasets.factory import build_dataset, build_datasets
        from depthwizard.datasets.mixed import MixedDataset
        if use_dem and dataset_name == "gamus":
            print("[error] GAMUS has no DEM source (non-georeferenced HDF5) "
                  "— --use-dem requires DFC2019.")
            return 1
        if use_dem:
            # pass the DEM fields through to the DFC adapter config
            dcfg["dem_dir"] = str(dem_dir) if dem_dir else None
            dcfg["synth_dem_fallback"] = synth_dem
            dcfg["synth_dem_sigma_m"] = args.synth_dem_sigma_m
            dcfg["synth_dem_gsd_m"] = args.synth_dem_gsd_m
        clamp_min_ds = float(dcfg.get("clamp_agl_min", 0.0))
        seed = int(tcfg.get("seed", 42))
        train_cfg_full = {"paths": paths, "dataset": dcfg}
        ds_all = build_datasets(
            train_cfg_full,
            crop_size=tcfg["crop_size"],
            augment=True,
            load_depth=True,
            depth_cache_dir=cache_dir,
            clamp_agl_min=clamp_min_ds,
            seed=seed,
        )
        ds_val = build_dataset(
            train_cfg_full,
            "val",
            crop_size=None,
            augment=False,
            load_depth=True,
            depth_cache_dir=cache_dir,
            clamp_agl_min=clamp_min_ds,
            seed=seed,
        )
        ds = {"train": ds_all["train"], "val": ds_val,
              "test": ds_all.get("test")}
        print(f"[i] dataset (factory): {dataset_name}  "
              f"train={len(ds['train'])} val={len(ds['val'])}")
        if isinstance(ds["train"], MixedDataset):
            mixed_weights_note = list(zip(ds["train"].source_names,
                                          ds["train"].weights))
            print(f"[i] MIXED sources (weight, non-optimized defaults): "
                  f"{mixed_weights_note}")
    else:
        # ---- legacy frozen DFC2019 path (Exp 1/2/3 preserved) ----------
        base = DFC2019Config(
            rgb_dir=Path(paths["rgb_dir"]), truth_dir=Path(paths["truth_dir"]),
            depth_cache_dir=cache_dir, load_depth=True,
            crop_size=tcfg["crop_size"], augment=True,
            clamp_agl_min=cfg["dataset"]["clamp_agl_min"] if "dataset" in cfg else 0.0,
            dem_dir=dem_dir,
            synth_dem_fallback=synth_dem,
            synth_dem_sigma_m=args.synth_dem_sigma_m,
            synth_dem_gsd_m=args.synth_dem_gsd_m,
            seed=tcfg.get("seed", 42))
        ds = discover_and_split(base, Path(paths["splits_json"]))
        # eval-time dataset needs FULL tiles -> rebuild val with crop=None, but
        # KEEP the DEM config so the DEM channel is produced in val too.
        ds["val"].cfg = replace(ds["val"].cfg, crop_size=None, augment=False)
        print(f"[i] dataset (legacy DFC2019): train={len(ds['train'])} "
              f"val={len(ds['val'])}")

    # --use-sem sanity: the dataset must actually PRODUCE the one-hot layer
    # (zero-filled semantics would silently train a privileged-input model
    # on nothing — guard loud, fail early).
    if use_sem:
        probe = ds["train"][0]
        if probe.get("sem_onehot") is None:
            print("[error] --use-sem but the dataset produces no semantic "
                  "layers. Use a config with a `dataset:` section and "
                  "load_semantics: true (default), or the DFC adapter.")
            return 1
        if probe["sem_onehot"].shape[0] != sem_classes:
            print(f"[error] semantic channel count mismatch: dataset gives "
                  f"{probe['sem_onehot'].shape[0]}, expected {sem_classes}.")
            return 1

    if args.max_train_tiles:
        tr = ds["train"]
        n = args.max_train_tiles
        if hasattr(tr, "tiles"):                       # DFC legacy
            setattr(tr, "tiles", getattr(tr, "tiles")[:n])
        elif hasattr(tr, "samples"):                   # adapters
            setattr(tr, "samples", getattr(tr, "samples")[:n])
        else:                                           # mixed: Subset view
            from torch.utils.data import Subset
            ds["train"] = Subset(tr, range(min(n, len(tr))))

    # DataLoader — mixed datasets use WeightedRandomSampler (per-source
    # weights travel WITH the dataset object; documented non-optimized).
    # collate_dict_none_safe: torch>=2.13 default_collate raises on the
    # optional None layers — restored historical None pass-through, results
    # identical across torch versions (required by the frozen DFC path).
    from depthwizard.datasets.base import collate_dict_none_safe
    from depthwizard.datasets.mixed import MixedDataset
    if isinstance(ds["train"], MixedDataset):
        w = ds["train"].per_sample_weights()
        sampler = torch.utils.data.WeightedRandomSampler(
            w, num_samples=len(w), replacement=True)
        dl = DataLoader(ds["train"],
                        batch_size=args.batch_size or tcfg["batch_size"],
                        sampler=sampler,
                        num_workers=args.workers if args.workers is not None
                        else tcfg.get("workers", 2),
                        pin_memory=(device == "cuda"), drop_last=True,
                        collate_fn=collate_dict_none_safe)
    else:
        dl = DataLoader(ds["train"], batch_size=args.batch_size or tcfg["batch_size"],
                        shuffle=True, num_workers=args.workers if args.workers is not None
                        else tcfg.get("workers", 2),
                        pin_memory=(device == "cuda"), drop_last=True,
                        collate_fn=collate_dict_none_safe)
    print(f"[i] train tiles={len(ds['train'])}  val tiles={len(ds['val'])} "
          f"(monitor subset={args.val_subset or tcfg['val_subset']})")

    # in_ch via the SINGLE source (channel order Dn | RGB | SEM | DEM). The
    # zero-weight head + a0,b0 bias init pins every variant to the affine
    # baseline at step 0 — ablation discipline.
    in_ch = derive_in_ch(use_rgb=use_rgb, use_sem=use_sem, use_dem=use_dem,
                         sem_classes=sem_classes)
    net = CalibrationNet(in_ch=in_ch,
                         widths=tuple(mcfg["widths"]),
                         a0=a0, b0=b0,
                         clamp_min=mcfg.get("clamp_min", 0.0),
                         sem_classes=sem_classes,
                         sem_aux_head=sem_aux_head).to(device)
    n_par = sum(p.numel() for p in net.parameters())
    print(f"[i] CalibrationNet in_ch={in_ch}  params={n_par:,}")

    opt = torch.optim.Adam(net.parameters(), lr=args.lr or tcfg["lr"],
                           weight_decay=tcfg.get("weight_decay", 1e-4))

    # CUDA AMP: reduce VRAM usage and improve throughput on NVIDIA GPUs.
    amp_enabled = (device == "cuda")
    amp_dtype = torch.float16 if amp_enabled else torch.float32

    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=amp_enabled,
    )

    print(f"[i] AMP enabled={amp_enabled} dtype={amp_dtype}")
    # ---- composite loss (Phase 4): weights default to the EXACT current
    # behavior (all extra terms off). CLI flags override the train config.
    loss_cfg = LossConfig(
        main=args.loss or tcfg.get("loss", "l1"),
        huber_delta=float(tcfg.get("huber_delta", 5.0)),
        w_grad=(args.w_grad if args.w_grad is not None
                else float(tcfg.get("w_grad", 0.0))),
        w_smooth=(args.w_smooth if args.w_smooth is not None
                  else float(tcfg.get("w_smooth", 0.0))),
        w_sem=(args.w_sem if args.w_sem is not None
               else float(tcfg.get("w_sem", 0.0))))
    if loss_cfg.w_sem > 0 and not sem_aux_head:
        print("[error] --w-sem > 0 requires --sem-aux-head (the auxiliary "
              "semantic head produces the logits the CE loss consumes).")
        return 1
    loss_fn = DepthLoss(loss_cfg)
    print(f"[i] loss: {loss_cfg}")
    grad_clip = float(tcfg.get("grad_clip", 5.0))
    epochs = args.epochs if args.epochs is not None else tcfg["epochs"]
    patience = args.patience if args.patience is not None else int(tcfg.get("patience", 8))
    k_sub = args.val_subset if args.val_subset is not None else tcfg["val_subset"]

    out_dir = Path(paths["outputs_dir"]) / "calib_net" / (args.out_tag or "dn_only")
    out_dir.mkdir(parents=True, exist_ok=True)

    # ONCE, before training: constructed at the TRUE base LR, T_max = total epochs.
    # Stepped ONCE PER EPOCH below — never inside the batch loop (worklog incident #2).
    # The --use-dem path uses the SAME scheduler instance and the SAME single
    # step-per-epoch discipline — only the channel count changes.
    sched = (torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
             if args.cosine else None)

    history, best_mae, best_epoch, bad = [], float("inf"), -1, 0
    for epoch in range(1, epochs + 1):
        t0, run_loss, nb = time.time(), 0.0, 0
        for batch in dl:
            dn = batch["dn"].to(device, non_blocking=True)
            agl = batch["agl"].to(device, non_blocking=True)
            rgb = batch["rgb"].to(device, non_blocking=True) if use_rgb else None
            dem = (batch["dem"].to(device, non_blocking=True)
                   if use_dem and batch.get("dem") is not None else None)
            sem = (batch["sem_onehot"].to(device, non_blocking=True)
                   if use_sem and batch.get("sem_onehot") is not None else None)
            opt.zero_grad(set_to_none=True)

            # CUDA AMP keeps model activations mostly in FP16 while
            # preserving optimizer/update stability with GradScaler.
            with torch.autocast(
                device_type="cuda" if amp_enabled else "cpu",
                dtype=torch.float16 if amp_enabled else torch.float32,
                enabled=amp_enabled,
            ):
                out = net(dn, rgb, dem, sem)
                pred = out["pred"]

                # Composite loss (Phase 4): extra terms contribute ONLY when
                # their weights are > 0.
                loss = loss_fn(
                    pred, agl,
                    rgb=(batch["rgb"].to(device, non_blocking=True)
                         if (use_rgb and loss_cfg.w_smooth > 0) else None),
                    sem_logits=out.get("sem_logits"),
                    sem_target=(
                        batch["sem_onehot"].to(device, non_blocking=True)
                        if (loss_cfg.w_sem > 0
                            and batch.get("sem_onehot") is not None)
                        else None
                    ),
                    sem_ignore=(
                        batch["sem_ignore"].to(device, non_blocking=True)
                        if (loss_cfg.w_sem > 0
                            and batch.get("sem_ignore") is not None)
                        else None
                    ),
                )

            scaler.scale(loss).backward()

            # Unscale before gradient clipping so grad_clip remains in
            # the original FP32 gradient scale.
            scaler.unscale_(opt)

            torch.nn.utils.clip_grad_norm_(
                net.parameters(),
                grad_clip,
            )

            scaler.step(opt)
            scaler.update()
            run_loss += float(loss.detach())
            nb += 1
        # ONCE per epoch, aligned with val_subset_mae — NOT inside the batch loop
        if sched is not None:
            sched.step()
        mae = val_subset_mae(net, ds["val"], k_sub, use_rgb, device,
                             use_dem=use_dem, use_sem=use_sem)
        history.append({"epoch": epoch, "train_loss": run_loss / max(nb, 1),
                        "val_subset_mae": mae, "lr": opt.param_groups[0]["lr"],
                        "sec": round(time.time() - t0, 1),
                        "loss_parts": loss_fn.last_parts})
        star = ""
        if mae < best_mae - 1e-4:
            best_mae, best_epoch, bad, star = mae, epoch, 0, "  <- best"
            torch.save({
                "model_state": net.state_dict(),
                "use_rgb": use_rgb,
                "use_dem": use_dem,           # Method-D challenger marker;
                "use_sem": use_sem,           # Exp 4/5 marker (Phase 4)
                "sem_classes": sem_classes,   # K of the one-hot block
                "sem_aux_head": sem_aux_head,
                "in_ch": in_ch,               # explicit, future-proofs the
                                              # checkpoint against future
                                              # variants (derive_in_ch source)
                "widths": list(mcfg["widths"]),
                "clamp_min": mcfg.get("clamp_min", 0.0),
                "affine_init": {"a": a0, "b": b0},
                "loss": loss_cfg.main,
                "loss_weights": {"w_grad": loss_cfg.w_grad,
                                 "w_smooth": loss_cfg.w_smooth,
                                 "w_sem": loss_cfg.w_sem},
                "epoch": epoch,
                "amp_enabled": amp_enabled,
                "scaler_state": scaler.state_dict(),
                "val_subset_mae": mae,
                "splits_json": str(paths["splits_json"]),
                "dataset": (dataset_name or "dfc2019"),
                "mixed_weights": mixed_weights_note,
                # SYNTHETIC-DEM-PROXY marker travels INTO the checkpoint so
                # any downstream evaluate/infer command can tell at a glance
                # whether the model trained on a synthetic DEM. Per
                # demprior.py's contract, NEVER present such a model's win
                # as a citable "DEM conditioning works" claim.
                "dem_source": ("SYNTHETIC-DEM-PROXY" if synth_dem
                               else (f"dem_dir:{dem_dir.name}"
                                     if dem_dir else None)),
                "created": datetime.now(timezone.utc).isoformat(),
            }, out_dir / "best.pt")
        else:
            bad += 1
        print(f"[ep {epoch:3d}] loss {run_loss/max(nb,1):7.4f}  "
              f"val_mae[{k_sub}] {mae:6.3f} m  ({time.time()-t0:5.1f}s){star}")
        dump_json({"history": history, "best": {"epoch": best_epoch, "mae": best_mae},
                   "config": {"use_rgb": use_rgb, "use_dem": use_dem,
                              "use_sem": use_sem, "sem_classes": sem_classes,
                              "sem_aux_head": sem_aux_head,
                              "in_ch": in_ch, "loss": loss_cfg.main,
                              "loss_weights": {"w_grad": loss_cfg.w_grad,
                                               "w_smooth": loss_cfg.w_smooth,
                                               "w_sem": loss_cfg.w_sem},
                              "epochs": epochs,
                              "lr": args.lr or tcfg["lr"], "batch": args.batch_size or tcfg["batch_size"],
                              "crop": tcfg["crop_size"], "train_tiles": len(ds["train"]),
                              "device": device,
                              "dataset": (dataset_name or "dfc2019"),
                              "mixed_weights": mixed_weights_note,
                              "dem_source": ("SYNTHETIC-DEM-PROXY" if synth_dem
                                              else (str(dem_dir) if dem_dir else None))}},
                  out_dir / "train_log.json")
        if bad >= patience:
            print(f"[i] early stop at epoch {epoch} (no improvement for {patience}).")
            break

    print(f"[i] best subset MAE {best_mae:.3f} m @ epoch {best_epoch} -> {out_dir/'best.pt'}")
    print("[i] now run: python model.py evaluate --config configs/phase2.yaml "
          "--checkpoint " + str(out_dir / "best.pt"))
    return 0
