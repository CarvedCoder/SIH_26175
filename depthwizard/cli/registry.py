"""Command registry — the single source of truth for the CLI surface.

Lazy-subcommand design (standard for heavy CLIs like git/docker):
    * ``model.py --help``          builds a SKELETON parser from the static
                                  table below — zero command-module imports,
                                  works before torch/transformers install.
    * ``model.py <cmd> ...``       imports ONLY that command's module, builds
                                  its full parser, runs it.

The (module, name, help) triples below are duplicated in each command
module's NAME/HELP — tests/test_cli.py asserts they stay in sync.

Governance note: the ``evaluate`` command mirrors the certified 09 eval
logic. FINAL / citable numbers come ONLY from it — every other command is
exploration, demonstration, or bookkeeping.
"""

from __future__ import annotations

import argparse
import importlib
import sys
from types import ModuleType
from typing import Dict, List, Optional

# (module under depthwizard.cli, subcommand name, one-line help)
# Canonical pipeline order.
SPECS: List[tuple] = [
    (
        "inspect_dataset",
        "inspect",
        "dataset structure/alignment/class inspection + quicklooks",
    ),
    (
        "make_splits",
        "splits",
        "freeze scene-level train/val/test splits (block mode default)",
    ),
    (
        "dataset_stats",
        "stats",
        "per-dataset AGL/Dn statistics + normalization verification "
        "(REQUIRED gate before mixed training)",
    ),
    (
        "precompute_depth",
        "depth",
        "precompute the Depth-Anything-V2 raw depth cache (.npy per tile)",
    ),
    (
        "fit_baseline",
        "fit-baseline",
        "fit the global affine baseline H = a*Dn + b (train split only)",
    ),
    (
        "eval_baseline",
        "eval-baseline",
        "masked evaluation of the global affine baseline + error maps",
    ),
    (
        "dummy_baselines",
        "dummies",
        "constant-predictor floors (zero / train-mean / train-median)",
    ),
    (
        "reference_table",
        "reference",
        "merge baseline + dummies into the frozen reference card / gates",
    ),
    (
        "train_calibration",
        "train",
        "train the height model (CalibrationNet | RDAH fine-tuning)",
    ),
    (
        "eval_calibration",
        "evaluate",
        "CITABLE eval of the height model (CalibrationNet | RDAH) vs frozen gates (val [+test])",
    ),
    (
        "eval_postprocess",
        "eval-postprocess",
        "ablation + boundary-aware eval of AGL post-processing on a frozen split",
    ),
    (
        "infer",
        "infer",
        "image -> AGL/DSM with the flagship (demo path, supports Track-2 anchoring)",
    ),
    (
        "bench_model",
        "bench",
        "benchmark a height backend (resolution sweep, params, peak VRAM)",
    ),
    (
        "eval_scene",
        "eval-scene",
        "predicted DSM vs truth AGL for one scene (diagnostic, NOT citable)",
    ),
    (
        "gt_check",
        "gt-check",
        "cross-check one predicted scene vs its AGL truth (frozen protocol)",
    ),
    ("diag_rgb", "diag", "tensor/init/gradient diagnostics for a pinned training run"),
    ("serve", "serve", "launch the FastAPI inference service used by the webapp"),
    ("disaster", "disaster", "run disaster assessment (building detection + damage) on a post-disaster image"),
]

COMMANDS: Dict[str, str] = {name: mod for mod, name, _h in SPECS}
HELPS: Dict[str, str] = {name: h for _m, name, h in SPECS}

_EPILOG = (
    "Pipeline order: inspect -> splits -> depth -> [stats: multi-"
    "dataset verification gate] -> fit-baseline -> eval-baseline -> "
    "dummies -> reference -> train -> evaluate -> infer | serve. "
    "Citable numbers: `evaluate` only."
)


def load_command(name: str) -> ModuleType:
    """Import a command module on demand."""
    if name not in COMMANDS:
        raise KeyError(f"unknown command '{name}'")
    return importlib.import_module(f"depthwizard.cli.{COMMANDS[name]}")


def _skeleton_parser() -> argparse.ArgumentParser:
    """All command names + helps, NO command-module imports."""
    ap = argparse.ArgumentParser(
        prog="model.py",
        description="DepthWizard (SIH26175) — single-view aerial RGB -> "
        "LiDAR-derived height (AGL) estimation + 3D flythrough.",
        epilog=_EPILOG,
    )
    sub = ap.add_subparsers(dest="command", metavar="<command>")
    for _mod, name, help_text in SPECS:
        sub.add_parser(name, help=help_text)
    return ap


def _full_parser(mod: ModuleType) -> argparse.ArgumentParser:
    """Full parser hosting exactly one command's real subparser."""
    ap = argparse.ArgumentParser(prog="model.py", epilog=_EPILOG)
    sub = ap.add_subparsers(dest="command", metavar="<command>")
    mod.add_parser(sub)
    return ap


def main(argv: Optional[List[str]] = None) -> int:
    argv = list(sys.argv[1:]) if argv is None else list(argv)

    # Stage 1: resolve the command token without importing any command module.
    cmd = argv[0] if (argv and argv[0] in COMMANDS) else None
    if cmd is None:
        skel = _skeleton_parser()
        args, extra = skel.parse_known_args(argv)
        # argparse auto-handles --help / bad-choice exits here.
        if not getattr(args, "command", None):
            skel.print_help()
            return 0
        cmd, argv = args.command, [args.command] + extra

    # Stage 2: import ONLY the invoked command, parse its real arguments.
    mod = load_command(cmd)
    ap = _full_parser(mod)
    args = ap.parse_args(argv)
    try:
        return mod.run(args) or 0
    except KeyboardInterrupt:
        print("\n[interrupt] aborted by user", file=sys.stderr)
        return 130
