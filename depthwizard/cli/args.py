"""Shared argparse / config helpers for CLI command modules.

Keeps per-command boilerplate tiny and the config-loading rules identical
across commands: YAML file -> dict, paths resolved lazily by consumers.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Optional

import yaml


def load_config(path: Path | str) -> Dict:
    """Read a YAML config into a dict (raises with a clear message if absent)."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"config not found: {path} — copy configs/phase1.yaml and edit "
            f"the paths for your machine.")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def add_config_arg(p: argparse.ArgumentParser, default: str) -> None:
    p.add_argument("--config", type=Path, default=Path(default),
                   help="YAML config (single source of truth for paths)")


def add_cache_subdir_arg(p: argparse.ArgumentParser) -> None:
    p.add_argument("--cache-subdir", default=None,
                   help="model-tagged subdir under depth_cache_dir "
                        "(auto-resolved when unambiguous)")


def add_device_arg(p: argparse.ArgumentParser) -> None:
    p.add_argument("--device", default=None,
                   help="cuda | cpu | auto (default: auto)")


def resolve_device(device: Optional[str]) -> str:
    """'auto' or None -> cuda if available (torch is imported only here)."""
    if device and device != "auto":
        return device
    import torch
    return "cuda" if torch.cuda.is_available() else "cpu"


def resolve_cache(paths: Dict, cache_subdir: Optional[str]) -> Path:
    """Depth-cache directory, never mixing backbones (see geo.resolve_cache_dir)."""
    from depthwizard.geo import resolve_cache_dir
    return resolve_cache_dir(paths["depth_cache_dir"], cache_subdir)
