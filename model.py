"""Depthizard (SIH26175) — unified entry point.

Every pipeline stage fires from this file:

    python main.py inspect      --rgb-dir ... --truth-dir ...     # dataset audit
    python main.py splits       --rgb-dir ... --truth-dir ...     # freeze splits
    python main.py depth        --rgb-dir ... --device auto       # DAv2 cache
    python main.py fit-baseline --config configs/phase1.yaml      # H = a*Dn + b
    python main.py eval-baseline --config configs/phase1.yaml     # masked eval
    python main.py dummies      --config configs/phase1.yaml      # constant floors
    python main.py reference    --config configs/phase1.yaml      # frozen gates
    python main.py train        --config configs/phase2.yaml      # calibration net
    python main.py evaluate     --config configs/phase2.yaml      # CITABLE numbers
    python main.py infer        --input scene.tif                 # demo DSM
    python main.py eval-scene   --pred dsm.tif --truth AGL.tif    # diagnostics
    python main.py gt-check     --pred dsm.npy --truth AGL.tif    # smoke protocol
    python main.py diag         --config configs/phase2.yaml      # run diagnostics
    python main.py serve        --port 8000                       # webapp backend

    python main.py --help        # all commands
    python main.py <command> --help

Governance (frozen): FINAL / citable numbers come ONLY from `evaluate`.
"""

import sys


def main() -> int:
    # Ensure the repo root is importable no matter where python is invoked
    # from (double-click, IDE, pytest, service worker...).
    from pathlib import Path

    root = Path(__file__).resolve().parent
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    from depthwizard.cli.registry import main as cli_main

    return cli_main()


if __name__ == "__main__":
    raise SystemExit(main())
