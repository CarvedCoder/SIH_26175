"""Depthizard (SIH26175) — unified entry point.

Every pipeline stage fires from this file:

    python model.py inspect      --rgb-dir ... --truth-dir ...     # dataset audit
    python model.py splits       --rgb-dir ... --truth-dir ...     # freeze splits
    python model.py depth        --rgb-dir ... --device auto       # DAv2 cache
    python model.py fit-baseline --config configs/phase1.yaml      # H = a*Dn + b
    python model.py eval-baseline --config configs/phase1.yaml     # masked eval
    python model.py dummies      --config configs/phase1.yaml      # constant floors
    python model.py reference    --config configs/phase1.yaml      # frozen gates
    python model.py train        --config configs/phase2.yaml      # calibration net
    python model.py train        --config configs/rdah_gamus.yaml # RDAH fine-tune
    python model.py evaluate     --config configs/phase2.yaml      # CITABLE numbers
    python model.py evaluate     --model rdah --dataset gamus \
        --checkpoint checkpoints/rdah/rdah_track1_best_model.pth   # rdah A/B
    python model.py infer        --input scene.tif                 # demo DSM (rdah default)
    python model.py bench        --architecture rdah               # resolution/VRAM sweep
    python model.py eval-scene   --pred dsm.tif --truth AGL.tif    # diagnostics
    python model.py gt-check     --pred dsm.npy --truth AGL.tif    # smoke protocol
    python model.py diag         --config configs/phase2.yaml      # run diagnostics
    python model.py serve        --port 8000                       # webapp backend

    python model.py --help        # all commands
    python model.py <command> --help

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
