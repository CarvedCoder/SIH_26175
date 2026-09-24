#!/usr/bin/env bash
# Ensure the serving checkpoint exists at the repo default path:
#   outputs/calib_net/gamus_rgb_grad/best.pt
#
# Primary path: the checkpoint ships with the repo (464K), so after a fresh
# clone this script is a no-op. Fallback path: if the file is missing (or
# forced), retrain it locally with the documented GAMUS recipe:
#
#   python model.py train --config configs/exp_local_gamus.yaml \
#       --out-tag gamus_rgb_grad
#
# (configs/exp_local_gamus.yaml is the exact recipe that produced the
# committed artifact: GAMUS local subset, use_rgb, 40 epochs, masked-L1.
# The dataset itself is NOT in git — see config header for where the local
# HF snapshot of earthflow/GAMUS must be mounted at `gamus_full`.)
#
# Usage: scripts/fetch_checkpoint.sh [--force]
set -euo pipefail

cd "$(git rev-parse --show-toplevel 2>/dev/null || echo .)"

CKPT="outputs/calib_net/gamus_rgb_grad/best.pt"

if [[ -f "$CKPT" && "${1:-}" != "--force" ]]; then
    echo "[fetch_checkpoint] OK: $CKPT already present"
    exit 0
fi

if [[ ! -d gamus_full ]]; then
    cat >&2 <<'EOF'
[fetch_checkpoint] Cannot retrain: the GAMUS dataset snapshot is missing.
Expected a local HF snapshot of earthflow/GAMUS at ./gamus_full
(layout: images/ | heights/ | classes/{train,val}).
Provide it, or copy a trained best.pt into outputs/calib_net/gamus_rgb_grad/
EOF
    exit 1
fi

echo "[fetch_checkpoint] training checkpoint -> $CKPT"
python model.py train --config configs/exp_local_gamus.yaml --out-tag gamus_rgb_grad
echo "[fetch_checkpoint] done: $CKPT"
