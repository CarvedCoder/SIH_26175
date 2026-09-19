#!/usr/bin/env bash
# V2 ablation suite on the local real-GAMUS harness (78 train / 48 val tiles).
# Same recipe as exp_baseline: 40 epochs, crop 512, batch 1, lr 1e-3, seed 42.
set -u
cd /home/carved/SIH_26175
PY=.venv/bin/python
CS="--cache-subdir depth_anything_v2_base_hf"
run() {
  tag=$1; cfg=$2
  if [ -f "outputs/calib_net/$tag/train_log.json" ] && [ -f "outputs/calib_net/$tag/best.pt" ]; then
    echo "[skip] $tag already trained"
    return
  fi
  echo "=== TRAIN $tag ($cfg) $(date -Is)"
  $PY model.py train --config "$cfg" --out-tag "$tag" $CS || { echo "[FAIL] train $tag"; return; }
  echo "=== EVAL $tag $(date -Is)"
  $PY model.py evaluate --config configs/exp_local_gamus.yaml --dataset gamus \
      --checkpoint "outputs/calib_net/$tag/best.pt" --splits val --error-maps 0 --out-tag "$tag" $CS \
      || echo "[FAIL] eval $tag"
}
run v2_a1_wider        configs/v2_local_a1_wider.yaml
run v2_a2_multiscale   configs/v2_local_a2_multiscale.yaml
run v2_a3_residual     configs/v2_local_a3_residual.yaml
run v2_a4_bounded      configs/v2_local_a4_bounded.yaml
run v2_a5_ms_residual  configs/v2_local_a5_ms_residual.yaml
run v2_a6_aspp         configs/v2_local_a6_aspp.yaml
run v2_a9_berhu        configs/v2_local_a9_berhu.yaml
echo "ALL DONE $(date -Is)"
