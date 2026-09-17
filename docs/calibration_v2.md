# CalibrationNet V2 — Research Stack

Status: **framework implemented; first ablation round complete on the
local GAMUS harness (§8).** This document is updated as results land.
Nothing in the V2
stack replaces the V1 pipeline; everything is behind configuration flags
and the V1 flagship (`outputs/calib_net/gamus_dfc_mixed_80_20/best.pt`)
remains the reference baseline.

---

## 1. Baseline

### Flagship reference (frozen — NOT reproducible in this environment)

```
checkpoint : outputs/calib_net/gamus_dfc_mixed_80_20/best.pt
training   : mixed DFC + GAMUS (80/20)
evaluation : GAMUS validation, 859 tiles
MAE 2.770 | RMSE 5.239 | r 0.643 | bias -0.749 | building 3.705
DC MAE 4.402 | PHL MAE 1.599
```

Manifest: `experiments/baselines/gamus_dfc_mixed_baseline.yaml`.

**Honesty note (Section-0 audit, recorded in `configs/exp_local_gamus.yaml`):
this environment has no DFC2019 rasters and no mixed stats-gate artifact,
so the 859-tile mixed protocol cannot be re-run here.** All V2 ablations
therefore use the **local real-GAMUS harness**: 78 official-GAMUS train
tiles + 48 val tiles (DC city), cached Depth-Anything-V2-Base Dn,
`configs/exp_local_gamus.yaml` recipe (crop 512, batch 1, epochs 40,
lr 1e-3, wd 1e-4, grad clip 5, masked-L1, seed 42, patience 10). The
baseline column of the comparison table is a **re-trained V1 CalibrationNet
under this exact recipe** — internally consistent with the V2 runs, but
**NOT comparable to the quoted 2.770 numbers**. Do not mix the two
protocols in any claim.

### Local-harness reference (this environment)

`outputs/calib_net/exp_baseline/` — full 48-tile val evaluation
(`eval_calib_gamus.json`):

| Metric | Value |
|---|---:|
| MAE | 4.459 m |
| RMSE | 6.623 m |
| Pearson r | 0.776 |
| bias | −1.291 m |
| building MAE | 3.960 m |
| per-tile MAE | 4.459 ± 1.861 m |

PHL MAE is n/a on this split (no PHL tiles) — never fabricated.

---

## 2. Architecture

Single modular `CalibrationNet` (depthwizard/calibration_net.py) replaces
the copy-per-variant approach. All variants share:

```
inputs -> encoder(widths) -> [context] -> decoder -> head -> H
```

### Parameterizations (`model.parameterization`)

| Mode | Formula | Init |
|---|---|---|
| `absolute_affine` (V1) | `H = clamp(a·Dn + b)` | head zero-weight, bias `(a0,b0)` |
| `residual_affine` | `H = clamp((a0+Δa)·Dn + (b0+Δb))` | head zero-init ⇒ starts exactly at `a0·Dn+b0` |
| `hybrid_residual` | `H = clamp((a0+Δa)·Dn + (b0+Δb) + ΔH)` | same |
| `residual_depth` | `H = clamp(a0·Dn + b0 + ΔH)` | same |

The zero-init head preserves the V1 invariant: **every variant starts
training exactly at the global affine baseline** and can only climb.
`a0,b0` always come from the checkpoint's `affine_init` (the frozen
`global_affine.json`), never hardcoded.

### Bounded residuals (`model.bounded: true`)

```
a = a0 + α·tanh(Δa),  b = b0 + β·tanh(Δb)      α=1.0, β=10.0
```

Keeps the calibration fields within physically plausible range of the
global affine. Diagnostics (min/max/mean/std of Δa, Δb, a, b) are logged
by the model summary; a saturated tanh (|Δ| pinned at the bound) means the
bounds were too tight — diagnose before widening.

### Configurable widths (`model.widths`)

`[16,32,64]` (V1) up to `[32,64,128,256]` without code changes; parameter
count printed at train start and stored in the checkpoint.

### Context module (`model.context`)

`none` (default) | `aspp` — ASPP-lite bottleneck (1×1, dil 2, dil 4,
global-pool branches at width/4, projected back), ~35k params at width 64.
Rationale: the ~50 px effective receptive field of the V1 encoder cannot
see the tile-level normalization context the affine fields depend on
(Chen et al., DeepLab ASPP, TPAMI 2017).

### Fusion (`fusion.mode`)

`early` (V1 concat) | `dual_encoder` — separate lightweight Dn and
RGB(+aux) encoders, bottleneck feature fusion (concat), summed skips.
Not assumed better; ablated.

### Semantic modes (`model.semantic_mode`)

`input` (V1 GT one-hot channels — privileged, legacy) | `auxiliary`
(predicted-semantics aux head, no GT input) | `joint` (predicted
semantics modulate the calibration decoder) | `none`. GT semantics are
never used at deployment; the loader maps legacy `sem_input` metadata
correctly and refuses incompatible checkpoints loudly.

### Uncertainty head (`use_uncertainty`)

Optional `log_var` output trained with the L1-scale heteroscedastic NLL
(`err·exp(−s) + s`, Kendall & Gal 2017 robust variant). Off by default.

---

## 3. Losses

Single source: `depthwizard/losses.py` (`DepthLoss` + `LossConfig`).
All extra terms default to 0 = exact pre-V2 behavior (pinned by tests).

| Term | Config | Notes |
|---|---|---|
| masked L1 (default) | `loss.main: l1` | matches MAE metric |
| Huber | `loss.main: huber` | forward Huber, δ=5 m |
| **BerHu** | `loss.main: berhu` | reverse Huber (Laina et al. 2016); L1 below c, quadratic above; c=0.2·max\|e\| per batch |
| gradient | `loss.gradient_weight` | L1 on first-order differences, masked |
| edge-aware smoothness | `loss.boundary_weight` | \|∇pred\|·exp(−\|∇rgb\|) |
| semantic CE | `loss.semantic_weight` | aux-head logits vs GT one-hot |
| height-balanced | `loss.height_balanced` | bins 0–10/10–30/>30 m ⇒ ×1/×2/×5 |
| uncertainty | `loss.uncertainty_weight` | heteroscedastic NLL on log_var |

---

## 4. Configuration

Everything is config-driven (`depthwizard/config.py` dataclasses;
`configs/v2_default.yaml` reproduces V1 exactly). CLI flags override YAML;
legacy configs (no `inputs:`/`loss:` sections) keep working unchanged.

```yaml
model:
  widths: [16, 32, 64, 128]
  parameterization: residual_affine   # absolute_affine | residual_affine | hybrid_residual | residual_depth
  bounded: true
  context: aspp                       # none | aspp
  semantic_mode: auxiliary            # input | auxiliary | joint | none
fusion: {mode: early, merge: concat}  # early | dual_encoder
inputs: {depth: true, rgb: true, confidence: false, semantic: false, dem: false}
loss: {main: berhu, gradient_weight: 0.0, boundary_weight: 0.0,
       semantic_weight: 0.0, uncertainty_weight: 0.0, height_balanced: false}
```

## 5. Checkpoint compatibility

V2 checkpoints store `semantic_mode`, `parameterization`, `bounded`,
`fusion_mode`, `context_module`, `use_uncertainty`. Loaders
(`tifops.load_calib_net`, `eval_calibration`) honor every field; legacy
checkpoints (pre-Exp-4 and Exp-4 `sem_input` era) rebuild bit-identically;
metadata/state-dict disagreement fails loudly (pinned in
`model_tests/test_v2_stack.py`).

---

## 6. Experiments (local GAMUS harness, 78/48 tiles)

All runs: identical recipe to `exp_baseline` (§1), one factor changed each.
Full 48-tile val evaluation via `model.py evaluate --dataset gamus`.
Runner: `scratch/run_v2_ablations.sh`; configs `configs/v2_local_a*.yaml`.

| ID | Change | Status |
|---|---|---|
| baseline | V1 re-train, local recipe | done |
| A1 | widths [32,64,128] | done |
| A2 | widths [16,32,64,128] | done |
| A3 | residual_affine | done |
| A4 | residual_affine + bounded | done |
| A5 | multi-scale + residual_affine | done |
| A6 | ASPP-lite context | done |
| A9 | BerHu loss | done |

## 7. Known limitations / failure modes

* The 859-tile mixed protocol is not runnable here; all local numbers are
  48-tile DC-city only (urban-dominated: 34/48 urban tiles).
* **Single seed, 48 tiles**: differences of ≲0.2 m MAE between runs are
  within noise. Trends, not verdicts.
* Height-balanced bins (10/30 m) are hardcoded pending a data-derived
  binning pass.
* `max_shift` (bounded ΔH scale, 10 m) is not yet configurable.
* Confidence-as-input (`inputs.confidence`) is plumbed in the dataset
  layer but no confidence channel producer is wired end-to-end yet.

## 8. Results

Full 48-tile val, identical protocol for every row
(`outputs/calib_net/<tag>/eval_calib_gamus.json`):

| Run | Params | best ep | MAE | RMSE | r | bias | Bld MAE | tile-MAE SD |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline (V1) | 117,138 | 33 | 4.459 | 6.623 | 0.776 | −1.291 | 3.960 | 1.861 |
| A1 wider [32,64,128] | 466,722 | 37 | 4.451 | 6.517 | 0.779 | **−0.271** | 3.937 | 1.510 |
| A2 multiscale [16,32,64,128] | 481,746 | 31 | 4.587 | 6.755 | 0.773 | −1.102 | 4.245 | — |
| A3 residual_affine | 117,138 | 36 | **4.329** | **6.429** | **0.790** | −1.243 | 3.996 | — |
| A4 residual+bounded | 117,138 | 36 | 4.676 | 6.918 | 0.761 | −1.789 | 3.875 | — |
| A5 multiscale+residual | 481,746 | 26 | 4.390 | 6.400 | 0.787 | −0.598 | 3.884 | — |
| A6 ASPP-lite | 141,971 | 13 | 4.628 | 6.820 | 0.766 | −1.701 | 4.287 | — |
| A9 BerHu | 117,138 | 25 | 4.593 | 6.625 | 0.772 | −1.092 | 4.434 | — |

### Reading the results (honest interpretation)

* **A3 (residual affine) is the best single change**: best MAE (−2.9%),
  best RMSE, best r — at *zero* extra parameters. The V1 diagnosis
  (free per-pixel a,b waste capacity re-learning the global affine) is
  supported.
* **A5 (multiscale + residual) improves the broadest set of criteria**:
  MAE −1.5%, RMSE −3.4%, r +0.011, |bias| −54% (−1.291 → −0.598),
  building MAE −1.9%. It is the recommended V2 candidate, at the cost of
  ~4× parameters (482k — still tiny).
* **A1 (wider) barely moves MAE but hugely improves bias** (−0.271) and
  per-tile variance. Capacity alone helps stability, not accuracy.
* **Negative results (reported, not hidden):**
  * **A4 (bounded residual) is worse than A3 on every global metric**
    (MAE +0.35, bias −1.789). The tanh bounds (α=1, β=10) either pinch
    legitimate field variation or hurt optimization at this data scale.
    Do not enable `bounded` without re-tuning the scales.
  * **A6 (ASPP-lite) overfits** — best epoch 13 vs 26–37 for everything
    else, and worse MAE/RMSE/bias. With 78 training tiles, the extra
    receptive field buys memorization, not generalization. Consistent
    with the earlier research report's overfit warning.
  * **A9 (BerHu) does not help here** (MAE 4.593, building MAE 4.434 —
    the worst building error of the suite). On this 78-tile subset the
    adaptive threshold may let the few tall-building pixels dominate.
    Keep BerHu available but not default.
  * **A2 (multiscale alone) is worse than baseline** (4.587); the wider
    A1 variant was the better capacity spend. Only helps when combined
    with residual parameterization (A5).

### Improvement of A5 (recommended) vs local baseline

| Metric | baseline | A5 | Δ | Δ% |
|---|---:|---:|---:|---:|
| MAE | 4.459 | 4.390 | −0.069 | −1.5% |
| RMSE | 6.623 | 6.400 | −0.223 | −3.4% |
| Pearson r | 0.776 | 0.787 | +0.011 | +1.4% |
| bias | −1.291 | −0.598 | \|bias\| −0.693 | −54% |
| building MAE | 3.960 | 3.884 | −0.076 | −1.9% |

**These are single-seed 48-tile numbers on DC-city-only data.** They
justify carrying residual-affine (+ optionally multiscale widths) into a
full mixed-train / 859-tile-val run on a machine that has the DFC data —
not for promoting any local number as final.

## 9. Reproduction

```bash
# 0) data: gamus_full symlink -> HF snapshot (78 train / 48 val tiles);
#    Dn cache outputs/depth_cache/depth_anything_v2_base_hf/
# 1) baseline re-train + eval (local protocol)
python model.py train --config configs/exp_local_gamus.yaml --use-rgb \
    --out-tag exp_baseline
python model.py evaluate --config configs/exp_local_gamus.yaml --dataset gamus \
    --checkpoint outputs/calib_net/exp_baseline/best.pt --splits val \
    --error-maps 0 --out-tag exp_baseline
# 2) any ablation (A1..A9)
bash scratch/run_v2_ablations.sh          # trains + evals all, skips done
# 3) unit/regression tests
.venv/bin/python -m pytest model_tests/ test_v2.py test_sem.py -q
```

## 10. References

* Laina et al., *Deeper Depth Prediction with Fully Convolutional
  Residual Networks*, 3DV 2016 — BerHu loss.
* Chen et al., *Rethinking Atrous Convolution (DeepLab v3)*, 2017 — ASPP.
* Kendall & Gal, *What Uncertainties Do We Need…*, NeurIPS 2017 —
  heteroscedastic NLL.
* Godard et al., *Unsupervised Monocular Depth Estimation with
  Left-Right Consistency*, CVPR 2017 — edge-aware smoothness.
* Xiong et al., *GAMUS*, arXiv:2305.14914 — dataset; long-tailed heights.
* Xiong et al., *THE Benchmark*, arXiv:2112.14985 — cross-dataset height
  transfer.
* Yang et al., *Depth Anything V2*, NeurIPS 2024 — Dn backbone.
