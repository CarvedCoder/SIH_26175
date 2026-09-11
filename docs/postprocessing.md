# DepthWizard AGL post-processing — algorithms, validation, and decisions

This document is the citable reference for the `depthwizard/postprocess/`
package: what each algorithm is, why it exists, what every parameter means,
what it costs, what it cannot do, and — most importantly — what the real
validation ablation measured before anything was accepted. It separates
**implemented** (validated on real data), **experimental** (implemented,
ablation evidence incomplete or negative), and **not implemented**.

Pipeline position (frozen, verified in `depthwizard/inference.py`):

```text
RGB → Depth Anything V2 (per-1024-tile Dn min-max) → CalibrationNet
    → AGL_raw (metric metres, clamped ≥ 0)
    → POST-PROCESSING  ← this package, BEFORE anchoring
    → DEM / ground-elev anchoring (unchanged arithmetic)
    → DSM   (DSM == AGL_refined + DEM exactly where the DEM is valid)
```

Design contract (binding, test-pinned in `model_tests/test_postprocess.py`):

* the refined signal stays in **metric metres** — no normalization, no
  per-tile rescale, ever (the mean/median/std shift is measured and reported
  on every run; `preserve_mean=True` can restore the raw mean exactly);
* `enabled=False` / `method="none"` is a **bit-identical passthrough** —
  the legacy inference path is unchanged and byte-equal (regression test);
* the raw AGL is **never overwritten** (`agl_raw.npy` is written alongside
  the refined product and equals the no-postprocess baseline exactly);
* NaN/nodata pixels keep their NaN pattern; no neighbour poisoning;
* refinement happens **before** DEM anchoring — the anchored DSM is
  `refined + DEM`, pure arithmetic, never filtered;
* **no ground truth of any kind** (depth or semantics) is used at
  deployment; GT semantics appear only inside evaluation metrics.

---

## 1. Methods

### 1.1 `none` — the baseline (V0)

Identity. Exists so every comparison has an honest denominator and so
`postprocess=false` is the frozen legacy API.

### 1.2 `median` — 3×3 median + conservative spikes (V1)

`spike_removal._shifted_med_mad` with radius 1. A tiny, honest denoiser.
Measured effect on real GAMUS validation tiles: ≈ 0 (the model output is
already locally smooth). Kept because it is the cheapest sanity anchor of
the whole ladder.

### 1.3 `guided` — RGB-guided filter (V2) — **selected method**

He, Sun, Tang, *Guided Image Filtering*, ECCV 2010. RGB luma `I` is the
guidance; the calibrated AGL `p` is the signal. Per (2r+1)² window:

```text
q_i = a_i·I_i + b_i
a_i = cov(I,p) / (var(I) + eps)          b_i = mean(p) − a_i·mean(I)
q   = mean(a)·I + mean(b)                (means over the same windows)
```

* `eps` is the **edge threshold** — guidance variance below it is treated
  as noise (smoothed away), above it is structure (kept sharp). Units:
  normalized-intensity² (1 grey level ≈ 1.5e-5).
* `guided_radius` — box window radius in px. Windows are computed with
  integral images → **O(N) per tile, independent of radius**.
* Exact for constant signals (a=0, b=c ⇒ q=c) — a flat calibrated surface
  cannot be shifted (unit test).
* NaN-aware masked box statistics; edge-replicate padding.

Implementation: `postprocess/guided.py`. This is the **only method the
ablation accepted for default use** (Section 6).

### 1.4 `bilateral` — joint bilateral filter (V3)

Cross/joint bilateral: spatial Gaussian × RGB-range Gaussian
(`postprocess/bilateral.py`, fully vectorized shift-accumulators, O(N·r²)
but constant memory). `bilateral_sigma_color` (RGB L1 units, [0,1]) and
`bilateral_sigma_space` (px) trade edge preservation against denoising.
Known weakness (documented, measured): satellite RGB contains texture
edges — road markings, tree crowns, shadows — that are not height edges;
the filter over-trusts image edges. That is why it exists only as a
baseline, not as the recommendation.

### 1.5 `wls` — edge-aware weighted least squares (V3b/V4)

Farbman et al. 2008-style energy, RGB-gated, solved with preconditioned
conjugate gradients (`postprocess/wls.py`):

```text
E(z) = Σ_i c_i (z_i − d_i)²  +  λ Σ_(i,j) w_ij (z_i − z_j)²
w_ij = exp(−(dRGB_ij/σ_rgb)²) · sem_sim_ij^γ
(C + λL) z = C d        — sparse SPD system, Jacobi-CG, guided warm start
```

* `d` is the (possibly TTA-fused) AGL in metres — **never rescaled**;
* `c_i` is the data weight (confidence-aware variant below);
* `wls_lambda` (λ): smoothness vs fidelity. λ=0 is the exact identity
  (unit test); λ=1 is close to a no-op on real data (measured); λ=24 still
  conservative (Section 6 sweep);
* `wls_sigma_rgb`: neighbour RGB L1 distance at which the smoothness
  weight falls to exp(−1);
* `wls_tol` / `wls_max_iter`: CG stopping rules.

The data term pins the solution to the neural prediction — the output can
only move where evidence allows, so the global height scale cannot drift
(mean shift measured on every run; ≤ 0.03 m in the accepted configs).

### 1.6 `conf_wls` — confidence-weighted WLS (V4)

Same solve with `c_i = clip(confidence,0,1)^w` floored at
`confidence_floor`. **Confidence is a relative weight, not a calibrated
probability** — the repository's model emits no uncertainty, so
`postprocess/confidence.py` constructs the map from deployment-observable
signals only:

1. `local_stability` — 1 − normalized |x − local median| (spikes get low
   confidence);
2. `edge_consistency` — penalizes height edges without RGB support (the
   classic monocular hallucination) while leaving textured flat ground
   alone;
3. `tta_agreement` — 1 − normalized per-pixel spread across the TTA
   ensemble, when TTA runs (the strongest available instability signal);
4. validity — non-finite AGL/RGB ⇒ exactly 0.

`confidence_weight` (exponent) and `confidence_floor` (minimum anchor, so
the solve can never run away) are the knobs. No calibration curve or
coverage claim is made or implied anywhere.

### 1.7 `semantic_wls` — predicted-semantics boundary gating (V5/V7)

The smoothness weight is multiplied by the **Bhattacharyya coefficient**
of the neighbouring predicted class distributions
(`postprocess/semantic.py`):

```text
sim(i,j) = Σ_k sqrt(p_ik · p_jk)        w_sem = sim^γ
```

* identical distributions ⇒ 1 (smooth);
* confident-but-different classes ⇒ ≈ 0 (no smoothing across the boundary
  — exactly the roof→ground bleeding the pipeline must prevent);
* uncertain pixels stay similar to everything (smoothable) — strictly
  softer and more informative than comparing argmax labels.

`semantic_edge_weight` (γ) sharpens boundary protection;
`semantic_hard_boundary=True` cuts all smoothing across any argmax change
(brittle, experiments only). No class-ID special-casing is needed: any
confident class change drives sim→0, which covers building↔ground,
building↔road, building↔vegetation generically.

**Sources of semantics (never GT at deployment):** the checkpoint's
*predicted* auxiliary head (`tifops.make_full_predict_fn` → softmax, fed
through `inference.predict_with_semantics`), an external model, or a
user-supplied raster. When none is available the variant degrades to
RGB-only WLS **and says so in the run metadata** (no fabrication).

**Measured honesty note (Section 6):** with the checkpoints trained in
this study the head's probabilities are boundary-blind, so
`semantic_wls` ≡ `wls` on real data. The mechanism itself is validated on
synthetic fixtures (Section 5).

### 1.8 `full` — composed pipeline (V8/V9)

`refinement.refine_agl` stage order (each stage optional and configured):

```text
validity mask → spike removal → optional TTA fusion →
core refinement (semantic+confidence+WLS) → optional planar stage →
calibration report (mean/median/std raw vs refined) → clamp(≥0)
```

`full` = semantic + confidence + WLS core; `planar=True` adds the
experimental plane stage; `tta=True` adds the ensemble. The method
dispatcher, presets, and the ablation ladder all go through the same
orchestrator, so the CLI preset `full`, the ablation entry `v8_full`, and
`PostProcessConfig(method="full")` are one code path.

### 1.9 Spike removal (conservative) — part of most variants

`postprocess/spike_removal.py`. A pixel is replaced by its local median
only when **all** of:

1. `|x − med_local| > spike_tau · (1.4826·MAD + 1e-3)` (robust z);
2. *isolation* — fewer than `spike_min_isolation` of the window's pixels
   deviate from their own medians **with the same sign**, **and** no
   4-neighbour is itself a strong same-sign outlier (contiguity: a roof
   edge/corner is a *connected band* of co-deviating pixels; a spike
   touches none);
3. the local RGB gradient is below `spike_max_rgb_grad` (edge-supported
   geometry is presumed real until proven otherwise).

Roof outlines fail 3 (strong image edges) and roof corners fail 2b (the
corner ring is connected) — both protected by unit tests. The
sign-awareness of 2a matters: at a step edge both sides deviate with
opposite signs, and an unsigned count would over-count support.

### 1.10 TTA — flip/rot180 ensemble (V6, opt-in)

`postprocess/tta.py`: identity + hflip + vflip + 180°, each a **full
honest pass** (transformed RGB → live Depth-Anything-V2 backbone →
per-tile min-max Dn → CalibrationNet), mapped back to the original grid,
fused per-pixel by **median** (robust to one bad variant) or mean. The
aligned per-variant stack feeds the confidence estimator as the
TTA-disagreement signal. Runtime ≈ 4× inference (measured 13.9 s/tile vs
~0.9 s on CPU 1024²) — **off by default**.

### 1.11 `planar` — EXPERIMENTAL robust local plane fitting

`postprocess/planar.py`: inside connected components of the **predicted**
building mask (≥ `planar_min_area` px), fit `z = a·x + b·y + c` by
IRLS/Huber; accept only if inlier fraction ≥ `planar_min_inlier_frac` and
robust inlier RMSE ≤ `planar_max_residual`; replace **inliers only**
(superstructures keep the prediction). Rejected components are untouched
(a sloped or multi-facet roof must never be flattened). Status:
**experimental** — see Section 6 for why it is not enabled.

---

## 2. Configuration

Everything lives in `postprocess/config.py::PostProcessConfig` (the single
source; no experimental constants scattered in the source). Field-by-field
meanings are in the dataclass comments; the important ones:

| field | meaning | default |
|---|---|---|
| `enabled` / `method` | master switch / ladder selector | `False` / `guided` |
| `guided_radius`, `guided_eps` | window radius (px); edge threshold (norm-intensity²) | 4, 1e-3 |
| `bilateral_sigma_color/space` | RGB range / spatial Gaussian | 0.1, 3.0 |
| `wls_lambda`, `wls_sigma_rgb` | smoothness weight; RGB edge scale | 1.0, 0.1 |
| `semantic_edge_weight`, `semantic_hard_boundary` | γ exponent; hard cut | 2.0, False |
| `confidence_weight`, `confidence_floor` | exponent; min data anchor | 1.0, 0.15 |
| `spike_removal`, `spike_tau`, `spike_min_isolation`, `spike_max_rgb_grad` | conservative outlier stage | True, 6.0, 0.25, 0.15 |
| `tta`, `tta_aggregation` | ensemble switch; median/mean | False, median |
| `planar*` | experimental plane stage gates | False, 400, 1.5, 0.6, 1.0 |
| `clamp_min`, `preserve_mean` | the model's own ≥0 clamp; exact mean restore | 0.0, False |

Presets (`config_from_preset`): `none | median | guided | bilateral | wls |
conf | semantic | planar | full` — the API surface of task §19, exposed by
`model.py infer --postprocess <preset> [--pp-param k=v ...]` and
`run_inference(postprocess=..., postprocess_params=...)` in the service.
Default behaviour (`none`) is **backward compatible**: the legacy path is
skipped entirely and no postprocess artefacts are written.

## 3. API and integration

* `refine_agl(agl, rgb_u8, config, sem_probs=None, confidence=None,
  tta_predict_fn=None) -> (agl_refined, PostProcessReport)` — pure,
  deterministic, no file IO. The report records method, config, runtime,
  changed-pixel count, spike count, WLS iterations, TTA augmentations,
  the **calibration audit** (raw/refined mean/median/std + shifts) and
  every degradation note.
* `inference.run_inference(..., postprocess="guided",
  postprocess_params={"guided_radius": 12}, tta=False)` — the frozen
  ordering; writes `agl_raw.npy`, `postprocess_meta.json`, `dsm.npy`
  (+ `dsm.tif` when georeferenced, CRS/transform preserved — test-pinned).
* Semantic exposure: `predict_with_semantics()` always returns a dict
  `{"pred", "sem_probs"}` (`sem_probs=None` for headless checkpoints —
  fixed during this work; previously a bare array crashed the caller).

Example (real run, real GAMUS val tile DC_02_26, CPU):

```bash
python model.py infer --config configs/gamus_postproc.yaml \
    --checkpoint outputs/calib_net/postproc_flagship_v2/best.pt \
    --input demo_DC_02_26.png --dn demo_DC_02_26_dn.npy \
    --out outputs/infer_demo_guided --device cpu --no-live \
    --postprocess guided --pp-param guided_radius=12 \
    --json-out outputs/infer_demo_guided/payload.json
```

## 4. Evaluation methodology

* **Frozen split governance:** the ablation runs on the frozen GAMUS
  **validation** split only (`outputs/splits/gamus_splits.json`, 24 real
  tiles, deterministic official-split order); the test split was never
  touched during method selection. Hyperparameters (guided radius, WLS
  λ) were swept on validation, as the research plan mandates.
* **Certified forward path:** every variant scores the SAME per-tile
  model output (`_forward_tile` — full-tile min-max Dn, ImageNet-normalized
  RGB, the checkpoint's own predicted semantics).
* **Metrics:** the frozen `depthwizard.metrics` height suite (MAE, RMSE,
  medAE, bias, Pearson r, negative fraction, building MAE/RMSE, per-class
  strata) **unchanged**, plus the new geometry suite
  (`postprocess/metrics.py`): gradient MAE/RMSE, boundary-band MAE/RMSE
  around GT class changes (building, building↔ground, building↔vegetation,
  …), discontinuity jump-ratio (mean |pred jump| / mean |GT jump| across
  GT building boundaries — 1.0 preserves discontinuities, <1 blurred
  edges), and 1024-px tile-seam error. Per-class region reporting
  (building/ground/vegetation/…) is included — global MAE alone is never
  the acceptance criterion.
* **Acceptance rules** (`postprocess/acceptance.py`, tolerances are CLI
  flags, never hard-coded): MAE/RMSE/building-MAE must improve or stay
  within tolerance; |bias|, boundary MAE, gradient MAE, seam MAE must not
  materially worsen; negative AGL fraction must not increase; the global
  mean shift must stay below `calib-tol` (absolute-height guard). The
  verdict is PASS only if every rule passes; missing metrics are recorded
  as *skipped*, never silently passed. **Visual quality is deliberately
  not a rule.** TTA-budget variants are judged against the baseline
  **restricted to the same tiles** (tile-aligned comparison).
* **Ablation runner:** `model.py eval-postprocess` (ladder V0…V9), writes
  `ablation.csv` + `ablation.json` + `report.md` + per-tile diagnostics
  (10-panel figures incl. raw depth, gradient-difference map, boundary
  band overlay; zoomed crops around roofs, building-ground edges,
  vegetation boundaries, roads, terrain transitions, worst outliers) and
  per-tile array dumps (`raw_depth_dn.npy`, `raw_agl.npy`,
  `agl_gt.npy`, `refined_agl_<variant>.npy`, `gt_sem_onehot.npy`).

## 5. Test suite (77 new tests, all green)

`model_tests/test_postprocess.py` covers: config validation and presets;
bit-identical passthrough when disabled; no input mutation; constant-AGL
invariance for **every** method (calibration guard); RGB-aligned edge
preservation (guided + WLS); NaN/nodata pattern preservation and
no-poisoning; the ≥0 clamp guard (and its opt-out); same-class smoothing
vs cross-class boundary protection (semantic WLS bleeds **4.5× less**
than RGB-only WLS on the weak-RGB-evidence fixture at λ=50 — the
mechanism, proven); calibration invariants incl. the `preserve_mean`
sign fix; spike removal (isolated spike removed, contiguous roof band and
RGB-edge-supported values kept, τ configurable); guided-filter eps
semantics; WLS internals (λ=0 identity, constant exact, confidence
pinning, invalid pixels); confidence properties; TTA round-trip
alignment, median robustness, and the mean-aggregation regression test;
planar accept/reject/skip; all metric functions; acceptance rules; and
six end-to-end `run_inference` integration tests (legacy-baseline
equality, `agl_raw.npy` == baseline, metadata contract, semantic-head
exposure + honest headless degradation, GeoTIFF CRS/transform
preservation, anchor-after-refine arithmetic). Full repository suite:
**291 tests passing**.

## 6. Ablation results (REAL GAMUS validation data)

Setup: checkpoint `postproc_flagship_v2` — CalibrationNet (Dn+RGB +
**predicted** semantic aux head; the GT-input Exp-4 design was measured
and *rejected*: its deployed zero-filled path collapses to
Pearson −0.28 vs +0.65 privileged, a documented train/deploy shift),
27 epochs × 24 real train tiles, CPU, 512-px crops. Frozen val split:
24 real 1024² GAMUS tiles. Same checkpoint, same tiles, same metrics for
every row; TTA rows cover the first 8 (6 for v9) tiles with
tile-aligned acceptance. Full table: `outputs/postprocess/ablation_r2_merged/`.

| variant | MAE | RMSE | bldg MAE | boundary MAE | grad MAE | calib shift | ms/tile | verdict |
|---|---|---|---|---|---|---|---|---|
| V0 raw | 5.268 | 7.631 | 5.195 | 4.632 | 0.3805 | — | ~1 | BASELINE |
| V1 median | 5.267 | 7.629 | 5.195 | 4.631 | 0.3786 | −0.003 | 6975* | PASS |
| V2 guided r=4 | 5.236 | 7.567 | 5.150 | 4.597 | 0.3710 | +0.003 | 6364* | PASS |
| V2 guided r=8 | 5.207 | 7.489 | 5.089 | 4.574 | 0.3712 | +0.011 | 6143* | PASS |
| **V2 guided r=12** | **5.201** | **7.436** | **5.032** | 4.575 | 0.3751 | +0.021 | 6688* | **PASS** |
| V2 guided r=16 | 5.217 | 7.411 | 4.982 | 4.597 | 0.3794 | +0.032 | 6788* | PASS |
| V3 bilateral | 5.250 | 7.599 | 5.178 | 4.610 | 0.3788 | −0.001 | 7948* | PASS |
| V3b WLS λ=1 | 5.265 | 7.625 | 5.191 | 4.628 | 0.3794 | −0.000 | 7107* | PASS |
| V3b WLS λ=24 | 5.242 | 7.579 | 5.164 | 4.607 | 0.3770 | −0.000 | 9148* | PASS |
| V4 conf-WLS λ=1 | 5.260 | 7.617 | 5.187 | 4.624 | 0.3778 | −0.008 | 9003* | PASS |
| V5 semantic-WLS λ=24 | 5.242 | 7.579 | 5.164 | 4.607 | 0.3770 | −0.000 | 9220* | PASS |
| V7 sem+conf WLS | 5.260 | 7.617 | 5.187 | 4.624 | 0.3778 | −0.008 | 8004* | PASS |
| V6 TTA median (8 tiles) | 4.857 | 7.290 | 4.711 | 4.193 | 0.3755 | **−0.296** | 13869 | FAIL |
| V8 full + TTA (8 tiles) | 4.850 | 7.278 | 4.705 | 4.187 | 0.3730 | **−0.303** | 20977 | FAIL |
| V9 full+planar+TTA (6 tiles) | 5.026 | 7.406 | 4.930 | 4.182 | 0.3809 | **−0.382** | 21480 | FAIL |

\* per-tile runtimes include the conservative spike-removal stage
(~5 s/tile of the ~6-8 s total on CPU 1024²); the pure guided filter is
O(N) ≈ 0.4 s/tile. On GPU the whole stage is milliseconds-scale.

Reading of the table (all deltas vs the same-tile V0):

* **Guided filter (r=12) is the selected method**: MAE −1.3 %, RMSE
  −2.5 %, building MAE −3.2 %, boundary MAE −1.2 %, gradient MAE −1.4 %,
  bias slightly *improved*, global mean shift +0.021 m (0.3 % of the mean
  AGL) — every acceptance rule passes, at negligible cost.
* Radius sweep matters more than method choice: r=4 → r=12 triples the
  RMSE gain (−0.06 → −0.19 m) before boundary metrics start to turn at
  r=16. The radius was selected **on validation** per protocol.
* **TTA median fusion gives the largest local gains** (MAE −0.41 m,
  building −0.49 m, boundary −0.44 m on its tile subset) **but fails the
  absolute-calibration guard**: median fusion systematically lowers the
  global mean by ~0.3 m and grows |bias| by ~0.3 m, at 4× inference cost.
  Honest verdict: FAIL as configured; a production TTA would need mean
  aggregation or `preserve_mean=True` plus a re-ablation.
* **Planar refinement** (V9 vs V8) improves boundary MAE further but
  degrades everything else and doubles the calibration shift — the
  predicted-mask components on this checkpoint mix structures. Stays
  experimental, off by default.
* **Semantic gating measured ≡ plain WLS on this data.** Diagnosis: the
  predicted head's probabilities are boundary-blind — mean Bhattacharyya
  similarity ≈ 0.999 **across GT class boundaries** (identical to
  within-class 0.999) even after retraining with w_sem=1.0; a 117k-param
  decoder trained for height regression does not produce pixel-sharp
  semantics at this budget. The mechanism is real (unit fixtures: 4.5×
  bleed reduction when probabilities are contrastive), but **no
  improvement is claimed** because none was measured. Enabling it
  requires a properly trained semantic predictor; the boundary-BC
  diagnostic above is the pre-flight check to run first.
* The model's own bias (−3.1 m) dominates all post-processing deltas —
  refinement fixes local geometry, never calibration error; that is
  CalibrationNet's job (and the reason the calibration guard exists).

## 7. Status ledger

**Implemented + validated on real data:** guided filter (selected),
median, joint bilateral, WLS (RGB-gated), confidence-weighted WLS,
conservative spike removal, boundary/gradient/seam/discontinuity metrics,
acceptance gate, ablation runner + diagnostics + array dumps, API/CLI
integration with backward-compatible default.

**Implemented, experimental (not enabled by default):** predicted-semantic
gating (mechanism proven on fixtures, no measured gain with the current
head), TTA ensemble (largest local gains, fails the calibration guard as
configured, 4× cost), robust planar refinement.

**Not implemented (deliberately, per protocol):** learned residual
refinement network (only after the ablation demonstrates remaining
systematic errors — the current ablation shows the classical ceiling is
reached at ~2.5 % RMSE improvement, so this decision stays open);
multi-scale inference fusion; shadow-based height constraints.

## 8. Failure modes and limitations

* Any RGB-guided method inherits **false image edges** (shadows, road
  markings, roof texture) as fake height structure; guided/WLS weight RGB
  evidence multiplicatively, so strongly textured flat ground is smoothed
  less. The grad/boundary metrics in the acceptance gate are the guard.
* TTA median fusion **shifts the global height scale** (~−0.3 m measured)
  — robust locally, biased globally. Use mean aggregation or
  `preserve_mean=True` if TTA is adopted, and re-run the acceptance gate.
* Semantic gating is only as good as the predicted head; a boundary-blind
  head silently degrades the variant to plain WLS (logged in the report
  notes; check the boundary-BC ratio before relying on it).
* Planar fitting can flatten genuinely sloped roofs when the residual
  gates are too loose; it is rejected wholesale on multi-facet structures
  only if the inlier gates fire — treat as research code.
* The spike stage's isolation window is a 7×7 default; structures smaller
  than ~3 px across that are *not* RGB-visible may be removed
  conservatively (this is the documented conservative direction).
* Absolute-height error (bias) is untouched by design — post-processing
  must never hide CalibrationNet's calibration error.

## 9. Computational cost (CPU, 1024×1024 tile, measured)

| stage | per tile |
|---|---|
| forward pass (CalibrationNet) | ~0.9 s |
| confidence map | ~1.5 s |
| spike removal (median/MAD stacks) | ~5 s |
| guided filter (any radius, O(N)) | ~0.4 s |
| bilateral (r=5) | ~1.5 s |
| WLS (CG ≤ 120 iters) | ~2-4 s |
| TTA (4 full passes) | ~13 s |
| whole `guided` preset (spikes+filter) | ~6.5 s |

All implementations are vectorized NumPy/integral-image/CG — no Python
pixel loops; large rasters go through the existing tiled inference
(per-1024-tile Dn contract) so memory stays O(tile); geospatial metadata
(CRS/transform/resolution/nodata/dtype/bounds) is copied verbatim from
the source raster to every written product.

## 10. Reproduction

```bash
# 1-4) data + checkpoint (network + CPU, ~35 min) — see
#       configs/gamus_postproc.yaml runbook header
python model.py splits  --dataset gamus --gamus-source hf --limit 24 \
    --out outputs/splits/gamus_splits.json
python model.py depth   --dataset gamus \
    --gamus-manifest outputs/splits/gamus_splits.json \
    --out-dir outputs/depth_cache --device cpu
python model.py fit-baseline --config configs/gamus_postproc.yaml --dataset gamus
python model.py train    --config configs/gamus_postproc.yaml --use-rgb \
    --sem-aux-head --w-sem 1.0 --out-tag postproc_flagship_v2 \
    --epochs 27 --batch-size 2
# 5) ablation (validation ONLY)
python model.py eval-postprocess --config configs/gamus_postproc.yaml \
    --checkpoint outputs/calib_net/postproc_flagship_v2/best.pt \
    --split val --out-tag ablation_r2
# tests
python -m pytest model_tests/ -q
```
