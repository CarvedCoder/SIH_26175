# DepthWizard Post-Processing — Final Implementation Report

**Deliverable bundle:** `depthwizard_postprocessing.zip` (full modified repository,
incl. all ablation outputs, checkpoints, docs) + this results folder.
**Working copy:** `/home/z/my-project/work/sih_repo`

---

## 1. Files changed (all inside the repository)

| file | change |
|---|---|
| `depthwizard/postprocess/refinement.py` | **bug fixes**: TTA 'mean' aggregation was silently ignored; `preserve_mean` added the shift instead of subtracting it (doubling it); `method="full"` was advertised but unimplemented (runtime ValueError) — now the composed semantic+confidence+WLS core + optional planar stage; TTA stack now feeds confidence estimation |
| `depthwizard/postprocess/spike_removal.py` | **bug fix**: isolation test was unsigned (its own docstring says same-sign) and had no contiguity check — roof *corners* were being clipped. Now sign-aware support + 4-neighbour same-sign contiguity protection |
| `depthwizard/postprocess/tta.py` | returns the orientation-aligned per-variant `stack` (feeds TTA-disagreement confidence) |
| `depthwizard/postprocess/config.py` | `full` preset now points at the implemented `full` method |
| `depthwizard/inference.py` | **bug fix**: `predict_with_semantics` returned a bare array for headless checkpoints → `run_inference` crashed with `IndexError` on any headless checkpoint + postprocessing. Now always returns `{"pred", "sem_probs"}` (legacy `predict()` unchanged) |
| `depthwizard/tifops.py` | reconstructs head-only checkpoints (`sem_input` flag, default preserves legacy bit-identically) |
| `depthwizard/calibration_net.py` | `sem_input` construction flag — enables the *documented but unwired* predicted-semantics deployment design (aux head WITHOUT privileged GT input channels). No existing checkpoint/weight behavior changes |
| `depthwizard/cli/train_calibration.py` | `--sem-aux-head` without `--use-sem` now trains the deployment-realistic variant; checkpoint records `sem_input` |
| `depthwizard/cli/eval_postprocess.py` | 10-panel diagnostics (added raw-depth + gradient-difference panels), **zoomed-crop figures** (roof / building-ground edge / vegetation boundary / road / terrain transition / worst outlier), per-tile array dumps (`raw_depth_dn`, `raw_agl`, `agl_gt`, `refined_agl_<variant>`, `gt_sem_onehot`), `--zoom-crop`/`--save-arrays-tiles` flags; **bug fixes**: latent wrong-relative-import (visualization had never successfully run), TTA activation keyed on variant *names* (missed `v8_full`), acceptance baselines now **tile-aligned** for TTA-budget variants |
| `model_tests/test_postprocess.py` | **NEW** — 77 tests (numerical/semantic/calibration/geospatial/regression + 6 end-to-end `run_inference` integration tests) |
| `configs/gamus_postproc.yaml` | **NEW** — reproducible real-data experiment config + runbook |
| `docs/postprocessing.md` | **NEW** — research documentation (algorithms, math, parameters, status ledger, failure modes, costs, ablation results, reproduction) |

Pre-existing infrastructure that was **audited, verified, and kept**
(`depthwizard/postprocess/` guided/bilateral/wls/semantic/confidence/
metrics/acceptance, `eval-postprocess` CLI, `infer --postprocess` API):
the package was already research-grade; this work fixed five real bugs in
it, tested it, and — for the first time — actually **ran** it on real data.

## 2. Commands executed (real-data pipeline)

```bash
python model.py splits  --dataset gamus --gamus-source hf --limit 24 --out outputs/splits/gamus_splits.json
python model.py depth   --dataset gamus --gamus-manifest outputs/splits/gamus_splits.json --out-dir outputs/depth_cache --device cpu
python model.py fit-baseline --config configs/gamus_postproc.yaml --dataset gamus
python model.py train    --config configs/gamus_postproc.yaml --use-rgb --sem-aux-head --w-sem 1.0 \
                         --out-tag postproc_flagship_v2 --epochs 27 --batch-size 2        # → val MAE 4.88 m
python model.py eval-postprocess --config configs/gamus_postproc.yaml \
                         --checkpoint outputs/calib_net/postproc_flagship_v2/best.pt --split val ...   # 10 chunks + λ/radius sweeps
python model.py infer   --input demo_DC_02_26.png --dn demo_DC_02_26_dn.npy \
                         --postprocess guided --pp-param guided_radius=12 ...              # example command
python -m pytest model_tests/ tests/ -q                                                   # 320 passed
```

Data: **real GAMUS** (earthflow/GAMUS via HuggingFace, official splits, 24
tiles/split), **real Depth-Anything-V2-Base** Dn cache (72 tiles), real GT
AGL + semantics. Checkpoint: 117k-param CalibrationNet, 27 epochs on CPU —
a budget model, identical for every ablation row (relative comparisons
valid; absolute quality not claimed).

## 3. Benchmark (frozen GAMUS validation split, 24 real tiles)

See `ablation.csv` / `report.md` for the full table with bias, per-class
strata, jump-ratio, runtimes, verdicts and failed-rule evidence.

| variant | MAE | RMSE | bldg MAE | boundary MAE | grad MAE | calib shift | verdict |
|---|---|---|---|---|---|---|---|
| V0 raw | 5.268 | 7.631 | 5.195 | 4.632 | 0.3805 | — | BASELINE |
| V1 median | 5.267 | 7.629 | 5.195 | 4.631 | 0.3786 | −0.003 | PASS |
| V2 guided r=4 | 5.236 | 7.567 | 5.150 | 4.597 | 0.3710 | +0.003 | PASS |
| **V2 guided r=12** | **5.201** | **7.436** | **5.032** | 4.575 | 0.3751 | +0.021 | **PASS — selected** |
| V3 bilateral | 5.250 | 7.599 | 5.178 | 4.610 | 0.3788 | −0.001 | PASS |
| WLS λ=24 | 5.242 | 7.579 | 5.164 | 4.607 | 0.3770 | −0.000 | PASS |
| conf-WLS λ=1 | 5.260 | 7.617 | 5.187 | 4.624 | 0.3778 | −0.008 | PASS |
| semantic-WLS λ=24 | 5.242 | 7.579 | 5.164 | 4.607 | 0.3770 | −0.000 | PASS (≡ WLS) |
| V6 TTA (8 tiles) | 4.857 | 7.290 | 4.711 | 4.193 | 0.3755 | **−0.296** | FAIL |
| V8 full+TTA (8 tiles) | 4.850 | 7.278 | 4.705 | 4.187 | 0.3730 | **−0.303** | FAIL |
| V9 +planar (6 tiles) | 5.026 | 7.406 | 4.930 | 4.182 | 0.3809 | **−0.382** | FAIL |

## 4. The task's mandatory questions

**Which post-processing method performs best?**
RGB-guided filter (He et al. 2010) at radius 12 + conservative spike removal
(`--postprocess guided --pp-param guided_radius=12`). Every metric class
improves simultaneously and it is the cheapest edge-aware option.

**How much does it improve MAE/RMSE?**
MAE 5.268→5.201 m (**−0.067 m, −1.3 %**); RMSE 7.631→7.436 m
(**−0.194 m, −2.5 %**); Pearson r and bias unchanged-or-better.
(These are honest, measured magnitudes — no visual-quality claims.)

**Does building accuracy improve?**
Yes: building MAE 5.195→5.032 m (**−0.164 m, −3.2 %**), the largest
relative gain of any region — consistent with denoising roofs while the
RGB guidance keeps outlines sharp.

**Does boundary accuracy improve?**
Yes: boundary-band MAE 4.632→4.575 m (**−1.2 %**) and gradient MAE
0.3805→0.3751 (−1.4 %); discontinuity jump-ratio not degraded. TTA improves
boundaries much more (−0.44 m) but fails the calibration guard.

**Does it preserve absolute height?**
Yes — measured, not assumed: global mean shift **+0.021 m** (0.3 % of mean
AGL), median/std reported in every run's `postprocess_meta.json`; the
acceptance gate rejects any variant whose mean shift exceeds tolerance
(TTA's −0.30 m was rejected exactly this way). No normalization exists
anywhere in the package.

**What is the inference-time cost?**
~6.5 s per 1024² tile on CPU (≈5 s of it is the median/MAD spike stage;
the guided filter itself is O(N) ≈ 0.4 s) vs ~0.9 s for the model forward
— milliseconds-scale on GPU, memory O(tile), fully vectorized, geospatial
metadata copied verbatim. TTA costs 4× inference and was measured at
13.9 s/tile.

**What failure cases remain?**
(i) RGB-guided methods inherit fake image edges (shadows, road markings,
roof texture) as fake height structure — guarded by the gradient/boundary
acceptance rules; (ii) the predicted semantic head is boundary-blind at
this training budget (Bhattacharyya 0.999 across GT boundaries), so
semantic gating currently adds nothing on real data — mechanism proven on
fixtures (4.5× bleed reduction), realization needs a stronger semantic
predictator; (iii) TTA median fusion systematically lowers the global
height scale; (iv) planar refinement degrades overall accuracy; (v) the
model's −3.1 m bias is untouched — by design, post-processing must never
hide CalibrationNet error.

**Should this be enabled by default?**
`guided r=12` + spike removal: **recommended default** on validation
evidence (all-metric PASS, negligible cost, calibration-safe) — enable via
`postprocess=guided, guided_radius=12`. The API default remains
`postprocess=none` (bit-identical legacy) until re-validated on the full
GAMUS split with a flagship checkpoint, per the no-cherry-picking rule.
TTA/planar/semantic: **not** by default.

## 5. Notable engineering findings (bonus)

1. The Exp-4 GT-semantics-**input** design collapses at deployment
   (zero-filled channels): Pearson −0.28 vs +0.65 privileged. The
   predicted-head design implemented here fixes it (Pearson +0.55
   deployed).
2. Five real bugs found by inspection/tests in the previously untested
   package (TTA mean-aggregation, preserve_mean sign, roof-corner
   clipping, unimplemented `full` method, headless-checkpoint crash) —
   all fixed and pinned by unit tests.
3. `--limit` is ignored when a GAMUS manifest is present (manifest wins)
   — documented behavior, worked around by freezing the manifest with the
   desired limit.
