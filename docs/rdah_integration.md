# DepthWizard × RDAH-Net Integration — Final Report

**Task**: replace the CalibrationNet height-calibration backend with a pretrained
RDAH-Net checkpoint as an *architecture/backend switch* (CalibrationNet fully
preserved), with `rdah` becoming the default backend after integration.

**Status**: ✅ COMPLETE — code integrated, 376/376 tests pass, real GAMUS
inference verified end-to-end (nDSM → DEM anchoring → absolute DSM).
Remaining items are *data-scale runs* (real GAMUS A/B numbers, GPU VRAM
benchmark) that require the local RTX 4050 — exact commands in §9/§10.

---

## 1. Compatibility report (pre-implementation analysis — condensed)

| # | Item | Finding |
|---|------|---------|
| 1 | CalibrationNet **input** contract | `net(dn, rgb, dem, sem, stats)` → dict; `dn` = min-max normalized [0,1] per tile; `rgb` = ImageNet-normalized; optional DEM/sem channels; `stats` = per-tile log-stats 4-vector |
| 2 | CalibrationNet **output** contract | `out["pred"]` = [B,1,H,W] clamped ≥ 0, affine `a(x,y)·Dn + b(x,y)` semantics |
| 3 | Depth Anything V2 output representation | RAW relative depth (disparity-like, unbounded scale); DepthWizard's frozen contracts store min-max normalized Dn **plus** the log-stats 4-vector, from which RAW is *exactly* reconstructible |
| 4 | GAMUS target representation | nDSM/AGL in metres (units undocumented in GAMUS itself — flagged, not fabricated) |
| 5 | Official RDAH **input** contract | `model(depth, img)`; `depth` [B,1,H,W] = RAW DAv2 × ~40 (0–255 range); `img` [B,3,H,W] ImageNet-normalized; H,W divisible by 128, ≤ 1024 |
| 6 | Official RDAH **output** contract | [B,1,H,W] height in **metres**, unclamped (nDSM; val SmoothL1 1.205 is metre-consistent) |
| 7 | Official checkpoint format | `{epoch, model_state_dict, optimizer_state_dict, loss}`, no `module.` prefix, 506 tensors, `weights_only=True`-safe; Track1: epoch 48, val loss 1.2051 |
| 8 | Preprocessing differences | **RGB identical** (uint8/255 → ImageNet). **Depth differs**: RDAH wants RAW×40, DepthWizard contracts carry min-max [0,1] + log-stats → RAW reconstructed exactly (log/exp roundtrip ~1e-6 relative), no frozen contract touched |
| 9 | Files needing modification | 10 modified / 6 added — exact list in §2/§3 |
| 10 | Blocking incompatibilities | **None.** The x40 constant is not explicitly documented in the official repo — derived via three independent lines of evidence (BN-stat forensics: depth-encoder stem input mean≈116/std≈24 ratio 0.21 = raw×40–46; empirical sweep on 7 real GAMUS val tiles: raw×{35–45} avg MAE 5.46–5.56 m vs min-max×255 6.41 m; the official `nyu_transform.ToTensor` never divides by 255). 40.0 sits centre-band and is exposed as `model.depth_scale` for sweeps |

---

## 2. Files ADDED (6)

| File | Purpose |
|------|---------|
| `depthwizard/rdah_net.py` | **Verbatim port** of the official `HeightPredTransformer` + 7 support classes from the official `test.py` (single commit 373bca2). Zero architectural edits — strict-loads the released checkpoint (0 missing / 0 unexpected keys) |
| `depthwizard/rdah.py` | Backend adapter: official preprocessing (RGB ImageNet once; depth RAW×40), CalibrationNet-compatible forward contract, checkpoint loading (MD5-verified auto-download from figshare, prefix stripping, loud mismatch failures), predict-fn factories |
| `model_tests/test_rdah.py` | 23 tests: loading/failure modes, shapes, preprocessing correctness (no double normalization), backend selection, geospatial honesty, anchoring, DSM writer |
| `configs/rdah_gamus.yaml` | RDAH fine-tuning config on GAMUS (paper recipe: Adam 1e-5, L1; RTX 4050 adaptation: 512 crops, batch 2, AMP) |
| `configs/rdah_smoke.yaml` | 1-epoch smoke config over `fake_gamus` (CI-safe) |
| `checkpoints/rdah/rdah_track1_best_model.pth` | Downloaded release checkpoint (65,516,400 bytes, MD5-verified) |

## 3. Files CHANGED (10)

| File | Change |
|------|--------|
| `depthwizard/tifops.py` | **Architecture registry**: `load_height_model(ckpt, architecture=...)` dispatches `calibration_net` \| `rdah`; `detect_architecture()` auto-detects from checkpoint payload; RDAH predict-fn factories hooked in |
| `depthwizard/config.py` | `model.architecture` field (default **`rdah`**), `model.checkpoint`, `model.pretrained`, `model.freeze_depth_anything`, `model.depth_scale`; `rdah:` section (`input_size`, `use_amp`). Legacy configs without `model.architecture` keep frozen Phase-2 behaviour |
| `depthwizard/inference.py` | `DepthWizardPredictor` made backend-agnostic; RDAH unclamped-nDSM semantics documented + reported; payload metadata gains `model_architecture` / `height_type` / `height_semantics` |
| `depthwizard/cli/train_calibration.py` | `train` command builds either architecture from config (fine-tuning support; CUDA autocast for RDAH) |
| `depthwizard/cli/eval_calibration.py` | `evaluate --model {calibration_net,rdah,auto}`; report adds parameter count, mean inference time per tile, peak VRAM (`torch.cuda.max_memory_allocated`) |
| `depthwizard/cli/bench_model.py` | New `bench` command: resolution sweep (default 256/512/1024), warm-up + repeats, peak VRAM, optional AMP — identical protocol for both backends |
| `depthwizard/cli/infer.py` | `infer --architecture {auto,calibration_net,rdah}` + `--depth-scale` overrides |
| `depthwizard/cli/registry.py` | Registry entry for `bench` |
| `configs/infer.yaml` | Default backend switched to `rdah` + pretrained Track1 checkpoint |
| `model.py` | Docstring runbook updated (train/evaluate/infer/bench for both backends) |

**Untouched (per spec)**: frontend, backend/app API routes, database, MinIO,
post-processing internals, experiments, GAMUS dataset/target generation,
anchoring semantics, tiling/mosaic, splits, metrics.

---

## 4. Exact checkpoint used

- **File**: `checkpoints/rdah/rdah_track1_best_model.pth`
- **Source**: figshare article **31986864** ("RDAH-Net checkpoints and Datasets", MIT license), file `63637257` (checkpoints-track1)
- **MD5**: `4fdd8769d2a05aee0ed40234aeceee09` — verified at download; auto-download re-verifies and deletes on mismatch
- **Size**: 65,516,400 bytes
- **Content**: `{epoch: 48, model_state_dict: 506 tensors, optimizer_state_dict, loss: 1.2051}` — official release format, no `module.` prefix, `weights_only=True`-safe
- **Provenance**: DFC2019-Track1 (1024×1024 US urban aerial) — the same imagery family as DepthWizard's GAMUS/DFC pipeline. Swiss/HK (512×512) variants are also wired in `RDAH_CHECKPOINTS` (variant `"swiss"` / `"hk"`).
- **Load verification**: strict state-dict load → **0 missing / 0 unexpected keys**; **5,371,663 parameters** (paper Table 4 "5.37 M" — exact match). Checkpoint name + parameter count printed at every load.

## 5. Exact preprocessing used (official recipe, reproduced exactly once)

**RGB** (identical to CalibrationNet recipe — applied ONCE, in `make_rdah_predict_fn` / the dataset transforms, never inside the model):
```
uint8 [H,W,3] → /255 → (x − ImageNet_mean) / ImageNet_std → [B,3,H,W]
```

**Depth** (DIFFERENT from CalibrationNet — never min-max, never /255, never ImageNet):
```
raw DAv2 relative depth × 40.0  →  [B,1,H,W] float32 (~0–255 range)
```
Recovered from the frozen contracts via exact log-stats inversion:
`lo = exp(stats[0]) − 1e-3; hi = exp(stats[1]) − 1e-3; raw = lo + dn·(hi − lo)`,
then `× depth_scale` once. Flat-tile guard mirrors `minmax_normalize`'s 0.5-everywhere behaviour so train/infer stay consistent.

**Output**: [B,1,H,W] nDSM **metres**, unclamped (ground may dip slightly < 0 — reported, never silently clamped). Labelled `height_type: "nDSM"` — never mislabelled as absolute terrain elevation.

**Resolution constraints**: H,W divisible by 128, ≤ 1024 (BlockAttention windowing + 64×64 PositionalEncoding buffer). Training crops 512; inference tiles handled by the existing tiling machinery.

---

## 6. Parameter counts (before → after)

| Backend | Parameters |
|---------|-----------|
| CalibrationNet flagship (in_ch=4, widths 16/32/64) | **117,138** |
| RDAH HeightPredTransformer (Track1) | **5,371,663** |

≈ 46× more parameters, still trivially 6-GB-VRAM-compatible (fp16 weights ≈ 10.7 MB). Depth Anything V2 ViT-B remains the frozen depth backbone (never trained — `freeze_depth_anything: true` is documented + cache-based).

## 7. One successful inference result (real GAMUS tile)

`DC_02_26` (1024×1024), live DAv2 ViT-B → RDAH Track1 → anchoring:

- **nDSM (metres)**: min −0.27 / median 0.32 / mean 1.72 / max 16.38; 1,165/1,048,576 pixels (0.11%) slightly negative — honest unclamped ground
- **Anchored absolute DSM** (constant datum 10 m): 9.73 – 26.38 m, labelled `ANCHORED (not learned)`
- **Payload metadata**: `model_architecture: "rdah"`, `height_type: "nDSM"`, `height_semantics: "nDSM + ground datum = absolute DSM (ANCHORED (not learned))"`, `model_tag: "rdah_rdah_track1_best_model_ep48"`
- **Georeference honesty**: source PNG has `crs=None` → payload reports `crs: "UNKNOWN"` / `transform: "UNKNOWN"`, no GeoTIFF emitted, **no fake georeference invented**; georeferenced GeoTIFF inputs keep CRS/transform propagated
- Wall-clock: 14.1 s on CPU (1024², batch 1) — GPU timings from `bench`
- Artifacts: `infer_out/rdah_gamus/` (dsm.npy + preview) and `infer_out/rdah_anchored/` (dsm.npy, dsm_anchored.npy, preview)

## 8. Test results

```
376 passed, 0 failed   (model_tests/, 43 s, CPU)
```
- **23 new RDAH tests** (`model_tests/test_rdah.py`): checkpoint exists/loads/incompatible-fails-loudly; shape contract [B,3,H,W]+[B,1,H,W]→[B,1,H,W]; preprocessing correctness (ImageNet exactly once, no double normalization; RAW×40 from log-stats roundtrip); backend selection through the registry (both architectures); geospatial path (CRS/transform propagation, DEM anchoring, DSM writer, no-fake-CRS behaviour)
- **353 regression tests unchanged and passing** — CalibrationNet path, anchoring, tiling, mosaic, metrics, losses, CLI, splits, datasets, post-processing all intact
- The 65 MB released checkpoint is used only by opt-in tests (auto-skip when absent); CI runs on tiny synthetic checkpoints in the official format

## 9. Unresolved items / honest caveats

1. **Real GAMUS A/B numbers pending** — the evaluate path is verified end-to-end (4-tile val run: MAE 4.91 m on a 1-epoch smoke model over synthetic `fake_gamus`), but citable numbers require the real GAMUS fine-tune on the local RTX 4050 (commands in §10). No gates: the frozen reference card is DFC-only.
2. **Depth-scale constant 40.0 is derived, not documented** — BN forensics + 7-tile empirical sweep (raw×{35–45} flat within 0.1 m MAE; min-max×255 clearly worse). Exposed as `model.depth_scale` / `--depth-scale` for A/B sweeps.
3. **GAMUS height units assumed metres** (undocumented upstream) — flagged in every eval report, never fabricated.
4. **RDAH has no semantic head** — `sem_probs: None` in the post-processing contract (degrades honestly, same as headless CalibNet checkpoints); supplying DEM/sem channels to the RDAH forward raises a loud `ValueError` (never silently dropped).
5. **Peak VRAM numbers need the local GPU** — `bench` reports `torch.cuda.max_memory_allocated` per resolution; on CPU it prints `n/a`. Run §10 command 4 on the RTX 4050.
6. **Anchoring semantics unchanged** (verified): RDAH nDSM + DEM = absolute DSM, arithmetic only.

## 10. Runbook — exact commands

```bash
cd depthwizard_repo

# ── RDAH inference (now the DEFAULT backend) ─────────────────────────
python model.py infer --input scene.tif                      # auto: rdah + Track1 ckpt
python model.py infer --input scene.tif --anchor-dem dem.tif   # absolute DSM via DEM
python model.py infer --input photo.png --json-out payload.json

# ── Switch BACK to CalibrationNet (one flag) ──────────────────────────
python model.py infer --input scene.tif \
    --architecture calibration_net \
    --checkpoint outputs/calib_net/rgb/best.pt

# ── A/B evaluation (identical split / target / masks / metrics) ───────
python model.py evaluate --model rdah             --dataset gamus \
    --config configs/rdah_gamus.yaml \
    --checkpoint checkpoints/rdah/rdah_track1_best_model.pth --splits val
python model.py evaluate --model calibration_net  --dataset gamus \
    --config configs/rdah_gamus.yaml \
    --checkpoint outputs/calib_net/rgb/best.pt --splits val

# ── Fine-tune pretrained RDAH on GAMUS (needs the DAv2 cache first) ───
python model.py depth --config configs/gamus.yaml --dataset gamus \
    --out-dir outputs/depth_cache --limit 512 --device auto
python model.py train --config configs/rdah_gamus.yaml --out-tag rdah_gamus

# ── Resolution / VRAM benchmark on the RTX 4050 ───────────────────────
python model.py bench --architecture rdah --device cuda \
    --resolutions 256 512 1024 --repeats 10          # add --amp for fp16
```

Config-level switching (YAML): `model.architecture: rdah` ↔ `calibration_net`
(+ `model.checkpoint`), in `configs/infer.yaml`, `configs/rdah_gamus.yaml`, etc.

---

## Acceptance criteria checklist

- [x] Official RDAH architecture integrated (verbatim port, zero edits)
- [x] Correct pretrained checkpoint loaded (Track1, MD5-verified, strict, epoch 48)
- [x] RGB preprocessing matches official (ImageNet, exactly once)
- [x] Depth preprocessing matches official (RAW×40 — BN-forensics + sweep verified)
- [x] Depth Anything V2 remains the frozen depth backbone
- [x] CalibrationNet still works as the alternative backend (registry parity verified)
- [x] RDAH selectable via config AND CLI (`--architecture` / `--model`)
- [x] RDAH output integrates with existing anchoring (nDSM + datum = DSM, verified)
- [x] Existing DSM writer works (npy + GeoTIFF path unchanged)
- [x] CRS/transform handling not broken (propagated; `UNKNOWN` reported honestly)
- [x] Tests pass (376/376, incl. 353 regression)
- [x] RDAH inference works on a real GAMUS sample (DC_02_26, §7)
- [x] Parameter count printed at every load (5,371,663)
- [~] Peak VRAM reported — wired into `bench`/`evaluate`; final numbers need the local GPU (`n/a` on CPU here)
- [x] No silent checkpoint mismatch (strict load + loud key-mismatch errors + payload validation)
- [x] No fake georeference generated (`crs: UNKNOWN` path verified)

---

## 11. Webapp backend switch (added after integration)

The frontend can switch the height-model backend per run — a segmented
"HEIGHT MODEL" control on the workspace Home panel (RDAH-Net default,
CalibrationNet one click away), persisted in `localStorage`
(`dw_model_backend`) and sent as `architecture` on **both** the process
and refine requests so a scene's refined tile stays consistent with its
full-scene product.

API contract (`backend/app/schemas/processing.py`):

    POST /scenes/{id}/process   { "architecture": "rdah" | "calibration_net" | "auto", ... }
    POST /scenes/{id}/refine    { "architecture": ..., ... }

Semantics (honest, no silent swaps):

  * `rdah` / `calibration_net` — explicit selection; checkpoint resolved
    per backend (`checkpoints/rdah/…` auto-download+MD5 for RDAH;
    `DW_CKPT` → `outputs/calib_net/postproc_flagship/best.pt` → tracked
    `best.pt` for CalibrationNet). A `DW_CKPT` that does not match the
    requested architecture fails the job with `INVALID_INPUT`.
  * `auto` (default) — legacy-client behaviour: follow the configured
    checkpoint (`DW_CKPT`'s detected architecture), else RDAH.
  * `GET /health` `model_loaded` is a cheap filesystem check — it never
    triggers the RDAH auto-download; the download happens inside the
    processing job that needs it.
  * The chosen architecture is persisted on the durable job record, so
    external workers execute it statelessly.

Verified end-to-end from the browser: CalibrationNet run →
`model_architecture: "calibration_net"`; RDAH run →
`model_architecture: "rdah"` (`model_tag: rdah_rdah_track1_best_model_ep48`).
