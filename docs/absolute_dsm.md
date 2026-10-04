# Absolute DSM: DEM acquisition, anchoring, provenance and evaluation

This document defines the vocabulary and the workflow for DepthWizard's
height products. The distinctions below are enforced in code and tests
(`model_tests/test_absolute_dsm.py`, `model_tests/test_dem_provider.py`,
`model_tests/test_eval_absolute.py`, `model_tests/test_output_types.py`).

## Vocabulary: model vs feature vs metric vs benchmark

| Term | Examples | Notes |
| --- | --- | --- |
| **Model** | TerraHeight-S, RDAH, Semantic-Aware CalibrationNet | Selectable height backends. Never call a model a "metric". |
| **Feature** | GeoTIFF support, GSD-aware inference, DEM anchoring, absolute DSM, 3D reconstruction, DINOv3 building detection, damage assessment, Route Assist | Capabilities of the product. |
| **Metric** | MAE, RMSE, bias, medAE, Pearson r, slope MAE, valid-pixel coverage | Numbers computed by evaluators. |
| **Benchmark** | GAMUS AGL benchmark; an organizer/reference absolute-DSM benchmark when one exists | A benchmark pairs a dataset/protocol with metrics. |

Height models available in the UI/API (`architecture`):

- `terraheight_s` — external pretrained GAMUS AGL model (RGB only, no
  depth cache; checkpoint never auto-downloaded).
- `rdah` — official RDAH-Net with the released Track-1 checkpoint
  (default; auto-downloaded and MD5-verified on first use).
- `calibration_net` — the Semantic-Aware CalibrationNet (legacy Phase-2
  net with the semantic auxiliary head).

**Where CalibrationNet actually sits:** code inspection (`depthwizard/
tifops.py`, `depthwizard/inference.py`, `depthwizard/terraheight.py`)
shows it is a standalone selectable backend, not a stage bolted after
TerraHeight-S or RDAH. RDAH explicitly rejects DEM/semantic inputs and
TerraHeight-S has no calibration stage. Its semantic-aware calibration
behavior is preserved unchanged. Every result records the executed model
(`meta.model_architecture`) plus `calibration_enabled` /
`calibration_model` in the provenance — `true` only when the
CalibrationNet backend itself produced the heights.

## AGL vs absolute DSM — the frontier

```
AGL MODEL EVALUATION
    pred height field  vs  truth *_AGL.tif
    commands: gt-check, eval-scene, evaluate (the citable one)
    protocol: the existing GAMUS AGL benchmark (UNCHANGED)

ABSOLUTE DSM EVALUATION
    pred absolute DSM  vs  reference absolute DSM
    command:   eval-absolute
    requires:  a genuine external elevation reference (a real DEM)
```

The ~1.3 m full-GAMUS validation MAE of the TerraHeight-S checkpoint and
the RDAH paper's Track-1 MAE are **AGL benchmark numbers**. They are not
absolute-DSM accuracy, and no output of this project may claim absolute
accuracy without `eval-absolute` evidence against a valid reference DSM.

Every scene payload records which side of the frontier it is on:

```json
"output_type": "relative_height" | "absolute_dsm" | "anchored_constant_dsm",
"absolute_reference_available": true | false
```

- `relative_height` — AGL only. No CRS invented for PNG/JPG inputs.
- `absolute_dsm` — `DSM_absolute = DEM_reference + AGL_prediction` with
  full DEM provenance. Only set when an external DEM was actually used.
- `anchored_constant_dsm` — AGL + a user-asserted constant datum
  (`--ground-elev`). Labelled `ANCHORED (not learned)`; the datum is
  user-asserted, so `absolute_reference_available` stays `false`.

## DEM acquisition (Part A)

Two ways to obtain a reference DEM:

1. **Manual (always available, no network):** `--anchor-dem <DEM.tif>`.
2. **Automatic:** `--dem-provider copernicus` (or `DW_DEM_PROVIDER`).
   Fetches Copernicus GLO-30 COGs from the AWS Open Data registry
   (no account needed), clips the needed windows, caches them under
   `outputs/dem_cache/copernicus30m/`, mosaics multi-tile footprints and
   returns full provenance (nominal 30 m resolution, EPSG:4326, vertical
   reference EGM2008 per the ESA GLO-30 product specification).

```bash
python model.py infer --input scene.tif --dem-provider copernicus
DW_DEM_PROVIDER=copernicus ./dw start     # backend-wide default
```

Rules (enforced by tests):

- Network/retrieval failures raise `DEM_UNAVAILABLE` **explicitly**, then
  the run degrades to the RELATIVE product with a loud notice. Synthetic
  or placeholder DEMs are never substituted; the test-only mock provider
  refuses to run in production paths.
- A non-georeferenced input with automatic DEM acquisition fails with
  `CRS_REQUIRED` — the footprint is unknown without a CRS.
- DEM tiles are cached; a repeat scene costs no network.

## DEM alignment (Part B)

`depthwizard.absolute_dsm.build_absolute_dsm` is the single alignment
implementation (shared by both inference orchestrators):

- input CRS, affine transform and dimensions are preserved exactly;
- the DEM is reprojected onto the image grid (bilinear);
- full-coverage is REQUIRED: partial overlap raises
  `DEM_COVERAGE_INSUFFICIENT` (never silent clipping);
- mismatched grids raise `GRID_MISMATCH` (arrays are never broadcast).

Provenance (Part C) is written to `absolute_dsm_provenance.json` and
mirrored into the payload (`meta.provenance`): height model + version,
calibration flag, height units/semantics, DEM source/identifier/CRS/
resolution/vertical reference/tiles/acquisition time, scene CRS/GSD/
bounds/shape, resampling method and the exact formula. Unknown fields are
`null` — never invented.

## Absolute DSM evaluation (Part H)

```bash
python model.py eval-absolute \
    --pred predicted_absolute_dsm.tif \
    --truth reference_absolute_dsm.tif \
    --json-out report.json
```

- Checks CRS on both rasters (`CRS_REQUIRED` otherwise); if grids differ,
  the prediction is reprojected onto the reference grid (the reference is
  the authority); zero overlap is `GRID_MISMATCH`.
- Nodata/NaN pixels are excluded; valid coverage % is reported.
- Metrics: MAE, RMSE, bias, medAE, Pearson r (+ slope MAE when the GSD
  is known; otherwise honestly `null`).
- The JSON report uses `kind: absolute_dsm_evaluation` and is kept
  strictly separate from the GAMUS AGL benchmark.

## Error codes (Part S)

`depthwizard.statuses` defines the machine-readable statuses:
`DEM_REQUIRED`, `DEM_UNAVAILABLE`, `DEM_COVERAGE_INSUFFICIENT`,
`CRS_REQUIRED`, `INVALID_GEOREFERENCE`, `VERTICAL_REFERENCE_UNKNOWN`,
`REFERENCE_DSM_REQUIRED_FOR_EVALUATION`, `GRID_MISMATCH`,
`INVALID_POLYGON`. Failures are explicit; the only automatic fallback is
absolute-DSM-unavailable → continue with the AGL/relative product.

## Known limitations

- Absolute DSM quality depends on the external DEM's own accuracy and
  vertical reference; Copernicus GLO-30 is 30 m nominal and EGM2008 —
  relative-datum mismatches with the scene's reference will surface as a
  constant bias, which `eval-absolute` reports as `bias`.
- No organizer reference absolute-DSM dataset ships with this repository;
  the evaluator is validated on controlled synthetic cases, and no
  organizer-level absolute accuracy is claimed.
- PNG/JPG inputs cannot reach absolute DSM at all (no georeferencing);
  they always produce relative height products.
