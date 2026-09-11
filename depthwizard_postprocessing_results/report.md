# Post-processing ablation — merged benchmark

Checkpoint: `postproc_flagship_v2` (Dn+RGB + PREDICTED semantic aux head, 27 epochs on 24 real GAMUS train tiles, CPU).
Split: frozen GAMUS **val** (24 tiles, 1024x1024). TTA rows cover the first 8 (6 for v9) tiles — acceptance verdicts are tile-aligned.

| variant | config | MAE | dMAE | RMSE | dRMSE | bldg MAE | dbldg | boundary | dbnd | grad | dgrad | bias | calib shift | ms/tile | n | verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| v0_raw | baseline (24 tiles) | 5.268 | +0.000 | 7.631 | +0.000 | 5.195 | +0.000 | 4.632 | +0.000 | 0.3805 | +0.0000 | -3.128 | — | 1 | 24 | BASELINE |
| v1_median | 3x3 median + spike removal | 5.267 | -0.001 | 7.629 | -0.001 | 5.195 | -0.000 | 4.631 | -0.001 | 0.3786 | -0.0019 | -3.131 | -0.0028 | 6975 | 24 | PASS |
| v2_guided | guided r=4, eps=1e-3 + spike removal | 5.236 | -0.033 | 7.567 | -0.064 | 5.150 | -0.045 | 4.597 | -0.035 | 0.3710 | -0.0095 | -3.125 | +0.0031 | 6364 | 24 | PASS |
| v2_guided | guided r=8 (radius sweep) | 5.207 | -0.061 | 7.489 | -0.141 | 5.089 | -0.106 | 4.574 | -0.058 | 0.3712 | -0.0093 | -3.117 | +0.0108 | 6143 | 24 | PASS |
| v2_guided | guided r=12 (radius sweep) | 5.201 | -0.067 | 7.436 | -0.194 | 5.032 | -0.164 | 4.575 | -0.057 | 0.3751 | -0.0054 | -3.106 | +0.0206 | 6688 | 24 | PASS |
| v2_guided | guided r=16 (radius sweep) | 5.217 | -0.051 | 7.411 | -0.220 | 4.982 | -0.213 | 4.597 | -0.035 | 0.3794 | -0.0011 | -3.095 | +0.0317 | 6788 | 24 | PASS |
| v3_bilateral | joint bilateral r=5 + spike removal | 5.250 | -0.018 | 7.599 | -0.031 | 5.178 | -0.017 | 4.610 | -0.022 | 0.3788 | -0.0017 | -3.129 | -0.0005 | 7948 | 24 | PASS |
| v3b_wls | WLS lambda=1 + spike removal | 5.265 | -0.003 | 7.625 | -0.006 | 5.191 | -0.005 | 4.628 | -0.004 | 0.3794 | -0.0011 | -3.128 | -0.0001 | 7107 | 24 | PASS |
| v3b_wls | WLS lambda=8 (lambda sweep) | 5.254 | -0.014 | 7.604 | -0.026 | 5.178 | -0.018 | 4.618 | -0.014 | 0.3782 | -0.0023 | -3.128 | -0.0001 | 7871 | 24 | PASS |
| v3b_wls | WLS lambda=24 (lambda sweep) | 5.242 | -0.026 | 7.579 | -0.052 | 5.164 | -0.032 | 4.607 | -0.025 | 0.3770 | -0.0035 | -3.128 | -0.0001 | 9148 | 24 | PASS |
| v5_semantic_wls | semantic WLS lambda=8 | 5.254 | -0.014 | 7.604 | -0.026 | 5.178 | -0.018 | 4.618 | -0.014 | 0.3782 | -0.0022 | -3.128 | -0.0001 | 7894 | 24 | PASS |
| v5_semantic_wls | semantic WLS lambda=24 | 5.242 | -0.026 | 7.579 | -0.052 | 5.164 | -0.032 | 4.607 | -0.025 | 0.3770 | -0.0035 | -3.128 | -0.0001 | 9220 | 24 | PASS |
| v4_conf_wls | confidence WLS lambda=1 | 5.260 | -0.008 | 7.617 | -0.013 | 5.187 | -0.008 | 4.624 | -0.008 | 0.3778 | -0.0027 | -3.137 | -0.0084 | 9003 | 24 | PASS |
| v5_semantic_wls | semantic WLS lambda=1 | 5.265 | -0.003 | 7.625 | -0.006 | 5.191 | -0.005 | 4.628 | -0.004 | 0.3794 | -0.0011 | -3.128 | -0.0001 | 7229 | 24 | PASS |
| v7_sem_conf_wls | semantic+conf WLS lambda=1 | 5.260 | -0.008 | 7.617 | -0.013 | 5.187 | -0.008 | 4.624 | -0.007 | 0.3778 | -0.0027 | -3.137 | -0.0084 | 8004 | 24 | PASS |
| v6_tta | TTA median fusion (8 tiles) | 4.857 | -0.411 | 7.290 | -0.341 | 4.711 | -0.485 | 4.193 | -0.439 | 0.3755 | -0.0050 | -2.444 | -0.2964 | 13869 | 8 | FAIL |
| v8_full | full pipeline + TTA (8 tiles) | 4.850 | -0.418 | 7.278 | -0.353 | 4.705 | -0.490 | 4.187 | -0.445 | 0.3730 | -0.0075 | -2.451 | -0.3028 | 20977 | 8 | FAIL |
| v9_full_planar | full + planar + TTA (6 tiles) | 5.026 | -0.242 | 7.406 | -0.225 | 4.930 | -0.265 | 4.182 | -0.450 | 0.3809 | +0.0004 | -2.192 | -0.3823 | 21480 | 6 | FAIL |