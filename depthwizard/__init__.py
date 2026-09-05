"""DepthWizard — single-view RGB -> DSM estimation (SIH26175).

Package layout (v2, modular — everything fires from the repo-root ``model.py``):

    depthwizard/                 CORE LIBRARY (frozen contracts — import, never rewrite)
      normalize.py                 Dn min-max + AGL cleaning  [SINGLE SOURCE OF TRUTH]
      metrics.py                   masked MAE/RMSE/MedAE/bias/r, pooled + per-tile
      splits.py                    scene-level tile/block splits + overlap guards
      geo.py                       rasterio I/O, tile pairing, quicklooks, cache resolve
      dataset.py                   DFC2019Dataset (joint crop/aug, ImageNet RGB)
      calibration_net.py           CalibrationNet H = clamp(a(x,y)*Dn + b(x,y), 0)
      tifops.py                    checkpoint loading + predict-fn factory
      backbone.py                  live Depth Anything V2 (cache-miss fallback)
      inference.py                 image -> AGL/DSM pipeline (CLI + service share it)
      anchoring.py                 AGL + ground/DTM -> absolute DSM (Track 2)
      streaming.py                 memory-light pooled metric accumulators
      cli/                        COMMAND MODULES (one per pipeline stage)
      service api                  see /service (FastAPI bridge for the webapp)

Design contract (kept stable across all phases, per blueprint Sec. 5/18):
    a "sample" is always  {rgb: [3,H,W], agl: [1,H,W], cls: [1,H,W],
                           dn: [1,H,W] (optional), meta: {...}}
    everything is pixel-aligned on the source raster grid.

Governance (frozen): FINAL / citable numbers come ONLY from the
``evaluate`` command (mirrors the certified 09 eval logic). Inference
outputs and the webapp are demonstrations, never citations.
"""

__version__ = "2.0.0"
