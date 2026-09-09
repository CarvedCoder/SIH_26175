"""DepthWizard inference service — the backend half of the webapp bridge.

Endpoints:
    GET  /health    -> liveness + model/device/config provenance
    POST /predict   -> multipart form:
                         image       (required) PNG/JPG/TIF
                         anchor_dem  (optional) DEM/DTM raster for Track-2
                         ground_elev (optional) constant datum, metres
                         mode        (optional) auto|crop|resize|tiles
                       -> scene payload JSON (see depthwizard.inference.

The service calls ``depthwizard.inference.run_inference`` — the SAME code
path as ``python model.py infer``. It never re-implements preprocessing, so
the webapp can never drift from the certified CLI forward pass.

Run (dev):   python model.py serve --port 8000
Run (prod):  uvicorn service.api:app --host 0.0.0.0 --port 8000

Environment overrides (all optional; configs/infer.yaml is the base):
    DW_ROOT        repo root (default: parent of this package's parent)
    DW_CKPT        checkpoint path
    DW_DEVICE      cuda | cpu | auto
    DW_CACHE_DIR   depth cache dir
    DW_BACKBONE    live DAv2 model id
    DW_NO_LIVE     "1" -> disable the live backbone
    DW_OUT_ROOT    where per-request outputs are written (default: <root>/outputs/service)
"""

from __future__ import annotations

import os
import shutil
import tempfile
import time
import uuid
from pathlib import Path
from typing import Optional

import yaml
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from depthwizard.inference import run_inference

# ---------------------------------------------------------------------------
# Configuration resolution: env > configs/infer.yaml > sane defaults
# ---------------------------------------------------------------------------

_ROOT = Path(os.environ.get("DW_ROOT", Path(__file__).resolve().parents[1]))
_CONFIG = _ROOT / "configs" / "infer.yaml"


def _cfg() -> dict:
    if _CONFIG.exists():
        with open(_CONFIG, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    return {}


def _paths() -> dict:
    return _cfg().get("paths", {})


def _infer_cfg() -> dict:
    return _cfg().get("infer", {})


def _ckpt_path() -> Path:
    env = os.environ.get("DW_CKPT")
    if env:
        return Path(env)
    p = _paths().get("outputs_dir", "outputs")
    tag = _infer_cfg().get("checkpoint", f"{p}/calib_net/rgb_cos/best.pt")
    return _ROOT / tag if not Path(tag).is_absolute() else Path(tag)


def _device() -> str:
    return os.environ.get("DW_DEVICE", _infer_cfg().get("device", "auto"))


def _cache_dir() -> Optional[Path]:
    env = os.environ.get("DW_CACHE_DIR")
    if env:
        return Path(env)
    c = _paths().get("depth_cache_dir")
    return (_ROOT / c) if c else None


def _backbone_id() -> str:
    # Phase 0.2: default aligned to ViT-B (matches the training cache; was
    # V2-Large-hf before — mismatched cached-vs-live Dn distributions, risk R9).
    return os.environ.get(
        "DW_BACKBONE",
        _infer_cfg().get("backbone", "depth-anything/Depth-Anything-V2-Base-hf"),
    )


def _live() -> bool:
    if os.environ.get("DW_NO_LIVE") == "1":
        return False
    return bool(_infer_cfg().get("live_backbone", True))


def _out_root() -> Path:
    env = os.environ.get("DW_OUT_ROOT")
    if env:
        return Path(env)
    return _ROOT / "outputs" / "service"


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="DepthWizard Inference Service",
    description="Backend bridge for the DepthWizard webapp. Same code path "
    "as `python model.py infer`.",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=os.environ.get("DW_CORS_ORIGINS", "*").split(","),
    allow_methods=["*"],
    allow_headers=["*"],
)

_predictor_state = {"last_used": None}  # diagnostics only


@app.get("/health")
def health() -> dict:
    ckpt = _ckpt_path()
    return {
        "status": "ok",
        "service": "depthwizard-inference",
        "root": str(_ROOT),
        "checkpoint": str(ckpt),
        "checkpoint_exists": ckpt.exists(),
        "device": _device(),
        "cache_dir": str(_cache_dir()) if _cache_dir() else None,
        "backbone": _backbone_id(),
        "live_backbone": _live(),
        "config": str(_CONFIG) if _CONFIG.exists() else "defaults",
        "last_predict": _predictor_state["last_used"],
    }


@app.post("/predict")
async def predict(
    image: UploadFile = File(...),
    anchor_dem: Optional[UploadFile] = File(None),
    ground_elev: Optional[float] = Form(None),
    mode: str = Form("auto"),
) -> JSONResponse:
    t0 = time.perf_counter()
    ckpt = _ckpt_path()
    if not ckpt.exists():
        raise HTTPException(
            status_code=503,
            detail=f"checkpoint missing: {ckpt} — train one or set DW_CKPT",
        )

    if mode not in ("auto", "crop", "resize", "tiles"):
        raise HTTPException(status_code=400, detail=f"bad mode '{mode}'")

    req_id = time.strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:6]
    work = tempfile.mkdtemp(prefix=f"dw-{req_id}-")
    try:
        # ---- persist uploads (rasterio needs real files) ----------------
        # The scratch filename PRESERVES the original stem (sanitized) so
        # the depth-cache lookup by stem still works for dataset tiles
        # (e.g. uploading JAX_004_006_RGB.tif hits JAX_004_006.npy).
        import re as _re

        orig_stem = Path(image.filename or "upload").stem or "upload"
        safe_stem = _re.sub(r"[^A-Za-z0-9_\-]", "_", orig_stem)[:80]
        suffix = Path(image.filename or "upload.png").suffix or ".png"
        img_path = Path(work) / f"{safe_stem}{suffix}"
        with open(img_path, "wb") as f:
            shutil.copyfileobj(image.file, f)

        dem_path = None
        if anchor_dem is not None and anchor_dem.filename:
            dem_suffix = Path(anchor_dem.filename).suffix or ".tif"
            dem_path = Path(work) / f"anchor_dem{dem_suffix}"
            with open(dem_path, "wb") as f:
                shutil.copyfileobj(anchor_dem.file, f)

        out_dir = _out_root() / req_id
        payload = run_inference(
            img_path,
            ckpt,
            out_dir=out_dir,
            device=_device(),
            mode=mode,
            dn_path=None,
            cache_dir=_cache_dir(),
            live_backbone=_live(),
            backbone_id=_backbone_id(),
            anchor_dem=dem_path,
            ground_elev=ground_elev,
            write_files=True,
        )
        payload["request_id"] = req_id
        payload["service_elapsed_sec"] = round(time.perf_counter() - t0, 2)
        _predictor_state["last_used"] = req_id
        return JSONResponse(payload)
    except HTTPException:
        raise
    except ValueError as e:  # honest user errors
        raise HTTPException(status_code=400, detail=str(e))
    except FileNotFoundError as e:
        raise HTTPException(status_code=400, detail=str(e))
    finally:
        # the outputs live under DW_OUT_ROOT; only scratch uploads die here
        shutil.rmtree(work, ignore_errors=True)
