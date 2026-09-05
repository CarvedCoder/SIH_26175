#!/usr/bin/env python3
"""End-to-end connectivity test: CLI infer + FastAPI service + payload contract.

Simulates the full webapp loop without a trained flagship:
  1. build a tiny untrained CalibrationNet checkpoint (affine init exact)
  2. write a synthetic RGB image (+ a georeferenced variant + a DEM)
  3. run `python model.py infer --json-out` (CLI transport)
  4. boot the FastAPI service, POST /predict (service transport)
  5. assert both payloads satisfy the frontend contract (lib/dw.ts)

Run from the repo root:  python tools/e2e_bridge_check.py
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

PASS, FAIL = "\033[32mPASS\033[0m", "\033[31mFAIL\033[0m"
failures = []


def check(name: str, cond: bool, extra: str = "") -> None:
    print(f"  [{PASS if cond else FAIL}] {name}{(' — ' + extra) if extra else ''}")
    if not cond:
        failures.append(name)


def make_checkpoint(path: Path) -> None:
    import torch
    from depthwizard.calibration_net import CalibrationNet

    net = CalibrationNet(in_ch=1, widths=(8, 16, 32), a0=2.0, b0=4.0)
    torch.save({
        "model_state": net.state_dict(),
        "use_rgb": False, "widths": [8, 16, 32], "clamp_min": 0.0,
        "affine_init": {"a": 2.0, "b": 4.0}, "loss": "l1", "epoch": 0,
        "val_subset_mae": None, "splits_json": "e2e",
    }, path)


def make_image(path: Path, h: int = 300, w: int = 300, georef: bool = False):
    import rasterio
    from rasterio.transform import from_origin

    rng = np.random.default_rng(7)
    rgb = rng.integers(0, 255, (h, w, 3), dtype=np.uint8)
    prof = dict(driver="GTiff", height=h, width=w, count=3, dtype="uint8")
    if georef:
        prof.update(crs="EPSG:32617", transform=from_origin(500000, 4000000, 1.0, 1.0))
    with rasterio.open(path, "w", **prof) as dst:
        dst.write(rgb.transpose(2, 0, 1))
    return rgb


def make_dem(path: Path) -> None:
    import rasterio
    from rasterio.transform import from_origin

    dem = np.full((64, 64), 42.0, dtype=np.float32)
    with rasterio.open(path, "w", driver="GTiff", height=64, width=64,
                       count=1, dtype="float32", crs="EPSG:32617",
                       transform=from_origin(500000, 4000000, 4.7, 4.7)) as dst:
        dst.write(dem, 1)


def payload_contract_ok(name: str, p: dict) -> None:
    check(f"{name}: ok flag", p.get("ok") is True)
    g = p.get("grid", {})
    check(f"{name}: grid shape/data agree",
          isinstance(g.get("height"), int) and isinstance(g.get("width"), int)
          and len(g.get("data", [])) == g.get("height", 0) * g.get("width", 0))
    check(f"{name}: rgb_png data URL",
          str(p.get("rgb_png", "")).startswith("data:image/png;base64,"))
    s = p.get("stats", {})
    check(f"{name}: stats keys", all(k in s for k in ("min", "mean", "median", "max", "neg")))
    check(f"{name}: georef honesty",
          p.get("georef", {}).get("crs") in ("UNKNOWN",) or isinstance(p.get("georef", {}).get("crs"), str))
    m = p.get("meta", {})
    check(f"{name}: meta provenance",
          all(k in m for k in ("model_tag", "device", "dn_source", "mode", "source_shape")))
    check(f"{name}: grid capped", max(g.get("height", 0), g.get("width", 0)) <= 512)


def main() -> int:
    work = Path(tempfile.mkdtemp(prefix="dw-e2e-"))
    print(f"== DepthWizard e2e bridge check (work={work}) ==")

    # ------------------------------------------------------------------
    ckpt = work / "best.pt"
    make_checkpoint(ckpt)
    img_plain = work / "plain.png"
    img_geo = work / "geo.tif"
    dem = work / "dem.tif"
    make_image(img_plain)
    make_image(img_geo, georef=True)
    make_dem(dem)

    cfg = work / "e2e_infer.yaml"
    cache = work / "cache" / "e2e"          # model-tagged cache subdir
    cache.mkdir(parents=True, exist_ok=True)
    cfg.write_text(f"""
paths:
  depth_cache_dir: {cache}
  outputs_dir: {work / "out"}
infer:
  checkpoint: {ckpt}
  device: auto
  mode: auto
  live_backbone: false
""")

    # ------------------------------------------------------------------
    print("\n[1/3] CLI transport: python model.py infer --json-out")
    json_out = work / "payload_cli.json"
    r = subprocess.run(
        [sys.executable, "model.py", "infer", "--config", str(cfg),
         "--input", str(img_plain), "--json-out", str(json_out),
         "--out", str(work / "out" / "cli"), "--no-live"],
        cwd=ROOT, capture_output=True, text=True, timeout=600)
    check("CLI fails honestly without any Dn source", r.returncode != 0)
    check("CLI failure names the Dn problem", "no Dn" in (r.stdout + r.stderr))
    # now provide the cache -> must succeed
    raw = np.tile(np.linspace(1, 5, 300), (300, 1)).astype(np.float32)
    np.save(cache / "plain.npy", raw)
    r2 = subprocess.run(
        [sys.executable, "model.py", "infer", "--config", str(cfg),
         "--input", str(img_plain), "--json-out", str(json_out),
         "--out", str(work / "out" / "cli2")],
        cwd=ROOT, capture_output=True, text=True, timeout=600)
    check("CLI with cache exit 0", r2.returncode == 0, r2.stderr[-300:] if r2.returncode else "")
    if json_out.exists():
        p = json.loads(json_out.read_text())
        payload_contract_ok("CLI+cache", p)
        check("CLI+cache: dn_source == cache", p["meta"]["dn_source"] == "cache")
        check("CLI+cache: affine exactness (min >= 0)",
              p["stats"]["min"] >= 0.0, f"min={p['stats']['min']:.3f}")

    # anchored run (georef image + DEM)
    json_anc = work / "payload_anchored.json"
    r3 = subprocess.run(
        [sys.executable, "model.py", "infer", "--config", str(cfg),
         "--input", str(img_geo), "--json-out", str(json_anc),
         "--out", str(work / "out" / "anc"), "--anchor-dem", str(dem),
         "--no-live"],
        cwd=ROOT, capture_output=True, text=True, timeout=600)
    # note: geo.tif stem is 'geo', cache has only 'plain.npy' -> no cache hit,
    # live off -> this MUST fail honestly:
    check("anchored run without Dn source fails honestly", r3.returncode != 0)
    np.save(cache / "geo.npy", raw)
    r3 = subprocess.run(
        [sys.executable, "model.py", "infer", "--config", str(cfg),
         "--input", str(img_geo), "--json-out", str(json_anc),
         "--out", str(work / "out" / "anc"), "--anchor-dem", str(dem),
         "--cache-dir", str(cache)],
        cwd=ROOT, capture_output=True, text=True, timeout=600)
    check("anchored CLI exit 0", r3.returncode == 0, r3.stderr[-300:] if r3.returncode else "")
    if json_anc.exists():
        p = json.loads(json_anc.read_text())
        payload_contract_ok("CLI+anchored", p)
        check("anchored label present",
              isinstance(p.get("anchored"), dict)
              and p["anchored"]["label"] == "ANCHORED (not learned)")
        check("anchored source is dem", str(p["anchored"]["source"]).startswith("dem:"))
        check("georef CRS surfaced", p["georef"]["crs"] == "EPSG:32617")
        check("anchored outputs include dsm_anchored",
              bool(p.get("outputs", {}).get("dsm_anchored")))
        check("stats line mentions anchored", "ANCHORED" in r3.stdout)

    # ------------------------------------------------------------------
    print("\n[2/3] Service transport: FastAPI /predict")
    import importlib
    import os
    os.environ["DW_ROOT"] = str(ROOT)
    os.environ["DW_CKPT"] = str(ckpt)
    os.environ["DW_CACHE_DIR"] = str(cache)
    os.environ["DW_NO_LIVE"] = "1"
    os.environ["DW_OUT_ROOT"] = str(work / "out" / "service")
    for mod in ("service.api", "depthwizard.inference"):
        sys.modules.pop(mod, None)
    api = importlib.import_module("service.api")
    from fastapi.testclient import TestClient

    with TestClient(api.app) as client:
        h = client.get("/health").json()
        check("/health ok", h.get("status") == "ok" and h.get("checkpoint_exists") is True)

        with open(img_plain, "rb") as f:
            resp = client.post("/predict",
                               files={"image": ("plain.png", f, "image/png")},
                               data={"mode": "resize"})
        check("/predict 200", resp.status_code == 200, resp.text[:300])
        if resp.status_code == 200:
            p = resp.json()
            payload_contract_ok("SERVICE", p)
            check("SERVICE: dn_source == cache", p["meta"]["dn_source"] == "cache")
            check("SERVICE: request_id assigned", bool(p.get("request_id")))

        # anchored via service
        with open(img_geo, "rb") as f, open(dem, "rb") as df:
            resp = client.post("/predict",
                               files={"image": ("geo.tif", f, "image/tiff"),
                                      "anchor_dem": ("dem.tif", df, "image/tiff")},
                               data={"ground_elev": ""})
        check("/predict anchored 200", resp.status_code == 200, resp.text[:300])
        if resp.status_code == 200:
            p = resp.json()
            check("SERVICE anchored label",
                  isinstance(p.get("anchored"), dict)
                  and p["anchored"]["label"] == "ANCHORED (not learned)")

        # bad mode -> 400
        with open(img_plain, "rb") as f:
            resp = client.post("/predict",
                               files={"image": ("plain.png", f, "image/png")},
                               data={"mode": "bogus"})
        check("bad mode -> 400", resp.status_code == 400)

        # no image -> 422
        resp = client.post("/predict", data={})
        check("missing image -> 4xx", 400 <= resp.status_code < 500)

    # ------------------------------------------------------------------
    print("\n[3/3] Summary")
    if failures:
        print(f"  {FAIL} {len(failures)} failing: {failures}")
        return 1
    print(f"  {PASS} all checks green — CLI and service emit the same contract")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
