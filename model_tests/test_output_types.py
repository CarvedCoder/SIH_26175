"""Model selection + output-type contracts (Parts E, O, U, D).

End-to-end on the torch-free-ish CalibrationNet path (synthetic
checkpoint, CPU, no downloads), plus a dispatch test for terraheight_s.

Pins:
  * requested model == executed model == recorded model
  * GeoTIFF + no DEM       -> output_type relative_height, CRS preserved,
                              absolute_reference_available false
  * GeoTIFF + valid DEM    -> output_type absolute_dsm, dsm_anchored.tif
                              written with CRS+transform preserved,
                              provenance present and truthful
  * PNG/JPG (no georef)    -> relative/image-space product, no fake CRS,
                              no absolute claim
  * downstream payload keys the frontend consumes stay intact
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest
import rasterio
from rasterio.crs import CRS
from rasterio.transform import from_origin

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

UTM = CRS.from_epsg(32617)
TF = from_origin(500000, 4000000, 1.0, 1.0)


@pytest.fixture(scope="module")
def calib_ckpt(tmp_path_factory):
    torch = pytest.importorskip("torch")
    from depthwizard.calibration_net import CalibrationNet

    tmp = tmp_path_factory.mktemp("ckpt")
    net = CalibrationNet(in_ch=1, widths=(8, 16, 32), a0=2.0, b0=4.0)
    p = tmp / "best.pt"
    torch.save({
        "model_state": net.state_dict(), "use_rgb": False,
        "widths": [8, 16, 32], "clamp_min": 0.0,
        "affine_init": {"a": 2.0, "b": 4.0}, "loss": "l1", "epoch": 0,
        "val_subset_mae": 0.0, "splits_json": "n/a",
    }, p)
    return p


def _write_rgb_geotiff(path, n=96, with_crs=True):
    rng = np.random.default_rng(0)
    rgb = rng.integers(0, 255, (n, n, 3), dtype=np.uint8)
    profile = {
        "driver": "GTiff", "height": n, "width": n, "count": 3,
        "dtype": "uint8",
        "crs": UTM if with_crs else None,
        "transform": TF if with_crs else from_origin(0, n, 1.0, 1.0),
    }
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(rgb.transpose(2, 0, 1))
    return path


def _write_dem(path, n=96):
    arr = np.full((n, n), 100.0, dtype=np.float32)
    with rasterio.open(
        path, "w", driver="GTiff", height=n, width=n, count=1,
        dtype="float32", crs=UTM, transform=TF,
    ) as dst:
        dst.write(arr, 1)
    return path


def _dn_path(tmp_path, n=96):
    xs = np.linspace(0.0, 1.0, n, dtype=np.float32)
    dn = np.tile(xs[None, :], (n, 1))
    p = tmp_path / "dn.npy"
    np.save(p, dn * 10.0)  # raw relative depth
    return p


def _run(tmp_path, calib_ckpt, image, **kw):
    from depthwizard.inference import run_inference

    out_dir = tmp_path / "out"
    payload = run_inference(
        image, calib_ckpt, out_dir=out_dir, device="cpu", mode="crop",
        dn_path=_dn_path(tmp_path, _img_size(image)),
        live_backbone=False, architecture="calibration_net", **kw,
    )
    return payload, out_dir


def _img_size(image):
    with rasterio.open(image) as src:
        return src.height


# ---------------------------------------------------------------------------
# requested == executed == recorded (Part U)
# ---------------------------------------------------------------------------

def test_requested_model_is_recorded_end_to_end(tmp_path, calib_ckpt):
    img = _write_rgb_geotiff(tmp_path / "img.tif")
    payload, _ = _run(tmp_path, calib_ckpt, img)
    meta = payload["meta"]
    assert meta["model_architecture"] == "calibration_net"  # requested
    assert meta["model_tag"]  # executed tag recorded
    assert meta["output_type"] == "relative_height"
    assert meta["absolute_reference_available"] is False
    assert "provenance" in meta


def test_terraheight_dispatch_preserves_model_selection(monkeypatch, tmp_path):
    """architecture='terraheight_s' MUST execute the terraheight path."""
    from depthwizard import inference as inf

    calls = {}

    def fake_th_inference(*args, **kwargs):
        calls["called"] = True
        calls["ckpt"] = args[1] if len(args) > 1 else kwargs.get("ckpt_path")
        return {"meta": {}}

    monkeypatch.setattr(inf, "run_terraheight_inference", fake_th_inference,
                        raising=False)
    # run_inference imports the function inside the branch
    import depthwizard.terraheight as th
    monkeypatch.setattr(th, "run_terraheight_inference", fake_th_inference)

    inf.run_inference(
        tmp_path / "any.tif", tmp_path / "ck.pth", out_dir=tmp_path / "o",
        architecture="terraheight_s",
    )
    assert calls.get("called") is True
    assert str(calls.get("ckpt")) == str(tmp_path / "ck.pth")


# ---------------------------------------------------------------------------
# CASE 3: GeoTIFF, NO DEM -> relative, honest
# ---------------------------------------------------------------------------

def test_geotiff_no_dem_is_relative_with_crs_preserved(tmp_path, calib_ckpt):
    img = _write_rgb_geotiff(tmp_path / "img.tif")
    payload, out_dir = _run(tmp_path, calib_ckpt, img)
    meta = payload["meta"]
    assert meta["output_type"] == "relative_height"
    assert meta["absolute_reference_available"] is False
    assert "NOT absolute terrain elevation" in meta["height_semantics"]
    # CRS + transform preserved on the dsm.tif product
    dsm_tif = out_dir / "dsm.tif"
    assert dsm_tif.exists()
    with rasterio.open(dsm_tif) as src:
        assert src.crs == UTM
        assert src.transform == TF
        assert (src.height, src.width) == (96, 96)
        assert src.dtypes[0] == "float32"
    assert not (out_dir / "dsm_anchored.tif").exists()


# ---------------------------------------------------------------------------
# CASE 1: GeoTIFF + valid DEM -> absolute DSM, provenance, preserved grid
# ---------------------------------------------------------------------------

def test_geotiff_with_dem_absolute_dsm_end_to_end(tmp_path, calib_ckpt):
    img = _write_rgb_geotiff(tmp_path / "img.tif")
    dem = _write_dem(tmp_path / "dtm.tif")
    payload, out_dir = _run(tmp_path, calib_ckpt, img, anchor_dem=dem)

    meta = payload["meta"]
    assert meta["output_type"] == "absolute_dsm"
    assert meta["absolute_reference_available"] is True
    assert meta["absolute_dsm_available"] is True
    prov = meta["provenance"]
    assert prov["dem_source"] == "local:dtm.tif"
    assert prov["scene_crs"] == str(UTM)
    assert prov["scene_gsd_m"] == pytest.approx(1.0)
    assert prov["height_model"] == "calibration_net"
    assert prov["height_units"] == "meters"
    assert prov["alignment"]["formula"] == "DSM_absolute = DEM_reference + AGL_prediction"

    # anchored product written on the EXACT input grid
    anchored = out_dir / "dsm_anchored.tif"
    assert anchored.exists()
    with rasterio.open(anchored) as src:
        assert src.crs == UTM
        assert src.transform == TF
        assert (src.height, src.width) == (96, 96)
        arr = src.read(1)
        assert arr.min() > 95.0  # DEM datum 100 m + non-negative AGL

    # machine-readable provenance file exists and agrees
    prov_file = out_dir / "absolute_dsm_provenance.json"
    assert prov_file.exists()
    doc = json.loads(prov_file.read_text())
    assert doc["dem_source"] == "local:dtm.tif"
    assert doc["height_model"] == "calibration_net"


# ---------------------------------------------------------------------------
# CASE 4: PNG (no georeferencing) -> relative, no fake CRS
# ---------------------------------------------------------------------------

def test_png_relative_no_fake_crs(tmp_path, calib_ckpt):
    from PIL import Image

    img_path = tmp_path / "photo.png"
    Image.fromarray(
        np.random.default_rng(1).integers(0, 255, (96, 96, 3), dtype=np.uint8)
    ).save(img_path)

    payload, out_dir = _run(tmp_path, calib_ckpt, img_path)
    meta = payload["meta"]
    assert meta["output_type"] == "relative_height"
    assert meta["absolute_reference_available"] is False
    assert payload["georef"]["crs"] == "UNKNOWN"
    assert payload["meta"]["pixel_size_m"] is None
    # no GeoTIFF twin is fabricated for a non-georeferenced input
    assert not (out_dir / "dsm.tif").exists()


def test_png_with_dem_requested_raises_crs_required(tmp_path, calib_ckpt):
    from PIL import Image

    img_path = tmp_path / "photo.png"
    Image.fromarray(np.zeros((96, 96, 3), np.uint8)).save(img_path)
    from depthwizard.statuses import CRS_REQUIRED, DWStatusError

    with pytest.raises(DWStatusError) as ei:
        _run(tmp_path, calib_ckpt, img_path, dem_provider="copernicus")
    assert ei.value.code == CRS_REQUIRED
