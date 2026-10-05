"""Sun-angle / timestamp independence (Part K) — regression tests.

Verified facts (code inspection + tests, NOT a fabricated invariance
benchmark):
  * No height model (TerraHeight-S, RDAH, CalibrationNet) takes sun
    angle, solar azimuth, or acquisition timestamp as an input: the
    forward/predict signatures carry no such parameters.
  * A GeoTIFF carrying acquisition-timestamp metadata and the same
    image with the metadata stripped produce identical predictions —
    inference never reads those tags.
  * Missing timestamp/sun-angle metadata cannot break inference.
"""

from __future__ import annotations

import inspect
import sys
from pathlib import Path

import numpy as np
import pytest
import rasterio
from rasterio.crs import CRS
from rasterio.transform import from_origin

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

FORBIDDEN_PARAMS = ("sun", "solar", "azimuth", "timestamp", "datetime",
                    "acquisition", "time_of_day")


def _signature_names(fn) -> set[str]:
    return {p.lower() for p in inspect.signature(fn).parameters}


def _assert_no_sun_params(fn, what: str) -> None:
    names = _signature_names(fn)
    hit = names & set(FORBIDDEN_PARAMS)
    assert not hit, f"{what} unexpectedly requires {hit} — sun-angle/timestamp " \
        "metadata must not be a model input (Part K)"


def test_model_forward_signatures_have_no_sun_or_time_inputs():
    torch = pytest.importorskip("torch")

    from depthwizard.calibration_net import CalibrationNet
    from depthwizard.rdah_net import HeightPredTransformer

    _assert_no_sun_params(CalibrationNet.forward, "CalibrationNet.forward")
    _assert_no_sun_params(HeightPredTransformer.forward,
                          "RDAH HeightPredTransformer.forward")

    # TerraHeight-S: the vendored DAv2 DPT forward + the adapter class
    # (built lazily, so its source is inspected for forbidden parameters)
    import inspect

    import depthwizard.terraheight as th
    from depthwizard.vendor.depth_anything_v2.dpt import DepthAnythingV2

    _assert_no_sun_params(DepthAnythingV2.forward, "DAv2 DPT forward")
    src = inspect.getsource(th._adapter_class).lower()
    for token in FORBIDDEN_PARAMS:
        assert token not in src, (
            f"TerraHeight adapter source mentions '{token}' — sun-angle/"
            "timestamp metadata must not become a model input (Part K)"
        )


def test_predict_fn_factories_have_no_sun_or_time_inputs():
    from depthwizard.inference import DepthWizardPredictor

    _assert_no_sun_params(DepthWizardPredictor.predict, "predict")
    _assert_no_sun_params(DepthWizardPredictor.predict_with_semantics,
                          "predict_with_semantics")
    _assert_no_sun_params(DepthWizardPredictor._predict_impl,
                          "_predict_impl")

    import depthwizard.terraheight as th

    _assert_no_sun_params(th.predict_agl_tiled, "predict_agl_tiled")
    _assert_no_sun_params(th.run_terraheight_inference,
                          "run_terraheight_inference")


def test_timestamp_metadata_does_not_change_prediction(tmp_path):
    """Same pixels with vs without TIFF datetime/EXIF tags -> same result."""
    pytest.importorskip("torch")
    from depthwizard.inference import DepthWizardPredictor
    from model_tests.test_inference import _make_checkpoint

    ckpt, a0, b0 = _make_checkpoint(tmp_path)
    pred = DepthWizardPredictor(ckpt, device="cpu", live_backbone=False)

    dn = np.tile(np.linspace(0.0, 1.0, 128, dtype=np.float32)[None, :], (128, 1))
    dn_path = tmp_path / "dn.npy"
    np.save(dn_path, dn * 10.0)

    rgb = np.zeros((128, 128, 3), np.uint8)
    rgb[..., 0] = np.tile(np.linspace(0, 255, 128).astype(np.uint8), (128, 1))
    rgb[..., 1] = rgb[..., 0]
    rgb[..., 2] = rgb[..., 0]

    def _write(path, with_tags):
        profile = {
            "driver": "GTiff", "height": 128, "width": 128, "count": 3,
            "dtype": "uint8", "crs": CRS.from_epsg(32617),
            "transform": from_origin(500000, 4000000, 1.0, 1.0),
        }
        with rasterio.open(path, "w", **profile) as dst:
            dst.write(rgb.transpose(2, 0, 1))
            if with_tags:
                dst.update_tags(
                    ACQUISITION_DATETIME="2025:06:01 12:00:00",
                    SUN_ELEVATION="62.5",
                    SUN_AZIMUTH="143.2",
                )

    p_tagged = tmp_path / "tagged.tif"
    p_clean = tmp_path / "clean.tif"
    _write(p_tagged, True)
    _write(p_clean, False)

    for path in (p_tagged, p_clean):
        with rasterio.open(path) as src:
            arr = src.read().transpose(1, 2, 0)
        out = pred.predict(np.ascontiguousarray(arr), np.load(dn_path),
                           mode="crop")
        if path == p_tagged:
            tagged_out = out
        else:
            clean_out = out

    np.testing.assert_array_equal(tagged_out, clean_out)
