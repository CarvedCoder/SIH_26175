"""Tests for the TerraHeight-S backend integration (depthwizard/terraheight.py).

Mirrors model_tests/test_rdah.py's acceptance surface (task Sec. 26):
  * checkpoint discovery + payload validation — missing/foreign payloads
    fail LOUDLY, nothing is auto-downloaded
  * architecture construction — the EXACT released DepthAnythingV2 config
    (encoder="vits", features=64, out_channels=[48,96,192,384]) via the
    vendored official implementation
  * preprocessing — uint8 -> /255 -> ImageNet normalization exactly once
  * output scaling — metres = raw x scale_m (normalized transform),
    clamped to >= 0 (both facts read from the checkpoint payload)
  * tiling — overlapping windows, edge padding, full-coverage stitching,
    output grid == input grid, cancellation
  * registry — architecture = "terraheight_s" through tifops; the
    rdah/calibration_net paths unchanged
  * end-to-end — RGB -> TerraHeight -> AGL artifacts -> payload, with CRS/
    transform preservation for georeferenced inputs and honest pixel-space
    handling otherwise

Everything runs on a tiny SYNTHETIC checkpoint in the OFFICIAL release
format (random-weight DepthAnythingV2 wrapper) so the suite never needs
the 99 MB download; the real checkpoint is exercised by the opt-in
released-checkpoint test (skipped when absent).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from depthwizard.terraheight import (
    ARCHITECTURE,
    DAV2_PATCH,
    HEIGHT_TYPE,
    TERRAHEIGHT_MEAN,
    TERRAHEIGHT_SCALE_M,
    TERRAHEIGHT_STD,
    default_checkpoint_path,
    is_terraheight_payload,
    load_terraheight_model,
    predict_agl_tiled,
    preprocess_rgb,
    terraheight_scale_m,
    validate_terraheight_payload,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
RELEASED_CKPT = REPO_ROOT / "best_model.pth"

#: Small valid inference size (multiple of 14) — keeps tests fast.
TILE = 112


# ---------------------------------------------------------------------------
# Fixtures — synthetic checkpoint in the OFFICIAL release format
# ---------------------------------------------------------------------------


def _release_payload(core: "torch.nn.Module") -> dict:
    """The official TerraHeight release payload shape (metadata fields are
    copied from the real release; weights are random)."""
    return {
        "program_version": 1,
        "model": {f"net.{k}": v for k, v in core.state_dict().items()},
        "model_config": {
            "encoder": "vits",
            "features": 64,
            "out_channels": [48, 96, 192, 384],
        },
        "transform": {
            "mode": "normalized",
            "scale_m": TERRAHEIGHT_SCALE_M,
            "train_clip_m": None,
            "units": "metres_AGL",
            "negative_training_targets": "clamp_zero",
        },
        "normalization": {"mean": list(TERRAHEIGHT_MEAN), "std": list(TERRAHEIGHT_STD)},
        "best_epoch": 7,
        "crop_size": 630,
        "validation_metrics": {"all": {"mae": 1.312, "rmse": 2.616, "corr": 0.9238}},
    }


@pytest.fixture(scope="module")
def th_ckpt(tmp_path_factory) -> Path:
    """Synthetic TerraHeight release checkpoint (module-scoped: one save)."""
    from depthwizard.vendor.depth_anything_v2.dpt import DepthAnythingV2

    torch.manual_seed(0)
    core = DepthAnythingV2(encoder="vits", features=64, out_channels=[48, 96, 192, 384])
    p = tmp_path_factory.mktemp("terraheight") / "best_model.pth"
    torch.save(_release_payload(core), p)
    return p


@pytest.fixture(scope="module")
def loaded(th_ckpt):
    return load_terraheight_model(th_ckpt, "cpu")


def _rgb(h: int = TILE, w: int = TILE, seed: int = 3) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(0, 255, size=(h, w, 3), dtype=np.uint8)


# ---------------------------------------------------------------------------
# Checkpoint discovery + payload validation
# ---------------------------------------------------------------------------


def test_default_checkpoint_path_found_from_repo_root():
    # the real repo carries the checkpoint at its root (or under
    # models/terraheight/) — discovery must find one of the candidates
    p = default_checkpoint_path(REPO_ROOT)
    assert p.exists()


def test_default_checkpoint_path_missing_raises(tmp_path):
    from depthwizard.terraheight import DEFAULT_CHECKPOINT_CANDIDATES

    with pytest.raises(FileNotFoundError) as e:
        default_checkpoint_path(tmp_path)
    # loud, actionable: names every searched location, no download
    for rel in DEFAULT_CHECKPOINT_CANDIDATES:
        assert str(tmp_path / rel) in str(e.value)
    assert "DW_CKPT_TERRAHEIGHT" in str(e.value)


def test_payload_validation_accepts_release_shape(th_ckpt):
    import torch as _t

    ckpt = _t.load(th_ckpt, map_location="cpu", weights_only=True)
    validate_terraheight_payload(ckpt)  # must not raise
    assert is_terraheight_payload(ckpt)


@pytest.mark.parametrize("drop", ["model", "model_config", "transform"])
def test_payload_validation_rejects_missing_keys(th_ckpt, drop):
    import torch as _t

    ckpt = _t.load(th_ckpt, map_location="cpu", weights_only=True)
    del ckpt[drop]
    with pytest.raises(ValueError, match=drop):
        validate_terraheight_payload(ckpt)


def test_payload_validation_rejects_foreign_payload():
    with pytest.raises(ValueError):
        validate_terraheight_payload({"model_state": {}, "use_rgb": True})
    with pytest.raises(ValueError):
        validate_terraheight_payload("not a dict")


def test_payload_validation_rejects_wrong_encoder(th_ckpt):
    import torch as _t

    ckpt = _t.load(th_ckpt, map_location="cpu", weights_only=True)
    ckpt["model_config"]["encoder"] = "vitl"
    with pytest.raises(ValueError, match="encoder"):
        validate_terraheight_payload(ckpt)


def test_payload_validation_rejects_unknown_transform_mode(th_ckpt):
    import torch as _t

    ckpt = _t.load(th_ckpt, map_location="cpu", weights_only=True)
    ckpt["transform"]["mode"] = "log1p"
    with pytest.raises(ValueError, match="normalized"):
        validate_terraheight_payload(ckpt)


def test_scale_read_from_checkpoint(th_ckpt):
    import torch as _t

    ckpt = _t.load(th_ckpt, map_location="cpu", weights_only=True)
    assert terraheight_scale_m(ckpt) == pytest.approx(TERRAHEIGHT_SCALE_M)
    ckpt["transform"]["scale_m"] = float("nan")
    with pytest.raises(ValueError, match="scale_m"):
        terraheight_scale_m(ckpt)


# ---------------------------------------------------------------------------
# Architecture construction + loading
# ---------------------------------------------------------------------------


def test_load_rebuilds_exact_released_architecture(loaded):
    from depthwizard.vendor.depth_anything_v2.dpt import DepthAnythingV2

    core = loaded.net.core
    assert isinstance(core, DepthAnythingV2)
    assert core.encoder == "vits"
    # official DPT head: scratch layer1_rn is `features` wide; the
    # fusion projects carry the published out_channels for vits
    assert core.depth_head.scratch.layer1_rn.weight.shape[0] == 64
    assert [p.weight.shape[0] for p in core.depth_head.projects] == [48, 96, 192, 384]
    assert loaded.architecture == ARCHITECTURE == "terraheight_s"
    assert loaded.height_type == HEIGHT_TYPE == "AGL"
    assert loaded.use_rgb is True
    assert loaded.needs_stats is False
    assert loaded.epoch == 7
    n_par = sum(p.numel() for p in loaded.net.parameters())
    assert n_par == 24_785_089  # released TerraHeight-S parameter count


def test_load_strips_module_prefix(tmp_path):
    """``module.net.*`` keys are tolerated (wrapper + DDP prefixes)."""
    from depthwizard.vendor.depth_anything_v2.dpt import DepthAnythingV2

    torch.manual_seed(0)
    core = DepthAnythingV2(encoder="vits", features=64, out_channels=[48, 96, 192, 384])
    p = tmp_path / "ckpt.pth"
    payload = _release_payload(core)
    payload["model"] = {f"module.{k}": v for k, v in payload["model"].items()}
    torch.save(payload, p)
    model = load_terraheight_model(p, "cpu")
    assert model.architecture == "terraheight_s"


def test_load_rejects_incomplete_state_dict(th_ckpt, tmp_path):
    import torch as _t

    ckpt = _t.load(th_ckpt, map_location="cpu", weights_only=True)
    first = next(iter(ckpt["model"]))
    del ckpt["model"][first]
    p = tmp_path / "truncated.pth"
    _t.save(ckpt, p)
    with pytest.raises(RuntimeError):
        load_terraheight_model(p, "cpu")


# ---------------------------------------------------------------------------
# Preprocessing + forward contract
# ---------------------------------------------------------------------------


def test_preprocess_rgb_imagenet_exactly_once():
    rgb = _rgb(28, 28)
    x = preprocess_rgb(rgb)
    assert x.shape == (3, 28, 28) and x.dtype == np.float32
    expected = (rgb[:, :, :3].transpose(2, 0, 1).astype(np.float32) / 255.0
                - np.asarray(TERRAHEIGHT_MEAN, np.float32).reshape(3, 1, 1)
                ) / np.asarray(TERRAHEIGHT_STD, np.float32).reshape(3, 1, 1)
    np.testing.assert_allclose(x, expected, rtol=1e-6)


def test_forward_shapes_and_batch(loaded):
    net = loaded.net
    x = torch.rand(2, 3, 28, 28)
    with torch.no_grad():
        out = net(x)
    assert out["pred"].shape == (2, 1, 28, 28)


def test_forward_rejects_non_multiple_of_patch(loaded):
    with pytest.raises(ValueError, match="14"):
        loaded.net(torch.rand(1, 3, 27, 28))


def test_forward_rejects_depth_cache_inputs(loaded):
    # TerraHeight is RGB-only: a Dn/stats input means the caller took the
    # RDAH path by mistake — loud error, never silent acceptance.
    x = torch.rand(1, 3, 28, 28)
    with pytest.raises(ValueError, match="RGB ONLY"):
        loaded.net(x, dn=torch.rand(1, 1, 28, 28))
    with pytest.raises(ValueError, match="RGB ONLY"):
        loaded.net(x, stats=torch.zeros(1, 4))


def test_output_is_metr_agl_clamped_nonnegative(loaded):
    # scale_m conversion: the wrapper multiplies the core's raw output by
    # scale_m; with a clamped core (ReLU inside DAv2) the result is >= 0.
    net = loaded.net
    x = torch.rand(1, 3, 28, 28)
    with torch.no_grad():
        out = net(x)["pred"]
    assert float(out.min()) >= 0.0
    with torch.no_grad():
        raw = net.core(x)
    np.testing.assert_allclose(
        out[0, 0].numpy(), np.clip(raw[0].numpy() * net.scale_m, 0.0, None),
        rtol=1e-5,
    )


# ---------------------------------------------------------------------------
# Predict-function factories (registry contract)
# ---------------------------------------------------------------------------


def test_predict_fn_contract(loaded):
    from depthwizard.tifops import make_predict_fn

    predict = make_predict_fn(loaded, "cpu")
    rgb = _rgb()
    out = predict(None, rgb, None)
    assert out.shape == (TILE, TILE) and out.dtype == np.float32
    assert np.isfinite(out).all() and (out >= 0).all()
    with pytest.raises(ValueError, match="RGB ONLY|does not consume"):
        predict(np.zeros((TILE, TILE), np.float32), rgb, None)  # dn supplied
    with pytest.raises(ValueError):
        predict(None, None, None)


def test_full_predict_fn_reports_no_semantics(loaded):
    from depthwizard.tifops import make_full_predict_fn

    out = make_full_predict_fn(loaded, "cpu")(None, _rgb(), None)
    assert out["sem_probs"] is None
    assert out["pred"].shape == (TILE, TILE)


# ---------------------------------------------------------------------------
# Tiled inference
# ---------------------------------------------------------------------------


def test_tiled_output_matches_input_grid(loaded):
    rgb = _rgb(h=150, w=210)
    agl, n = predict_agl_tiled(loaded, rgb, device="cpu", tile_size=56,
                               overlap=14, log=lambda *_: None)
    assert agl.shape == (150, 210)
    assert n >= 4  # 150x210 with 56 tiles + snapping covers with overlaps
    assert np.isfinite(agl).all()


def test_tiled_single_small_tile_edge_padded(loaded):
    rgb = _rgb(h=40, w=50)
    agl, n = predict_agl_tiled(loaded, rgb, device="cpu", tile_size=56,
                               overlap=14, log=lambda *_: None)
    assert agl.shape == (40, 50) and n == 1


def test_tiled_defaults_use_published_crop(loaded):
    from depthwizard.terraheight import TERRAHEIGHT_DEFAULT_OVERLAP, TERRAHEIGHT_TILE_SIZE

    assert TERRAHEIGHT_TILE_SIZE == 630
    assert TERRAHEIGHT_DEFAULT_OVERLAP == 157  # 0.25 overlap, the ckpt's own


@pytest.mark.parametrize("tile,overlap", [(63, 14), (56, 56), (0, 0), (56, -1)])
def test_tiled_rejects_invalid_tiling(loaded, tile, overlap):
    with pytest.raises(ValueError):
        predict_agl_tiled(loaded, _rgb(28, 28), device="cpu",
                          tile_size=tile, overlap=overlap, log=lambda *_: None)


def test_tiled_honours_cancellation(loaded):
    from depthwizard.inference import InferenceCancelled

    with pytest.raises(InferenceCancelled):
        predict_agl_tiled(
            loaded, _rgb(h=150, w=150), device="cpu", tile_size=56, overlap=14,
            should_cancel=lambda: True, log=lambda *_: None,
        )


def test_tiled_batch_size_one_only(loaded):
    with pytest.raises(NotImplementedError):
        predict_agl_tiled(loaded, _rgb(28, 28), device="cpu", batch_size=2,
                          log=lambda *_: None)


# ---------------------------------------------------------------------------
# Registry (tifops) — one entry point for all backends
# ---------------------------------------------------------------------------


def test_registry_detects_and_loads_terraheight(th_ckpt, loaded):
    from depthwizard.tifops import detect_architecture, load_height_model

    assert detect_architecture(th_ckpt) == "terraheight_s"
    m = load_height_model(th_ckpt, "cpu", architecture="terraheight_s")
    assert m.architecture == "terraheight_s"
    assert m.tag == "terraheight_s_best_model"
    # auto-detect path (no explicit architecture) lands in the same loader
    m2 = load_height_model(th_ckpt, "cpu")
    assert m2.architecture == "terraheight_s"


def test_registry_rejects_unknown_architecture(loaded):
    from depthwizard.tifops import load_height_model

    with pytest.raises(ValueError, match="unknown architecture"):
        load_height_model(
            loaded.checkpoint, "cpu", architecture="terraheight"
        )


def test_registry_rdah_path_unchanged(tmp_path):
    """RDAH regression: the rdah branch still loads an official-format
    checkpoint (the terraheight addition must not disturb it)."""
    from depthwizard.rdah_net import HeightPredTransformer
    from depthwizard.tifops import detect_architecture, load_height_model

    p = tmp_path / "rdah.pth"
    torch.save({"epoch": 1, "model_state_dict": HeightPredTransformer().state_dict()}, p)
    assert detect_architecture(p) == "rdah"
    m = load_height_model(p, "cpu")
    assert m.architecture == "rdah" and m.height_type == "nDSM"


# ---------------------------------------------------------------------------
# End-to-end: RGB -> TerraHeight -> artifacts -> payload
# ---------------------------------------------------------------------------


def _write_geotiff(path: Path, h=140, w=160) -> None:
    import rasterio
    from rasterio.transform import from_origin

    rng = np.random.default_rng(5)
    data = rng.integers(0, 255, size=(3, h, w), dtype=np.uint8)
    with rasterio.open(
        path, "w", driver="GTiff", height=h, width=w, count=3, dtype="uint8",
        crs="EPSG:32633", transform=from_origin(500000, 5000000, 0.33, 0.33),
    ) as dst:
        dst.write(data)


def test_run_inference_georeferenced_preserves_crs(th_ckpt, tmp_path):
    import rasterio

    from depthwizard.inference import run_inference

    tif = tmp_path / "scene.tif"
    _write_geotiff(tif, h=140, w=168)

    payload = run_inference(
        tif, th_ckpt, out_dir=tmp_path / "out", device="cpu",
        architecture="terraheight_s", terraheight_tile_size=56,
        terraheight_overlap=14,
    )
    meta = payload["meta"]
    assert meta["model_architecture"] == "terraheight_s"
    assert meta["height_type"] == "AGL"
    assert "directly from RGB" in meta["dn_source"]
    assert meta["height_model_label"] == "TerraHeight-S"
    # standard downstream contract intact
    assert (tmp_path / "out" / "dsm.npy").exists()
    # TerraHeight provenance artifacts
    assert (tmp_path / "out" / "terraheight_meta.json").exists()
    assert (tmp_path / "out" / "terraheight_preview.png").exists()
    agl_tif = tmp_path / "out" / "terraheight_agl.tif"
    assert agl_tif.exists()
    with rasterio.open(tif) as src, rasterio.open(agl_tif) as out:
        assert out.crs == src.crs == rasterio.crs.CRS.from_epsg(32633)
        assert out.transform == src.transform
        assert (out.height, out.width) == (src.height, src.width)
        agl = out.read(1)
    assert agl.shape == (140, 168)
    assert np.isfinite(agl).all() and (agl >= 0).all()
    th_meta = json.loads((tmp_path / "out" / "terraheight_meta.json").read_text())
    assert th_meta["architecture"] == "terraheight_s"
    assert th_meta["height_type"] == "AGL"
    assert th_meta["parameter_count"] == 24_785_089
    assert th_meta["tile_size"] == 56 and th_meta["tile_stride"] == 42
    assert th_meta["batch_size"] == 1
    assert th_meta["georeferenced"] is True
    assert th_meta["input_dimensions"] == [140, 168]
    assert "scale_m" in th_meta
    assert payload["semantic"]["available"] is False


def test_run_inference_non_georeferenced_no_fake_crs(th_ckpt, tmp_path):
    from PIL import Image

    from depthwizard.inference import run_inference

    png = tmp_path / "photo.png"
    Image.fromarray(_rgb(120, 140)).save(png)
    payload = run_inference(
        png, th_ckpt, out_dir=tmp_path / "out", device="cpu",
        architecture="terraheight_s", terraheight_tile_size=56,
        terraheight_overlap=14,
    )
    assert payload["meta"]["model_architecture"] == "terraheight_s"
    assert payload["georef"]["crs"] == "UNKNOWN"
    assert payload["terraheight"]["georeferenced"] is False
    assert "pixel-space" in payload["terraheight"]["georef_state"]
    assert (tmp_path / "out" / "terraheight_agl.npy").exists()
    assert not (tmp_path / "out" / "terraheight_agl.tif").exists()
    assert not (tmp_path / "out" / "dsm.tif").exists()  # never fabricated


def test_run_inference_auto_detects_terraheight_checkpoint(th_ckpt, tmp_path):
    from PIL import Image

    from depthwizard.inference import run_inference

    png = tmp_path / "photo.png"
    Image.fromarray(_rgb(56, 56)).save(png)
    payload = run_inference(
        png, th_ckpt, out_dir=tmp_path / "out", device="cpu",
        terraheight_tile_size=56, terraheight_overlap=14,
    )
    assert payload["meta"]["model_architecture"] == "terraheight_s"


# ---------------------------------------------------------------------------
# Opt-in: the REAL released checkpoint (skipped when absent)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not RELEASED_CKPT.exists(), reason="released TerraHeight-S checkpoint not present")
def test_released_checkpoint_end_to_end(tmp_path):
    import rasterio

    from depthwizard.inference import run_inference

    tif = tmp_path / "scene.tif"
    _write_geotiff(tif, h=630, w=630)
    payload = run_inference(
        tif, RELEASED_CKPT, out_dir=tmp_path / "out", device="cpu",
        architecture="terraheight_s",
    )
    assert payload["meta"]["model_architecture"] == "terraheight_s"
    with rasterio.open(tmp_path / "out" / "terraheight_agl.tif") as out:
        agl = out.read(1)
    assert np.isfinite(agl).all() and (agl >= 0).all()
    stats = payload["stats"]
    assert stats["max"] <= float(agl.max()) + 1e-3
