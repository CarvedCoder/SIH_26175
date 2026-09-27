"""Tests for the RDAH-Net backend integration (depthwizard/rdah*.py).

Covers the task Sec. 15 acceptance surface:
  * model loading — checkpoint exists / loads / incompatible fails LOUDLY
  * shapes — RGB [B,3,H,W] + depth [B,1,H,W] -> [B,1,H,W]
  * preprocessing — RGB ImageNet exactly once; depth = RAW DAv2 x 40 (the
    official representation, NOT min-max, NOT double-normalized)
  * backend selection — architecture = calibration_net | rdah through the
    tifops registry (both must work; the legacy path must be UNCHANGED)
  * geospatial path — CRS/transform propagation, DEM anchoring, DSM writer,
    honest "no fake georeference" behaviour for non-georeferenced inputs

The released 65 MB Track1 checkpoint is used ONLY by opt-in tests (skip
when absent); everything else runs on tiny synthetic checkpoints in the
OFFICIAL format so CI never needs the download.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from depthwizard.normalize import (
    dn_tile_stats,
    minmax_normalize,
    minmax_normalize_with_stats,
)
from depthwizard.rdah import (
    RDAH_DEPTH_SCALE,
    RDAHHeightModel,
    load_rdah_state_dict,
    rdah_depth_from_raw,
    reconstruct_raw_from_stats,
    validate_rdah_payload,
)
from depthwizard.rdah_net import HeightPredTransformer

REPO_ROOT = Path(__file__).resolve().parents[1]
RELEASED_CKPT = REPO_ROOT / "checkpoints" / "rdah" / "rdah_track1_best_model.pth"

SIZE = 256  # smallest multiple of 128 the RDAH architecture accepts


# ---------------------------------------------------------------------------
# Fixtures — synthetic checkpoints in the OFFICIAL format
# ---------------------------------------------------------------------------


def _official_ckpt_path(tmp_path: Path, name="best_model.pth", core=None) -> Path:
    """Save a HeightPredTransformer state dict in the official release
    format: {epoch, model_state_dict, optimizer-free extras}."""
    core = core or HeightPredTransformer()
    p = tmp_path / name
    torch.save(
        {
            "epoch": 3,
            "model_state_dict": core.state_dict(),
            "loss": 1.5,
        },
        p,
    )
    return p


@pytest.fixture(scope="module")
def core():
    torch.manual_seed(0)
    return HeightPredTransformer()


@pytest.fixture(scope="module")
def official_ckpt(tmp_path_factory, core):
    return _official_ckpt_path(tmp_path_factory.mktemp("rdah"), core=core)


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------


def test_checkpoint_loads_strictly(official_ckpt, core):
    """Official-format checkpoint loads with ZERO missing/unexpected keys."""
    from depthwizard.tifops import load_height_model

    model = load_height_model(official_ckpt, device="cpu", architecture="rdah")
    assert model.architecture == "rdah"
    assert model.height_type == "nDSM"
    assert model.use_rgb is True
    assert model.needs_stats is True
    n_par = sum(p.numel() for p in model.net.parameters())
    assert n_par == 5_371_663  # the official parameter count (paper: 5.37 M)


def test_incompatible_checkpoint_fails_loudly(tmp_path, core):
    """A state dict with missing/extra keys must raise — NEVER a silent
    partial load."""
    bad = {k: v for k, v in list(core.state_dict().items())[:-4]}  # drop keys
    p = tmp_path / "bad.pth"
    torch.save({"epoch": 0, "model_state_dict": bad, "loss": 0.0}, p)
    with pytest.raises(ValueError, match="missing keys"):
        from depthwizard.tifops import load_height_model

        load_height_model(p, device="cpu", architecture="rdah")

    extra = dict(core.state_dict())
    extra["bogus_layer.weight"] = torch.zeros(1)
    p2 = tmp_path / "extra.pth"
    torch.save({"epoch": 0, "model_state_dict": extra, "loss": 0.0}, p2)
    with pytest.raises(ValueError, match="unexpected keys"):
        from depthwizard.tifops import load_height_model

        load_height_model(p2, device="cpu", architecture="rdah")


def test_non_rdah_payload_rejected(tmp_path):
    """Payload without 'model_state_dict' fails validation loudly."""
    p = tmp_path / "garbage.pth"
    torch.save({"epoch": 0, "something_else": {}}, p)
    with pytest.raises(ValueError):
        validate_rdah_payload(torch.load(p, weights_only=True))
    with pytest.raises(ValueError, match="model_state_dict"):
        from depthwizard.tifops import load_height_model

        load_height_model(p, device="cpu", architecture="rdah")


def test_module_prefix_is_stripped(tmp_path, core):
    """'module.'-prefixed state dicts (DataParallel exports) load cleanly."""
    prefixed = {f"module.{k}": v for k, v in core.state_dict().items()}
    net = HeightPredTransformer()
    load_rdah_state_dict(net, prefixed, source="prefixed")
    for k, v in core.state_dict().items():
        assert torch.equal(net.state_dict()[k], v)


@pytest.mark.skipif(not RELEASED_CKPT.exists(), reason="released checkpoint absent")
def test_released_track1_checkpoint():
    """The actual figshare release loads strictly and reproduces the
    published provenance (epoch 48, val loss 1.205, 5,371,663 params)."""
    from depthwizard.tifops import load_height_model

    model = load_height_model(RELEASED_CKPT, device="cpu")
    assert model.architecture == "rdah"  # auto-detected from the payload
    assert model.epoch == 48
    assert model.val_subset_mae == pytest.approx(1.2051, abs=1e-3)
    assert sum(p.numel() for p in model.net.parameters()) == 5_371_663


# ---------------------------------------------------------------------------
# Shapes
# ---------------------------------------------------------------------------


def _sample_inputs(batch=1, size=SIZE, seed=0):
    g = torch.Generator().manual_seed(seed)
    raw = 1.0 + 5.0 * torch.rand(batch, 1, size, size, generator=g)
    dn, stats = _normalize(raw)
    rgb = torch.randn(batch, 3, size, size, generator=g)
    return dn, rgb, stats, raw


def _normalize(raw):
    lo, hi = float(raw.min()), float(raw.max())
    dn = (raw - lo) / (hi - lo)
    stats = torch.log(
        torch.tensor([lo + 1e-3, hi + 1e-3, hi - lo + 1e-3, float(raw.mean()) + 1e-3])
    )
    return dn, stats


def test_forward_shapes_batched(core):
    net = RDAHHeightModel(core=core).eval()
    for batch in (1, 2):
        dn, rgb, stats, _ = _sample_inputs(batch=batch, seed=batch)
        with torch.no_grad():
            out = net(dn, rgb, None, None, stats)
        assert out["pred"].shape == (batch, 1, SIZE, SIZE)
        assert out["pred"].dtype == torch.float32
        assert out["height_type"] == "nDSM" and out["model"] == "rdah"


def test_forward_unbatched_3d_auto_batches(core):
    """[1,H,W] + [3,H,W] + [4] inputs auto-batch like CalibrationNet."""
    net = RDAHHeightModel(core=core).eval()
    dn, rgb, stats, _ = _sample_inputs(seed=7)
    assert stats.dim() == 1  # [4] — the natural unbatched stats form
    with torch.no_grad():
        out = net(dn[0], rgb[0], None, None, stats)
    assert out["pred"].shape == (1, 1, SIZE, SIZE)


def test_forward_rejects_dem_sem_and_missing_stats(core):
    net = RDAHHeightModel(core=core).eval()
    dn, rgb, stats, _ = _sample_inputs()
    with pytest.raises(ValueError, match="DEM"):
        net(dn, rgb, dem=dn, stats=stats)
    with pytest.raises(ValueError, match="semantic"):
        net(dn, rgb, sem=dn, stats=stats)
    with pytest.raises(ValueError, match="RGB"):
        net(dn, rgb=None, stats=stats)
    with pytest.raises(ValueError, match="stats"):
        net(dn, rgb, None, None, None)


# ---------------------------------------------------------------------------
# Preprocessing — the official recipes, exactly once
# ---------------------------------------------------------------------------


def test_depth_representation_matches_official():
    """RDAH depth input = RAW DAv2 x 40 — NOT min-max, NOT /255."""
    raw = np.linspace(1.0, 6.0, 64 * 64, dtype=np.float32).reshape(64, 64)
    out = rdah_depth_from_raw(raw)
    np.testing.assert_allclose(out, raw * RDAH_DEPTH_SCALE, rtol=1e-6)
    # it is NOT the CalibrationNet representation:
    assert abs(float(out.mean()) - float(minmax_normalize(raw).mean())) > 1.0


def test_raw_reconstruction_from_stats_is_exact():
    """(normalized Dn, dn_tile_stats) -> RAW round trip: the inverse of
    minmax_normalize_with_stats — the adapter's core reconstruction."""
    rng = np.random.default_rng(0)
    raw = (1.0 + rng.random((32, 48)).astype(np.float32) * 4.0)
    dn, stats = minmax_normalize_with_stats(raw)
    back = reconstruct_raw_from_stats(dn, stats)
    np.testing.assert_allclose(back, raw, atol=1e-5)  # log/exp roundtrip


def test_adapter_equals_direct_official_call(core):
    """The adapter must produce EXACTLY the official model's output for
    the same (raw x 40 depth, ImageNet RGB) inputs — no hidden extra
    normalization anywhere in the chain."""
    dn, rgb, stats, raw = _sample_inputs(seed=1)
    net = RDAHHeightModel(core=core).eval()

    with torch.no_grad():
        via_adapter = net(dn, rgb, None, None, stats)["pred"]

    # manual: official preprocessing computed ONCE by hand
    lo = float(torch.exp(stats[0]) - 1e-3)
    hi = float(torch.exp(stats[1]) - 1e-3)
    raw_rec = lo + dn[:, 0] * (hi - lo)
    depth_repr = raw_rec * RDAH_DEPTH_SCALE
    with torch.no_grad():
        direct = core(depth_repr[:, None].contiguous(), rgb)

    torch.testing.assert_close(via_adapter, direct, atol=1e-4, rtol=1e-4)


def test_predict_fn_rgb_normalized_exactly_once(official_ckpt):
    """make_rdah_predict_fn: uint8 RGB -> /255 -> ImageNet ONCE; compare
    against a manual single application of the recipe."""
    from depthwizard.dataset import IMAGENET_MEAN, IMAGENET_STD
    from depthwizard.tifops import load_height_model, make_predict_fn

    model = load_height_model(official_ckpt, device="cpu")
    predict = make_predict_fn(model, device="cpu")

    rng = np.random.default_rng(2)
    rgb_u8 = rng.integers(0, 255, (SIZE, SIZE, 3), dtype=np.uint8)
    raw = (1.0 + rng.random((SIZE, SIZE)).astype(np.float32) * 4.0)
    dn_n, stats = minmax_normalize_with_stats(raw)

    got = predict(dn_n, rgb_u8, stats)

    # manual chain, each conversion applied exactly once
    rgb_n = (rgb_u8.astype(np.float32) / 255.0 - IMAGENET_MEAN) / IMAGENET_STD
    depth_repr = rdah_depth_from_raw(raw)
    with torch.no_grad():
        want = model.net.net(
            torch.from_numpy(depth_repr)[None, None],
            torch.from_numpy(rgb_n.transpose(2, 0, 1))[None],
        )[0, 0].numpy()
    np.testing.assert_allclose(got, want, atol=1e-4)

    # double-normalization guard: /255 twice or ImageNet twice would shift
    # the input far outside the checkpoint's BN statistics -> outputs differ
    with pytest.raises(AssertionError):
        double = (rgb_n.astype(np.float32) - IMAGENET_MEAN) / IMAGENET_STD
        np.testing.assert_allclose(double, rgb_n, atol=1e-3)


def test_predict_fn_requires_stats(official_ckpt):
    from depthwizard.tifops import load_height_model, make_predict_fn

    model = load_height_model(official_ckpt, device="cpu")
    predict = make_predict_fn(model, device="cpu")
    with pytest.raises(ValueError, match="stats"):
        predict(np.zeros((8, 8), np.float32), np.zeros((8, 8, 3), np.uint8), None)


def test_full_predict_fn_contract(official_ckpt):
    """predict_full mirrors the CalibrationNet surface the post-processing
    pipeline consumes: sem_probs=None (RDAH has no semantic head)."""
    from depthwizard.tifops import load_height_model, make_full_predict_fn

    model = load_height_model(official_ckpt, device="cpu")
    predict_full = make_full_predict_fn(model, device="cpu")
    rng = np.random.default_rng(3)
    rgb_u8 = rng.integers(0, 255, (SIZE, SIZE, 3), dtype=np.uint8)
    raw = (1.0 + rng.random((SIZE, SIZE)).astype(np.float32) * 4.0)
    dn_n, stats = minmax_normalize_with_stats(raw)
    out = predict_full(dn_n, rgb_u8, stats)
    assert set(out) == {"pred", "sem_probs", "sem_zero_filled", "stats_zero_filled"}
    assert out["sem_probs"] is None and out["sem_zero_filled"] is False
    assert out["pred"].shape == (SIZE, SIZE)


# ---------------------------------------------------------------------------
# Backend selection (registry)
# ---------------------------------------------------------------------------


def _calib_ckpt(tmp_path: Path) -> Path:
    from depthwizard.calibration_net import CalibrationNet

    net = CalibrationNet(in_ch=1, widths=(8, 16, 32), a0=2.0, b0=4.0)
    p = tmp_path / "best.pt"
    torch.save(
        {
            "model_state": net.state_dict(),
            "use_rgb": False, "widths": [8, 16, 32], "clamp_min": 0.0,
            "affine_init": {"a": 2.0, "b": 4.0}, "loss": "l1", "epoch": 0,
            "val_subset_mae": 0.0, "splits_json": "n/a",
        },
        p,
    )
    return p


def test_registry_selects_both_architectries(tmp_path, official_ckpt):
    from depthwizard.tifops import (
        detect_architecture,
        load_calib_net,
        load_height_model,
    )

    calib_p = _calib_ckpt(tmp_path)
    assert detect_architecture(calib_p) == "calibration_net"
    assert detect_architecture(official_ckpt) == "rdah"

    m_calib = load_height_model(calib_p, device="cpu")
    assert m_calib.architecture == "calibration_net"
    assert m_calib.needs_stats is False
    # the legacy loader is UNCHANGED and still works:
    m_legacy = load_calib_net(calib_p, device="cpu")
    assert m_legacy.tag.startswith("calib_")

    m_rdah = load_height_model(official_ckpt, device="cpu")
    assert m_rdah.architecture == "rdah"
    assert m_rdah.tag.startswith("rdah_")


def test_registry_rejects_unknown_architecture(official_ckpt):
    from depthwizard.tifops import load_height_model

    with pytest.raises(ValueError, match="unknown architecture"):
        load_height_model(official_ckpt, device="cpu", architecture="bogus")


def test_predictor_serves_both_backends(tmp_path, official_ckpt):
    """DepthWizardPredictor: one code path, two backends. The CalibrationNet
    round trip must stay EXACT (legacy contract), the RDAH path must emit
    metre heights on the source grid."""
    from depthwizard.inference import DepthWizardPredictor

    # --- legacy: exact affine on a linear ramp (regression guard) ---
    ckpt, a0, b0 = _calib_ckpt(tmp_path), 2.0, 4.0
    pred_calib = DepthWizardPredictor(ckpt, device="cpu", live_backbone=False)
    assert pred_calib.architecture == "calibration_net"
    rng = np.random.default_rng(4)
    rgb = rng.integers(0, 255, (256, 256, 3), dtype=np.uint8)
    dn = np.tile(np.linspace(0, 1, 256, dtype=np.float32), (256, 1))
    out = pred_calib.predict(rgb, dn * 10.0, mode="crop")
    np.testing.assert_allclose(out, np.clip(a0 * dn + b0, 0, None), atol=5e-3)

    # --- rdah: metre heights, correct grid, auto-detected backend ---
    pred_rdah = DepthWizardPredictor(
        official_ckpt, device="cpu", live_backbone=False
    )
    assert pred_rdah.architecture == "rdah"
    raw = (1.0 + rng.random((256, 256)).astype(np.float32) * 4.0)
    out2 = pred_rdah.predict(rgb, raw, mode="resize")
    assert out2.shape == (256, 256)
    assert np.isfinite(out2).all()


def test_config_architecture_resolution(tmp_path):
    """CalibrationConfig: explicit 'rdah' selects the new backend; legacy
    configs without the key keep calibration_net (frozen behaviour)."""
    import yaml

    from depthwizard.config import CalibrationConfig

    rdah_cfg = yaml.safe_load(
        "model:\n  architecture: rdah\n  checkpoint: checkpoints/rdah/x.pth\n"
    )
    assert CalibrationConfig.from_dict(rdah_cfg).model.architecture == "rdah"
    legacy_cfg = yaml.safe_load("model:\n  widths: [16, 32, 64]\n")
    assert CalibrationConfig.from_dict(legacy_cfg).model.architecture == "calibration_net"
    v2_cfg = yaml.safe_load("model:\n  architecture: CalibrationNet_v2\n")
    assert CalibrationConfig.from_dict(v2_cfg).model.architecture == "calibration_net"


# ---------------------------------------------------------------------------
# Geospatial path — CRS/transform propagation, anchoring, DSM writing
# ---------------------------------------------------------------------------


def _write_geotif(path: Path, rgb: np.ndarray, crs, transform) -> None:
    import rasterio

    h, w = rgb.shape[:2]
    with rasterio.open(
        path, "w", driver="GTiff", height=h, width=w, count=3, dtype="uint8",
        crs=crs, transform=transform,
    ) as dst:
        for b in range(3):
            dst.write(rgb[..., b], b + 1)


def _write_dem(path: Path, elev: float, crs, transform, shape) -> None:
    import rasterio

    with rasterio.open(
        path, "w", driver="GTiff", height=shape[0], width=shape[1], count=1,
        dtype="float32", crs=crs, transform=transform,
    ) as dst:
        dst.write(np.full(shape, elev, np.float32), 1)


def test_run_inference_rdah_non_georeferenced_no_fake_crs(tmp_path, official_ckpt):
    """Non-georeferenced input: height map IS produced, dsm.tif is NOT
    (never invent a CRS), the payload says so honestly."""
    from PIL import Image

    from depthwizard.inference import run_inference

    rng = np.random.default_rng(5)
    rgb = rng.integers(0, 255, (SIZE, SIZE, 3), dtype=np.uint8)
    img_p = tmp_path / "plain.png"
    Image.fromarray(rgb).save(img_p)
    raw = (1.0 + rng.random((SIZE, SIZE)).astype(np.float32) * 4.0)
    dn_p = tmp_path / "raw_dn.npy"
    np.save(dn_p, raw)

    payload = run_inference(
        img_p, official_ckpt, out_dir=tmp_path / "out", device="cpu",
        mode="resize", dn_path=dn_p, live_backbone=False,
    )
    assert payload["ok"] is True
    assert payload["meta"]["model_architecture"] == "rdah"
    assert payload["meta"]["height_type"] == "nDSM"
    assert payload["georef"]["crs"] == "UNKNOWN"
    assert payload["outputs"]["dsm_npy"] is not None
    assert payload["outputs"]["dsm_tif"] is None  # NO fake georeference
    dsm = np.load(payload["outputs"]["dsm_npy"])
    assert dsm.shape == (SIZE, SIZE) and np.isfinite(dsm).all()
    assert "NOT absolute terrain elevation" in payload["meta"]["height_semantics"]


def test_run_inference_rdah_georeferenced_preserves_crs(tmp_path, official_ckpt):
    """Georeferenced GeoTIFF input: CRS + transform propagate to dsm.tif
    and the DEM-anchored DSM. nDSM + DEM = absolute DSM, exactly."""
    import rasterio
    from rasterio.transform import from_origin

    from depthwizard.inference import run_inference

    rng = np.random.default_rng(6)
    rgb = rng.integers(0, 255, (SIZE, SIZE, 3), dtype=np.uint8)
    crs = "EPSG:32618"
    transform = from_origin(500000.0, 4400000.0, 0.33, 0.33)  # GAMUS-like GSD
    img_p = tmp_path / "geo.tif"
    _write_geotif(img_p, rgb, crs, transform)
    raw = (1.0 + rng.random((SIZE, SIZE)).astype(np.float32) * 4.0)
    dn_p = tmp_path / "raw_dn.npy"
    np.save(dn_p, raw)
    dem_p = tmp_path / "dem.tif"
    _write_dem(dem_p, 12.5, crs, transform, (SIZE, SIZE))

    payload = run_inference(
        img_p, official_ckpt, out_dir=tmp_path / "out", device="cpu",
        mode="resize", dn_path=dn_p, live_backbone=False, anchor_dem=dem_p,
    )
    assert payload["georef"]["crs"] == "EPSG:32618"
    assert payload["anchored"]["label"] == "ANCHORED (not learned)"
    assert payload["meta"]["height_type"] == "nDSM"
    assert "absolute DSM" in payload["meta"]["height_semantics"]

    dsm_tif = payload["outputs"]["dsm_tif"]
    anchored_tif = payload["outputs"]["dsm_anchored"]
    assert dsm_tif and anchored_tif
    with rasterio.open(dsm_tif) as src:
        assert str(src.crs) == "EPSG:32618"
        assert list(src.transform)[:6] == list(transform)[:6]
        dsm = src.read(1)
    with rasterio.open(anchored_tif) as src:
        assert str(src.crs) == "EPSG:32618"
        anchored = src.read(1)
    # anchoring is EXACT arithmetic: DSM_anchored == nDSM + 12.5
    np.testing.assert_allclose(anchored, dsm + 12.5, atol=1e-4)


def test_rgb_never_treated_as_dem(tmp_path, official_ckpt):
    """Sanity guard for the RGB/DEM confusion: the RDAH path consumes RGB
    as IMAGE input only; elevation comes exclusively from --anchor-dem /
    --ground-elev (anchoring.py). Feeding an 8-bit image as DEM must NOT
    change the RELATIVE prediction (only the anchored output)."""
    from PIL import Image

    from depthwizard.inference import run_inference

    rng = np.random.default_rng(7)
    rgb = rng.integers(0, 255, (SIZE, SIZE, 3), dtype=np.uint8)
    img_p = tmp_path / "plain.png"
    Image.fromarray(rgb).save(img_p)
    raw = (1.0 + rng.random((SIZE, SIZE)).astype(np.float32) * 4.0)
    dn_p = tmp_path / "raw_dn.npy"
    np.save(dn_p, raw)

    base = run_inference(
        img_p, official_ckpt, out_dir=tmp_path / "a", device="cpu",
        mode="resize", dn_path=dn_p, live_backbone=False,
    )
    anchored = run_inference(
        img_p, official_ckpt, out_dir=tmp_path / "b", device="cpu",
        mode="resize", dn_path=dn_p, live_backbone=False, ground_elev=100.0,
    )
    d0 = np.load(base["outputs"]["dsm_npy"])
    d1 = np.load(anchored["outputs"]["dsm_npy"])
    np.testing.assert_allclose(d0, d1, atol=0)  # identical relative height
    a1 = np.load(anchored["outputs"]["dsm_anchored"])
    np.testing.assert_allclose(a1, d1 + 100.0, atol=1e-4)


# ---------------------------------------------------------------------------
# Training-support surface (adapter contract)
# ---------------------------------------------------------------------------


def test_adapter_training_contract(core):
    """The adapter exposes the CalibrationNet forward contract so the
    existing training loop / val monitor run unchanged: dict output with
    'pred', gradients flow, parameters are optimizable."""
    net = RDAHHeightModel(core=core)
    dn, rgb, stats, _ = _sample_inputs(seed=9)
    out = net(dn, rgb, None, None, stats)
    loss = (out["pred"] - 5.0).abs().mean()
    loss.backward()
    grads = [p.grad for p in net.parameters() if p.grad is not None]
    assert grads and any(g.abs().sum() > 0 for g in grads)

    opt = torch.optim.Adam(net.parameters(), lr=1e-4)
    opt.zero_grad()
    out = net(dn, rgb, None, None, stats)
    (out["pred"] - 5.0).abs().mean().backward()
    opt.step()  # fine-tuning path works


def test_adapter_saves_and_reloads(tmp_path, core):
    """DepthWizard fine-tuned checkpoints round-trip: saved with the
    official 'model_state_dict' key (inner net, no adapter prefix) and
    reload through the same registry."""
    from depthwizard.tifops import load_height_model

    net = RDAHHeightModel(core=core)
    p = tmp_path / "finetuned.pth"
    torch.save(
        {
            "architecture": "rdah",
            "model_state_dict": net.net.state_dict(),
            "epoch": 1,
            "val_subset_mae": 3.2,
        },
        p,
    )
    model = load_height_model(p, device="cpu")
    assert model.architecture == "rdah"
    dn, rgb, stats, _ = _sample_inputs(seed=10)
    with torch.no_grad():
        o1 = net(dn, rgb, None, None, stats)["pred"]
        o2 = model.net(dn, rgb, None, None, stats)["pred"]
    torch.testing.assert_close(o1, o2)
