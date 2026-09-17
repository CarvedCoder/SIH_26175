"""Tests for depthwizard.imgio — the input preprocessing path.

Pins:
  * every PNG/JPG mode (RGB / grayscale / alpha / palette / 16-bit /
    CMYK) decodes to the SAME contract: uint8 [H,W,3], values [0,255],
    C-contiguous;
  * palette PNGs expand to palette COLORS (never raw indices);
  * 16-bit / float data is range-checked — the old naive
    ``astype(np.uint8)`` wrapped modulo 256 (300 -> 44) and is gone;
  * NaN/Inf rasters and corrupted files fail LOUDLY (InvalidImageError);
  * GeoTIFF keeps CRS/transform; PNG/JPG get None (never invented);
  * Depth Anything V2 and CalibrationNet each apply /255 + ImageNet
    normalization EXACTLY ONCE (captured at the model boundary).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from depthwizard.imgio import (
    InvalidImageError,
    preprocess_input_image,
    to_uint8,
    validate_rgb_u8,
)
from depthwizard.inference import read_image

# ---------------------------------------------------------------------------
# Synthetic image builders (Pillow + rasterio, no network)
# ---------------------------------------------------------------------------


def _gradient_rgb(h: int = 32, w: int = 48) -> np.ndarray:
    a = np.zeros((h, w, 3), dtype=np.uint8)
    a[..., 0] = np.linspace(0, 220, w, dtype=np.uint8)[None, :]
    a[..., 1] = 100
    a[..., 2] = 40
    a[0, 0] = (255, 0, 0)  # unique sentinel pixel for order checks
    return a


def _write_png(path: Path, arr: np.ndarray) -> Path:
    from PIL import Image

    Image.fromarray(arr).save(path)
    return path


def _assert_contract(rgb: np.ndarray) -> None:
    """The ONE contract every loader must satisfy."""
    assert isinstance(rgb, np.ndarray)
    assert rgb.dtype == np.uint8
    assert rgb.ndim == 3 and rgb.shape[2] == 3
    assert rgb.shape[0] >= 1 and rgb.shape[1] >= 1
    assert int(rgb.min()) >= 0 and int(rgb.max()) <= 255
    assert np.isfinite(rgb.astype(np.float64)).all()
    assert rgb.flags["C_CONTIGUOUS"]


# ---------------------------------------------------------------------------
# 1-4: plain RGB + grayscale (PNG lossless-exact, JPG structural)
# ---------------------------------------------------------------------------


def test_read_rgb_png_exact(tmp_path):
    src = _gradient_rgb()
    p = _write_png(tmp_path / "rgb.png", src)
    rgb, meta = preprocess_input_image(p)
    _assert_contract(rgb)
    np.testing.assert_array_equal(rgb, src)  # PNG is lossless
    assert meta["format"] == "PNG"
    assert meta["pil_mode"] == "RGB"
    assert meta["georeferenced"] is False
    assert meta["crs"] is None and meta["transform"] is None


def test_read_rgb_jpg(tmp_path):
    from PIL import Image

    src = _gradient_rgb()
    p = tmp_path / "rgb.jpg"
    Image.fromarray(src, "RGB").save(p, quality=95)
    rgb, meta = preprocess_input_image(p)
    _assert_contract(rgb)
    assert meta["format"] == "JPEG"
    # lossy but close: high per-pixel agreement on the VARYING channel
    # (the other channels are constant — correlation is meaningless there)
    assert np.corrcoef(rgb[..., 0].ravel(), src[..., 0].ravel())[0, 1] > 0.99


def test_read_grayscale_png_replicated(tmp_path):
    g = np.linspace(0, 255, 32 * 48, dtype=np.uint8).reshape(32, 48)
    p = _write_png(tmp_path / "gray.png", g)
    rgb, meta = preprocess_input_image(p)
    _assert_contract(rgb)
    np.testing.assert_array_equal(rgb[..., 0], g)
    np.testing.assert_array_equal(rgb[..., 1], g)  # replicated, not dropped
    np.testing.assert_array_equal(rgb[..., 2], g)
    assert meta["pil_mode"] == "L"


def test_read_grayscale_jpg_replicated(tmp_path):
    from PIL import Image

    g = np.linspace(0, 255, 16 * 24, dtype=np.uint8).reshape(16, 24)
    p = tmp_path / "gray.jpg"
    Image.fromarray(g, "L").save(p, quality=95)
    rgb, _meta = preprocess_input_image(p)
    _assert_contract(rgb)
    assert np.corrcoef(rgb[..., 0].ravel(), g.ravel())[0, 1] > 0.99
    np.testing.assert_array_equal(rgb[..., 0], rgb[..., 2])


# ---------------------------------------------------------------------------
# 5-6: alpha + palette policy
# ---------------------------------------------------------------------------


def test_read_rgba_png_drops_alpha_keeps_rgb(tmp_path):
    src = _gradient_rgb()
    alpha = np.full(src.shape[:2], 255, dtype=np.uint8)
    alpha[-4:, -4:] = 0  # fully transparent corner
    rgba = np.dstack([src, alpha])
    p = _write_png(tmp_path / "rgba.png", rgba)
    rgb, meta = preprocess_input_image(p)
    _assert_contract(rgb)
    np.testing.assert_array_equal(rgb, src)  # RGB preserved verbatim
    assert meta["pil_mode"] == "RGBA"


def test_read_palette_png_expands_colors_not_indices(tmp_path):
    from PIL import Image

    # indices into a REAL color palette (built from a color gradient, so
    # palette colors are distinguishable from index values)
    idx = (np.linspace(0, 255, 64, dtype=np.uint8)).reshape(8, 8)
    pal = Image.fromarray(idx).convert("RGB")  # grayscale gradient first...
    pal = pal.convert("P", palette=Image.Palette.ADAPTIVE, colors=32)
    p = tmp_path / "palette.png"
    pal.save(p)

    with Image.open(p) as im:  # ground truth: PIL's own palette expansion
        expected = np.asarray(im.convert("RGB"))
        raw_indices = np.asarray(im)

    rgb, meta = preprocess_input_image(p)
    _assert_contract(rgb)
    np.testing.assert_array_equal(rgb, expected)  # COLORS, not indices
    # guard the test itself: indices alone must NOT reproduce the colors
    assert not np.array_equal(
        np.repeat(raw_indices[..., None], 3, axis=2), expected
    )
    assert meta["pil_mode"] == "P"


def test_read_palette_png_with_transparency(tmp_path):
    from PIL import Image

    idx = np.zeros((8, 8), dtype=np.uint8)
    idx[0, 0] = 255
    pal = Image.fromarray(idx).convert("P", palette=Image.Palette.ADAPTIVE)
    pal.save(tmp_path / "p_t.png", transparency=0)
    rgb, meta = preprocess_input_image(tmp_path / "p_t.png")
    _assert_contract(rgb)
    assert meta["pil_mode"] == "P"


# ---------------------------------------------------------------------------
# 7-8: corrupted / unsupported / missing inputs fail LOUDLY
# ---------------------------------------------------------------------------


def test_corrupted_png_raises_invalid_image(tmp_path):
    p = tmp_path / "corrupt.png"
    p.write_bytes(b"\x89PNG\r\n\x1a\nGARBAGE-NOT-A-PNG")
    with pytest.raises(InvalidImageError, match="corrupt.png"):
        preprocess_input_image(p)


def test_text_with_image_extension_raises(tmp_path):
    p = tmp_path / "text.png"
    p.write_bytes(b"hello, not an image at all")
    with pytest.raises(InvalidImageError):
        preprocess_input_image(p)


def test_unknown_extension_via_rasterio_wraps_cleanly(tmp_path):
    p = tmp_path / "junk.txt"
    p.write_bytes(b"definitely not a raster")
    with pytest.raises(InvalidImageError, match="junk.txt"):
        preprocess_input_image(p)


def test_missing_file_raises_filenotfound(tmp_path):
    with pytest.raises(FileNotFoundError):
        preprocess_input_image(tmp_path / "nope.png")


def test_truncated_png_raises(tmp_path):
    from PIL import Image

    p = tmp_path / "trunc.png"
    Image.fromarray(_gradient_rgb()).save(p)
    data = p.read_bytes()
    p.write_bytes(data[: len(data) // 3])  # keep only the header
    with pytest.raises(InvalidImageError):
        preprocess_input_image(p)


# ---------------------------------------------------------------------------
# 9-10: tiny + normal images
# ---------------------------------------------------------------------------


def test_very_small_images(tmp_path):
    one = np.zeros((1, 1, 3), dtype=np.uint8)
    rgb, _ = preprocess_input_image(_write_png(tmp_path / "one.png", one))
    _assert_contract(rgb)
    assert rgb.shape == (1, 1, 3)

    tiny = _gradient_rgb(3, 5)
    rgb, meta = preprocess_input_image(_write_png(tmp_path / "tiny.png", tiny))
    _assert_contract(rgb)
    assert rgb.shape == (3, 5, 3)
    assert meta["original_size"] == [5, 3]
    assert meta["processed_size"] == [5, 3]


def test_normal_aerial_size_contract_and_metadata(tmp_path):
    rng = np.random.default_rng(11)
    src = rng.integers(0, 255, (256, 256, 3), dtype=np.uint8)
    rgb, meta = preprocess_input_image(_write_png(tmp_path / "aerial.png", src))
    _assert_contract(rgb)
    assert rgb.shape == (256, 256, 3)
    for key in (
        "source",
        "format",
        "pil_mode",
        "original_size",
        "processed_size",
        "channels",
        "dtype",
        "georeferenced",
    ):
        assert key in meta
    assert meta["channels"] == 3
    assert meta["dtype"] == "uint8"
    assert meta["original_size"] == [256, 256]


# ---------------------------------------------------------------------------
# dtype policy: range-checked, NEVER modulo-wrapped
# ---------------------------------------------------------------------------


def test_16bit_grayscale_png_no_modulo_wrap(tmp_path):
    g16 = np.array([[300, 511, 60000, 0]], dtype=np.uint16)
    p = tmp_path / "g16.png"
    from PIL import Image

    Image.fromarray(g16).save(p)  # uint16 2-D -> PIL mode "I;16"
    rgb, meta = preprocess_input_image(p)
    _assert_contract(rgb)
    row = rgb[0]  # [4, 3] after gray replication
    assert row.shape == (4, 3)
    # monotonic and NOT value % 256 (the old bug: 300 -> 44, 60000 -> 96)
    assert not np.array_equal(row[:, 0], g16[0] % 256)
    assert row[0, 0] < row[1, 0] < row[2, 0]  # order preserved
    assert row[2, 0] == 255 and row[3, 0] == 0  # full range used
    assert meta["pil_mode"] == "I;16"


def test_to_uint8_uint16_in_byte_range_is_bit_exact():
    # 8-bit content stored in uint16: direct cast, NO rescale (bit-exact)
    out = to_uint8(np.array([[0, 128, 255]], dtype=np.uint16), "t")
    np.testing.assert_array_equal(out, [[0, 128, 255]])


def test_to_uint8_uint16_rescale_is_monotonic_full_range():
    a = np.array([0, 100, 2047, 4095], dtype=np.uint16)  # 12-bit-in-16
    out = to_uint8(a, "t")
    assert out.max() == 255
    assert np.all(np.diff(out) > 0)  # strictly monotonic, no wrap
    # linear by the observed max, not by 65535
    assert out[1] == round(100 * 255 / 4095)


def test_to_uint8_float01_scaled_float_range_checked():
    f = np.array([0.0, 0.5, 1.0], dtype=np.float32)
    np.testing.assert_array_equal(to_uint8(f, "t"), [0, 128, 255])
    f8 = np.array([0.0, 127.5, 255.0], dtype=np.float32)
    np.testing.assert_array_equal(to_uint8(f8, "t"), [0, 128, 255])
    with pytest.raises(InvalidImageError, match="not an image range"):
        to_uint8(np.array([[-50.0, 300.0]], dtype=np.float32), "t")


def test_to_uint8_rejects_nan_inf_and_negative_ints():
    with pytest.raises(InvalidImageError, match="non-finite"):
        to_uint8(np.array([[1.0, np.nan]], dtype=np.float32), "t")
    with pytest.raises(InvalidImageError, match="non-finite"):
        to_uint8(np.array([[1.0, np.inf]], dtype=np.float32), "t")
    with pytest.raises(InvalidImageError, match="negative"):
        to_uint8(np.array([[-3, 7]], dtype=np.int16), "t")


def test_uint16_geotiff_rescaled_not_wrapped(tmp_path):
    import rasterio

    data = np.array([3779, 256, 257], dtype=np.uint16)[:, None, None]  # [3,1,1]
    p = tmp_path / "u16.tif"
    with rasterio.open(
        p, "w", driver="GTiff", height=1, width=1, count=3, dtype="uint16"
    ) as dst:
        dst.write(data)
    rgb, meta = preprocess_input_image(p)
    _assert_contract(rgb)
    pixel = rgb[0, 0]
    assert not np.array_equal(pixel, data[:, 0, 0] % 256)  # old wrap bug
    assert pixel[0] == 255 and pixel[1] == pixel[2]  # monotonic rescale
    assert meta["original_dtype"] == "uint16"


def test_float32_geotiff_unit_interval_scaled(tmp_path):
    import rasterio

    data = np.linspace(0.0, 1.0, 3 * 8 * 8, dtype=np.float32).reshape(3, 8, 8)
    p = tmp_path / "f01.tif"
    with rasterio.open(
        p, "w", driver="GTiff", height=8, width=8, count=3, dtype="float32"
    ) as dst:
        dst.write(data)
    rgb, _ = preprocess_input_image(p)
    _assert_contract(rgb)
    assert rgb.max() == 255 and rgb.min() == 0


def test_float32_geotiff_nan_inf_raises_loudly(tmp_path):
    import rasterio

    data = np.full((3, 8, 8), 128.0, dtype=np.float32)
    data[0, 0, 0] = np.nan
    data[1, 0, 0] = np.inf
    p = tmp_path / "fnan.tif"
    with rasterio.open(
        p, "w", driver="GTiff", height=8, width=8, count=3, dtype="float32"
    ) as dst:
        dst.write(data)
    with pytest.raises(InvalidImageError, match="non-finite"):
        preprocess_input_image(p)


# ---------------------------------------------------------------------------
# GeoTIFF path: georeferencing preserved; PNG path: never invented
# ---------------------------------------------------------------------------


def test_geotiff_preserves_crs_and_pixels(tmp_path):
    import rasterio
    from rasterio.transform import from_origin

    src = _gradient_rgb()
    p = tmp_path / "geo.tif"
    tf = from_origin(500000, 5600000, 1.0, 1.0)
    with rasterio.open(
        p,
        "w",
        driver="GTiff",
        height=src.shape[0],
        width=src.shape[1],
        count=3,
        dtype="uint8",
        crs="EPSG:32632",
        transform=tf,
    ) as dst:
        dst.write(src.transpose(2, 0, 1))

    rgb, meta = preprocess_input_image(p)
    _assert_contract(rgb)
    np.testing.assert_array_equal(rgb, src)
    assert meta["georeferenced"] is True
    assert meta["crs"] is not None and "32632" in str(meta["crs"])
    assert meta["format"] == "GTiff"

    # read_image backward-compat surface (used by run_inference/anchoring)
    rgb2, profile = read_image(p)
    assert profile["_crs_obj"] is not None
    assert profile["_transform_obj"] == tf
    np.testing.assert_array_equal(rgb2, src)


def test_png_never_gets_fake_georeferencing(tmp_path):
    rgb, _ = preprocess_input_image(_write_png(tmp_path / "p.png", _gradient_rgb()))
    _assert_contract(rgb)
    _, profile = read_image(tmp_path / "p.png")
    assert profile["_crs_obj"] is None
    assert profile["_transform_obj"] is None


def test_gray_geotiff_single_band_replicated(tmp_path):
    import rasterio

    g = np.linspace(0, 255, 8 * 8, dtype=np.uint8).reshape(8, 8)
    p = tmp_path / "g8.tif"
    with rasterio.open(
        p, "w", driver="GTiff", height=8, width=8, count=1, dtype="uint8"
    ) as dst:
        dst.write(g[None])
    rgb, _ = preprocess_input_image(p)
    np.testing.assert_array_equal(rgb[..., 0], g)
    np.testing.assert_array_equal(rgb[..., 1], g)


# ---------------------------------------------------------------------------
# validate_rgb_u8 contract rejections
# ---------------------------------------------------------------------------


def test_validate_rejects_bad_inputs():
    with pytest.raises(InvalidImageError):
        validate_rgb_u8(np.zeros((8, 8), dtype=np.uint8))  # 2-D grayscale
    with pytest.raises(InvalidImageError):
        validate_rgb_u8(np.zeros((8, 8, 4), dtype=np.uint8))  # RGBA leaked
    with pytest.raises(InvalidImageError):
        validate_rgb_u8(np.zeros((8, 8, 3), dtype=np.float32))  # unnormalized
    with pytest.raises(InvalidImageError):
        validate_rgb_u8(np.zeros((0, 8, 3), dtype=np.uint8))  # empty
    validate_rgb_u8(np.zeros((8, 8, 3), dtype=np.uint8))  # OK: no raise


# ---------------------------------------------------------------------------
# 11: large image -> tile inference (preprocessing feeds EVERY tile)
# ---------------------------------------------------------------------------


def _make_rgb_checkpoint(tmp_path, a0=2.0, b0=4.0):
    """use_rgb=True checkpoint (in_ch=4) — the flagship configuration."""
    torch = pytest.importorskip("torch")
    from depthwizard.calibration_net import CalibrationNet, derive_in_ch

    net = CalibrationNet(
        in_ch=derive_in_ch(use_rgb=True, use_sem=False, use_dem=False),
        widths=(8, 16, 32),
        a0=a0,
        b0=b0,
    )
    ckpt = {
        "model_state": net.state_dict(),
        "use_rgb": True, "widths": [8, 16, 32], "clamp_min": 0.0,
        "affine_init": {"a": a0, "b": b0}, "loss": "l1", "epoch": 0,
        "val_subset_mae": 0.0, "splits_json": "n/a",
    }
    p = tmp_path / "best_rgb.pt"
    torch.save(ckpt, p)
    return p


def test_large_png_enters_tile_inference_with_contract(tmp_path):
    """2100x1100 PNG: read_image -> predict(tiles) must feed six uint8
    1024x1024 RGB tiles with per-tile-normalized Dn — RGB preprocessing is
    done ONCE up front, never per tile with a different distribution."""
    torch = pytest.importorskip("torch")
    from depthwizard.inference import DepthWizardPredictor, TILE

    h, w = 2100, 1100  # 3 x 2 tiles
    rng = np.random.default_rng(5)
    src = rng.integers(0, 255, (h, w, 3), dtype=np.uint8)
    rgb_u8, _ = read_image(_write_png(tmp_path / "big.png", src))
    _assert_contract(rgb_u8)
    assert rgb_u8.shape == (h, w, 3)

    ckpt = _make_rgb_checkpoint(tmp_path)
    pred = DepthWizardPredictor(ckpt, device="cpu", live_backbone=False)

    seen: list[tuple[np.ndarray, np.ndarray]] = []

    def capture_predict(dn, rgb=None):
        seen.append((dn.copy(), None if rgb is None else rgb.copy()))
        return np.zeros(dn.shape, dtype=np.float32)

    pred._predict_fn = capture_predict
    raw_dn = np.linspace(0.0, 10.0, h * w, dtype=np.float32).reshape(h, w)
    out = pred.predict(rgb_u8, raw_dn, mode="tiles")
    assert out.shape == (h, w)

    ny, nx = -(-h // TILE), -(-w // TILE)
    assert len(seen) == ny * nx == 6
    for dn, rgb_tile in seen:
        assert rgb_tile.dtype == np.uint8  # model sees RAW uint8, ONCE
        assert rgb_tile.shape == (TILE, TILE, 3)
        assert dn.min() == pytest.approx(0.0, abs=1e-6)
        assert dn.max() == pytest.approx(1.0, abs=1e-6)  # per-tile Dn only
    assert torch.is_tensor  # keeps importorskip honest


# ---------------------------------------------------------------------------
# Normalization applied EXACTLY ONCE at each model boundary
# ---------------------------------------------------------------------------


def test_depth_anything_preprocess_normalizes_exactly_once():
    """Captures the pixel_values tensor handed to the HF model: with a
    constant-color input the bicubic resize is exact, so the tensor must
    equal ONE application of (x/255 - mean)/std — pre-normalized input
    would land far from these values."""
    torch = pytest.importorskip("torch")
    from depthwizard.backbone import DepthAnythingBackbone

    bb = DepthAnythingBackbone(model_id="fake", device="cpu")
    captured: list = []

    class _Out:
        predicted_depth = torch.rand(1, 13, 13)

    class _FakeHF:
        def to(self, device):
            return self

        def eval(self):
            return self

        def __call__(self, pixel_values=None, **kw):
            captured.append(pixel_values.clone())
            return _Out()

    bb._model = _FakeHF()
    bb._torch = torch

    rgb = np.full((64, 64, 3), (200, 100, 50), dtype=np.uint8)
    bb.raw_depth(rgb)
    assert len(captured) == 1

    x = captured[0]
    assert x.shape == (1, 3, 518, 518) and x.dtype == torch.float32
    for c, (v, m, s) in enumerate(
        zip((200, 100, 50), (0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ):
        expected = (v / 255.0 - m) / s
        assert torch.allclose(x[0, c], torch.tensor(expected), atol=1e-6)


def test_calibrationnet_rgb_normalized_exactly_once(tmp_path):
    """make_predict_fn must apply /255 + ImageNet exactly once — captured
    rgb_t is compared against the formula byte-for-byte."""
    torch = pytest.importorskip("torch")
    from depthwizard.dataset import IMAGENET_MEAN, IMAGENET_STD
    from depthwizard.tifops import LoadedModel, make_predict_fn

    captured: list = []

    class _FakeNet:
        def eval(self):
            return self

        def __call__(self, dn_t, rgb_t):
            captured.append(rgb_t.clone())
            return {
                "pred": torch.zeros((1, 1, dn_t.shape[-2], dn_t.shape[-1]))
            }

    model = LoadedModel(
        net=_FakeNet(),
        use_rgb=True,
        widths=(8, 16, 32),
        clamp_min=0.0,
        affine_init={"a": 2.0, "b": 4.0},
        epoch=0,
        checkpoint=Path("fake.pt"),
        val_subset_mae=None,
    )
    predict = make_predict_fn(model, device="cpu")

    rng = np.random.default_rng(3)
    rgb = rng.integers(0, 255, (16, 16, 3), dtype=np.uint8)
    dn = rng.random((16, 16)).astype(np.float32)
    predict(dn, rgb)

    assert len(captured) == 1
    expected = ((rgb.astype(np.float32) / 255.0) - IMAGENET_MEAN) / IMAGENET_STD
    expected_t = torch.from_numpy(
        np.ascontiguousarray(expected.transpose(2, 0, 1))[None]
    )
    assert torch.equal(captured[0], expected_t)  # EXACTLY once, exact math


# ---------------------------------------------------------------------------
# End-to-end (no network): explicit Dn + real checkpoint -> full outputs
# ---------------------------------------------------------------------------


def test_run_inference_png_end_to_end_no_backbone(tmp_path):
    """PNG (not georeferenced) -> dsm.npy + preview, NO dsm.tif, and the
    payload keeps the honest georef contract."""
    pytest.importorskip("torch")
    from model_tests.test_inference import _make_checkpoint
    from depthwizard.inference import run_inference

    ckpt, a0, b0 = _make_checkpoint(tmp_path)
    h, w = 256, 384
    rgb = _gradient_rgb(h, w)
    p = _write_png(tmp_path / "scene.png", rgb)

    dn = np.tile(np.linspace(0.0, 1.0, w, dtype=np.float32), (h, 1))
    dn_path = tmp_path / "scene_dn.npy"
    np.save(dn_path, dn * 10.0)  # RAW (unnormalized), as resolve_dn expects

    payload = run_inference(
        p,
        ckpt,
        out_dir=tmp_path / "out",
        device="cpu",
        mode="resize",
        dn_path=dn_path,
        live_backbone=False,
    )
    assert payload["ok"] is True
    assert payload["georef"]["crs"] == "UNKNOWN"
    assert payload["meta"]["pixel_size_m"] is None
    assert (tmp_path / "out" / "dsm.npy").exists()
    assert (tmp_path / "out" / "dsm_preview.png").exists()
    assert not (tmp_path / "out" / "dsm.tif").exists()  # no invented CRS

    dsm = np.load(tmp_path / "out" / "dsm.npy")
    assert dsm.shape == (h, w)
    expected = np.clip(a0 * dn + b0, 0, None)
    np.testing.assert_allclose(dsm, expected, atol=2e-2)


def test_run_inference_geotiff_end_to_end_writes_tif(tmp_path):
    """GeoTIFF keeps CRS/transform through the WHOLE pipeline."""
    pytest.importorskip("torch")
    import rasterio
    from rasterio.transform import from_origin
    from model_tests.test_inference import _make_checkpoint
    from depthwizard.inference import run_inference

    ckpt, a0, b0 = _make_checkpoint(tmp_path)
    h, w = 256, 256
    src = _gradient_rgb(h, w)
    p = tmp_path / "scene.tif"
    tf = from_origin(500000, 5600000, 1.0, 1.0)
    with rasterio.open(
        p, "w", driver="GTiff", height=h, width=w, count=3, dtype="uint8",
        crs="EPSG:32632", transform=tf,
    ) as dst:
        dst.write(src.transpose(2, 0, 1))

    dn = np.tile(np.linspace(0.0, 1.0, w, dtype=np.float32), (h, 1))
    dn_path = tmp_path / "scene_dn.npy"
    np.save(dn_path, dn * 10.0)

    payload = run_inference(
        p,
        ckpt,
        out_dir=tmp_path / "out",
        device="cpu",
        mode="crop",
        dn_path=dn_path,
        live_backbone=False,
    )
    assert payload["georef"]["crs"].startswith("EPSG")
    out_tif = tmp_path / "out" / "dsm.tif"
    assert out_tif.exists()
    with rasterio.open(out_tif) as ds:
        assert ds.crs is not None and "32632" in str(ds.crs)
        assert ds.transform == tf  # preserved end-to-end
