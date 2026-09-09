"""Alignment + contract tests for DFC2019Dataset.

These tests exist because geospatial alignment bugs are silent: outputs LOOK
fine while being numerically wrong (blueprint risk register, Sec. 24).
Run:  pytest tests/ -q
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from depthwizard.dataset import DFC2019Config, DFC2019Dataset
from depthwizard.geo import TilePaths
from tools.make_fake_dataset import make_tile, write_tif


@pytest.fixture(scope="module")
def fake_root(tmp_path_factory):
    root = tmp_path_factory.mktemp("fake")
    rgb_dir = root / "rgb"; truth_dir = root / "truth"
    stems = []
    for i in range(1, 5):
        stem = f"JAX_{i:03d}_{i:03d}"
        rgb, agl, cls = make_tile(seed=i, size=256)
        write_tif(rgb_dir / f"{stem}_RGB.tif", rgb, 3, "uint8")
        write_tif(truth_dir / f"{stem}_AGL.tif", agl, 1, "float32")
        write_tif(truth_dir / f"{stem}_CLS.tif", cls, 1, "uint8")
        stems.append(stem)
    return root, rgb_dir, truth_dir, stems


def _tiles(rgb_dir, truth_dir, stems):
    return [TilePaths(stem=s,
                      rgb=rgb_dir / f"{s}_RGB.tif",
                      agl=truth_dir / f"{s}_AGL.tif",
                      cls=truth_dir / f"{s}_CLS.tif") for s in stems]


def _cfg(rgb_dir, truth_dir, **kw):
    d = dict(rgb_dir=rgb_dir, truth_dir=truth_dir, load_depth=False,
             crop_size=None, augment=False)
    if kw.get("depth_cache_dir") is not None:
        d["load_depth"] = True
    d.update(kw)
    return DFC2019Config(**d)


def test_sample_contract_shapes(fake_root):
    root, rgb_dir, truth_dir, stems = fake_root
    ds = DFC2019Dataset(_tiles(rgb_dir, truth_dir, stems), _cfg(rgb_dir, truth_dir))
    s = ds[0]
    assert s["rgb"].shape == (3, 256, 256) and s["rgb"].dtype == torch.float32
    assert s["agl"].shape == (1, 256, 256) and s["agl"].dtype == torch.float32
    assert s["cls"].shape == (1, 256, 256) and s["cls"].dtype == torch.int64
    assert s["dn"] is None
    assert s["meta"]["stem"] == stems[0]


def test_agl_clamped_nonnegative(fake_root):
    root, rgb_dir, truth_dir, stems = fake_root
    ds = DFC2019Dataset(_tiles(rgb_dir, truth_dir, stems), _cfg(rgb_dir, truth_dir))
    for i in range(len(ds)):
        assert float(ds[i]["agl"].min()) >= 0.0


def test_crop_alignment_across_layers(fake_root):
    """The crop applied to rgb MUST equal the crop applied to agl/cls."""
    root, rgb_dir, truth_dir, stems = fake_root
    cfg = _cfg(rgb_dir, truth_dir, crop_size=64, augment=True, seed=7)
    ds = DFC2019Dataset(_tiles(rgb_dir, truth_dir, stems), cfg)
    s = ds[2]
    h = w = 64
    assert s["rgb"].shape == (3, h, w) and s["agl"].shape == (1, h, w)

    # rebuild the same window from raw data and compare against un-normalized rgb
    from depthwizard.geo import read_tile
    raw = read_tile(ds.tiles[2])
    y0, x0 = s["meta"]["y0"], s["meta"]["x0"]
    ref_rgb = raw["rgb"][y0:y0 + h, x0:x0 + w, :]
    k, fh, fv = s["meta"]["rot90"], s["meta"]["flip_h"], s["meta"]["flip_v"]
    ref = np.rot90(ref_rgb, k=k)
    if fh:
        ref = np.flip(ref, axis=1)
    if fv:
        ref = np.flip(ref, axis=0)
    ref = np.ascontiguousarray(ref).astype(np.float32) / 255.0
    got = s["rgb"].permute(1, 2, 0).numpy()
    # undo ImageNet normalization for comparison
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    got = got * std + mean
    assert np.allclose(got, ref, atol=1e-4), "rgb crop/aug mismatch — alignment broken"


def test_determinism_same_seed(fake_root):
    root, rgb_dir, truth_dir, stems = fake_root
    cfg = _cfg(rgb_dir, truth_dir, crop_size=64, augment=True, seed=123)
    ds_a = DFC2019Dataset(_tiles(rgb_dir, truth_dir, stems), cfg)
    ds_b = DFC2019Dataset(_tiles(rgb_dir, truth_dir, stems),
                          _cfg(rgb_dir, truth_dir, crop_size=64, augment=True, seed=123))
    a, b = ds_a[1], ds_b[1]
    assert torch.equal(a["agl"], b["agl"])
    assert a["meta"] == b["meta"]


def test_depth_cache_contract(fake_root, tmp_path):
    root, rgb_dir, truth_dir, stems = fake_root
    cache = tmp_path / "cache"
    cache.mkdir()
    rng = np.random.default_rng(0)
    for s in stems:
        np.save(cache / f"{s}.npy", rng.random((256, 256)).astype(np.float32))
    ds = DFC2019Dataset(_tiles(rgb_dir, truth_dir, stems),
                        _cfg(rgb_dir, truth_dir, depth_cache_dir=cache))
    s = ds[0]
    assert s["dn"] is not None and s["dn"].shape == (1, 256, 256)
    assert 0.0 <= float(s["dn"].min()) and float(s["dn"].max()) <= 1.0


def test_depth_cache_shape_mismatch_raises(fake_root, tmp_path):
    root, rgb_dir, truth_dir, stems = fake_root
    cache = tmp_path / "cache2"
    cache.mkdir()
    np.save(cache / f"{stems[0]}.npy", np.zeros((128, 128), np.float32))
    ds = DFC2019Dataset(_tiles(rgb_dir, truth_dir, stems),
                        _cfg(rgb_dir, truth_dir, depth_cache_dir=cache))
    with pytest.raises(ValueError):
        _ = ds[0]


# ---------------------------------------------------------------------------
# DEM prior (Method-D ablation) — joint-transform alignment + tag honesty
# ---------------------------------------------------------------------------

def test_crop_alignment_across_layers_with_dem(fake_root):
    """Method-D ablation alignment pin: the DEM layer MUST be cropped/flipped
    by the SAME window and transform as rgb/agl/cls/dn. Drift here would
    feed the calibration net a DEM that disagrees with the AGL it's
    predicting — a silent-wrong-number bug (worklog Section 4 class).

    Uses the synthetic-DEM fallback so the test doesn't depend on a real
    DEM cache. The sample's meta.dem_tag MUST be the literal
    SYNTHETIC-DEM-PROXY string (the contract every output / worklog line
    MUST carry in this case).
    """
    root, rgb_dir, truth_dir, stems = fake_root
    cfg = _cfg(rgb_dir, truth_dir, crop_size=64, augment=True, seed=7,
               synth_dem_fallback=True, synth_dem_sigma_m=4.0)
    ds = DFC2019Dataset(_tiles(rgb_dir, truth_dir, stems), cfg)
    s = ds[2]
    h = w = 64
    assert s["rgb"].shape == (3, h, w) and s["agl"].shape == (1, h, w)
    # DEM contract: present, float32, [1,h,w], tag = SYNTHETIC-DEM-PROXY
    assert s["dem"] is not None
    assert s["dem"].shape == (1, h, w)
    assert s["dem"].dtype == torch.float32
    assert s["meta"]["dem_tag"] == "SYNTHETIC-DEM-PROXY"

    # Rebuild the DEM from the FULL AGL the SAME way the dataset does
    # (synth BEFORE crop/aug), then apply the same window+transform. The
    # Gaussian conv uses reflect padding, which is NOT shift-invariant —
    # so deriving the DEM from the *cropped* AGL would give a different
    # boundary result; matching the dataset's order of operations is the
    # alignment pin.
    from depthwizard.geo import read_tile
    from depthwizard.demprior import synth_dem_from_agl
    raw = read_tile(ds.tiles[2])
    ref_full_dem = synth_dem_from_agl(raw["agl"],
                                      sigma_m=cfg.synth_dem_sigma_m)
    y0, x0 = s["meta"]["y0"], s["meta"]["x0"]
    ref_dem_crop = ref_full_dem[y0:y0 + h, x0:x0 + w]
    k, fh, fv = s["meta"]["rot90"], s["meta"]["flip_h"], s["meta"]["flip_v"]
    ref_dem_crop = np.rot90(ref_dem_crop, k=k)
    if fh:
        ref_dem_crop = np.flip(ref_dem_crop, axis=1)
    if fv:
        ref_dem_crop = np.flip(ref_dem_crop, axis=0)
    ref_dem_crop = np.ascontiguousarray(ref_dem_crop).astype(np.float32)
    got_dem = s["dem"][0].numpy()
    assert np.allclose(got_dem, ref_dem_crop, atol=1e-5), \
        "DEM crop/aug mismatch — Method-D alignment broken"


def test_dataset_dem_layer_off_by_default(fake_root):
    """When dem_dir is None AND synth_dem_fallback is False, the dataset
    produces NO dem layer and meta.dem_tag is None — i.e. the pre-Method-D
    behavior is preserved exactly. No silent synthesis."""
    root, rgb_dir, truth_dir, stems = fake_root
    cfg = _cfg(rgb_dir, truth_dir, crop_size=64, augment=True, seed=7)
    ds = DFC2019Dataset(_tiles(rgb_dir, truth_dir, stems), cfg)
    s = ds[0]
    assert s["dem"] is None
    assert s["meta"]["dem_tag"] is None


def test_dataset_real_dem_layer_loads_from_disk(fake_root, tmp_path):
    """When dem_dir is set, the dataset loads {stem}.tif single-band DEMs
    from it and tags the sample 'dem:<filename>'. CRS-alignment with the
    (CRS-less, Track-1) tile is OUT OF SCOPE here — the worklog Section 4
    honesty rule says a Track-1 tile cannot be honestly aligned to a real
    DEM; that's a separate data-dependency. This test only pins the
    cache-lookup + shape + tag contract."""
    root, rgb_dir, truth_dir, stems = fake_root
    dem_dir = tmp_path / "dem"
    dem_dir.mkdir()
    from tools.make_fake_dataset import write_tif
    rng = np.random.default_rng(99)
    for s in stems:
        dem = rng.random((256, 256)).astype(np.float32) * 50.0
        write_tif(dem_dir / f"{s}.tif", dem, 1, "float32")
    cfg = _cfg(rgb_dir, truth_dir, dem_dir=dem_dir)
    ds = DFC2019Dataset(_tiles(rgb_dir, truth_dir, stems), cfg)
    s = ds[0]
    assert s["dem"] is not None and s["dem"].shape == (1, 256, 256)
    assert s["meta"]["dem_tag"].startswith("dem:")


def test_dataset_real_dem_shape_mismatch_raises(fake_root, tmp_path):
    """DEM cache shape != tile grid shape -> ValueError (no silent broadcast)."""
    root, rgb_dir, truth_dir, stems = fake_root
    dem_dir = tmp_path / "dem_mismatch"
    dem_dir.mkdir()
    from tools.make_fake_dataset import write_tif
    write_tif(dem_dir / f"{stems[0]}.tif",
              np.zeros((128, 128), np.float32), 1, "float32")
    cfg = _cfg(rgb_dir, truth_dir, dem_dir=dem_dir)
    ds = DFC2019Dataset(_tiles(rgb_dir, truth_dir, stems), cfg)
    with pytest.raises(ValueError, match="out of sync"):
        _ = ds[0]


def test_dataset_real_dem_missing_raises(fake_root, tmp_path):
    """dem_dir set but the {stem}.tif file is absent -> FileNotFoundError
    (no silent fall-through to the synthetic path)."""
    root, rgb_dir, truth_dir, stems = fake_root
    dem_dir = tmp_path / "dem_missing"
    dem_dir.mkdir()
    cfg = _cfg(rgb_dir, truth_dir, dem_dir=dem_dir)
    ds = DFC2019Dataset(_tiles(rgb_dir, truth_dir, stems), cfg)
    with pytest.raises(FileNotFoundError, match="DEM cache miss"):
        _ = ds[0]
