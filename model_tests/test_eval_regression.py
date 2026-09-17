"""Regression pins for the evaluate path after the GAMUS integration.

The user constraint "Preserve existing DFC2019 Exp 1/2/3 behavior" is
enforced HERE by construction:

  * evaluate_split (modified: namespaced cache read + Phase-5 extensions)
    must produce pooled metrics BIT-IDENTICAL to the ORIGINAL pre-change
    logic (reimplemented verbatim below as _reference_frozen_eval) on the
    same fixtures;
  * the same numbers must arise from the NAMESPACED cache layout (the
    cache-path fallback works);
  * the adapter evaluation path (evaluate_split_adapter over
    DFC2019Adapter) must produce the SAME pooled numbers as the frozen
    path — this is what makes cross-dataset evaluation trustworthy;
  * metrics extensions (building/slope/scene) are additive and consistent.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from depthwizard.calibration_net import CalibrationNet
from depthwizard.dataset import DFC2019Config
from depthwizard.geo import TilePaths, dump_json
from depthwizard.metrics import pooled_metrics
from depthwizard.normalize import clean_agl, minmax_normalize, valid_target_mask
from tools.make_fake_dataset import make_tile, write_tif


@pytest.fixture(scope="module")
def eval_world(tmp_path_factory):
    root = tmp_path_factory.mktemp("eval_world")
    rgb_dir = root / "rgb"
    truth_dir = root / "truth"
    stems = []
    for i in range(1, 10):
        stem = f"JAX_{i:03d}_{i:03d}"
        rgb, agl, cls = make_tile(seed=i, size=128)
        write_tif(rgb_dir / f"{stem}_RGB.tif", rgb, 3, "uint8")
        write_tif(truth_dir / f"{stem}_AGL.tif", agl, 1, "float32")
        write_tif(truth_dir / f"{stem}_CLS.tif", cls, 1, "uint8")
        stems.append(stem)
    splits = {"train": stems[:5], "val": stems[5:7], "test": stems[7:]}
    splits_json = root / "splits.json"
    dump_json({"splits": splits}, splits_json)

    # depth caches: LEGACY flat layout AND namespaced layout (same content)
    rng = np.random.default_rng(0)
    legacy = root / "cache" / "vit_b"                 # model-tagged dir
    legacy.mkdir(parents=True)
    ns = legacy / "dfc2019"
    ns.mkdir()
    for s in stems:
        raw = (rng.random((128, 128)) * 10).astype(np.float32)
        np.save(legacy / f"{s}.npy", raw)
        np.save(ns / f"{s}.npy", raw)

    # a fixed checkpoint: in_ch=1 dn-only, exact affine init (a0=2, b0=4)
    net = CalibrationNet(in_ch=1, widths=(8, 16, 32), a0=2.0, b0=4.0)
    net.eval()
    return root, rgb_dir, truth_dir, stems, splits_json, legacy, net


def _tiles(rgb_dir, truth_dir, stems):
    return [TilePaths(stem=s, rgb=rgb_dir / f"{s}_RGB.tif",
                      agl=truth_dir / f"{s}_AGL.tif",
                      cls=truth_dir / f"{s}_CLS.tif") for s in stems]


def _reference_frozen_eval(net, tiles, cache_dir, device="cpu"):
    """The ORIGINAL pre-GAMUS evaluate_split loop, verbatim (legacy cache
    read; no extensions). This is the frozen behavior we must reproduce."""
    from depthwizard.geo import read_tile
    pooled_p, pooled_t, pooled_m = [], [], []
    for t in tiles:
        raw = np.load(cache_dir / f"{t.stem}.npy")
        dn = torch.from_numpy(
            minmax_normalize(raw)[None, None].astype(np.float32)).to(device)
        with torch.no_grad():
            pred = net(dn, None)["pred"][0, 0].cpu().numpy()
        agl = clean_agl(read_tile(t)["agl"])
        m = valid_target_mask(agl)
        pooled_p.append(pred[m].ravel())
        pooled_t.append(agl[m].ravel())
        pooled_m.append(m[m].ravel())
    return pooled_metrics(pooled_p, pooled_t, pooled_m)


class TestFrozenEvaluateRegression:
    def test_pooled_identical_to_pre_change_logic(self, eval_world):
        root, rgb_dir, truth_dir, stems, splits_json, legacy, net = eval_world
        from depthwizard.cli.eval_calibration import evaluate_split
        from depthwizard.dataset import DFC2019Dataset

        tiles = _tiles(rgb_dir, truth_dir, stems[5:])      # val+test
        ds = DFC2019Dataset(tiles, DFC2019Config(
            rgb_dir=rgb_dir, truth_dir=truth_dir,
            depth_cache_dir=legacy, load_depth=True, crop_size=None))
        pooled, _summary, _city, _per, ext = evaluate_split(
            net, ds, use_rgb=False, device="cpu", cache_dir=legacy)
        ref = _reference_frozen_eval(net, tiles, legacy)
        for k in ("n", "mae", "rmse", "bias", "pearson_r", "neg_frac_pred"):
            assert pooled[k] == pytest.approx(ref[k], abs=1e-12), \
                f"frozen eval metric {k} DRIFTED: {pooled[k]} vs {ref[k]}"

    def test_namespaced_cache_gives_same_numbers(self, eval_world):
        """Only the namespaced layout present -> identical pooled metrics
        (cache-path fallback)."""
        root, rgb_dir, truth_dir, stems, _, legacy, net = eval_world
        import shutil
        only_ns = root / "cache_only_ns" / "vit_b"
        only_ns.mkdir(parents=True)
        shutil.copytree(legacy / "dfc2019", only_ns / "dfc2019")
        # reference uses the LEGACY files' content — identical by construction
        from depthwizard.cli.eval_calibration import evaluate_split
        from depthwizard.dataset import DFC2019Dataset

        tiles = _tiles(rgb_dir, truth_dir, stems[5:])
        ds = DFC2019Dataset(tiles, DFC2019Config(
            rgb_dir=rgb_dir, truth_dir=truth_dir,
            depth_cache_dir=only_ns, load_depth=True, crop_size=None))
        pooled, *_ = evaluate_split(net, ds, use_rgb=False, device="cpu",
                                    cache_dir=only_ns)
        ref = _reference_frozen_eval(net, tiles, legacy)
        assert pooled["mae"] == pytest.approx(ref["mae"], abs=1e-12)
        assert pooled["rmse"] == pytest.approx(ref["rmse"], abs=1e-12)

    def test_adapter_eval_path_matches_frozen(self, eval_world):
        """evaluate_split_adapter over DFC2019Adapter == frozen path pooled
        (the equivalence that makes cross-dataset numbers trustworthy)."""
        root, rgb_dir, truth_dir, stems, splits_json, legacy, net = eval_world
        from depthwizard.cli.eval_calibration import evaluate_split_adapter
        from depthwizard.datasets import DFC2019Adapter
        from depthwizard.tifops import LoadedModel

        tiles = _tiles(rgb_dir, truth_dir, stems[5:])
        ds = DFC2019Adapter(tiles, DFC2019Config(
            rgb_dir=rgb_dir, truth_dir=truth_dir,
            depth_cache_dir=legacy, load_depth=True, crop_size=None))
        model = LoadedModel(net=net, use_rgb=False, widths=(8, 16, 32),
                            clamp_min=0.0, affine_init={"a": 2.0, "b": 4.0},
                            epoch=0, checkpoint=Path("pin"),
                            val_subset_mae=None)
        pooled, *_ = evaluate_split_adapter(model, ds, "cpu", "dfc2019")
        ref = _reference_frozen_eval(net, tiles, legacy)
        for k in ("n", "mae", "rmse", "bias", "pearson_r"):
            assert pooled[k] == pytest.approx(ref[k], abs=1e-9), \
                f"adapter eval {k} drifted: {pooled[k]} vs {ref[k]}"

    def test_extensions_are_additive_and_sane(self, eval_world):
        root, rgb_dir, truth_dir, stems, _, legacy, net = eval_world
        from depthwizard.cli.eval_calibration import evaluate_split
        from depthwizard.dataset import DFC2019Dataset

        tiles = _tiles(rgb_dir, truth_dir, stems[5:])
        ds = DFC2019Dataset(tiles, DFC2019Config(
            rgb_dir=rgb_dir, truth_dir=truth_dir,
            depth_cache_dir=legacy, load_depth=True, crop_size=None))
        pooled, _s, _c, _p, ext = evaluate_split(
            net, ds, use_rgb=False, device="cpu", cache_dir=legacy)
        # extensions exist, are finite, and slope was REFUSED (no GSD on DFC)
        assert ext["building_mae_mean"] is not None
        assert np.isfinite(ext["building_mae_mean"])
        assert ext["slope"] is not None and "REFUSED" in str(ext["slope"])
        assert ext["scene_types"]                     # labels assigned
