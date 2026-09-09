"""Tests for the multi-dataset adapter interface (GAMUS integration).

Covers:
  * DFC2019Adapter: the frozen DFC path + one-hot semantics + provenance meta,
    byte-equal to the legacy DFC2019Dataset when built with the same seed;
  * the namespaced depth cache (new layout first, legacy fallback);
  * GAMUSDataset on synthetic HDF5 fixtures (network-free): official-layout
    reads, dtype inconsistency, honesty meta, joint-transform alignment of
    sem layers, deterministic sample ids, limit, manifest reuse;
  * MixedDataset gating: mixing is REFUSED until a verified stats artifact
    exists (binding user constraint);
  * factory: dfc2019 via legacy paths config == adapter pipeline.

Run:  pytest model_tests/test_datasets_interface.py -q
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from depthwizard.dataset import DFC2019Config, DFC2019Dataset
from depthwizard.datasets import (DFC2019Adapter, GAMUSConfig, GAMUSDataset,
                                  MixedDataset, build_datasets,
                                  discover_and_split_adapter)
from depthwizard.datasets.gamus import (GAMUSSample, build_gamus_datasets,
                                        list_gamus_samples, read_gamus_h5)
from depthwizard.geo import TilePaths, depth_npy_candidates
from tools.make_fake_dataset import make_tile, write_tif

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from make_fake_gamus import make_gamus_tile, write_h5


# ---------------------------------------------------------------------------
# DFC2019 fixtures (same shape as test_dataset.py)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def dfc_root(tmp_path_factory):
    root = tmp_path_factory.mktemp("fake_dfc")
    rgb_dir = root / "rgb"; truth_dir = root / "truth"
    splits = {"train": [], "val": [], "test": []}
    stems = []
    for i in range(1, 10):
        stem = f"JAX_{i:03d}_{i:03d}"
        rgb, agl, cls = make_tile(seed=i, size=256)
        write_tif(rgb_dir / f"{stem}_RGB.tif", rgb, 3, "uint8")
        write_tif(truth_dir / f"{stem}_AGL.tif", agl, 1, "float32")
        write_tif(truth_dir / f"{stem}_CLS.tif", cls, 1, "uint8")
        stems.append(stem)
    splits["train"] = stems[:5]
    splits["val"] = stems[5:7]
    splits["test"] = stems[7:]
    import json
    (root / "splits.json").write_text(json.dumps({"splits": splits}))
    return root, rgb_dir, truth_dir, stems, root / "splits.json"


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


# ---------------------------------------------------------------------------
# DFC2019Adapter
# ---------------------------------------------------------------------------

class TestDFC2019Adapter:
    def test_semantic_layers_added_and_correct(self, dfc_root):
        root, rgb_dir, truth_dir, stems, _ = dfc_root
        ds = DFC2019Adapter(_tiles(rgb_dir, truth_dir, stems),
                            _cfg(rgb_dir, truth_dir))
        s = ds[0]
        assert s["sem_onehot"].shape == (6, 256, 256)
        assert s["sem_onehot"].dtype == torch.float32
        assert s["sem_ignore"].shape == (1, 256, 256)
        assert s["sem_ignore"].dtype == torch.bool
        assert s["meta"]["dataset"] == "dfc2019"
        assert s["meta"]["sample_id"] == stems[0]
        assert s["meta"]["sem_legend"] == "dfc2019"
        # fake tiles contain ids {2,5,6,9,65}; 65 -> ignore
        assert bool(s["sem_ignore"].any())
        # one-hot consistent with raw cls at every pixel
        from depthwizard.datasets import semantic_layers
        ref_onehot, ref_ignore, _ = semantic_layers(
            s["cls"][0].numpy(), "dfc2019")
        assert torch.equal(s["sem_onehot"],
                           torch.from_numpy(ref_onehot))
        assert torch.equal(s["sem_ignore"][0],
                           torch.from_numpy(ref_ignore))

    def test_adapter_byte_equal_to_frozen_dataset(self, dfc_root):
        """Same seed -> identical rgb/agl/cls/dn/meta vs DFC2019Dataset.
        The adapter must not perturb the frozen path in any way."""
        root, rgb_dir, truth_dir, stems, _ = dfc_root
        cfg_kw = dict(crop_size=64, augment=True, seed=7)
        legacy = DFC2019Dataset(_tiles(rgb_dir, truth_dir, stems),
                                _cfg(rgb_dir, truth_dir, **cfg_kw))
        adapter = DFC2019Adapter(_tiles(rgb_dir, truth_dir, stems),
                                 _cfg(rgb_dir, truth_dir, **cfg_kw))
        for i in (0, 2, 4):
            a, b = legacy[i], adapter[i]
            assert torch.equal(a["rgb"], b["rgb"])
            assert torch.equal(a["agl"], b["agl"])
            assert torch.equal(a["cls"], b["cls"])
            for k in ("stem", "h", "w", "y0", "x0", "rot90", "flip_h",
                      "flip_v", "dem_tag"):
                assert a["meta"][k] == b["meta"][k], f"meta[{k}] drifted"

    def test_sem_layers_jointly_transformed(self, dfc_root):
        """sem layers must follow the SAME window/rot/flip as rgb — verified
        by reconstructing one-hot from the transformed cls (per-pixel mapping
        commutes with spatial transforms, so they must be EQUAL)."""
        root, rgb_dir, truth_dir, stems, _ = dfc_root
        ds = DFC2019Adapter(_tiles(rgb_dir, truth_dir, stems),
                            _cfg(rgb_dir, truth_dir, crop_size=64,
                                 augment=True, seed=11))
        s = ds[3]
        from depthwizard.datasets import semantic_layers
        ref_onehot, ref_ignore, _ = semantic_layers(
            s["cls"][0].numpy(), "dfc2019")
        assert torch.equal(s["sem_onehot"], torch.from_numpy(ref_onehot))
        assert torch.equal(s["sem_ignore"][0], torch.from_numpy(ref_ignore))

    def test_load_semantics_false_disables_layers(self, dfc_root):
        root, rgb_dir, truth_dir, stems, _ = dfc_root
        ds = DFC2019Adapter(_tiles(rgb_dir, truth_dir, stems),
                            _cfg(rgb_dir, truth_dir), load_semantics=False)
        s = ds[0]
        assert s["sem_onehot"] is None and s["sem_ignore"] is None
        assert "sem_legend" not in s["meta"]

    def test_discover_and_split_adapter_builds_three_splits(self, dfc_root):
        root, rgb_dir, truth_dir, stems, splits_json = dfc_root
        out = discover_and_split_adapter(_cfg(rgb_dir, truth_dir), splits_json)
        assert set(out) == {"train", "val", "test"}
        assert all(isinstance(d, DFC2019Adapter) for d in out.values())
        assert len(out["train"]) == 5 and len(out["val"]) == 2


# ---------------------------------------------------------------------------
# Namespaced depth cache
# ---------------------------------------------------------------------------

class TestDepthCacheNamespacing:
    def test_candidates_order_and_legacy(self, tmp_path):
        c = depth_npy_candidates(tmp_path, "dfc2019", "JAX_004_006")
        assert c[0] == tmp_path / "dfc2019" / "JAX_004_006.npy"
        assert c[1] == tmp_path / "JAX_004_006.npy"      # legacy fallback
        g = depth_npy_candidates(tmp_path, "gamus", "DC_01_25")
        assert g == [tmp_path / "gamus" / "DC_01_25.npy"]

    def _cache_setup(self, tmp_path, stems, layout):
        cache = tmp_path / "depth_cache" / "vit_b" if layout == "tag" \
            else tmp_path / "depth_cache"
        cache.mkdir(parents=True, exist_ok=True)
        rng = np.random.default_rng(0)
        for s in stems:
            np.save(cache / f"{s}.npy",
                    rng.random((256, 256)).astype(np.float32))
        return cache

    @pytest.mark.parametrize("layout", ["flat", "tag"])
    def test_legacy_cache_still_loads(self, dfc_root, tmp_path, layout):
        """Pre-GAMUS caches (flat or model-tagged) must keep working."""
        root, rgb_dir, truth_dir, stems, _ = dfc_root
        cache = self._cache_setup(tmp_path, stems, layout)
        ds = DFC2019Dataset(_tiles(rgb_dir, truth_dir, stems),
                            _cfg(rgb_dir, truth_dir, depth_cache_dir=cache))
        s = ds[0]
        assert s["dn"] is not None and s["dn"].shape == (1, 256, 256)

    def test_namespaced_cache_loads(self, dfc_root, tmp_path):
        root, rgb_dir, truth_dir, stems, _ = dfc_root
        model_dir = tmp_path / "depth_cache" / "vit_b"
        ns = model_dir / "dfc2019"
        ns.mkdir(parents=True)
        rng = np.random.default_rng(1)
        for s in stems:
            np.save(ns / f"{s}.npy", rng.random((256, 256)).astype(np.float32))
        ds = DFC2019Dataset(_tiles(rgb_dir, truth_dir, stems),
                            _cfg(rgb_dir, truth_dir, depth_cache_dir=model_dir))
        assert ds[0]["dn"] is not None

    def test_shape_mismatch_still_raises(self, dfc_root, tmp_path):
        root, rgb_dir, truth_dir, stems, _ = dfc_root
        model_dir = tmp_path / "depth_cache" / "vit_b"
        ns = model_dir / "dfc2019"
        ns.mkdir(parents=True)
        np.save(ns / f"{stems[0]}.npy", np.zeros((128, 128), np.float32))
        ds = DFC2019Dataset(_tiles(rgb_dir, truth_dir, stems),
                            _cfg(rgb_dir, truth_dir,
                                 depth_cache_dir=model_dir))
        with pytest.raises(ValueError, match="out of sync"):
            _ = ds[0]


# ---------------------------------------------------------------------------
# GAMUS fixtures + adapter
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def gamus_root(tmp_path_factory):
    root = tmp_path_factory.mktemp("fake_gamus")
    k = 0
    for split in ("train", "val", "test"):
        for i in range(1, 3):
            k += 1
            sid = f"DC_{i:02d}_{k:02d}"
            cls_dtype = "float32" if k % 3 == 0 else "uint8"
            rgb, agl, cls = make_gamus_tile(seed=700 + k, size=128,
                                            cls_dtype=cls_dtype)
            write_h5(root / "images" / split / f"{sid}_RGB.h5", rgb)
            write_h5(root / "heights" / split / f"{sid}_AGL.h5", agl)
            write_h5(root / "classes" / split / f"{sid}_CLS.h5", cls)
    return root


def _gamus_cfg(gamus_root, **kw):
    d = dict(source="local", local_root=gamus_root, load_depth=False,
             crop_size=None, augment=False)
    if kw.get("depth_cache_dir") is not None:
        d["load_depth"] = True
    d.update(kw)
    return GAMUSConfig(**d)


class TestGAMUSDataset:
    def test_discovery_and_official_splits(self, gamus_root):
        per_split, problems = list_gamus_samples(_gamus_cfg(gamus_root))
        assert problems == []
        assert set(per_split) == {"train", "val", "test"}
        assert all(len(v) == 2 for v in per_split.values())
        # deterministic sorted order
        ids = [s.sample_id for s in per_split["train"]]
        assert ids == sorted(ids)

    def test_sample_contract(self, gamus_root):
        ds = build_gamus_datasets(_gamus_cfg(gamus_root))["train"]
        s = ds[0]
        assert s["rgb"].shape == (3, 128, 128) and s["rgb"].dtype == torch.float32
        assert s["agl"].shape == (1, 128, 128) and s["agl"].dtype == torch.float32
        assert s["cls"].shape == (1, 128, 128) and s["cls"].dtype == torch.int64
        assert float(s["agl"].min()) >= 0.0                # clean_agl
        assert s["sem_onehot"].shape == (6, 128, 128)
        assert s["sem_ignore"].shape == (1, 128, 128)
        # GAMUS 0 is a real class -> NO ignore in the fixture data
        assert not bool(s["sem_ignore"].any())
        assert np.allclose(s["sem_onehot"].sum(0).numpy(), 1.0)

    def test_honesty_meta_fields(self, gamus_root):
        ds = build_gamus_datasets(_gamus_cfg(gamus_root))["train"]
        m = ds[0]["meta"]
        assert m["dataset"] == "gamus"
        assert m["sample_id"] == ds.samples[0].sample_id
        assert m["split"] == "train"
        assert m["georef"] == "none"
        assert "nDSM" in m["height_semantics"]
        assert "ASSUMED" in m["units"]                     # flagged, not fabricated
        assert m["gsd_m"] == 0.33

    def test_cls_dtype_inconsistency_handled(self, gamus_root):
        """Fixture stores every 3rd CLS as float32 (like the real release);
        both dtypes must yield identical int64 cls tensors."""
        # the float32 tiles land in val (k=3) and test (k=6) — scan all splits
        found_float = 0
        for split in ("train", "val", "test"):
            ds = build_gamus_datasets(_gamus_cfg(gamus_root))[split]
            for s_obj in ds.samples:
                raw = read_gamus_h5(gamus_root / "classes" / s_obj.split /
                                    f"{s_obj.sample_id}_CLS.h5")
                if raw.dtype == np.float32:
                    found_float += 1
                    # integral values cast to the same ids the adapter reads
                    sample = ds[ds.samples.index(s_obj)]
                    assert sample["cls"].dtype == torch.int64
                    assert np.array_equal(sample["cls"][0].numpy(),
                                          raw.astype(np.int64))
        assert found_float >= 1                              # fixture covers it

    def test_joint_transform_alignment_sem(self, gamus_root):
        ds = build_gamus_datasets(_gamus_cfg(gamus_root, crop_size=64,
                                             augment=True, seed=7))["train"]
        s = ds[1]
        from depthwizard.datasets import semantic_layers
        ref_onehot, ref_ignore, _ = semantic_layers(s["cls"][0].numpy(),
                                                    "gamus")
        assert torch.equal(s["sem_onehot"], torch.from_numpy(ref_onehot))
        assert torch.equal(s["sem_ignore"][0], torch.from_numpy(ref_ignore))
        assert s["rgb"].shape == (3, 64, 64)

    def test_determinism_same_seed(self, gamus_root):
        a = build_gamus_datasets(_gamus_cfg(gamus_root, crop_size=64,
                                            augment=True, seed=123))["val"]
        b = build_gamus_datasets(_gamus_cfg(gamus_root, crop_size=64,
                                            augment=True, seed=123))["val"]
        sa, sb = a[0], b[0]
        assert torch.equal(sa["agl"], sb["agl"])
        assert sa["meta"] == sb["meta"]

    def test_limit_deterministic(self, gamus_root):
        cfg = _gamus_cfg(gamus_root, limit=1)
        per_split, _ = list_gamus_samples(cfg)
        assert all(len(v) == 1 for v in per_split.values())

    def test_manifest_reuse(self, gamus_root, tmp_path):
        man = tmp_path / "gamus_splits.json"
        cfg = _gamus_cfg(gamus_root, save_manifest=man)
        per_split, _ = list_gamus_samples(cfg)
        assert man.exists()
        # rebuild from manifest only (no dir walk needed)
        import shutil
        moved = tmp_path / "moved_root"
        shutil.copytree(gamus_root, moved)
        cfg2 = GAMUSConfig(source="local", local_root=moved, manifest=man)
        per_split2, _ = list_gamus_samples(cfg2)
        assert {k: [x.sample_id for x in v] for k, v in per_split.items()} == \
               {k: [x.sample_id for x in v] for k, v in per_split2.items()}

    def test_gamus_depth_cache_namespaced(self, gamus_root, tmp_path):
        model_dir = tmp_path / "depth_cache" / "vit_b"
        ns = model_dir / "gamus"
        ns.mkdir(parents=True)
        rng = np.random.default_rng(5)
        ds = build_gamus_datasets(_gamus_cfg(gamus_root))["train"]
        for s_obj in ds.samples:
            np.save(ns / f"{s_obj.sample_id}.npy",
                    rng.random((128, 128)).astype(np.float32))
        ds2 = build_gamus_datasets(_gamus_cfg(gamus_root,
                                              depth_cache_dir=model_dir))["train"]
        s = ds2[0]
        assert s["dn"] is not None and s["dn"].shape == (1, 128, 128)
        assert 0.0 <= float(s["dn"].min()) and float(s["dn"].max()) <= 1.0

    def test_hf_source_requires_network_or_manifest(self, gamus_root):
        """source=hf without a manifest would hit the network — the config
        path must not be silently local."""
        cfg = GAMUSConfig(source="hf", manifest=None)
        assert cfg.source == "hf"

    def test_dataset4eo_backend_refuses_with_rationale(self):
        with pytest.raises(NotImplementedError, match="float16"):
            GAMUSConfig(source="hf", backend="dataset4eo")


# ---------------------------------------------------------------------------
# MixedDataset gating
# ---------------------------------------------------------------------------

class TestMixedGating:
    def test_refuses_without_stats(self, gamus_root, dfc_root):
        with pytest.raises(ValueError, match="GATED"):
            build_datasets({"dataset": {
                "name": "mixed",
                "sources": [{"name": "gamus",
                             "source": "local",
                             "local_root": str(gamus_root)},
                            {"name": "dfc2019"}]}},
                load_depth=False)

    def test_refuses_when_stats_unverified(self, gamus_root, tmp_path):
        stats = tmp_path / "stats.json"
        stats.write_text('{"datasets": {"gamus": {}, "dfc2019": {}}, '
                         '"verified": false}')
        with pytest.raises(ValueError, match="verified"):
            build_datasets({"dataset": {
                "name": "mixed", "verified_stats": str(stats),
                "sources": [{"name": "gamus", "source": "local",
                             "local_root": str(gamus_root)},
                            {"name": "dfc2019"}]}},
                load_depth=False)

    def test_refuses_when_stats_incomplete(self, gamus_root, tmp_path):
        stats = tmp_path / "stats.json"
        stats.write_text('{"datasets": {"gamus": {}}, "verified": true}')
        with pytest.raises(ValueError, match="dfc2019"):
            build_datasets({"dataset": {
                "name": "mixed", "verified_stats": str(stats),
                "sources": [{"name": "gamus", "source": "local",
                             "local_root": str(gamus_root)},
                            {"name": "dfc2019"}]}},
                load_depth=False)

    def test_builds_when_verified(self, gamus_root, dfc_root, tmp_path):
        import json
        root, rgb_dir, truth_dir, stems, splits_json = dfc_root
        stats = tmp_path / "stats.json"
        stats.write_text(json.dumps({
            "datasets": {"gamus": {}, "dfc2019": {}}, "verified": True}))
        out = build_datasets({
            "paths": {},
            "dataset": {
                "name": "mixed", "verified_stats": str(stats),
                "sources": [
                    {"name": "gamus", "source": "local",
                     "local_root": str(gamus_root), "weight": 2.0},
                    {"name": "dfc2019", "rgb_dir": str(rgb_dir),
                     "truth_dir": str(truth_dir),
                     "splits_json": str(splits_json)},
                ]}}, load_depth=False)
        assert set(out) == {"train", "val", "test"}
        mixed = out["train"]
        assert isinstance(mixed, MixedDataset)
        assert mixed.source_names == ["gamus", "dfc2019"]
        s = mixed[0]
        assert s["meta"]["mixed_source"] == "gamus"
        assert s["meta"]["mixed_source_weight"] == 2.0
        # per-sample weights: 2.0 per gamus sample, 1.0 per dfc sample
        w = mixed.per_sample_weights()
        assert set(w) == {2.0, 1.0}


# ---------------------------------------------------------------------------
# Factory: legacy paths config == DFC adapter pipeline
# ---------------------------------------------------------------------------

class TestFactory:
    def test_legacy_config_builds_dfc_adapter(self, dfc_root):
        root, rgb_dir, truth_dir, stems, splits_json = dfc_root
        cfg = {"paths": {"rgb_dir": str(rgb_dir),
                         "truth_dir": str(truth_dir),
                         "splits_json": str(splits_json)},
               "dataset": {"name": "dfc2019", "load_semantics": True}}
        out = build_datasets(cfg, load_depth=False)
        assert set(out) == {"train", "val", "test"}
        assert isinstance(out["train"], DFC2019Adapter)
        s = out["train"][0]
        assert s["sem_onehot"] is not None

    def test_no_dataset_section_defaults_to_dfc(self, dfc_root):
        root, rgb_dir, truth_dir, stems, splits_json = dfc_root
        cfg = {"paths": {"rgb_dir": str(rgb_dir),
                         "truth_dir": str(truth_dir),
                         "splits_json": str(splits_json)}}
        out = build_datasets(cfg, load_depth=False)
        assert isinstance(out["train"], DFC2019Adapter)

    def test_unknown_name_raises(self):
        with pytest.raises(ValueError, match="unknown dataset"):
            build_datasets({"dataset": {"name": "nope"}}, load_depth=False)
