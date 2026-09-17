"""Tests for the verified semantic legends -> project-class mapping.

Pins the ANTI-FABRICATION contract of depthwizard/datasets/semantics.py:
    * one-hot over the 6 project classes, canonical order;
    * explicit ignore mask (DFC 65 void -> ignore; GAMUS 0 -> REAL class
      'other', NOT ignore — the two datasets differ deliberately);
    * unknown raw ids -> ignore + reported (never guessed, never crash);
    * raw ids preserved (mapping never rewrites the input).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from depthwizard.datasets.semantics import (NUM_PROJECT_CLASSES, PROJECT_CLASSES,
                                             PROJECT_CLASS_TO_INDEX,
                                             RAW_LEGENDS,
                                             class_to_project, legend_report,
                                             semantic_layers)


def test_project_class_system_frozen():
    assert PROJECT_CLASSES == ("building", "vegetation", "road", "water",
                               "ground", "other")
    assert NUM_PROJECT_CLASSES == 6
    assert PROJECT_CLASS_TO_INDEX["building"] == 0
    assert PROJECT_CLASS_TO_INDEX["other"] == 5


def test_legends_match_verified_sources():
    # DFC2019 LAS legend (pubgeo/dfc2019 data/README.md)
    assert RAW_LEGENDS["dfc2019"] == {
        2: "Ground", 5: "Trees", 6: "Buildings", 9: "Water",
        17: "Bridge/elevated road", 65: "Unlabeled (void)",
    }
    # GAMUS legend (RSI-MMSegmentation README)
    assert RAW_LEGENDS["gamus"] == {
        0: "others (background)", 1: "ground", 2: "low vegetation",
        3: "buildings", 4: "water", 5: "road", 6: "tree",
    }


def test_dfc2019_mapping_and_ignore():
    assert class_to_project("dfc2019", 6) == "building"
    assert class_to_project("dfc2019", 5) == "vegetation"
    assert class_to_project("dfc2019", 2) == "ground"
    assert class_to_project("dfc2019", 9) == "water"
    assert class_to_project("dfc2019", 17) == "road"      # bridge -> road
    assert class_to_project("dfc2019", 65) is None       # void -> IGNORE

    cls = np.array([[6, 2, 65], [5, 9, 17]], dtype=np.int32)
    onehot, ignore, unmapped = semantic_layers(cls, "dfc2019")
    assert onehot.shape == (6, 2, 3)
    assert unmapped == []
    # building channel set exactly at the 6
    assert onehot[0, 0, 0] == 1.0 and onehot[0].sum() == 1
    # 65 -> ignore, all-zero one-hot column
    assert ignore[0, 2] and onehot[:, 0, 2].sum() == 0
    # every non-ignored pixel is exactly one-hot
    valid = ~ignore
    assert np.allclose(onehot.sum(axis=0)[valid], 1.0)


def test_gamus_mapping_zero_is_other_not_ignore():
    # The critical DFC-vs-GAMUS distinction: GAMUS 0 is a REAL class.
    assert class_to_project("gamus", 0) == "other"
    assert class_to_project("gamus", 3) == "building"
    assert class_to_project("gamus", 2) == "vegetation"   # low veg
    assert class_to_project("gamus", 6) == "vegetation"   # tree (merged)
    assert class_to_project("gamus", 5) == "road"

    cls = np.array([[0, 1, 3], [4, 5, 6]], dtype=np.uint8)
    onehot, ignore, unmapped = semantic_layers(cls, "gamus")
    assert not ignore.any() and unmapped == []
    assert onehot[5, 0, 0] == 1.0                        # 0 -> other channel
    assert np.allclose(onehot.sum(axis=0), 1.0)          # fully dense


def test_unknown_ids_go_to_ignore_and_are_reported():
    cls = np.array([[2, 99], [6, 200]], dtype=np.int32)
    onehot, ignore, unmapped = semantic_layers(cls, "dfc2019")
    # 99 and 200 are not in the verified DFC legend
    assert sorted(unmapped) == [99, 200]
    assert ignore[0, 1] and ignore[1, 1]
    assert onehot[:, 0, 1].sum() == 0 and onehot[:, 1, 1].sum() == 0
    assert onehot[0, 1, 0] == 1.0                        # 6 -> building OK


def test_float_cls_dtype_inconsistency_handled():
    """The real GAMUS release stores CLS as uint8 OR float32 — both must
    map identically (values are integral either way)."""
    a = np.array([[0, 1, 3], [4, 5, 6]], dtype=np.uint8)
    b = a.astype(np.float32)
    ha, ia, _ = semantic_layers(a, "gamus")
    hb, ib, _ = semantic_layers(b, "gamus")
    assert np.array_equal(ha, hb) and np.array_equal(ia, ib)


def test_raw_ids_never_mutated():
    cls = np.array([[6, 65, 2]], dtype=np.int32)
    original = cls.copy()
    semantic_layers(cls, "dfc2019")
    assert np.array_equal(cls, original)


def test_unknown_dataset_raises():
    import pytest
    with pytest.raises(KeyError):
        class_to_project("nope", 1)
    with pytest.raises(KeyError):
        semantic_layers(np.zeros((2, 2), np.int32), "nope")


def test_legend_report_roundtrip():
    for ds in ("dfc2019", "gamus"):
        rep = legend_report(ds)
        assert rep["dataset"] == ds
        assert rep["project_classes"] == list(PROJECT_CLASSES)
        # every mapped name is a project class or the IGNORE marker
        for name in rep["project_mapping"].values():
            assert name == "IGNORE" or name in PROJECT_CLASS_TO_INDEX
