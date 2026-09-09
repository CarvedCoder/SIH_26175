"""depthwizard.datasets — multi-dataset adapter package (GAMUS integration).

Introduces GAMUS alongside DFC2019 through ONE common sample contract while
keeping the frozen ``depthwizard.dataset.DFC2019Dataset`` path
behavior-identical (model_tests/test_dataset.py passes unmodified).

Sample contract (superset of the frozen DFC2019 contract — keys ALWAYS
present, values None when the layer is absent, so torch default_collate
stays consistent):

    rgb         [3,H,W]  float32         ImageNet-normalized
    agl         [1,H,W]  float32         clean_agl (>= clamp_min)
    cls         [1,H,W]  int64           RAW dataset class ids (legend: meta)
    dn          [1,H,W]  float32 | None  min-max normalized relative depth
    dem         [1,H,W]  float32 | None  DEM prior (real or SYNTHETIC tag)
    sem_onehot  [6,H,W]  float32 | None  ONE-HOT over PROJECT_CLASSES
    sem_ignore  [1,H,W]  bool     | None explicit semantic ignore mask
    meta        dict                     stem/h/w/y0/x0/rot90/flips/dem_tag
                                           + dataset/sample_id/sem_legend
                                           + source-specific honesty fields

Modules:
    transforms.py  joint crop/flip/rot (single source of truth; the frozen
                   DFC2019Dataset delegates here too)
    semantics.py   VERIFIED legends -> project classes (anti-fabrication)
    base.py        BaseDepthDataset (shared __getitem__ pipeline)
    dfc2019.py     DFC2019Adapter (thin subclass of the frozen dataset)
    gamus.py       GAMUSDataset (raw HDF5, hf lazy download | local dir)
    mixed.py       MixedDataset (GATED on verified per-dataset stats)
    factory.py     build_dataset(s) from the YAML ``dataset:`` section

Import hygiene: dfc2019 / mixed / factory are imported LAZILY (module
__getattr__) because they import ``depthwizard.dataset`` — which imports
this package's transforms lazily in return. Do not add module-level imports
of those three here.
"""

from __future__ import annotations

from . import base, gamus, semantics, transforms
from .base import AdapterConfig, BaseDepthDataset
from .semantics import (IGNORE, NUM_PROJECT_CLASSES, PROJECT_CLASSES,
                        PROJECT_CLASS_TO_INDEX, PROJECT_MAPS, RAW_LEGENDS,
                        class_to_project, legend_report, semantic_layers)
from .transforms import joint_crop, joint_flip_rot

__all__ = [
    "AdapterConfig", "BaseDepthDataset",
    "IGNORE", "NUM_PROJECT_CLASSES", "PROJECT_CLASSES",
    "PROJECT_CLASS_TO_INDEX", "PROJECT_MAPS", "RAW_LEGENDS",
    "class_to_project", "legend_report", "semantic_layers",
    "joint_crop", "joint_flip_rot",
    # lazy (module __getattr__):
    "build_dataset", "build_datasets",
    "DFC2019Adapter", "discover_and_split_adapter",
    "GAMUSConfig", "GAMUSDataset", "GAMUSSample",
    "build_gamus_datasets", "list_gamus_samples",
    "MixedDataset", "build_mixed_datasets",
]

_LAZY = {
    "build_dataset": "factory",
    "build_datasets": "factory",
    "DFC2019Adapter": "dfc2019",
    "discover_and_split_adapter": "dfc2019",
    "GAMUSConfig": "gamus",
    "GAMUSDataset": "gamus",
    "GAMUSSample": "gamus",
    "build_gamus_datasets": "gamus",
    "list_gamus_samples": "gamus",
    "MixedDataset": "mixed",
    "build_mixed_datasets": "mixed",
}


def __getattr__(name):
    mod_name = _LAZY.get(name)
    if mod_name is None:
        raise AttributeError(f"module 'depthwizard.datasets' has no "
                             f"attribute {name!r}")
    import importlib
    mod = importlib.import_module(f".{mod_name}", __name__)
    return getattr(mod, name)
