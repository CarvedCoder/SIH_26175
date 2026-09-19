import yaml
from dataclasses import dataclass, field, fields
from typing import List, Optional

@dataclass
class ModelConfig:
    architecture: str = "CalibrationNet_v2"
    widths: List[int] = field(default_factory=lambda: [16, 32, 64])
    parameterization: str = "absolute_affine"
    bounded: bool = False
    semantic_mode: str = "input"
    context: str = "none"
    clamp_min: float = 0.0
    film: bool = False

@dataclass
class FusionConfig:
    mode: str = "early"
    merge: str = "concat"

@dataclass
class InputsConfig:
    depth: bool = True
    rgb: bool = True
    confidence: bool = False
    semantic: bool = False
    dem: bool = False

@dataclass
class LossWeightsConfig:
    main: str = "l1"
    gradient_weight: float = 0.0
    boundary_weight: float = 0.0
    semantic_weight: float = 0.0
    uncertainty_weight: float = 0.0
    height_balanced: bool = False
    huber_delta: float = 5.0
    berhu_c: float = 0.0

@dataclass
class DatasetMixingConfig:
    mode: str = "ratio"
    dfc_weight: float = 0.2
    gamus_weight: float = 0.8

@dataclass
class DatasetConfig:
    name: str = "mixed"
    mixing: DatasetMixingConfig = field(default_factory=DatasetMixingConfig)
    clamp_agl_min: float = 0.0

@dataclass
class TrainConfig:
    epochs: int = 100
    batch_size: int = 16
    lr: float = 0.001
    weight_decay: float = 0.0001
    crop_size: int = 256
    val_subset: int = 859
    grad_clip: float = 5.0
    patience: int = 8
    seed: int = 42

@dataclass
class PathsConfig:
    rgb_dir: str = "images/train"
    truth_dir: str = "heights/train"
    splits_json: str = "splits.json"
    affine_json: str = "outputs/calib_net/global_affine.json"
    outputs_dir: str = "outputs"
    depth_cache_dir: str = "outputs/depth_cache"

@dataclass
class CalibrationConfig:
    paths: PathsConfig = field(default_factory=PathsConfig)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    inputs: InputsConfig = field(default_factory=InputsConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    fusion: FusionConfig = field(default_factory=FusionConfig)
    loss: LossWeightsConfig = field(default_factory=LossWeightsConfig)
    train: TrainConfig = field(default_factory=TrainConfig)

    @staticmethod
    def _section(dataclass_cls, d: dict):
        """Build a section dataclass, ignoring keys this dataclass doesn't
        know (legacy configs carry extra keys like model.use_rgb,
        dataset.source/backend/local_root, train.workers — the train CLI
        still reads those from the raw YAML, so dropping them here is safe)."""
        known = {f.name for f in fields(dataclass_cls)}
        return dataclass_cls(**{k: v for k, v in d.items() if k in known})

    @classmethod
    def from_dict(cls, d: dict) -> 'CalibrationConfig':
        ds = d.get("dataset", {})
        return cls(
            paths=cls._section(PathsConfig, d.get("paths", {})),
            dataset=DatasetConfig(
                name=ds.get("name", "mixed"),
                mixing=cls._section(DatasetMixingConfig, ds.get("mixing", {})),
                clamp_agl_min=ds.get("clamp_agl_min", 0.0),
            ),
            inputs=cls._section(InputsConfig, d.get("inputs", {})),
            model=cls._section(ModelConfig, d.get("model", {})),
            fusion=cls._section(FusionConfig, d.get("fusion", {})),
            loss=cls._section(LossWeightsConfig, d.get("loss", {})),
            train=cls._section(TrainConfig, d.get("train", {})),
        )

    @classmethod
    def from_yaml(cls, path: str) -> 'CalibrationConfig':
        with open(path, "r", encoding="utf-8") as f:
            d = yaml.safe_load(f)
        return cls.from_dict(d)
