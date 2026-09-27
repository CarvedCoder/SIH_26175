import yaml
from dataclasses import dataclass, field, fields
from typing import List, Optional

@dataclass
class ModelConfig:
    """Height-model section (RDAH integration).

    architecture:
      "rdah"            official RDAH-Net backend (DEFAULT for new
                        configs — pretrained HeightPredTransformer;
                        depth input = raw DAv2 x depth_scale, output =
                        nDSM metres; see depthwizard/rdah.py)
      "calibration_net" the legacy Phase-2 net (H = clamp(a*Dn + b, 0))
      Legacy value "CalibrationNet_v2" maps to calibration_net.
    NOTE: configs that carry NO model.architecture key keep the frozen
    legacy behaviour in the train CLI (calibration_net) so every existing
    experiment config is unaffected; the DATACLASS default is the new
    backend. Inference auto-detects from the checkpoint when unspecified.
    """
    architecture: str = "rdah"
    widths: List[int] = field(default_factory=lambda: [16, 32, 64])
    parameterization: str = "absolute_affine"
    bounded: bool = False
    semantic_mode: str = "input"
    context: str = "none"
    clamp_min: float = 0.0
    film: bool = False
    # ---- RDAH backend knobs (ignored by calibration_net) ----
    checkpoint: Optional[str] = None  # default: checkpoints/rdah/<track1>.pth
    pretrained: bool = True            # load the released weights
    depth_scale: float = 40.0          # raw DAv2 depth x scale (verified)


@dataclass
class RDAHConfig:
    """RDAH-specific runtime settings (no duplication with ModelConfig —
    only resolution/AMP concerns that belong to the backend itself)."""
    input_size: int = 1024        # Track1 training resolution (multiples of 128)
    use_amp: bool = True          # autocast on CUDA (RTX-4050-friendly)


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
    rdah: RDAHConfig = field(default_factory=RDAHConfig)
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
        model_raw = d.get("model", {})
        arch = model_raw.get("architecture")
        if arch in ("CalibrationNet_v2", None, ""):
            # legacy configs (no model.architecture key) keep the frozen
            # Phase-2 behaviour; only an explicit "rdah" switches backend
            arch = "calibration_net"
        return cls(
            paths=cls._section(PathsConfig, d.get("paths", {})),
            dataset=DatasetConfig(
                name=ds.get("name", "mixed"),
                mixing=cls._section(DatasetMixingConfig, ds.get("mixing", {})),
                clamp_agl_min=ds.get("clamp_agl_min", 0.0),
            ),
            inputs=cls._section(InputsConfig, d.get("inputs", {})),
            model=ModelConfig(
                architecture=arch,
                **{k: v for k, v in model_raw.items()
                   if k in ("widths", "parameterization", "bounded",
                            "semantic_mode", "context", "clamp_min", "film",
                            "checkpoint", "pretrained", "depth_scale")},
            ),
            rdah=cls._section(RDAHConfig, d.get("rdah", {})),
            fusion=cls._section(FusionConfig, d.get("fusion", {})),
            loss=cls._section(LossWeightsConfig, d.get("loss", {})),
            train=cls._section(TrainConfig, d.get("train", {})),
        )

    @classmethod
    def from_yaml(cls, path: str) -> 'CalibrationConfig':
        with open(path, "r", encoding="utf-8") as f:
            d = yaml.safe_load(f)
        return cls.from_dict(d)
