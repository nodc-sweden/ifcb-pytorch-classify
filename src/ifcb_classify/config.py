from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path

import yaml


def _expand_date_placeholders(value: str) -> str:
    """Expand date placeholders like {year}, {month}, {day} in path strings."""
    now = datetime.now(timezone.utc)
    return value.format(
        year=now.strftime("%Y"),
        month=now.strftime("%m"),
        day=now.strftime("%d"),
        date=now.strftime("%Y%m%d"),
    )


@dataclass(frozen=True)
class TrainConfig:
    data_dir: str = "training_data/V1"
    dataset_version: str = "V1"
    val_split: float = 0.2
    image_width: int = 224
    image_height: int = 224
    mean: float | None = None
    std: float | None = None
    transform: str = "dataset_squarepad_augmented"
    model: str = "resnet50"
    pretrained: bool = True
    lr: float = 0.0001
    batch_size: int = 64
    epochs: int = 20
    num_workers: int = 0
    seed: int = 42
    output_dir: str = "output"
    checkpoint_metric: str = "weighted_f1"
    tracker: str = "csv"
    mlflow_uri: str | None = None
    wandb_project: str | None = None
    experiment_name: str = "ifcb-classify"
    sweep_params: dict | None = None
    min_class_images: int | None = None
    manual_include_classes: list[str] | None = None
    plots: bool = False

    def __post_init__(self):
        if not (0.0 < self.val_split < 1.0):
            raise ValueError(f"val_split must be between 0 and 1 exclusive, got {self.val_split}")
        if self.lr <= 0:
            raise ValueError(f"lr must be positive, got {self.lr}")
        if self.batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {self.batch_size}")
        if self.epochs < 1:
            raise ValueError(f"epochs must be >= 1, got {self.epochs}")
        if self.image_width < 1 or self.image_height < 1:
            raise ValueError(f"image dimensions must be positive, got {self.image_width}x{self.image_height}")


@dataclass(frozen=True)
class InferConfig:
    input_path: str = ""
    model_checkpoint: str = ""
    output_dir: str = "output/class_scores"
    batch_size: int = 64
    num_workers: int = 0
    thresholds_path: str | None = None
    threshold_default: float = 0.0
    device: str = "auto"
    classifier_name: str | None = None
    overwrite: bool = False
    classes_path: str | None = None
    model_name: str | None = None
    num_threads: int | None = None

    def __post_init__(self):
        if self.batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {self.batch_size}")
        if self.num_threads is not None and self.num_threads < 1:
            raise ValueError(f"num_threads must be >= 1, got {self.num_threads}")


def load_config(yaml_path: str | Path, config_cls: type, overrides: dict | None = None):
    with open(yaml_path) as f:
        data = yaml.safe_load(f) or {}
    if overrides:
        data.update({k: v for k, v in overrides.items() if v is not None})
    filtered = {k: v for k, v in data.items() if k in config_cls.__dataclass_fields__}
    for k, v in filtered.items():
        if isinstance(v, str) and "{" in v:
            filtered[k] = _expand_date_placeholders(v)
    return config_cls(**filtered)


def config_to_dict(config) -> dict:
    return asdict(config)
