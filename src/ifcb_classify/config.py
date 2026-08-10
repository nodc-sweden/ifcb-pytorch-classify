"""Typed configuration objects for the train and infer commands.

Both :class:`TrainConfig` and :class:`InferConfig` are frozen dataclasses whose
defaults define the pipeline's out-of-the-box behaviour and whose
``__post_init__`` validates the most error-prone fields. :func:`load_config`
reads a YAML file, applies CLI overrides, expands date placeholders in path-like
string values, drops unknown keys, and constructs the requested dataclass.

The same :func:`load_config` is reused for the chain-counting configs, which is
why it takes the dataclass type as an argument rather than hard-coding one.
"""

from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

import yaml


def _expand_date_placeholders(value: str) -> str:
    """Expand date placeholders like {year}, {month}, {day}, {date} in path strings."""
    now = datetime.now(UTC)
    replacements = {
        "year": now.strftime("%Y"),
        "month": now.strftime("%m"),
        "day": now.strftime("%d"),
        "date": now.strftime("%Y%m%d"),
    }
    for key, val in replacements.items():
        value = value.replace(f"{{{key}}}", val)
    return value


# Scalar fields of ``metrics.MetricsResult`` that a checkpoint can be selected on.
# Kept in sync with that dataclass by hand to avoid importing torch at config load.
VALID_CHECKPOINT_METRICS = (
    "accuracy",
    "precision",
    "recall",
    "f1",
    "weighted_f1",
    "auprc",
    "auroc",
)


@dataclass(frozen=True)
class TrainConfig:
    """Validated settings for a training run (see ``configs/train_default.yaml``).

    Frozen so a config can't be mutated mid-run; sweeps build new instances
    instead. ``sweep_params`` maps field names to lists of values to grid over;
    ``manual_include_classes`` force-keeps named classes that would otherwise be
    dropped by ``min_class_images``.
    """

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
        """Validate numeric ranges that would otherwise fail deep inside training."""
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
        if self.checkpoint_metric not in VALID_CHECKPOINT_METRICS:
            valid = ", ".join(VALID_CHECKPOINT_METRICS)
            raise ValueError(
                f"checkpoint_metric must be one of: {valid}; got {self.checkpoint_metric!r}"
            )


VALID_OUTPUT_FORMATS = ("h5", "csv", "mat", "csv-labels")


@dataclass(frozen=True)
class InferConfig:
    """Validated settings for an inference run (see ``configs/infer_default.yaml``).

    ``thresholds_path`` and ``classes_path`` are optional because both can be
    auto-detected next to the checkpoint. ``chain_counting`` holds the raw YAML
    block (parsed lazily into a ``ChainCountingConfig`` only if counting runs).
    ``allow_unsafe`` permits loading legacy raw-state-dict checkpoints.
    ``output_format`` selects which class-scores file(s) to write per bin: any of
    ``h5`` (default, IFCB Dashboard class_scores v3), ``csv`` (the dashboard's
    per-ROI scores export), ``mat`` (dashboard-ingestible v1 ``.mat``) and
    ``csv-labels`` (the ClassiPyR/iRfcb per-ROI resolved-label CSV). Accepts a
    single value, a comma-separated string, a YAML list, or ``"all"``.
    """

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
    allow_unsafe: bool = False
    chain_counting: dict | None = None
    output_format: str | list[str] = "h5"

    def __post_init__(self):
        """Validate the numeric fields most likely to be misconfigured."""
        if self.batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {self.batch_size}")
        if self.num_threads is not None and self.num_threads < 1:
            raise ValueError(f"num_threads must be >= 1, got {self.num_threads}")
        self.resolved_formats()  # validate output_format early

    def resolved_formats(self) -> tuple[str, ...]:
        """Return the requested output formats as an ordered, de-duplicated tuple.

        Parses ``output_format`` (string, comma-separated string, list, or the
        special value ``"all"``) into a subset of :data:`VALID_OUTPUT_FORMATS`,
        preserving request order. Raises ``ValueError`` on an unknown format.
        """
        raw = self.output_format
        items = raw.split(",") if isinstance(raw, str) else list(raw)
        items = [str(x).strip().lower() for x in items if str(x).strip()]
        if not items:
            items = ["h5"]

        resolved: list[str] = []
        for item in items:
            candidates = VALID_OUTPUT_FORMATS if item == "all" else (item,)
            for fmt in candidates:
                if fmt not in VALID_OUTPUT_FORMATS:
                    valid = ", ".join(VALID_OUTPUT_FORMATS)
                    raise ValueError(f"Unknown output format {fmt!r}; valid: {valid} (or 'all')")
                if fmt not in resolved:
                    resolved.append(fmt)
        return tuple(resolved)


def load_config(yaml_path: str | Path, config_cls: type, overrides: dict | None = None):
    """Load a YAML file into ``config_cls``, applying overrides and date expansion.

    Non-``None`` ``overrides`` (typically CLI flags) take precedence over YAML
    values. String values containing ``{`` are passed through
    :func:`_expand_date_placeholders`. Keys not declared on ``config_cls`` are
    silently dropped, so unrelated YAML keys won't raise.
    """
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
    """Return a plain dict of a config dataclass (for logging and reconstruction)."""
    return asdict(config)
