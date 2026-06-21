from dataclasses import dataclass, field

DEFAULT_IOU = 0.3
DEFAULT_CONF = 0.25


@dataclass(frozen=True)
class ChainModelSpec:
    """A single per-taxon detector: weights plus NMS/confidence thresholds."""

    weights: str
    iou: float = DEFAULT_IOU
    conf: float = DEFAULT_CONF


@dataclass(frozen=True)
class ChainCountingConfig:
    """Inference-time chain counting: which classifier labels get counted, and how.

    Built from the ``chain_counting`` block of an inference YAML config. The keys
    of ``models`` must match the classifier's output labels exactly; multiple
    labels may point at the same weights (e.g. several Thalassiosira species and
    the genus-level class sharing one detector).
    """

    enabled: bool = False
    models: dict[str, ChainModelSpec] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict) -> "ChainCountingConfig":
        enabled = bool(data.get("enabled", False))
        default_iou = float(data.get("iou", DEFAULT_IOU))
        default_conf = float(data.get("conf", DEFAULT_CONF))

        models: dict[str, ChainModelSpec] = {}
        for name, spec in (data.get("models") or {}).items():
            if isinstance(spec, str):
                spec = {"weights": spec}
            weights = spec.get("weights")
            if not weights:
                raise ValueError(f"chain_counting model '{name}' is missing 'weights'")
            iou = float(spec.get("iou", default_iou))
            conf = float(spec.get("conf", default_conf))
            if not (0.0 <= iou <= 1.0):
                raise ValueError(f"chain_counting model '{name}': iou must be in [0, 1], got {iou}")
            if not (0.0 <= conf <= 1.0):
                raise ValueError(f"chain_counting model '{name}': conf must be in [0, 1], got {conf}")
            models[name] = ChainModelSpec(weights=weights, iou=iou, conf=conf)

        if enabled and not models:
            raise ValueError("chain_counting is enabled but no models are configured")

        return cls(enabled=enabled, models=models)


@dataclass(frozen=True)
class ChainTrainConfig:
    """Configuration for training a per-taxon YOLO chain-counting detector.

    ``data`` may point either at a YOLO ``data.yaml`` file or at a dataset
    directory containing one (``data.local.yaml`` is preferred over
    ``data.yaml`` when both are present).
    """

    class_name: str = ""
    data: str = ""
    model: str = "yolo11n.pt"
    epochs: int = 100
    imgsz: int = 640
    batch: int = 16
    device: str = "cpu"
    patience: int = 20
    project: str = "output/chains"
    name: str | None = None

    def __post_init__(self):
        if not self.class_name:
            raise ValueError("class_name is required")
        if not self.data:
            raise ValueError("data (path to YOLO data.yaml or dataset directory) is required")
        if self.epochs < 1:
            raise ValueError(f"epochs must be >= 1, got {self.epochs}")
        if self.imgsz < 1:
            raise ValueError(f"imgsz must be >= 1, got {self.imgsz}")
        if self.batch < 1:
            raise ValueError(f"batch must be >= 1, got {self.batch}")
        if self.patience < 0:
            raise ValueError(f"patience must be >= 0, got {self.patience}")


@dataclass(frozen=True)
class ChainEvalConfig:
    """Configuration for validating a chain detector's counts against manual counts.

    ``counts_csv`` must have a filename column and an integer count column;
    images are resolved by name under ``images``. Running the same ``weights``
    against several species' test sets is how you check that one genus-level
    detector generalises across species.
    """

    weights: str = ""
    images: str = ""
    counts_csv: str = ""
    conf: float = 0.25
    ious: tuple[float, ...] = (0.3, 0.5, 0.7)
    limit: int = 0
    output: str | None = None
    file_col: str = "file_name"
    count_col: str = "cell_count"

    def __post_init__(self):
        if not self.weights:
            raise ValueError("weights is required")
        if not self.images:
            raise ValueError("images (directory of test images) is required")
        if not self.counts_csv:
            raise ValueError("counts_csv (CSV of manual counts) is required")
        if not (0.0 <= self.conf <= 1.0):
            raise ValueError(f"conf must be in [0, 1], got {self.conf}")
        if not self.ious:
            raise ValueError("at least one iou value is required")
        for iou in self.ious:
            if not (0.0 <= iou <= 1.0):
                raise ValueError(f"iou values must be in [0, 1], got {iou}")
        if self.limit < 0:
            raise ValueError(f"limit must be >= 0, got {self.limit}")
