from dataclasses import dataclass


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
