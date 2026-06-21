import logging
from pathlib import Path

from ifcb_classify.chains.config import ChainTrainConfig

logger = logging.getLogger(__name__)

_CHAINS_EXTRA_HINT = (
    "Chain counting requires the optional 'chains' extra. "
    'Install it with: uv pip install -e ".[chains]"'
)


def resolve_data_yaml(data: str) -> Path:
    """Resolve a YOLO data config from a file path or a dataset directory.

    A directory is searched for ``data.local.yaml`` first, then ``data.yaml``.
    """
    path = Path(data)
    if path.is_dir():
        for candidate in ("data.local.yaml", "data.yaml"):
            if (path / candidate).exists():
                return path / candidate
        raise FileNotFoundError(f"No data.local.yaml or data.yaml found in directory: {path}")
    if path.is_file():
        return path
    raise FileNotFoundError(f"Data config not found: {path}")


def train_chain_detector(config: ChainTrainConfig) -> Path:
    """Train a single-class YOLO detector for one chain-forming taxon.

    Returns the path to the best checkpoint (``best.pt``).
    """
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise ImportError(_CHAINS_EXTRA_HINT) from exc

    data_yaml = resolve_data_yaml(config.data)
    name = config.name or f"chains_{config.class_name}_{Path(config.model).stem}"

    logger.info(
        "Training chain detector for %s: model=%s epochs=%d imgsz=%d batch=%d device=%s",
        config.class_name,
        config.model,
        config.epochs,
        config.imgsz,
        config.batch,
        config.device,
    )
    logger.info("Data config: %s", data_yaml)

    model = YOLO(config.model)
    results = model.train(
        data=str(data_yaml),
        epochs=config.epochs,
        imgsz=config.imgsz,
        batch=config.batch,
        device=config.device,
        patience=config.patience,
        project=config.project,
        name=name,
    )

    best = Path(results.save_dir) / "weights" / "best.pt"
    logger.info("Chain detector trained. Best weights: %s", best)
    return best
