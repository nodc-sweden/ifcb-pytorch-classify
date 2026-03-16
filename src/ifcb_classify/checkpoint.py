import logging
from pathlib import Path

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class CheckpointManager:
    def __init__(self, output_dir: str, metric_name: str = "weighted_f1", mode: str = "max"):
        self._output_dir = Path(output_dir)
        self._metric_name = metric_name
        self._mode = mode
        self._best_value = float("-inf") if mode == "max" else float("inf")
        self._best_path: Path | None = None

    def maybe_save(
        self,
        model: nn.Module,
        metric_value: float,
        run_name: str,
        epoch: int,
        class_names: list[str],
        config: dict,
    ) -> bool:
        improved = (
            metric_value > self._best_value if self._mode == "max" else metric_value < self._best_value
        )
        if not improved:
            return False

        self._best_value = metric_value
        self._output_dir.mkdir(parents=True, exist_ok=True)

        if self._best_path and self._best_path.exists():
            self._best_path.unlink()

        new_path = self._output_dir / f"{run_name}_best.pt"
        tmp_path = new_path.with_suffix(".pt.tmp")
        torch.save(
            {
                "state_dict": model.state_dict(),
                "epoch": epoch,
                "metric_name": self._metric_name,
                "metric_value": metric_value,
                "class_names": class_names,
                "config": config,
            },
            tmp_path,
        )

        if self._best_path and self._best_path.exists():
            self._best_path.unlink()
        tmp_path.rename(new_path)
        self._best_path = new_path
        logger.info("Saved best model: %s (epoch %d, %s=%.4f)", self._best_path.name, epoch, self._metric_name, metric_value)
        return True


def load_checkpoint(path: str | Path, model_name: str | None = None, classes_path: str | None = None) -> dict:
    path = Path(path)
    try:
        data = torch.load(path, map_location="cpu", weights_only=True)
    except Exception:
        logger.warning(
            "Safe load failed for %s — falling back to unsafe load. "
            "Only load checkpoints from trusted sources.",
            path,
        )
        data = torch.load(path, map_location="cpu", weights_only=False)

    # Our pipeline checkpoints have "state_dict" and "config" keys
    if isinstance(data, dict) and "state_dict" in data and "config" in data:
        return data

    # Legacy checkpoint: raw state_dict (just weight tensors)
    state_dict = data
    class_names = _load_class_names(path, classes_path)
    resolved_model = model_name or _guess_model_name(state_dict)

    logger.info("Legacy checkpoint detected — model=%s, %d classes", resolved_model, len(class_names))

    return {
        "state_dict": state_dict,
        "class_names": class_names,
        "config": {
            "model": resolved_model,
            "image_width": 224,
            "image_height": 224,
            "transform": "dataset_squarepad",
        },
    }


def _load_class_names(checkpoint_path: Path, classes_path: str | None) -> list[str]:
    if classes_path:
        p = Path(classes_path)
    else:
        p = checkpoint_path.parent / "classes.txt"

    if not p.exists():
        raise FileNotFoundError(
            f"No classes.txt found at {p}. Supply --classes pointing to a class list file."
        )

    return [line.strip() for line in p.read_text().splitlines() if line.strip()]


def _guess_model_name(state_dict: dict) -> str:
    keys = set(state_dict.keys())
    if any(k.startswith("layer4") for k in keys) and "fc.weight" in keys:
        return "resnet50"
    if any(k.startswith("features") for k in keys) and "classifier.1.weight" in keys:
        return "efficientnet_b0"
    return "resnet50"
