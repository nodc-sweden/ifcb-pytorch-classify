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

        self._best_path = self._output_dir / f"{run_name}_best.pt"
        torch.save(
            {
                "state_dict": model.state_dict(),
                "epoch": epoch,
                "metric_name": self._metric_name,
                "metric_value": metric_value,
                "class_names": class_names,
                "config": config,
            },
            self._best_path,
        )
        logger.info("Saved best model: %s (epoch %d, %s=%.4f)", self._best_path.name, epoch, self._metric_name, metric_value)
        return True


def load_checkpoint(path: str | Path) -> dict:
    return torch.load(path, map_location="cpu", weights_only=False)
