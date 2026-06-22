"""Saving and loading model checkpoints.

:class:`CheckpointManager` keeps only the single best checkpoint seen so far
during a run, judged by a monitored metric. :func:`load_checkpoint` reads a
checkpoint back for inference and transparently handles two formats:

* **Pipeline checkpoints** — a dict with ``state_dict``, ``class_names`` and
  ``config`` keys, as written by ``CheckpointManager``.
* **Legacy checkpoints** — a bare ``state_dict`` saved outside this pipeline.
  These need a sidecar ``classes.txt`` and the architecture is inferred (or
  supplied via ``model_name``). Loading them requires ``allow_unsafe=True``
  because they fall outside ``torch.load``'s safe ``weights_only`` mode.
"""

import logging
from pathlib import Path

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class CheckpointManager:
    """Persist the single best model of a run, keyed on a monitored metric.

    Only one ``{run_name}_best.pt`` file is kept on disk: each improvement
    overwrites the previous best. Saves go through a temp file + atomic rename so
    a crash mid-write can't leave a corrupt checkpoint.
    """

    def __init__(self, output_dir: str, metric_name: str = "weighted_f1", mode: str = "max"):
        """Configure where checkpoints go and which metric direction is "better".

        ``mode="max"`` treats higher metric values as improvements (e.g. F1);
        ``mode="min"`` treats lower as better (e.g. loss).
        """
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
        """Save the model iff ``metric_value`` improves on the best so far.

        Returns ``True`` when a new checkpoint was written, ``False`` otherwise.
        The saved payload bundles the weights with the epoch, metric, class names
        and training config so inference can reconstruct the model standalone.
        """
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


def load_checkpoint(
    path: str | Path,
    model_name: str | None = None,
    classes_path: str | None = None,
    allow_unsafe: bool = False,
) -> dict:
    """Load a checkpoint into a normalised ``{state_dict, class_names, config}`` dict.

    Pipeline checkpoints are returned as-is. A bare state-dict is treated as a
    legacy checkpoint: class names come from ``classes_path`` (or a sibling
    ``classes.txt``) and the architecture from ``model_name`` (or an inference
    based on the weight keys). ``allow_unsafe`` is required to fall back to
    ``torch.load(weights_only=False)`` when the safe load fails — only enable it
    for checkpoints you trust, since unpickling can execute arbitrary code.
    """
    path = Path(path)
    try:
        data = torch.load(path, map_location="cpu", weights_only=True)
    except Exception:
        if not allow_unsafe:
            raise RuntimeError(
                f"Safe load failed for {path}. If you trust this checkpoint, "
                "re-run with --allow-unsafe."
            )
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
    """Read class names (one per line) from ``classes_path`` or a sibling classes.txt.

    Raises :class:`FileNotFoundError` with a hint if no class list can be found,
    since a legacy checkpoint is unusable without one.
    """
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
    """Best-effort guess of the architecture from a legacy state-dict's key names.

    Recognises ResNet-50 and EfficientNetV2-S layouts; defaults to ``resnet50``.
    Pass ``--model-name`` explicitly when the guess is wrong.
    """
    keys = set(state_dict.keys())
    if any(k.startswith("layer4") for k in keys) and "fc.weight" in keys:
        return "resnet50"
    if any(k.startswith("features") for k in keys) and "classifier.1.weight" in keys:
        return "efficientnet_v2_s"
    return "resnet50"
