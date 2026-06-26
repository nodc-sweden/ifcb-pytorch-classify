"""Weights & Biases experiment-tracking backend (optional ``[wandb]`` extra)."""

import numpy as np


class WandbTracker:
    """Log runs, config, metrics and confusion matrices to Weights & Biases.

    ``wandb`` is imported in ``__init__`` so the dependency is only required when
    this backend is selected. W&B's confusion-matrix plot wants per-sample
    ``(y_true, y_pred)`` lists, so the dense matrix is expanded back into pairs
    before logging.
    """

    def __init__(self, project: str = "ifcb-classify"):
        """Store the target W&B project; the run is created in ``begin_run``."""
        import wandb

        self._wandb = wandb
        self._project = project

    def begin_run(self, run_name: str, params: dict) -> None:
        """Initialise a W&B run under the project with the given config."""
        self._wandb.init(project=self._project, name=run_name, config=params)

    def log_metrics(self, metrics: dict, step: int) -> None:
        """Log this epoch's metrics to the active W&B run."""
        self._wandb.log(metrics, step=step)

    def log_confusion_matrix(self, cm: np.ndarray, class_names: list[str], step: int) -> None:
        """Log a W&B confusion-matrix plot for ``step``.

        W&B builds the plot from per-sample ``(y_true, y_pred)`` pairs, so the
        dense count matrix is expanded back into one pair per counted sample.
        """
        y_true = []
        y_pred = []
        for true_idx in range(cm.shape[0]):
            for pred_idx in range(cm.shape[1]):
                count = int(cm[true_idx, pred_idx])
                y_true.extend([true_idx] * count)
                y_pred.extend([pred_idx] * count)

        self._wandb.log({
            f"confusion_matrix_epoch_{step}": self._wandb.plot.confusion_matrix(
                probs=None,
                y_true=y_true,
                preds=y_pred,
                class_names=class_names,
            )
        })

    def end_run(self) -> None:
        """Finish and upload the active W&B run."""
        self._wandb.finish()
