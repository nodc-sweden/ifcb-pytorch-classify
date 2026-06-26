"""The structural interface every experiment-tracking backend implements."""

from typing import Protocol

import numpy as np


class ExperimentTracker(Protocol):
    """Protocol for experiment trackers (structural — no inheritance required).

    The training loop calls ``begin_run`` once, then ``log_metrics`` and
    ``log_confusion_matrix`` per epoch, then ``end_run``. Any class with these
    methods satisfies the protocol; see the CSV/MLflow/W&B implementations.
    """

    def begin_run(self, run_name: str, params: dict) -> None:
        """Start a run named ``run_name`` and record its hyperparameters."""
        ...

    def log_metrics(self, metrics: dict, step: int) -> None:
        """Log a flat ``{name: value}`` dict of metrics for epoch ``step``."""
        ...

    def log_confusion_matrix(self, cm: np.ndarray, class_names: list[str], step: int) -> None:
        """Log a ``(C, C)`` confusion matrix (``cm[true, pred]``) for epoch ``step``."""
        ...

    def end_run(self) -> None:
        """Finalise the current run (flush/close resources)."""
        ...
