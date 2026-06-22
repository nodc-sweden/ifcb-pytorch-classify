"""MLflow experiment-tracking backend (optional ``[mlflow]`` extra)."""

import os
import tempfile

import numpy as np
import pandas as pd


class MlflowTracker:
    """Log runs, params, metrics and confusion-matrix artifacts to MLflow.

    ``mlflow`` is imported in ``__init__`` so the dependency is only required
    when this backend is actually selected. Confusion matrices are written to a
    temp CSV and uploaded as run artifacts (then the temp file is removed).
    """

    def __init__(self, tracking_uri: str | None = None, experiment_name: str = "ifcb-classify"):
        """Point MLflow at ``tracking_uri`` (if given) and select the experiment."""
        import mlflow

        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment_name)
        self._mlflow = mlflow

    def begin_run(self, run_name: str, params: dict) -> None:
        """Start an MLflow run and log its hyperparameters."""
        self._mlflow.start_run(run_name=run_name)
        self._mlflow.log_params(params)

    def log_metrics(self, metrics: dict, step: int) -> None:
        """Log this epoch's metrics to the active MLflow run."""
        self._mlflow.log_metrics(metrics, step=step)

    def log_confusion_matrix(self, cm: np.ndarray, class_names: list[str], step: int) -> None:
        """Upload the labelled confusion matrix as a per-epoch CSV artifact."""
        df = pd.DataFrame(cm, index=class_names, columns=class_names)
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w") as f:
            df.to_csv(f)
            tmp_path = f.name
        try:
            self._mlflow.log_artifact(tmp_path, artifact_path=f"confusion_matrices/epoch_{step}")
        finally:
            os.unlink(tmp_path)

    def end_run(self) -> None:
        """Close the active MLflow run."""
        self._mlflow.end_run()
