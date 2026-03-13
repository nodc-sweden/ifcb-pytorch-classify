import tempfile
from pathlib import Path

import numpy as np
import pandas as pd


class MlflowTracker:
    def __init__(self, tracking_uri: str | None = None, experiment_name: str = "ifcb-classify"):
        import mlflow

        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment_name)
        self._mlflow = mlflow

    def begin_run(self, run_name: str, params: dict) -> None:
        self._mlflow.start_run(run_name=run_name)
        self._mlflow.log_params(params)

    def log_metrics(self, metrics: dict, step: int) -> None:
        self._mlflow.log_metrics(metrics, step=step)

    def log_confusion_matrix(self, cm: np.ndarray, class_names: list[str], step: int) -> None:
        df = pd.DataFrame(cm, index=class_names, columns=class_names)
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w") as f:
            df.to_csv(f)
            self._mlflow.log_artifact(f.name, artifact_path=f"confusion_matrices/epoch_{step}")

    def end_run(self) -> None:
        self._mlflow.end_run()
