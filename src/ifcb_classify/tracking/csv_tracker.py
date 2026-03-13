import os
from pathlib import Path

import numpy as np
import pandas as pd


class CsvTracker:
    def __init__(self, output_dir: str = "results"):
        self._output_dir = Path(output_dir)
        self._run_name = ""
        self._run_data: list[dict] = []
        self._params: dict = {}

    def begin_run(self, run_name: str, params: dict) -> None:
        self._run_name = run_name
        self._run_data = []
        self._params = params

    def log_metrics(self, metrics: dict, step: int) -> None:
        row = {"epoch": step, **metrics, **self._params}
        self._run_data.append(row)
        self._output_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(self._run_data).to_csv(self._output_dir / f"{self._run_name}.csv", index=False)

    def log_confusion_matrix(self, cm: np.ndarray, class_names: list[str], step: int) -> None:
        cm_dir = self._output_dir / "confusion_matrix" / self._run_name
        cm_dir.mkdir(parents=True, exist_ok=True)
        df = pd.DataFrame(cm, index=class_names, columns=class_names)
        df.to_csv(cm_dir / f"{self._run_name}_{step}.csv")

    def end_run(self) -> None:
        pass
