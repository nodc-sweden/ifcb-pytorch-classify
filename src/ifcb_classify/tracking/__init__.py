"""Experiment tracking backends."""

from ifcb_classify.tracking.base import ExperimentTracker
from ifcb_classify.tracking.csv_tracker import CsvTracker


def create_tracker(tracker_type: str, **kwargs) -> ExperimentTracker:
    if tracker_type == "csv":
        return CsvTracker(output_dir=kwargs.get("output_dir", "results"))

    if tracker_type == "mlflow":
        from ifcb_classify.tracking.mlflow_tracker import MlflowTracker

        return MlflowTracker(
            tracking_uri=kwargs.get("mlflow_uri"),
            experiment_name=kwargs.get("experiment_name", "ifcb-classify"),
        )

    if tracker_type == "wandb":
        from ifcb_classify.tracking.wandb_tracker import WandbTracker

        return WandbTracker(project=kwargs.get("wandb_project", "ifcb-classify"))

    if tracker_type == "none":
        return _NullTracker()

    raise ValueError(f"Unknown tracker type: {tracker_type}")


class _NullTracker:
    def begin_run(self, run_name, params):
        pass

    def log_metrics(self, metrics, step):
        pass

    def log_confusion_matrix(self, cm, class_names, step):
        pass

    def end_run(self):
        pass
