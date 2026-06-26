"""Experiment tracking backends.

All backends implement the :class:`ExperimentTracker` protocol (see
:mod:`ifcb_classify.tracking.base`), so training code logs through one interface
regardless of destination. :func:`create_tracker` is the factory the training
loop calls. The optional ``mlflow``/``wandb`` backends import their heavy
dependencies lazily, so they only need to be installed if actually selected.
"""

from ifcb_classify.tracking.base import ExperimentTracker
from ifcb_classify.tracking.csv_tracker import CsvTracker


def create_tracker(tracker_type: str, **kwargs) -> ExperimentTracker:
    """Construct the tracker named ``tracker_type``.

    Supported types: ``"csv"`` (default file logging), ``"mlflow"``, ``"wandb"``
    and ``"none"`` (a no-op). Relevant ``kwargs`` (e.g. ``output_dir``,
    ``mlflow_uri``, ``wandb_project``, ``experiment_name``) are forwarded to the
    chosen backend. Raises ``ValueError`` for an unknown type.
    """
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
    """No-op tracker used when ``tracker="none"``; ignores everything logged."""

    def begin_run(self, run_name, params):
        pass

    def log_metrics(self, metrics, step):
        pass

    def log_confusion_matrix(self, cm, class_names, step):
        pass

    def end_run(self):
        pass
