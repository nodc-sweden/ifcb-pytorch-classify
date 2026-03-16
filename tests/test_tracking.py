from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from ifcb_classify.tracking import create_tracker
from ifcb_classify.tracking.csv_tracker import CsvTracker


def test_create_csv_tracker():
    tracker = create_tracker("csv", output_dir="/tmp/test")
    assert isinstance(tracker, CsvTracker)


def test_create_none_tracker():
    tracker = create_tracker("none")
    # Should not raise on any method
    tracker.begin_run("test", {"lr": 0.01})
    tracker.log_metrics({"acc": 0.9}, step=1)
    tracker.log_confusion_matrix(np.eye(3), ["A", "B", "C"], step=1)
    tracker.end_run()


def test_create_unknown_tracker_raises():
    with pytest.raises(ValueError, match="Unknown tracker type"):
        create_tracker("invalid")


def test_csv_tracker_writes_metrics(tmp_path):
    tracker = CsvTracker(output_dir=str(tmp_path))
    tracker.begin_run("run1", {"model": "resnet50"})
    tracker.log_metrics({"acc": 0.8, "loss": 0.5}, step=1)
    tracker.log_metrics({"acc": 0.9, "loss": 0.3}, step=2)

    csv_path = tmp_path / "run1.csv"
    assert csv_path.exists()
    df = pd.read_csv(csv_path)
    assert len(df) == 2
    assert "acc" in df.columns
    assert "model" in df.columns


def test_csv_tracker_writes_confusion_matrix(tmp_path):
    tracker = CsvTracker(output_dir=str(tmp_path))
    tracker.begin_run("run1", {})
    cm = np.array([[5, 1], [2, 8]])
    tracker.log_confusion_matrix(cm, ["A", "B"], step=3)

    cm_path = tmp_path / "confusion_matrix" / "run1" / "run1_3.csv"
    assert cm_path.exists()


def test_mlflow_tracker_log_confusion_matrix_cleans_up(tmp_path):
    mock_mlflow = MagicMock()
    mock_mlflow.log_artifact = MagicMock()

    with patch.dict("sys.modules", {"mlflow": mock_mlflow}):
        from ifcb_classify.tracking.mlflow_tracker import MlflowTracker

        tracker = MlflowTracker.__new__(MlflowTracker)
        tracker._mlflow = mock_mlflow

        cm = np.array([[5, 1], [2, 8]])
        tracker.log_confusion_matrix(cm, ["A", "B"], step=1)

        mock_mlflow.log_artifact.assert_called_once()
        # The temp file should have been cleaned up
        artifact_path = mock_mlflow.log_artifact.call_args[0][0]
        import os
        assert not os.path.exists(artifact_path)


def test_wandb_tracker_log_confusion_matrix():
    mock_wandb = MagicMock()
    mock_wandb.plot.confusion_matrix = MagicMock(return_value="mock_plot")

    with patch.dict("sys.modules", {"wandb": mock_wandb}):
        from ifcb_classify.tracking.wandb_tracker import WandbTracker

        tracker = WandbTracker.__new__(WandbTracker)
        tracker._wandb = mock_wandb

        cm = np.array([[5, 1], [2, 8]])
        tracker.log_confusion_matrix(cm, ["A", "B"], step=1)

        mock_wandb.plot.confusion_matrix.assert_called_once()
        call_kwargs = mock_wandb.plot.confusion_matrix.call_args
        # Verify actual data is passed (not None)
        assert call_kwargs[1]["y_true"] is not None
        assert call_kwargs[1]["preds"] is not None
        assert len(call_kwargs[1]["y_true"]) == 16  # 5+1+2+8
        assert len(call_kwargs[1]["preds"]) == 16
