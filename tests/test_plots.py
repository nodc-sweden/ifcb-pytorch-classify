import numpy as np
import pytest

from ifcb_classify.plots import generate_evaluation_plots


@pytest.fixture
def sample_epoch_metrics():
    return [
        {"train_loss": 2.0, "val_loss": 2.1, "train_accuracy": 0.3, "val_accuracy": 0.28},
        {"train_loss": 1.5, "val_loss": 1.6, "train_accuracy": 0.5, "val_accuracy": 0.45},
        {"train_loss": 1.0, "val_loss": 1.2, "train_accuracy": 0.7, "val_accuracy": 0.62},
    ]


@pytest.fixture
def sample_class_names():
    return ["classA", "classB", "classC", "classD"]


@pytest.fixture
def sample_confusion_matrix():
    return np.array([
        [10, 2, 0, 1],
        [1, 15, 3, 0],
        [0, 1, 12, 2],
        [2, 0, 1, 8],
    ])


@pytest.fixture
def sample_class_metrics():
    return {
        "classA": {"class_name": "classA", "f1": 0.85, "precision": 0.80, "recall": 0.90, "support": 50, "threshold": 0.4},
        "classB": {"class_name": "classB", "f1": 0.75, "precision": 0.70, "recall": 0.80, "support": 30, "threshold": 0.5},
        "classC": {"class_name": "classC", "f1": 0.90, "precision": 0.92, "recall": 0.88, "support": 80, "threshold": 0.35},
        "classD": {"class_name": "classD", "f1": 0.60, "precision": 0.55, "recall": 0.65, "support": 15, "threshold": 0.6},
    }


def test_generate_static_plots(tmp_path, sample_epoch_metrics, sample_confusion_matrix, sample_class_names, sample_class_metrics):
    paths = generate_evaluation_plots(
        output_dir=tmp_path,
        run_name="test-run",
        epoch_metrics=sample_epoch_metrics,
        confusion_matrix=sample_confusion_matrix,
        class_names=sample_class_names,
        class_metrics=sample_class_metrics,
    )

    filenames = {p.name for p in paths}
    assert "training_curves.png" in filenames
    assert "per_class_f1.png" in filenames
    assert "precision_recall_scatter.png" in filenames
    assert "class_support_histogram.png" in filenames
    assert "top_confused_pairs.png" in filenames

    for p in paths:
        assert p.exists()
        assert p.stat().st_size > 0


def test_generate_plots_without_class_metrics(tmp_path, sample_epoch_metrics, sample_confusion_matrix, sample_class_names):
    paths = generate_evaluation_plots(
        output_dir=tmp_path,
        run_name="test-run",
        epoch_metrics=sample_epoch_metrics,
        confusion_matrix=sample_confusion_matrix,
        class_names=sample_class_names,
        class_metrics=None,
    )

    filenames = {p.name for p in paths}
    assert "training_curves.png" in filenames
    assert "top_confused_pairs.png" in filenames
    assert "per_class_f1.png" not in filenames
    assert "precision_recall_scatter.png" not in filenames
    assert "class_support_histogram.png" not in filenames


def test_generate_plots_empty_epoch_metrics(tmp_path, sample_confusion_matrix, sample_class_names):
    paths = generate_evaluation_plots(
        output_dir=tmp_path,
        run_name="test-run",
        epoch_metrics=[],
        confusion_matrix=sample_confusion_matrix,
        class_names=sample_class_names,
    )

    filenames = {p.name for p in paths}
    assert "training_curves.png" not in filenames
    assert "top_confused_pairs.png" in filenames


def test_generate_plots_large_class_count(tmp_path):
    """Verify plots work at realistic scale (200 classes)."""
    num_classes = 200
    class_names = [f"class_{i:03d}" for i in range(num_classes)]
    cm = np.random.randint(0, 10, size=(num_classes, num_classes))
    np.fill_diagonal(cm, np.random.randint(50, 200, size=num_classes))

    class_metrics = {
        name: {
            "class_name": name,
            "f1": np.random.uniform(0.3, 1.0),
            "precision": np.random.uniform(0.3, 1.0),
            "recall": np.random.uniform(0.3, 1.0),
            "support": np.random.randint(10, 500),
            "threshold": np.random.uniform(0.2, 0.8),
        }
        for name in class_names
    }

    epoch_metrics = [
        {"train_loss": 2.0 - i * 0.1, "val_loss": 2.1 - i * 0.08, "train_accuracy": 0.3 + i * 0.05, "val_accuracy": 0.28 + i * 0.04}
        for i in range(10)
    ]

    paths = generate_evaluation_plots(
        output_dir=tmp_path,
        run_name="large-run",
        epoch_metrics=epoch_metrics,
        confusion_matrix=cm,
        class_names=class_names,
        class_metrics=class_metrics,
    )

    assert len(paths) >= 5
    for p in paths:
        assert p.exists()
        assert p.stat().st_size > 0


def test_interactive_plots_with_plotly(tmp_path, sample_epoch_metrics, sample_confusion_matrix, sample_class_names, sample_class_metrics):
    pytest.importorskip("plotly")

    paths = generate_evaluation_plots(
        output_dir=tmp_path,
        run_name="test-run",
        epoch_metrics=sample_epoch_metrics,
        confusion_matrix=sample_confusion_matrix,
        class_names=sample_class_names,
        class_metrics=sample_class_metrics,
    )

    filenames = {p.name for p in paths}
    assert "confusion_matrix_interactive.html" in filenames
    assert "per_class_metrics_table.html" in filenames

    for p in paths:
        if p.suffix == ".html":
            content = p.read_text()
            assert len(content) > 100


def test_confused_pairs_zero_confusion(tmp_path):
    """No confused pairs plot when confusion matrix is diagonal."""
    cm = np.diag([10, 20, 30])
    class_names = ["a", "b", "c"]

    paths = generate_evaluation_plots(
        output_dir=tmp_path,
        run_name="diag-run",
        epoch_metrics=[],
        confusion_matrix=cm,
        class_names=class_names,
    )

    filenames = {p.name for p in paths}
    assert "top_confused_pairs.png" not in filenames
