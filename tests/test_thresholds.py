import json

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from ifcb_classify.thresholds import compute_optimal_thresholds, load_thresholds_json, save_thresholds_and_metrics


def test_save_thresholds_and_metrics(tmp_path):
    class_names = ["ClassA", "ClassB", "ClassC"]
    thresholds = np.array([0.5, 0.3, 0.7])
    class_metrics = {
        "ClassA": {"class_name": "ClassA", "threshold": 0.5, "f1": 0.9, "precision": 0.85, "recall": 0.95, "support": 100},
        "ClassB": {"class_name": "ClassB", "threshold": 0.3, "f1": 0.8, "precision": 0.75, "recall": 0.85, "support": 80},
        "ClassC": {"class_name": "ClassC", "threshold": 0.7, "f1": 0.7, "precision": 0.65, "recall": 0.75, "support": 60},
    }

    json_path = save_thresholds_and_metrics(
        str(tmp_path), "run1", best_epoch=10, class_names=class_names,
        thresholds=thresholds, class_metrics=class_metrics,
    )

    assert json_path.exists()
    with open(json_path) as f:
        data = json.load(f)
    assert data["num_classes"] == 3
    assert data["best_epoch"] == 10
    assert "macro_F1" in data
    assert "weighted_F1" in data

    classes_path = tmp_path / "run1_classes.txt"
    assert classes_path.exists()
    assert classes_path.read_text().splitlines() == class_names


def test_load_thresholds_json_by_class_name(tmp_path):
    data = {
        "class_metrics": {
            "A": {"threshold": 0.5},
            "B": {"threshold": 0.3},
            "C": {"threshold": 0.7},
        }
    }
    json_path = tmp_path / "thresholds.json"
    with open(json_path, "w") as f:
        json.dump(data, f)

    thresholds = load_thresholds_json(json_path, ["A", "B", "C"])
    np.testing.assert_array_almost_equal(thresholds, [0.5, 0.3, 0.7])


def test_load_thresholds_json_reordered_classes(tmp_path):
    data = {
        "class_metrics": {
            "A": {"threshold": 0.5},
            "B": {"threshold": 0.3},
            "C": {"threshold": 0.7},
        }
    }
    json_path = tmp_path / "thresholds.json"
    with open(json_path, "w") as f:
        json.dump(data, f)

    # Load with different class order — thresholds should follow names, not position
    thresholds = load_thresholds_json(json_path, ["C", "A", "B"])
    np.testing.assert_array_almost_equal(thresholds, [0.7, 0.5, 0.3])


def test_load_thresholds_json_legacy_int_keys(tmp_path):
    data = {
        "class_metrics": {
            "0": {"threshold": 0.5},
            "1": {"threshold": 0.3},
            "2": {"threshold": 0.7},
        }
    }
    json_path = tmp_path / "thresholds.json"
    with open(json_path, "w") as f:
        json.dump(data, f)

    thresholds = load_thresholds_json(json_path, ["A", "B", "C"])
    np.testing.assert_array_almost_equal(thresholds, [0.5, 0.3, 0.7])


def test_compute_optimal_thresholds():
    """Test threshold computation with a simple model that outputs known logits."""
    num_classes = 3
    model = nn.Linear(4, num_classes)

    torch.manual_seed(42)
    images = torch.randn(30, 4)
    labels = torch.tensor([0] * 10 + [1] * 10 + [2] * 10)
    dataset = TensorDataset(images, labels)
    loader = DataLoader(dataset, batch_size=10)

    class_names = ["ClassA", "ClassB", "ClassC"]
    thresholds, class_metrics = compute_optimal_thresholds(model, loader, torch.device("cpu"), class_names)

    assert thresholds.shape == (num_classes,)
    assert len(class_metrics) == num_classes
    for c in range(num_classes):
        name = class_names[c]
        assert 0.0 <= thresholds[c] <= 1.0
        m = class_metrics[name]
        assert m["class_name"] == name
        assert "threshold" in m
        assert "f1" in m
        assert "precision" in m
        assert "recall" in m
        assert m["support"] == 10


def test_load_thresholds_json_partial(tmp_path):
    data = {
        "class_metrics": {
            "A": {"threshold": 0.5},
        }
    }
    json_path = tmp_path / "thresholds.json"
    with open(json_path, "w") as f:
        json.dump(data, f)

    thresholds = load_thresholds_json(json_path, ["A", "B", "C"])
    assert thresholds[0] == 0.5
    assert np.isnan(thresholds[1])
    assert np.isnan(thresholds[2])
