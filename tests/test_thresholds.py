import json

import numpy as np

from ifcb_classify.thresholds import load_thresholds_json, save_thresholds_and_metrics


def test_save_thresholds_and_metrics(tmp_path):
    class_names = ["ClassA", "ClassB", "ClassC"]
    thresholds = np.array([0.5, 0.3, 0.7])
    class_metrics = {
        0: {"class_name": "ClassA", "threshold": 0.5, "f1": 0.9, "precision": 0.85, "recall": 0.95, "support": 100},
        1: {"class_name": "ClassB", "threshold": 0.3, "f1": 0.8, "precision": 0.75, "recall": 0.85, "support": 80},
        2: {"class_name": "ClassC", "threshold": 0.7, "f1": 0.7, "precision": 0.65, "recall": 0.75, "support": 60},
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


def test_load_thresholds_json(tmp_path):
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


def test_load_thresholds_json_partial(tmp_path):
    data = {
        "class_metrics": {
            "0": {"threshold": 0.5},
        }
    }
    json_path = tmp_path / "thresholds.json"
    with open(json_path, "w") as f:
        json.dump(data, f)

    thresholds = load_thresholds_json(json_path, ["A", "B", "C"])
    assert thresholds[0] == 0.5
    assert np.isnan(thresholds[1])
    assert np.isnan(thresholds[2])
