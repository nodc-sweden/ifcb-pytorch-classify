import json

import h5py
import numpy as np

from ifcb_classify.hdf5_output import write_class_scores


def test_write_class_scores(tmp_path):
    n_rois, n_classes = 10, 3
    scores = np.random.rand(n_rois, n_classes).astype(np.float64)
    # Normalise to make them look like probabilities
    scores /= scores.sum(axis=1, keepdims=True)
    class_labels = ["ClassA", "ClassB", "ClassC"]
    roi_numbers = np.arange(1, n_rois + 1, dtype=np.int64)
    thresholds = np.array([0.5, 0.3, np.nan])

    output_path = tmp_path / "test_class_v3.h5"
    write_class_scores(output_path, scores, class_labels, roi_numbers, "test_model", thresholds)

    with h5py.File(output_path, "r") as f:
        assert set(f.keys()) == {
            "output_scores",
            "class_labels",
            "roi_numbers",
            "classifier_name",
            "class_name_auto",
            "class_name",
            "thresholds",
        }
        assert f["output_scores"].shape == (n_rois, n_classes)
        assert f["output_scores"].dtype == np.float64
        assert len(f["class_labels"]) == n_classes
        assert len(f["roi_numbers"]) == n_rois
        assert len(f["class_name_auto"]) == n_rois
        assert len(f["class_name"]) == n_rois
        assert len(f["thresholds"]) == n_classes

        # Verify threshold logic: class C has NaN threshold, so all should be classified
        # class A (threshold 0.5) and B (threshold 0.3) may produce "unclassified"
        class_names = [x.decode() if isinstance(x, bytes) else x for x in f["class_name"][:]]
        for name in class_names:
            assert name in ("ClassA", "ClassB", "ClassC", "unclassified")

        # No cell_count dataset when cell_counts is not provided (backward compatible)
        assert "cell_count" not in f


def test_write_class_scores_with_cell_counts(tmp_path):
    n_rois = 4
    scores = np.array([[0.9, 0.1], [0.2, 0.8], [0.7, 0.3], [0.4, 0.6]])
    class_labels = ["Skeletonema", "Other"]
    roi_numbers = np.arange(1, n_rois + 1, dtype=np.int32)
    thresholds = np.array([np.nan, np.nan])
    cell_counts = np.array([5, -1, 8, -1], dtype=np.int32)
    models = {"Skeletonema": {"weights": "best.pt", "iou": 0.3, "conf": 0.25}}

    output_path = tmp_path / "with_counts.h5"
    write_class_scores(
        output_path, scores, class_labels, roi_numbers, "test_model", thresholds,
        cell_counts=cell_counts, cell_counter_models=models,
    )

    with h5py.File(output_path, "r") as f:
        assert "cell_count" in f
        np.testing.assert_array_equal(f["cell_count"][:], [5, -1, 8, -1])
        assert f["cell_count"].dtype == np.int32
        assert json.loads(f.attrs["cell_counter_models"]) == models


def test_write_class_scores_cell_counts_length_mismatch(tmp_path):
    import pytest

    scores = np.array([[0.9, 0.1], [0.2, 0.8]])
    with pytest.raises(ValueError, match="chain counts"):
        write_class_scores(
            tmp_path / "bad.h5", scores, ["A", "B"], np.array([1, 2], dtype=np.int32),
            "m", np.array([np.nan, np.nan]), cell_counts=np.array([1], dtype=np.int32),
        )
