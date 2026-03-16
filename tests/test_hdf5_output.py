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
