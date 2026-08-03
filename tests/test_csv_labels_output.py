import numpy as np
import pandas as pd
import pytest

from ifcb_classify.csv_labels_output import write_class_labels_csv


def test_labels_csv_columns_and_values(tmp_path):
    scores = np.array([[0.97, 0.02, 0.01], [0.20, 0.36, 0.44]], dtype=np.float64)
    roi_numbers = np.array([3, 4], dtype=np.int32)
    class_name_auto = ["Guinardia_delicatula", "Scrippsiella_group"]
    class_name = ["Guinardia_delicatula", "unclassified"]  # ROI 4 below threshold

    out = tmp_path / "D20230314T003836_IFCB134.csv"
    write_class_labels_csv(out, scores, roi_numbers, class_name_auto, class_name, "D20230314T003836_IFCB134")

    df = pd.read_csv(out)
    assert list(df.columns) == ["file_name", "class_name", "class_name_auto", "score"]
    assert df["file_name"].tolist() == [
        "D20230314T003836_IFCB134_00003.png",
        "D20230314T003836_IFCB134_00004.png",
    ]
    assert df["class_name"].tolist() == class_name
    assert df["class_name_auto"].tolist() == class_name_auto
    # score is the winning (max) score per ROI
    np.testing.assert_allclose(df["score"].to_numpy(), [0.97, 0.44])


def test_labels_csv_cell_count_optional(tmp_path):
    scores = np.array([[0.9, 0.1], [0.2, 0.8]])
    out = tmp_path / "c.csv"
    write_class_labels_csv(out, scores, np.array([1, 2]), ["A", "B"], ["A", "B"], "LID",
                           cell_counts=np.array([5, -1], dtype=np.int32))
    df = pd.read_csv(out)
    assert "cell_count" in df.columns
    assert df["cell_count"].tolist() == [5, -1]

    # Omitted by default
    out2 = tmp_path / "nc.csv"
    write_class_labels_csv(out2, scores, np.array([1, 2]), ["A", "B"], ["A", "B"], "LID")
    assert "cell_count" not in pd.read_csv(out2).columns


def test_labels_csv_length_mismatch(tmp_path):
    scores = np.array([[0.9, 0.1]])
    with pytest.raises(ValueError, match="class_name_auto"):
        write_class_labels_csv(tmp_path / "bad.csv", scores, np.array([1]), ["A", "B"], ["A"], "LID")
