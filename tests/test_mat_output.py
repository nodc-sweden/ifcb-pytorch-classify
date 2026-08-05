import numpy as np
import pandas as pd
import pytest
from scipy.io import loadmat

from ifcb_classify.mat_output import write_class_scores_mat


def _write(tmp_path):
    scores = np.array([[0.97, 0.02, 0.01], [0.20, 0.36, 0.44]], dtype=np.float64)
    class_labels = ["Guinardia_delicatula", "Cryptomonadales", "Scrippsiella_group"]
    roi_numbers = np.array([3, 4], dtype=np.int32)
    class_name_auto = ["Guinardia_delicatula", "Scrippsiella_group"]
    class_name = ["Guinardia_delicatula", "unclassified"]  # 2nd ROI below threshold
    out = tmp_path / "D20230314T003836_IFCB134_class_v1.mat"
    write_class_scores_mat(out, scores, class_labels, roi_numbers, class_name_auto, class_name, "MyClassifier V6")
    return out, scores, class_labels, class_name_auto, class_name


def test_mat_rejects_roi_numbers_beyond_uint16(tmp_path):
    """roinum is stored as uint16; out-of-range values must not wrap silently.

    A cast alone turned ROI 70000 into 4464, which misassociates every
    downstream biovolume and count with no error and no warning.
    """
    scores = np.array([[0.5, 0.5]], dtype=np.float64)
    out = tmp_path / "big_class_v1.mat"

    with pytest.raises(ValueError, match="65535"):
        write_class_scores_mat(
            out, scores, ["A", "B"], np.array([70000], dtype=np.int64),
            ["A"], ["A"], "clf",
        )
    assert not out.exists()


def test_mat_accepts_the_largest_representable_roi_number(tmp_path):
    scores = np.array([[0.5, 0.5]], dtype=np.float64)
    out = tmp_path / "edge_class_v1.mat"

    write_class_scores_mat(
        out, scores, ["A", "B"], np.array([65535], dtype=np.int64), ["A"], ["A"], "clf",
    )

    assert loadmat(out, squeeze_me=True)["roinum"] == 65535


def test_mat_rejects_negative_roi_numbers(tmp_path):
    scores = np.array([[0.5, 0.5]], dtype=np.float64)
    with pytest.raises(ValueError, match="between 0 and 65535"):
        write_class_scores_mat(
            tmp_path / "neg_class_v1.mat", scores, ["A", "B"],
            np.array([-1], dtype=np.int64), ["A"], ["A"], "clf",
        )


def test_mat_scores_roundtrip_via_pyifcb_logic(tmp_path):
    out, scores, class_labels, _, _ = _write(tmp_path)
    mat = loadmat(out, squeeze_me=True)
    # class2useTB carries a trailing "unclassified" that pyifcb strips on read.
    assert list(mat["class2useTB"]) == [*class_labels, "unclassified"]

    # Reproduce pyifcb's _class_scores_v1: drop the last label, index by roinum.
    read_labels = [str(x) for x in mat["class2useTB"][:-1]]
    df = pd.DataFrame(mat["TBscores"], columns=read_labels).set_index(mat["roinum"])
    assert list(df.columns) == class_labels
    assert df.index.tolist() == [3, 4]
    np.testing.assert_allclose(df.to_numpy(), scores)


def test_mat_carries_irfcb_fields(tmp_path):
    out, _, _, class_name_auto, class_name = _write(tmp_path)
    mat = loadmat(out, squeeze_me=True)
    # Fields iRfcb reads from a v1 class .mat
    assert [str(x) for x in mat["TBclass"]] == class_name_auto
    assert [str(x) for x in mat["TBclass_above_threshold"]] == class_name
    assert str(mat["classifierName"]) == "MyClassifier V6"
    # roinum is written as uint16 (matching iRfcb's own writer)
    assert loadmat(out)["roinum"].dtype == np.uint16


def test_mat_cell_count_written_when_provided(tmp_path):
    scores = np.array([[0.9, 0.1], [0.2, 0.8]], dtype=np.float64)
    out = tmp_path / "c.mat"
    write_class_scores_mat(
        out, scores, ["A", "B"], np.array([1, 2]), ["A", "B"], ["A", "B"], "clf",
        cell_counts=np.array([5, -1], dtype=np.int32),
    )
    mat = loadmat(out, squeeze_me=True)
    np.testing.assert_array_equal(mat["cell_count"], [5, -1])

    # Omitted by default (backward compatible)
    out2 = tmp_path / "nc.mat"
    write_class_scores_mat(out2, scores, ["A", "B"], np.array([1, 2]), ["A", "B"], ["A", "B"], "clf")
    assert "cell_count" not in loadmat(out2)


def test_mat_length_mismatch(tmp_path):
    scores = np.array([[0.9, 0.1]])
    with pytest.raises(ValueError, match="ROI numbers"):
        write_class_scores_mat(tmp_path / "bad.mat", scores, ["A", "B"], np.array([1, 2]), ["A"], ["A"], "clf")


def test_mat_contains_no_struct_variables(tmp_path):
    """iRfcb's native reader aborts the whole file on any struct variable, so the
    .mat must carry only cell/char/numeric arrays. A struct loads back through
    scipy with a named (kind 'V') dtype — assert none does, including the full
    field set that a chain-counting run writes."""
    scores = np.array([[0.9, 0.1], [0.2, 0.8]], dtype=np.float64)
    out = tmp_path / "s.mat"
    write_class_scores_mat(
        out, scores, ["A", "B"], np.array([1, 2]), ["A", "B"], ["A", "B"], "clf",
        cell_counts=np.array([5, -1], dtype=np.int32),
    )
    for name, value in loadmat(out).items():
        if name.startswith("__"):
            continue
        assert value.dtype.names is None, f"{name} is a struct; iRfcb cannot read this file"
