import numpy as np
import pandas as pd
import pytest

from ifcb_classify.csv_output import write_class_scores_csv


def test_write_class_scores_csv_dashboard_format(tmp_path):
    scores = np.array([[0.97, 0.02, 0.01], [0.20, 0.36, 0.44]], dtype=np.float64)
    class_labels = ["Guinardia_delicatula", "Cryptomonadales", "Scrippsiella_group"]
    roi_numbers = np.array([3, 4], dtype=np.int32)
    bin_lid = "D20230314T003836_IFCB134"

    out = tmp_path / "out.csv"
    write_class_scores_csv(out, scores, class_labels, roi_numbers, bin_lid)

    # Read back the way pyifcb/pandas would: pid index + one column per class.
    df = pd.read_csv(out)
    assert list(df.columns) == ["pid", *class_labels]
    # pid = {bin_lid}_{roi_number:05d}
    assert df["pid"].tolist() == ["D20230314T003836_IFCB134_00003", "D20230314T003836_IFCB134_00004"]
    np.testing.assert_allclose(df[class_labels].to_numpy(), scores)


def test_write_class_scores_csv_length_mismatch(tmp_path):
    scores = np.array([[0.9, 0.1]])
    with pytest.raises(ValueError, match="ROI numbers"):
        write_class_scores_csv(tmp_path / "bad.csv", scores, ["A", "B"], np.array([1, 2]), "LID")
