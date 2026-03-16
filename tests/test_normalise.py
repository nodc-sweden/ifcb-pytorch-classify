import shutil
from pathlib import Path

from ifcb_classify.normalise import compute_dataset_stats

FIXTURES = Path(__file__).parent / "fixtures" / "training_data"


def test_compute_dataset_stats(tmp_path):
    data_dir = tmp_path / "training_data"
    shutil.copytree(FIXTURES, data_dir)

    mean, std = compute_dataset_stats(str(data_dir), transform_name="dataset_squarepad", width=32, height=32)
    assert 0.0 <= mean <= 1.0
    assert 0.0 < std <= 1.0


def test_compute_dataset_stats_fullpad(tmp_path):
    data_dir = tmp_path / "training_data"
    shutil.copytree(FIXTURES, data_dir)

    mean, std = compute_dataset_stats(str(data_dir), transform_name="dataset_fullpad", width=32, height=32)
    assert 0.0 <= mean <= 1.0
    assert 0.0 < std <= 1.0
