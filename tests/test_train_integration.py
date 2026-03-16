from pathlib import Path

import pytest

from ifcb_classify.config import TrainConfig
from ifcb_classify.train import train_main

FIXTURES = Path(__file__).parent / "fixtures" / "training_data"


@pytest.mark.slow
def test_train_one_epoch(tmp_path):
    config = TrainConfig(
        data_dir=str(FIXTURES),
        model="resnet18",
        transform="dataset_squarepad",
        image_width=32,
        image_height=32,
        epochs=1,
        batch_size=8,
        lr=0.01,
        output_dir=str(tmp_path),
        tracker="csv",
        val_split=0.3,
        num_workers=0,
    )
    train_main(config)

    # Check that a checkpoint was saved
    checkpoints = list(tmp_path.glob("*.pt"))
    assert len(checkpoints) == 1

    # Check that CSV metrics were written
    csvs = list(tmp_path.glob("*.csv"))
    assert len(csvs) >= 1

    # Check that thresholds JSON was written
    jsons = list(tmp_path.glob("*thresholds*.json"))
    assert len(jsons) == 1

    # Check that classes.txt was written
    classes_files = list(tmp_path.glob("*classes.txt"))
    assert len(classes_files) == 1
