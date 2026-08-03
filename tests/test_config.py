from datetime import UTC, datetime

import pytest
import yaml

from ifcb_classify.config import InferConfig, TrainConfig, load_config


def test_load_train_config(tmp_path):
    cfg = {"data_dir": "/data/V2", "model": "resnet18", "epochs": 10}
    yaml_path = tmp_path / "train.yaml"
    yaml_path.write_text(yaml.dump(cfg))

    config = load_config(yaml_path, TrainConfig)
    assert config.data_dir == "/data/V2"
    assert config.model == "resnet18"
    assert config.epochs == 10
    assert config.batch_size == 64  # default


def test_load_config_with_overrides(tmp_path):
    cfg = {"data_dir": "/data/V1", "epochs": 5}
    yaml_path = tmp_path / "train.yaml"
    yaml_path.write_text(yaml.dump(cfg))

    config = load_config(yaml_path, TrainConfig, overrides={"epochs": 20, "lr": 0.01})
    assert config.epochs == 20
    assert config.lr == 0.01


def test_infer_config_defaults():
    config = InferConfig(input_path="/bins", model_checkpoint="/model.pt")
    assert config.batch_size == 64
    assert config.device == "auto"
    assert config.threshold_default == 0.0
    assert config.resolved_formats() == ("h5",)


def test_infer_config_resolved_formats():
    assert InferConfig(output_format="csv").resolved_formats() == ("csv",)
    # comma-separated string, order preserved and de-duplicated
    assert InferConfig(output_format="h5,csv,mat").resolved_formats() == ("h5", "csv", "mat")
    assert InferConfig(output_format="csv, csv , h5").resolved_formats() == ("csv", "h5")
    # YAML list form
    assert InferConfig(output_format=["mat", "h5"]).resolved_formats() == ("mat", "h5")
    # csv-labels (hyphenated token) parses
    assert InferConfig(output_format="h5,csv-labels").resolved_formats() == ("h5", "csv-labels")
    # "all" expands to every format
    assert InferConfig(output_format="all").resolved_formats() == ("h5", "csv", "mat", "csv-labels")


def test_infer_config_invalid_format():
    with pytest.raises(ValueError, match="Unknown output format"):
        InferConfig(output_format="parquet")


def test_date_placeholder_expansion(tmp_path):
    cfg = {
        "input_path": "/ifcb/data/{year}",
        "model_checkpoint": "/models/best.pt",
        "output_dir": "/ifcb/output/{year}",
    }
    yaml_path = tmp_path / "infer.yaml"
    yaml_path.write_text(yaml.dump(cfg))

    config = load_config(yaml_path, InferConfig)
    year = datetime.now(UTC).strftime("%Y")
    assert config.input_path == f"/ifcb/data/{year}"
    assert config.output_dir == f"/ifcb/output/{year}"


def test_date_placeholder_month_day(tmp_path):
    cfg = {"data_dir": "/data/{year}/{month}/{day}", "model": "resnet18"}
    yaml_path = tmp_path / "train.yaml"
    yaml_path.write_text(yaml.dump(cfg))

    config = load_config(yaml_path, TrainConfig)
    now = datetime.now(UTC)
    assert config.data_dir == f"/data/{now:%Y}/{now:%m}/{now:%d}"


def test_train_config_invalid_val_split():
    with pytest.raises(ValueError, match="val_split"):
        TrainConfig(val_split=0.0)


def test_train_config_invalid_val_split_above_one():
    with pytest.raises(ValueError, match="val_split"):
        TrainConfig(val_split=1.0)


def test_train_config_negative_lr():
    with pytest.raises(ValueError, match="lr"):
        TrainConfig(lr=-0.001)


def test_train_config_zero_batch_size():
    with pytest.raises(ValueError, match="batch_size"):
        TrainConfig(batch_size=0)


def test_train_config_zero_epochs():
    with pytest.raises(ValueError, match="epochs"):
        TrainConfig(epochs=0)


def test_train_config_negative_image_dims():
    with pytest.raises(ValueError, match="image dimensions"):
        TrainConfig(image_width=0, image_height=224)


def test_infer_config_zero_batch_size():
    with pytest.raises(ValueError, match="batch_size"):
        InferConfig(batch_size=0)
