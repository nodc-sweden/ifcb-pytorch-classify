import tempfile
from pathlib import Path

import yaml

from ifcb_classify.config import TrainConfig, InferConfig, load_config


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
