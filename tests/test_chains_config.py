import pytest

from ifcb_classify.chains.config import ChainTrainConfig
from ifcb_classify.chains.train import resolve_data_yaml


def test_valid_config():
    cfg = ChainTrainConfig(class_name="Skeletonema", data="datasets/skeletonema")
    assert cfg.class_name == "Skeletonema"
    assert cfg.model == "yolo11s.pt"
    assert cfg.device == "cpu"


def test_class_name_required():
    with pytest.raises(ValueError, match="class_name is required"):
        ChainTrainConfig(data="datasets/skeletonema")


def test_data_required():
    with pytest.raises(ValueError, match="data .* is required"):
        ChainTrainConfig(class_name="Skeletonema")


@pytest.mark.parametrize(
    "field,value,msg",
    [
        ("epochs", 0, "epochs must be >= 1"),
        ("imgsz", 0, "imgsz must be >= 1"),
        ("batch", 0, "batch must be >= 1"),
        ("patience", -1, "patience must be >= 0"),
    ],
)
def test_numeric_validation(field, value, msg):
    kwargs = {"class_name": "Skeletonema", "data": "datasets/skeletonema", field: value}
    with pytest.raises(ValueError, match=msg):
        ChainTrainConfig(**kwargs)


def test_resolve_data_yaml_prefers_local(tmp_path):
    (tmp_path / "data.yaml").write_text("names:\n  0: x\n")
    (tmp_path / "data.local.yaml").write_text("names:\n  0: x\n")
    assert resolve_data_yaml(str(tmp_path)).name == "data.local.yaml"


def test_resolve_data_yaml_falls_back(tmp_path):
    (tmp_path / "data.yaml").write_text("names:\n  0: x\n")
    assert resolve_data_yaml(str(tmp_path)).name == "data.yaml"


def test_resolve_data_yaml_file_path(tmp_path):
    f = tmp_path / "custom.yaml"
    f.write_text("names:\n  0: x\n")
    assert resolve_data_yaml(str(f)) == f


def test_resolve_data_yaml_missing_dir(tmp_path):
    with pytest.raises(FileNotFoundError):
        resolve_data_yaml(str(tmp_path))


def test_resolve_data_yaml_missing_path():
    with pytest.raises(FileNotFoundError):
        resolve_data_yaml("/nonexistent/path/data.yaml")
