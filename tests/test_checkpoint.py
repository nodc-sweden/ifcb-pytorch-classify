import pytest
import torch
import torch.nn as nn

from ifcb_classify.checkpoint import CheckpointManager, _guess_model_name, _load_class_names


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(4, 2)

    def forward(self, x):
        return self.fc(x)


def test_maybe_save_improves(tmp_path):
    mgr = CheckpointManager(str(tmp_path), metric_name="acc", mode="max")
    model = TinyModel()

    saved = mgr.maybe_save(model, 0.8, "run1", epoch=1, class_names=["A", "B"], config={"lr": 0.01})
    assert saved
    assert (tmp_path / "run1_best.pt").exists()


def test_maybe_save_no_improvement(tmp_path):
    mgr = CheckpointManager(str(tmp_path), metric_name="acc", mode="max")
    model = TinyModel()

    mgr.maybe_save(model, 0.8, "run1", epoch=1, class_names=["A", "B"], config={})
    saved = mgr.maybe_save(model, 0.7, "run1", epoch=2, class_names=["A", "B"], config={})
    assert not saved


def test_maybe_save_replaces_old_checkpoint(tmp_path):
    mgr = CheckpointManager(str(tmp_path), metric_name="acc", mode="max")
    model = TinyModel()

    mgr.maybe_save(model, 0.8, "run1", epoch=1, class_names=["A", "B"], config={})
    mgr.maybe_save(model, 0.9, "run1", epoch=2, class_names=["A", "B"], config={})

    files = list(tmp_path.glob("*.pt"))
    assert len(files) == 1


def test_maybe_save_min_mode(tmp_path):
    mgr = CheckpointManager(str(tmp_path), metric_name="loss", mode="min")
    model = TinyModel()

    assert mgr.maybe_save(model, 0.5, "run1", epoch=1, class_names=["A"], config={})
    assert mgr.maybe_save(model, 0.3, "run1", epoch=2, class_names=["A"], config={})
    assert not mgr.maybe_save(model, 0.4, "run1", epoch=3, class_names=["A"], config={})


def test_checkpoint_contains_metadata(tmp_path):
    mgr = CheckpointManager(str(tmp_path), metric_name="f1", mode="max")
    model = TinyModel()
    mgr.maybe_save(model, 0.9, "run1", epoch=5, class_names=["A", "B"], config={"lr": 0.01})

    data = torch.load(tmp_path / "run1_best.pt", map_location="cpu", weights_only=False)
    assert data["epoch"] == 5
    assert data["metric_value"] == 0.9
    assert data["class_names"] == ["A", "B"]
    assert data["config"] == {"lr": 0.01}


def test_load_class_names(tmp_path):
    classes_file = tmp_path / "classes.txt"
    classes_file.write_text("ClassA\nClassB\nClassC\n")
    result = _load_class_names(tmp_path / "model.pt", str(classes_file))
    assert result == ["ClassA", "ClassB", "ClassC"]


def test_load_class_names_missing_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        _load_class_names(tmp_path / "model.pt", None)


def test_guess_model_name_resnet():
    state_dict = {"layer4.0.conv1.weight": None, "fc.weight": None, "fc.bias": None}
    assert _guess_model_name(state_dict) == "resnet50"


def test_guess_model_name_efficientnet():
    state_dict = {"features.0.weight": None, "classifier.1.weight": None}
    assert _guess_model_name(state_dict) == "efficientnet_b0"


def test_guess_model_name_fallback():
    state_dict = {"some.random.key": None}
    assert _guess_model_name(state_dict) == "resnet50"
