import pytest
import torch.nn as nn

from ifcb_classify.models.factory import get_model
from ifcb_classify.models.registry import MODELS, ModelSpec


def _recording_spec(captured: dict) -> ModelSpec:
    """A ModelSpec whose constructor records the ``weights`` argument it received."""

    def constructor(weights=None):
        captured["weights"] = weights
        model = nn.Module()
        model.fc = nn.Linear(4, 2)
        return model

    return ModelSpec(constructor, "fc", 4)


def test_all_models_registered():
    assert len(MODELS) >= 40


def test_get_model_resnet50():
    model = get_model("resnet50", num_classes=6)
    assert model.fc.out_features == 6


def test_get_model_unknown_raises():
    with pytest.raises(ValueError, match="Unknown model: nonexistent_model"):
        get_model("nonexistent_model", num_classes=6)


def test_custom_model():
    model = get_model("custom", num_classes=10)
    assert model is not None


def test_pretrained_true_requests_registry_weights(monkeypatch):
    captured: dict = {}
    monkeypatch.setitem(MODELS, "recording_arch", _recording_spec(captured))
    get_model("recording_arch", num_classes=3)
    assert captured["weights"] == "DEFAULT"


def test_pretrained_false_trains_from_scratch(monkeypatch):
    captured: dict = {}
    monkeypatch.setitem(MODELS, "recording_arch", _recording_spec(captured))
    get_model("recording_arch", num_classes=3, pretrained=False)
    assert captured["weights"] is None
