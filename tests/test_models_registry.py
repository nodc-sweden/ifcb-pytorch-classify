import pytest

from ifcb_classify.models.factory import get_model
from ifcb_classify.models.registry import MODELS


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
