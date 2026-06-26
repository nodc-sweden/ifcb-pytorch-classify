import sys
import types
from pathlib import Path

import numpy as np
import pytest

from ifcb_classify.chains.config import ChainCountingConfig, ChainModelSpec
from ifcb_classify.chains.counter import ChainCounter


# --- config parsing/validation ---------------------------------------------

def test_from_dict_basic():
    cfg = ChainCountingConfig.from_dict({
        "enabled": True,
        "models": {"Skeletonema": {"weights": "a.pt", "iou": 0.3}},
    })
    assert cfg.enabled is True
    assert cfg.models["Skeletonema"] == ChainModelSpec(weights="a.pt", iou=0.3, conf=0.25)


def test_from_dict_string_shorthand():
    cfg = ChainCountingConfig.from_dict({"enabled": True, "models": {"X": "w.pt"}})
    assert cfg.models["X"].weights == "w.pt"


def test_from_dict_defaults_applied():
    cfg = ChainCountingConfig.from_dict({
        "enabled": True, "iou": 0.4, "conf": 0.1,
        "models": {"X": {"weights": "w.pt"}},
    })
    assert cfg.models["X"].iou == 0.4
    assert cfg.models["X"].conf == 0.1


def test_from_dict_disabled_default():
    cfg = ChainCountingConfig.from_dict({})
    assert cfg.enabled is False
    assert cfg.models == {}


def test_from_dict_missing_weights():
    with pytest.raises(ValueError, match="missing 'weights'"):
        ChainCountingConfig.from_dict({"enabled": True, "models": {"X": {"iou": 0.3}}})


def test_from_dict_enabled_without_models():
    with pytest.raises(ValueError, match="no models are configured"):
        ChainCountingConfig.from_dict({"enabled": True, "models": {}})


@pytest.mark.parametrize("iou", [-0.1, 1.5])
def test_from_dict_bad_iou(iou):
    with pytest.raises(ValueError, match="iou must be in"):
        ChainCountingConfig.from_dict({"enabled": True, "models": {"X": {"weights": "w.pt", "iou": iou}}})


# --- counter (with a fake ultralytics) -------------------------------------

class _FakeResult:
    def __init__(self, n):
        self.boxes = list(range(n))


class _FakeYOLO:
    """Returns a fixed box count derived from the weights filename digits."""

    def __init__(self, weights):
        self.weights = weights
        # Count is encoded as the trailing token of the filename stem (e.g. "Skeletonema_7").
        self._n = int(Path(weights).stem.split("_")[-1])

    def __call__(self, image, iou=None, conf=None, verbose=False):
        return [_FakeResult(self._n)]


@pytest.fixture
def fake_ultralytics(monkeypatch):
    mod = types.ModuleType("ultralytics")
    mod.YOLO = _FakeYOLO
    monkeypatch.setitem(sys.modules, "ultralytics", mod)
    return mod


def _make_config(tmp_path, mapping):
    models = {}
    for name, n in mapping.items():
        # Encode the desired count in the filename so _FakeYOLO is deterministic.
        w = tmp_path / f"{name}_{n}.pt"
        w.write_bytes(b"x")
        models[name] = {"weights": str(w)}
    return ChainCountingConfig.from_dict({"enabled": True, "models": models})


def test_counter_handles(tmp_path, fake_ultralytics):
    counter = ChainCounter(_make_config(tmp_path, {"Skeletonema": 5}))
    assert counter.handles("Skeletonema") is True
    assert counter.handles("Other") is False


def test_counter_counts(tmp_path, fake_ultralytics):
    counter = ChainCounter(_make_config(tmp_path, {"Skeletonema": 7}))
    assert counter.count(object(), "Skeletonema") == 7


def test_counter_lazy_and_cached(tmp_path, fake_ultralytics):
    counter = ChainCounter(_make_config(tmp_path, {"Skeletonema": 3}))
    assert counter._models == {}          # nothing loaded yet
    counter.count(object(), "Skeletonema")
    counter.count(object(), "Skeletonema")
    assert list(counter._models) == ["Skeletonema"]  # loaded once, cached


def test_counter_missing_weights(tmp_path, fake_ultralytics):
    cfg = ChainCountingConfig.from_dict({"enabled": True, "models": {"X": {"weights": str(tmp_path / "nope.pt")}}})
    with pytest.raises(FileNotFoundError, match="weights not found"):
        ChainCounter(cfg)


def test_counter_metadata(tmp_path, fake_ultralytics):
    counter = ChainCounter(_make_config(tmp_path, {"Skeletonema": 2}))
    meta = counter.models_metadata()
    assert set(meta["Skeletonema"]) == {"weights", "iou", "conf"}


# --- infer-level gating (thresholded class_name) ----------------------------

class _StubCounter:
    """Counts = 10 for handled classes; records which images it saw."""

    def __init__(self, handled):
        self._handled = set(handled)
        self.seen = []

    def handles(self, class_name):
        return class_name in self._handled

    def count(self, image, class_name):
        self.seen.append((image, class_name))
        return 10

    def models_metadata(self):
        return {name: {"weights": "x.pt", "iou": 0.3, "conf": 0.25} for name in self._handled}


def test_compute_chain_counts_gates_on_thresholded_name():
    from ifcb_classify.infer import _compute_chain_counts

    class_names = ["Skeletonema", "Other"]
    scores = np.array([
        [0.9, 0.1],    # Skeletonema, 0.9 -> above threshold, counted
        [0.2, 0.8],    # Other -> not handled
        [0.55, 0.45],  # Skeletonema, 0.55 -> below 0.7 threshold -> "unclassified"
    ])
    thresholds = np.array([0.7, np.nan])  # Skeletonema needs >=0.7
    raw_images = ["img0", "img1", "img2"]
    counter = _StubCounter(handled=["Skeletonema"])

    counts, meta = _compute_chain_counts(scores, class_names, thresholds, counter, raw_images)

    # ROI 0 counted; ROI 1 is Other (-1); ROI 2 falls below threshold -> "unclassified" (-1)
    np.testing.assert_array_equal(counts, [10, -1, -1])
    assert counter.seen == [("img0", "Skeletonema")]
    assert "Skeletonema" in meta


def test_compute_chain_counts_disabled():
    from ifcb_classify.infer import _compute_chain_counts

    scores = np.array([[0.9, 0.1]])
    counts, meta = _compute_chain_counts(scores, ["A", "B"], np.array([np.nan, np.nan]), None, None)
    assert counts is None and meta is None
