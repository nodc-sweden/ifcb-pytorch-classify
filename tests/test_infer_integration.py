from pathlib import Path

import h5py
import numpy as np
import pytest
import torch
from PIL import Image

from ifcb_classify import infer as infer_mod
from ifcb_classify.config import TrainConfig, InferConfig
from ifcb_classify.infer import _batch_predict, _classify_directory, _derive_classifier_name, _has_pending_bins, _load_thresholds
from ifcb_classify.train import train_main

FIXTURES = Path(__file__).parent / "fixtures"
BIN_PATH = FIXTURES / "bins" / "D20220519T124533_IFCB134.roi"


def test_batch_predict():
    model = torch.nn.Linear(3 * 32 * 32, 2)
    model.eval()
    images = [torch.rand(3, 32, 32).flatten() for _ in range(10)]
    scores = _batch_predict(model, images, torch.device("cpu"), batch_size=4)
    assert scores.shape == (10, 2)
    assert np.allclose(scores.sum(axis=1), 1.0, atol=1e-5)


def test_derive_classifier_name():
    config = InferConfig(model_checkpoint="/models/my_model/best.pt")
    name = _derive_classifier_name(config, {"model": "resnet50", "dataset_version": "V1"})
    assert name == "my_model"


def test_derive_classifier_name_fallback():
    config = InferConfig(model_checkpoint="best.pt")
    name = _derive_classifier_name(config, {"model": "resnet50", "dataset_version": "V1"})
    assert name == "resnet50_V1"


def test_load_thresholds_default():
    config = InferConfig(model_checkpoint="nonexistent/best.pt", threshold_default=0.5)
    thresholds = _load_thresholds(config, ["A", "B", "C"])
    np.testing.assert_array_equal(thresholds, [0.5, 0.5, 0.5])


def test_has_pending_bins_single_file(tmp_path):
    assert _has_pending_bins(BIN_PATH, tmp_path) is True

    # Create the output file to simulate already-classified
    (tmp_path / "D20220519T124533_IFCB134_class.h5").touch()
    assert _has_pending_bins(BIN_PATH, tmp_path) is False


def test_has_pending_bins_directory(tmp_path):
    bins_dir = FIXTURES / "bins"
    assert _has_pending_bins(bins_dir, tmp_path) is True

    (tmp_path / "D20220519T124533_IFCB134_class.h5").touch()
    assert _has_pending_bins(bins_dir, tmp_path) is False


def test_has_pending_bins_empty_directory(tmp_path):
    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()
    assert _has_pending_bins(empty_dir, tmp_path) is False


# --- _classify_directory (with stubbed bin I/O) -----------------------------

class _FakeBin:
    """A context-manageable stand-in for a pyifcb bin object."""

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def _two_roi_images(_bin):
    """Yield two trivial 8x8 RGB ROIs, mimicking iter_bin_images."""
    return [(0, Image.new("RGB", (8, 8))), (1, Image.new("RGB", (8, 8)))]


def _wire_directory(monkeypatch, lids):
    """Stub iter_directory_bins/iter_bin_images so no real bins are needed."""
    monkeypatch.setattr(infer_mod, "iter_directory_bins", lambda d: [(lid, _FakeBin()) for lid in lids])
    monkeypatch.setattr(infer_mod, "iter_bin_images", _two_roi_images)


def _tiny_model_and_transform():
    """A Linear model over flattened 3x8x8 ROIs and a matching transform."""
    model = torch.nn.Linear(3 * 8 * 8, 2)
    model.eval()
    transform = lambda img: torch.rand(3 * 8 * 8)  # noqa: E731 - test shim
    return model, transform


def test_classify_directory_writes_outputs(tmp_path, monkeypatch):
    _wire_directory(monkeypatch, ["D1", "D2"])
    model, transform = _tiny_model_and_transform()

    _classify_directory(
        tmp_path, model, transform, torch.device("cpu"), 8,
        ["A", "B"], np.array([np.nan, np.nan]), "clf", tmp_path, overwrite=False,
    )

    assert (tmp_path / "D1_class.h5").exists()
    assert (tmp_path / "D2_class.h5").exists()
    with h5py.File(tmp_path / "D1_class.h5", "r") as f:
        assert f["roi_numbers"][:].tolist() == [0, 1]
        assert "chain_count" not in f


def test_classify_directory_skips_existing(tmp_path, monkeypatch):
    _wire_directory(monkeypatch, ["D1"])
    model, transform = _tiny_model_and_transform()
    # Pre-create an empty output: skip leaves it untouched (writing would make
    # it a valid, non-empty HDF5 file).
    existing = tmp_path / "D1_class.h5"
    existing.touch()

    _classify_directory(
        tmp_path, model, transform, torch.device("cpu"), 8,
        ["A", "B"], np.array([np.nan, np.nan]), "clf", tmp_path, overwrite=False,
    )

    assert existing.stat().st_size == 0


def test_classify_directory_with_counter(tmp_path, monkeypatch):
    _wire_directory(monkeypatch, ["D1"])
    model, transform = _tiny_model_and_transform()

    class _StubCounter:
        def handles(self, class_name):
            return True  # count every ROI

        def count(self, image, class_name):
            return 7

        def models_metadata(self):
            return {"A": {"weights": "x.pt", "iou": 0.3, "conf": 0.25}}

    _classify_directory(
        tmp_path, model, transform, torch.device("cpu"), 8,
        ["A", "B"], np.array([np.nan, np.nan]), "clf", tmp_path, overwrite=False,
        counter=_StubCounter(),
    )

    with h5py.File(tmp_path / "D1_class.h5", "r") as f:
        np.testing.assert_array_equal(f["chain_count"][:], [7, 7])
        assert "chain_counter_models" in f.attrs


@pytest.mark.slow
def test_train_then_infer(tmp_path):
    """End-to-end: train a tiny model, then run inference on a real bin."""
    model_dir = tmp_path / "model"
    train_config = TrainConfig(
        data_dir=str(FIXTURES / "training_data"),
        model="resnet18",
        transform="dataset_squarepad",
        image_width=32,
        image_height=32,
        epochs=1,
        batch_size=8,
        lr=0.01,
        output_dir=str(model_dir),
        tracker="none",
        val_split=0.3,
        num_workers=0,
    )
    train_main(train_config)

    checkpoint = list(model_dir.glob("*.pt"))[0]

    from ifcb_classify.infer import infer_main

    output_dir = tmp_path / "class_scores"
    infer_config = InferConfig(
        input_path=str(BIN_PATH),
        model_checkpoint=str(checkpoint),
        output_dir=str(output_dir),
        batch_size=8,
        device="cpu",
    )
    infer_main(infer_config)

    h5_files = list(output_dir.glob("*.h5"))
    assert len(h5_files) == 1
    assert "D20220519T124533_IFCB134" in h5_files[0].name
