import shutil
from pathlib import Path

import h5py
import numpy as np
import pytest
import torch
from PIL import Image

from ifcb_classify import infer as infer_mod
from ifcb_classify.config import InferConfig, TrainConfig
from ifcb_classify.data.ifcb_bin import BinFiles
from ifcb_classify.infer import (
    _batch_predict,
    _classify_directory,
    _derive_classifier_name,
    _has_pending_bins,
    _load_thresholds,
)
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
    assert _has_pending_bins(BIN_PATH, tmp_path, ("h5",)) is True

    # Create the output file to simulate already-classified
    (tmp_path / "D20220519T124533_IFCB134_class.h5").touch()
    assert _has_pending_bins(BIN_PATH, tmp_path, ("h5",)) is False


def test_has_pending_bins_directory(tmp_path):
    bins_dir = FIXTURES / "bins"
    assert _has_pending_bins(bins_dir, tmp_path, ("h5",)) is True

    (tmp_path / "D20220519T124533_IFCB134_class.h5").touch()
    assert _has_pending_bins(bins_dir, tmp_path, ("h5",)) is False


def test_has_pending_bins_empty_directory(tmp_path):
    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()
    assert _has_pending_bins(empty_dir, tmp_path, ("h5",)) is False


def test_has_pending_bins_directory_ignores_undiscoverable_bins(tmp_path):
    """Pending work must be judged by the same enumeration that does the work.

    ifcbkit's directory discovery skips filesets with no .hdr. Counting those as
    pending made infer load the model, process nothing, and report them pending
    again on every subsequent run.
    """
    bins_dir = tmp_path / "bins"
    bins_dir.mkdir()
    for suffix in (".roi", ".adc"):
        src = BIN_PATH.with_suffix(suffix)
        (bins_dir / src.name).write_bytes(src.read_bytes())

    assert _has_pending_bins(bins_dir, tmp_path, ("h5",)) is False


def test_has_pending_bins_multiformat(tmp_path):
    # A bin already written to h5 is still pending when csv is also requested.
    (tmp_path / "D20220519T124533_IFCB134_class.h5").touch()
    assert _has_pending_bins(BIN_PATH, tmp_path, ("h5", "csv")) is True

    (tmp_path / "D20220519T124533_IFCB134_class.csv").touch()
    assert _has_pending_bins(BIN_PATH, tmp_path, ("h5", "csv")) is False


# --- _classify_directory (with stubbed bin I/O) -----------------------------

def _fake_bin(lid):
    """A stand-in for the BinFiles handle iter_directory_bins yields."""
    return BinFiles(lid=lid, adc_path=Path(f"{lid}.adc"), roi_path=Path(f"{lid}.roi"))


def _two_roi_images(_bin):
    """Yield two trivial 8x8 RGB ROIs, mimicking iter_bin_images."""
    return [(0, Image.new("RGB", (8, 8))), (1, Image.new("RGB", (8, 8)))]


def _wire_directory(monkeypatch, lids):
    """Stub iter_directory_bins/iter_bin_images so no real bins are needed."""
    monkeypatch.setattr(infer_mod, "iter_directory_bins", lambda d: [(lid, _fake_bin(lid)) for lid in lids])
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
        ["A", "B"], np.array([np.nan, np.nan]), "clf", tmp_path, False, ("h5",),
    )

    assert (tmp_path / "D1_class.h5").exists()
    assert (tmp_path / "D2_class.h5").exists()
    with h5py.File(tmp_path / "D1_class.h5", "r") as f:
        assert f["roi_numbers"][:].tolist() == [0, 1]
        assert "cell_count" not in f


def test_classify_directory_all_formats(tmp_path, monkeypatch):
    _wire_directory(monkeypatch, ["D1"])
    model, transform = _tiny_model_and_transform()

    _classify_directory(
        tmp_path, model, transform, torch.device("cpu"), 8,
        ["A", "B"], np.array([np.nan, np.nan]), "clf", tmp_path, False, ("h5", "csv", "mat", "csv-labels"),
    )

    assert (tmp_path / "D1_class.h5").exists()
    assert (tmp_path / "D1_class.csv").exists()
    assert (tmp_path / "D1_class_v1.mat").exists()
    # csv-labels uses the bare {lid}.csv name (iRfcb convention)
    assert (tmp_path / "D1.csv").exists()

    import pandas as pd

    df = pd.read_csv(tmp_path / "D1_class.csv")
    assert list(df.columns) == ["pid", "A", "B"]
    assert df["pid"].tolist() == ["D1_00000", "D1_00001"]

    labels = pd.read_csv(tmp_path / "D1.csv")
    assert list(labels.columns) == ["file_name", "class_name", "class_name_auto", "score"]
    assert labels["file_name"].tolist() == ["D1_00000.png", "D1_00001.png"]


# --- pretrained plumbed through to inference --------------------------------

def _write_thresholds_json(path, thresholds_by_class):
    """Write the {run}_thresholds_and_metrics.json layout training produces."""
    import json

    payload = {
        "model_name": path.stem,
        "best_epoch": 1,
        "num_classes": len(thresholds_by_class),
        "class_metrics": {
            name: {"class_name": name, "threshold": value, "f1": 0.5, "precision": 0.5, "recall": 0.5, "support": 3}
            for name, value in thresholds_by_class.items()
        },
    }
    path.write_text(json.dumps(payload))


def _stub_inference(monkeypatch, tmp_path, train_config):
    """Wire infer_main's heavy dependencies to stubs, recording get_model's args.

    Returns the dict the fake get_model records its call into.
    """
    model = torch.nn.Linear(3 * 8 * 8, 2)
    recorded: dict = {}

    def fake_get_model(name, num_classes, pretrained=True):
        recorded.update(name=name, num_classes=num_classes, pretrained=pretrained)
        return model

    monkeypatch.setattr(infer_mod, "get_model", fake_get_model)
    monkeypatch.setattr(
        infer_mod,
        "load_checkpoint",
        lambda *a, **k: {"state_dict": model.state_dict(), "class_names": ["A", "B"], "config": train_config},
    )
    monkeypatch.setattr(infer_mod, "build_transform", lambda *a, **k: (lambda img: torch.rand(3 * 8 * 8)))
    monkeypatch.setattr(infer_mod, "iter_bin_images", _two_roi_images)
    return recorded


_TRAIN_CONFIG = {"model": "resnet18", "transform": "dataset_squarepad", "image_width": 8, "image_height": 8}


def test_infer_honours_pretrained_false_from_checkpoint(tmp_path, monkeypatch):
    """A from-scratch checkpoint must be rebuilt from scratch at inference.

    torchvision's ``weights=`` also reshapes some architectures (inception_v3
    forces ``transform_input=True``), so rebuilding a ``pretrained: false``
    checkpoint with pretrained weights silently changes preprocessing.
    """
    recorded = _stub_inference(monkeypatch, tmp_path, {**_TRAIN_CONFIG, "pretrained": False})

    infer_mod.infer_main(InferConfig(input_path=str(BIN_PATH), model_checkpoint="m.pt", output_dir=str(tmp_path)))

    assert recorded["pretrained"] is False


def test_infer_defaults_pretrained_true_for_legacy_checkpoints(tmp_path, monkeypatch):
    """Legacy checkpoints synthesise a config with no ``pretrained`` key."""
    recorded = _stub_inference(monkeypatch, tmp_path, dict(_TRAIN_CONFIG))

    infer_mod.infer_main(InferConfig(input_path=str(BIN_PATH), model_checkpoint="m.pt", output_dir=str(tmp_path)))

    assert recorded["pretrained"] is True


# --- threshold auto-detection ------------------------------------------------

def test_load_thresholds_autodetects_training_output(tmp_path):
    """Training writes {run}_thresholds_and_metrics.json; inference must find it."""
    (tmp_path / "RUN_best.pt").touch()
    _write_thresholds_json(tmp_path / "RUN_thresholds_and_metrics.json", {"A": 0.25, "B": 0.75})

    config = InferConfig(model_checkpoint=str(tmp_path / "RUN_best.pt"), threshold_default=0.0)
    np.testing.assert_allclose(_load_thresholds(config, ["A", "B"]), [0.25, 0.75])


def test_load_thresholds_prefers_the_matching_run(tmp_path):
    """With several runs in one output dir, pick the one matching the checkpoint."""
    (tmp_path / "RUN_A_best.pt").touch()
    _write_thresholds_json(tmp_path / "RUN_A_thresholds_and_metrics.json", {"A": 0.25, "B": 0.75})
    _write_thresholds_json(tmp_path / "RUN_B_thresholds_and_metrics.json", {"A": 0.9, "B": 0.9})

    config = InferConfig(model_checkpoint=str(tmp_path / "RUN_A_best.pt"), threshold_default=0.0)
    np.testing.assert_allclose(_load_thresholds(config, ["A", "B"]), [0.25, 0.75])


def test_load_thresholds_does_not_guess_between_runs(tmp_path, caplog):
    """Ambiguous candidates must fall back to the default rather than guess."""
    (tmp_path / "model_best.pt").touch()
    _write_thresholds_json(tmp_path / "RUN_A_thresholds_and_metrics.json", {"A": 0.25, "B": 0.75})
    _write_thresholds_json(tmp_path / "RUN_B_thresholds_and_metrics.json", {"A": 0.9, "B": 0.9})

    config = InferConfig(model_checkpoint=str(tmp_path / "model_best.pt"), threshold_default=0.4)
    with caplog.at_level("WARNING"):
        np.testing.assert_allclose(_load_thresholds(config, ["A", "B"]), [0.4, 0.4])
    assert "thresholds" in caplog.text.lower()


def test_load_thresholds_autodetects_legacy_plain_name(tmp_path):
    """A hand-placed thresholds.json keeps working."""
    (tmp_path / "RUN_best.pt").touch()
    _write_thresholds_json(tmp_path / "thresholds.json", {"A": 0.3, "B": 0.6})

    config = InferConfig(model_checkpoint=str(tmp_path / "RUN_best.pt"), threshold_default=0.0)
    np.testing.assert_allclose(_load_thresholds(config, ["A", "B"]), [0.3, 0.6])


def test_load_thresholds_warns_when_none_found(tmp_path, caplog):
    """Falling back to a flat default must not be silent."""
    (tmp_path / "RUN_best.pt").touch()
    config = InferConfig(model_checkpoint=str(tmp_path / "RUN_best.pt"), threshold_default=0.0)

    with caplog.at_level("WARNING"):
        _load_thresholds(config, ["A", "B"])
    assert "no thresholds file" in caplog.text.lower()


def test_classify_directory_skips_existing(tmp_path, monkeypatch):
    _wire_directory(monkeypatch, ["D1"])
    model, transform = _tiny_model_and_transform()
    # Pre-create an empty output: skip leaves it untouched (writing would make
    # it a valid, non-empty HDF5 file).
    existing = tmp_path / "D1_class.h5"
    existing.touch()

    _classify_directory(
        tmp_path, model, transform, torch.device("cpu"), 8,
        ["A", "B"], np.array([np.nan, np.nan]), "clf", tmp_path, False, ("h5",),
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
        ["A", "B"], np.array([np.nan, np.nan]), "clf", tmp_path, False, ("h5", "mat"),
        counter=_StubCounter(),
    )

    with h5py.File(tmp_path / "D1_class.h5", "r") as f:
        np.testing.assert_array_equal(f["cell_count"][:], [7, 7])
        assert "cell_counter_models" in f.attrs

    # Chain counts also land in the mat (for iRfcb's cell-count summariser)
    from scipy.io import loadmat

    np.testing.assert_array_equal(loadmat(tmp_path / "D1_class_v1.mat", squeeze_me=True)["cell_count"], [7, 7])


def test_adding_format_preserves_existing_output(tmp_path, monkeypatch):
    """Adding csv to already-h5-classified bins must not rewrite the h5.

    The h5 was written with chain counts; a later csv-adding run without a counter
    must leave the h5 (and its cell_count) byte-for-byte intact.
    """
    model, transform = _tiny_model_and_transform()

    class _StubCounter:
        def handles(self, class_name):
            return True

        def count(self, image, class_name):
            return 3

        def models_metadata(self):
            return {"A": {"weights": "x.pt", "iou": 0.3, "conf": 0.25}}

    # First pass: h5 only, with counts.
    _wire_directory(monkeypatch, ["D1"])
    _classify_directory(
        tmp_path, model, transform, torch.device("cpu"), 8,
        ["A", "B"], np.array([np.nan, np.nan]), "clf", tmp_path, False, ("h5",),
        counter=_StubCounter(),
    )
    h5_path = tmp_path / "D1_class.h5"
    original_bytes = h5_path.read_bytes()

    # Second pass: request h5+csv, no counter, no overwrite.
    _wire_directory(monkeypatch, ["D1"])
    _classify_directory(
        tmp_path, model, transform, torch.device("cpu"), 8,
        ["A", "B"], np.array([np.nan, np.nan]), "clf", tmp_path, False, ("h5", "csv"),
    )

    # csv created, h5 untouched (counts preserved).
    assert (tmp_path / "D1_class.csv").exists()
    assert h5_path.read_bytes() == original_bytes
    with h5py.File(h5_path, "r") as f:
        np.testing.assert_array_equal(f["cell_count"][:], [3, 3])


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

    checkpoint = next(iter(model_dir.glob("*.pt")))

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


def _augmented_checkpoint(tmp_path, num_classes=6):
    """Write a pipeline checkpoint whose config names an augmented transform.

    Built rather than trained: a model fitted on the tiny fixture set saturates
    to softmax scores of exactly 1.0, where perturbing the input changes nothing
    and the regression below cannot fail. Random weights over several classes
    keep the scores off the rails, which is what makes the check sensitive.
    """
    from ifcb_classify.models.factory import get_model

    torch.manual_seed(0)
    model = get_model("resnet18", num_classes, pretrained=False)
    path = tmp_path / "model" / "run_best.pt"
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "class_names": [f"Class{i}" for i in range(num_classes)],
            "config": {
                "model": "resnet18",
                "pretrained": False,
                "transform": "dataset_squarepad_augmented",
                "image_width": 32,
                "image_height": 32,
                "seed": 42,
            },
        },
        path,
    )
    return path


def test_augmented_checkpoint_scores_independent_of_run_position(tmp_path):
    """Regression: a bin must score the same alone as it does mid-directory.

    Inference used to rebuild the training transform verbatim, so ColorJitter and
    the random flips ran on every ROI. The transform is built once and reused for
    the whole run, so the RNG advanced from bin to bin and a bin's scores depended
    on how many ROIs preceded it. Re-running the same command stayed reproducible
    -- infer_main re-seeds every invocation -- so this has to compare a single-bin
    run against a directory run, not two identical ones.
    """
    checkpoint = _augmented_checkpoint(tmp_path)

    # Copies of the one fixture under different LIDs, which are only labels. The
    # target needs a bin scored before it for the RNG to have moved on, and it is
    # bracketed by an earlier and a later LID so that holds whichever direction
    # discovery happens to enumerate in — otherwise a change there would leave
    # the test passing while silently testing nothing.
    bins_dir = tmp_path / "bins"
    bins_dir.mkdir()
    target = "D20220519T124533_IFCB134"
    for lid in ("D20220519T010101_IFCB134", target, "D20220519T235959_IFCB134"):
        for ext in (".adc", ".roi", ".hdr"):
            shutil.copy(BIN_PATH.with_suffix(ext), bins_dir / f"{lid}{ext}")

    from ifcb_classify.infer import infer_main

    def classify(input_path, out_dir):
        infer_main(InferConfig(
            input_path=str(input_path),
            model_checkpoint=str(checkpoint),
            output_dir=str(out_dir),
            batch_size=8,
            device="cpu",
            overwrite=True,
        ))
        with h5py.File(Path(out_dir) / f"{target}_class.h5", "r") as f:
            return f["output_scores"][:]

    alone = classify(BIN_PATH, tmp_path / "alone")
    in_directory = classify(bins_dir, tmp_path / "directory")

    # Guard the guard: a saturated model would make this pass for the wrong reason.
    assert alone.max(axis=1).min() < 0.99, "model too confident for this test to be sensitive"
    np.testing.assert_array_equal(alone, in_directory)
