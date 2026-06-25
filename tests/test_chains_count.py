import json
from pathlib import Path

import h5py
import numpy as np
import pytest

from ifcb_classify.chains import count as count_mod
from ifcb_classify.hdf5_output import write_class_scores


# --- helpers ----------------------------------------------------------------

class _StubCounter:
    """Counts = number-of-pixels stand-in (here: fixed 10) for handled classes."""

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


def _write_scores_file(path, class_labels, names, roi_numbers, thresholds):
    """Write a minimal class-scores file whose thresholded class_name == names."""
    # Build scores that argmax to the desired (already-thresholded) names. Using
    # NaN thresholds means class_name == argmax label, so names drive the file.
    n = len(names)
    scores = np.full((n, len(class_labels)), 0.01)
    for j, name in enumerate(names):
        scores[j, class_labels.index(name)] = 0.99
    write_class_scores(
        path, scores, class_labels, np.asarray(roi_numbers, dtype=np.int32),
        "test-classifier", np.asarray(thresholds, dtype=np.float64),
    )


# --- _index_bins ------------------------------------------------------------

def test_index_bins_single_file(tmp_path):
    roi = tmp_path / "D20200101T000000_IFCB100.roi"
    roi.write_bytes(b"")
    idx = count_mod._index_bins(roi)
    assert idx == {"D20200101T000000_IFCB100": roi}


def test_index_bins_directory(tmp_path):
    (tmp_path / "a.roi").write_bytes(b"")
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "b.roi").write_bytes(b"")
    idx = count_mod._index_bins(tmp_path)
    assert set(idx) == {"a", "b"}


def test_index_bins_missing(tmp_path):
    with pytest.raises(FileNotFoundError):
        count_mod._index_bins(tmp_path / "nope")


# --- _count_one_file --------------------------------------------------------

def test_count_one_file_writes_counts(tmp_path, monkeypatch):
    class_labels = ["Skeletonema", "Other"]
    h5 = tmp_path / "D1_class.h5"
    _write_scores_file(h5, class_labels, ["Skeletonema", "Other", "Skeletonema"], [5, 6, 7], [np.nan, np.nan])

    # Bin yields one image per ROI target; image payload is the target int.
    monkeypatch.setattr(count_mod, "iter_bin_images", lambda p: [(5, "img5"), (6, "img6"), (7, "img7")])

    counter = _StubCounter(handled=["Skeletonema"])
    count_mod._count_one_file(h5, {"D1": Path("ignored.roi")}, counter, overwrite=False)

    with h5py.File(h5, "r") as f:
        counts = f["chain_count"][:]
        meta = json.loads(f.attrs["chain_counter_models"])
    # ROIs 0 and 2 are Skeletonema (counted=10); ROI 1 is Other (-1).
    np.testing.assert_array_equal(counts, [10, -1, 10])
    # Only the gated ROIs' images were passed to the counter.
    assert {img for img, _ in counter.seen} == {"img5", "img7"}
    assert "Skeletonema" in meta


def test_count_one_file_skips_when_already_counted(tmp_path, monkeypatch):
    class_labels = ["Skeletonema"]
    h5 = tmp_path / "D1_class.h5"
    _write_scores_file(h5, class_labels, ["Skeletonema"], [5], [np.nan])
    # Seed an existing chain_count dataset.
    with h5py.File(h5, "a") as f:
        f.create_dataset("chain_count", data=np.array([99], dtype=np.int32))

    called = []
    monkeypatch.setattr(count_mod, "iter_bin_images", lambda p: called.append(p) or [])
    counter = _StubCounter(handled=["Skeletonema"])
    count_mod._count_one_file(h5, {"D1": Path("x.roi")}, counter, overwrite=False)

    with h5py.File(h5, "r") as f:
        assert f["chain_count"][:].tolist() == [99]  # untouched
    assert called == []  # bin never opened


def test_count_one_file_overwrite_recounts(tmp_path, monkeypatch):
    class_labels = ["Skeletonema"]
    h5 = tmp_path / "D1_class.h5"
    _write_scores_file(h5, class_labels, ["Skeletonema"], [5], [np.nan])
    with h5py.File(h5, "a") as f:
        f.create_dataset("chain_count", data=np.array([99], dtype=np.int32))

    monkeypatch.setattr(count_mod, "iter_bin_images", lambda p: [(5, "img5")])
    counter = _StubCounter(handled=["Skeletonema"])
    count_mod._count_one_file(h5, {"D1": Path("x.roi")}, counter, overwrite=True)

    with h5py.File(h5, "r") as f:
        assert f["chain_count"][:].tolist() == [10]


def test_count_one_file_no_gated_writes_all_sentinel(tmp_path, monkeypatch):
    class_labels = ["Other"]
    h5 = tmp_path / "D1_class.h5"
    _write_scores_file(h5, class_labels, ["Other", "Other"], [5, 6], [np.nan])

    called = []
    monkeypatch.setattr(count_mod, "iter_bin_images", lambda p: called.append(p) or [])
    counter = _StubCounter(handled=["Skeletonema"])
    count_mod._count_one_file(h5, {"D1": Path("x.roi")}, counter, overwrite=False)

    with h5py.File(h5, "r") as f:
        np.testing.assert_array_equal(f["chain_count"][:], [-1, -1])
    assert called == []  # no countable ROIs, so the bin is never opened


def test_count_one_file_missing_bin_writes_sentinel(tmp_path):
    class_labels = ["Skeletonema"]
    h5 = tmp_path / "D1_class.h5"
    _write_scores_file(h5, class_labels, ["Skeletonema"], [5], [np.nan])
    counter = _StubCounter(handled=["Skeletonema"])
    # bin_index has no entry for D1 -> still write an all-sentinel dataset so the
    # schema stays uniform with the no-countable-ROIs case.
    count_mod._count_one_file(h5, {}, counter, overwrite=False)
    with h5py.File(h5, "r") as f:
        np.testing.assert_array_equal(f["chain_count"][:], [-1])


# --- count_main integration --------------------------------------------------

def test_count_main_requires_enabled_block(tmp_path):
    from ifcb_classify.config import InferConfig

    config = InferConfig(input_path=str(tmp_path), output_dir=str(tmp_path), chain_counting=None)
    with pytest.raises(ValueError, match="enabled 'chain_counting' block"):
        count_mod.count_main(config)
