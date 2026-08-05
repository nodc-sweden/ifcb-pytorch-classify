import numpy as np
import pytest
import torch

from ifcb_classify.hdf5_output import write_class_scores
from ifcb_classify.provenance import build_provenance, checkpoint_sha256


def test_build_provenance_records_what_determines_a_score():
    p = build_provenance("dataset_squarepad", "resnet50")

    assert p["transform"] == "dataset_squarepad"
    assert p["model_architecture"] == "resnet50"
    for key in ("ifcb_classify_version", "python_version", "torch_version", "torchvision_version"):
        assert p[key]
    # exact type, not isinstance: torch.__version__ is a str SUBCLASS that numpy
    # renders as fixed-width unicode, which h5py cannot store as an attribute.
    assert all(type(v) is str for v in p.values()), {k: type(v) for k, v in p.items()}


def test_checkpoint_sha256_identifies_by_content(tmp_path):
    a, b, c = tmp_path / "a.pt", tmp_path / "b.pt", tmp_path / "c.pt"
    a.write_bytes(b"weights")
    b.write_bytes(b"weights")  # same content, different name
    c.write_bytes(b"other")

    assert checkpoint_sha256(a) == checkpoint_sha256(b)
    assert checkpoint_sha256(a) != checkpoint_sha256(c)


def test_missing_checkpoint_does_not_break_provenance(tmp_path):
    """Provenance is metadata; failing to record it must never fail a run."""
    assert checkpoint_sha256(tmp_path / "nope.pt") is None

    p = build_provenance("dataset_squarepad", "resnet50", tmp_path / "nope.pt")
    assert "checkpoint_sha256" not in p
    assert p["transform"] == "dataset_squarepad"


def test_provenance_written_as_hdf5_attributes(tmp_path):
    out = tmp_path / "D1_class.h5"
    write_class_scores(
        out,
        np.array([[0.7, 0.3]]),
        ["A", "B"],
        np.array([1], dtype=np.int32),
        "my_model",
        np.array([np.nan, np.nan]),
        provenance=build_provenance("dataset_squarepad", "resnet50"),
    )

    import h5py

    with h5py.File(out, "r") as f:
        assert f.attrs["transform"] == "dataset_squarepad"
        assert f.attrs["model_architecture"] == "resnet50"
        assert f.attrs["torch_version"] == torch.__version__
        # still a valid class_scores v3 file
        assert f["output_scores"].shape == (1, 2)
        assert [v.decode() for v in f["class_labels"][:]] == ["A", "B"]


def test_output_without_provenance_is_unchanged(tmp_path):
    """Omitting it must not add attributes — older readers see the same file."""
    out = tmp_path / "D1_class.h5"
    write_class_scores(
        out, np.array([[0.7, 0.3]]), ["A", "B"], np.array([1], dtype=np.int32),
        "my_model", np.array([np.nan, np.nan]),
    )

    import h5py

    with h5py.File(out, "r") as f:
        assert dict(f.attrs) == {}


@pytest.mark.parametrize("runs", [2])
def test_identical_runs_stay_byte_identical(tmp_path, runs):
    """No timestamp: two identical runs must produce comparable files."""
    written = []
    for i in range(runs):
        out = tmp_path / f"run{i}.h5"
        write_class_scores(
            out, np.array([[0.7, 0.3]]), ["A", "B"], np.array([1], dtype=np.int32),
            "my_model", np.array([np.nan, np.nan]),
            provenance=build_provenance("dataset_squarepad", "resnet50"),
        )
        written.append(out)

    import h5py

    attrs = []
    for w in written:
        with h5py.File(w, "r") as f:
            attrs.append(dict(f.attrs))
    assert attrs[0] == attrs[1]


def test_version_comes_from_installed_metadata():
    """pyproject.toml is the single source; this reads it back via the install."""
    from importlib.metadata import version

    assert build_provenance("dataset_squarepad", "resnet50")["ifcb_classify_version"] == version("ifcb-classify")


def test_uninstalled_package_records_unknown(monkeypatch):
    """Running from a bare checkout must still produce a usable record."""
    from importlib.metadata import PackageNotFoundError

    import ifcb_classify.provenance as mod

    def missing(_name):
        raise PackageNotFoundError

    monkeypatch.setattr(mod, "version", missing)
    p = mod.build_provenance("dataset_squarepad", "resnet50")
    assert p["ifcb_classify_version"] == "unknown"
    assert p["torch_version"]
