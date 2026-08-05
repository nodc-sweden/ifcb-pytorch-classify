"""Tests for ``scripts/recompute_thresholds.py``.

The script is the migration path for every model trained before the validation
split stopped being augmented: each owner runs it once, over files that are the
only record of their model's previous operating point. So the destructive edges
are what these test.
"""

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

SCRIPT = Path(__file__).parent.parent / "scripts" / "recompute_thresholds.py"


def _load_script():
    """Import the script by path — ``scripts/`` is not an importable package."""
    spec = importlib.util.spec_from_file_location("recompute_thresholds", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


rt = _load_script()

# The thresholds that "shipped" with the fixture model, deliberately far from
# anything a refit would produce so that a real comparison cannot report no change.
SHIPPED_THRESHOLDS = {"Alpha": 0.11, "Beta": 0.93}


def _shipped_json(**overrides):
    data = {
        "model_name": "myrun",
        "best_epoch": 3,
        "num_classes": 2,
        "class_metrics": {
            name: {"threshold": value, "f1": 0.5, "precision": 0.5, "recall": 0.5, "support": 8}
            for name, value in SHIPPED_THRESHOLDS.items()
        },
        "macro_F1": 0.5,
        "weighted_F1": 0.5,
    }
    data.update(overrides)
    return data


@pytest.fixture
def refit_fixture(tmp_path):
    """A pipeline checkpoint, its training data, and the thresholds that shipped with it."""
    from ifcb_classify.models.factory import get_model

    data_dir = tmp_path / "data"
    rng = np.random.default_rng(0)
    for cls in SHIPPED_THRESHOLDS:
        (data_dir / cls).mkdir(parents=True)
        for i in range(20):
            Image.fromarray(rng.integers(0, 255, (24, 30, 3), dtype=np.uint8)).save(data_dir / cls / f"{i}.png")

    model_dir = tmp_path / "mymodel"
    model_dir.mkdir()
    torch.manual_seed(0)
    model = get_model("resnet18", 2, pretrained=False)
    checkpoint = model_dir / "myrun_best.pt"
    torch.save(
        {
            "state_dict": model.state_dict(),
            "class_names": list(SHIPPED_THRESHOLDS),
            "config": {
                "model": "resnet18",
                "pretrained": False,
                "transform": "dataset_squarepad_augmented",
                "image_width": 32,
                "image_height": 32,
                "data_dir": str(data_dir),
                "val_split": 0.4,
                "seed": 42,
            },
        },
        checkpoint,
    )

    shipped = model_dir / "myrun_thresholds_and_metrics.json"
    shipped.write_text(json.dumps(_shipped_json(), indent=4))
    (model_dir / "myrun_classes.txt").write_text("Alpha\nBeta\n")
    return checkpoint, shipped, tmp_path


def _run(monkeypatch, checkpoint, out, *extra):
    monkeypatch.setattr(
        sys, "argv",
        ["recompute_thresholds.py", "--model", str(checkpoint), "--out", str(out), "--device", "cpu", *extra],
    )
    rt.main()


def _reported_max_delta(output: str) -> float:
    """Pull the 'max' figure out of the 'thresholds changed by' line."""
    line = next(line for line in output.splitlines() if "thresholds changed by" in line)
    return float(line.split("max")[1].strip())


def test_refuses_to_overwrite_the_shipped_thresholds(monkeypatch, refit_fixture):
    """The documented usage must not silently destroy the model's operating point."""
    checkpoint, shipped, _ = refit_fixture
    before = shipped.read_text()

    with pytest.raises(SystemExit, match="--in-place"):
        _run(monkeypatch, checkpoint, checkpoint.parent)

    assert shipped.read_text() == before


def test_in_place_compares_against_the_file_it_replaces(monkeypatch, refit_fixture, capsys):
    """Regression: reading the previous file after writing compared it with itself.

    The write lands on exactly the file being compared against, so a read-after-write
    reported 'median 0.0000, max 0.0000' — no change at all — for a refit that moved
    every threshold. That is the one number the user is told to judge by.
    """
    checkpoint, shipped, _ = refit_fixture

    _run(monkeypatch, checkpoint, checkpoint.parent, "--in-place")

    output = capsys.readouterr().out
    assert _reported_max_delta(output) > 0.05, output
    assert "2/2 classes moved by more than 0.05" in output

    refit = json.loads(shipped.read_text())
    assert refit["validation_transform"] == "dataset_squarepad"
    for name, old in SHIPPED_THRESHOLDS.items():
        assert refit["class_metrics"][name]["threshold"] != pytest.approx(old)


def test_out_elsewhere_leaves_the_shipped_thresholds_alone(monkeypatch, refit_fixture, capsys):
    checkpoint, shipped, tmp_path = refit_fixture
    before = shipped.read_text()

    _run(monkeypatch, checkpoint, tmp_path / "refit")

    output = capsys.readouterr().out
    assert shipped.read_text() == before
    assert _reported_max_delta(output) > 0.05, output
    assert "NOT installed automatically" in output
    assert (tmp_path / "refit" / "myrun_thresholds_and_metrics.json").is_file()


def test_in_place_backs_up_what_it_replaces(monkeypatch, refit_fixture):
    """The move sizes only print after the write, so consent is not informed consent."""
    checkpoint, shipped, _ = refit_fixture
    before = shipped.read_text()

    _run(monkeypatch, checkpoint, checkpoint.parent, "--in-place")

    backup = shipped.with_suffix(".json.bak")
    assert backup.read_text() == before
    # Must not be resolvable as a thresholds file in its own right.
    assert backup not in set(checkpoint.parent.glob("*_thresholds_and_metrics.json"))


def test_in_place_says_the_refit_is_live(monkeypatch, refit_fixture, capsys):
    """--in-place installs the refit, so the closing message must not claim otherwise."""
    checkpoint, _, _ = refit_fixture

    _run(monkeypatch, checkpoint, checkpoint.parent, "--in-place")

    output = capsys.readouterr().out
    assert "NOT installed automatically" not in output
    assert "inference will" in output


def test_previous_thresholds_matched_by_run_name(tmp_path):
    """Several runs in one directory: the comparison must use this model's file."""
    (tmp_path / "other_thresholds_and_metrics.json").write_text(json.dumps({"class_metrics": {}}))
    mine = tmp_path / "myrun_thresholds_and_metrics.json"
    mine.write_text(json.dumps(_shipped_json()))

    found = rt._read_previous(tmp_path, "myrun")

    assert found is not None
    assert found[0] == mine


def test_ambiguous_previous_thresholds_are_not_guessed(tmp_path, capsys):
    """Comparing against an unrelated run would put a meaningless number in front
    of the decision this script exists to inform."""
    for name in ("aaa", "zzz"):
        (tmp_path / f"{name}_thresholds_and_metrics.json").write_text(json.dumps(_shipped_json()))

    assert rt._read_previous(tmp_path, "myrun") is None
    assert "not guessing between them" in capsys.readouterr().out


def test_unreadable_previous_thresholds_do_not_crash(tmp_path, capsys):
    (tmp_path / "myrun_thresholds_and_metrics.json").write_text("{not json")

    assert rt._read_previous(tmp_path, "myrun") is None
    assert "could not read" in capsys.readouterr().out


def test_incomparable_previous_thresholds_report_no_statistic(capsys):
    """A previous file describing other classes yields all-NaN deltas; summarising
    those would print 'nan' rather than say the files are not comparable."""
    previous = (Path("old.json"), _shipped_json(class_metrics={"Gamma": {"threshold": 0.5}}))

    rt._report_change(previous, ["Alpha", "Beta"], np.array([0.4, 0.6]), 0.7)

    output = capsys.readouterr().out
    assert "not comparable" in output
    assert "nan" not in output


def test_legacy_checkpoint_is_refused_with_a_clear_error(tmp_path):
    """A legacy bare state-dict gets a synthesised config with no split to rebuild."""
    legacy = {"model": "resnet18", "image_width": 224, "image_height": 224, "transform": "dataset_squarepad"}

    with pytest.raises(SystemExit, match="val_split"):
        rt._require_training_config(legacy, tmp_path / "legacy_best.pt", None)


def test_data_dir_override_satisfies_a_missing_data_dir(tmp_path):
    """--data-dir supplies what the config lacks; val_split still cannot be guessed."""
    config = {"transform": "dataset_squarepad", "val_split": 0.2}

    rt._require_training_config(config, tmp_path / "m.pt", tmp_path / "data")

    with pytest.raises(SystemExit, match="data_dir"):
        rt._require_training_config({"val_split": 0.2}, tmp_path / "m.pt", None)


def test_unsafe_checkpoint_loading_is_opt_in(monkeypatch, refit_fixture):
    """Unpickling can execute arbitrary code, so it must not be on by default."""
    checkpoint, _, tmp_path = refit_fixture
    seen = {}

    def spy(path, *args, **kwargs):
        seen["allow_unsafe"] = kwargs.get("allow_unsafe")
        raise SystemExit("stop here")

    monkeypatch.setattr(rt, "load_checkpoint", spy)
    with pytest.raises(SystemExit):
        _run(monkeypatch, checkpoint, tmp_path / "refit")
    assert seen["allow_unsafe"] is False

    with pytest.raises(SystemExit):
        _run(monkeypatch, checkpoint, tmp_path / "refit", "--allow-unsafe")
    assert seen["allow_unsafe"] is True
