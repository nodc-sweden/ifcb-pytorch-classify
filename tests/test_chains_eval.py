import sys
import types
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from ifcb_classify.chains.config import ChainEvalConfig
from ifcb_classify.chains.eval import compute_count_metrics, evaluate_counts, load_counts_csv


# --- metrics ----------------------------------------------------------------

def test_compute_count_metrics_perfect():
    m = compute_count_metrics(np.array([3, 5, 2]), np.array([3, 5, 2]))
    assert m["mae"] == 0.0
    assert m["exact_acc"] == 1.0
    assert m["within1"] == 1.0
    assert m["mean_bias"] == 0.0
    assert m["total_manual"] == 10
    assert m["total_pred"] == 10


def test_compute_count_metrics_errors():
    m = compute_count_metrics(np.array([5, 5, 5, 5]), np.array([5, 6, 7, 3]))
    # diffs: 0, +1, +2, -2 -> MAE 1.25, exact 1/4, within1 2/4, bias +0.25
    assert m["mae"] == pytest.approx(1.25)
    assert m["exact_acc"] == pytest.approx(0.25)
    assert m["within1"] == pytest.approx(0.5)
    assert m["mean_bias"] == pytest.approx(0.25)


def test_compute_count_metrics_shape_mismatch():
    with pytest.raises(ValueError, match="shape mismatch"):
        compute_count_metrics(np.array([1, 2]), np.array([1]))


def test_compute_count_metrics_empty():
    with pytest.raises(ValueError, match="no counts"):
        compute_count_metrics(np.array([]), np.array([]))


# --- CSV loading ------------------------------------------------------------

def test_load_counts_csv(tmp_path):
    csv = tmp_path / "counts.csv"
    csv.write_text("file_name,cell_count\na.png,3\nb.png,7\n")
    rows = load_counts_csv(str(csv), "file_name", "cell_count")
    assert rows == [("a.png", 3), ("b.png", 7)]


def test_load_counts_csv_missing_column(tmp_path):
    csv = tmp_path / "counts.csv"
    csv.write_text("name,count\na.png,3\n")
    with pytest.raises(ValueError, match="must have columns"):
        load_counts_csv(str(csv), "file_name", "cell_count")


def test_load_counts_csv_non_integer_count_reports_line(tmp_path):
    csv = tmp_path / "counts.csv"
    csv.write_text("file_name,cell_count\na.png,3\nb.png,not_a_number\n")
    with pytest.raises(ValueError, match="line 3: column 'cell_count'='not_a_number'"):
        load_counts_csv(str(csv), "file_name", "cell_count")


# --- config validation ------------------------------------------------------

def test_eval_config_requires_fields():
    with pytest.raises(ValueError, match="weights is required"):
        ChainEvalConfig(images="x", counts_csv="y")
    with pytest.raises(ValueError, match="images .* is required"):
        ChainEvalConfig(weights="w", counts_csv="y")
    with pytest.raises(ValueError, match="counts_csv .* is required"):
        ChainEvalConfig(weights="w", images="x")


@pytest.mark.parametrize("iou", [-0.1, 1.5])
def test_eval_config_bad_iou(iou):
    with pytest.raises(ValueError, match="iou values must be in"):
        ChainEvalConfig(weights="w", images="x", counts_csv="y", ious=(iou,))


def test_eval_config_bad_conf():
    with pytest.raises(ValueError, match="conf must be in"):
        ChainEvalConfig(weights="w", images="x", counts_csv="y", conf=2.0)


# --- evaluate_counts with a fake ultralytics --------------------------------

class _FakeResult:
    def __init__(self, n):
        self.boxes = list(range(n))


class _FakeYOLO:
    """Predicts a box count parsed from each image filename stem (e.g. img_4 -> 4)."""

    def __init__(self, weights):
        self.weights = weights

    def __call__(self, source, iou=None, conf=None, verbose=False):
        return [_FakeResult(int(Path(p).stem.split("_")[-1])) for p in source]


@pytest.fixture
def fake_ultralytics(monkeypatch):
    mod = types.ModuleType("ultralytics")
    mod.YOLO = _FakeYOLO
    monkeypatch.setitem(sys.modules, "ultralytics", mod)
    return mod


def test_evaluate_counts_end_to_end(tmp_path, fake_ultralytics):
    images = tmp_path / "imgs"
    images.mkdir()
    # filename encodes the predicted count; manual counts deliberately differ by ROI
    spec = {"img_3.png": 3, "img_5.png": 4, "img_2.png": 2}  # file -> manual count
    lines = ["file_name,cell_count"]
    for name, manual in spec.items():
        Image.fromarray(np.zeros((8, 8, 3), dtype=np.uint8)).save(images / name)
        lines.append(f"{name},{manual}")
    csv = tmp_path / "counts.csv"
    csv.write_text("\n".join(lines) + "\n")

    out = tmp_path / "results.csv"
    config = ChainEvalConfig(
        weights=str(tmp_path / "best.pt"),
        images=str(images),
        counts_csv=str(csv),
        ious=(0.3,),
        output=str(out),
    )
    summary = evaluate_counts(config)

    assert len(summary) == 1
    m = summary[0]
    # predicted (from filenames): 3,5,2 ; manual: 3,4,2 -> one off-by-one on img_5
    assert m["iou"] == 0.3
    assert m["total_pred"] == 10
    assert m["total_manual"] == 9
    assert m["exact_acc"] == pytest.approx(2 / 3)
    assert m["within1"] == 1.0
    assert out.exists()
    assert "manual_count" in out.read_text()
