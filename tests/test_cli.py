import numpy as np
import pytest
from PIL import Image

from ifcb_classify.cli import build_parser, run_cli


def test_train_parser():
    parser = build_parser()
    args = parser.parse_args(["train", "--config", "train.yaml"])
    assert args.command == "train"
    assert args.config == "train.yaml"


def test_train_parser_with_overrides():
    parser = build_parser()
    args = parser.parse_args([
        "train", "--config", "train.yaml",
        "--model", "convnext_tiny",
        "--lr", "0.001",
        "--epochs", "30",
        "--batch-size", "128",
    ])
    assert args.model == "convnext_tiny"
    assert args.lr == 0.001
    assert args.epochs == 30
    assert args.batch_size == 128


def test_infer_parser():
    parser = build_parser()
    args = parser.parse_args(["infer", "--config", "infer.yaml"])
    assert args.command == "infer"


def test_infer_parser_cli_only():
    parser = build_parser()
    args = parser.parse_args([
        "infer",
        "--input", "/path/to/bins",
        "--model", "model.pt",
        "--output", "/path/to/output",
    ])
    assert args.input_path == "/path/to/bins"
    assert args.model_checkpoint == "model.pt"
    assert args.output_dir == "/path/to/output"


def test_chains_train_parser():
    parser = build_parser()
    args = parser.parse_args([
        "chains-train",
        "--class-name", "Skeletonema",
        "--data", "/data/skeletonema",
        "--model", "yolo11x.pt",
        "--epochs", "200",
        "--device", "0",
    ])
    assert args.command == "chains-train"
    assert args.class_name == "Skeletonema"
    assert args.data == "/data/skeletonema"
    assert args.model == "yolo11x.pt"
    assert args.epochs == 200
    assert args.device == "0"


def test_chains_train_parser_with_config():
    parser = build_parser()
    args = parser.parse_args(["chains-train", "--config", "chains.yaml"])
    assert args.command == "chains-train"
    assert args.config == "chains.yaml"


def test_run_cli_chains_train_missing_args_raises():
    with pytest.raises(SystemExit, match="--config or both --class-name and --data"):
        run_cli(["chains-train", "--class-name", "Skeletonema"])


def test_chains_eval_parser():
    parser = build_parser()
    args = parser.parse_args([
        "chains-eval",
        "--weights", "best.pt",
        "--images", "/data/test",
        "--counts-csv", "/data/test/counts.csv",
        "--ious", "0.3,0.5",
        "--limit", "50",
    ])
    assert args.command == "chains-eval"
    assert args.weights == "best.pt"
    assert args.counts_csv == "/data/test/counts.csv"
    assert args.ious == "0.3,0.5"
    assert args.limit == 50


def test_run_cli_chains_eval_missing_args_raises():
    with pytest.raises(SystemExit, match="--config or all of --weights, --images and --counts-csv"):
        run_cli(["chains-eval", "--weights", "best.pt"])


def test_normalise_parser():
    parser = build_parser()
    args = parser.parse_args(["normalise", "--data-dir", "/data"])
    assert args.command == "normalise"
    assert args.data_dir == "/data"


def test_missing_command_raises():
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args([])


def test_run_cli_normalise(tmp_path, capsys):
    classes = ["A", "B"]
    for cls in classes:
        cls_dir = tmp_path / cls
        cls_dir.mkdir()
        for i in range(3):
            img = Image.fromarray(np.zeros((32, 32, 3), dtype=np.uint8))
            img.save(cls_dir / f"img_{i}.png")

    run_cli(["normalise", "--data-dir", str(tmp_path)])
    captured = capsys.readouterr()
    assert "mean:" in captured.out
    assert "std:" in captured.out


def test_run_cli_infer_missing_args_raises():
    with pytest.raises(SystemExit, match="--config or both --input and --model"):
        run_cli(["infer", "--input", "/some/path"])


def test_infer_parser_allow_unsafe():
    parser = build_parser()
    args = parser.parse_args(["infer", "--config", "infer.yaml", "--allow-unsafe"])
    assert args.allow_unsafe is True


def test_infer_parser_allow_unsafe_default():
    parser = build_parser()
    args = parser.parse_args(["infer", "--config", "infer.yaml"])
    assert args.allow_unsafe is False


def test_infer_parser_no_count():
    parser = build_parser()
    args = parser.parse_args(["infer", "--config", "infer.yaml", "--no-count"])
    assert args.no_count is True


def test_infer_parser_no_count_default():
    parser = build_parser()
    args = parser.parse_args(["infer", "--config", "infer.yaml"])
    assert args.no_count is False


def test_list_models_parser():
    parser = build_parser()
    args = parser.parse_args(["list-models"])
    assert args.command == "list-models"


def test_list_models_output(capsys):
    from ifcb_classify.models.registry import available_models

    run_cli(["list-models"])
    out = capsys.readouterr().out
    names = available_models()
    # header reports the count, and every registered model is listed
    assert f"{len(names)} model architectures available" in out
    assert "resnet50" in out
    for name in names:
        assert name in out


def test_available_models_matches_registry():
    from ifcb_classify.models.registry import MODELS, available_models

    assert available_models() == sorted(MODELS)
