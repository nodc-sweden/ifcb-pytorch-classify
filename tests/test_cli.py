import pytest
import numpy as np
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
