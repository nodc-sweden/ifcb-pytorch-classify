import pytest

from ifcb_classify.cli import build_parser


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
