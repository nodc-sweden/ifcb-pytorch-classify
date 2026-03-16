import argparse
import logging


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="ifcb-classify", description="IFCB image classification pipeline")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # --- train ---
    train_parser = subparsers.add_parser("train", help="Train a classification model")
    train_parser.add_argument("--config", required=True, help="Path to training YAML config")
    train_parser.add_argument("--data-dir", dest="data_dir")
    train_parser.add_argument("--model")
    train_parser.add_argument("--transform")
    train_parser.add_argument("--lr", type=float)
    train_parser.add_argument("--batch-size", dest="batch_size", type=int)
    train_parser.add_argument("--epochs", type=int)
    train_parser.add_argument("--num-workers", dest="num_workers", type=int)
    train_parser.add_argument("--seed", type=int)
    train_parser.add_argument("--output-dir", dest="output_dir")
    train_parser.add_argument("--tracker", choices=["csv", "mlflow", "wandb", "none"])
    train_parser.add_argument("--image-width", dest="image_width", type=int)
    train_parser.add_argument("--image-height", dest="image_height", type=int)
    train_parser.add_argument("--val-split", dest="val_split", type=float)
    train_parser.add_argument("--mean", type=float)
    train_parser.add_argument("--std", type=float)
    train_parser.add_argument("--dataset-version", dest="dataset_version")
    train_parser.add_argument("--checkpoint-metric", dest="checkpoint_metric")
    train_parser.add_argument("--mlflow-uri", dest="mlflow_uri")
    train_parser.add_argument("--wandb-project", dest="wandb_project")
    train_parser.add_argument("--experiment-name", dest="experiment_name")
    train_parser.add_argument("--min-class-images", dest="min_class_images", type=int, help="Exclude classes with fewer images")
    train_parser.add_argument("--plots", action="store_true", default=None, help="Generate evaluation plots after training")
    train_parser.add_argument("-v", "--verbose", action="store_true")

    # --- infer ---
    infer_parser = subparsers.add_parser("infer", help="Run inference on IFCB bins")
    infer_parser.add_argument("--config", help="Path to inference YAML config")
    infer_parser.add_argument("--input", dest="input_path", help="Path to bin file or directory")
    infer_parser.add_argument("--model", dest="model_checkpoint", help="Path to model checkpoint .pt")
    infer_parser.add_argument("--output", dest="output_dir")
    infer_parser.add_argument("--batch-size", dest="batch_size", type=int)
    infer_parser.add_argument("--num-workers", dest="num_workers", type=int)
    infer_parser.add_argument("--thresholds", dest="thresholds_path")
    infer_parser.add_argument("--threshold-default", dest="threshold_default", type=float)
    infer_parser.add_argument("--device", choices=["auto", "cpu", "cuda"])
    infer_parser.add_argument("--classifier-name", dest="classifier_name")
    infer_parser.add_argument("--classes", dest="classes_path", help="Path to classes.txt (auto-detected from model dir if not set)")
    infer_parser.add_argument("--model-name", dest="model_name", help="Model architecture name for legacy checkpoints (e.g. resnet50)")
    infer_parser.add_argument("--overwrite", action="store_true", default=False, help="Overwrite existing output files (default: skip)")
    infer_parser.add_argument("--num-threads", dest="num_threads", type=int, help="Limit CPU threads for inference (default: all cores)")
    infer_parser.add_argument("-v", "--verbose", action="store_true")

    # --- normalise ---
    norm_parser = subparsers.add_parser("normalise", help="Compute dataset mean and std")
    norm_parser.add_argument("--data-dir", dest="data_dir", required=True)
    norm_parser.add_argument("--transform", default="dataset_fullpad")
    norm_parser.add_argument("--width", type=int, default=224)
    norm_parser.add_argument("--height", type=int, default=224)
    norm_parser.add_argument("-v", "--verbose", action="store_true")

    return parser


def run_cli(args=None) -> None:
    parser = build_parser()
    parsed = parser.parse_args(args)

    log_level = logging.DEBUG if getattr(parsed, "verbose", False) else logging.INFO
    logging.basicConfig(level=log_level, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    if parsed.command == "train":
        _run_train(parsed)
    elif parsed.command == "infer":
        _run_infer(parsed)
    elif parsed.command == "normalise":
        _run_normalise(parsed)


def _run_train(parsed) -> None:
    from ifcb_classify.config import TrainConfig, load_config
    from ifcb_classify.train import train_main

    overrides = {k: v for k, v in vars(parsed).items() if k not in ("command", "config", "verbose") and v is not None}
    config = load_config(parsed.config, TrainConfig, overrides)
    train_main(config)


def _run_infer(parsed) -> None:
    from ifcb_classify.config import InferConfig, load_config
    from ifcb_classify.infer import infer_main

    overrides = {k: v for k, v in vars(parsed).items() if k not in ("command", "config", "verbose") and v is not None}

    if parsed.config:
        config = load_config(parsed.config, InferConfig, overrides)
    else:
        if not parsed.input_path or not parsed.model_checkpoint:
            raise SystemExit("Either --config or both --input and --model are required")
        config = InferConfig(**{k: v for k, v in overrides.items() if k in InferConfig.__dataclass_fields__})

    infer_main(config)


def _run_normalise(parsed) -> None:
    from ifcb_classify.normalise import compute_dataset_stats

    mean, std = compute_dataset_stats(
        data_dir=parsed.data_dir,
        transform_name=parsed.transform,
        width=parsed.width,
        height=parsed.height,
    )
    print(f"mean: {mean:.4f}")
    print(f"std: {std:.4f}")
