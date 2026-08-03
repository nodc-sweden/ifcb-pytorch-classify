"""Command-line interface: argument parsing and command dispatch.

This module defines the ``ifcb-classify`` CLI (also reachable as
``python -m ifcb_classify``). :func:`build_parser` declares the subcommands —
``train``, ``infer``, ``chains-count``, ``chains-train``, ``chains-eval``,
``normalise`` and ``list-models`` — and :func:`run_cli` dispatches the parsed
arguments to a small ``_run_*`` handler per command.

Each handler resolves a config object from a ``--config`` YAML file and/or CLI
overrides, then calls into the matching pipeline module. Heavy imports (torch,
ultralytics, the pipeline modules) are deferred into the handlers so that
``--help`` and argument parsing stay fast and don't require optional extras.
Only CLI flags that the user actually set (non-``None``) become config
overrides, so YAML defaults are preserved.
"""

import argparse
import logging


def build_parser() -> argparse.ArgumentParser:
    """Build the top-level argument parser with one subparser per command."""
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
    infer_parser.add_argument("--format", dest="output_format", help="Output format(s): h5 (default), csv, mat, csv-labels, comma-separated (e.g. h5,csv-labels), or 'all'")
    infer_parser.add_argument("--overwrite", action="store_true", default=None, help="Overwrite existing output files (default: skip)")
    infer_parser.add_argument("--num-threads", dest="num_threads", type=int, help="Limit CPU threads for inference (default: all cores)")
    infer_parser.add_argument("--allow-unsafe", dest="allow_unsafe", action="store_true", default=None, help="Allow unsafe checkpoint loading for legacy .pt files")
    infer_parser.add_argument("--no-count", dest="no_count", action="store_true", default=False, help="Disable chain counting even if enabled in the config")
    infer_parser.add_argument("-v", "--verbose", action="store_true")

    # --- chains-count ---
    chains_count_parser = subparsers.add_parser(
        "chains-count",
        help="Add chain counts to already-classified bins without re-running the classifier",
    )
    chains_count_parser.add_argument("--config", help="Path to inference YAML config (with a chain_counting block)")
    chains_count_parser.add_argument("--input", dest="input_path", help="Path to raw bin file or directory (for ROI pixels)")
    chains_count_parser.add_argument("--output", dest="output_dir", help="Directory of existing *_class.h5 files to update")
    chains_count_parser.add_argument("--overwrite", action="store_true", default=None, help="Re-count files that already have chain counts")
    chains_count_parser.add_argument("--num-threads", dest="num_threads", type=int, help="Limit CPU threads (default: all cores)")
    chains_count_parser.add_argument("-v", "--verbose", action="store_true")

    # --- chains-train ---
    chains_train_parser = subparsers.add_parser(
        "chains-train", help="Train a YOLO chain-counting detector for a chain-forming taxon"
    )
    chains_train_parser.add_argument("--config", help="Path to chain-training YAML config")
    chains_train_parser.add_argument("--class-name", dest="class_name", help="Taxon name (e.g. Skeletonema)")
    chains_train_parser.add_argument("--data", help="Path to YOLO data.yaml or dataset directory")
    chains_train_parser.add_argument("--model", help="Pretrained YOLO weights to fine-tune (e.g. yolo11n.pt, yolo11x.pt)")
    chains_train_parser.add_argument("--epochs", type=int)
    chains_train_parser.add_argument("--imgsz", type=int)
    chains_train_parser.add_argument("--batch", type=int)
    chains_train_parser.add_argument("--device", help="'cpu', or GPU index like '0'")
    chains_train_parser.add_argument("--patience", type=int, help="Early-stopping patience")
    chains_train_parser.add_argument("--project", help="Output directory for training runs")
    chains_train_parser.add_argument("--name", help="Run name (default derived from class and model)")
    chains_train_parser.add_argument("-v", "--verbose", action="store_true")

    # --- chains-eval ---
    chains_eval_parser = subparsers.add_parser(
        "chains-eval", help="Validate a chain detector's counts against manual counts"
    )
    chains_eval_parser.add_argument("--config", help="Path to chain-eval YAML config")
    chains_eval_parser.add_argument("--weights", help="Path to detector weights (best.pt)")
    chains_eval_parser.add_argument("--images", help="Directory of test images")
    chains_eval_parser.add_argument("--counts-csv", dest="counts_csv", help="CSV of manual counts")
    chains_eval_parser.add_argument("--conf", type=float, help="Confidence threshold")
    chains_eval_parser.add_argument("--ious", help="Comma-separated NMS IoU values to sweep (e.g. 0.3,0.5,0.7)")
    chains_eval_parser.add_argument("--limit", type=int, help="Evaluate only the first N images (0 = all)")
    chains_eval_parser.add_argument("--output", help="Optional path to write per-image results CSV")
    chains_eval_parser.add_argument("--file-col", dest="file_col", help="Filename column in the CSV")
    chains_eval_parser.add_argument("--count-col", dest="count_col", help="Count column in the CSV")
    chains_eval_parser.add_argument("-v", "--verbose", action="store_true")

    # --- normalise ---
    norm_parser = subparsers.add_parser("normalise", help="Compute dataset mean and std")
    norm_parser.add_argument("--data-dir", dest="data_dir", required=True)
    norm_parser.add_argument("--transform", default="dataset_fullpad")
    norm_parser.add_argument("--width", type=int, default=224)
    norm_parser.add_argument("--height", type=int, default=224)
    norm_parser.add_argument("-v", "--verbose", action="store_true")

    # --- list-models ---
    subparsers.add_parser("list-models", help="List the model architectures accepted by 'train --model'")

    return parser


def run_cli(args=None) -> None:
    """Parse ``args`` (defaults to ``sys.argv``), configure logging, and dispatch.

    ``-v/--verbose`` raises the log level to DEBUG. Each command routes to its
    ``_run_*`` handler.
    """
    parser = build_parser()
    parsed = parser.parse_args(args)

    log_level = logging.DEBUG if getattr(parsed, "verbose", False) else logging.INFO
    logging.basicConfig(level=log_level, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    if parsed.command == "train":
        _run_train(parsed)
    elif parsed.command == "infer":
        _run_infer(parsed)
    elif parsed.command == "chains-count":
        _run_chains_count(parsed)
    elif parsed.command == "chains-train":
        _run_chains_train(parsed)
    elif parsed.command == "chains-eval":
        _run_chains_eval(parsed)
    elif parsed.command == "normalise":
        _run_normalise(parsed)
    elif parsed.command == "list-models":
        _run_list_models(parsed)


def _run_train(parsed) -> None:
    """Handle ``train``: load the training config (with CLI overrides) and run it."""
    from ifcb_classify.config import TrainConfig, load_config
    from ifcb_classify.train import train_main

    overrides = {k: v for k, v in vars(parsed).items() if k not in ("command", "config", "verbose") and v is not None}
    config = load_config(parsed.config, TrainConfig, overrides)
    train_main(config)


def _run_infer(parsed) -> None:
    """Handle ``infer``: build the inference config and run the pipeline.

    Accepts either a ``--config`` file or the ``--input``/``--model`` pair. The
    ``--no-count`` flag strips any ``chain_counting`` block so counting is
    skipped even when the config enables it.
    """
    from ifcb_classify.config import InferConfig, load_config
    from ifcb_classify.infer import infer_main

    overrides = {k: v for k, v in vars(parsed).items() if k not in ("command", "config", "verbose", "no_count") and v is not None}

    if parsed.config:
        config = load_config(parsed.config, InferConfig, overrides)
    else:
        if not parsed.input_path or not parsed.model_checkpoint:
            raise SystemExit("Either --config or both --input and --model are required")
        config = InferConfig(**{k: v for k, v in overrides.items() if k in InferConfig.__dataclass_fields__})

    if parsed.no_count and config.chain_counting:
        from dataclasses import replace
        config = replace(config, chain_counting=None)

    infer_main(config)


def _run_chains_count(parsed) -> None:
    """Handle ``chains-count``: backfill chain counts onto existing class-score files.

    Reuses the inference config (``input_path`` = raw bins, ``output_dir`` = the
    directory of existing ``*_class.h5`` files, plus the ``chain_counting``
    block). Accepts either a ``--config`` file or the ``--input``/``--output``
    pair; an enabled ``chain_counting`` block is required either way.
    """
    from ifcb_classify.chains.count import count_main
    from ifcb_classify.config import InferConfig, load_config

    overrides = {k: v for k, v in vars(parsed).items() if k not in ("command", "config", "verbose") and v is not None}

    if parsed.config:
        config = load_config(parsed.config, InferConfig, overrides)
    else:
        if not parsed.input_path or not parsed.output_dir:
            raise SystemExit("Either --config or both --input and --output are required")
        config = InferConfig(**{k: v for k, v in overrides.items() if k in InferConfig.__dataclass_fields__})

    count_main(config)


def _run_chains_train(parsed) -> None:
    """Handle ``chains-train``: train one YOLO detector and print the best weights.

    Accepts either a ``--config`` file or the ``--class-name``/``--data`` pair.
    """
    from ifcb_classify.chains.config import ChainTrainConfig
    from ifcb_classify.chains.train import train_chain_detector
    from ifcb_classify.config import load_config

    overrides = {k: v for k, v in vars(parsed).items() if k not in ("command", "config", "verbose") and v is not None}

    if parsed.config:
        config = load_config(parsed.config, ChainTrainConfig, overrides)
    else:
        if not parsed.class_name or not parsed.data:
            raise SystemExit("Either --config or both --class-name and --data are required")
        config = ChainTrainConfig(**{k: v for k, v in overrides.items() if k in ChainTrainConfig.__dataclass_fields__})

    best = train_chain_detector(config)
    print(f"Best weights: {best}")


def _run_chains_eval(parsed) -> None:
    """Handle ``chains-eval``: evaluate a detector and print the per-IoU summary.

    The comma-separated ``--ious`` string is parsed into a tuple of floats before
    building the config. Accepts either ``--config`` or all of
    ``--weights``/``--images``/``--counts-csv``.
    """
    from ifcb_classify.chains.config import ChainEvalConfig
    from ifcb_classify.chains.eval import evaluate_counts
    from ifcb_classify.config import load_config

    overrides = {k: v for k, v in vars(parsed).items() if k not in ("command", "config", "verbose") and v is not None}
    if isinstance(overrides.get("ious"), str):
        overrides["ious"] = tuple(float(x) for x in overrides["ious"].split(","))

    if parsed.config:
        config = load_config(parsed.config, ChainEvalConfig, overrides)
    else:
        if not parsed.weights or not parsed.images or not parsed.counts_csv:
            raise SystemExit("Either --config or all of --weights, --images and --counts-csv are required")
        config = ChainEvalConfig(**{k: v for k, v in overrides.items() if k in ChainEvalConfig.__dataclass_fields__})

    summary = evaluate_counts(config)
    _print_eval_summary(summary)


def _print_eval_summary(summary) -> None:
    """Print the chain-eval per-IoU metrics as an aligned text table."""
    print(f"{'IoU':>5} {'MAE':>7} {'Bias':>7} {'Exact':>7} {'Within1':>8} {'Manual':>8} {'Pred':>8}")
    for m in summary:
        print(
            f"{m['iou']:5.2f} {m['mae']:7.3f} {m['mean_bias']:7.2f} "
            f"{m['exact_acc']:7.1%} {m['within1']:8.1%} {m['total_manual']:8d} {m['total_pred']:8d}"
        )


def _run_normalise(parsed) -> None:
    """Handle ``normalise``: compute and print the dataset mean and std."""
    from ifcb_classify.normalise import compute_dataset_stats

    mean, std = compute_dataset_stats(
        data_dir=parsed.data_dir,
        transform_name=parsed.transform,
        width=parsed.width,
        height=parsed.height,
    )
    print(f"mean: {mean:.4f}")
    print(f"std: {std:.4f}")


def _run_list_models(parsed) -> None:
    """Handle ``list-models``: print the model names accepted by ``train --model``."""
    from ifcb_classify.models.registry import available_models

    names = available_models()
    print(f"{len(names)} model architectures available (use with 'train --model <name>'):")
    for name in names:
        print(f"  {name}")
