"""Training entry point and the per-epoch train/validate loop.

``train_main`` is the top-level function the CLI calls for the ``train`` command.
It dispatches to either a single run or a hyperparameter sweep (one run per
combination of swept values). Each run:

1. builds the train/validation datasets, model, optimiser and experiment tracker;
2. runs ``config.epochs`` of training and validation;
3. checkpoints the best model by ``config.checkpoint_metric`` and, when a new
   best is found, recomputes and saves per-class decision thresholds; and
4. optionally renders evaluation plots once the run finishes.

The functions are split so the orchestration (``_train_run``) stays readable and
the actual tensor work lives in the small ``_train_epoch`` / ``_validate_epoch``
helpers.
"""

import logging

import torch
from torch import nn
from torch.utils.data import DataLoader

from ifcb_classify.checkpoint import CheckpointManager
from ifcb_classify.config import TrainConfig, config_to_dict
from ifcb_classify.data.datasets import create_training_datasets
from ifcb_classify.device import get_device
from ifcb_classify.metrics import MetricsCalculator
from ifcb_classify.models.factory import get_model
from ifcb_classify.seed import set_seed
from ifcb_classify.sweep import generate_sweep_runs
from ifcb_classify.thresholds import compute_optimal_thresholds, save_thresholds_and_metrics
from ifcb_classify.plots import generate_evaluation_plots
from ifcb_classify.tracking import create_tracker

logger = logging.getLogger(__name__)


def train_main(config: TrainConfig) -> None:
    """Run training from a resolved :class:`TrainConfig`.

    Seeds RNGs for reproducibility, auto-selects the device (GPU if available),
    and dispatches to a hyperparameter sweep when ``config.sweep_params`` is set,
    otherwise to a single run.
    """
    set_seed(config.seed)
    device = get_device("auto")
    logger.info("Using device: %s", device)

    if config.sweep_params:
        _train_sweep(config, device)
    else:
        _train_single(config, device)


def _train_sweep(config: TrainConfig, device: torch.device) -> None:
    """Run one training run per combination in the sweep grid.

    For each combination yielded by :func:`generate_sweep_runs`, a fresh
    ``TrainConfig`` is built by overlaying the swept values onto the base config
    (only keys that are real config fields are applied).
    """
    sweep_params = config.sweep_params
    for run in generate_sweep_runs(sweep_params):
        run_dict = run._asdict()
        run_config = TrainConfig(
            **{
                **config_to_dict(config),
                **{k: v for k, v in run_dict.items() if k in TrainConfig.__dataclass_fields__},
            }
        )
        run_name = _build_run_name(run_config, run_dict)
        _train_run(run_config, device, run_name, run_dict)


def _train_single(config: TrainConfig, device: torch.device) -> None:
    """Run a single training run with a descriptive, parameter-encoded run name."""
    run_name = f"{config.dataset_version}-{config.model}_{config.transform}_b{config.batch_size}_lr{config.lr}_e{config.epochs}"
    _train_run(config, device, run_name, config_to_dict(config))


def _train_run(config: TrainConfig, device: torch.device, run_name: str, run_params: dict) -> None:
    """Execute one full training run.

    Builds the datasets, data loaders, model, loss, optimiser, experiment
    tracker, checkpoint manager and metrics calculator, then delegates the epoch
    loop to :func:`_run_training_loop` and the post-run work (plots, tracker
    shutdown) to :func:`_finalize_run`. ``run_params`` is the flat dict logged to
    the experiment tracker for this run.
    """
    logger.info("Starting run: %s", run_name)

    data = create_training_datasets(
        data_dir=config.data_dir,
        transform_name=config.transform,
        width=config.image_width,
        height=config.image_height,
        val_split=config.val_split,
        mean=config.mean,
        std=config.std,
        seed=config.seed,
        min_class_images=config.min_class_images,
        manual_include_classes=config.manual_include_classes,
    )
    class_names = data["class_names"]
    num_classes = data["num_classes"]

    train_loader = DataLoader(
        data["train"],
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
    )
    val_loader = DataLoader(
        data["val"],
        batch_size=config.batch_size,
        num_workers=config.num_workers,
    )

    model = get_model(config.model, num_classes).to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)

    tracker = create_tracker(
        config.tracker,
        output_dir=config.output_dir,
        mlflow_uri=config.mlflow_uri,
        wandb_project=config.wandb_project,
        experiment_name=config.experiment_name,
    )
    checkpoint_mgr = CheckpointManager(
        output_dir=config.output_dir,
        metric_name=config.checkpoint_metric,
    )
    metrics_calc = MetricsCalculator(num_classes)

    tracker.begin_run(run_name, run_params)

    all_epoch_metrics, best_class_metrics, best_confusion_matrix = _run_training_loop(
        config, model, train_loader, val_loader, loss_fn, optimizer,
        device, tracker, checkpoint_mgr, metrics_calc, class_names, run_name,
    )

    _finalize_run(config, run_name, all_epoch_metrics, best_confusion_matrix, class_names, best_class_metrics, tracker)


def _run_training_loop(
    config, model, train_loader, val_loader, loss_fn, optimizer,
    device, tracker, checkpoint_mgr, metrics_calc, class_names, run_name,
):
    """Run the epoch loop and return ``(epoch_metrics, best_class_metrics, best_cm)``.

    For every epoch it trains, validates, logs metrics and the confusion matrix
    to the tracker, and asks the checkpoint manager to save if the monitored
    metric improved. Each time a new best is saved it also recomputes per-class
    optimal thresholds on the validation set and remembers that epoch's confusion
    matrix and class metrics (used later for plotting). ``best_class_metrics`` and
    ``best_cm`` are ``None`` if no checkpoint was ever saved.
    """
    all_epoch_metrics: list[dict] = []
    best_class_metrics: dict | None = None
    best_confusion_matrix = None

    for epoch in range(1, config.epochs + 1):
        train_loss, train_acc = _train_epoch(model, train_loader, loss_fn, optimizer, device, config.model)
        val_loss, val_acc = _validate_epoch(model, val_loader, loss_fn, device, metrics_calc)

        results = metrics_calc.compute()
        log_data = {
            "train_loss": train_loss,
            "train_accuracy": train_acc,
            "val_loss": val_loss,
            "val_accuracy": val_acc,
            "precision": results.precision,
            "recall": results.recall,
            "f1": results.f1,
            "weighted_f1": results.weighted_f1,
            "auprc": results.auprc,
            "auroc": results.auroc,
        }

        all_epoch_metrics.append(log_data)
        tracker.log_metrics(log_data, step=epoch)
        tracker.log_confusion_matrix(results.confusion_matrix.numpy(), class_names, step=epoch)

        metric_value = getattr(results, config.checkpoint_metric)
        saved = checkpoint_mgr.maybe_save(
            model, metric_value, run_name, epoch, class_names, config_to_dict(config)
        )

        if saved:
            logger.info("Computing per-class optimal thresholds...")
            thresholds, class_metrics = compute_optimal_thresholds(model, val_loader, device, class_names)
            save_thresholds_and_metrics(config.output_dir, run_name, epoch, class_names, thresholds, class_metrics)
            best_class_metrics = class_metrics
            best_confusion_matrix = results.confusion_matrix.numpy()

        logger.info(
            "Epoch %d/%d — train_loss=%.4f, val_loss=%.4f, %s=%.4f%s",
            epoch,
            config.epochs,
            train_loss,
            val_loss,
            config.checkpoint_metric,
            metric_value,
            " [saved]" if saved else "",
        )

        metrics_calc.reset()

    return all_epoch_metrics, best_class_metrics, best_confusion_matrix


def _finalize_run(config, run_name, all_epoch_metrics, best_confusion_matrix, class_names, best_class_metrics, tracker):
    """Render evaluation plots (if enabled), close the tracker and free GPU memory."""
    if config.plots and best_confusion_matrix is not None:
        logger.info("Generating evaluation plots...")
        generate_evaluation_plots(
            output_dir=config.output_dir,
            run_name=run_name,
            epoch_metrics=all_epoch_metrics,
            confusion_matrix=best_confusion_matrix,
            class_names=class_names,
            class_metrics=best_class_metrics,
        )

    tracker.end_run()
    torch.cuda.empty_cache()
    logger.info("Run complete: %s", run_name)


def _train_epoch(model, loader, loss_fn, optimizer, device, model_name):
    """Run one training pass over ``loader`` and return ``(avg_loss, accuracy)``.

    ``inception_v3`` is special-cased: in training mode it returns a
    ``(logits, aux_logits)`` tuple, so we discard the auxiliary output here.
    """
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad(set_to_none=True)

        if model_name == "inception_v3":
            preds, _ = model(images)
        else:
            preds = model(images)

        loss = loss_fn(preds, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        correct += preds.argmax(dim=1).eq(labels).sum().item()
        total += labels.size(0)

    if total == 0:
        return 0.0, 0.0
    return total_loss / len(loader), correct / total


def _validate_epoch(model, loader, loss_fn, device, metrics_calc):
    """Run one validation pass and return ``(avg_loss, accuracy)``.

    Also feeds predictions and labels to ``metrics_calc`` so the richer metrics
    (precision/recall/F1/AUROC/confusion matrix) can be computed afterwards.
    """
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            preds = model(images)
            total_loss += loss_fn(preds, labels).item()
            correct += preds.argmax(dim=1).eq(labels).sum().item()
            total += labels.size(0)
            metrics_calc.update(preds, labels)

    if total == 0:
        return 0.0, 0.0
    return total_loss / len(loader), correct / total


def _build_run_name(config: TrainConfig, run_dict: dict) -> str:
    """Build a sweep run name from the dataset version and the swept values."""
    parts = [config.dataset_version]
    for k, v in run_dict.items():
        parts.append(f"{k}{v}")
    return "-".join(parts)
