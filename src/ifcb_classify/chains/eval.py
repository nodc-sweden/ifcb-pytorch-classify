import csv
import logging
from pathlib import Path

import numpy as np

from ifcb_classify.chains.config import ChainEvalConfig

logger = logging.getLogger(__name__)

_CHAINS_EXTRA_HINT = (
    "Chain counting requires the optional 'chains' extra. "
    'Install it with: uv pip install -e ".[chains]"'
)


def compute_count_metrics(manual: np.ndarray, predicted: np.ndarray) -> dict:
    """Count-accuracy metrics comparing predicted vs manual cell counts."""
    manual = np.asarray(manual)
    predicted = np.asarray(predicted)
    if manual.shape != predicted.shape:
        raise ValueError(f"shape mismatch: manual {manual.shape} vs predicted {predicted.shape}")
    if len(manual) == 0:
        raise ValueError("no counts to evaluate")

    err = predicted - manual
    return {
        "n": int(len(manual)),
        "mae": float(np.abs(err).mean()),
        "mean_bias": float(err.mean()),
        "exact_acc": float((err == 0).mean()),
        "within1": float((np.abs(err) <= 1).mean()),
        "total_manual": int(manual.sum()),
        "total_pred": int(predicted.sum()),
    }


def load_counts_csv(path: str, file_col: str, count_col: str) -> list[tuple[str, int]]:
    """Load (filename, manual_count) rows from a CSV."""
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None or file_col not in reader.fieldnames or count_col not in reader.fieldnames:
            raise ValueError(
                f"counts CSV must have columns '{file_col}' and '{count_col}'; got {reader.fieldnames}"
            )
        return [(row[file_col], int(row[count_col])) for row in reader]


def evaluate_counts(config: ChainEvalConfig) -> list[dict]:
    """Run the detector over a labelled test set and return per-IoU metrics.

    Returns one summary dict per IoU value (each includes the ``iou`` key).
    """
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise ImportError(_CHAINS_EXTRA_HINT) from exc

    rows = load_counts_csv(config.counts_csv, config.file_col, config.count_col)
    if config.limit:
        rows = rows[: config.limit]
    if not rows:
        raise ValueError("no rows to evaluate in counts CSV")

    images_dir = Path(config.images)
    files = [str(images_dir / name) for name, _ in rows]
    manual = np.array([count for _, count in rows])

    model = YOLO(config.weights)

    summary: list[dict] = []
    per_iou_preds: dict[float, np.ndarray] = {}
    for iou in config.ious:
        preds: list[int] = []
        for i in range(0, len(files), 64):
            for result in model(files[i : i + 64], iou=iou, conf=config.conf, verbose=False):
                preds.append(len(result.boxes))
        predicted = np.array(preds)
        metrics = compute_count_metrics(manual, predicted)
        metrics["iou"] = float(iou)
        summary.append(metrics)
        per_iou_preds[float(iou)] = predicted
        logger.info(
            "iou=%.2f  MAE=%.3f  exact=%.1f%%  within1=%.1f%%  total %d vs %d",
            iou, metrics["mae"], 100 * metrics["exact_acc"], 100 * metrics["within1"],
            metrics["total_manual"], metrics["total_pred"],
        )

    if config.output:
        _write_results_csv(config.output, rows, manual, per_iou_preds)
        logger.info("Per-image results written to %s", config.output)

    return summary


def _write_results_csv(output: str, rows, manual, per_iou_preds) -> None:
    ious = sorted(per_iou_preds)
    header = ["file_name", "manual_count"]
    for iou in ious:
        header += [f"pred_iou_{iou}", f"diff_iou_{iou}"]
    with open(output, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for j, (name, _) in enumerate(rows):
            row = [name, int(manual[j])]
            for iou in ious:
                pred = int(per_iou_preds[iou][j])
                row += [pred, pred - int(manual[j])]
            writer.writerow(row)
