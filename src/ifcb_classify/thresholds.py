"""Per-class decision thresholds derived from the validation set.

A single global cutoff is wrong for imbalanced multiclass data, so after each
best epoch the pipeline computes an F1-optimal threshold *per class* from the
precision-recall curve (:func:`compute_optimal_thresholds`), saves them with
metrics and a ``classes.txt`` (:func:`save_thresholds_and_metrics`), and at
inference time loads them back (:func:`load_thresholds_json`). ROIs whose top
score falls below the class threshold are labelled ``"unclassified"`` (see
:mod:`ifcb_classify.hdf5_output`).
"""

import json
import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score, precision_recall_curve, precision_score, recall_score

logger = logging.getLogger(__name__)

# Key recording which transform the validation split was read through. It was
# introduced by the same change that stopped augmenting that split (see the
# CHANGELOG entry "Inference no longer applies the training augmentation"), so
# its presence is what dates a thresholds file: written by that release or later
# if it is there, earlier if it is not. Nothing else in the file distinguishes
# the two, which is why the key exists rather than a version comparison — a
# thresholds JSON records no version, and a hand-written one never did.
VALIDATION_TRANSFORM_KEY = "validation_transform"


def thresholds_fitted_on_augmented(path: str | Path) -> bool:
    """Whether a thresholds JSON was fitted against an augmented validation split.

    True when the file records no ``"validation_transform"``, whether the key is
    absent or present but null. Every file written before that split stopped
    being augmented lacks it, as does any hand-written one, and those thresholds
    were fitted on randomly jittered and flipped images.

    This is a check you can run by eye — look for ``"validation_transform"`` in
    the JSON with a transform name against it — which is deliberate, since the
    file carries no version of its own.
    An unreadable or non-JSON file returns False: this only drives an advisory
    warning, and failing to parse it is not evidence of anything.
    """
    try:
        with open(path) as f:
            return json.load(f).get(VALIDATION_TRANSFORM_KEY) is None
    except (OSError, json.JSONDecodeError):
        return False


def compute_optimal_thresholds(
    model: torch.nn.Module,
    val_loader,
    device: torch.device,
    class_names: list[str],
) -> tuple[np.ndarray, dict]:
    """Compute per-class optimal F1 thresholds from the validation set.

    Returns (thresholds_array, class_metrics_dict).
    """
    all_scores = []
    all_labels = []

    model.eval()
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            logits = model(images)
            probs = F.softmax(logits, dim=1).cpu()
            all_scores.append(probs)
            all_labels.append(labels)

    all_scores = torch.cat(all_scores).numpy()
    all_labels = torch.cat(all_labels).numpy()
    num_classes = all_scores.shape[1]

    optimal_thresholds = []
    class_metrics = {}

    for c in range(num_classes):
        y_true = (all_labels == c).astype(int)
        y_score = all_scores[:, c]
        precisions, recalls, thresholds = precision_recall_curve(y_true, y_score)
        f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-10)
        best_idx = np.argmax(f1_scores)
        best_thresh = float(thresholds[best_idx]) if best_idx < len(thresholds) else 0.5
        optimal_thresholds.append(best_thresh)

        y_pred = (y_score >= best_thresh).astype(int)
        class_metrics[class_names[c]] = {
            "class_name": class_names[c],
            "threshold": best_thresh,
            "f1": float(f1_score(y_true, y_pred)),
            "precision": float(precision_score(y_true, y_pred, zero_division=0)),
            "recall": float(recall_score(y_true, y_pred, zero_division=0)),
            "support": int(y_true.sum()),
        }

    return np.array(optimal_thresholds, dtype=np.float64), class_metrics


def save_thresholds_and_metrics(
    output_dir: str | Path,
    run_name: str,
    best_epoch: int,
    class_names: list[str],
    thresholds: np.ndarray,
    class_metrics: dict,
    validation_transform: str | None = None,
) -> Path:
    """Save per-class thresholds, metrics JSON, and classes.txt.

    ``validation_transform`` records which transform the validation split was read
    through, and dates the file: only releases that stopped augmenting that split
    write it. Its absence is what marks thresholds fitted against randomly
    jittered and flipped images — see :data:`VALIDATION_TRANSFORM_KEY` and
    :func:`thresholds_fitted_on_augmented`.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    metrics_output = {
        "model_name": run_name,
        "best_epoch": best_epoch,
        "num_classes": len(class_names),
        VALIDATION_TRANSFORM_KEY: validation_transform,
        "class_metrics": class_metrics,
        "macro_F1": float(np.mean([m["f1"] for m in class_metrics.values()])),
        "weighted_F1": float(np.average(
            [m["f1"] for m in class_metrics.values()],
            weights=[m["support"] for m in class_metrics.values()],
        )),
    }

    json_path = output_path / f"{run_name}_thresholds_and_metrics.json"
    with open(json_path, "w") as f:
        json.dump(metrics_output, f, indent=4)
    logger.info("Saved thresholds and metrics: %s", json_path.name)

    classes_path = output_path / f"{run_name}_classes.txt"
    with open(classes_path, "w") as f:
        f.writelines(f"{name}\n" for name in class_names)
    logger.info("Saved class list: %s", classes_path.name)

    return json_path


def load_thresholds_json(path: str | Path, class_names: list[str]) -> np.ndarray:
    """Load per-class thresholds from the JSON format produced by training."""
    with open(path) as f:
        data = json.load(f)

    class_metrics = data["class_metrics"]
    class_name_to_idx = {name: i for i, name in enumerate(class_names)}
    thresholds = np.full(len(class_names), np.nan, dtype=np.float64)
    for key, metrics in class_metrics.items():
        if key in class_name_to_idx:
            thresholds[class_name_to_idx[key]] = metrics["threshold"]
        else:
            # Legacy format: keys are integer indices as strings
            try:
                idx = int(key)
            except ValueError:
                logger.warning("Unknown class in thresholds file: %s", key)
                continue
            if idx < len(thresholds):
                thresholds[idx] = metrics["threshold"]
    return thresholds
