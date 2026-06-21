import json
from pathlib import Path

import h5py
import numpy as np


def resolve_class_names(
    scores: np.ndarray,
    class_labels: list[str],
    thresholds: np.ndarray,
) -> tuple[list[str], list[str]]:
    """Resolve per-ROI class names from scores.

    Returns ``(class_name_auto, class_name)`` where ``class_name_auto`` is the
    raw argmax label and ``class_name`` applies per-class thresholds, falling
    back to ``"unclassified"`` when the top score is below its threshold.
    """
    best_class_idx = np.argmax(scores, axis=1)
    class_name_auto = [class_labels[i] for i in best_class_idx]

    class_name = []
    for j in range(len(best_class_idx)):
        idx = best_class_idx[j]
        threshold = thresholds[idx]
        if np.isnan(threshold) or scores[j, idx] >= threshold:
            class_name.append(class_labels[idx])
        else:
            class_name.append("unclassified")

    return class_name_auto, class_name


def write_class_scores(
    output_path: str | Path,
    scores: np.ndarray,
    class_labels: list[str],
    roi_numbers: np.ndarray,
    classifier_name: str,
    thresholds: np.ndarray,
    chain_counts: np.ndarray | None = None,
    chain_counter_models: dict | None = None,
) -> None:
    """Write IFCB Dashboard class_scores v3 HDF5 file.

    Args:
        output_path: Path for the .h5 file.
        scores: Float64 array of shape (N, C) — ROIs x classes.
        class_labels: List of class names, length C.
        roi_numbers: Integer array of ROI target numbers, length N.
        classifier_name: Name of the classifier model.
        thresholds: Float64 array of per-class thresholds, length C. Use NaN where not set.
        chain_counts: Optional int array of per-ROI cell counts, length N. ``-1``
            marks ROIs that were not counted (class not configured for counting).
            When omitted, no chain-count data is written (fully backward compatible).
        chain_counter_models: Optional provenance mapping (class -> {weights, iou, conf})
            stored as a JSON attribute when chain_counts is written.
    """
    n_rois, n_classes = scores.shape
    if len(class_labels) != n_classes:
        raise ValueError(f"Expected {n_classes} class labels, got {len(class_labels)}")
    if len(roi_numbers) != n_rois:
        raise ValueError(f"Expected {n_rois} ROI numbers, got {len(roi_numbers)}")
    if len(thresholds) != n_classes:
        raise ValueError(f"Expected {n_classes} thresholds, got {len(thresholds)}")
    if chain_counts is not None and len(chain_counts) != n_rois:
        raise ValueError(f"Expected {n_rois} chain counts, got {len(chain_counts)}")

    class_name_auto, class_name = resolve_class_names(scores, class_labels, thresholds)

    str_dtype = h5py.string_dtype()

    with h5py.File(output_path, "w") as f:
        f.create_dataset("output_scores", data=scores.astype(np.float64))
        f.create_dataset("class_labels", data=class_labels, dtype=str_dtype)
        f.create_dataset("roi_numbers", data=roi_numbers.astype(np.int32))
        f.create_dataset("classifier_name", data=[classifier_name], dtype=str_dtype)
        f.create_dataset("class_name_auto", data=class_name_auto, dtype=str_dtype)
        f.create_dataset("class_name", data=class_name, dtype=str_dtype)
        f.create_dataset("thresholds", data=thresholds.astype(np.float64))
        if chain_counts is not None:
            f.create_dataset("chain_count", data=np.asarray(chain_counts, dtype=np.int32))
            if chain_counter_models is not None:
                f.attrs["chain_counter_models"] = json.dumps(chain_counter_models)
