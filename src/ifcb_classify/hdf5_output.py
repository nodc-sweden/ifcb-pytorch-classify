from pathlib import Path

import h5py
import numpy as np


def write_class_scores(
    output_path: str | Path,
    scores: np.ndarray,
    class_labels: list[str],
    roi_numbers: np.ndarray,
    classifier_name: str,
    thresholds: np.ndarray,
) -> None:
    """Write IFCB Dashboard class_scores v3 HDF5 file.

    Args:
        output_path: Path for the .h5 file.
        scores: Float64 array of shape (N, C) — ROIs x classes.
        class_labels: List of class names, length C.
        roi_numbers: Integer array of ROI target numbers, length N.
        classifier_name: Name of the classifier model.
        thresholds: Float64 array of per-class thresholds, length C. Use NaN where not set.
    """
    n_rois, n_classes = scores.shape
    if len(class_labels) != n_classes:
        raise ValueError(f"Expected {n_classes} class labels, got {len(class_labels)}")
    if len(roi_numbers) != n_rois:
        raise ValueError(f"Expected {n_rois} ROI numbers, got {len(roi_numbers)}")
    if len(thresholds) != n_classes:
        raise ValueError(f"Expected {n_classes} thresholds, got {len(thresholds)}")


    best_class_idx = np.argmax(scores, axis=1)
    class_name_auto = [class_labels[i] for i in best_class_idx]

    class_name = []
    for j in range(n_rois):
        idx = best_class_idx[j]
        threshold = thresholds[idx]
        if np.isnan(threshold) or scores[j, idx] >= threshold:
            class_name.append(class_labels[idx])
        else:
            class_name.append("unclassified")

    str_dtype = h5py.string_dtype()

    with h5py.File(output_path, "w") as f:
        f.create_dataset("output_scores", data=scores.astype(np.float64))
        f.create_dataset("class_labels", data=class_labels, dtype=str_dtype)
        f.create_dataset("roi_numbers", data=roi_numbers.astype(np.int32))
        f.create_dataset("classifier_name", data=[classifier_name], dtype=str_dtype)
        f.create_dataset("class_name_auto", data=class_name_auto, dtype=str_dtype)
        f.create_dataset("class_name", data=class_name, dtype=str_dtype)
        f.create_dataset("thresholds", data=thresholds.astype(np.float64))
