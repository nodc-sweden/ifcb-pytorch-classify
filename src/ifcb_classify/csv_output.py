"""Writer for the IFCB Dashboard class-scores CSV format.

Reproduces the CSV the IFCB Dashboard serves at ``{bin}_class_v3.csv``: one row
per ROI, indexed by ``pid`` (``{bin_lid}_{roi_number:05d}``), with one column per
class label holding the softmax score. It is scores-only — the same data as
pyifcb's ``class_scores()`` DataFrame — so it round-trips with the dashboard and
pyifcb ecosystem. Derived labels, thresholds and cell counts are intentionally
omitted to match the dashboard export's columns and ``pid`` layout.
"""

from pathlib import Path

import numpy as np
import pandas as pd


def write_class_scores_csv(
    output_path: str | Path,
    scores: np.ndarray,
    class_labels: list[str],
    roi_numbers: np.ndarray,
    bin_lid: str,
) -> None:
    """Write a dashboard-format class-scores CSV file.

    Args:
        output_path: Path for the .csv file.
        scores: Float array of shape (N, C) — ROIs x classes.
        class_labels: List of class names, length C (the CSV columns).
        roi_numbers: Integer array of ROI target numbers, length N.
        bin_lid: Bin identifier used to build each row's ``pid`` index.
    """
    n_rois, n_classes = scores.shape
    if len(class_labels) != n_classes:
        raise ValueError(f"Expected {n_classes} class labels, got {len(class_labels)}")
    if len(roi_numbers) != n_rois:
        raise ValueError(f"Expected {n_rois} ROI numbers, got {len(roi_numbers)}")

    df = pd.DataFrame(scores.astype(np.float64), columns=class_labels)
    df.index = [f"{bin_lid}_{int(rn):05d}" for rn in roi_numbers]
    df.index.name = "pid"
    df.to_csv(output_path)
