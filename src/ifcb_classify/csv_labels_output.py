"""Writer for the ClassiPyR / iRfcb per-ROI class-labels CSV format.

Unlike the dashboard scores CSV (:mod:`ifcb_classify.csv_output`), this is the
*resolved-label* CSV that iRfcb and
[ClassiPyR](https://github.com/EuropeanIFCBGroup/ClassiPyR) consume. iRfcb's
``read_class_file`` parses these columns:

* ``file_name`` — ``{bin_lid}_{roi_number:05d}.png`` (iRfcb derives the ROI number
  from it).
* ``class_name`` — the threshold-applied class (``"unclassified"`` below threshold).
* ``class_name_auto`` — the winning (argmax) class, ignoring thresholds.
* ``score`` — the winning class's confidence.
* ``cell_count`` — *(optional)* per-ROI chain count (``-1`` where not counted),
  written only when counting ran; iRfcb reads it for cell-abundance summaries.

The file is named ``{bin_lid}.csv`` (no ``_class`` suffix) to match iRfcb's
convention: iRfcb resolves a ``.csv``'s sample name by stripping only ``.csv``, so
this is what lets its folder scanners pick the file up under the correct sample.
"""

from pathlib import Path

import numpy as np
import pandas as pd


def write_class_labels_csv(
    output_path: str | Path,
    scores: np.ndarray,
    roi_numbers: np.ndarray,
    class_name_auto: list[str],
    class_name: list[str],
    bin_lid: str,
    cell_counts: np.ndarray | None = None,
) -> None:
    """Write a ClassiPyR/iRfcb-format per-ROI class-labels CSV.

    Args:
        output_path: Path for the .csv file (conventionally ``{bin_lid}.csv``).
        scores: Float array of shape (N, C); the winning score per ROI is stored.
        roi_numbers: Integer array of ROI target numbers, length N.
        class_name_auto: Winning (argmax) class per ROI, length N.
        class_name: Threshold-applied class per ROI, length N.
        bin_lid: Bin identifier used to build each row's ``file_name``.
        cell_counts: Optional int array of per-ROI cell counts, length N (``-1``
            where not counted). Adds a ``cell_count`` column when provided.
    """
    n_rois = scores.shape[0]
    for label, seq in (("ROI numbers", roi_numbers), ("class_name_auto", class_name_auto), ("class_name", class_name)):
        if len(seq) != n_rois:
            raise ValueError(f"Expected {n_rois} {label}, got {len(seq)}")
    if cell_counts is not None and len(cell_counts) != n_rois:
        raise ValueError(f"Expected {n_rois} cell counts, got {len(cell_counts)}")

    columns = {
        "file_name": [f"{bin_lid}_{int(rn):05d}.png" for rn in roi_numbers],
        "class_name": list(class_name),
        "class_name_auto": list(class_name_auto),
        "score": scores.max(axis=1).astype(np.float64),
    }
    if cell_counts is not None:
        columns["cell_count"] = np.asarray(cell_counts, dtype=np.int32)

    pd.DataFrame(columns).to_csv(output_path, index=False)
