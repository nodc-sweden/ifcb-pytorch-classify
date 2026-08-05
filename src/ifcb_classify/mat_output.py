"""Writer for the IFCB Dashboard / ifcb-analysis v1 class-scores MATLAB format.

Produces ``{bin_lid}_class_v1.mat`` files that are both ingestible by the IFCB
Dashboard (via pyifcb's v1 reader) and processable by
[iRfcb](https://europeanifcbgroup.github.io/iRfcb/) (e.g. ``ifcb_extract_biovolumes``,
``ifcb_summarize_class_counts``). The field set matches iRfcb's own
``ifcb_save_classification`` writer:

* ``class2useTB`` — class labels with a trailing ``"unclassified"`` (pyifcb's v1
  reader drops the last entry, so appending it keeps the real classes intact).
* ``TBscores`` — the ``N x C`` score matrix (double).
* ``roinum`` — ROI target numbers (uint16 column).
* ``TBclass`` — the winning (argmax) class per ROI.
* ``TBclass_above_threshold`` — the threshold-applied class per ROI
  (``"unclassified"`` below its class threshold).
* ``classifierName`` — the classifier model name.

When chain counting ran, a per-ROI ``cell_count`` field (int32, ``-1`` where not
counted) is also written, mirroring the HDF5 output's ``cell_count`` dataset, so
iRfcb's ``ifcb_summarize_cell_counts`` can read chain counts from the ``.mat`` too.

A ``provenance`` struct records what produced the scores (see
:mod:`ifcb_classify.provenance`). Both it and ``cell_count`` are additive: MATLAB
readers resolve variables by name, so one that does not know these simply does
not ask for them.
"""

from pathlib import Path

import numpy as np
from scipy.io import savemat


def write_class_scores_mat(
    output_path: str | Path,
    scores: np.ndarray,
    class_labels: list[str],
    roi_numbers: np.ndarray,
    class_name_auto: list[str],
    class_name: list[str],
    classifier_name: str,
    cell_counts: np.ndarray | None = None,
    provenance: dict[str, str] | None = None,
) -> None:
    """Write an iRfcb/Dashboard-compatible v1 class-scores ``.mat`` file.

    Args:
        output_path: Path for the .mat file (conventionally ``{lid}_class_v1.mat``).
        scores: Float array of shape (N, C) — ROIs x classes.
        class_labels: List of class names, length C. ``"unclassified"`` is appended
            to form ``class2useTB`` so pyifcb's ``[:-1]`` read leaves the real
            classes intact.
        roi_numbers: Integer array of ROI target numbers, length N.
        class_name_auto: Winning (argmax) class per ROI, length N (``TBclass``).
        class_name: Threshold-applied class per ROI, length N
            (``TBclass_above_threshold``).
        classifier_name: Classifier model name (``classifierName``).
        cell_counts: Optional int array of per-ROI cell counts, length N (``-1``
            marks ROIs that were not counted). Written as a ``cell_count`` field
            when provided; omitted otherwise (fully backward compatible).
        provenance: Optional string mapping describing what produced the scores
            (see :mod:`ifcb_classify.provenance`), written as a ``provenance``
            struct. Readers look variables up by name and ignore unknown ones, so
            this is additive in the same way ``cell_count`` already is.
    """
    n_rois, n_classes = scores.shape
    if len(class_labels) != n_classes:
        raise ValueError(f"Expected {n_classes} class labels, got {len(class_labels)}")
    for name, seq in (("ROI numbers", roi_numbers), ("class_name_auto", class_name_auto), ("class_name", class_name)):
        if len(seq) != n_rois:
            raise ValueError(f"Expected {n_rois} {name}, got {len(seq)}")
    if cell_counts is not None and len(cell_counts) != n_rois:
        raise ValueError(f"Expected {n_rois} cell counts, got {len(cell_counts)}")

    # roinum is a uint16 field in this format. A plain cast wraps out-of-range
    # values silently (70000 becomes 4464), which misassociates every score and
    # count in the file, so refuse instead. The h5 and csv-labels outputs use
    # int32 and can carry such a bin.
    roi_numbers = np.asarray(roi_numbers)
    if n_rois and (roi_numbers.min() < 0 or roi_numbers.max() > 65535):
        raise ValueError(
            f"ROI numbers must be between 0 and 65535 to fit the .mat 'roinum' field, "
            f"but this bin spans {roi_numbers.min()} to {roi_numbers.max()}. "
            "Write this bin to h5 or csv-labels instead."
        )

    variables = {
        "class2useTB": np.array([*class_labels, "unclassified"], dtype=object),
        "TBscores": scores.astype(np.float64),
        "roinum": np.asarray(roi_numbers, dtype=np.uint16).reshape(-1, 1),
        "TBclass": np.array(list(class_name_auto), dtype=object).reshape(-1, 1),
        "TBclass_above_threshold": np.array(list(class_name), dtype=object).reshape(-1, 1),
        "classifierName": np.array([[classifier_name]], dtype=object),
    }
    if cell_counts is not None:
        variables["cell_count"] = np.asarray(cell_counts, dtype=np.int32).reshape(-1, 1)
    if provenance:
        variables["provenance"] = {k: str(v) for k, v in provenance.items()}

    savemat(output_path, variables, do_compression=True)
