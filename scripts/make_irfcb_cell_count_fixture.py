"""Generate a chain-count test fixture for iRfcb's ``ifcb_summarize_cell_counts``.

Writes two bins' worth of class-scores files (``{lid}_class_v1.mat`` and a
matching ``{lid}_class.h5``) with a controlled per-ROI ``cell_count`` distribution
— chains, a single-detected ROI, a zero-detected ROI, and not-counted (``-1``)
ROIs, plus one below-threshold ``unclassified`` ROI. See the sibling ``README.md``
for the per-ROI table and the golden expected summary.

Run with a Python environment that has ``ifcb-classify`` installed:

    python scripts/make_irfcb_cell_count_fixture.py [output_dir]

Default output dir: ``test_data/irfcb_cell_counts``.
"""

import sys
from pathlib import Path

import numpy as np

from ifcb_classify.hdf5_output import resolve_class_names, write_class_scores
from ifcb_classify.mat_output import write_class_scores_mat

CLASSES = ["Skeletonema_marinoi", "Thalassiosira_spp", "Mesodinium_rubrum", "Dinophysis_acuminata"]
THRESHOLDS = np.array([np.nan, np.nan, np.nan, 0.90])  # Dinophysis needs 0.90
CLASSIFIER_NAME = "SMHI ResNet50 V6"
COUNTER_MODELS = {
    "Skeletonema_marinoi": {"weights": "chains_skeletonema_yolo11n.pt", "iou": 0.30, "conf": 0.25},
    "Thalassiosira_spp": {"weights": "chains_thalassiosira_yolo11n.pt", "iou": 0.30, "conf": 0.25},
}

# Per bin: (roi_number, winning_class_index, winning_probability, cell_count)
BINS = {
    "D20230314T001205_IFCB134": [
        (12, 0, 0.95, 5), (28, 0, 0.92, 8), (39, 0, 0.88, 1), (48, 0, 0.85, 0),
        (61, 1, 0.90, 3), (62, 1, 0.87, 4),
        (68, 2, 0.93, -1), (69, 2, 0.91, -1),
    ],
    "D20230314T003836_IFCB134": [
        (5, 0, 0.94, 6), (17, 0, 0.90, 2),
        (23, 1, 0.89, 2),
        (31, 2, 0.92, -1),
        (44, 3, 0.50, -1),  # winner below its 0.90 threshold -> unclassified
    ],
}


def _build(rows):
    """Turn ``(roinum, winner_idx, winner_prob, count)`` rows into scores/roinum/counts."""
    n, c = len(rows), len(CLASSES)
    scores = np.zeros((n, c))
    roinum = np.zeros(n, dtype=np.int32)
    counts = np.zeros(n, dtype=np.int32)
    for i, (rn, wi, wp, count) in enumerate(rows):
        scores[i, :] = (1.0 - wp) / (c - 1)
        scores[i, wi] = wp
        roinum[i] = rn
        counts[i] = count
    return scores, roinum, counts


def main(out_dir):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for lid, rows in BINS.items():
        scores, roinum, counts = _build(rows)
        name_auto, name = resolve_class_names(scores, CLASSES, THRESHOLDS)
        write_class_scores(
            out / f"{lid}_class.h5", scores, CLASSES, roinum, CLASSIFIER_NAME, THRESHOLDS,
            cell_counts=counts, cell_counter_models=COUNTER_MODELS,
        )
        write_class_scores_mat(
            out / f"{lid}_class_v1.mat", scores, CLASSES, roinum, name_auto, name,
            CLASSIFIER_NAME, cell_counts=counts,
        )
        print(f"{lid}: {len(rows)} ROIs, cell_count={counts.tolist()}")
    print(f"Wrote fixture to {out}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "test_data/irfcb_cell_counts")
