"""Build the chain-count test fixture for iRfcb's ``ifcb_summarize_cell_counts``.

The fixture is a curated **subset of real inference output** — two bins classified
by ``SMHI-NIVA-SYKE-SAMS-SZN-ResNet50-V6`` with genuine YOLO chain counts. Every
row's ROI number, scores, thresholded class and ``cell_count`` are copied verbatim
from the classifier's own ``{lid}_class.h5``; nothing is hand-authored. This
matters: an earlier version of this fixture used invented labels and counts, and
because the numbers looked biological someone reasoned from them and reached a
wrong conclusion. Deriving the fixture from real files is what stops that from
recurring — this script cannot manufacture biology, it can only carry real rows
across.

For each bin it writes both formats from one source of truth so they stay
consistent:

* ``{lid}_class.h5``    — class_scores v3 HDF5 (the richer canonical format).
* ``{lid}_class_v1.mat`` — the iRfcb/Dashboard ``.mat`` (the file under test).

``KEEP`` lists, per bin, the real ROI numbers to retain. They are *selected*, not
invented: the selection preserves the fixture's full discriminating power (a long
chain, a mid chain, a single cell, a counted-but-empty ``0`` box, ``-1``
not-counted ROIs of several taxa, and threshold-demoted ``unclassified`` rows)
while keeping the files small. ROI numbers are kept verbatim, so each one still
resolves to a real image in the raw ``.roi``.

The small subset ``.h5`` files are committed (real and only ~50 KB each) as the
fixture's source of truth; the ``.mat`` is regenerated from them by this script.
So the fixture is self-contained in the repo and needs no external data for normal
use — only re-curating *which* ROIs it keeps calls for the full bins.

Usage (needs ``ifcb-classify`` installed)::

    # default: rebuild the .mat from the existing subset .h5 in output_dir
    python scripts/make_irfcb_cell_count_fixture.py

    # re-curate the subset from the full, un-subset real bins
    python scripts/make_irfcb_cell_count_fixture.py --source /path/to/full/bins

With no ``--source``, the script reads the subset ``{lid}_class.h5`` already in
``output_dir`` and just refreshes the ``.mat`` from it — no multi-MB originals
needed. Point ``--source`` at the full bins only to change *which* ROIs the subset
keeps (the ``KEEP`` list below); that path is a caller's argument, never hard-coded.
If no source ``.h5`` can be found, the script errors rather than fabricating.
"""

import argparse
from pathlib import Path

import h5py
import numpy as np

from ifcb_classify.hdf5_output import resolve_class_names, write_class_scores
from ifcb_classify.mat_output import write_class_scores_mat

OUT_DEFAULT = "test_data/irfcb_cell_counts"

# Per bin: the real ROI numbers to keep, chosen to cover every distinct case the
# summariser must handle. Order is irrelevant to the summaries; sorted for
# readability. See the sibling README for the per-ROI table these produce.
KEEP = {
    # No cell_count==0 ROI exists in this bin (the counter never returned an
    # empty box here); the 0-box case is covered by the other bin.
    "D20230314T001205_IFCB134": [
        2, 3, 4, 5, 8, 10, 11, 24, 122, 135, 143, 447, 601, 660, 689,
    ],
    "D20230314T003836_IFCB134": [
        2, 3, 6, 9, 16, 19, 24, 62, 86, 93, 100, 194, 201, 337, 504, 644, 873, 1025,
    ],
}


def _strs(arr):
    """Decode an h5py string dataset to a plain list of ``str``."""
    return [x.decode() if isinstance(x, bytes) else x for x in arr]


def _read_bin(path):
    """Read the arrays and provenance this fixture needs from a class_scores .h5."""
    with h5py.File(path, "r") as f:
        return {
            "scores": f["output_scores"][:],
            "class_labels": _strs(f["class_labels"][:]),
            "roi_numbers": f["roi_numbers"][:].astype(np.int32),
            "classifier_name": _strs(f["classifier_name"][:])[0],
            "thresholds": f["thresholds"][:],
            "cell_count": f["cell_count"][:].astype(np.int32),
            "cell_counter_models": f.attrs.get("cell_counter_models"),
        }


def _subset(data, keep):
    """Select the ``keep`` ROIs (in ``keep`` order), erroring on any that are absent."""
    pos = {int(r): i for i, r in enumerate(data["roi_numbers"])}
    missing = [r for r in keep if r not in pos]
    if missing:
        raise SystemExit(f"ROIs {missing} not present in source bin — cannot subset")
    idx = [pos[r] for r in keep]
    out = dict(data)
    out["scores"] = data["scores"][idx]
    out["roi_numbers"] = data["roi_numbers"][idx]
    out["cell_count"] = data["cell_count"][idx]
    return out


def _resolve_source(lid, source_dir, out_dir):
    """Find the ``.h5`` to build from: the explicit ``--source`` dir when given,
    else the existing subset in ``out_dir``."""
    search = [source_dir, out_dir] if source_dir else [out_dir]
    for d in search:
        p = Path(d) / f"{lid}_class.h5"
        if p.exists():
            return p
    tried = ", ".join(str(Path(d) / f"{lid}_class.h5") for d in search)
    raise SystemExit(
        f"No source .h5 for {lid} (looked at: {tried}). Pass --source pointing at "
        "the full real bins to re-curate."
    )


def main(out_dir, source_dir):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for lid, keep in KEEP.items():
        keep = sorted(keep)
        src = _resolve_source(lid, source_dir, out_dir)
        data = _read_bin(src)

        # Curate vs. refresh is decided by the source's *contents*, not its folder:
        # a full bin (superset of KEEP) is subset and its .h5 (re)written; a source
        # that already holds exactly KEEP is used as-is and only its .mat refreshed.
        present = sorted(int(r) for r in data["roi_numbers"])
        if present == keep:
            sub, curating = data, False
        elif set(keep) <= set(present):
            sub, curating = _subset(data, keep), True
        else:
            missing = sorted(set(keep) - set(present))
            raise SystemExit(f"Source {src} is missing KEEP ROIs {missing} for {lid}")

        # class_name(_auto) are recomputed from the real scores and thresholds, so
        # the two output formats agree by construction. Guard the invariant the
        # whole fixture exists to protect: a counted 'unclassified' ROI is
        # impossible in real output.
        name_auto, name = resolve_class_names(sub["scores"], sub["class_labels"], sub["thresholds"])
        bad = [int(r) for r, n, c in zip(sub["roi_numbers"], name, sub["cell_count"], strict=True)
               if n == "unclassified" and c != -1]
        if bad:
            raise SystemExit(f"{lid}: 'unclassified' ROIs {bad} carry a cell_count != -1")

        # Only (re)write the .h5 when curating from a full bin; when the source is
        # already the subset, leave it untouched and just refresh the .mat from it.
        if curating:
            write_class_scores(
                out / f"{lid}_class.h5", sub["scores"], sub["class_labels"],
                sub["roi_numbers"], sub["classifier_name"], sub["thresholds"],
                cell_counts=sub["cell_count"],
                cell_counter_models=_models_dict(sub["cell_counter_models"]),
            )
        mat_path = out / f"{lid}_class_v1.mat"
        write_class_scores_mat(
            mat_path, sub["scores"], sub["class_labels"],
            sub["roi_numbers"], name_auto, name, sub["classifier_name"],
            cell_counts=sub["cell_count"],
        )
        _normalize_mat_header(mat_path)
        counted = int((sub["cell_count"] >= 1).sum())
        print(f"{lid}: {len(keep)} ROIs ({counted} counted) from {src}")
    print(f"Wrote fixture to {out}")


def _models_dict(raw):
    """The cell_counter_models attr is stored as a JSON string; parse for re-write."""
    import json
    return json.loads(raw) if isinstance(raw, str) else raw


def _normalize_mat_header(path):
    """Overwrite the 116-byte MAT text header (bytes 0-115 only; the version and
    endian markers at 124-127 are untouched), which ``savemat`` stamps with the
    wall clock. Making the byte image depend only on the data lets ``md5sum``
    serve as a regeneration check: an unchanged checksum proves a re-run produced
    identical data, which a timestamped header would otherwise mask."""
    text = b"MATLAB 5.0 MAT-file, generated by make_irfcb_cell_count_fixture.py"
    with open(path, "r+b") as f:
        f.write(text.ljust(116, b" "))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("output_dir", nargs="?", default=OUT_DEFAULT)
    ap.add_argument("--source", default=None,
                    help="directory holding the full real {lid}_class.h5 bins, to "
                         "re-curate the subset; omit to rebuild the .mat from the "
                         "existing subset .h5 in output_dir")
    args = ap.parse_args()
    main(args.output_dir, args.source)
