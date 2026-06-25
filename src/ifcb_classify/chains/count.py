"""Count-only pass: add chain counts to already-classified bins.

``count_main`` backfills the additive ``chain_count`` dataset onto existing
``{lid}_class.h5`` files without re-running the classifier. It reuses the final
(thresholded) ``class_name`` already stored in each file to decide which ROIs to
count, reads only those ROIs' pixels from the matching raw ``.roi`` bin, runs the
configured :class:`~ifcb_classify.chains.counter.ChainCounter`, and writes the
counts back into the same file in place.

This is the path to take when classifications are final but you want counts —
e.g. after training a new detector — so you don't pay the (dominant) ResNet cost
again. It reuses the same inference YAML as ``infer`` (``input_path`` = raw bins,
``output_dir`` = the directory of existing class-score files, ``chain_counting``
= the detector block).
"""

import json
import logging
from pathlib import Path

import h5py
import numpy as np

from ifcb_classify.config import InferConfig
from ifcb_classify.data.ifcb_bin import get_bin_lid, iter_bin_images

logger = logging.getLogger(__name__)


def count_main(config: InferConfig) -> None:
    """Backfill chain counts onto existing class-score files from a resolved config.

    Walks every ``*_class.h5`` under ``config.output_dir``, counts cells for the
    ROIs whose stored ``class_name`` is configured for counting, and writes the
    ``chain_count`` dataset back in place. Files that already carry counts are
    skipped unless ``config.overwrite`` is set. Raises :class:`ValueError` if the
    config has no enabled ``chain_counting`` block.
    """
    if config.num_threads is not None:
        import torch

        torch.set_num_threads(config.num_threads)
        logger.info("CPU threads limited to %d", config.num_threads)

    counter = _build_counter(config)

    scores_dir = Path(config.output_dir)
    if not scores_dir.is_dir():
        raise FileNotFoundError(f"Class-scores directory not found: {scores_dir}")

    bin_index = _index_bins(Path(config.input_path))
    logger.info("Indexed %d raw bins under %s", len(bin_index), config.input_path)

    h5_files = sorted(scores_dir.glob("*_class.h5"))
    if not h5_files:
        logger.warning("No *_class.h5 files found in %s", scores_dir)
        return

    for h5_path in h5_files:
        _count_one_file(h5_path, bin_index, counter, config.overwrite)


def _build_counter(config: InferConfig):
    """Construct a ChainCounter, raising if the config has no enabled detector."""
    from ifcb_classify.chains.config import ChainCountingConfig
    from ifcb_classify.chains.counter import ChainCounter

    cc_config = ChainCountingConfig.from_dict(config.chain_counting or {})
    if not cc_config.enabled:
        raise ValueError(
            "chains-count requires an enabled 'chain_counting' block in the config"
        )
    counter = ChainCounter(cc_config)
    logger.info("Chain counting enabled for: %s", ", ".join(sorted(cc_config.models)))
    return counter


def _index_bins(input_path: Path) -> dict[str, Path]:
    """Map bin LID -> path to a raw bin file, for the input file or directory.

    For a directory, every ``.roi`` found recursively is indexed by its LID so a
    class-score file can be matched back to its source bin.
    """
    if input_path.is_file():
        return {get_bin_lid(input_path): input_path}
    if input_path.is_dir():
        return {get_bin_lid(p): p for p in input_path.rglob("*.roi")}
    raise FileNotFoundError(f"Input path not found: {input_path}")


def _count_one_file(h5_path: Path, bin_index: dict[str, Path], counter, overwrite: bool) -> None:
    """Count one class-score file and write its ``chain_count`` dataset in place.

    Skips files that already carry counts (unless ``overwrite``). Files with no
    countable ROIs — and files whose source bin can't be found — still get an
    all-``-1`` dataset, so every processed file ends up with a uniform schema.
    """
    lid = h5_path.name.removesuffix("_class.h5")

    with h5py.File(h5_path, "r") as f:
        if "chain_count" in f and not overwrite:
            logger.info("Skipping (already counted): %s", h5_path.name)
            return
        class_name = [_decode(v) for v in f["class_name"][:]]
        roi_numbers = f["roi_numbers"][:].astype(np.int32)

    counts = np.full(len(class_name), -1, dtype=np.int32)
    gated = {int(roi_numbers[j]): j for j, cn in enumerate(class_name) if counter.handles(cn)}

    if gated:
        bin_path = bin_index.get(lid)
        if bin_path is None:
            # No source bin: still write the all-sentinel dataset (below) so the
            # schema stays uniform with the no-countable-ROIs case.
            logger.warning("No raw bin found for %s — leaving uncounted", lid)
        else:
            images = _load_images(bin_path, set(gated))
            for target, j in gated.items():
                img = images.get(target)
                if img is None:
                    logger.warning("ROI %d not found in bin %s — leaving uncounted", target, lid)
                    continue
                counts[j] = counter.count(img, class_name[j])

    _write_counts(h5_path, counts, counter.models_metadata())
    n_counted = int((counts >= 0).sum())
    total_cells = int(counts[counts >= 0].sum())
    logger.info(
        "Updated: %s (%d ROIs, %d chain-counted, %d cells)",
        h5_path.name, len(class_name), n_counted, total_cells,
    )


def _load_images(bin_path: Path, targets: set[int]) -> dict[int, object]:
    """Read only the requested ROI targets from a raw bin into a target->image map."""
    return {t: img for t, img in iter_bin_images(bin_path) if t in targets}


def _write_counts(h5_path: Path, counts: np.ndarray, models_meta: dict) -> None:
    """Write (or replace) the ``chain_count`` dataset and provenance attribute."""
    with h5py.File(h5_path, "a") as f:
        if "chain_count" in f:
            del f["chain_count"]
        f.create_dataset("chain_count", data=counts.astype(np.int32))
        f.attrs["chain_counter_models"] = json.dumps(models_meta)


def _decode(value) -> str:
    """Decode an HDF5 string value (bytes or str) to a plain ``str``."""
    return value.decode() if isinstance(value, bytes) else str(value)
