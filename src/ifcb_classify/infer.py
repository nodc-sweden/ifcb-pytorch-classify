"""Inference pipeline: classify raw IFCB bins and write HDF5 class-score files.

``infer_main`` is the entry point the CLI calls for the ``infer`` command. Given
a trained checkpoint and an input path (a single bin or a directory of bins) it:

1. skips work early if every bin already has an output file (unless ``overwrite``);
2. loads the model and rebuilds the exact transform used at training time;
3. resolves per-class decision thresholds and, optionally, a chain counter;
4. runs batched softmax inference over each bin's ROIs; and
5. writes one class-scores file per bin in each requested output format —
   ``h5`` (IFCB Dashboard class_scores v3, the default; see
   :mod:`ifcb_classify.hdf5_output`), ``csv`` (:mod:`ifcb_classify.csv_output`)
   and/or ``mat`` (:mod:`ifcb_classify.mat_output`).

The private ``_classify_*`` / ``_batch_predict`` / ``_write_output`` helpers carry
a long positional argument list; this is deliberate to keep a single linear data
path through the pipeline rather than threading a context object.
"""

import logging
from pathlib import Path

import numpy as np
import torch
import yaml

from ifcb_classify.checkpoint import load_checkpoint
from ifcb_classify.config import InferConfig
from ifcb_classify.data.datasets import build_transform
from ifcb_classify.data.ifcb_bin import get_bin_lid, iter_bin_images, iter_directory_bins
from ifcb_classify.device import get_device
from ifcb_classify.hdf5_output import resolve_class_names, write_class_scores
from ifcb_classify.models.factory import get_model
from ifcb_classify.seed import set_seed

logger = logging.getLogger(__name__)


def infer_main(config: InferConfig) -> None:
    """Run inference from a resolved :class:`InferConfig`.

    Loads the model only after confirming there is pending work, then dispatches
    to the single-file or directory classifier depending on ``input_path``.
    Raises :class:`FileNotFoundError` if the input path is neither a file nor a
    directory.
    """
    if config.num_threads is not None:
        torch.set_num_threads(config.num_threads)
        logger.info("CPU threads limited to %d", config.num_threads)

    input_path = Path(config.input_path)
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    formats_early = config.resolved_formats()
    if not config.overwrite and not _has_pending_bins(input_path, output_dir, formats_early):
        logger.info("No new bins to classify — skipping model load")
        return

    checkpoint = load_checkpoint(
        config.model_checkpoint,
        model_name=config.model_name,
        classes_path=config.classes_path,
        allow_unsafe=config.allow_unsafe,
    )
    train_config = checkpoint["config"]
    class_names = checkpoint["class_names"]
    num_classes = len(class_names)

    device = get_device(config.device)
    logger.info("Using device: %s", device)

    set_seed(train_config.get("seed", 42))
    model = get_model(train_config["model"], num_classes)
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device)
    model.eval()

    transform = build_transform(
        train_config["transform"],
        train_config["image_width"],
        train_config["image_height"],
        train_config.get("mean"),
        train_config.get("std"),
    )

    thresholds = _load_thresholds(config, class_names)
    classifier_name = config.classifier_name or _derive_classifier_name(config, train_config)
    counter = _build_chain_counter(config)
    formats = config.resolved_formats()
    logger.info("Writing output format(s): %s", ", ".join(formats))
    if counter is not None and not ({"h5", "mat", "csv-labels"} & set(formats)):
        logger.warning("Chain counting is enabled but none of 'h5', 'mat', 'csv-labels' is an output format; cell counts are only stored in those formats, so no counts will be written.")

    if input_path.is_file():
        _classify_bin_file(input_path, model, transform, device, config.batch_size, class_names, thresholds, classifier_name, output_dir, config.overwrite, formats, counter)
    elif input_path.is_dir():
        _classify_directory(input_path, model, transform, device, config.batch_size, class_names, thresholds, classifier_name, output_dir, config.overwrite, formats, counter)
    else:
        raise FileNotFoundError(f"Input path not found: {input_path}")


def _build_chain_counter(config: InferConfig):
    """Construct a ChainCounter from the inference config, or None if disabled."""
    block = config.chain_counting
    if not block:
        return None

    from ifcb_classify.chains.config import ChainCountingConfig

    cc_config = ChainCountingConfig.from_dict(block)
    if not cc_config.enabled:
        return None

    from ifcb_classify.chains.counter import ChainCounter

    counter = ChainCounter(cc_config)
    logger.info("Chain counting enabled for: %s", ", ".join(sorted(cc_config.models)))
    return counter


def _has_pending_bins(input_path: Path, output_dir: Path, formats: tuple[str, ...]) -> bool:
    """Check whether any bin is missing at least one requested output file.

    A bin is "done" only when every requested format's file already exists, so a
    bin classified to h5 but not yet to a newly added csv/mat format still counts
    as pending. For directories, uses rglob to find bins in subdirectories.
    """
    if input_path.is_file():
        lid = get_bin_lid(input_path)
        return not _bin_outputs_complete(output_dir, lid, formats)

    if input_path.is_dir():
        # Sort reverse so newest samples (by name, which encodes timestamp) are checked first
        roi_files = sorted(input_path.rglob("*.roi"), reverse=True)
        for roi_file in roi_files:
            lid = get_bin_lid(roi_file)
            if not _bin_outputs_complete(output_dir, lid, formats):
                return True
        return False

    return True


# Per-format output-file suffixes. ``mat`` uses the ``_v1`` name pyifcb's v1
# reader (and thus the dashboard) resolves for MATLAB class-scores files.
# ``csv-labels`` uses a bare ``{lid}.csv`` because iRfcb resolves a csv's sample
# name by stripping only ``.csv`` — a ``_class`` suffix would misname the sample.
_FORMAT_SUFFIX = {"h5": "_class.h5", "csv": "_class.csv", "mat": "_class_v1.mat", "csv-labels": ".csv"}


def _output_path_for_lid(output_dir: Path, lid: str, fmt: str = "h5") -> Path:
    """Return the output path for a bin LID in the given format."""
    return output_dir / f"{lid}{_FORMAT_SUFFIX[fmt]}"


def _bin_outputs_complete(output_dir: Path, lid: str, formats: tuple[str, ...]) -> bool:
    """True when every requested format's output file already exists for ``lid``."""
    return all(_output_path_for_lid(output_dir, lid, fmt).exists() for fmt in formats)


def _classify_bin_file(
    bin_path, model, transform, device, batch_size, class_names, thresholds, classifier_name, output_dir, overwrite, formats, counter=None
):
    """Classify a single bin file and write its class-scores output(s).

    Skips silently if every requested format already exists and ``overwrite`` is
    false. When ``counter`` is provided, the untransformed PIL images are also
    collected so the chain counter can run on them.
    """
    lid = get_bin_lid(bin_path)

    if not overwrite and _bin_outputs_complete(output_dir, lid, formats):
        logger.info("Skipping (already exists): %s", lid)
        return

    logger.info("Classifying bin: %s", lid)

    target_numbers = []
    images = []
    # When counting, the untransformed PILs for the whole bin are held until
    # after classification: which ROIs need a count isn't known until the
    # thresholded class is decided, so every ROI's image must be retained. This
    # roughly doubles peak memory for the bin during counting.
    raw_images = [] if counter is not None else None
    for target_num, img in iter_bin_images(bin_path):
        target_numbers.append(target_num)
        images.append(transform(img))
        if raw_images is not None:
            raw_images.append(img)

    if not images:
        logger.warning("No images in bin: %s", lid)
        return

    scores = _batch_predict(model, images, device, batch_size)
    _write_output(output_dir, lid, scores, class_names, target_numbers, classifier_name, thresholds, formats, overwrite, counter, raw_images)


def _classify_directory(
    dir_path, model, transform, device, batch_size, class_names, thresholds, classifier_name, output_dir, overwrite, formats, counter=None
):
    """Classify every bin in a directory, skipping ones already classified.

    Bins are discovered up front but read one at a time, so only a single bin's
    ROIs are in memory while scanning large directories.
    """
    for lid, fbin in iter_directory_bins(dir_path):
        if not overwrite and _bin_outputs_complete(output_dir, lid, formats):
            logger.info("Skipping (already exists): %s", lid)
            continue

        logger.info("Classifying bin: %s", lid)

        target_numbers = []
        images = []
        # See _classify_bin_file: raw PILs are retained for the whole bin when
        # counting, since the countable ROIs aren't known until classification.
        raw_images = [] if counter is not None else None
        for target_num, img in iter_bin_images(fbin):
            target_numbers.append(target_num)
            images.append(transform(img))
            if raw_images is not None:
                raw_images.append(img)

        if not images:
            logger.warning("No images in bin: %s", lid)
            continue

        scores = _batch_predict(model, images, device, batch_size)
        _write_output(output_dir, lid, scores, class_names, target_numbers, classifier_name, thresholds, formats, overwrite, counter, raw_images)


def _batch_predict(model, images, device, batch_size):
    """Run the model over pre-transformed images in batches.

    Returns an ``(N, num_classes)`` float array of softmax probabilities, moved
    back to the CPU and concatenated across batches.
    """
    all_scores = []
    with torch.no_grad():
        for i in range(0, len(images), batch_size):
            batch = torch.stack(images[i : i + batch_size]).to(device)
            logits = model(batch)
            probs = torch.nn.functional.softmax(logits, dim=1)
            all_scores.append(probs.cpu().numpy())
    return np.concatenate(all_scores, axis=0)


def _write_output(output_dir, lid, scores, class_names, target_numbers, classifier_name, thresholds, formats, overwrite, counter=None, raw_images=None):
    """Write the requested class-scores file(s) for a bin.

    ``formats`` is any subset of ``("h5", "csv", "mat", "csv-labels")``. A bin is
    reclassified whenever *any* requested format is missing, but each format is
    only written if its own file is absent (or ``overwrite`` is set) — so adding a
    new format to already-processed bins leaves the existing files (e.g. an ``h5``
    carrying chain counts) untouched. ``h5``, ``mat`` and ``csv-labels`` store
    chain counts and resolved/thresholded classes (``mat``/``csv-labels`` are the
    iRfcb/ClassiPyR layouts); ``csv`` stays scores-only to match the dashboard's
    export.
    """
    roi_numbers = np.array(target_numbers, dtype=np.int32)

    def _target(fmt):
        """Return the output path for ``fmt`` if it should be (re)written, else None."""
        path = _output_path_for_lid(output_dir, lid, fmt)
        return path if (overwrite or not path.exists()) else None

    h5_path = _target("h5") if "h5" in formats else None
    csv_path = _target("csv") if "csv" in formats else None
    mat_path = _target("mat") if "mat" in formats else None
    labels_path = _target("csv-labels") if "csv-labels" in formats else None

    # Chain counts are stored by the h5, mat and csv-labels outputs; compute them
    # once if any of those will actually be written.
    cell_counts, models_meta = None, None
    if h5_path is not None or mat_path is not None or labels_path is not None:
        cell_counts, models_meta = _compute_chain_counts(scores, class_names, thresholds, counter, raw_images)

    # The mat and csv-labels outputs both need the resolved/thresholded classes.
    class_name_auto, class_name = None, None
    if mat_path is not None or labels_path is not None:
        class_name_auto, class_name = resolve_class_names(scores, class_names, thresholds)

    written = []

    if h5_path is not None:
        write_class_scores(
            h5_path, scores, class_names, roi_numbers, classifier_name, thresholds,
            cell_counts=cell_counts, cell_counter_models=models_meta,
        )
        written.append(h5_path)

    if csv_path is not None:
        from ifcb_classify.csv_output import write_class_scores_csv

        write_class_scores_csv(csv_path, scores, class_names, roi_numbers, lid)
        written.append(csv_path)

    if mat_path is not None:
        from ifcb_classify.mat_output import write_class_scores_mat

        write_class_scores_mat(
            mat_path, scores, class_names, roi_numbers, class_name_auto, class_name,
            classifier_name, cell_counts=cell_counts,
        )
        written.append(mat_path)

    if labels_path is not None:
        from ifcb_classify.csv_labels_output import write_class_labels_csv

        write_class_labels_csv(
            labels_path, scores, roi_numbers, class_name_auto, class_name, lid,
            cell_counts=cell_counts,
        )
        written.append(labels_path)

    if not written:
        logger.info("Nothing to write for %s (all requested formats exist)", lid)
        return

    names = ", ".join(p.name for p in written)
    if cell_counts is not None:
        n_counted = int((cell_counts >= 0).sum())
        total_cells = int(cell_counts[cell_counts >= 0].sum())
        logger.info(
            "Wrote: %s (%d ROIs, %d chain-counted, %d cells)",
            names, len(target_numbers), n_counted, total_cells,
        )
    else:
        logger.info("Wrote: %s (%d ROIs)", names, len(target_numbers))


def _compute_chain_counts(scores, class_names, thresholds, counter, raw_images):
    """Count cells for ROIs whose thresholded class is configured for counting.

    Returns ``(cell_counts, models_metadata)`` or ``(None, None)`` when counting
    is disabled. Uncounted ROIs get the sentinel ``-1``.
    """
    if counter is None:
        return None, None

    _, class_name = resolve_class_names(scores, class_names, thresholds)
    counts = np.full(len(class_name), -1, dtype=np.int32)
    for j, cn in enumerate(class_name):
        if counter.handles(cn):
            counts[j] = counter.count(raw_images[j], cn)
    return counts, counter.models_metadata()


def _load_thresholds(config: InferConfig, class_names: list[str]) -> np.ndarray:
    """Resolve per-class decision thresholds, ordered to match ``class_names``.

    Resolution order: an explicit ``thresholds_path`` (JSON or YAML), else a
    ``thresholds.json`` auto-detected next to the checkpoint, else a flat array
    of ``config.threshold_default`` for every class. Classes absent from a file
    get ``NaN``, which downstream code treats as "no threshold" (accept argmax).
    """
    path = config.thresholds_path

    # Auto-detect thresholds.json from model directory
    if not path:
        model_dir = Path(config.model_checkpoint).parent
        candidate = model_dir / "thresholds.json"
        if candidate.exists():
            logger.info("Auto-detected thresholds: %s", candidate)
            path = str(candidate)

    if path:
        if path.endswith(".json"):
            from ifcb_classify.thresholds import load_thresholds_json
            return load_thresholds_json(path, class_names)
        with open(path) as f:
            data = yaml.safe_load(f)
        return np.array([data.get(c, np.nan) for c in class_names], dtype=np.float64)

    return np.full(len(class_names), config.threshold_default, dtype=np.float64)


def _derive_classifier_name(config: InferConfig, train_config: dict) -> str:
    """Derive the classifier name stored in the output when none was configured.

    Prefers the checkpoint's parent directory name; falls back to
    ``{model}_{dataset_version}`` for legacy checkpoints saved at the repo root.
    """
    # For legacy checkpoints, use the model directory name
    model_dir = Path(config.model_checkpoint).parent
    dir_name = model_dir.name
    if dir_name and dir_name != ".":
        return dir_name
    return f"{train_config['model']}_{train_config.get('dataset_version', '')}"
