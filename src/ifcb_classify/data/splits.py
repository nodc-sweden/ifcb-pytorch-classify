"""Which images train the model and which validate it.

:func:`split_indices` turns the dataset's labels into two index lists. Three
splits are available:

* ``random`` — the split every model trained before the strategy existed used:
  one shuffle of all images, ignoring their class. Kept bit-for-bit, because a
  checkpoint that records no strategy has its validation set rebuilt from it.
* ``stratified`` — each class is shuffled and cut on its own, so every class
  contributes its share to validation however rare it is.
* ``stratified`` with ``groups`` — whole groups (IFCB bins, or days) are kept on
  one side. Images from one bin are near-duplicates of each other; splitting
  them across the boundary lets the model be validated on what it has, in
  effect, already seen.

The split draws only from its own seeded generator, never from the global RNGs,
so it does not move when unrelated code consumes random numbers before it.
"""

import logging
import math
import os
import re
import warnings
from collections.abc import Sequence

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold, train_test_split

from ifcb_classify.config import LEGACY_SPLIT_STRATEGY, VALID_SPLIT_GROUPS, VALID_SPLIT_STRATEGIES

logger = logging.getLogger(__name__)

__all__ = ["LEGACY_SPLIT_STRATEGY", "group_keys", "split_indices"]

# ``D20230128T124402_IFCB172`` at the start of a ROI filename: the bin, and
# within it the calendar date.
_BIN_PATTERN = re.compile(r"^(D(\d{8})T\d{6}_IFCB\d+)")
_GROUP_PART = {"bin": 1, "day": 2}

# How far the realised validation fraction may drift from ``val_split`` before
# it is worth a warning rather than a line in the log.
_FRACTION_TOLERANCE = 0.02
_MAX_NAMED = 10


def group_keys(paths: Sequence[str], by: str) -> list[str]:
    """Return the group each image belongs to, read from its filename.

    ``by`` is ``"bin"`` (one IFCB sample) or ``"day"`` (the calendar date, across
    instruments). A filename without a bin id cannot be grouped, so it becomes a
    group of its own — split like any single image — and the count is logged.
    """
    if by not in VALID_SPLIT_GROUPS:
        raise ValueError(f"split_group_by must be one of: {', '.join(VALID_SPLIT_GROUPS)}; got {by!r}")

    part = _GROUP_PART[by]
    matches = [_BIN_PATTERN.match(os.path.basename(path)) for path in paths]
    unmatched = [str(path) for path, match in zip(paths, matches, strict=True) if match is None]
    if unmatched:
        logger.warning(
            "%d of %d filenames carry no bin id (D<date>T<time>_IFCB<n>), so each is split on its own "
            "rather than with its %s, e.g. %s",
            len(unmatched), len(paths), by, ", ".join(unmatched[:_MAX_NAMED]),
        )
    return [
        match.group(part) if match else f"file:{path}"
        for path, match in zip(paths, matches, strict=True)
    ]


def split_indices(
    targets: Sequence[int],
    val_split: float,
    seed: int,
    strategy: str = LEGACY_SPLIT_STRATEGY,
    groups: Sequence[str] | None = None,
    class_names: Sequence[str] | None = None,
) -> tuple[list[int], list[int]]:
    """Split ``range(len(targets))`` into ``(train, validation)`` index lists.

    ``targets`` holds one class label per image. ``groups``, when given, holds
    one group key per image (see :func:`group_keys`) and requires the
    ``stratified`` strategy. ``class_names`` only makes the log messages
    readable. The result is deterministic given ``seed``.
    """
    if strategy not in VALID_SPLIT_STRATEGIES:
        raise ValueError(f"split_strategy must be one of: {', '.join(VALID_SPLIT_STRATEGIES)}; got {strategy!r}")
    if groups is not None and strategy != "stratified":
        raise ValueError(f"A grouped split requires split_strategy 'stratified', got {strategy!r}")
    if groups is not None and len(groups) != len(targets):
        raise ValueError(f"groups and targets differ in length ({len(groups)} and {len(targets)})")

    labels = np.asarray(targets)
    if strategy == "random":
        train, val = train_test_split(list(range(len(labels))), test_size=val_split, random_state=seed)
    elif groups is None:
        train, val = _stratified_split(labels, val_split, seed)
    else:
        train, val = _grouped_split(labels, groups, val_split, seed, class_names)

    _report_split(labels, val, val_split, class_names)
    return train, val


def _stratified_split(labels: np.ndarray, val_split: float, seed: int) -> tuple[list[int], list[int]]:
    """Shuffle and cut each class on its own."""
    rng = np.random.default_rng(seed)
    train: list[int] = []
    val: list[int] = []
    for label in np.unique(labels):
        members = rng.permutation(np.flatnonzero(labels == label))
        n_val = _validation_count(len(members), val_split)
        val.extend(members[:n_val].tolist())
        train.extend(members[n_val:].tolist())
    return sorted(train), sorted(val)


def _validation_count(n_images: int, val_split: float) -> int:
    """How many of a class's images validate: its share, but never all or (given two) none.

    A single image can only train. Rounds half up; ``round()`` rounds half to
    even, which would send 2.5 and 3.5 to different sides.
    """
    if n_images < 2:
        return 0
    return min(max(math.floor(n_images * val_split + 0.5), 1), n_images - 1)


def _grouped_split(
    labels: np.ndarray,
    groups: Sequence[str],
    val_split: float,
    seed: int,
    class_names: Sequence[str] | None,
) -> tuple[list[int], list[int]]:
    """Split whole groups, keeping each class's share as close as the groups allow.

    Whole groups only come in folds of ``1/k``, so the validation fraction is the
    nearest such fraction to ``val_split``. Single-image classes are left out of
    the folds and train only, as in the ungrouped split.
    """
    codes = _separable_group_codes(labels, groups, class_names)
    pool = np.flatnonzero(np.bincount(labels)[labels] > 1)
    n_splits = max(2, math.floor(1 / val_split + 0.5))
    n_groups = len(np.unique(codes[pool]))
    if n_groups < n_splits:
        raise ValueError(
            f"A grouped split with val_split={val_split} needs at least {n_splits} groups, but the "
            f"dataset has {n_groups}. Use a coarser val_split or drop split_group_by."
        )

    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    with warnings.catch_warnings():
        # sklearn warns about classes smaller than n_splits; _report_split says
        # what that means here (which classes ended up without validation images).
        warnings.simplefilter("ignore", UserWarning)
        folds = [pool[fold] for _, fold in splitter.split(pool, labels[pool], codes[pool])]

    val = _best_fold(labels, pool, folds)
    train = np.setdiff1d(np.arange(len(labels)), val)
    untrained = np.setdiff1d(labels, labels[train])
    if len(untrained):
        raise ValueError(
            "The grouped split left no training images for: "
            f"{_describe(untrained, class_names)}. Try another seed, or exclude them with min_class_images."
        )
    return train.tolist(), val.tolist()


def _separable_group_codes(labels: np.ndarray, groups: Sequence[str], class_names: Sequence[str] | None) -> np.ndarray:
    """Integer group ids, with classes confined to one group split by image instead.

    Keeping such a class's only group whole would put the entire class on one
    side: no images to learn it from, or none to validate it on. Its images get
    a group each, which gives up the isolation for that class alone.
    """
    _, codes = np.unique(np.asarray(groups), return_inverse=True)
    pairs = np.unique(np.stack([labels, codes], axis=1), axis=0)
    groups_per_class = np.bincount(pairs[:, 0], minlength=labels.max() + 1)
    confined = (groups_per_class == 1) & (np.bincount(labels) > 1)
    if not confined.any():
        return codes

    logger.warning(
        "%d classes have all their images in a single group and are split by image instead: %s",
        confined.sum(), _describe(np.flatnonzero(confined), class_names),
    )
    own = confined[labels]
    separated = codes.copy()
    separated[own] = codes.max() + 1 + np.arange(own.sum())
    return separated


def _best_fold(labels: np.ndarray, pool: np.ndarray, folds: list[np.ndarray]) -> np.ndarray:
    """The fold that, as validation, leaves the fewest classes without training or validation images."""
    n_classes = len(np.unique(labels[pool]))

    def classes_missing(fold: np.ndarray) -> tuple[int, int]:
        in_train = len(np.unique(labels[np.setdiff1d(pool, fold)]))
        return n_classes - in_train, n_classes - len(np.unique(labels[fold]))

    return min(folds, key=classes_missing)


def _report_split(labels: np.ndarray, val: Sequence[int], val_split: float, class_names: Sequence[str] | None) -> None:
    """Log the realised split, and warn where it falls short of what was asked for."""
    fraction = len(val) / len(labels) if len(labels) else 0.0
    logger.info(
        "Split: %d training / %d validation images (%.1f%%)", len(labels) - len(val), len(val), 100 * fraction
    )
    if abs(fraction - val_split) > _FRACTION_TOLERANCE:
        logger.warning(
            "Validation holds %.1f%% of the images, not the val_split of %.1f%% that was asked for: "
            "small classes and whole groups cannot be divided exactly.",
            100 * fraction, 100 * val_split,
        )
    unvalidated = np.setdiff1d(labels, labels[np.asarray(val, dtype=int)])
    if len(unvalidated):
        logger.warning(
            "%d classes have no validation images, so no validation score or threshold can be "
            "fitted for them: %s",
            len(unvalidated), _describe(unvalidated, class_names),
        )


def _describe(labels: Sequence[int], class_names: Sequence[str] | None) -> str:
    """Name the classes behind ``labels``, shortened when there are many."""
    names = [str(class_names[label]) if class_names is not None else str(label) for label in labels]
    if len(names) <= _MAX_NAMED:
        return ", ".join(names)
    return f"{', '.join(names[:_MAX_NAMED])} and {len(names) - _MAX_NAMED} more"
