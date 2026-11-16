"""Tests for the train/validation index split (``ifcb_classify.data.splits``)."""

import logging
import random
from collections import Counter

import numpy as np
import pytest
from sklearn.model_selection import train_test_split

from ifcb_classify.data.splits import LEGACY_SPLIT_STRATEGY, group_keys, split_indices


def _targets(*sizes):
    """Labels for classes of the given sizes, in class order (as ImageFolder yields them)."""
    return [label for label, size in enumerate(sizes) for _ in range(size)]


def _per_class(targets, indices):
    return Counter(targets[i] for i in indices)


# --- legacy random split ----------------------------------------------------


@pytest.mark.parametrize("n", [15, 137])
def test_random_split_is_the_legacy_sklearn_call(n):
    """Checkpoints trained before the strategy existed rebuild their split from this.

    Anything but the identical call, on the identical input, moves images across
    the train/validation boundary for every existing model.
    """
    targets = [i % 3 for i in range(n)]
    expected_train, expected_val = train_test_split(list(range(n)), test_size=0.2, random_state=42)

    train, val = split_indices(targets, val_split=0.2, seed=42, strategy="random")

    assert train == expected_train
    assert val == expected_val


def test_legacy_strategy_is_random():
    assert LEGACY_SPLIT_STRATEGY == "random"


def test_unknown_strategy_raises():
    with pytest.raises(ValueError, match="split_strategy"):
        split_indices(_targets(5, 5), val_split=0.2, seed=42, strategy="clever")


# --- stratified split -------------------------------------------------------


def test_stratified_gives_each_class_its_share():
    targets = _targets(1000, 10, 5)
    train, val = split_indices(targets, val_split=0.2, seed=42, strategy="stratified")

    assert _per_class(targets, val) == {0: 200, 1: 2, 2: 1}
    assert _per_class(targets, train) == {0: 800, 1: 8, 2: 4}


def test_stratified_is_disjoint_and_complete():
    targets = _targets(37, 11, 3)
    train, val = split_indices(targets, val_split=0.3, seed=1, strategy="stratified")

    assert set(train).isdisjoint(val)
    assert sorted(train + val) == list(range(len(targets)))


def test_stratified_is_deterministic_and_follows_the_seed():
    targets = _targets(50, 50)
    first = split_indices(targets, val_split=0.2, seed=7, strategy="stratified")
    again = split_indices(targets, val_split=0.2, seed=7, strategy="stratified")
    other = split_indices(targets, val_split=0.2, seed=8, strategy="stratified")

    assert first == again
    assert first != other


def test_stratified_ignores_the_global_rngs():
    """Training seeds the global RNGs first; the split must not depend on their state."""
    targets = _targets(50, 50)
    random.seed(0)
    np.random.seed(0)
    first = split_indices(targets, val_split=0.2, seed=7, strategy="stratified")
    random.seed(123)
    np.random.seed(123)
    again = split_indices(targets, val_split=0.2, seed=7, strategy="stratified")

    assert first == again


def test_single_image_class_trains_only_and_is_reported(caplog):
    targets = _targets(10, 1)
    with caplog.at_level(logging.WARNING):
        train, val = split_indices(
            targets, val_split=0.2, seed=42, strategy="stratified", class_names=["Common", "Lonely"]
        )

    assert 10 in train
    assert _per_class(targets, val) == {0: 2}
    assert "Lonely" in caplog.text


def test_two_image_class_gets_one_of_each():
    targets = _targets(10, 2)
    train, val = split_indices(targets, val_split=0.2, seed=42, strategy="stratified")

    assert _per_class(targets, val)[1] == 1
    assert _per_class(targets, train)[1] == 1


def test_small_classes_never_lose_all_their_training_images():
    """A large val_split on a tiny class must still leave something to train on."""
    targets = _targets(3, 3)
    train, _ = split_indices(targets, val_split=0.9, seed=42, strategy="stratified")

    assert _per_class(targets, train) == {0: 1, 1: 1}


def test_more_classes_than_validation_slots_does_not_raise():
    """sklearn's ``stratify=`` refuses this; a class-per-folder dataset must not."""
    targets = _targets(5, 5, 5)
    _, val = split_indices(targets, val_split=0.1, seed=42, strategy="stratified")

    assert _per_class(targets, val) == {0: 1, 1: 1, 2: 1}


# --- group keys -------------------------------------------------------------


def test_group_key_reads_the_bin_from_a_roi_filename():
    paths = ["/data/Tripos/D20230128T124402_IFCB172_t24_x828_y134_32015732.png", "D20230314T001205_IFCB134_00012.png"]

    assert group_keys(paths, "bin") == ["D20230128T124402_IFCB172", "D20230314T001205_IFCB134"]


def test_group_key_by_day_is_the_calendar_date():
    paths = ["a/D20230128T124402_IFCB172_00001.png", "b/D20230128T235959_IFCB134_00002.png"]

    assert group_keys(paths, "day") == ["20230128", "20230128"]


def test_unmatched_filenames_become_their_own_group_and_are_reported(caplog):
    paths = ["A/img_0.png", "A/img_1.png", "B/img_0.png", "B/D20230128T124402_IFCB172_00001.png"]
    with caplog.at_level(logging.WARNING):
        keys = group_keys(paths, "bin")

    assert len(set(keys)) == 4
    assert keys[3] == "D20230128T124402_IFCB172"
    assert "3 of 4" in caplog.text
    assert "img_0.png" in caplog.text


def test_unknown_group_by_raises():
    with pytest.raises(ValueError, match="split_group_by"):
        group_keys(["D20230128T124402_IFCB172_00001.png"], "week")


# --- grouped split ----------------------------------------------------------


def _grouped_dataset(n_classes=4, n_groups=40, per_group=6, seed=0):
    """Every group holds images of two classes, so groups cut across classes."""
    rng = np.random.default_rng(seed)
    targets, groups = [], []
    for g in range(n_groups):
        for label in rng.choice(n_classes, size=2, replace=False):
            targets.extend([int(label)] * per_group)
            groups.extend([f"bin{g:03d}"] * per_group)
    order = np.argsort(targets, kind="stable")
    return [targets[i] for i in order], [groups[i] for i in order]


def test_grouped_split_keeps_every_group_on_one_side():
    targets, groups = _grouped_dataset()
    train, val = split_indices(targets, val_split=0.2, seed=42, strategy="stratified", groups=groups)

    assert {groups[i] for i in train}.isdisjoint({groups[i] for i in val})
    assert sorted(train + val) == list(range(len(targets)))
    assert set(_per_class(targets, val)) == set(targets)
    assert abs(len(val) / len(targets) - 0.2) < 0.05


def test_grouped_split_is_deterministic_and_follows_the_seed():
    targets, groups = _grouped_dataset()
    first = split_indices(targets, val_split=0.2, seed=3, strategy="stratified", groups=groups)
    again = split_indices(targets, val_split=0.2, seed=3, strategy="stratified", groups=groups)
    other = split_indices(targets, val_split=0.2, seed=4, strategy="stratified", groups=groups)

    assert first == again
    assert first != other


def test_class_confined_to_one_group_is_split_by_image_and_reported(caplog):
    """Keeping its only group whole would leave the class with no training or no validation images."""
    targets, groups = _grouped_dataset(n_classes=3)
    lonely = max(targets) + 1
    targets = [*targets, *[lonely] * 10]
    groups = [*groups, *["bin999"] * 10]
    names = ["A", "B", "C", "OneBin"]

    with caplog.at_level(logging.WARNING):
        train, val = split_indices(
            targets, val_split=0.2, seed=42, strategy="stratified", groups=groups, class_names=names
        )

    assert _per_class(targets, train)[lonely] >= 1
    assert _per_class(targets, val)[lonely] >= 1
    assert "OneBin" in caplog.text


def test_single_image_class_trains_only_when_grouped():
    targets, groups = _grouped_dataset(n_classes=3)
    lonely = max(targets) + 1
    targets = [*targets, lonely]
    groups = [*groups, groups[0]]

    for seed in range(5):
        train, _ = split_indices(targets, val_split=0.2, seed=seed, strategy="stratified", groups=groups)
        assert len(targets) - 1 in train


def test_grouped_split_needs_enough_groups():
    targets = _targets(20, 20)
    groups = ["binA"] * 10 + ["binB"] * 10 + ["binA"] * 10 + ["binB"] * 10

    with pytest.raises(ValueError, match="groups"):
        split_indices(targets, val_split=0.2, seed=42, strategy="stratified", groups=groups)


def test_grouping_requires_the_stratified_strategy():
    targets, groups = _grouped_dataset()

    with pytest.raises(ValueError, match="stratified"):
        split_indices(targets, val_split=0.2, seed=42, strategy="random", groups=groups)


def test_groups_must_match_targets_in_length():
    with pytest.raises(ValueError, match="length"):
        split_indices(_targets(5, 5), val_split=0.2, seed=42, strategy="stratified", groups=["a", "b"])


def test_unachievable_fraction_is_reported(caplog):
    """Whole groups come in folds of 1/k, so 0.3 cannot be met exactly."""
    targets, groups = _grouped_dataset()
    with caplog.at_level(logging.WARNING):
        _, val = split_indices(targets, val_split=0.3, seed=42, strategy="stratified", groups=groups)

    assert abs(len(val) / len(targets) - 1 / 3) < 0.05
    assert "val_split" in caplog.text


def test_class_left_without_training_images_is_an_error(monkeypatch):
    """Training on with a class the model never sees would fail silently."""
    targets = _targets(10, 10)
    groups = [f"bin{i}" for i in range(20)]

    class _EveryFoldTakesAWholeClass:
        def __init__(self, **_):
            pass

        def split(self, pool, labels, codes):
            for label in np.unique(labels):
                yield np.flatnonzero(labels != label), np.flatnonzero(labels == label)

    monkeypatch.setattr("ifcb_classify.data.splits.StratifiedGroupKFold", _EveryFoldTakesAWholeClass)

    with pytest.raises(ValueError, match="no training images for: Alpha"):
        split_indices(
            targets, val_split=0.5, seed=42, strategy="stratified", groups=groups, class_names=["Alpha", "Beta"]
        )


def test_long_class_lists_are_shortened_in_messages(caplog):
    targets = _targets(10, *[1] * 15)
    with caplog.at_level(logging.WARNING):
        split_indices(targets, val_split=0.2, seed=42, strategy="stratified")

    assert "15 classes have no validation images" in caplog.text
    assert "and 5 more" in caplog.text
