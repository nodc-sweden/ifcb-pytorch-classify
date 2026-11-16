import shutil
from pathlib import Path

from ifcb_classify.data.datasets import create_training_datasets, filter_classes

FIXTURES = Path(__file__).parent / "fixtures" / "training_data"


def test_create_training_datasets_from_fixtures():
    data = create_training_datasets(
        data_dir=str(FIXTURES),
        transform_name="dataset_squarepad",
        width=32,
        height=32,
        val_split=0.2,
    )
    assert data["num_classes"] == 2
    assert set(data["class_names"]) == {"Mesodinium_major", "Strombidium-like"}
    assert len(data["train"]) + len(data["val"]) == 30


def test_training_dataset_outputs_correct_shape():
    data = create_training_datasets(
        data_dir=str(FIXTURES),
        transform_name="dataset_squarepad",
        width=32,
        height=32,
    )
    img, label = data["train"][0]
    assert img.shape == (3, 32, 32)
    assert isinstance(label, int)


def test_filter_classes_min_images(tmp_path):
    """Copy fixtures to tmp so symlinks work on any filesystem."""
    data_dir = tmp_path / "training_data"
    shutil.copytree(FIXTURES, data_dir)

    # Mesodinium_major has 14 images, Strombidium-like has 16
    _filtered_dir, filtered = filter_classes(str(data_dir), min_images=15)
    assert "Strombidium-like" in filtered
    assert "Mesodinium_major" not in filtered


def test_filter_classes_manual_include(tmp_path):
    data_dir = tmp_path / "training_data"
    shutil.copytree(FIXTURES, data_dir)

    _filtered_dir, filtered = filter_classes(
        str(data_dir), min_images=100, manual_include=["Mesodinium_major"],
    )
    assert "Mesodinium_major" in filtered


def test_all_non_normalised_transforms_work():
    non_normalised = [
        "dataset", "dataset_squarepad", "dataset_fullpad",
        "dataset_squarepad_augmented", "dataset_fullpad_augmented",
    ]
    for name in non_normalised:
        data = create_training_datasets(
            data_dir=str(FIXTURES),
            transform_name=name,
            width=32,
            height=32,
        )
        img, _ = data["train"][0]
        assert img.shape == (3, 32, 32), f"Transform {name} produced wrong shape"


# --- split strategies -------------------------------------------------------


def _val_paths(data):
    samples = data["val"].dataset.samples
    return [Path(samples[i][0]) for i in data["val"].indices]


def _train_paths(data):
    samples = data["train"].dataset.samples
    return [Path(samples[i][0]) for i in data["train"].indices]


def test_default_split_is_the_legacy_random_split():
    """Callers that pass no strategy (old checkpoints, old scripts) keep the old split."""
    from sklearn.model_selection import train_test_split

    data = create_training_datasets(str(FIXTURES), "dataset_squarepad", 32, 32, val_split=0.2)
    expected_train, expected_val = train_test_split(list(range(30)), test_size=0.2, random_state=42)

    assert list(data["train"].indices) == expected_train
    assert list(data["val"].indices) == expected_val


def test_stratified_split_gives_each_class_its_share():
    # Mesodinium_major has 14 images, Strombidium-like has 16
    data = create_training_datasets(
        str(FIXTURES), "dataset_squarepad", 32, 32, val_split=0.2, split_strategy="stratified",
    )
    per_class = {}
    for path in _val_paths(data):
        per_class[path.parent.name] = per_class.get(path.parent.name, 0) + 1

    assert per_class == {"Mesodinium_major": 3, "Strombidium-like": 3}
    assert len(data["train"]) == 24


def test_bin_grouped_split_keeps_each_bin_on_one_side():
    data = create_training_datasets(
        str(FIXTURES), "dataset_squarepad", 32, 32, val_split=0.2,
        split_strategy="stratified", split_group_by="bin",
    )
    train_bins = {p.name[:24] for p in _train_paths(data)}
    val_bins = {p.name[:24] for p in _val_paths(data)}

    assert val_bins
    assert train_bins.isdisjoint(val_bins)
    assert len(data["train"]) + len(data["val"]) == 30


def test_day_grouped_split_falls_back_for_a_single_day_class(caplog):
    """Every Mesodinium_major image is from one day; Strombidium-like spans two."""
    import logging

    with caplog.at_level(logging.WARNING):
        data = create_training_datasets(
            str(FIXTURES), "dataset_squarepad", 32, 32, val_split=0.2,
            split_strategy="stratified", split_group_by="day",
        )
    train, val = _train_paths(data), _val_paths(data)

    def days(paths, cls):
        return {p.name[1:9] for p in paths if p.parent.name == cls}

    assert days(train, "Strombidium-like").isdisjoint(days(val, "Strombidium-like"))
    assert days(train, "Mesodinium_major")
    assert days(val, "Mesodinium_major")
    assert "Mesodinium_major" in caplog.text
