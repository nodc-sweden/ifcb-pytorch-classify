import numpy as np
import torch
from PIL import Image

from ifcb_classify.data.datasets import (
    TRANSFORM_NAMES,
    build_transform,
    create_training_datasets,
    eval_transform_name,
)


def test_build_transform_squarepad():
    t = build_transform("dataset_squarepad", 224, 224)
    assert t is not None


def test_create_training_datasets(tiny_imagefolder):
    data_dir, classes = tiny_imagefolder
    data = create_training_datasets(str(data_dir), "dataset_squarepad", 224, 224, val_split=0.2)
    assert "train" in data
    assert "val" in data
    assert data["num_classes"] == 3
    assert data["class_names"] == classes
    assert len(data["train"]) + len(data["val"]) == 15


def test_eval_transform_name_strips_augmentation():
    assert eval_transform_name("dataset_squarepad_augmented") == "dataset_squarepad"
    assert eval_transform_name("dataset_fullpad_augmented") == "dataset_fullpad"
    assert eval_transform_name("dataset_squarepad_augmented_normalised") == "dataset_squarepad_normalised"
    assert eval_transform_name("dataset_fullpad_augmented_normalised") == "dataset_fullpad_normalised"


def test_eval_transform_name_passes_through_non_augmented():
    for name in ["dataset", "dataset_squarepad", "dataset_fullpad_normalised", "dataset_reflectpad"]:
        assert eval_transform_name(name) == name


def test_every_eval_transform_is_a_known_name():
    for name in TRANSFORM_NAMES:
        assert eval_transform_name(name) in TRANSFORM_NAMES


def test_every_eval_transform_is_deterministic():
    """The whole point of the eval variants: same input, same tensor, every time."""
    image = Image.fromarray(np.random.randint(0, 255, (24, 40, 3), dtype=np.uint8))
    for name in TRANSFORM_NAMES:
        transform = build_transform(eval_transform_name(name), 32, 32, mean=0.5, std=0.2)
        assert torch.equal(transform(image), transform(image)), f"{name} is not deterministic at eval"


def test_validation_split_is_not_augmented(tiny_imagefolder):
    """Thresholds and val metrics are computed on this split, so it must be stable."""
    data_dir, _ = tiny_imagefolder
    data = create_training_datasets(str(data_dir), "dataset_squarepad_augmented", 32, 32, val_split=0.4)

    first, _ = data["val"][0]
    second, _ = data["val"][0]
    assert torch.equal(first, second)

    # Stable is necessary but not sufficient — it must also be the right image,
    # put through the augmentation-free counterpart of the training transform.
    val_subset = data["val"]
    path, _ = val_subset.dataset.samples[val_subset.indices[0]]
    expected = build_transform("dataset_squarepad", 32, 32)(Image.open(path).convert("RGB"))
    assert torch.equal(first, expected)


def test_training_split_is_still_augmented(tiny_imagefolder):
    """De-augmenting validation must not disable augmentation for training."""
    data_dir, _ = tiny_imagefolder
    data = create_training_datasets(str(data_dir), "dataset_squarepad_augmented", 32, 32, val_split=0.4)

    first, _ = data["train"][0]
    second, _ = data["train"][0]
    assert not torch.equal(first, second)


def test_split_shares_one_file_ordering(tiny_imagefolder):
    """One index list addresses both splits, so they must share one ordering.

    Train and val differ only in their transform. If they ever enumerated the
    directory separately, a tree that changed between the two scans would shift
    what the validation indices point at and mislabel the split in silence.
    """
    data_dir, classes = tiny_imagefolder
    data = create_training_datasets(str(data_dir), "dataset_squarepad_augmented", 32, 32, val_split=0.4)

    train_ds, val_ds = data["train"].dataset, data["val"].dataset
    assert train_ds.samples is val_ds.samples
    assert train_ds.class_to_idx == val_ds.class_to_idx
    assert data["class_names"] == classes
    assert train_ds.transform is not val_ds.transform
