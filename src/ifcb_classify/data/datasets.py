"""Dataset construction and the named image-transform pipelines.

Two things live here:

* :func:`build_transform` — turns a transform *name* (one of
  :data:`TRANSFORM_NAMES`) into a torchvision ``Compose``. Names encode three
  independent choices: padding strategy (none / ``squarepad`` / ``fullpad`` /
  ``reflectpad``), whether augmentation is applied, and whether the result is
  normalised (which requires precomputed ``mean``/``std`` from the ``normalise``
  command). The same name must be used at train and inference time.
* :func:`create_training_datasets` — wraps an ``ImageFolder`` (class-per-folder
  layout) and splits it into train/validation ``Subset``s, optionally first
  filtering out rare classes via :func:`filter_classes`.
"""

import logging
import os
import shutil
from pathlib import Path

import torch
from sklearn.model_selection import train_test_split
from torchvision import datasets
from torchvision.transforms import v2 as transforms

from ifcb_classify.data.transforms import FullPad, ReflectPad, SquarePad

logger = logging.getLogger(__name__)

TRANSFORM_NAMES = [
    "dataset",
    "dataset_normalised",
    "dataset_squarepad",
    "dataset_squarepad_normalised",
    "dataset_fullpad",
    "dataset_fullpad_normalised",
    "dataset_reflectpad",
    "dataset_squarepad_augmented",
    "dataset_fullpad_augmented",
    "dataset_squarepad_augmented_normalised",
    "dataset_fullpad_augmented_normalised",
]

_AUGMENTATION = [
    transforms.ColorJitter(brightness=0.3, contrast=0.3),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
]


def _make_mean_std(mean: float, std: float) -> tuple[list[float], list[float]]:
    """Replicate single-channel stats to 3 channels for grayscale-to-RGB models."""
    return [mean, mean, mean], [std, std, std]


def build_transform(
    name: str,
    width: int = 224,
    height: int = 224,
    mean: float | None = None,
    std: float | None = None,
) -> transforms.Compose:
    """Build the torchvision transform pipeline named ``name``.

    All variants first convert to 3-channel grayscale float tensors (IFCB images
    are single-channel, but the pretrained backbones expect RGB). ``_normalised``
    variants require ``mean`` and ``std`` and append a ``Normalize`` step;
    ``_augmented`` variants insert colour-jitter and random flips. Raises
    ``ValueError`` for an unknown name (see :data:`TRANSFORM_NAMES`).
    """
    grayscale = transforms.Grayscale(num_output_channels=3)
    base = [
        grayscale,
        transforms.ToImage(),
        transforms.ToDtype(torch.float32, scale=True),
    ]

    if name == "dataset":
        return transforms.Compose([*base, transforms.Resize((width, height), antialias=True)])

    if name == "dataset_normalised":
        _require_stats(mean, std, name)
        m, s = _make_mean_std(mean, std)
        return transforms.Compose([
            *base,
            transforms.Resize((width, height), antialias=True),
            transforms.Normalize(mean=m, std=s),
        ])

    if name == "dataset_squarepad":
        return transforms.Compose([*base, SquarePad(), transforms.Resize((width, height), antialias=True)])

    if name == "dataset_squarepad_normalised":
        _require_stats(mean, std, name)
        m, s = _make_mean_std(mean, std)
        return transforms.Compose([
            *base,
            SquarePad(),
            transforms.Resize((width, height), antialias=True),
            transforms.Normalize(mean=m, std=s),
        ])

    if name == "dataset_fullpad":
        return transforms.Compose([
            *base,
            FullPad(width, height),
            transforms.Resize((width, height), antialias=True),
        ])

    if name == "dataset_fullpad_normalised":
        _require_stats(mean, std, name)
        m, s = _make_mean_std(mean, std)
        return transforms.Compose([
            *base,
            FullPad(width, height),
            transforms.Resize((width, height), antialias=True),
            transforms.Normalize(mean=m, std=s),
        ])

    if name == "dataset_reflectpad":
        return transforms.Compose([*base, ReflectPad(width, height), transforms.Resize((width, height), antialias=True)])

    # Augmented variants
    if name == "dataset_squarepad_augmented":
        return transforms.Compose([
            *base, SquarePad(), transforms.Resize((width, height), antialias=True), *_AUGMENTATION,
        ])

    if name == "dataset_fullpad_augmented":
        return transforms.Compose([
            *base, FullPad(width, height), transforms.Resize((width, height), antialias=True), *_AUGMENTATION,
        ])

    if name == "dataset_squarepad_augmented_normalised":
        _require_stats(mean, std, name)
        m, s = _make_mean_std(mean, std)
        return transforms.Compose([
            *base, SquarePad(), transforms.Resize((width, height), antialias=True),
            *_AUGMENTATION, transforms.Normalize(mean=m, std=s),
        ])

    if name == "dataset_fullpad_augmented_normalised":
        _require_stats(mean, std, name)
        m, s = _make_mean_std(mean, std)
        return transforms.Compose([
            *base, FullPad(width, height), transforms.Resize((width, height), antialias=True),
            *_AUGMENTATION, transforms.Normalize(mean=m, std=s),
        ])

    raise ValueError(f"Unknown transform: {name}. Available: {TRANSFORM_NAMES}")


def filter_classes(
    data_dir: str,
    min_images: int = 50,
    manual_include: list[str] | None = None,
) -> tuple[str, list[str]]:
    """Filter class folders by minimum image count.

    Returns (filtered_dir, filtered_class_names). Creates a temporary
    _filtered_dataset directory with symlinks to qualifying classes.
    """
    manual_include = manual_include or []
    data_path = Path(data_dir)
    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}

    class_folders = sorted([
        f.name for f in data_path.iterdir()
        if f.is_dir() and not f.name.startswith("_")
    ])

    filtered = []
    for cls in class_folders:
        cls_path = data_path / cls
        num_images = sum(1 for f in cls_path.iterdir() if f.suffix.lower() in image_extensions)
        if num_images >= min_images or cls in manual_include:
            filtered.append(cls)

    logger.info("Class filtering: %d/%d classes pass (min_images=%d)", len(filtered), len(class_folders), min_images)

    filtered_root = data_path / "_filtered_dataset"
    if filtered_root.exists():
        shutil.rmtree(filtered_root)
    filtered_root.mkdir()

    for cls in filtered:
        src = data_path / cls
        dst = filtered_root / cls
        try:
            os.symlink(src.resolve(), dst, target_is_directory=True)
        except OSError:
            # Windows without symlink privileges — hard-link individual files
            dst.mkdir()
            for img in src.iterdir():
                if img.suffix.lower() in image_extensions:
                    os.link(img, dst / img.name)

    return str(filtered_root), filtered


def create_training_datasets(
    data_dir: str,
    transform_name: str,
    width: int = 224,
    height: int = 224,
    val_split: float = 0.2,
    mean: float | None = None,
    std: float | None = None,
    seed: int = 42,
    min_class_images: int | None = None,
    manual_include_classes: list[str] | None = None,
) -> dict:
    """Build train/validation datasets from a class-per-folder image directory.

    When ``min_class_images`` is set, rare classes are first filtered out (see
    :func:`filter_classes`). The split is deterministic given ``seed``. Returns a
    dict with ``train`` / ``val`` (``Subset``s), ``class_names`` and
    ``num_classes``.
    """
    effective_dir = data_dir
    if min_class_images is not None:
        effective_dir, _ = filter_classes(data_dir, min_class_images, manual_include_classes)

    transform = build_transform(transform_name, width, height, mean, std)
    dataset = datasets.ImageFolder(effective_dir, transform=transform)
    train_idx, val_idx = train_test_split(list(range(len(dataset))), test_size=val_split, random_state=seed)
    return {
        "train": torch.utils.data.Subset(dataset, train_idx),
        "val": torch.utils.data.Subset(dataset, val_idx),
        "class_names": dataset.classes,
        "num_classes": len(dataset.classes),
    }


def _require_stats(mean, std, name):
    """Raise a helpful error if a normalised transform was requested without stats."""
    if mean is None or std is None:
        raise ValueError(f"Transform '{name}' requires mean and std. Run `ifcb-classify normalise` first.")
