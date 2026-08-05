"""Compute dataset mean/std for the ``_normalised`` transform variants.

Backs the ``normalise`` CLI command. The printed ``mean``/``std`` go into a
training config so that ``*_normalised`` transforms can standardise inputs. Stats
are computed in a single streaming pass (Welford's algorithm) using the
non-normalised version of the chosen transform.
"""

import torch
from torchvision import datasets

from ifcb_classify.data.datasets import build_transform, eval_transform_name


def compute_dataset_stats(
    data_dir: str,
    transform_name: str = "dataset_fullpad",
    width: int = 224,
    height: int = 224,
    batch_size: int = 1000,
) -> tuple[float, float]:
    """Compute per-channel mean and std for a training dataset.

    Uses Welford's online algorithm to compute mean and variance in a single
    pass, avoiding the need to iterate the dataset twice.

    Measures the images as they are scored: the ``_normalised`` suffix is stripped
    (these stats are what normalisation needs, so it cannot already be applied)
    and any augmentation is dropped. The augmented pipelines normalise *after*
    jittering, so augmented stats would whiten training inputs more exactly — but
    the same stored ``mean``/``std`` are reused at inference, where no jitter
    runs, so clean stats keep training and inference consistent and make the
    numbers a property of the dataset rather than of the RNG.
    """
    base_name = eval_transform_name(transform_name).replace("_normalised", "")
    transform = build_transform(base_name, width, height)
    dataset = datasets.ImageFolder(data_dir, transform=transform)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, num_workers=0)

    count = 0
    mean = 0.0
    m2 = 0.0

    for batch, _ in loader:
        batch_pixels = batch.numel()
        batch_mean = batch.mean().item()
        batch_var = batch.var().item()

        new_count = count + batch_pixels
        delta = batch_mean - mean
        mean += delta * batch_pixels / new_count
        m2 += batch_var * batch_pixels + delta * delta * count * batch_pixels / new_count
        count = new_count

    std = (m2 / count) ** 0.5 if count > 0 else 0.0
    return mean, std
