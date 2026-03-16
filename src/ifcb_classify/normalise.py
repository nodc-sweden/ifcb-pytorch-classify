import torch
from torchvision import datasets

from ifcb_classify.data.datasets import build_transform


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

    Uses a non-normalised transform variant (strips _normalised suffix if present).
    """
    base_name = transform_name.replace("_normalised", "")
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
