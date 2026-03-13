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

    Uses a non-normalised transform variant (strips _normalised suffix if present).
    """
    base_name = transform_name.replace("_normalised", "")
    transform = build_transform(base_name, width, height)
    dataset = datasets.ImageFolder(data_dir, transform=transform)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, num_workers=0)

    num_pixels = len(dataset) * width * height * 3

    total_sum = 0.0
    for batch, _ in loader:
        total_sum += batch.sum().item()
    mean = total_sum / num_pixels

    sum_sq_error = 0.0
    for batch, _ in loader:
        sum_sq_error += ((batch - mean) ** 2).sum().item()
    std = (sum_sq_error / num_pixels) ** 0.5

    return mean, std
