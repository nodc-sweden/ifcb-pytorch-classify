"""Selecting the compute device (CPU / CUDA / MPS)."""

import torch


def get_device(force: str = "auto") -> torch.device:
    """Resolve a torch device from a selector string.

    ``"auto"`` prefers CUDA, then Apple MPS, then CPU. ``"cpu"``/``"cuda"`` force
    that device; any other string is passed straight to ``torch.device``.
    Training auto-selects (GPU when present); inference defaults to CPU.
    """
    if force == "cpu":
        return torch.device("cpu")
    if force == "cuda":
        return torch.device("cuda")
    if force == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(force)
