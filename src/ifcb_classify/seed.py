"""Reproducibility: seed every RNG the pipeline touches."""

import os
import random

import numpy as np
import torch


def set_seed(seed: int = 42) -> None:
    """Seed Python, NumPy and torch RNGs and force deterministic cuDNN.

    Disabling cuDNN benchmarking trades some GPU throughput for run-to-run
    reproducibility. Call once at the start of training or inference.
    """
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)
