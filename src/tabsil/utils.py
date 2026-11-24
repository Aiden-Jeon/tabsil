"""Device and seed helpers."""

import random

import numpy as np
import torch


def set_seed(seed: int = 42) -> None:
    """Seed the Python, NumPy and PyTorch random number generators."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def get_device(prefer: str | None = None) -> torch.device:
    """The fastest available device (cuda > mps > cpu).

    With `prefer`, that device is used as is, e.g. `get_device("cpu")`.
    """
    if prefer is not None:
        return torch.device(prefer)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
