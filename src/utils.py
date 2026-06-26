"""
Utility functions for the MNIST classifier.

This module provides common "plumbing" that every ML project needs:
  1. Reproducibility: setting random seeds so you get the same results every run.
  2. Hardware detection: automatically using GPU if available, otherwise CPU.
"""

import random
import numpy as np
import torch


def set_seed(seed: int = 42) -> None:
    """
    Set all random number generators to a fixed seed.

    Why this matters:
    - Neural networks use randomness everywhere (weight initialization,
      data shuffling, dropout). Without a fixed seed, every run gives
      slightly different results.
    - Setting the seed means you (and anyone else) can exactly reproduce
      your training runs.

    What we set:
    - Python's built-in `random` (used by some data loaders).
    - NumPy's random (used by some preprocessing).
    - PyTorch's random (used by weight initialization, dropout, etc.).
    - CuDNN flags: deterministic ops run slightly slower but give
      identical results across runs on GPU.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # These two CuDNN settings force deterministic behavior on GPU.
    # deterministic=True: always pick the same convolution algorithm.
    # benchmark=False: don't auto-tune for speed (which depends on run-to-run variance).
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    """
    Automatically detect and return the best available compute device.

    Priority:
    1. CUDA GPU (NVIDIA)  -> fastest for deep learning
    2. CPU                -> works everywhere, slower but fine for MNIST

    Usage:
        device = get_device()
        model.to(device)
        tensor.to(device)

    Returns:
        torch.device: "cuda" if an NVIDIA GPU is available, else "cpu".
    """
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
