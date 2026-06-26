"""
Data loading module for MNIST.

Responsibility:
  - Download the MNIST dataset (if not already present).
  - Apply preprocessing (convert images to tensors, normalize pixel values).
  - Package into PyTorch DataLoaders for efficient batching during training/testing.

What is MNIST?
  - 70,000 grayscale images of handwritten digits (0-9).
  - Each image is 28 pixels tall × 28 pixels wide = 784 pixels total.
  - Split: 60,000 training images + 10,000 test images.
  - Standard benchmark for "hello world" image classification.
"""

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def get_mnist_loaders(
    batch_size: int = 64,
    data_root: str = "data",
    num_workers: int = 2,
) -> tuple[DataLoader, DataLoader]:
    """
    Download MNIST (if needed) and create training/test DataLoaders.

    Args:
        batch_size: How many images to process at once.
        data_root: Directory where MNIST will be stored.
        num_workers: How many CPU subprocesses to use for loading data.

    Returns:
        Tuple of (train_loader, test_loader).

    --- How DataLoaders work ---
    Instead of loading one image at a time, a DataLoader:
      1. Groups images into batches (e.g., 64 images at once).
      2. Shuffles them randomly for training (so the model sees a
         different order every epoch).
      3. Uses multiple CPU workers so that while the GPU trains on
         batch N, the CPU is already loading batch N+1.

    --- The transform pipeline ---
    Every image goes through two steps before the model sees it:

    Step 1: ToTensor()
      - Converts a PIL image (0-255 integer values) to a PyTorch tensor.
      - Shape changes: (28, 28) -> (1, 28, 28).
        - First dimension (1) = the "channel". Grayscale = 1 channel.
        - Color images would have 3 channels (RGB).
      - Values change: integers 0-255 -> floats 0.0-1.0.
        - This helps the neural network learn more stably.

    Step 2: Normalize(mean, std)
      - Transforms pixel values from range [0.0, 1.0] to have a specific
        mean and standard deviation.
      - Formula: output = (input - mean) / std
      - (0.1307,) is the mean pixel value across all MNIST training images.
      - (0.3081,) is the standard deviation of pixels across all images.
      - After normalization, most pixel values fall roughly in [-1, 1].
      - Why normalize? Neural networks train faster and more reliably when
        input values are centered around 0 with a reasonable spread.
    """
    # Compose chains multiple transforms into a single pipeline.
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])

    # torchvision.datasets.MNIST handles downloading automatically.
    # The 'train=True' gives us 60,000 training samples.
    train_dataset = datasets.MNIST(
        root=data_root, train=True, download=True, transform=transform
    )

    # The 'train=False' gives us 10,000 test samples.
    test_dataset = datasets.MNIST(
        root=data_root, train=False, download=True, transform=transform
    )

    # DataLoader wraps a Dataset and provides:
    #   - Batching: groups samples into batches of `batch_size`.
    #   - Shuffling: randomizes order each epoch (crucial for training stability).
    #   - Parallel loading: `num_workers` processes load data simultaneously.
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,          # Shuffle training data so order doesn't bias the model.
        num_workers=num_workers,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,         # Don't shuffle test data (order doesn't matter for evaluation).
        num_workers=num_workers,
    )

    return train_loader, test_loader
