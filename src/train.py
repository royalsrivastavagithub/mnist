"""
Training module.

This module handles:
  1. The training loop: forward pass -> loss -> backward pass -> weight update.
  2. Validation loop: evaluating the model on test data at the end of each epoch.
  3. Checkpointing: saving the model whenever it achieves a new best accuracy.

Key concepts explained:

  --- Loss (CrossEntropyLoss) ---
  Measures how wrong the model's predictions are.
  For a single image: if the true label is "3", the model outputs 10 scores.
  Cross-entropy loss is high when the model gives a low score to "3" and
  high scores to other digits. The goal is to minimize this loss.

  --- Backpropagation (loss.backward()) ---
  The magic of deep learning. PyTorch automatically computes the gradient
  (derivative) of the loss with respect to every parameter in the model.
  This tells us: "if we increase this weight slightly, does the loss go up or down?"

  --- Optimizer (Adam) ---
  Uses gradients to update the weights.
  Adam is adaptive: it adjusts the learning rate for each parameter individually.
  Think of it as "smart gradient descent" that's faster and more stable than plain SGD.

  --- Learning Rate Scheduler ---
  Reduces the learning rate during training.
  Start with big steps (lr=0.001), then reduce by half every 5 epochs.
  This helps the model converge more precisely later in training.
"""

import time
import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from tqdm import tqdm
from src.utils import get_device


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    """
    Train the model for one complete pass through the training data.

    This is the core training step. For each batch of images:

    1. Move data to GPU (or CPU).
    2. Zero out gradients from the previous step.
       (PyTorch accumulates gradients by default, so we must reset them.)
    3. Forward pass: give the model images, get predictions.
    4. Compute loss: how wrong were the predictions?
    5. Backward pass: compute gradients for every parameter.
    6. Optimizer step: update all parameters using the gradients.

    Args:
        model: The neural network to train.
        loader: DataLoader providing batches of (images, labels).
        optimizer: Algorithm that updates model weights (e.g., Adam).
        criterion: Loss function (e.g., CrossEntropyLoss).
        device: "cuda" or "cpu".

    Returns:
        Average loss over all batches in this epoch.
    """
    model.train()
    total_loss = 0.0

    pbar = tqdm(loader, desc="  Training", unit="batch")

    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        pbar.set_postfix(Loss=f"{loss.item():.4f}")

    return total_loss / len(loader)


@torch.no_grad()
def validate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    """
    Evaluate the model on a dataset (usually test set) without training.

    The @torch.no_grad() decorator tells PyTorch we don't need gradients.
    This is important because:
      - Without gradients, the computation uses MUCH less memory.
      - Without gradients, the computation is faster.
      - We never want to accidentally update weights during validation.

    Args:
        model: The neural network to evaluate.
        loader: DataLoader (usually test set).
        criterion: Loss function.
        device: "cuda" or "cpu".

    Returns:
        Tuple of (average_loss, accuracy_percent).
    """
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)

        loss = criterion(outputs, labels)
        total_loss += loss.item()

        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    avg_loss = total_loss / len(loader)
    accuracy = correct / total * 100
    return avg_loss, accuracy


def train(
    model: nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
    epochs: int = 10,
    lr: float = 1e-3,
    checkpoint_path: str = "models/best_model.pth",
) -> dict[str, list[float]]:
    """
    Full training pipeline: loop over epochs, train + validate, save best model.

    Args:
        model: The model to train.
        train_loader: Training data.
        test_loader: Test data (used for validation after each epoch).
        epochs: Number of complete passes through the training data.
        lr: Initial learning rate for Adam optimizer.
        checkpoint_path: File path to save the best model weights.

    Returns:
        Dictionary containing training history:
            history["train_loss"]:   loss after each epoch.
            history["val_loss"]:     validation loss after each epoch.
            history["val_acc"]:      validation accuracy (%) after each epoch.
            history["epoch_times"]:  duration of each epoch in seconds.
            history["total_time"]:   total training time in seconds.
    """
    device = get_device()
    model.to(device)
    print(f"Device: {device}")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    best_acc = 0.0
    history = {"train_loss": [], "val_loss": [], "val_acc": [], "epoch_times": []}

    t_start = time.time()
    for epoch in range(1, epochs + 1):
        t0 = time.time()
        print(f"\nEpoch [{epoch}/{epochs}]")
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc = validate(model, test_loader, criterion, device)
        scheduler.step()
        dt = time.time() - t0

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["epoch_times"].append(dt)

        print(f"  Train Loss: {train_loss:.4f}  Val Loss: {val_loss:.4f}  Val Acc: {val_acc:.2f}%  Time: {dt:.1f}s")

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), checkpoint_path)
            print(f"  -> New best model saved (acc: {val_acc:.2f}%)")

        current_lr = optimizer.param_groups[0]["lr"]
        print(f"  LR: {current_lr:.6f}")

    history["total_time"] = time.time() - t_start
    print(f"\nTraining complete. Best accuracy: {best_acc:.2f}%")
    print(f"Total time: {history['total_time']:.1f}s")
    return history
