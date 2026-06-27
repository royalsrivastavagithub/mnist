"""
Plotting module.

Generates all PNG visualizations for a training run:
  1. loss.png         - training and validation loss curves
  2. accuracy.png     - validation accuracy curve
  3. confusion_matrix.png - heatmap of predictions vs true labels
  4. wrong_predictions.png - grid of images the model misclassified
  5. sample_predictions.png - grid of correctly classified images
"""

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path


def plot_loss(history: dict, save_path: str | Path) -> None:
    plt.figure(figsize=(8, 5))
    epochs = range(1, len(history["train_loss"]) + 1)
    plt.plot(epochs, history["train_loss"], "bo-", label="Training Loss")
    plt.plot(epochs, history["val_loss"], "ro-", label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss over Epochs")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_accuracy(history: dict, save_path: str | Path) -> None:
    plt.figure(figsize=(8, 5))
    epochs = range(1, len(history["val_acc"]) + 1)
    plt.plot(epochs, history["val_acc"], "go-", label="Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.title("Validation Accuracy over Epochs")
    plt.ylim(0, 100)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_confusion_matrix(conf_matrix: np.ndarray, save_path: str | Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(conf_matrix, cmap="Blues", interpolation="nearest")
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    ax.set_title("Confusion Matrix")
    ax.set_xticks(range(10))
    ax.set_yticks(range(10))

    for i in range(10):
        for j in range(10):
            ax.text(j, i, str(conf_matrix[i, j]),
                    ha="center", va="center",
                    color="white" if conf_matrix[i, j] > conf_matrix.max() / 2 else "black")

    fig.colorbar(im, ax=ax, shrink=0.8)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_wrong_predictions(
    images: list[torch.Tensor],
    true_labels: list[int],
    pred_labels: list[int],
    save_path: str | Path,
    max_samples: int = 25,
) -> None:
    n = min(len(images), max_samples)
    cols = 5
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2, rows * 2))
    axes = axes.flatten() if hasattr(axes, "flatten") else [axes]

    for i in range(n):
        img = images[i].squeeze().cpu().numpy()
        axes[i].imshow(img, cmap="gray")
        axes[i].set_title(f"True: {true_labels[i]}\nPred: {pred_labels[i]}", fontsize=9, color="red")
        axes[i].axis("off")

    for i in range(n, len(axes)):
        axes[i].axis("off")

    plt.suptitle(f"Misclassified Images ({n} shown)", fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_sample_predictions(
    images: list[torch.Tensor],
    true_labels: list[int],
    pred_labels: list[int],
    save_path: str | Path,
    max_samples: int = 25,
) -> None:
    n = min(len(images), max_samples)
    cols = 5
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2, rows * 2))
    axes = axes.flatten() if hasattr(axes, "flatten") else [axes]

    for i in range(n):
        img = images[i].squeeze().cpu().numpy()
        axes[i].imshow(img, cmap="gray")
        axes[i].set_title(f"True: {true_labels[i]}\nPred: {pred_labels[i]}", fontsize=9, color="green")
        axes[i].axis("off")

    for i in range(n, len(axes)):
        axes[i].axis("off")

    plt.suptitle(f"Correctly Classified Images ({n} shown)", fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
