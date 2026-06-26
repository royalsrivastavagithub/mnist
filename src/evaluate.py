"""
Evaluation module.

After the model is trained, we want to know: how good is it really?

This module computes:
  1. Overall test accuracy (what % of digits are correctly classified).
  2. Per-class accuracy (which digits does the model struggle with?).
  3. Confusion matrix (a grid showing what the model predicted vs. the truth).
  4. Collects sample images (correct and wrong) for visualization.

The confusion matrix is especially useful. For example, if the model often
confuses "4" with "9", we see a high number at position [4, 9] in the matrix.
This tells us what to improve.
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from src.utils import get_device


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader) -> dict:
    """
    Run the model on the entire dataset and compute comprehensive metrics.

    Args:
        model: Trained neural network.
        loader: DataLoader (usually the test set).

    Returns:
        Dictionary with:
            "accuracy": Overall accuracy (0-100%).
            "confusion_matrix": 10x10 numpy array.
            "per_class_accuracy": Dict mapping digit (0-9) -> accuracy (0-100%).
            "wrong_images": List of image tensors misclassified.
            "wrong_true": List of true labels for wrong predictions.
            "wrong_pred": List of predicted labels for wrong predictions.
            "correct_images": List of image tensors correctly classified.
            "correct_true": List of true labels for correct predictions.
            "correct_pred": List of predicted labels for correct predictions.
    """
    device = get_device()
    model.to(device)
    model.eval()

    all_preds = []
    all_labels = []
    correct = 0
    total = 0

    wrong_images, wrong_true, wrong_pred = [], [], []
    correct_images, correct_true, correct_pred = [], [], []

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)

        _, predicted = torch.max(outputs, 1)

        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

        # Collect samples for plotting (first 25 of each).
        for i in range(len(labels)):
            if predicted[i] != labels[i] and len(wrong_images) < 25:
                wrong_images.append(images[i].cpu())
                wrong_true.append(labels[i].item())
                wrong_pred.append(predicted[i].item())
            elif predicted[i] == labels[i] and len(correct_images) < 25:
                correct_images.append(images[i].cpu())
                correct_true.append(labels[i].item())
                correct_pred.append(predicted[i].item())

            if len(wrong_images) >= 25 and len(correct_images) >= 25:
                break

    accuracy = correct / total * 100

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    conf_matrix = np.zeros((10, 10), dtype=int)
    for t, p in zip(all_labels, all_preds):
        conf_matrix[t, p] += 1

    per_class_acc = {}
    for i in range(10):
        total_cls = (all_labels == i).sum()
        correct_cls = (all_labels[all_labels == i] == all_preds[all_labels == i]).sum()
        per_class_acc[i] = float(correct_cls / total_cls * 100) if total_cls > 0 else 0.0

    return {
        "accuracy": accuracy,
        "confusion_matrix": conf_matrix,
        "per_class_accuracy": per_class_acc,
        "wrong_images": wrong_images,
        "wrong_true": wrong_true,
        "wrong_pred": wrong_pred,
        "correct_images": correct_images,
        "correct_true": correct_true,
        "correct_pred": correct_pred,
    }
