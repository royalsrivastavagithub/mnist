"""
Inference module.

After training, we want to use the model to make predictions on new images.

This module provides two functions:
  1. load_model: Load saved weights into a model architecture.
  2. predict: Make a prediction on a single image.

Note: load_model requires you to create the model architecture FIRST, then
load the saved state dictionary. This is because the saved weights are just
numbers — you need the model structure to know which number goes where.
"""

import torch
import torch.nn as nn
from src.utils import get_device


def load_model(model: nn.Module, checkpoint_path: str) -> nn.Module:
    """
    Load saved model weights from a checkpoint file.

    Usage:
        model = build_model("cnn")           # Create the architecture.
        model = load_model(model, "models/cnn_best.pth")  # Load the weights.

    How it works:
      1. torch.load reads the file and returns a state dictionary.
         weights_only=True is a security measure (prevents arbitrary code execution
         from malicious checkpoint files).
      2. model.load_state_dict(state) loads the weights into the model's layers.
         It matches each saved tensor to the corresponding layer by name.
      3. model.to(device) ensures weights are on the right device (GPU/CPU).
      4. model.eval() switches to evaluation mode (disables dropout etc.).

    Args:
        model: A model instance with the same architecture as the saved one.
        checkpoint_path: Path to the .pth file.

    Returns:
        The model with loaded weights, in evaluation mode.
    """
    device = get_device()

    # torch.load reads the checkpoint file.
    # map_location ensures weights are loaded to the correct device
    # (e.g., if saved on GPU but running on CPU).
    state = torch.load(checkpoint_path, map_location=device, weights_only=True)

    # Load the state dict into the model.
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


def predict(model: nn.Module, image: torch.Tensor) -> tuple[int, torch.Tensor]:
    """
    Predict the digit in a single image.

    Args:
        model: A trained model (in evaluation mode).
        image: A 3D tensor of shape (1, 28, 28) — the image to classify.

    Returns:
        Tuple of (predicted_digit, class_probabilities).
            - predicted_digit: Integer 0-9.
            - class_probabilities: Tensor of shape (10,) with values 0.0-1.0,
              summing to 1. This is the model's confidence for each digit.

    --- Understanding the output ---

    The model outputs raw scores called "logits" — arbitrary numbers that
    can be negative, positive, large, or small. These aren't intuitive.

    We apply softmax to convert logits to probabilities:
        softmax(x_i) = exp(x_i) / sum(exp(x_j))

    This gives us nice interpretable numbers:
        - Each between 0.0 and 1.0.
        - They sum to 1.0.
        - Larger logit = higher probability.

    Example output:
        predicted = 7
        probabilities = [0.01, 0.00, 0.02, 0.01, 0.03, 0.01, 0.01, 0.88, 0.02, 0.01]
        -> Model is 88% confident it's a "7".
    """
    device = get_device()
    model.to(device)

    # Add a batch dimension: (1, 28, 28) -> (1, 1, 28, 28).
    # The model expects a batch dimension even for single images.
    image = image.unsqueeze(0).to(device)

    with torch.no_grad():
        # Forward pass: get raw logits for the image.
        logits = model(image)  # shape: (1, 10)

        # Convert logits to probabilities using softmax.
        probs = torch.softmax(logits, dim=1)  # shape: (1, 10)

        # Find the digit with the highest probability.
        predicted = torch.argmax(probs, dim=1).item()  # scalar int

    # Remove batch dimension from probabilities and move to CPU.
    return predicted, probs.squeeze(0).cpu()
