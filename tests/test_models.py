"""
Tests for the model module.

These tests verify that:
  1. Each model produces the correct output shape.
  2. The build_model factory returns the right type.
  3. An unknown model name raises an error.
  4. The models can do a forward + backward pass (gradients flow correctly).

Why test models?
  - A simple refactor (changing layer sizes) could break shape assumptions.
  - Tests catch these issues instantly instead of at 2 AM during training.
  - Demonstrates software engineering discipline in an ML project.
"""

import torch
import pytest
from src.model import MLP, CNN, ViT, build_model


def test_mlp_output_shape():
    """MLP should output (batch_size, 10) for any batch size."""
    model = MLP()
    x = torch.randn(4, 1, 28, 28)  # Batch of 4 MNIST images.
    out = model(x)
    # 10 classes = digits 0 through 9.
    assert out.shape == (4, 10), f"Expected (4, 10), got {out.shape}"


def test_cnn_output_shape():
    """CNN should output (batch_size, 10) for any batch size."""
    model = CNN()
    x = torch.randn(4, 1, 28, 28)
    out = model(x)
    assert out.shape == (4, 10), f"Expected (4, 10), got {out.shape}"


def test_vit_output_shape():
    """ViT should output (batch_size, 10) for any batch size."""
    model = ViT()
    x = torch.randn(4, 1, 28, 28)
    out = model(x)
    assert out.shape == (4, 10), f"Expected (4, 10), got {out.shape}"


def test_build_model_mlp():
    """build_model('mlp') should return an MLP instance."""
    model = build_model("mlp")
    assert isinstance(model, MLP)


def test_build_model_cnn():
    """build_model('cnn') should return a CNN instance."""
    model = build_model("cnn")
    assert isinstance(model, CNN)


def test_build_model_vit():
    """build_model('vit') should return a ViT instance."""
    model = build_model("vit")
    assert isinstance(model, ViT)


def test_build_model_unknown():
    """build_model with an unknown name should raise ValueError."""
    with pytest.raises(ValueError):
        build_model("unknown")


@pytest.mark.parametrize("model_cls", [MLP, CNN, ViT])
def test_models_forward_and_backward(model_cls):
    """
    Every model should support:
      1. Forward pass: input -> output without errors.
      2. Loss computation: comparing output to random labels.
      3. Backward pass: gradients should flow to all parameters.
         (All parameters should have non-None gradients after .backward().)

    This is a "sanity check" — if gradients don't flow, the model can't learn.
    """
    model = model_cls()
    x = torch.randn(2, 1, 28, 28)  # Small batch of 2.
    y = torch.randint(0, 10, (2,))  # Random labels.
    out = model(x)

    # CrossEntropyLoss expects (batch, num_classes) logits and (batch,) labels.
    loss = torch.nn.functional.cross_entropy(out, y)

    # Backward pass: compute gradients.
    loss.backward()

    # Check that every parameter received a gradient.
    # If any parameter has grad=None, something is wrong (e.g., a disconnected layer).
    for param in model.parameters():
        assert param.grad is not None, f"Parameter {param.shape} has no gradient!"
