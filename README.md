# MNIST Classifier

Classify handwritten digits (0-9) using multiple neural network architectures — **MLP**, **CNN**, and **Vision Transformer (ViT)**.

Built with **PyTorch**. Every run produces a timestamped output folder with plots, metrics, and a PDF report — ready for your portfolio.

## Quick Start

```bash
# Install dependencies
uv sync

# Train the CNN (default) for 10 epochs
uv run python main.py

# Fast test run (1 epoch, smallest model)
uv run python main.py --model mlp --epochs 1 --batch-size 256
```

## Models

| Model | Parameters | Speed | Accuracy | Description |
|-------|-----------|-------|----------|-------------|
| `mlp` | ~269K | Fastest | ~97% | Multi-Layer Perceptron — simplest, flattens the image |
| `cnn` | ~1.2M | Medium | ~99% | Convolutional Neural Network — standard for images |
| `vit` | ~289K | Slowest | ~98% | Vision Transformer — attention-based, modern |

## Usage

```bash
# Train
uv run python main.py --model cnn --epochs 10 --mode train

# Train + evaluate (recommended)
uv run python main.py --model mlp --epochs 10 --mode train-eval

# Evaluate only (needs existing checkpoint)
uv run python main.py --model cnn --mode eval

# Full training options
uv run python main.py --model vit --epochs 15 --batch-size 128 --lr 0.0005
```

### CLI Arguments

| Argument | Default | Choices | Description |
|----------|---------|---------|-------------|
| `--model` | `cnn` | `mlp`, `cnn`, `vit` | Model architecture |
| `--mode` | `train` | `train`, `eval`, `train-eval` | What to do |
| `--epochs` | `10` | any int | Training epochs |
| `--batch-size` | `64` | any int | Images per batch |
| `--lr` | `0.001` | any float | Learning rate |
| `--seed` | `42` | any int | Random seed |

## Output Structure

Each run creates a timestamped folder:

```
outputs/run_2026-06-27_18-30/
├── report.pdf              # PDF report with all metrics and plots
├── loss.png                # Training + validation loss curves
├── accuracy.png            # Validation accuracy curve
├── confusion_matrix.png    # Heatmap of predictions vs truth
├── wrong_predictions.png   # Grid of misclassified images
├── sample_predictions.png  # Grid of correct predictions
├── metrics.json            # Accuracy, per-class accuracy, confusion matrix
├── training_stats.json     # Hardware, timing, epoch-by-epoch table
├── config.json             # CLI arguments + model params
├── model_summary.txt       # Model architecture printout
└── best_model.pth          # Best checkpoint weights
```

## Project Structure

```
mnist-classifier/
├── .gitignore
├── README.md
├── pyproject.toml          # Dependencies: torch, torchvision, numpy, tqdm, matplotlib, fpdf2
├── main.py                 # CLI entry point
├── src/
│   ├── __init__.py
│   ├── utils.py            # set_seed(), get_device()
│   ├── dataset.py          # MNIST download + DataLoaders
│   ├── model.py            # MLP, CNN, ViT model definitions
│   ├── train.py            # Training loop, validation, checkpointing
│   ├── evaluate.py         # Test accuracy, confusion matrix, metrics
│   ├── predict.py          # Load checkpoint + predict on single image
│   ├── plots.py            # Matplotlib plot functions (5 plot types)
│   └── report.py           # PDF report generation (fpdf2)
├── tests/
│   ├── conftest.py
│   └── test_models.py      # Model shape + gradient flow tests (10 tests)
└── data/MNIST/             # Auto-downloaded dataset
```

## Run Tests

```bash
uv run pytest tests/ -v
```

## Dependencies

- **torch** — Core ML framework
- **torchvision** — MNIST dataset
- **numpy** — Array operations
- **tqdm** — Progress bars
- **matplotlib** — Plots (PNG output)
- **fpdf2** — PDF report generation
