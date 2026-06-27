"""
Main entry point for the MNIST classifier.

This script ties everything together:
  1. Parses command-line arguments to configure the run.
  2. Sets the random seed for reproducibility.
  3. Builds the requested model architecture.
  4. Loads the MNIST dataset.
  5. Trains the model (optional).
  6. Evaluates the model on the test set (optional).
  7. Generates plots, metrics JSON, and a PDF report.

Usage:
    # Train CNN (default) for 10 epochs
    python main.py

    # Train MLP for 5 epochs
    python main.py --model mlp --epochs 5

    # Train ViT for 15 epochs and then evaluate
    python main.py --model vit --epochs 15 --mode train-eval

    # Just evaluate an existing model
    python main.py --model cnn --mode eval

    # Custom batch size and learning rate
    python main.py --model cnn --batch-size 128 --lr 0.0005
"""

import argparse
import json
import platform
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

from src.dataset import get_mnist_loaders
from src.model import build_model
from src.train import train
from src.evaluate import evaluate
from src.utils import set_seed, get_device
from src.plots import (
    plot_loss,
    plot_accuracy,
    plot_confusion_matrix,
    plot_wrong_predictions,
    plot_sample_predictions,
)
from src.report import generate_report

PROJECT_ROOT = Path(__file__).parent


def main() -> None:
    parser = argparse.ArgumentParser(
        description="MNIST Classifier - Train and evaluate multiple model architectures"
    )

    parser.add_argument(
        "--model",
        type=str,
        default="cnn",
        choices=["mlp", "cnn", "vit"],
        help="Model architecture: mlp (simple), cnn (standard), vit (transformer)",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="train",
        choices=["train", "eval", "train-eval"],
        help="'train' to only train, 'eval' to only evaluate, 'train-eval' to do both",
    )
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=64, help="Samples per batch")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")

    args = parser.parse_args()

    set_seed(args.seed)

    model = build_model(args.model)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {args.model.upper()} ({num_params:,} params)")

    train_loader, test_loader = get_mnist_loaders(
        batch_size=args.batch_size,
        data_root=str(PROJECT_ROOT / "data"),
    )
    print(f"Training samples: {len(train_loader.dataset):,}")
    print(f"Test samples: {len(test_loader.dataset):,}")

    # Create run directory.
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = PROJECT_ROOT / "outputs" / f"run_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"Run directory: {run_dir}")

    # Save config.
    config = {
        "model": args.model,
        "mode": args.mode,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "seed": args.seed,
        "num_params": num_params,
    }
    with open(run_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    # Save model summary.
    with open(run_dir / "model_summary.txt", "w") as f:
        f.write(str(model))
        f.write(f"\n\nTotal parameters: {num_params:,}\n")

    if args.mode in ("train", "train-eval"):
        print(f"\n{'='*50}")
        print(f"Starting training with {args.model.upper()}...")
        print(f"{'='*50}")

        checkpoint_path = str(run_dir / "best_model.pth")

        history = train(
            model=model,
            train_loader=train_loader,
            test_loader=test_loader,
            epochs=args.epochs,
            lr=args.lr,
            checkpoint_path=checkpoint_path,
        )

        print(f"\nTraining summary for {args.model.upper()}:")
        print(f"  Best validation accuracy: {max(history['val_acc']):.2f}%")
        print(f"  Final training loss: {history['train_loss'][-1]:.4f}")
        print(f"  Final validation loss: {history['val_loss'][-1]:.4f}")

        # Generate training plots.
        plot_loss(history, run_dir / "loss.png")
        plot_accuracy(history, run_dir / "accuracy.png")
        print(f"Plots saved: loss.png, accuracy.png")

        # Save training stats (metrics table + timing + hardware).
        device = get_device()
        gpu_name = torch.cuda.get_device_name(0) if device.type == "cuda" else "N/A"
        epochs_list = []
        for i in range(len(history["train_loss"])):
            epochs_list.append({
                "epoch": i + 1,
                "train_loss": round(history["train_loss"][i], 4),
                "val_loss": round(history["val_loss"][i], 4),
                "val_acc": round(history["val_acc"][i], 2),
                "time_seconds": round(history["epoch_times"][i], 1),
            })

        stats = {
            "hardware": {
                "device": str(device),
                "gpu": gpu_name,
                "cpu": platform.processor() or "unknown",
            },
            "timing": {
                "total_seconds": round(history["total_time"], 1),
                "avg_per_epoch": round(history["total_time"] / len(history["epoch_times"]), 1),
            },
            "epochs": epochs_list,
        }
        with open(run_dir / "training_stats.json", "w") as f:
            json.dump(stats, f, indent=2)

    if args.mode in ("eval", "train-eval"):
        print(f"\n{'='*50}")
        print(f"Evaluating {args.model.upper()} on test set...")
        print(f"{'='*50}")

        results = evaluate(model, test_loader)

        # Save metrics as JSON.
        metrics = {
            "accuracy": results["accuracy"],
            "per_class_accuracy": {str(k): v for k, v in results["per_class_accuracy"].items()},
            "confusion_matrix": results["confusion_matrix"].tolist(),
        }
        with open(run_dir / "metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)

        # Generate evaluation plots.
        plot_confusion_matrix(results["confusion_matrix"], run_dir / "confusion_matrix.png")
        plot_wrong_predictions(
            results["wrong_images"], results["wrong_true"], results["wrong_pred"],
            run_dir / "wrong_predictions.png",
        )
        plot_sample_predictions(
            results["correct_images"], results["correct_true"], results["correct_pred"],
            run_dir / "sample_predictions.png",
        )
        print(f"Plots saved: confusion_matrix.png, wrong_predictions.png, sample_predictions.png")

        print(f"\nOverall Test Accuracy: {results['accuracy']:.2f}%")
        print(f"\n--- Per-Class Accuracy ---")
        for digit, acc in results["per_class_accuracy"].items():
            bar_length = int(acc / 2)
            bar = "#" * bar_length
            print(f"  Digit {digit}: {acc:5.2f}%  {bar}")

        print(f"\n--- Confusion Matrix ---")
        print("       Predicted digit")
        print("       ", end="")
        for i in range(10):
            print(f"{i:4d} ", end="")
        print("\n       " + "-" * 50)

        for i in range(10):
            print(f"True {i}: ", end="")
            for j in range(10):
                print(f"{results['confusion_matrix'][i, j]:4d} ", end="")
            print()

        print(f"\n--- Most Common Mistakes ---")
        confusion = results["confusion_matrix"]
        np.fill_diagonal(confusion, 0)
        mistakes = []
        for i in range(10):
            for j in range(10):
                if confusion[i, j] > 0:
                    mistakes.append((confusion[i, j], i, j))
        mistakes.sort(reverse=True)
        for count, true, pred in mistakes[:5]:
            print(f"  Digit {true} misclassified as {pred}: {count} times")

        print(f"\nWrong predictions collected: {len(results['wrong_images'])}")
        print(f"Correct predictions collected: {len(results['correct_images'])}")

    # Generate PDF report.
    generate_report(run_dir)
    print(f"\nAll outputs saved to: {run_dir}")


if __name__ == "__main__":
    main()
