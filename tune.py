"""
Hyperparameter tuning script.

Automatically searches over model types, learning rates, batch sizes,
and epochs to find the best configuration for MNIST classification.

Usage:
    # Grid search over all combinations (36 runs)
    uv run python tune.py

    # Random search with 15 trials
    uv run python tune.py --search random --trials 15

    # Grid search, MLP only, 3 epochs
    uv run python tune.py --search grid --model mlp --epochs 3

    # Narrow grid: specific values
    uv run python tune.py --model cnn --lr 0.01,0.001 --bs 64,128 --epochs 5,10
"""

import argparse
import itertools
import json
import random
import time
from datetime import datetime
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

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

# Full search space used for grid search (unless overridden by CLI).
GRID_SPACE = {
    "model": ["mlp", "cnn"],
    "lr": [0.01, 0.001, 0.0005],
    "batch_size": [32, 64, 128],
    "epochs": [5, 10],
}
# Total: 2 x 3 x 3 x 2 = 36 runs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Hyperparameter tuning for MNIST")
    parser.add_argument(
        "--search", type=str, default="grid", choices=["grid", "random"],
        help="Search strategy",
    )
    parser.add_argument(
        "--trials", type=int, default=20,
        help="Number of random search trials",
    )
    parser.add_argument(
        "--model", type=str, default="all",
        help="Filter: mlp, cnn, vit, or 'all'",
    )
    parser.add_argument(
        "--lr", type=str, default="grid",
        help="Override: 'grid' or comma-separated like '0.01,0.001'",
    )
    parser.add_argument(
        "--bs", type=str, default="grid",
        help="Override: 'grid' or comma-separated like '32,64,128'",
    )
    parser.add_argument(
        "--epochs", type=str, default="grid",
        help="Override: 'grid' or comma-separated like '3,5,10'",
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def build_search_configs(args: argparse.Namespace) -> list[dict]:
    """Build the list of config dictionaries to try."""
    space = GRID_SPACE.copy()

    if args.model != "all":
        space["model"] = args.model.split(",")
    if args.lr != "grid":
        space["lr"] = [float(x) for x in args.lr.split(",")]
    if args.bs != "grid":
        space["batch_size"] = [int(x) for x in args.bs.split(",")]
    if args.epochs != "grid":
        space["epochs"] = [int(x) for x in args.epochs.split(",")]

    if args.search == "grid":
        keys = list(space.keys())
        combos = list(itertools.product(*space.values()))
        configs = [dict(zip(keys, combo)) for combo in combos]
    else:
        keys = list(space.keys())
        configs = []
        for _ in range(args.trials):
            cfg = {}
            for k in keys:
                cfg[k] = random.choice(space[k])
            configs.append(cfg)

    return configs


def run_name(config: dict) -> str:
    """Generate a descriptive run directory name from config."""
    return (
        f"{config['model']}"
        f"_lr{config['lr']}"
        f"_bs{config['batch_size']}"
        f"_e{config['epochs']}"
    )


def _make_loaders(batch_size: int, data_root: str) -> tuple[DataLoader, DataLoader]:
    """Create train/test DataLoaders with a specific batch size."""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    train_ds = datasets.MNIST(root=data_root, train=True, download=True, transform=transform)
    test_ds = datasets.MNIST(root=data_root, train=False, download=True, transform=transform)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=2)
    return train_loader, test_loader


def run_one(
    config: dict,
    run_dir: Path,
    data_root: str,
    base_seed: int,
    run_index: int,
) -> dict:
    """Train and evaluate one configuration. Returns result dict."""
    seed = base_seed + run_index
    set_seed(seed)

    print(f"\n{'=' * 60}")
    print(f"Run {run_index + 1}: {run_name(config)}")
    print(f"{'=' * 60}")

    run_dir.mkdir(parents=True, exist_ok=True)

    model = build_model(config["model"])
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {config['model'].upper()} ({num_params:,} params)")

    # Save config.
    with open(run_dir / "config.json", "w") as f:
        json.dump({**config, "num_params": num_params, "seed": seed}, f, indent=2)

    # Save model summary.
    with open(run_dir / "model_summary.txt", "w") as f:
        f.write(str(model))
        f.write(f"\n\nTotal parameters: {num_params:,}\n")

    # Data loaders with this run's batch size.
    train_loader, test_loader = _make_loaders(config["batch_size"], data_root)

    # Train.
    checkpoint_path = str(run_dir / "best_model.pth")
    t_start = time.time()
    history = train(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        epochs=config["epochs"],
        lr=config["lr"],
        checkpoint_path=checkpoint_path,
    )
    total_time = time.time() - t_start

    # Evaluate.
    results = evaluate(model, test_loader)

    # Save metrics.
    metrics = {
        "accuracy": results["accuracy"],
        "per_class_accuracy": {str(k): v for k, v in results["per_class_accuracy"].items()},
        "confusion_matrix": results["confusion_matrix"].tolist(),
    }
    with open(run_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    # Generate plots.
    plot_loss(history, run_dir / "loss.png")
    plot_accuracy(history, run_dir / "accuracy.png")
    plot_confusion_matrix(results["confusion_matrix"], run_dir / "confusion_matrix.png")
    plot_wrong_predictions(
        results["wrong_images"], results["wrong_true"], results["wrong_pred"],
        run_dir / "wrong_predictions.png",
    )
    plot_sample_predictions(
        results["correct_images"], results["correct_true"], results["correct_pred"],
        run_dir / "sample_predictions.png",
    )
    generate_report(run_dir)

    # Build result summary.
    best_val_acc = max(history["val_acc"]) if history["val_acc"] else 0
    result = {
        "model": config["model"],
        "lr": config["lr"],
        "batch_size": config["batch_size"],
        "epochs": config["epochs"],
        "seed": seed,
        "num_params": num_params,
        "best_val_acc": round(best_val_acc, 2),
        "test_accuracy": round(results["accuracy"], 2),
        "final_train_loss": round(history["train_loss"][-1], 4),
        "final_val_loss": round(history["val_loss"][-1], 4),
        "total_time_seconds": round(total_time, 1),
    }

    print(f"  -> Accuracy: {result['test_accuracy']:.2f}%  Time: {total_time:.1f}s")
    return result


def save_comparison(results: list[dict], tune_dir: Path) -> None:
    # Sort by test accuracy (descending).
    results.sort(key=lambda r: r["test_accuracy"], reverse=True)

    # Save JSON.
    with open(tune_dir / "comparison.json", "w") as f:
        json.dump(results, f, indent=2)

    # Generate PDF.
    _comparison_pdf(results, tune_dir / "comparison.pdf")

    # Generate chart.
    _comparison_chart(results, tune_dir / "comparison_summary.png")

    # Print winner.
    best = results[0]
    print(f"\n{'=' * 60}")
    print(f"BEST: {best['model'].upper()}")
    print(f"  lr={best['lr']}  bs={best['batch_size']}  epochs={best['epochs']}")
    print(f"  Test Accuracy: {best['test_accuracy']:.2f}%")
    print(f"  Time: {best['total_time_seconds']:.1f}s")
    print(f"{'=' * 60}")


def _comparison_pdf(results: list[dict], save_path: Path) -> None:
    from fpdf import FPDF

    pdf = FPDF()
    pdf.add_page("L")

    pdf.set_font("Helvetica", "B", 18)
    pdf.cell(0, 14, "Hyperparameter Tuning Results", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 10)
    pdf.cell(0, 8,
             f"Total runs: {len(results)}  |  Best: {results[0]['model'].upper()} "
             f"{results[0]['test_accuracy']:.2f}%",
             align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(6)

    # Table.
    col_w = [14, 20, 20, 20, 16, 24, 24, 24, 18]
    headers = ["Rank", "Model", "LR", "Batch", "Epochs", "Test Acc %", "Val Acc %", "Train Loss", "Time (s)"]

    pdf.set_font("Helvetica", "B", 8)
    pdf.set_fill_color(50, 50, 150)
    pdf.set_text_color(255, 255, 255)
    for i, h in enumerate(headers):
        pdf.cell(col_w[i], 7, h, border=1, fill=True, align="C")
    pdf.ln()
    pdf.set_text_color(0, 0, 0)

    for rank, r in enumerate(results, 1):
        fill = (240, 240, 240) if rank % 2 == 0 else (255, 255, 255)
        pdf.set_fill_color(*fill)
        pdf.set_font("Courier", "", 7)
        row = [
            str(rank), r["model"].upper(), str(r["lr"]),
            str(r["batch_size"]), str(r["epochs"]),
            f"{r['test_accuracy']:.2f}%", f"{r['best_val_acc']:.2f}%",
            f"{r['final_train_loss']:.4f}", f"{r['total_time_seconds']:.1f}",
        ]
        for j, val in enumerate(row):
            pdf.cell(col_w[j], 6, val, border=1, fill=True, align="C")
        pdf.ln()

    # Per-model best.
    pdf.add_page("P")
    pdf.set_font("Helvetica", "B", 16)
    pdf.cell(0, 12, "Best Configuration Per Model Type", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(6)
    pdf.set_font("Courier", "", 10)
    seen = set()
    for r in results:
        if r["model"] not in seen:
            seen.add(r["model"])
            pdf.cell(0, 6,
                     f"  {r['model'].upper():5s}   Acc={r['test_accuracy']:.2f}%  "
                     f"lr={r['lr']}  bs={r['batch_size']}  e={r['epochs']}  "
                     f"time={r['total_time_seconds']:.1f}s",
                     new_x="LMARGIN", new_y="NEXT")

    pdf.output(str(save_path))


def _comparison_chart(results: list[dict], save_path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    top = results[:10]
    labels = [
        f"{r['model'].upper()} lr={r['lr']} bs={r['batch_size']} e={r['epochs']}"
        for r in top
    ]
    accs = [r["test_accuracy"] for r in top]
    colors = {
        "mlp": "#2196F3", "cnn": "#FF9800", "vit": "#9C27B0",
    }
    bar_colors = [colors.get(r["model"], "#999") for r in top]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Bar chart.
    bars = ax1.barh(range(len(top)), accs, color=bar_colors)
    ax1.set_yticks(range(len(top)))
    ax1.set_yticklabels(labels, fontsize=7)
    ax1.invert_yaxis()
    ax1.set_xlabel("Test Accuracy (%)")
    ax1.set_title("Top 10 Configurations")
    for bar, acc in zip(bars, accs):
        ax1.text(bar.get_width() + 0.05, bar.get_y() + bar.get_height() / 2,
                 f"{acc:.2f}%", va="center", fontsize=8)

    # Scatter plot.
    for m in sorted(set(r["model"] for r in results)):
        pts = [r for r in results if r["model"] == m]
        ax2.scatter(
            [p["total_time_seconds"] for p in pts],
            [p["test_accuracy"] for p in pts],
            c=colors.get(m, "#000"), label=m.upper(), s=35, alpha=0.7,
        )
    ax2.set_xlabel("Training Time (s)")
    ax2.set_ylabel("Test Accuracy (%)")
    ax2.set_title("Accuracy vs Training Time")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    configs = build_search_configs(args)
    print(f"Search: {args.search}")
    print(f"Configurations: {len(configs)}")
    for i, cfg in enumerate(configs):
        if i < 5 or i >= len(configs) - 3:
            print(f"  {i + 1:>3}. {run_name(cfg)}")
        elif i == 5:
            print(f"  ...")
    print()

    # Create tune output directory.
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    tune_dir = PROJECT_ROOT / "outputs" / f"tune_{args.search}_{timestamp}"
    tune_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output dir: {tune_dir}")

    # First download of MNIST will happen on the first run.
    print("Starting runs...")

    all_results = []
    for i, config in enumerate(configs):
        run_dir = tune_dir / run_name(config)
        try:
            result = run_one(
                config, run_dir,
                data_root=str(PROJECT_ROOT / "data"),
                base_seed=args.seed,
                run_index=i,
            )
            all_results.append(result)
        except Exception as e:
            print(f"\n  [SKIP] {run_name(config)} failed: {e}")

    if all_results:
        save_comparison(all_results, tune_dir)
        print(f"\nComparison saved to {tune_dir}/")
        print(f"  comparison.json")
        print(f"  comparison.pdf")
        print(f"  comparison_summary.png")
    else:
        print("\nNo successful runs.")


if __name__ == "__main__":
    main()
