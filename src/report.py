"""
Report module.

Generates a PDF report summarizing the training run.
Uses fpdf2 (lightweight, no LaTeX dependency).
"""

import json
from pathlib import Path
from fpdf import FPDF


def _load_json(path: Path) -> dict | None:
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return None


def _section_title(pdf: FPDF, title: str) -> None:
    pdf.set_font("Helvetica", "B", 14)
    pdf.ln(4)
    pdf.cell(0, 10, title, new_x="LMARGIN", new_y="NEXT")


def _write_hardware(pdf: FPDF, stats: dict) -> None:
    _section_title(pdf, "Hardware")
    hw = stats["hardware"]
    pdf.set_font("Courier", "", 10)
    pdf.cell(0, 5, f"  Device:    {hw['device']}", new_x="LMARGIN", new_y="NEXT")
    pdf.cell(0, 5, f"  GPU:       {hw['gpu']}", new_x="LMARGIN", new_y="NEXT")
    pdf.cell(0, 5, f"  CPU:       {hw['cpu']}", new_x="LMARGIN", new_y="NEXT")


def _write_timing(pdf: FPDF, stats: dict) -> None:
    _section_title(pdf, "Training Time")
    tm = stats["timing"]
    total = tm["total_seconds"]
    minutes = int(total // 60)
    seconds = total % 60
    pdf.set_font("Courier", "", 10)
    pdf.cell(0, 5, f"  Total:           {total:.1f}s  ({minutes}m {seconds:.0f}s)", new_x="LMARGIN", new_y="NEXT")
    pdf.cell(0, 5, f"  Average/epoch:   {tm['avg_per_epoch']:.1f}s", new_x="LMARGIN", new_y="NEXT")


def _write_metrics_table(pdf: FPDF, stats: dict) -> None:
    _section_title(pdf, "Training History")

    col_w = [14, 26, 26, 26, 26, 26, 26]
    headers = ["Epoch", "Train Loss", "Val Loss", "Val Acc %", "Time (s)"]
    col_w = pdf.w / 5

    pdf.set_font("Helvetica", "B", 9)
    pdf.set_fill_color(230, 230, 230)
    for header in headers:
        pdf.cell(col_w, 7, header, border=1, fill=True, align="C")
    pdf.ln()

    pdf.set_font("Courier", "", 9)
    for ep in stats["epochs"]:
        row_color = (250, 250, 250) if ep["epoch"] % 2 == 0 else (255, 255, 255)
        pdf.set_fill_color(*row_color)
        pdf.cell(col_w, 6, str(ep["epoch"]), border=1, fill=True, align="C")
        pdf.cell(col_w, 6, f"{ep['train_loss']:.4f}", border=1, fill=True, align="C")
        pdf.cell(col_w, 6, f"{ep['val_loss']:.4f}", border=1, fill=True, align="C")
        pdf.cell(col_w, 6, f"{ep['val_acc']:.2f}%", border=1, fill=True, align="C")
        pdf.cell(col_w, 6, f"{ep['time_seconds']:.1f}", border=1, fill=True, align="C")
        pdf.ln()


def _write_final_metrics(pdf: FPDF, metrics: dict) -> None:
    _section_title(pdf, "Final Metrics")
    pdf.set_font("Courier", "", 10)
    pdf.cell(0, 6, f"  Test Accuracy: {metrics['accuracy']:.2f}%", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(2)

    pdf.set_font("Helvetica", "B", 11)
    pdf.cell(0, 8, "  Per-Class Accuracy:", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Courier", "", 10)
    for digit, acc in sorted(metrics["per_class_accuracy"].items()):
        bar = "#" * int(acc / 2)
        pdf.cell(0, 5, f"    Digit {digit}: {acc:5.2f}%  {bar}", new_x="LMARGIN", new_y="NEXT")

    # Top misclassifications.
    conf = metrics.get("confusion_matrix", [])
    if conf:
        pdf.ln(2)
        pdf.set_font("Helvetica", "B", 11)
        pdf.cell(0, 8, "  Top Misclassifications:", new_x="LMARGIN", new_y="NEXT")
        pdf.set_font("Courier", "", 10)
        # Build mistakes list: (count, true, pred)
        mistakes = []
        for i in range(10):
            for j in range(10):
                if i != j and conf[i][j] > 0:
                    mistakes.append((conf[i][j], i, j))
        mistakes.sort(reverse=True)
        for count, true, pred in mistakes[:5]:
            pdf.cell(0, 5, f"    Digit {true} confused with {pred}: {count} times", new_x="LMARGIN", new_y="NEXT")


def _write_images(pdf: FPDF, run_dir: Path) -> None:
    images_to_include = [
        ("Loss Curves", "loss.png"),
        ("Accuracy Curve", "accuracy.png"),
        ("Confusion Matrix", "confusion_matrix.png"),
        ("Sample Predictions", "sample_predictions.png"),
        ("Misclassified Images", "wrong_predictions.png"),
    ]
    for title, filename in images_to_include:
        filepath = run_dir / filename
        if filepath.exists():
            _section_title(pdf, title)
            pdf.image(str(filepath), x=10, w=180)
            pdf.ln(3)


def _write_config(pdf: FPDF, config: dict) -> None:
    _section_title(pdf, "Configuration")
    pdf.set_font("Courier", "", 10)
    for key, value in config.items():
        pdf.cell(0, 5, f"  {key}: {value}", new_x="LMARGIN", new_y="NEXT")


def generate_report(run_dir: str | Path) -> None:
    run_dir = Path(run_dir)
    report_path = run_dir / "report.pdf"
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    # Title
    pdf.set_font("Helvetica", "B", 20)
    pdf.cell(0, 15, "MNIST Classifier Report", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 11)
    pdf.cell(0, 8, f"Run: {run_dir.name}", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(4)

    config = _load_json(run_dir / "config.json")
    stats = _load_json(run_dir / "training_stats.json")
    metrics = _load_json(run_dir / "metrics.json")

    # Hardware + Timing section.
    if stats:
        _write_hardware(pdf, stats)
        _write_timing(pdf, stats)

    # Training history table.
    if stats and stats.get("epochs"):
        _write_metrics_table(pdf, stats)

    # Final metrics section.
    if metrics:
        _write_final_metrics(pdf, metrics)

    # Plots (images).
    _write_images(pdf, run_dir)

    # Config at the end.
    if config:
        _write_config(pdf, config)

    pdf.output(str(report_path))
    print(f"Report saved: {report_path}")
