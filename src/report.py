"""
Report module.

Generates a PDF report summarizing the training run.
Uses fpdf2 (lightweight, no LaTeX dependency).
"""

from pathlib import Path
from fpdf import FPDF
import numpy as np


def generate_report(run_dir: str | Path) -> None:
    run_dir = Path(run_dir)
    report_path = run_dir / "report.pdf"
    pdf = FPDF()
    pdf.add_page()

    # Title
    pdf.set_font("Helvetica", "B", 20)
    pdf.cell(0, 15, "MNIST Classifier Report", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(5)

    # Run info
    pdf.set_font("Helvetica", "", 12)
    pdf.cell(0, 8, f"Run: {run_dir.name}", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(5)

    # Insert images
    images_to_include = [
        ("Training & Validation Loss", "loss.png"),
        ("Validation Accuracy", "accuracy.png"),
        ("Confusion Matrix", "confusion_matrix.png"),
        ("Sample Predictions", "sample_predictions.png"),
        ("Misclassified Images", "wrong_predictions.png"),
    ]

    for title, filename in images_to_include:
        filepath = run_dir / filename
        if filepath.exists():
            pdf.set_font("Helvetica", "B", 14)
            pdf.cell(0, 10, title, new_x="LMARGIN", new_y="NEXT")
            pdf.image(str(filepath), x=10, w=180)
            pdf.ln(5)

    # Metrics
    metrics_path = run_dir / "metrics.json"
    if metrics_path.exists():
        import json
        with open(metrics_path) as f:
            metrics = json.load(f)
        pdf.set_font("Helvetica", "B", 14)
        pdf.cell(0, 10, "Metrics", new_x="LMARGIN", new_y="NEXT")
        pdf.set_font("Courier", "", 10)
        pdf.cell(0, 6, f"Test Accuracy: {metrics['accuracy']:.2f}%", new_x="LMARGIN", new_y="NEXT")
        pdf.ln(2)
        pdf.set_font("Helvetica", "B", 11)
        pdf.cell(0, 8, "Per-Class Accuracy:", new_x="LMARGIN", new_y="NEXT")
        pdf.set_font("Courier", "", 10)
        for digit, acc in sorted(metrics["per_class_accuracy"].items()):
            bar = "#" * int(acc / 2)
            pdf.cell(0, 5, f"  Digit {digit}: {acc:5.2f}%  {bar}", new_x="LMARGIN", new_y="NEXT")

    # Config
    config_path = run_dir / "config.json"
    if config_path.exists():
        import json
        with open(config_path) as f:
            config = json.load(f)
        pdf.ln(5)
        pdf.set_font("Helvetica", "B", 14)
        pdf.cell(0, 10, "Configuration", new_x="LMARGIN", new_y="NEXT")
        pdf.set_font("Courier", "", 10)
        for key, value in config.items():
            pdf.cell(0, 5, f"  {key}: {value}", new_x="LMARGIN", new_y="NEXT")

    pdf.output(str(report_path))
    print(f"Report saved: {report_path}")
