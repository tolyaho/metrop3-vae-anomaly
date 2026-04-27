from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.metrics import anomaly_metrics

COLORS = {
    "VAE": "#2E7D32",
    "Isolation Forest": "#E68613",
}


def _resolve(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _optimize_threshold_by_f1(scores: np.ndarray, labels: np.ndarray) -> tuple[float, float]:
    labels = labels.astype(np.int32)
    if len(scores) == 0 or np.unique(labels).size < 2:
        return float(np.percentile(scores, 95.0)), 0.0
    thresholds = np.unique(np.quantile(scores, np.linspace(0.01, 0.99, 200)))
    best_threshold = float(thresholds[len(thresholds) // 2])
    best_f1 = -1.0
    for threshold in thresholds:
        pred = (scores > threshold).astype(np.int32)
        f1 = anomaly_metrics(labels, pred, scores, float(threshold))["f1"]
        if f1 > best_f1:
            best_f1 = float(f1)
            best_threshold = float(threshold)
    return best_threshold, best_f1


def _vae_rows(run_dir: Path) -> list[dict]:
    train_scores = np.load(run_dir / "train_scores.npy")
    val_scores = np.load(run_dir / "val_scores.npy")
    test_scores = np.load(run_dir / "test_scores.npy")
    val_labels = np.load(run_dir / "val_labels.npy")
    test_labels = np.load(run_dir / "test_labels.npy")

    thresholds = {
        "train_p98": float(np.percentile(train_scores, 98)),
    }
    threshold, _ = _optimize_threshold_by_f1(val_scores, val_labels)
    thresholds["val_f1"] = float(threshold)

    rows = []
    for method, threshold in thresholds.items():
        preds = (test_scores > threshold).astype(np.int32)
        row = {
            "model": "VAE",
            "run_id": run_dir.name,
            "threshold_method": method,
            "split": "test",
        }
        row.update(anomaly_metrics(test_labels, preds, test_scores, threshold))
        rows.append(row)
    return rows


def _if_rows(run_dir: Path) -> list[dict]:
    scores = np.load(run_dir / "scores.npz")
    thresholds = json.loads((run_dir / "thresholds.json").read_text())
    test_scores = scores["test_scores"]
    test_labels = scores["test_labels"]

    rows = []
    for method in ("train_p98", "val_f1"):
        if method not in thresholds:
            continue
        threshold = float(thresholds[method]["threshold"])
        preds = (test_scores >= threshold).astype(np.int32)
        row = {
            "model": "Isolation Forest",
            "run_id": run_dir.name,
            "threshold_method": method,
            "split": "test",
        }
        row.update(anomaly_metrics(test_labels, preds, test_scores, threshold))
        rows.append(row)
    return rows


def _plot_comparison(df: pd.DataFrame, out_path: Path) -> None:
    metrics = ["precision", "recall", "f1", "roc_auc", "pr_auc"]
    metric_labels = ["Precision", "Recall", "F1", "ROC-AUC", "PR-AUC"]
    x = np.arange(len(metrics), dtype=float)

    fig = plt.figure(figsize=(10.5, 5.8))
    gs = fig.add_gridspec(1, 2, width_ratios=[2.2, 1.0], wspace=0.22)
    ax_main = fig.add_subplot(gs[0, 0])
    ax_diag = fig.add_subplot(gs[0, 1])

    fair = df[df["threshold_method"] == "train_p98"].copy()
    width = 0.34
    for i, row in enumerate(fair.itertuples()):
        vals = [getattr(row, m) for m in metrics]
        offset = (i - (len(fair) - 1) / 2.0) * width
        bars = ax_main.bar(
            x + offset,
            vals,
            width=width,
            label=row.model,
            color=COLORS.get(row.model, "#666666"),
            alpha=0.92,
            edgecolor="white",
            linewidth=0.7,
        )
        for bar, val in zip(bars, vals):
            ax_main.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.018,
                f"{val:.2f}",
                ha="center",
                va="bottom",
                fontsize=8,
                color="#222222",
            )

    ax_main.set_title("Main Fair Comparison: Train-Only Threshold")
    ax_main.set_xlabel("Metric")
    ax_main.set_ylabel("Score")
    ax_main.set_xticks(x)
    ax_main.set_xticklabels(metric_labels)
    ax_main.set_ylim(0, 1.14)
    ax_main.grid(axis="y", alpha=0.24, linestyle="--")
    ax_main.legend(frameon=False, loc="lower right")
    if set(fair["model"]) >= {"VAE", "Isolation Forest"}:
        vae_row = fair[fair["model"] == "VAE"].iloc[0]
        if_row = fair[fair["model"] == "Isolation Forest"].iloc[0]
        fp_ratio = if_row["fp"] / max(1, vae_row["fp"])
        ax_main.text(
            0.02,
            0.97,
            f"At train_p98: VAE has higher F1 ({vae_row['f1']:.3f} vs {if_row['f1']:.3f}); "
            f"IF has {fp_ratio:.1f}x more false positives.",
            transform=ax_main.transAxes,
            fontsize=8.5,
            color="#333333",
            va="top",
            bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "edgecolor": "#DDDDDD", "alpha": 0.92},
        )

    diag = df[df["threshold_method"] == "val_f1"].copy()
    diag_metrics = ["precision", "recall", "f1"]
    diag_labels = ["Precision", "Recall", "F1"]
    dx = np.arange(len(diag_metrics), dtype=float)
    dwidth = 0.34
    for i, row in enumerate(diag.itertuples()):
        vals = [getattr(row, m) for m in diag_metrics]
        offset = (i - (len(diag) - 1) / 2.0) * dwidth
        bars = ax_diag.bar(
            dx + offset,
            vals,
            width=dwidth,
            label=row.model,
            color=COLORS.get(row.model, "#666666"),
            alpha=0.42,
            edgecolor="#333333",
            linewidth=0.5,
            hatch="//",
        )
        for bar, val in zip(bars, vals):
            ax_diag.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.018,
                f"{val:.2f}",
                ha="center",
                va="bottom",
                fontsize=8,
                color="#333333",
            )

    ax_diag.set_title("Diagnostic: Val-F1 Threshold")
    ax_diag.set_xlabel("Metric")
    ax_diag.set_xticks(dx)
    ax_diag.set_xticklabels(diag_labels)
    ax_diag.set_ylim(0, 1.14)
    ax_diag.grid(axis="y", alpha=0.24, linestyle="--")
    ax_diag.text(
        0.02,
        0.04,
        "Uses validation labels\nnot the main fair claim",
        transform=ax_diag.transAxes,
        fontsize=8,
        color="#444444",
        va="bottom",
    )

    fig.suptitle("VAE vs Isolation Forest on Raw MetroPT3 Windows", fontsize=14, fontweight="bold")
    fig.subplots_adjust(top=0.82, bottom=0.16, left=0.07, right=0.98)
    plt.savefig(out_path, dpi=200)
    plt.close()


def _annotate_rows(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["threshold_scope"] = np.where(
        out["threshold_method"] == "train_p98",
        "train-only",
        "validation-label diagnostic",
    )
    out["is_primary"] = out["threshold_method"] == "train_p98"
    out["report_note"] = np.where(
        out["is_primary"],
        "Primary fair unsupervised comparison.",
        "Diagnostic only; threshold selected with validation labels.",
    )
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare saved VAE and Isolation Forest baseline runs.")
    parser.add_argument("--vae-run-dir", required=True)
    parser.add_argument("--if-run-dir", required=True)
    parser.add_argument("--output-table", default="reports/tables/baseline_comparison_latest.csv")
    parser.add_argument("--output-figure", default="reports/figures/baseline_comparison_latest.png")
    args = parser.parse_args()

    vae_run_dir = _resolve(args.vae_run_dir)
    if_run_dir = _resolve(args.if_run_dir)
    out_table = _resolve(args.output_table)
    out_figure = _resolve(args.output_figure)
    structured_table = PROJECT_ROOT / "reports" / "tables" / "baselines" / "baseline_comparison_latest.csv"
    primary_table = PROJECT_ROOT / "reports" / "tables" / "baselines" / "baseline_comparison_primary_train_p98.csv"
    structured_figure = PROJECT_ROOT / "reports" / "figures" / "baselines" / "baseline_comparison_latest.png"
    out_table.parent.mkdir(parents=True, exist_ok=True)
    out_figure.parent.mkdir(parents=True, exist_ok=True)
    structured_table.parent.mkdir(parents=True, exist_ok=True)
    primary_table.parent.mkdir(parents=True, exist_ok=True)
    structured_figure.parent.mkdir(parents=True, exist_ok=True)

    rows = _vae_rows(vae_run_dir) + _if_rows(if_run_dir)
    df = _annotate_rows(pd.DataFrame(rows))
    df.to_csv(out_table, index=False)
    df.to_csv(structured_table, index=False)
    df[df["is_primary"]].to_csv(primary_table, index=False)
    _plot_comparison(df, out_figure)
    _plot_comparison(df, structured_figure)

    print(df[["model", "threshold_method", "precision", "recall", "f1", "roc_auc", "pr_auc"]].to_string(index=False))
    print(f"Saved table: {out_table}")
    print(f"Saved structured table: {structured_table}")
    print(f"Saved primary table: {primary_table}")
    print(f"Saved figure: {out_figure}")
    print(f"Saved structured figure: {structured_figure}")


if __name__ == "__main__":
    main()
