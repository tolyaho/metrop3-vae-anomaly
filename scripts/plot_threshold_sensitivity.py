"""Sweep the train-score percentile threshold and plot per-model F1, precision, recall.

Useful for the paper to show *threshold robustness*: the deployed system's
decision rule is "flag windows whose anomaly score exceeds the p-th
percentile of training scores".  The figure tells the reader how
sensitive each model is to the chosen percentile.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.metrics import binary_metrics  # noqa: E402
from src.plotting import MODEL_COLORS, PALETTE, apply_paper_style, save_figure, style_axes  # noqa: E402


def _resolve(path):
    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _latest_subdir(p):
    if not p.exists():
        return None
    children = sorted(c for c in p.iterdir() if c.is_dir())
    return children[-1] if children else None


def _vae_scores(run_dir):
    return (
        np.load(run_dir / "train_scores.npy").astype(np.float64),
        np.load(run_dir / "test_scores.npy").astype(np.float64),
        np.load(run_dir / "test_labels.npy").astype(np.int32),
    )


def _baseline_scores(run_dir):
    data = np.load(run_dir / "scores.npz")
    return (
        data["train_scores"].astype(np.float64),
        data["test_scores"].astype(np.float64),
        data["test_labels"].astype(np.int32),
    )


def _sweep(train_scores, test_scores, test_labels, percentiles):
    rows = []
    for p in percentiles:
        threshold = float(np.percentile(train_scores, p))
        preds = (test_scores > threshold).astype(np.int32)
        m = binary_metrics(test_labels, preds)
        rows.append(
            {
                "percentile": float(p),
                "threshold": threshold,
                "precision": m["precision"],
                "recall": m["recall"],
                "f1": m["f1"],
                "fp": m["fp"],
                "tp": m["tp"],
            }
        )
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description="Sweep threshold percentile and plot F1/precision/recall vs percentile.")
    parser.add_argument("--vae-run-dir", default=None)
    parser.add_argument("--baselines", nargs="*", default=None, help="Optional <kind>:<run_dir> entries.")
    parser.add_argument("--output", default="reports/figures/threshold_sensitivity.png")
    parser.add_argument("--output-table", default="reports/tables/threshold_sensitivity.csv")
    parser.add_argument("--percentile-min", type=float, default=90.0)
    parser.add_argument("--percentile-max", type=float, default=99.9)
    parser.add_argument("--percentile-points", type=int, default=99)
    args = parser.parse_args()

    apply_paper_style()
    percentiles = np.linspace(args.percentile_min, args.percentile_max, args.percentile_points)

    sources = []
    vae_run = _resolve(args.vae_run_dir) if args.vae_run_dir else _latest_subdir(_resolve("models/vae_runs"))
    if vae_run is not None and (vae_run / "test_scores.npy").exists():
        sources.append(("VAE (dense)", vae_run, "vae"))

    baseline_kinds = {
        "isolation_forest": "Isolation Forest",
        "pca_recon": "PCA reconstruction",
        "oc_svm": "OC-SVM",
        "lof": "LOF",
    }
    explicit = {}
    if args.baselines:
        for entry in args.baselines:
            kind, _, path = entry.partition(":")
            explicit[kind.strip()] = _resolve(path.strip())
    for kind, label in baseline_kinds.items():
        run = explicit.get(kind) or _latest_subdir(_resolve(f"models/baseline_runs/{kind}"))
        if run and (run / "scores.npz").exists():
            sources.append((label, run, "baseline"))

    if not sources:
        raise SystemExit("No model runs found.")

    all_rows = []
    for label, run, kind in sources:
        if kind == "vae":
            train, test, y = _vae_scores(run)
        else:
            train, test, y = _baseline_scores(run)
        df = _sweep(train, test, y, percentiles)
        df.insert(0, "model", label)
        df.insert(1, "source", str(run.relative_to(PROJECT_ROOT)))
        all_rows.append(df)

    table = pd.concat(all_rows, ignore_index=True)
    out_table = _resolve(args.output_table)
    out_table.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(out_table, index=False)

    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.6), sharex=True)
    metric_panels = [("precision", "Precision"), ("recall", "Recall"), ("f1", "F1")]
    for ax, (col, pretty) in zip(axes, metric_panels):
        for label in table["model"].unique():
            sub = table[table["model"] == label]
            color = MODEL_COLORS.get(label.split(" (")[0], MODEL_COLORS.get(label, PALETTE["primary"]))
            ax.plot(sub["percentile"], sub[col], color=color, linewidth=1.9, label=label)
        ax.set_title(f"Test {pretty} vs train-score percentile")
        ax.set_xlabel("Training-score percentile (threshold)")
        ax.set_ylabel(pretty)
        ax.set_ylim(0, 1.02)
        ax.set_xlim(args.percentile_min, args.percentile_max)
        ax.axvline(98, color=PALETTE["subtle"], linestyle=":", linewidth=1)
        style_axes(ax, grid_axis="both")
        if col == "f1":
            ax.legend(frameon=False, loc="lower left", fontsize=8.6)

    fig.suptitle("Threshold sensitivity (train-score percentile sweep)", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out = _resolve(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    save_figure(fig, [out, out.with_suffix(".pdf")])

    print(f"Saved figure: {out}")
    print(f"Saved table: {out_table}")


if __name__ == "__main__":
    main()
