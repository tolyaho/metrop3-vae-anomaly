"""Generate grid comparison figures (multi-seed aware).

Reads ``grid_aggregated.csv`` written by
``scripts/collect_experiment_results.py`` and produces error-bar
comparisons of layer / window / architecture sweeps at the
``train_p98`` threshold and on the ``test`` split.

Falls back to ``grid_results.csv`` if the aggregated file does not
exist (single-seed studies).
"""

from __future__ import annotations

import argparse
import ast
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.plotting import PALETTE, apply_paper_style, save_figure, style_axes  # noqa: E402

LAYER_ORDER = [
    "layers_64_32",
    "layers_128_64",
    "layers_256_128",
    "layers_128_64_32",
    "layers_256_128_64",
]
WINDOW_ORDER = ["win30", "win60", "win120"]
ARCH_ORDER = ["arch_conv1d", "arch_lstm"]


def _resolve(path):
    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _load_aggregated(study_dir):
    candidates = [
        study_dir / "tables" / "grid_aggregated.csv",
        PROJECT_ROOT / "reports" / "tables" / "grid_aggregated.csv",
    ]
    for path in candidates:
        if path.exists():
            return pd.read_csv(path), True
    candidates_raw = [
        study_dir / "tables" / "grid_results.csv",
        PROJECT_ROOT / "reports" / "tables" / "grid_results.csv",
    ]
    for path in candidates_raw:
        if path.exists():
            df = pd.read_csv(path)
            df = df.rename(
                columns={
                    "precision": "precision_mean",
                    "recall": "recall_mean",
                    "f1": "f1_mean",
                    "roc_auc": "roc_auc_mean",
                    "pr_auc": "pr_auc_mean",
                    "accuracy": "accuracy_mean",
                    "balanced_accuracy": "balanced_accuracy_mean",
                }
            )
            for col in ["precision", "recall", "f1", "roc_auc", "pr_auc", "accuracy", "balanced_accuracy"]:
                df[f"{col}_std"] = 0.0
            df["base_experiment_id"] = df.get("base_experiment_id", df["experiment_id"])
            df["n_seeds"] = 1
            return df, False
    raise FileNotFoundError("Run collect_experiment_results.py before plotting.")


def _hidden_label(value):
    try:
        parsed = ast.literal_eval(value)
        return "-".join(str(v) for v in parsed)
    except Exception:
        return str(value).replace("[", "").replace("]", "")


def _filter_group(df, group, *, threshold="train_p98", split="test"):
    out = df[
        (df["experiment_group"] == group)
        & (df["split"] == split)
        & (df["threshold_method"] == threshold)
    ].copy()
    if group == "layers":
        order = {exp_id: i for i, exp_id in enumerate(LAYER_ORDER)}
        out["order"] = out["base_experiment_id"].map(order).fillna(999).astype(int)
        out["label"] = out["hidden"].map(_hidden_label)
    elif group == "windows":
        order = {exp_id: i for i, exp_id in enumerate(WINDOW_ORDER)}
        out["order"] = out["base_experiment_id"].map(order).fillna(999).astype(int)
        out["label"] = out["window_size"].astype(str) + " steps"
    else:
        order = {exp_id: i for i, exp_id in enumerate(ARCH_ORDER)}
        out["order"] = out["base_experiment_id"].map(order).fillna(999).astype(int)
        out["label"] = out["architecture"].astype(str)
    return out.sort_values(["order", "base_experiment_id"])


def _bar_with_error(
    data,
    metric,
    *,
    title,
    xlabel,
    ylabel,
    out_paths,
):
    if data.empty:
        return
    mean_col = f"{metric}_mean"
    std_col = f"{metric}_std"
    fig, ax = plt.subplots(figsize=(max(7, 1.5 * len(data) + 4), 5.0))
    best_idx = data[mean_col].astype(float).idxmax()
    colors = [PALETTE["secondary"] if idx == best_idx else PALETTE["primary"] for idx in data.index]
    bars = ax.bar(
        data["label"],
        data[mean_col].astype(float),
        yerr=data[std_col].astype(float),
        capsize=4,
        color=colors,
        edgecolor="white",
        linewidth=0.7,
        error_kw={"elinewidth": 1.0, "ecolor": PALETTE["ink"]},
    )
    for bar, mean_val, std_val in zip(bars, data[mean_col], data[std_col]):
        text = f"{float(mean_val):.3f}"
        if float(std_val) > 0:
            text += f"\n+/-{float(std_val):.3f}"
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            float(mean_val) + float(std_val) + 0.018,
            text,
            ha="center",
            va="bottom",
            fontsize=8.5,
            color=PALETTE["ink"],
        )
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_ylim(0, max(1.05, float((data[mean_col] + data[std_col]).max()) * 1.18))
    plt.setp(ax.get_xticklabels(), rotation=20, ha="right")
    style_axes(ax, grid_axis="y")
    fig.tight_layout()
    save_figure(fig, out_paths)


def _precision_recall(data, *, title, xlabel, out_paths):
    if data.empty:
        return
    x = np.arange(len(data))
    fig, ax = plt.subplots(figsize=(max(7, 1.5 * len(data) + 4), 5.0))
    ax.errorbar(
        x,
        data["precision_mean"].astype(float),
        yerr=data["precision_std"].astype(float),
        marker="o",
        linewidth=1.9,
        label="Precision",
        color=PALETTE["primary"],
        capsize=3,
    )
    ax.errorbar(
        x,
        data["recall_mean"].astype(float),
        yerr=data["recall_std"].astype(float),
        marker="s",
        linewidth=1.9,
        label="Recall",
        color=PALETTE["warm"],
        capsize=3,
    )
    ax.errorbar(
        x,
        data["f1_mean"].astype(float),
        yerr=data["f1_std"].astype(float),
        marker="D",
        linewidth=2.2,
        label="F1",
        color=PALETTE["accent"],
        capsize=3,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(data["label"], rotation=20, ha="right")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.05)
    ax.set_title(title)
    style_axes(ax, grid_axis="y")
    ax.legend(frameon=False, loc="lower right")
    fig.tight_layout()
    save_figure(fig, out_paths)


def _summary_md(df, study_dir):
    test_p98 = df[(df["split"] == "test") & (df["threshold_method"] == "train_p98")].copy()
    best_by_group = (
        test_p98.sort_values(["experiment_group", "f1_mean"], ascending=[True, False])
        .groupby("experiment_group", as_index=False)
        .head(1)
    )
    lines = [
        "# Wrap-up experiment summary",
        "",
        f"Study directory: `{study_dir}`",
        "",
        "Reported numbers are mean +/- std across seed replicates at the `train_p98` threshold on the test split.",
        "",
        "## Best test results by group",
        "",
        "| Group | Base experiment | Window | Hidden | Architecture | Seeds | F1 | ROC-AUC | PR-AUC | Precision | Recall |",
        "|---|---|---:|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in best_by_group.itertuples():
        lines.append(
            f"| {row.experiment_group} | {row.base_experiment_id} | {row.window_size} | `{row.hidden}` | "
            f"{row.architecture} | {int(row.n_seeds)} | "
            f"{row.f1_mean:.3f}+/-{row.f1_std:.3f} | "
            f"{row.roc_auc_mean:.3f}+/-{row.roc_auc_std:.3f} | "
            f"{row.pr_auc_mean:.3f}+/-{row.pr_auc_std:.3f} | "
            f"{row.precision_mean:.3f}+/-{row.precision_std:.3f} | "
            f"{row.recall_mean:.3f}+/-{row.recall_std:.3f} |"
        )
    lines.extend(
        [
            "",
            "## All grid runs at train_p98",
            "",
            "| Group | Base experiment | Window | Hidden | Architecture | Seeds | F1 | ROC-AUC | PR-AUC |",
            "|---|---|---:|---|---|---:|---:|---:|---:|",
        ]
    )
    for row in test_p98.sort_values(["experiment_group", "f1_mean"], ascending=[True, False]).itertuples():
        lines.append(
            f"| {row.experiment_group} | {row.base_experiment_id} | {row.window_size} | `{row.hidden}` | "
            f"{row.architecture} | {int(row.n_seeds)} | "
            f"{row.f1_mean:.3f}+/-{row.f1_std:.3f} | "
            f"{row.roc_auc_mean:.3f}+/-{row.roc_auc_std:.3f} | "
            f"{row.pr_auc_mean:.3f}+/-{row.pr_auc_std:.3f} |"
        )
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description="Generate grid comparison figures.")
    parser.add_argument("--study-dir", required=True)
    args = parser.parse_args()

    apply_paper_style()
    study_dir = _resolve(args.study_dir)
    df, used_aggregated = _load_aggregated(study_dir)
    fig_dir = study_dir / "figures"
    report_dir = PROJECT_ROOT / "reports" / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    panels = [
        ("layers", "Hidden layers", "Layer comparison: F1 at train_p98 (test)", "layer_comparison_f1_train_p98.png"),
        ("windows", "Window size", "Window comparison: F1 at train_p98 (test)", "window_comparison_f1_train_p98.png"),
        ("architectures", "Architecture", "Architecture comparison: F1 at train_p98 (test)", "architecture_comparison_f1_train_p98.png"),
    ]
    for group, xlabel, title, fname in panels:
        sub = _filter_group(df, group)
        _bar_with_error(
            sub,
            "f1",
            title=title,
            xlabel=xlabel,
            ylabel="Test F1",
            out_paths=[fig_dir / fname, report_dir / fname],
        )
        _bar_with_error(
            sub,
            "roc_auc",
            title=title.replace("F1", "ROC-AUC"),
            xlabel=xlabel,
            ylabel="Test ROC-AUC",
            out_paths=[fig_dir / fname.replace("_f1_", "_auc_"), report_dir / fname.replace("_f1_", "_auc_")],
        )
        _precision_recall(
            sub,
            title=title.replace("F1", "Precision / Recall / F1"),
            xlabel=xlabel,
            out_paths=[
                fig_dir / fname.replace("_f1_", "_pr_"),
                report_dir / fname.replace("_f1_", "_pr_"),
            ],
        )

    summary = _summary_md(df, study_dir.relative_to(PROJECT_ROOT))
    (study_dir / "grid_summary.md").write_text(summary, encoding="utf-8")
    (PROJECT_ROOT / "reports" / "grid_summary.md").write_text(summary, encoding="utf-8")
    print(f"Saved figures to: {fig_dir}")
    print(f"Saved report figures to: {report_dir}")
    print(f"Saved summary: {PROJECT_ROOT / 'reports' / 'grid_summary.md'}")
    if not used_aggregated:
        print("(Used grid_results.csv -- no aggregated CSV present yet.)")


if __name__ == "__main__":
    main()
