from __future__ import annotations

import argparse
import ast
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]

COLORS = {
    "primary": "#2E7D32",
    "secondary": "#4C78A8",
    "muted": "#B8C7D9",
    "accent": "#E68613",
    "bad": "#B9B9B9",
    "grid": "#DADADA",
}

LAYER_ORDER = [
    "layers_64_32",
    "layers_128_64",
    "layers_256_128",
    "layers_128_64_32",
    "layers_256_128_64",
]
WINDOW_ORDER = ["win30", "win60", "win120"]


def _resolve(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _load_results(study_dir: Path) -> pd.DataFrame:
    candidates = [
        study_dir / "tables" / "wrapup_grid_results.csv",
        PROJECT_ROOT / "reports" / "tables" / "wrapup_grid_results.csv",
    ]
    for path in candidates:
        if path.exists():
            return pd.read_csv(path)
    raise FileNotFoundError("Run collect_experiment_results.py before plotting.")


def _hidden_label(value: str) -> str:
    try:
        parsed = ast.literal_eval(value)
        return "-".join(str(v) for v in parsed)
    except Exception:
        return str(value).replace("[", "").replace("]", "")


def _style_axes(ax) -> None:
    ax.grid(axis="y", alpha=0.25, linestyle="--", color=COLORS["grid"])
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _annotate_bars(ax, bars, fmt: str = "{:.3f}") -> None:
    for bar in bars:
        val = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            val + 0.018,
            fmt.format(val),
            ha="center",
            va="bottom",
            fontsize=8.5,
            color="#222222",
        )


def _test_p98(df: pd.DataFrame, group: str) -> pd.DataFrame:
    out = df[
        (df["experiment_group"] == group)
        & (df["split"] == "test")
        & (df["threshold_method"] == "train_p98")
    ].copy()
    if group == "layers":
        order = {exp_id: i for i, exp_id in enumerate(LAYER_ORDER)}
        out["order"] = out["experiment_id"].map(order).fillna(999).astype(int)
        out["label"] = out["hidden"].map(_hidden_label)
    else:
        order = {exp_id: i for i, exp_id in enumerate(WINDOW_ORDER)}
        out["order"] = out["experiment_id"].map(order).fillna(999).astype(int)
        out["label"] = out["window_size"].astype(str) + " steps"
    return out.sort_values(["order", "experiment_id"])


def _save_all(fig_path: Path, report_path: Path) -> None:
    plt.tight_layout()
    plt.savefig(fig_path, dpi=200, bbox_inches="tight")
    if fig_path.resolve() != report_path.resolve():
        plt.savefig(report_path, dpi=200, bbox_inches="tight")
    plt.close()


def _bar_metric(df: pd.DataFrame, group: str, metric: str, ylabel: str, title: str, filename: str, fig_dir: Path, report_dir: Path) -> None:
    data = _test_p98(df, group)
    best_idx = data[metric].astype(float).idxmax()
    colors = [COLORS["primary"] if idx == best_idx else COLORS["secondary"] for idx in data.index]
    fig, ax = plt.subplots(figsize=(8.6, 5.0))
    bars = ax.bar(data["label"], data[metric], color=colors, alpha=0.92, edgecolor="white", linewidth=0.8)
    _annotate_bars(ax, bars)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xlabel("Hidden layers" if group == "layers" else "Window size")
    ax.set_ylabel(ylabel)
    ax.set_xticks(np.arange(len(data)))
    ax.set_xticklabels(data["label"], rotation=25 if group == "layers" else 0, ha="right" if group == "layers" else "center")
    ax.set_ylim(0, 1.10 if metric in {"f1", "roc_auc"} else max(1.0, float(data[metric].max()) * 1.18))
    _style_axes(ax)
    plt.sca(ax)
    _save_all(fig_dir / filename, report_dir / filename)


def _precision_recall(df: pd.DataFrame, group: str, title: str, filename: str, fig_dir: Path, report_dir: Path) -> None:
    data = _test_p98(df, group)
    x = np.arange(len(data))
    fig, ax = plt.subplots(figsize=(8.6, 5.0))
    ax.plot(x, data["precision"], marker="o", linewidth=2, label="Precision", color=COLORS["primary"])
    ax.plot(x, data["recall"], marker="o", linewidth=2, label="Recall", color=COLORS["accent"])
    for i, row in enumerate(data.itertuples()):
        ax.text(i, row.precision + 0.025, f"{row.precision:.2f}", ha="center", fontsize=8, color=COLORS["primary"])
        ax.text(i, row.recall + 0.025, f"{row.recall:.2f}", ha="center", fontsize=8, color=COLORS["accent"])
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xlabel("Hidden layers" if group == "layers" else "Window size")
    ax.set_ylabel("Score")
    ax.set_xticks(x)
    ax.set_xticklabels(data["label"], rotation=25 if group == "layers" else 0, ha="right" if group == "layers" else "center")
    ax.set_ylim(0, 1.08)
    ax.legend(frameon=False)
    _style_axes(ax)
    _save_all(fig_dir / filename, report_dir / filename)


def _final_model_comparison(df: pd.DataFrame, fig_dir: Path, report_dir: Path) -> None:
    rows = []
    test_p98 = df[(df["split"] == "test") & (df["threshold_method"] == "train_p98")].copy()
    if not test_p98.empty:
        best = test_p98.sort_values("f1", ascending=False).iloc[0]
        rows.append({
            "label": f"Best grid VAE\n{best['experiment_id']}",
            "precision": best["precision"],
            "recall": best["recall"],
            "f1": best["f1"],
            "roc_auc": best["roc_auc"],
            "pr_auc": best["pr_auc"],
        })

    baseline_path = PROJECT_ROOT / "reports" / "tables" / "baselines" / "baseline_comparison_latest.csv"
    if baseline_path.exists():
        base = pd.read_csv(baseline_path)
        for row in base.itertuples():
            if row.threshold_method == "train_p98":
                rows.append({
                    "label": f"{row.model}\n{row.threshold_method}",
                    "precision": row.precision,
                    "recall": row.recall,
                    "f1": row.f1,
                    "roc_auc": row.roc_auc,
                    "pr_auc": row.pr_auc,
                })

    if not rows:
        return

    data = pd.DataFrame(rows)
    metrics = ["precision", "recall", "f1", "roc_auc", "pr_auc"]
    labels = ["Precision", "Recall", "F1", "ROC-AUC", "PR-AUC"]
    x = np.arange(len(metrics), dtype=float)
    width = 0.8 / max(1, len(data))
    palette = [COLORS["primary"], COLORS["secondary"], COLORS["accent"], COLORS["muted"]]
    fig, ax = plt.subplots(figsize=(10, 5.4))
    for i, row in enumerate(data.itertuples()):
        offsets = x + (i - (len(data) - 1) / 2.0) * width
        bars = ax.bar(offsets, [getattr(row, m) for m in metrics], width=width, label=row.label, alpha=0.92, color=palette[i % len(palette)], edgecolor="white", linewidth=0.6)
        for bar, metric in zip(bars, metrics):
            if metric == "f1":
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.018, f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=8)
    ax.set_title("Final Model Comparison at train_p98", fontsize=13, fontweight="bold")
    ax.set_xlabel("Metric")
    ax.set_ylabel("Score")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1.10)
    ax.legend(fontsize=8, frameon=False, loc="center left", bbox_to_anchor=(1.01, 0.5))
    _style_axes(ax)
    _save_all(fig_dir / "final_model_comparison.png", report_dir / "final_model_comparison.png")


def _summary_md(df: pd.DataFrame, study_dir: Path) -> str:
    test_p98 = df[(df["split"] == "test") & (df["threshold_method"] == "train_p98")].copy()
    best_by_group = (
        test_p98.sort_values(["experiment_group", "f1"], ascending=[True, False])
        .groupby("experiment_group", as_index=False)
        .head(1)
    )
    lines = [
        "# Wrap-Up Experiment Summary",
        "",
        f"Study directory: `{study_dir}`",
        "",
        "Main threshold: `train_p98` from training reconstruction scores.",
        "Validation-label thresholds are kept only as diagnostics.",
        "",
        "## Best Test Results By Group",
        "",
        "| Group | Experiment | Window | Hidden | Precision | Recall | F1 | ROC-AUC | PR-AUC |",
        "|---|---|---:|---|---:|---:|---:|---:|---:|",
    ]
    for row in best_by_group.itertuples():
        lines.append(
            f"| {row.experiment_group} | {row.experiment_id} | {row.window_size} | `{row.hidden}` | "
            f"{row.precision:.3f} | {row.recall:.3f} | {row.f1:.3f} | {row.roc_auc:.3f} | {row.pr_auc:.3f} |"
        )
    lines.extend([
        "",
        "## All Fresh Grid Runs at train_p98",
        "",
        "| Group | Experiment | Window | Hidden | Precision | Recall | F1 | ROC-AUC | PR-AUC |",
        "|---|---|---:|---|---:|---:|---:|---:|---:|",
    ])
    for row in test_p98.sort_values(["experiment_group", "f1"], ascending=[True, False]).itertuples():
        lines.append(
            f"| {row.experiment_group} | {row.experiment_id} | {row.window_size} | `{row.hidden}` | "
            f"{row.precision:.3f} | {row.recall:.3f} | {row.f1:.3f} | {row.roc_auc:.3f} | {row.pr_auc:.3f} |"
        )

    baseline_path = PROJECT_ROOT / "reports" / "tables" / "baselines" / "baseline_comparison_primary_train_p98.csv"
    if baseline_path.exists():
        base = pd.read_csv(baseline_path)
        lines.extend([
            "",
            "## Final Baseline Context",
            "",
            "| Model | Run | Precision | Recall | F1 | ROC-AUC | PR-AUC |",
            "|---|---|---:|---:|---:|---:|---:|",
        ])
        for row in base.itertuples():
            lines.append(
                f"| {row.model} | `{row.run_id}` | {row.precision:.3f} | {row.recall:.3f} | "
                f"{row.f1:.3f} | {row.roc_auc:.3f} | {row.pr_auc:.3f} |"
            )
    lines.extend([
        "",
        "## Notes",
        "",
        "- The comparison uses raw, unscaled windows.",
        "- `val_f1` rows are diagnostic because they use validation labels.",
        "- The main model-selection threshold is `train_p98`.",
    ])
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate wrap-up grid comparison figures.")
    parser.add_argument("--study-dir", required=True)
    args = parser.parse_args()

    study_dir = _resolve(args.study_dir)
    df = _load_results(study_dir)
    fig_dir = study_dir / "figures"
    report_dir = PROJECT_ROOT / "reports" / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    _bar_metric(df, "layers", "f1", "Test F1", "Layer Comparison: F1 at train_p98", "layer_comparison_f1_train_p98.png", fig_dir, report_dir)
    _bar_metric(df, "layers", "roc_auc", "Test ROC-AUC", "Layer Comparison: ROC-AUC at train_p98", "layer_comparison_auc_train_p98.png", fig_dir, report_dir)
    _precision_recall(df, "layers", "Layer Comparison: Precision/Recall at train_p98", "layer_precision_recall_train_p98.png", fig_dir, report_dir)

    _bar_metric(df, "windows", "f1", "Test F1", "Window Comparison: F1 at train_p98", "window_comparison_f1_train_p98.png", fig_dir, report_dir)
    _bar_metric(df, "windows", "roc_auc", "Test ROC-AUC", "Window Comparison: ROC-AUC at train_p98", "window_comparison_auc_train_p98.png", fig_dir, report_dir)
    _precision_recall(df, "windows", "Window Comparison: Precision/Recall at train_p98", "window_precision_recall_train_p98.png", fig_dir, report_dir)
    _final_model_comparison(df, fig_dir, report_dir)

    summary = _summary_md(df, study_dir.relative_to(PROJECT_ROOT))
    (study_dir / "wrapup_experiment_summary.md").write_text(summary, encoding="utf-8")
    (PROJECT_ROOT / "reports" / "wrapup_experiment_summary.md").write_text(summary, encoding="utf-8")
    print(f"Saved figures to: {fig_dir}")
    print(f"Saved report figures to: {report_dir}")
    print(f"Saved summary: {PROJECT_ROOT / 'reports' / 'wrapup_experiment_summary.md'}")


if __name__ == "__main__":
    main()
