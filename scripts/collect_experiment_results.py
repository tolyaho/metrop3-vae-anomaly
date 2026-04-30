"""Collect grid run summaries into per-run CSV + multi-seed aggregate CSV.

Outputs (under ``study_dir/tables`` and ``reports/tables``):

  * ``grid_results.csv``         -- one row per (run, threshold, split).
  * ``grid_aggregated.csv``      -- mean +/- std over seeds for each
    (group, base_experiment_id, threshold, split).
  * ``grid_best_by_group.csv``   -- the best aggregated row per group
    using ``train_p98`` + test split.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.metrics import anomaly_metrics  # noqa: E402


REQUIRED_COLUMNS = [
    "study_id",
    "experiment_group",
    "base_experiment_id",
    "experiment_id",
    "run_id",
    "run_dir",
    "architecture",
    "window_size",
    "stride",
    "hidden",
    "latent_dim",
    "beta",
    "seed",
    "scaling",
    "threshold_method",
    "threshold_value",
    "split",
    "precision",
    "recall",
    "f1",
    "roc_auc",
    "pr_auc",
    "accuracy",
    "balanced_accuracy",
    "tn",
    "fp",
    "fn",
    "tp",
    "n_train_windows",
    "n_val_windows",
    "n_test_windows",
    "n_val_positive",
    "n_test_positive",
    "train_score_mean",
    "train_score_std",
    "train_score_p98",
    "test_score_mean_normal",
    "test_score_mean_anomaly",
]

AGGREGATED_METRIC_COLS = ["precision", "recall", "f1", "roc_auc", "pr_auc", "accuracy", "balanced_accuracy"]


def _resolve(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _threshold_label(percentile: float) -> str:
    return f"train_p{f'{percentile:g}'.replace('.', '')}"


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


def _thresholds(study_cfg: dict, run_dir: Path) -> dict[str, float]:
    train_scores = np.load(run_dir / "train_scores.npy")
    val_scores = np.load(run_dir / "val_scores.npy")
    val_labels = np.load(run_dir / "val_labels.npy")
    threshold_cfg = study_cfg.get("thresholds", {})
    out: dict[str, float] = {}
    for percentile in threshold_cfg.get("train_percentiles", [98, 99, 99.5]):
        out[_threshold_label(float(percentile))] = float(np.percentile(train_scores, float(percentile)))
    if threshold_cfg.get("include_val_f1", True) and len(val_scores) and np.unique(val_labels).size >= 2:
        threshold, _ = _optimize_threshold_by_f1(val_scores, val_labels)
        out["val_f1"] = float(threshold)
    return out


def _run_dirs(study_dir: Path) -> list[Path]:
    runs_root = study_dir / "runs"
    if not runs_root.exists():
        return []
    return sorted(p for p in runs_root.glob("*/*/*") if (p / "summary.json").exists())


def _row_base(study_id: str, run_dir: Path, summary: dict) -> dict:
    run_cfg = summary.get("run_config", {})
    grid_meta = run_cfg.get("grid_metadata", {})
    source_meta = summary.get("source_metadata", {})
    source_params = source_meta.get("params", {})
    vae_cfg = summary.get("vae_config", {})
    train_cfg = summary.get("train_config", {})

    train_scores = np.load(run_dir / "train_scores.npy")
    test_scores = np.load(run_dir / "test_scores.npy")
    val_labels = np.load(run_dir / "val_labels.npy")
    test_labels = np.load(run_dir / "test_labels.npy")

    return {
        "study_id": study_id,
        "experiment_group": grid_meta.get("experiment_group", ""),
        "base_experiment_id": grid_meta.get("base_experiment_id", grid_meta.get("experiment_id", run_dir.parent.name)),
        "experiment_id": grid_meta.get("experiment_id", run_dir.parent.name),
        "run_id": run_dir.name,
        "run_dir": str(run_dir.relative_to(PROJECT_ROOT)),
        "architecture": vae_cfg.get("architecture", "dense"),
        "window_size": int(grid_meta.get("window_size", source_params.get("window_size", 0))),
        "stride": int(grid_meta.get("stride", source_params.get("stride", 0))),
        "hidden": json.dumps(vae_cfg.get("hidden_units", [])),
        "latent_dim": int(vae_cfg.get("latent_dim", 0)),
        "beta": float(train_cfg.get("beta", 0.0)),
        "seed": int(grid_meta.get("seed", train_cfg.get("random_seed", 0))),
        "scaling": grid_meta.get("scaling", "none" if source_params.get("scale_features") is False else "standard"),
        "n_train_windows": int(summary.get("train_samples", len(train_scores))),
        "n_val_windows": int(summary.get("val_samples", len(val_labels))),
        "n_test_windows": int(summary.get("test_samples", len(test_labels))),
        "n_val_positive": int(np.sum(val_labels == 1)),
        "n_test_positive": int(np.sum(test_labels == 1)),
        "train_score_mean": float(np.mean(train_scores)),
        "train_score_std": float(np.std(train_scores)),
        "train_score_p98": float(np.percentile(train_scores, 98)),
        "test_score_mean_normal": float(np.mean(test_scores[test_labels == 0])) if np.any(test_labels == 0) else float("nan"),
        "test_score_mean_anomaly": float(np.mean(test_scores[test_labels == 1])) if np.any(test_labels == 1) else float("nan"),
    }


def _collect_run(study_id: str, study_cfg: dict, run_dir: Path) -> list[dict]:
    summary = _load_json(run_dir / "summary.json")
    base = _row_base(study_id, run_dir, summary)
    thresholds = _thresholds(study_cfg, run_dir)
    split_data = {
        "val": (np.load(run_dir / "val_scores.npy"), np.load(run_dir / "val_labels.npy")),
        "test": (np.load(run_dir / "test_scores.npy"), np.load(run_dir / "test_labels.npy")),
    }

    rows: list[dict] = []
    for method, threshold in thresholds.items():
        for split, (scores, labels) in split_data.items():
            preds = (scores > threshold).astype(np.int32)
            row = dict(base)
            row["threshold_method"] = method
            row["threshold_value"] = float(threshold)
            row["split"] = split
            row.update(anomaly_metrics(labels, preds, scores, threshold))
            row = {col: row.get(col, "") for col in REQUIRED_COLUMNS}
            rows.append(row)
    return rows


def _aggregate(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate seed replicates: mean and std for the metric columns."""
    if df.empty:
        return df

    group_cols = [
        "study_id",
        "experiment_group",
        "base_experiment_id",
        "architecture",
        "window_size",
        "stride",
        "hidden",
        "latent_dim",
        "beta",
        "scaling",
        "threshold_method",
        "split",
    ]
    agg: dict[str, tuple[str, str]] = {}
    for col in AGGREGATED_METRIC_COLS:
        agg[f"{col}_mean"] = (col, "mean")
        agg[f"{col}_std"] = (col, "std")
    agg["n_seeds"] = ("seed", "count")
    agg["seeds"] = ("seed", lambda s: ",".join(str(int(x)) for x in sorted(s)))
    agg["threshold_value_mean"] = ("threshold_value", "mean")
    agg["threshold_value_std"] = ("threshold_value", "std")

    grouped = df.groupby(group_cols, dropna=False).agg(**agg).reset_index()

    for col in AGGREGATED_METRIC_COLS:
        std_col = f"{col}_std"
        grouped[std_col] = grouped[std_col].fillna(0.0)
    return grouped


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect VAE grid results into CSV tables (incl. multi-seed aggregates).")
    parser.add_argument("--study-dir", required=True)
    args = parser.parse_args()

    study_dir = _resolve(args.study_dir)
    summary = _load_json(study_dir / "summary.json")
    study_id = summary.get("study_id", study_dir.name)
    study_cfg = summary.get("config", {})
    table_dir = study_dir / "tables"
    table_dir.mkdir(parents=True, exist_ok=True)
    reports_table_dir = PROJECT_ROOT / "reports" / "tables"
    reports_table_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    for run_dir in _run_dirs(study_dir):
        print(f"Collecting: {run_dir}")
        rows.extend(_collect_run(study_id, study_cfg, run_dir))
    if not rows:
        raise FileNotFoundError(f"No VAE run summaries found under {study_dir / 'runs'}")

    df = pd.DataFrame(rows, columns=REQUIRED_COLUMNS)
    out_local = table_dir / "grid_results.csv"
    out_global = reports_table_dir / "grid_results.csv"
    df.to_csv(out_local, index=False)
    df.to_csv(out_global, index=False)

    aggregated = _aggregate(df)
    agg_local = table_dir / "grid_aggregated.csv"
    agg_global = reports_table_dir / "grid_aggregated.csv"
    aggregated.to_csv(agg_local, index=False)
    aggregated.to_csv(agg_global, index=False)

    test_p98 = aggregated[
        (aggregated["split"] == "test") & (aggregated["threshold_method"] == "train_p98")
    ].copy()
    best = (
        test_p98.sort_values(["experiment_group", "f1_mean"], ascending=[True, False])
        .groupby("experiment_group", as_index=False)
        .head(1)
    )
    best_local = table_dir / "grid_best_by_group.csv"
    best_global = reports_table_dir / "grid_best_by_group.csv"
    best.to_csv(best_local, index=False)
    best.to_csv(best_global, index=False)

    print(f"Saved: {out_local}")
    print(f"Saved: {out_global}")
    print(f"Saved: {agg_local}")
    print(f"Saved: {agg_global}")
    print(f"Saved: {best_local}")
    print(f"Saved: {best_global}")


if __name__ == "__main__":
    main()
