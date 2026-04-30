"""Unified runner for the classical anomaly-detection baselines.

Trains one of {isolation_forest, pca_recon, oc_svm, lof} on the same flattened
windows that the VAE consumes, then writes scores, thresholds, metrics, and
paper-ready figures (score histogram, ROC, PR, timeline) to:

    models/baseline_runs/<model>/<run_id>/
    reports/tables/baselines/<model>/<run_id>_metrics.csv
    reports/figures/baselines/<model>/<run_id>/

The threshold logic exactly matches ``scripts/train_isolation_forest.py``
(percentiles of train scores + optional val-F1) so all four baselines are
directly comparable to the VAE table.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib  # noqa: F401  (kept for backward compatibility)
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve, roc_curve

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.baselines.classical import (  # noqa: E402
    BaselineResult,
    score_isolation_forest,
    score_lof,
    score_oc_svm,
    score_pca_reconstruction,
)
from src.evaluation.metrics import anomaly_metrics, pr_auc_binary, roc_auc_binary  # noqa: E402
from src.plotting import PALETTE, apply_paper_style, save_figure, style_axes  # noqa: E402


MODEL_REGISTRY = {
    "isolation_forest": ("Isolation Forest", score_isolation_forest),
    "pca_recon": ("PCA reconstruction", score_pca_reconstruction),
    "oc_svm": ("OC-SVM", score_oc_svm),
    "lof": ("LOF", score_lof),
}


def _resolve(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _load_config(path: str | Path) -> dict:
    with _resolve(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def _run_id() -> str:
    return pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")


def _load_window_run(run_dir: Path):
    metadata = json.loads((run_dir / "metadata.json").read_text())
    train_w = np.load(run_dir / "train_windows.npy")
    val_w = np.load(run_dir / "val_windows.npy")
    test_w = np.load(run_dir / "test_windows.npy")
    train_y = np.load(run_dir / "train_window_labels.npy")
    val_y = np.load(run_dir / "val_window_labels.npy")
    test_y = np.load(run_dir / "test_window_labels.npy")

    if train_w.ndim == 2:
        params = metadata["params"]
        w = 1 if params.get("point_mode", False) else int(params["window_size"])
        n_features = len(metadata["feature_cols"])
        train_w = train_w.reshape((-1, w, n_features))
        val_w = val_w.reshape((-1, w, n_features)) if len(val_w) else val_w
        test_w = test_w.reshape((-1, w, n_features))

    return (
        train_w.astype(np.float32),
        val_w.astype(np.float32),
        test_w.astype(np.float32),
        train_y.astype(np.int32),
        val_y.astype(np.int32),
        test_y.astype(np.int32),
        metadata,
    )


def _optimize_threshold_by_f1(scores: np.ndarray, labels: np.ndarray) -> tuple[float, float]:
    labels = labels.astype(np.int32)
    if len(scores) == 0 or np.unique(labels).size < 2:
        return float(np.percentile(scores, 95.0)), 0.0
    thresholds = np.unique(np.quantile(scores, np.linspace(0.01, 0.99, 200)))
    best_threshold = float(thresholds[len(thresholds) // 2])
    best_f1 = -1.0
    for threshold in thresholds:
        pred = (scores >= threshold).astype(np.int32)
        f1 = anomaly_metrics(labels, pred, scores, float(threshold))["f1"]
        if f1 > best_f1:
            best_f1 = float(f1)
            best_threshold = float(threshold)
    return best_threshold, best_f1


def _threshold_label(percentile: float) -> str:
    return f"train_p{f'{percentile:g}'.replace('.', '')}"


def _thresholds(
    train_scores: np.ndarray,
    val_scores: np.ndarray,
    val_labels: np.ndarray,
    cfg: dict,
) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for percentile in cfg.get("train_percentiles", [98, 99, 99.5]):
        label = _threshold_label(float(percentile))
        out[label] = {
            "method": label,
            "source": "train_scores",
            "percentile": float(percentile),
            "threshold": float(np.percentile(train_scores, float(percentile))),
            "fair_unsupervised": True,
        }

    if cfg.get("include_val_f1", True) and len(val_scores) and np.unique(val_labels).size >= 2:
        threshold, best_f1 = _optimize_threshold_by_f1(val_scores, val_labels)
        out["val_f1"] = {
            "method": "val_f1",
            "source": "validation_labels",
            "threshold": float(threshold),
            "best_val_f1": float(best_f1),
            "fair_unsupervised": False,
        }
    return out


def _evaluate_all(
    thresholds: dict[str, dict],
    scores_by_split: dict[str, np.ndarray],
    labels_by_split: dict[str, np.ndarray],
) -> list[dict]:
    rows: list[dict] = []
    for threshold_method, info in thresholds.items():
        threshold = float(info["threshold"])
        for split in ("val", "test"):
            scores = scores_by_split[split]
            labels = labels_by_split[split]
            preds = (scores >= threshold).astype(np.int32)
            row = {"threshold_method": threshold_method, "split": split}
            row.update(anomaly_metrics(labels, preds, scores, threshold))
            rows.append(row)
    return rows


def _save_predictions(
    path: Path, labels: np.ndarray, scores: np.ndarray, thresholds: dict[str, dict]
) -> None:
    df = pd.DataFrame(
        {
            "window_index": np.arange(len(scores), dtype=np.int64),
            "label": labels.astype(np.int32),
            "score": scores.astype(np.float64),
        }
    )
    for name, info in thresholds.items():
        df[f"pred_{name}"] = (scores >= float(info["threshold"])).astype(np.int32)
    df.to_csv(path, index=False)


def _plot_score_hist(
    paths: list[Path],
    labels: np.ndarray,
    scores: np.ndarray,
    threshold: float,
    *,
    model_name: str,
    threshold_label: str,
) -> None:
    if len(scores) == 0:
        return
    lo, hi = np.percentile(scores, [0.5, 99.8])
    if hi <= lo:
        hi = lo + 1e-6
    bins = np.linspace(lo, hi, 90)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(
        scores[labels == 0],
        bins=bins,
        alpha=0.78,
        label="Normal",
        density=True,
        color=PALETTE["normal"],
        edgecolor="white",
        linewidth=0.4,
    )
    ax.hist(
        scores[labels == 1],
        bins=bins,
        alpha=0.72,
        label="Anomaly",
        density=True,
        color=PALETTE["anomaly"],
        edgecolor="white",
        linewidth=0.4,
    )
    ax.axvline(threshold, color=PALETTE["threshold"], linestyle="--", linewidth=1.5, label=threshold_label)
    ax.set_title(f"{model_name} -- test score distribution")
    ax.set_xlabel("Anomaly score (higher = more anomalous)")
    ax.set_ylabel("Density")
    ax.legend(frameon=False, loc="upper right")
    style_axes(ax)
    fig.tight_layout()
    save_figure(fig, paths)


def _plot_roc(paths: list[Path], labels: np.ndarray, scores: np.ndarray, *, model_name: str) -> None:
    if len(labels) == 0 or np.unique(labels).size < 2:
        return
    fpr, tpr, _ = roc_curve(labels, scores)
    auc = roc_auc_binary(labels, scores)
    fig, ax = plt.subplots(figsize=(7.2, 5.4))
    ax.plot(fpr, tpr, color=PALETTE["primary"], linewidth=2.0, label=f"{model_name} (AUC={auc:.3f})")
    ax.plot([0, 1], [0, 1], color=PALETTE["subtle"], linestyle="--", linewidth=1)
    ax.set_title(f"{model_name} -- test ROC")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.legend(frameon=False, loc="lower right")
    style_axes(ax)
    fig.tight_layout()
    save_figure(fig, paths)


def _plot_pr(paths: list[Path], labels: np.ndarray, scores: np.ndarray, *, model_name: str) -> None:
    if len(labels) == 0 or np.unique(labels).size < 2:
        return
    precision, recall, _ = precision_recall_curve(labels, scores)
    auc = pr_auc_binary(labels, scores)
    prevalence = float(np.mean(labels == 1))
    fig, ax = plt.subplots(figsize=(7.2, 5.4))
    ax.plot(recall, precision, color=PALETTE["primary"], linewidth=2.0, label=f"{model_name} (AP={auc:.3f})")
    ax.axhline(prevalence, color=PALETTE["subtle"], linestyle="--", linewidth=1, label=f"prevalence={prevalence:.3f}")
    ax.set_title(f"{model_name} -- test precision-recall")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.legend(frameon=False, loc="upper right")
    style_axes(ax)
    fig.tight_layout()
    save_figure(fig, paths)


def _event_spans(labels: np.ndarray) -> list[tuple[int, int]]:
    idx = np.flatnonzero(labels == 1)
    if len(idx) == 0:
        return []
    spans: list[tuple[int, int]] = []
    start = int(idx[0])
    prev = int(idx[0])
    for value in idx[1:]:
        value = int(value)
        if value > prev + 1:
            spans.append((start, prev))
            start = value
        prev = value
    spans.append((start, prev))
    return spans


def _plot_timeline(
    paths: list[Path],
    labels: np.ndarray,
    scores: np.ndarray,
    threshold: float,
    *,
    model_name: str,
    threshold_label: str,
) -> None:
    if len(scores) == 0:
        return
    x = np.arange(len(scores), dtype=np.int64)
    fig, ax = plt.subplots(figsize=(11, 4.6))
    for i, (start, end) in enumerate(_event_spans(labels)):
        ax.axvspan(start, end, color=PALETTE["event"], alpha=0.22, label="True anomaly interval" if i == 0 else None)
    ax.plot(x, scores, linewidth=0.7, color=PALETTE["primary"], label="Anomaly score")
    ax.axhline(threshold, color=PALETTE["threshold"], linestyle="--", linewidth=1.4, label=threshold_label)
    ax.set_title(f"{model_name} -- test timeline")
    ax.set_xlabel("Test window index")
    ax.set_ylabel("Anomaly score")
    ax.legend(frameon=False, loc="upper right")
    style_axes(ax)
    fig.tight_layout()
    save_figure(fig, paths)


def _run_baseline(model_key: str, model_cfg: dict, *, train_w, val_w, test_w, train_y) -> BaselineResult:
    if model_key not in MODEL_REGISTRY:
        raise SystemExit(
            f"Unknown baseline '{model_key}'. Choose from: {sorted(MODEL_REGISTRY)}"
        )
    _, fn = MODEL_REGISTRY[model_key]
    return fn(train_w, val_w, test_w, train_y, **model_cfg)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train one of the classical anomaly-detection baselines.")
    parser.add_argument(
        "--config",
        required=True,
        help="Baseline JSON config (must include 'model_kind' and 'window_dir').",
    )
    args = parser.parse_args()

    apply_paper_style()

    cfg = _load_config(args.config)
    model_key = cfg["model_kind"]
    model_pretty, _ = MODEL_REGISTRY[model_key]
    window_dir = _resolve(cfg["window_dir"])
    output_root = _resolve(cfg.get("output_dir", f"models/baseline_runs/{model_key}"))
    table_dir = _resolve(cfg.get("reports_table_dir", "reports/tables"))
    fig_dir = _resolve(cfg.get("reports_figure_dir", "reports/figures"))
    output_root.mkdir(parents=True, exist_ok=True)
    table_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    print(f"Model: {model_pretty}")
    print(f"Window run: {window_dir}")
    train_w, val_w, test_w, train_y, val_y, test_y, metadata = _load_window_run(window_dir)
    print(f"Loaded windows: train={train_w.shape}, val={val_w.shape}, test={test_w.shape}")
    print(
        f"Train label balance: positives={int(np.sum(train_y == 1))}, normals={int(np.sum(train_y == 0))}"
    )

    print(f"Fitting {model_pretty}...")
    result = _run_baseline(
        model_key,
        cfg.get("model", {}),
        train_w=train_w,
        val_w=val_w,
        test_w=test_w,
        train_y=train_y,
    )

    thresholds = _thresholds(result.train_scores, result.val_scores, val_y, cfg.get("thresholds", {}))
    rows = _evaluate_all(
        thresholds,
        scores_by_split={"val": result.val_scores, "test": result.test_scores},
        labels_by_split={"val": val_y, "test": test_y},
    )

    run_id = _run_id()
    run_dir = output_root / run_id
    run_table_dir = run_dir / "tables"
    run_fig_dir = run_dir / "figures"
    structured_table_dir = table_dir / "baselines" / model_key
    structured_fig_dir = fig_dir / "baselines" / model_key / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    run_table_dir.mkdir(parents=True, exist_ok=True)
    run_fig_dir.mkdir(parents=True, exist_ok=True)
    structured_table_dir.mkdir(parents=True, exist_ok=True)
    structured_fig_dir.mkdir(parents=True, exist_ok=True)
    print(f"Saving run: {run_dir}")

    (run_dir / "config.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    np.savez(
        run_dir / "scores.npz",
        train_scores=result.train_scores.astype(np.float32),
        val_scores=result.val_scores.astype(np.float32),
        test_scores=result.test_scores.astype(np.float32),
        train_labels=train_y.astype(np.int32),
        val_labels=val_y.astype(np.int32),
        test_labels=test_y.astype(np.int32),
    )
    (run_dir / "thresholds.json").write_text(json.dumps(thresholds, indent=2), encoding="utf-8")

    metrics_df = pd.DataFrame(rows)
    metrics_df.insert(0, "run_id", run_id)
    metrics_df.insert(1, "model", model_pretty)
    metrics_df.insert(2, "model_kind", model_key)
    metrics_df.insert(3, "window_dir", str(window_dir.relative_to(PROJECT_ROOT)))
    metrics_json = {
        "run_id": run_id,
        "model": model_pretty,
        "model_kind": model_key,
        "window_dir": str(window_dir.relative_to(PROJECT_ROOT)),
        "source_metadata": metadata,
        "thresholds": thresholds,
        "metrics": rows,
        "extras": result.extras,
    }
    (run_dir / "metrics.json").write_text(json.dumps(metrics_json, indent=2), encoding="utf-8")
    report_metrics_csv = structured_table_dir / f"{run_id}_metrics.csv"
    run_metrics_csv = run_table_dir / "metrics.csv"
    metrics_df.to_csv(report_metrics_csv, index=False)
    metrics_df.to_csv(run_metrics_csv, index=False)

    _save_predictions(run_dir / "predictions_val.csv", val_y, result.val_scores, thresholds)
    _save_predictions(run_dir / "predictions_test.csv", test_y, result.test_scores, thresholds)

    train_p98 = float(thresholds["train_p98"]["threshold"])
    _plot_score_hist(
        [run_fig_dir / "score_hist_test_train_p98.png", structured_fig_dir / "score_hist_test_train_p98.png"],
        test_y,
        result.test_scores,
        train_p98,
        model_name=model_pretty,
        threshold_label="train_p98",
    )
    _plot_roc(
        [run_fig_dir / "roc_curve_test.png", structured_fig_dir / "roc_curve_test.png"],
        test_y,
        result.test_scores,
        model_name=model_pretty,
    )
    _plot_pr(
        [run_fig_dir / "pr_curve_test.png", structured_fig_dir / "pr_curve_test.png"],
        test_y,
        result.test_scores,
        model_name=model_pretty,
    )
    _plot_timeline(
        [run_fig_dir / "timeline_test_train_p98.png", structured_fig_dir / "timeline_test_train_p98.png"],
        test_y,
        result.test_scores,
        train_p98,
        model_name=model_pretty,
        threshold_label="train_p98",
    )

    print(metrics_df[["threshold_method", "split", "precision", "recall", "f1", "roc_auc", "pr_auc"]].to_string(index=False))
    print(f"Saved run metrics: {run_metrics_csv}")
    print(f"Saved report metrics: {report_metrics_csv}")
    print(f"Saved run figures: {run_fig_dir}")
    print(f"Saved report figures: {structured_fig_dir}")


if __name__ == "__main__":
    main()
