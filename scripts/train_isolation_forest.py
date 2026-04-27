from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import precision_recall_curve, roc_curve

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.metrics import anomaly_metrics, pr_auc_binary, roc_auc_binary

COLORS = {
    "normal": "#2C7FB8",
    "anomaly": "#D95F02",
    "score": "#315C99",
    "threshold": "#222222",
    "event": "#F4A582",
}


def _resolve(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _load_config(path: str | Path) -> dict:
    with _resolve(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def _run_id() -> str:
    return pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")


def _flatten(windows: np.ndarray) -> np.ndarray:
    return windows.reshape(len(windows), -1).astype(np.float32)


def _load_window_run(run_dir: Path):
    metadata = json.loads((run_dir / "metadata.json").read_text())
    train_windows = np.load(run_dir / "train_windows.npy")
    val_windows = np.load(run_dir / "val_windows.npy")
    test_windows = np.load(run_dir / "test_windows.npy")
    train_labels = np.load(run_dir / "train_window_labels.npy")
    val_labels = np.load(run_dir / "val_window_labels.npy")
    test_labels = np.load(run_dir / "test_window_labels.npy")
    return (
        train_windows.astype(np.float32),
        val_windows.astype(np.float32),
        test_windows.astype(np.float32),
        train_labels.astype(np.int32),
        val_labels.astype(np.int32),
        test_labels.astype(np.int32),
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
    text = f"{percentile:g}".replace(".", "")
    return f"train_p{text}"


def _thresholds(train_scores: np.ndarray, val_scores: np.ndarray, val_labels: np.ndarray, cfg: dict) -> dict[str, dict]:
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
            row = {
                "threshold_method": threshold_method,
                "split": split,
            }
            row.update(anomaly_metrics(labels, preds, scores, threshold))
            rows.append(row)
    return rows


def _save_predictions(path: Path, labels: np.ndarray, scores: np.ndarray, thresholds: dict[str, dict]) -> None:
    df = pd.DataFrame({
        "window_index": np.arange(len(scores), dtype=np.int64),
        "label": labels.astype(np.int32),
        "score": scores.astype(np.float64),
    })
    for name, info in thresholds.items():
        df[f"pred_{name}"] = (scores >= float(info["threshold"])).astype(np.int32)
    df.to_csv(path, index=False)


def _style_axes(ax) -> None:
    ax.grid(axis="y", alpha=0.25, linestyle="--")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _save_figure(fig, paths: list[Path]) -> None:
    for path in paths:
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=200)


def _plot_score_hist(paths: list[Path], labels: np.ndarray, scores: np.ndarray, threshold: float) -> None:
    lo, hi = np.percentile(scores, [0.5, 99.8])
    bins = np.linspace(lo, hi, 90)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(scores[labels == 0], bins=bins, alpha=0.76, label="Normal", density=True, color=COLORS["normal"])
    ax.hist(scores[labels == 1], bins=bins, alpha=0.70, label="Anomaly", density=True, color=COLORS["anomaly"])
    ax.axvline(threshold, color=COLORS["threshold"], linestyle="--", linewidth=1.7, label="train_p98")
    ax.set_title("Isolation Forest Test Score Distribution")
    ax.set_xlabel("Anomaly score (-decision_function)")
    ax.set_ylabel("Density")
    ax.legend(frameon=False)
    _style_axes(ax)
    fig.tight_layout()
    _save_figure(fig, paths)
    plt.close(fig)


def _plot_roc(paths: list[Path], labels: np.ndarray, scores: np.ndarray) -> None:
    fpr, tpr, _ = roc_curve(labels, scores)
    auc = roc_auc_binary(labels, scores)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(fpr, tpr, color=COLORS["score"], linewidth=2.0, label=f"Isolation Forest (AUC={auc:.3f})")
    ax.plot([0, 1], [0, 1], color="#999999", linestyle="--", linewidth=1)
    ax.set_title("Isolation Forest Test ROC Curve")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.legend(frameon=False, loc="lower right")
    _style_axes(ax)
    fig.tight_layout()
    _save_figure(fig, paths)
    plt.close(fig)


def _plot_pr(paths: list[Path], labels: np.ndarray, scores: np.ndarray) -> None:
    precision, recall, _ = precision_recall_curve(labels, scores)
    auc = pr_auc_binary(labels, scores)
    prevalence = float(np.mean(labels == 1)) if len(labels) else 0.0
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(recall, precision, color=COLORS["score"], linewidth=2.0, label=f"Isolation Forest (AP={auc:.3f})")
    ax.axhline(prevalence, color="#999999", linestyle="--", linewidth=1, label=f"Prevalence={prevalence:.3f}")
    ax.set_title("Isolation Forest Test Precision-Recall Curve")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.legend(frameon=False)
    _style_axes(ax)
    fig.tight_layout()
    _save_figure(fig, paths)
    plt.close(fig)


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


def _plot_timeline(paths: list[Path], labels: np.ndarray, scores: np.ndarray, threshold: float) -> None:
    x = np.arange(len(scores), dtype=np.int64)
    fig, ax = plt.subplots(figsize=(10, 4.8))
    for i, (start, end) in enumerate(_event_spans(labels)):
        ax.axvspan(start, end, color=COLORS["event"], alpha=0.22, label="True anomaly interval" if i == 0 else None)
    ax.plot(x, scores, linewidth=0.75, color=COLORS["score"], label="Anomaly score")
    ax.axhline(threshold, color=COLORS["threshold"], linestyle="--", linewidth=1.4, label="train_p98")
    ax.set_title("Isolation Forest Test Timeline")
    ax.set_xlabel("Test window index")
    ax.set_ylabel("Anomaly score")
    ax.legend(frameon=False, loc="upper right")
    _style_axes(ax)
    fig.tight_layout()
    _save_figure(fig, paths)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train an unsupervised Isolation Forest baseline on normal windows.")
    parser.add_argument("--config", default="configs/baselines/isolation_forest_window60_noscale.json")
    args = parser.parse_args()

    cfg = _load_config(args.config)
    window_dir = _resolve(cfg["window_dir"])
    output_root = _resolve(cfg.get("output_dir", "models/baseline_runs/isolation_forest"))
    table_dir = _resolve(cfg.get("reports_table_dir", "reports/tables"))
    fig_dir = _resolve(cfg.get("reports_figure_dir", "reports/figures"))
    output_root.mkdir(parents=True, exist_ok=True)
    table_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    print(f"Window run: {window_dir}")
    train_w, val_w, test_w, train_y, val_y, test_y, metadata = _load_window_run(window_dir)
    x_train = _flatten(train_w)
    x_val = _flatten(val_w)
    x_test = _flatten(test_w)
    print(f"Loaded windows: train={x_train.shape}, val={x_val.shape}, test={x_test.shape}")
    print(f"Training labels: positives={int(np.sum(train_y == 1))}, normals={int(np.sum(train_y == 0))}")

    normal_mask = train_y == 0
    model_cfg = cfg.get("model", {})
    model = IsolationForest(**model_cfg)
    print("Training Isolation Forest on normal training windows only...")
    model.fit(x_train[normal_mask])

    print("Scoring train/val/test windows...")
    train_scores = -model.decision_function(x_train)
    val_scores = -model.decision_function(x_val)
    test_scores = -model.decision_function(x_test)
    thresholds = _thresholds(train_scores, val_scores, val_y, cfg.get("thresholds", {}))
    rows = _evaluate_all(
        thresholds,
        scores_by_split={"val": val_scores, "test": test_scores},
        labels_by_split={"val": val_y, "test": test_y},
    )

    run_id = _run_id()
    run_dir = output_root / run_id
    run_table_dir = run_dir / "tables"
    run_fig_dir = run_dir / "figures"
    structured_table_dir = table_dir / "baselines" / "isolation_forest"
    structured_fig_dir = fig_dir / "baselines" / "isolation_forest" / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    run_table_dir.mkdir(parents=True, exist_ok=True)
    run_fig_dir.mkdir(parents=True, exist_ok=True)
    structured_table_dir.mkdir(parents=True, exist_ok=True)
    structured_fig_dir.mkdir(parents=True, exist_ok=True)
    print(f"Saving run: {run_dir}")

    (run_dir / "config.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    joblib.dump(model, run_dir / "model.joblib")
    np.savez(
        run_dir / "scores.npz",
        train_scores=train_scores.astype(np.float32),
        val_scores=val_scores.astype(np.float32),
        test_scores=test_scores.astype(np.float32),
        train_labels=train_y.astype(np.int32),
        val_labels=val_y.astype(np.int32),
        test_labels=test_y.astype(np.int32),
    )
    (run_dir / "thresholds.json").write_text(json.dumps(thresholds, indent=2), encoding="utf-8")

    metrics_df = pd.DataFrame(rows)
    metrics_df.insert(0, "run_id", run_id)
    metrics_df.insert(1, "model", "Isolation Forest")
    metrics_df.insert(2, "window_dir", str(window_dir.relative_to(PROJECT_ROOT)))
    metrics_json = {
        "run_id": run_id,
        "model": "Isolation Forest",
        "window_dir": str(window_dir.relative_to(PROJECT_ROOT)),
        "source_metadata": metadata,
        "thresholds": thresholds,
        "metrics": rows,
    }
    (run_dir / "metrics.json").write_text(json.dumps(metrics_json, indent=2), encoding="utf-8")
    report_metrics_csv = structured_table_dir / f"{run_id}_metrics.csv"
    run_metrics_csv = run_table_dir / "metrics.csv"
    metrics_df.to_csv(report_metrics_csv, index=False)
    metrics_df.to_csv(run_metrics_csv, index=False)

    _save_predictions(run_dir / "predictions_val.csv", val_y, val_scores, thresholds)
    _save_predictions(run_dir / "predictions_test.csv", test_y, test_scores, thresholds)

    train_p98 = float(thresholds["train_p98"]["threshold"])
    _plot_score_hist(
        [run_fig_dir / "score_hist_test_train_p98.png", structured_fig_dir / "score_hist_test_train_p98.png"],
        test_y,
        test_scores,
        train_p98,
    )
    _plot_roc(
        [run_fig_dir / "roc_curve_test.png", structured_fig_dir / "roc_curve_test.png"],
        test_y,
        test_scores,
    )
    _plot_pr(
        [run_fig_dir / "pr_curve_test.png", structured_fig_dir / "pr_curve_test.png"],
        test_y,
        test_scores,
    )
    _plot_timeline(
        [run_fig_dir / "timeline_test_train_p98.png", structured_fig_dir / "timeline_test_train_p98.png"],
        test_y,
        test_scores,
        train_p98,
    )

    print(metrics_df[["threshold_method", "split", "precision", "recall", "f1", "roc_auc", "pr_auc"]].to_string(index=False))
    print(f"Saved run metrics: {run_metrics_csv}")
    print(f"Saved report metrics: {report_metrics_csv}")
    print(f"Saved run figures: {run_fig_dir}")
    print(f"Saved report figures: {structured_fig_dir}")


if __name__ == "__main__":
    main()
