from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score


def binary_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    y_true = y_true.astype(np.int32)
    y_pred = y_pred.astype(np.int32)

    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    accuracy = (tp + tn) / max(1, len(y_true))
    specificity = tn / (tn + fp) if (tn + fp) else 0.0
    balanced_accuracy = 0.5 * (recall + specificity)

    return {
        "accuracy": accuracy,
        "balanced_accuracy": balanced_accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


def roc_auc_binary(y_true: np.ndarray, scores: np.ndarray) -> float:
    y_true = y_true.astype(np.int32)
    scores = scores.astype(np.float64)

    pos = y_true == 1
    neg = y_true == 0
    n_pos = int(np.sum(pos))
    n_neg = int(np.sum(neg))
    if n_pos == 0 or n_neg == 0:
        return float("nan")

    order = np.argsort(scores)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(scores) + 1, dtype=np.float64)
    sum_pos = float(np.sum(ranks[pos]))
    auc = (sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return float(auc)


def pr_auc_binary(y_true: np.ndarray, scores: np.ndarray) -> float:
    y_true = y_true.astype(np.int32)
    scores = scores.astype(np.float64)
    if len(y_true) == 0 or np.unique(y_true).size < 2:
        return float("nan")
    return float(average_precision_score(y_true, scores))


def anomaly_metrics(y_true: np.ndarray, y_pred: np.ndarray, scores: np.ndarray, threshold: float) -> dict[str, float]:
    y_true = y_true.astype(np.int32)
    y_pred = y_pred.astype(np.int32)
    out = binary_metrics(y_true, y_pred)
    out["roc_auc"] = roc_auc_binary(y_true, scores)
    out["pr_auc"] = pr_auc_binary(y_true, scores)
    out["threshold"] = float(threshold)
    out["positive_rate_pred"] = float(np.mean(y_pred == 1)) if len(y_pred) else 0.0
    out["positive_rate_true"] = float(np.mean(y_true == 1)) if len(y_true) else 0.0
    return out
