"""Evaluate a saved VAE run with multiple threshold strategies and save a comparison table."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.metrics import binary_metrics, roc_auc_binary
from src.models.vae_anomaly_detector import classify, optimize_threshold_by_f1


def evaluate_with_threshold(
    run_dir: Path,
    threshold: float,
    threshold_label: str,
) -> dict:
    train_scores = np.load(run_dir / "train_scores.npy")
    val_scores   = np.load(run_dir / "val_scores.npy")
    val_labels   = np.load(run_dir / "val_labels.npy")
    test_scores  = np.load(run_dir / "test_scores.npy")
    test_labels  = np.load(run_dir / "test_labels.npy")

    rows = []
    for split, scores, labels in [
        ("val", val_scores, val_labels),
        ("test", test_scores, test_labels),
    ]:
        preds = classify(scores, threshold)
        m = binary_metrics(labels, preds)
        roc = roc_auc_binary(labels, scores)
        row = {
            "run_id": run_dir.name,
            "threshold_rule": threshold_label,
            "threshold": threshold,
            "split": split,
        }
        row.update(m)
        row["roc_auc"] = roc
        rows.append(row)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--output-dir", default="reports/tables")
    args = parser.parse_args()

    run_dir = (PROJECT_ROOT / args.run_dir).resolve()
    output_dir = (PROJECT_ROOT / args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    train_scores = np.load(run_dir / "train_scores.npy")
    val_scores   = np.load(run_dir / "val_scores.npy")
    val_labels   = np.load(run_dir / "val_labels.npy")

    # Collect threshold strategies
    thresholds: list[tuple[str, float]] = []

    th_vf1, vf1 = optimize_threshold_by_f1(val_scores, val_labels)
    thresholds.append((f"val_f1 (val_f1={vf1:.4f})", th_vf1))

    for pct in [95, 97, 98, 99]:
        th = float(np.percentile(train_scores, pct))
        thresholds.append((f"train_p{pct}", th))

    mean3std = float(np.mean(train_scores) + 3.0 * np.std(train_scores))
    thresholds.append(("train_mean3std", mean3std))

    all_rows = []
    for label, th in thresholds:
        rows = evaluate_with_threshold(run_dir, th, label)
        all_rows.extend(rows)

    df = pd.DataFrame(all_rows)
    out_csv = output_dir / f"{run_dir.name}_threshold_comparison.csv"
    df.to_csv(out_csv, index=False)

    print(df[["threshold_rule", "split", "threshold", "precision", "recall", "f1", "roc_auc"]].to_string(index=False))
    print(f"\nSaved: {out_csv}")

    # Update summary.json with val_f1 metrics
    summary_path = run_dir / "summary.json"
    if summary_path.exists():
        summary = json.loads(summary_path.read_text())
        test_row = df[(df["threshold_rule"].str.startswith("val_f1")) & (df["split"] == "test")].iloc[0].to_dict()
        val_row  = df[(df["threshold_rule"].str.startswith("val_f1")) & (df["split"] == "val")].iloc[0].to_dict()
        summary["threshold"] = th_vf1
        summary["threshold_info"] = {"method": "val_f1", "best_val_f1": vf1}
        summary["metrics"] = {k: test_row.get(k) for k in ["accuracy","precision","recall","f1","tp","tn","fp","fn","roc_auc"]}
        summary["validation_metrics"] = {k: val_row.get(k) for k in ["accuracy","precision","recall","f1","tp","tn","fp","fn","roc_auc"]}
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"Updated summary.json with val_f1 threshold={th_vf1:.4f}")


if __name__ == "__main__":
    main()
