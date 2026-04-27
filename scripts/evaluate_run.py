from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.metrics import binary_metrics, roc_auc_binary


def _resolve(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _metrics_for(run_dir: Path, split: str) -> dict:
    labels_path = run_dir / f"{split}_labels.npy"
    preds_path = run_dir / f"{split}_predictions.npy"
    scores_path = run_dir / f"{split}_scores.npy"
    if not labels_path.exists() or not preds_path.exists():
        return {}
    labels = np.load(labels_path)
    preds = np.load(preds_path)
    metrics = binary_metrics(labels, preds)
    if scores_path.exists():
        scores = np.load(scores_path)
        metrics["roc_auc"] = roc_auc_binary(labels, scores)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Recompute metrics for a saved VAE run.")
    parser.add_argument("--run-dir", required=True, help="Path to a saved run under models/vae_runs.")
    parser.add_argument("--output-dir", default="reports/tables", help="Directory for metrics table outputs.")
    args = parser.parse_args()

    run_dir = _resolve(args.run_dir)
    output_dir = _resolve(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_path = run_dir / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {}

    rows = []
    for split in ("train", "val", "test"):
        metrics = _metrics_for(run_dir, split)
        if metrics:
            row = {"run_id": run_dir.name, "split": split}
            row.update(metrics)
            rows.append(row)

    table = pd.DataFrame(rows)
    out_csv = output_dir / f"{run_dir.name}_metrics.csv"
    out_json = output_dir / f"{run_dir.name}_metrics.json"
    table.to_csv(out_csv, index=False)
    out_json.write_text(json.dumps({"summary": summary, "metrics": rows}, indent=2), encoding="utf-8")

    print(table.to_string(index=False) if len(table) else "No saved labels/predictions found.")
    print(f"Metrics CSV: {out_csv}")
    print(f"Metrics JSON: {out_json}")


if __name__ == "__main__":
    main()
