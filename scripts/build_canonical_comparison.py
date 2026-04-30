"""Assemble the canonical model comparison table used in the report.

Pulls aggregated VAE metrics (mean +/- std across seeds for the chosen
``base_experiment_id``) from ``grid_aggregated.csv`` and per-baseline
metrics from each ``models/baseline_runs/<kind>/<run>/metrics.json`` so that
every model is reported under the same threshold rule and on the same split.

Outputs:
  * reports/tables/baselines/baseline_comparison_primary_train_p98.csv
  * reports/tables/baselines/baseline_comparison_primary_val_f1.csv
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


METRIC_COLS = ["precision", "recall", "f1", "roc_auc", "pr_auc", "tp", "fp", "fn", "tn", "balanced_accuracy", "accuracy"]


def _resolve(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else PROJECT_ROOT / p


def _latest_subdir(p: Path) -> Path | None:
    if not p.exists():
        return None
    children = sorted(c for c in p.iterdir() if c.is_dir())
    return children[-1] if children else None


def _baseline_row(run_dir: Path, threshold_method: str, split: str) -> dict | None:
    metrics = json.loads((run_dir / "metrics.json").read_text())
    for entry in metrics["metrics"]:
        if entry["threshold_method"] == threshold_method and entry["split"] == split:
            return {
                "model": metrics["model"],
                "model_kind": metrics.get("model_kind"),
                "run_id": metrics["run_id"],
                "threshold_method": threshold_method,
                "split": split,
                "n_seeds": 1,
                **{f"{c}_mean": entry.get(c) for c in METRIC_COLS},
                **{f"{c}_std": 0.0 for c in METRIC_COLS},
                "threshold": entry.get("threshold"),
                "source": str(run_dir.relative_to(PROJECT_ROOT)),
            }
    return None


def _vae_row(aggregated_csv: Path, base_experiment_id: str, threshold_method: str, split: str) -> dict | None:
    df = pd.read_csv(aggregated_csv)
    df = df[(df["base_experiment_id"] == base_experiment_id) & (df["threshold_method"] == threshold_method) & (df["split"] == split)]
    if df.empty:
        return None
    row = df.iloc[0]
    out = {
        "model": f"VAE ({row['architecture']}, hidden={row['hidden']}, window={int(row['window_size'])})",
        "model_kind": f"vae_{row['architecture']}",
        "run_id": base_experiment_id,
        "threshold_method": threshold_method,
        "split": split,
        "n_seeds": int(row["n_seeds"]),
    }
    for c in METRIC_COLS:
        out[f"{c}_mean"] = row.get(f"{c}_mean")
        out[f"{c}_std"] = row.get(f"{c}_std")
    out["threshold"] = row.get("threshold_value_mean")
    out["source"] = str(aggregated_csv.relative_to(PROJECT_ROOT))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Assemble canonical model comparison.")
    parser.add_argument("--aggregated-csv", default="reports/tables/grid_aggregated.csv")
    parser.add_argument("--vae-base-id", default="win120", help="Base experiment id of the headline VAE in the grid.")
    parser.add_argument("--baselines-root", default="models/baseline_runs")
    parser.add_argument(
        "--baseline-kinds",
        nargs="*",
        default=["isolation_forest", "pca_recon", "oc_svm", "lof"],
    )
    parser.add_argument("--output-dir", default="reports/tables/baselines")
    args = parser.parse_args()

    aggregated_csv = _resolve(args.aggregated_csv)
    out_dir = _resolve(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for threshold_method in ["train_p98", "val_f1"]:
        rows: list[dict] = []
        vae_row = _vae_row(aggregated_csv, args.vae_base_id, threshold_method, "test")
        if vae_row is not None:
            rows.append(vae_row)
        for kind in args.baseline_kinds:
            kind_dir = _resolve(args.baselines_root) / kind
            run = _latest_subdir(kind_dir)
            if run is None:
                print(f"WARN: no run found for {kind} under {kind_dir}; skipping.")
                continue
            row = _baseline_row(run, threshold_method, "test")
            if row is not None:
                rows.append(row)
        if not rows:
            continue
        df = pd.DataFrame(rows)
        out_path = out_dir / f"baseline_comparison_primary_{threshold_method}.csv"
        df.to_csv(out_path, index=False)
        print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
