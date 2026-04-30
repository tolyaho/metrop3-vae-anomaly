"""Overlay test-set ROC and Precision-Recall curves for all models.

Reads scores from:

  * VAE run (a single ``models/vae_runs/<run_id>/`` produced by ``train_vae.py``),
  * each classical baseline run (``models/baseline_runs/<model>/<run_id>/scores.npz``).

Produces a side-by-side ROC + PR figure (paper-ready) and a per-model
metrics table. Defaults pick the most recent run for each model so the
script can be invoked with no arguments after a full retraining sweep.

Optionally include a multi-seed VAE *grid* row by passing
``--vae-grid-dir <study_dir> --vae-grid-base-id <base_experiment_id>``;
the script then plots the seed-average ROC/PR curve with a shaded band
for the seed-wise standard deviation.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve, roc_curve

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.metrics import pr_auc_binary, roc_auc_binary  # noqa: E402
from src.plotting import MODEL_COLORS, PALETTE, apply_paper_style, save_figure, style_axes  # noqa: E402


BASELINE_DIRS = {
    "Isolation Forest": "models/baseline_runs/isolation_forest",
    "PCA reconstruction": "models/baseline_runs/pca_recon",
    "OC-SVM": "models/baseline_runs/oc_svm",
    "LOF": "models/baseline_runs/lof",
}


def _resolve(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _latest_subdir(p: Path) -> Path | None:
    if not p.exists():
        return None
    children = sorted(c for c in p.iterdir() if c.is_dir())
    return children[-1] if children else None


def _load_vae_scores(run_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    return (
        np.load(run_dir / "test_scores.npy").astype(np.float64),
        np.load(run_dir / "test_labels.npy").astype(np.int32),
    )


def _load_baseline_scores(run_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    data = np.load(run_dir / "scores.npz")
    return data["test_scores"].astype(np.float64), data["test_labels"].astype(np.int32)


def _interp_curve(x_target: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    order = np.argsort(x)
    return np.interp(x_target, x[order], y[order])


def _add_vae_grid(
    rows: list[dict],
    fpr_grid: np.ndarray,
    rec_grid: np.ndarray,
    label_to_curves_roc: dict,
    label_to_curves_pr: dict,
    label_to_metrics: dict,
    *,
    grid_dir: Path,
    base_id: str,
) -> None:
    runs_root = grid_dir / "runs"
    seed_runs: list[Path] = []
    for parent in runs_root.glob(f"*/{base_id}_s*"):
        for child in parent.iterdir():
            if child.is_dir() and (child / "test_scores.npy").exists():
                seed_runs.append(child)
                break
    if not seed_runs:
        candidate = runs_root / f"layers/{base_id}"
        if candidate.exists():
            for child in candidate.iterdir():
                if (child / "test_scores.npy").exists():
                    seed_runs.append(child)
                    break
    if not seed_runs:
        print(f"WARN: no VAE grid runs found for base_id={base_id}; skipping seed-aggregated VAE.")
        return

    fprs, tprs, recalls, precisions, aucs, aps = [], [], [], [], [], []
    for run in seed_runs:
        scores, labels = _load_vae_scores(run)
        if np.unique(labels).size < 2:
            continue
        fpr, tpr, _ = roc_curve(labels, scores)
        prec, rec, _ = precision_recall_curve(labels, scores)
        fprs.append(_interp_curve(fpr_grid, fpr, tpr))
        recalls.append(_interp_curve(rec_grid, rec[::-1], prec[::-1]))
        aucs.append(roc_auc_binary(labels, scores))
        aps.append(pr_auc_binary(labels, scores))

    if not fprs:
        return

    tpr_arr = np.vstack(fprs)
    pr_arr = np.vstack(recalls)
    roc_mean = tpr_arr.mean(axis=0)
    roc_std = tpr_arr.std(axis=0)
    pr_mean = pr_arr.mean(axis=0)
    pr_std = pr_arr.std(axis=0)
    auc_mean = float(np.mean(aucs))
    auc_std = float(np.std(aucs))
    ap_mean = float(np.mean(aps))
    ap_std = float(np.std(aps))

    label = f"VAE (dense, {len(seed_runs)} seeds)"
    label_to_curves_roc[label] = (roc_mean, roc_std)
    label_to_curves_pr[label] = (pr_mean, pr_std)
    label_to_metrics[label] = {
        "auc_mean": auc_mean,
        "auc_std": auc_std,
        "ap_mean": ap_mean,
        "ap_std": ap_std,
        "n_seeds": len(seed_runs),
    }
    rows.append(
        {
            "model": label,
            "roc_auc": auc_mean,
            "roc_auc_std": auc_std,
            "pr_auc": ap_mean,
            "pr_auc_std": ap_std,
            "n_seeds": len(seed_runs),
            "source": str(grid_dir.relative_to(PROJECT_ROOT)),
        }
    )


def _add_single(
    rows: list[dict],
    fpr_grid: np.ndarray,
    rec_grid: np.ndarray,
    label_to_curves_roc: dict,
    label_to_curves_pr: dict,
    label_to_metrics: dict,
    *,
    label: str,
    scores: np.ndarray,
    labels: np.ndarray,
    source: str,
) -> None:
    if np.unique(labels).size < 2:
        return
    fpr, tpr, _ = roc_curve(labels, scores)
    prec, rec, _ = precision_recall_curve(labels, scores)
    auc = roc_auc_binary(labels, scores)
    ap = pr_auc_binary(labels, scores)
    roc_mean = _interp_curve(fpr_grid, fpr, tpr)
    pr_mean = _interp_curve(rec_grid, rec[::-1], prec[::-1])
    label_to_curves_roc[label] = (roc_mean, np.zeros_like(roc_mean))
    label_to_curves_pr[label] = (pr_mean, np.zeros_like(pr_mean))
    label_to_metrics[label] = {
        "auc_mean": auc,
        "auc_std": 0.0,
        "ap_mean": ap,
        "ap_std": 0.0,
        "n_seeds": 1,
    }
    rows.append(
        {
            "model": label,
            "roc_auc": auc,
            "roc_auc_std": 0.0,
            "pr_auc": ap,
            "pr_auc_std": 0.0,
            "n_seeds": 1,
            "source": source,
        }
    )


def _plot(
    label_to_curves_roc: dict,
    label_to_curves_pr: dict,
    label_to_metrics: dict,
    fpr_grid: np.ndarray,
    rec_grid: np.ndarray,
    *,
    prevalence: float,
    out_paths: list[Path],
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13.2, 5.4))
    ax_roc, ax_pr = axes

    for label, (mean, std) in label_to_curves_roc.items():
        color = MODEL_COLORS.get(label.split(" (")[0], MODEL_COLORS.get(label, PALETTE["primary"]))
        m = label_to_metrics[label]
        legend = f"{label} (AUC={m['auc_mean']:.3f}"
        legend += f"+/-{m['auc_std']:.3f}" if m["auc_std"] > 0 else ""
        legend += ")"
        ax_roc.plot(fpr_grid, mean, color=color, linewidth=2.0, label=legend)
        if np.any(std > 0):
            ax_roc.fill_between(fpr_grid, mean - std, mean + std, color=color, alpha=0.18, linewidth=0)
    ax_roc.plot([0, 1], [0, 1], color=PALETTE["subtle"], linestyle="--", linewidth=1)
    ax_roc.set_title("Test ROC")
    ax_roc.set_xlabel("False positive rate")
    ax_roc.set_ylabel("True positive rate")
    ax_roc.set_xlim(0, 1)
    ax_roc.set_ylim(0, 1.02)
    style_axes(ax_roc, grid_axis="both")
    ax_roc.legend(frameon=False, loc="lower right", fontsize=9)

    for label, (mean, std) in label_to_curves_pr.items():
        color = MODEL_COLORS.get(label.split(" (")[0], MODEL_COLORS.get(label, PALETTE["primary"]))
        m = label_to_metrics[label]
        legend = f"{label} (AP={m['ap_mean']:.3f}"
        legend += f"+/-{m['ap_std']:.3f}" if m["ap_std"] > 0 else ""
        legend += ")"
        ax_pr.plot(rec_grid, mean, color=color, linewidth=2.0, label=legend)
        if np.any(std > 0):
            ax_pr.fill_between(rec_grid, mean - std, mean + std, color=color, alpha=0.18, linewidth=0)
    ax_pr.axhline(prevalence, color=PALETTE["subtle"], linestyle="--", linewidth=1, label=f"prevalence={prevalence:.3f}")
    ax_pr.set_title("Test precision-recall")
    ax_pr.set_xlabel("Recall")
    ax_pr.set_ylabel("Precision")
    ax_pr.set_xlim(0, 1)
    ax_pr.set_ylim(0, 1.02)
    style_axes(ax_pr, grid_axis="both")
    ax_pr.legend(frameon=False, loc="lower left", fontsize=9)

    fig.suptitle("Test-set ROC and PR overlay across models", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    save_figure(fig, out_paths)


def main() -> None:
    parser = argparse.ArgumentParser(description="Overlay test-set ROC and PR curves across all models.")
    parser.add_argument("--vae-run-dir", default=None, help="Path to VAE run directory (defaults to most recent).")
    parser.add_argument("--vae-grid-dir", default=None, help="Path to a multi-seed grid study dir.")
    parser.add_argument("--vae-grid-base-id", default="layers_128_64", help="Base experiment id within the grid to aggregate.")
    parser.add_argument(
        "--baseline-run-dir",
        nargs="*",
        default=None,
        help="Optional list of <model_kind>:<run_dir> entries; defaults to most recent run per model.",
    )
    parser.add_argument(
        "--output",
        default="reports/figures/models_overlay_test.png",
        help="Output figure path (PNG).",
    )
    parser.add_argument(
        "--output-table",
        default="reports/tables/models_overlay_test.csv",
        help="Output CSV summary path.",
    )
    args = parser.parse_args()

    apply_paper_style()

    fpr_grid = np.linspace(0, 1, 401)
    rec_grid = np.linspace(0, 1, 401)
    rows: list[dict] = []
    label_to_curves_roc: dict = {}
    label_to_curves_pr: dict = {}
    label_to_metrics: dict = {}

    if args.vae_grid_dir:
        grid_dir = _resolve(args.vae_grid_dir)
        _add_vae_grid(
            rows,
            fpr_grid,
            rec_grid,
            label_to_curves_roc,
            label_to_curves_pr,
            label_to_metrics,
            grid_dir=grid_dir,
            base_id=args.vae_grid_base_id,
        )
    else:
        vae_run = _resolve(args.vae_run_dir) if args.vae_run_dir else _latest_subdir(_resolve("models/vae_runs"))
        if vae_run is not None and (vae_run / "test_scores.npy").exists():
            scores, labels = _load_vae_scores(vae_run)
            _add_single(
                rows,
                fpr_grid,
                rec_grid,
                label_to_curves_roc,
                label_to_curves_pr,
                label_to_metrics,
                label="VAE (dense)",
                scores=scores,
                labels=labels,
                source=str(vae_run.relative_to(PROJECT_ROOT)),
            )

    explicit_baseline_runs: dict[str, Path] = {}
    if args.baseline_run_dir:
        for entry in args.baseline_run_dir:
            kind, _, path = entry.partition(":")
            explicit_baseline_runs[kind.strip()] = _resolve(path.strip())

    for model_label, default_path in BASELINE_DIRS.items():
        kind = model_label.lower().replace(" ", "_").replace("-", "_")
        if kind in explicit_baseline_runs:
            run = explicit_baseline_runs[kind]
        else:
            kind_alias = {
                "isolation_forest": "isolation_forest",
                "pca_reconstruction": "pca_recon",
                "oc_svm": "oc_svm",
                "lof": "lof",
            }[kind]
            run = _latest_subdir(_resolve(BASELINE_DIRS[model_label].replace(BASELINE_DIRS[model_label].split("/")[-1], kind_alias)))
        if run is None or not (run / "scores.npz").exists():
            print(f"WARN: no run for {model_label} under {default_path}; skipping.")
            continue
        scores, labels = _load_baseline_scores(run)
        _add_single(
            rows,
            fpr_grid,
            rec_grid,
            label_to_curves_roc,
            label_to_curves_pr,
            label_to_metrics,
            label=model_label,
            scores=scores,
            labels=labels,
            source=str(run.relative_to(PROJECT_ROOT)),
        )

    if not rows:
        raise SystemExit("No models found to overlay; check the run paths.")

    test_labels_for_prev: np.ndarray | None = None
    for label in label_to_curves_roc:
        if "VAE" in label and args.vae_grid_dir is None:
            vae_run = _resolve(args.vae_run_dir) if args.vae_run_dir else _latest_subdir(_resolve("models/vae_runs"))
            if vae_run is not None and (vae_run / "test_labels.npy").exists():
                test_labels_for_prev = np.load(vae_run / "test_labels.npy")
            break
    if test_labels_for_prev is None:
        for kind in ("isolation_forest", "pca_recon", "oc_svm", "lof"):
            run = _latest_subdir(_resolve(f"models/baseline_runs/{kind}"))
            if run and (run / "scores.npz").exists():
                test_labels_for_prev = np.load(run / "scores.npz")["test_labels"]
                break
    prevalence = float(np.mean(test_labels_for_prev == 1)) if test_labels_for_prev is not None else 0.05

    out = _resolve(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    _plot(
        label_to_curves_roc,
        label_to_curves_pr,
        label_to_metrics,
        fpr_grid,
        rec_grid,
        prevalence=prevalence,
        out_paths=[out, out.with_suffix(".pdf")],
    )

    table = pd.DataFrame(rows).sort_values("roc_auc", ascending=False)
    out_table = _resolve(args.output_table)
    out_table.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(out_table, index=False)
    print(table.to_string(index=False))
    print(f"Saved figure: {out}")
    print(f"Saved figure: {out.with_suffix('.pdf')}")
    print(f"Saved table: {out_table}")


if __name__ == "__main__":
    main()
