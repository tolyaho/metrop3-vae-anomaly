"""
Publication-ready analysis plots for the DDA4210 MetroPT3 VAE anomaly detection project.

Generates three figures:
  fig1_score_distributions.{pdf,png}  – Reconstruction score distributions + thresholds
  fig2_baseline_comparison.{pdf,png}  – VAE vs Logistic Regression vs XGBoost
  fig3_timeseries_simulation.{pdf,png} – Detection timeline with FN markers

Run:
    python scripts/plot_analysis.py [--run-dir models/vae_runs/<id>]

Defaults to the primary best run: models/vae_runs/20260424_140621
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
from scipy.stats import gaussian_kde
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT / "src"))
from metrics import binary_metrics, roc_auc_binary

# ── Constants ──────────────────────────────────────────────────────────────────
WINDOW_SIZE = 60
STRIDE      = 10
N_FEATURES  = 15

COLORS = {
    "normal":    "#2166AC",
    "anomaly":   "#D6604D",
    "overlap":   "#A0A0A0",
    "vae":       "#1B7837",
    "lr":        "#762A83",
    "xgb":       "#E07B00",
    "threshold_valf1": "#1A1A1A",
    "threshold_p98":   "#888888",
    "fn_marker": "#8B0000",
    "event_fill": "#FFCDD2",
    "score_line": "#3A6FA6",
}

# ── Matplotlib style ───────────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.dpi":       150,
    "savefig.dpi":      300,
    "font.family":      "DejaVu Sans",
    "font.size":        11,
    "axes.titlesize":   13,
    "axes.titleweight": "bold",
    "axes.labelsize":   11,
    "legend.fontsize":  9.5,
    "xtick.labelsize":  9.5,
    "ytick.labelsize":  9.5,
    "axes.spines.top":  False,
    "axes.spines.right": False,
    "axes.grid":        True,
    "grid.alpha":       0.30,
    "grid.linestyle":   "--",
    "axes.axisbelow":   True,
})


# ══════════════════════════════════════════════════════════════════════════════
# Data loading helpers
# ══════════════════════════════════════════════════════════════════════════════

def load_run(run_dir: Path) -> dict:
    summary = json.loads((run_dir / "summary.json").read_text())
    return {
        "summary":      summary,
        "threshold":    float(summary["threshold"]),
        "train_scores": np.load(run_dir / "train_scores.npy").astype(np.float32),
        "val_scores":   np.load(run_dir / "val_scores.npy").astype(np.float32),
        "test_scores":  np.load(run_dir / "test_scores.npy").astype(np.float32),
        "train_labels": np.load(run_dir / "train_labels.npy").astype(np.int32),
        "val_labels":   np.load(run_dir / "val_labels.npy").astype(np.int32),
        "test_labels":  np.load(run_dir / "test_labels.npy").astype(np.int32),
        "test_preds":   np.load(run_dir / "test_predictions.npy").astype(np.int32),
    }


def window_stats(windows: np.ndarray) -> np.ndarray:
    """Per-feature mean and std across the time axis → (N, 2*F)."""
    return np.concatenate([windows.mean(axis=1), windows.std(axis=1)], axis=1)


# ══════════════════════════════════════════════════════════════════════════════
# Figure 1 — Score distributions + thresholds
# ══════════════════════════════════════════════════════════════════════════════

def plot_score_distributions(data: dict, fig_dir: Path) -> None:
    train_scores = data["train_scores"]
    val_scores   = data["val_scores"]
    test_scores  = data["test_scores"]
    val_labels   = data["val_labels"]
    test_labels  = data["test_labels"]
    train_labels = data["train_labels"]
    threshold    = data["threshold"]
    train_p98    = float(np.percentile(train_scores, 98))

    normal_scores = np.concatenate([
        train_scores[train_labels == 0],
        val_scores[val_labels == 0],
        test_scores[test_labels == 0],
    ])
    anomaly_scores = np.concatenate([
        val_scores[val_labels == 1],
        test_scores[test_labels == 1],
    ])

    # Display range: clip extreme normal tail while keeping all anomaly visible
    x_min = float(normal_scores.min()) * 0.97
    x_max = max(float(np.percentile(normal_scores, 99.8)),
                float(anomaly_scores.max())) * 1.02
    xs = np.linspace(x_min, x_max, 800)

    kde_normal  = gaussian_kde(normal_scores,  bw_method="scott")(xs)
    kde_anomaly = gaussian_kde(anomaly_scores, bw_method="scott")(xs)
    overlap     = np.minimum(kde_normal, kde_anomaly)

    fig, ax = plt.subplots(figsize=(9, 4.8))

    # Filled distributions
    ax.fill_between(xs, kde_normal,  alpha=0.20, color=COLORS["normal"])
    ax.fill_between(xs, kde_anomaly, alpha=0.20, color=COLORS["anomaly"])
    ax.fill_between(xs, overlap,     alpha=0.35, color=COLORS["overlap"],
                    label="Overlap region", hatch="////", linewidth=0.0)

    # Outline curves
    ax.plot(xs, kde_normal,  lw=2.0, color=COLORS["normal"],
            label=f"Normal  (n={len(normal_scores):,})")
    ax.plot(xs, kde_anomaly, lw=2.0, color=COLORS["anomaly"],
            label=f"Anomaly (n={len(anomaly_scores):,})")

    # Threshold lines
    ax.axvline(threshold,  ls="--", lw=2.0, color=COLORS["threshold_valf1"],
               label=f"Val-F1 threshold  = {threshold:,.0f}")
    ax.axvline(train_p98, ls=":",  lw=1.8, color=COLORS["threshold_p98"],
               label=f"Train P98 threshold = {train_p98:,.0f}")

    # Annotation: highlight threshold region
    y_top = max(kde_normal.max(), kde_anomaly.max())
    ax.annotate(
        "Classification\nboundary",
        xy=(threshold, y_top * 0.55),
        xytext=(threshold + (x_max - x_min) * 0.07, y_top * 0.75),
        arrowprops=dict(arrowstyle="->", color="#333333", lw=1.2),
        fontsize=9, color="#333333",
    )

    ax.set_xlim(x_min, x_max)
    ax.set_xlabel("Reconstruction MSE Score")
    ax.set_ylabel("Density")
    ax.set_title("Reconstruction Score Distributions: Normal vs. Anomaly Windows")
    ax.legend(loc="upper right", framealpha=0.92, edgecolor="#CCCCCC")

    fig.tight_layout()
    _save(fig, fig_dir / "fig1_score_distributions")
    print(f"  train_p98 = {train_p98:,.1f}   val_f1 threshold = {threshold:,.1f}")


# ══════════════════════════════════════════════════════════════════════════════
# Figure 2 — Baseline comparison
# ══════════════════════════════════════════════════════════════════════════════

def plot_baseline_comparison(data: dict, windows_dir: Path, fig_dir: Path) -> None:
    val_labels  = data["val_labels"]
    test_labels = data["test_labels"]
    test_scores = data["test_scores"]
    test_preds  = data["test_preds"]

    n_val  = len(val_labels)
    n_test = len(test_labels)

    val_win  = np.load(windows_dir / "val_windows.npy").reshape(n_val,  WINDOW_SIZE, N_FEATURES)
    test_win = np.load(windows_dir / "test_windows.npy").reshape(n_test, WINDOW_SIZE, N_FEATURES)

    X_val  = window_stats(val_win)
    y_val  = val_labels
    X_test = window_stats(test_win)
    y_test = test_labels

    neg_pos = float((y_val == 0).sum()) / max(1, int((y_val == 1).sum()))
    print(f"  Baseline training set: {n_val:,} windows "
          f"({int((y_val==1).sum())} anomaly, {int((y_val==0).sum())} normal) — neg:pos ≈ {neg_pos:.0f}:1")

    # Logistic Regression
    lr_pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            class_weight="balanced", max_iter=2000,
            C=0.5, random_state=42,
        )),
    ])
    lr_pipe.fit(X_val, y_val)
    lr_preds = lr_pipe.predict(X_test)
    lr_proba = lr_pipe.predict_proba(X_test)[:, 1]

    # XGBoost
    xgb_clf = XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        scale_pos_weight=neg_pos,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        verbosity=0,
    )
    xgb_clf.fit(X_val, y_val)
    xgb_preds = xgb_clf.predict(X_test)
    xgb_proba = xgb_clf.predict_proba(X_test)[:, 1]

    # Aggregate metrics
    vae_m = binary_metrics(test_labels, test_preds)
    vae_m["roc_auc"] = roc_auc_binary(test_labels, test_scores)

    lr_m = binary_metrics(y_test, lr_preds)
    lr_m["roc_auc"] = roc_auc_binary(y_test, lr_proba)

    xgb_m = binary_metrics(y_test, xgb_preds)
    xgb_m["roc_auc"] = roc_auc_binary(y_test, xgb_proba)

    print("\n  ── Baseline Results (test set) ───────────────────────────────────")
    for name, m in [("VAE (ours, unsup.)", vae_m),
                    ("Logistic Regression", lr_m),
                    ("XGBoost", xgb_m)]:
        print(f"  {name:24s}  P={m['precision']:.3f}  R={m['recall']:.3f}"
              f"  F1={m['f1']:.3f}  AUC={m['roc_auc']:.3f}")

    metric_keys   = ["f1", "precision", "recall", "roc_auc"]
    metric_labels = ["F1-Score", "Precision", "Recall", "ROC-AUC"]
    models = [
        ("VAE (unsupervised)", vae_m, COLORS["vae"]),
        ("Logistic Regression", lr_m,  COLORS["lr"]),
        ("XGBoost",            xgb_m, COLORS["xgb"]),
    ]

    x     = np.arange(len(metric_keys))
    w     = 0.23
    fig, ax = plt.subplots(figsize=(9, 5.0))

    for i, (name, m, color) in enumerate(models):
        vals   = [m[k] for k in metric_keys]
        offset = (i - 1) * w
        bars   = ax.bar(x + offset, vals, w,
                        label=name, color=color,
                        alpha=0.88, edgecolor="white", linewidth=0.6)
        for bar, val in zip(bars, vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.013,
                f"{val:.3f}",
                ha="center", va="bottom", fontsize=8.0, color="#222222",
            )

    ax.set_xticks(x)
    ax.set_xticklabels(metric_labels, fontsize=10.5)
    ax.set_ylim(0, 1.16)
    ax.set_ylabel("Score")
    ax.set_title(
        "VAE vs. Supervised Baselines — Test Set Performance\n"
        "Baselines trained on labelled validation set; VAE trained on unlabelled normal data only"
    )
    ax.legend(loc="lower right", framealpha=0.92, edgecolor="#CCCCCC")

    # Subtle note about training setup
    ax.text(
        0.01, 0.97,
        "★ VAE requires no anomaly labels during training",
        transform=ax.transAxes,
        fontsize=8.5, color=COLORS["vae"], va="top",
        style="italic",
    )

    fig.tight_layout()
    _save(fig, fig_dir / "fig2_baseline_comparison")


# ══════════════════════════════════════════════════════════════════════════════
# Figure 3 — Time-series simulation
# ══════════════════════════════════════════════════════════════════════════════

def plot_timeseries_simulation(data: dict, split_dir: Path, fig_dir: Path) -> None:
    test_scores = data["test_scores"]
    test_labels = data["test_labels"]
    test_preds  = data["test_preds"]
    threshold   = data["threshold"]
    train_p98   = float(np.percentile(data["train_scores"], 98))

    # ── Recover per-window timestamps ─────────────────────────────────────────
    test_df = pd.read_csv(
        split_dir / "test_split.csv",
        low_memory=False,
        parse_dates=["timestamp"],
    )
    ts_arr = test_df["timestamp"].values
    n_win  = len(test_scores)

    last_idxs  = [i * STRIDE + WINDOW_SIZE - 1 for i in range(n_win)]
    timestamps = pd.to_datetime([ts_arr[idx] for idx in last_idxs])

    # ── Identify contiguous anomaly event spans (gap > 50 windows → new event) ─
    anom_idx = np.where(test_labels == 1)[0]
    events: list[tuple] = []
    if len(anom_idx):
        seg_start = anom_idx[0]
        prev      = anom_idx[0]
        for idx in anom_idx[1:]:
            if idx - prev > 50:
                events.append((timestamps[seg_start], timestamps[prev]))
                seg_start = idx
            prev = idx
        events.append((timestamps[seg_start], timestamps[prev]))

    # Assign failure IDs based on temporal order
    event_names   = [f"F{3+j}" for j in range(len(events))]
    event_shades  = ["#FFCDD2", "#FFCCBB"]

    # ── Smooth scores for the main trend line ─────────────────────────────────
    scores_series = pd.Series(test_scores.astype(float), index=timestamps)
    smoothed      = scores_series.rolling("4h", min_periods=1).mean()

    # False negatives: anomaly windows predicted as normal
    fn_mask = (test_labels == 1) & (test_preds == 0)
    fn_ts   = timestamps[fn_mask]
    fn_sc   = test_scores[fn_mask]

    print(f"\n  Test period: {timestamps.min().date()} → {timestamps.max().date()}")
    for name, (t0, t1) in zip(event_names, events):
        n_anom = int(np.sum((timestamps >= t0) & (timestamps <= t1) & (test_labels == 1)))
        print(f"  {name}: {t0.date()} → {t1.date()}  ({n_anom} anomalous windows)")
    print(f"  False negatives: {fn_mask.sum():,} windows  "
          f"({fn_mask.sum() / max(1, int(test_labels.sum())) * 100:.1f}% of true anomalies)")

    y_min = float(test_scores.min()) * 0.95
    y_max = float(test_scores.max()) * 1.05

    # ── Layout: main panel (score over time) + binary signal strip ────────────
    fig, axes = plt.subplots(
        2, 1, figsize=(14, 8),
        gridspec_kw={"height_ratios": [3.5, 1.0], "hspace": 0.06},
        sharex=True,
        layout="constrained",
    )
    ax_main, ax_sig = axes

    # Event shading (main panel)
    for j, ((t0, t1), name) in enumerate(zip(events, event_names)):
        shade = event_shades[j % len(event_shades)]
        ax_main.axvspan(t0, t1, alpha=0.50, color=shade, zorder=0, linewidth=0)
        # Event label just below the top spine (axes-transform y)
        ax_main.text(
            t0 + (t1 - t0) / 2, 0.96, name,
            transform=ax_main.get_xaxis_transform(),
            ha="center", va="top", fontsize=11, fontweight="bold",
            color="#B71C1C", alpha=0.95,
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="#B71C1C",
                      lw=0.8, alpha=0.85),
        )

    # Raw score (thin, transparent) + smoothed trend
    ax_main.plot(timestamps, test_scores, lw=0.4, color=COLORS["score_line"],
                 alpha=0.25, zorder=1)
    ax_main.plot(smoothed.index, smoothed.values, lw=1.4,
                 color=COLORS["score_line"], alpha=0.85, zorder=2,
                 label="Reconstruction score (4 h rolling mean)")

    # Threshold lines
    ax_main.axhline(threshold, ls="--", lw=2.0, color=COLORS["threshold_valf1"],
                    zorder=3, label=f"Val-F1 threshold ({threshold:,.0f})")
    ax_main.axhline(train_p98, ls=":",  lw=1.8, color=COLORS["threshold_p98"],
                    zorder=3, label=f"Train P98 threshold ({train_p98:,.0f})")

    # False-negative markers
    if fn_mask.sum() > 0:
        ax_main.scatter(
            fn_ts, fn_sc,
            marker="x", s=28, linewidths=1.4,
            color=COLORS["fn_marker"], zorder=6,
            label=f"False negatives ({fn_mask.sum():,} windows)",
            alpha=0.80,
        )

    # Legend patch for true anomaly events
    from matplotlib.patches import Patch
    event_patch = Patch(facecolor=event_shades[0], alpha=0.6, label="True anomaly event (ground truth)")
    handles, labels_leg = ax_main.get_legend_handles_labels()
    ax_main.legend(
        handles=[event_patch] + handles,
        labels=["True anomaly event (ground truth)"] + labels_leg,
        loc="upper left", framealpha=0.92, edgecolor="#CCCCCC", fontsize=9,
    )

    ax_main.set_ylim(y_min, y_max)
    ax_main.set_ylabel("Reconstruction MSE Score")
    ax_main.set_title(
        "Anomaly Detection over Test Period — MetroPT3 Compressor (Jun–Jul 2020)",
        pad=10,
    )

    # ── Binary signal strip ───────────────────────────────────────────────────
    for j, ((t0, t1), name) in enumerate(zip(events, event_names)):
        shade = event_shades[j % len(event_shades)]
        ax_sig.axvspan(t0, t1, alpha=0.50, color=shade, zorder=0, linewidth=0)

    ax_sig.fill_between(
        timestamps, test_labels.astype(float), step="post",
        alpha=0.55, color=COLORS["anomaly"], zorder=1, label="Ground truth (anomaly=1)",
    )
    ax_sig.step(
        timestamps, test_preds.astype(float), where="post",
        lw=1.3, color=COLORS["vae"], alpha=0.90, zorder=2, label="VAE prediction",
    )
    ax_sig.set_ylim(-0.05, 1.55)
    ax_sig.set_yticks([0, 1])
    ax_sig.set_yticklabels(["Normal", "Anomaly"], fontsize=9)
    ax_sig.set_ylabel("Label", labelpad=4)
    ax_sig.set_xlabel("Date (2020)", labelpad=6)
    ax_sig.legend(loc="upper right", framealpha=0.92, edgecolor="#CCCCCC", fontsize=9)

    # Format x-axis dates
    import matplotlib.dates as mdates
    ax_sig.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    ax_sig.xaxis.set_major_locator(mdates.WeekdayLocator(interval=1))
    plt.setp(ax_sig.xaxis.get_majorticklabels(), rotation=30, ha="right")

    _save(fig, fig_dir / "fig3_timeseries_simulation")


# ══════════════════════════════════════════════════════════════════════════════
# Utilities
# ══════════════════════════════════════════════════════════════════════════════

def _save(fig: plt.Figure, stem: Path) -> None:
    for ext in ("pdf", "png"):
        p = stem.with_suffix(f".{ext}")
        fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {stem}.{{pdf,png}}")


# ══════════════════════════════════════════════════════════════════════════════
# Entry point
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(description="Generate analysis figures for a VAE run.")
    parser.add_argument(
        "--run-dir",
        default="models/vae_runs/20260424_140621",
        help="Path to the saved VAE run directory (relative to project root).",
    )
    parser.add_argument(
        "--split-dir",
        default="dataset/preprocessed/splits/20260424_005227",
        help="Path to the preprocessed split directory (contains *_split.csv files).",
    )
    parser.add_argument(
        "--windows-dir",
        default="dataset/processed_windows/20260424_135957",
        help="Path to the processed windows directory (contains *.npy files).",
    )
    parser.add_argument(
        "--output-dir",
        default="reports/figures",
        help="Directory to save generated figures.",
    )
    args = parser.parse_args()

    def _resolve(p: str) -> Path:
        path = Path(p)
        return path if path.is_absolute() else PROJECT_ROOT / path

    run_dir     = _resolve(args.run_dir)
    split_dir   = _resolve(args.split_dir)
    windows_dir = _resolve(args.windows_dir)
    fig_dir     = _resolve(args.output_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)

    print(f"Run directory : {run_dir}")
    print(f"Split directory: {split_dir}")
    print(f"Windows dir   : {windows_dir}")
    print(f"Output        : {fig_dir}\n")

    data = load_run(run_dir)
    th = data["threshold"]
    tr_p98 = float(np.percentile(data["train_scores"], 98))
    vae_m = binary_metrics(data["test_labels"], data["test_preds"])
    vae_m["roc_auc"] = roc_auc_binary(data["test_labels"], data["test_scores"])
    print(f"VAE (val_f1 threshold={th:,.0f}): "
          f"F1={vae_m['f1']:.4f}  P={vae_m['precision']:.4f}  "
          f"R={vae_m['recall']:.4f}  AUC={vae_m['roc_auc']:.4f}")
    print(f"Train P98 threshold: {tr_p98:,.0f}\n")

    print("── Figure 1: Score Distributions ──────────────────────────────────")
    plot_score_distributions(data, fig_dir)

    print("\n── Figure 2: Baseline Comparison ──────────────────────────────────")
    plot_baseline_comparison(data, windows_dir, fig_dir)

    print("\n── Figure 3: Time-Series Simulation ───────────────────────────────")
    plot_timeseries_simulation(data, split_dir, fig_dir)

    print(f"\nDone. All figures saved to: {fig_dir}")


if __name__ == "__main__":
    main()
