from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


COLORS = {
    "normal": "#2F6F73",
    "anomaly": "#D64545",
    "train": "#B7C1C8",
    "background": "#F7F8FA",
    "ink": "#1F2933",
    "muted": "#697586",
    "grid": "#D8DEE6",
}


def _resolve(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _default_run_dir() -> Path:
    """Pick the best standalone saved VAE run by test F1."""
    runs = sorted(
        p for p in (PROJECT_ROOT / "models" / "vae_runs").iterdir()
        if p.is_dir() and (p / "encoder.keras").exists()
    )
    if not runs:
        raise FileNotFoundError("No saved VAE run with encoder.keras was found.")

    ranked = []
    for run_dir in runs:
        summary_path = run_dir / "summary.json"
        if not summary_path.exists():
            continue
        summary = _load_json(summary_path)
        metrics = summary.get("metrics", {})
        ranked.append((float(metrics.get("f1", -1.0)), float(metrics.get("roc_auc", -1.0)), run_dir))

    return max(ranked, key=lambda row: (row[0], row[1]))[2] if ranked else runs[-1]


def _stratified_indices(labels: np.ndarray, max_points: int, seed: int) -> np.ndarray:
    labels = labels.astype(np.int32)
    n = len(labels)
    if n <= max_points:
        return np.arange(n)

    rng = np.random.default_rng(seed)
    anomaly_idx = np.flatnonzero(labels == 1)
    normal_idx = np.flatnonzero(labels == 0)

    anomaly_take = min(len(anomaly_idx), max(1, int(max_points * 0.35)))
    normal_take = max_points - anomaly_take
    if anomaly_take < len(anomaly_idx):
        anomaly_idx = rng.choice(anomaly_idx, size=anomaly_take, replace=False)
    if normal_take < len(normal_idx):
        normal_idx = rng.choice(normal_idx, size=normal_take, replace=False)

    idx = np.concatenate([normal_idx, anomaly_idx])
    rng.shuffle(idx)
    return np.sort(idx)


def _extract_z_mean(encoder, windows: np.ndarray, batch_size: int) -> np.ndarray:
    import tensorflow as tf

    ds = tf.data.Dataset.from_tensor_slices(windows).batch(batch_size)
    latent = []
    for batch in ds:
        z_mean, _, _ = encoder(batch, training=False)
        latent.append(z_mean.numpy())
    return np.concatenate(latent, axis=0) if latent else np.empty((0, 0), dtype=np.float32)


def _load_split_windows(source_run_dir: Path, split: str):
    from src.models.vae_anomaly_detector import load_window_run

    train_windows, val_windows, test_windows, train_labels, val_labels, test_labels, metadata = load_window_run(source_run_dir)
    split_map = {
        "train": (train_windows, train_labels),
        "val": (val_windows, val_labels),
        "test": (test_windows, test_labels),
    }
    windows, labels = split_map[split]
    if len(windows) == 0:
        raise ValueError(f"Split '{split}' has no windows in {source_run_dir}.")
    return windows, labels, metadata


def _axis_limits(points: np.ndarray) -> tuple[tuple[float, float], tuple[float, float]]:
    x = points[:, 0]
    y = points[:, 1]
    x_pad = max(1e-6, (np.nanpercentile(x, 99.5) - np.nanpercentile(x, 0.5)) * 0.08)
    y_pad = max(1e-6, (np.nanpercentile(y, 99.5) - np.nanpercentile(y, 0.5)) * 0.08)
    return (
        (float(np.nanpercentile(x, 0.5) - x_pad), float(np.nanpercentile(x, 99.5) + x_pad)),
        (float(np.nanpercentile(y, 0.5) - y_pad), float(np.nanpercentile(y, 99.5) + y_pad)),
    )


def _style_axis(ax) -> None:
    ax.set_facecolor(COLORS["background"])
    ax.grid(True, color=COLORS["grid"], linewidth=0.75, alpha=0.55)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#AAB4BF")
    ax.spines["bottom"].set_color("#AAB4BF")
    ax.tick_params(colors=COLORS["muted"], labelsize=9)


def _plot_projection(
    points: np.ndarray,
    labels: np.ndarray,
    scores: np.ndarray | None,
    title: str,
    subtitle: str,
    xlabel: str,
    ylabel: str,
    output_path: Path,
) -> None:
    normal = labels == 0
    anomaly = labels == 1

    fig, ax = plt.subplots(figsize=(10.5, 7.2))
    fig.patch.set_facecolor("white")
    _style_axis(ax)

    if normal.any():
        ax.scatter(
            points[normal, 0],
            points[normal, 1],
            s=13,
            c=COLORS["normal"],
            alpha=0.34,
            linewidths=0,
            rasterized=True,
        )
    if anomaly.any():
        if scores is not None and np.isfinite(scores[anomaly]).any():
            anomaly_colors = scores[anomaly]
            scatter = ax.scatter(
                points[anomaly, 0],
                points[anomaly, 1],
                s=30,
                c=anomaly_colors,
                cmap="Reds",
                alpha=0.94,
                edgecolors="white",
                linewidths=0.35,
                rasterized=True,
            )
            cbar = fig.colorbar(scatter, ax=ax, fraction=0.036, pad=0.025)
            cbar.set_label("Anomaly reconstruction score", color=COLORS["muted"])
            cbar.ax.tick_params(colors=COLORS["muted"], labelsize=8)
        else:
            ax.scatter(
                points[anomaly, 0],
                points[anomaly, 1],
                s=30,
                c=COLORS["anomaly"],
                alpha=0.94,
                edgecolors="white",
                linewidths=0.35,
                rasterized=True,
            )

    ax.set_title(title, loc="left", fontsize=16, fontweight="bold", color=COLORS["ink"], pad=20)
    ax.text(
        0.0,
        1.012,
        subtitle,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=10,
        color=COLORS["muted"],
    )
    ax.set_xlabel(xlabel, fontsize=10.5, color=COLORS["ink"])
    ax.set_ylabel(ylabel, fontsize=10.5, color=COLORS["ink"])
    xlim, ylim = _axis_limits(points)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)

    legend_handles = [
        Line2D([0], [0], marker="o", linestyle="", markerfacecolor=COLORS["normal"], markeredgecolor="none", markersize=7, alpha=0.75, label=f"Normal windows ({normal.sum():,})"),
        Line2D([0], [0], marker="o", linestyle="", markerfacecolor=COLORS["anomaly"], markeredgecolor="white", markersize=8, label=f"Anomaly windows ({anomaly.sum():,})"),
    ]
    ax.legend(handles=legend_handles, frameon=True, facecolor="white", edgecolor="#D6DCE3", loc="upper right")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=240, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize VAE latent space for normal/anomaly clusters.")
    parser.add_argument("--run-dir", default=None, help="Saved VAE run directory. Defaults to best standalone VAE run.")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"], help="Window split to visualize.")
    parser.add_argument("--output-dir", default="reports/figures", help="Directory for report PNG outputs.")
    parser.add_argument("--max-points", type=int, default=12000, help="Maximum points in each visualization.")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip-tsne", action="store_true", help="Only generate the PCA projection.")
    args = parser.parse_args()

    # Import registers custom Keras layers before load_model.
    import tensorflow as tf
    from src.models.vae_anomaly_detector import Sampling

    run_dir = _resolve(args.run_dir) if args.run_dir else _default_run_dir()
    output_dir = _resolve(args.output_dir)
    summary = _load_json(run_dir / "summary.json")
    source_run_dir = _resolve(summary["source_processed_windows_run"])

    windows, labels, metadata = _load_split_windows(source_run_dir, args.split)
    idx = _stratified_indices(labels, args.max_points, args.seed)
    sampled_windows = windows[idx]
    sampled_labels = labels[idx]

    scores_path = run_dir / f"{args.split}_scores.npy"
    sampled_scores = np.load(scores_path)[idx] if scores_path.exists() else None

    encoder = tf.keras.models.load_model(
        run_dir / "encoder.keras",
        compile=False,
        custom_objects={"Sampling": Sampling},
    )
    latent = _extract_z_mean(encoder, sampled_windows, batch_size=args.batch_size)
    scaled_latent = StandardScaler().fit_transform(latent)

    pca = PCA(n_components=2, random_state=args.seed)
    pca_points = pca.fit_transform(scaled_latent)
    variance = pca.explained_variance_ratio_ * 100.0

    run_label = summary.get("run_id", run_dir.name)
    model_bits = summary.get("vae_config", {})
    hidden = model_bits.get("hidden_units", "?")
    latent_dim = model_bits.get("latent_dim", latent.shape[1])
    window_size = metadata.get("params", {}).get("window_size", "?")
    subtitle = (
        f"Run {run_label} | split={args.split} | window={window_size} | "
        f"latent_dim={latent_dim} | hidden={hidden}"
    )

    stem = f"vae_latent_space_{run_label}_{args.split}"
    _plot_projection(
        pca_points,
        sampled_labels,
        sampled_scores,
        "VAE Latent Space: PCA Projection",
        f"{subtitle} | PC1={variance[0]:.1f}% variance, PC2={variance[1]:.1f}%",
        "Principal component 1",
        "Principal component 2",
        output_dir / f"{stem}_pca.png",
    )

    if not args.skip_tsne:
        perplexity = min(40, max(5, (len(sampled_labels) - 1) // 3))
        tsne = TSNE(
            n_components=2,
            perplexity=perplexity,
            init="pca",
            learning_rate="auto",
            random_state=args.seed,
        )
        tsne_points = tsne.fit_transform(scaled_latent)
        _plot_projection(
            tsne_points,
            sampled_labels,
            sampled_scores,
            "VAE Latent Space: t-SNE Cluster View",
            f"{subtitle} | stratified sample={len(sampled_labels):,}, perplexity={perplexity}",
            "t-SNE dimension 1",
            "t-SNE dimension 2",
            output_dir / f"{stem}_tsne.png",
        )

    print(f"Run directory: {run_dir}")
    print(f"Source windows: {source_run_dir}")
    print(f"Saved PCA figure: {output_dir / f'{stem}_pca.png'}")
    if not args.skip_tsne:
        print(f"Saved t-SNE figure: {output_dir / f'{stem}_tsne.png'}")


if __name__ == "__main__":
    main()
