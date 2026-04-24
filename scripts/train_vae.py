from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT / "src"))


def _resolve(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def load_config(path: str | Path) -> dict:
    with _resolve(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def _git_commit() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception:
        return None
    return result.stdout.strip()


def _select_window_run(root: Path, name: str) -> Path:
    from vae_anomaly_detector import find_latest_processed_window_run, find_processed_window_run_by_name

    if name == "latest":
        return find_latest_processed_window_run(root, require_val=True)
    return find_processed_window_run_by_name(root, name)


def _cap_arrays(windows: np.ndarray, labels: np.ndarray, cap: int | None) -> tuple[np.ndarray, np.ndarray]:
    if cap is not None and len(windows) > cap:
        return windows[:cap], labels[:cap]
    return windows, labels


def main() -> None:
    parser = argparse.ArgumentParser(description="Train, score, threshold, and save a VAE anomaly run.")
    parser.add_argument("--config", default="configs/experiments/dense_point_beta0.json", help="Experiment JSON config.")
    args = parser.parse_args()

    from vae_anomaly_detector import (
        ThresholdConfig,
        TrainConfig,
        VAEConfig,
        classify,
        find_latest_processed_window_run,
        find_processed_window_run_by_name,
        load_window_run,
        mse_reconstruction_scores,
        save_training_artifacts,
        select_threshold,
        train_vae,
    )

    cfg = load_config(args.config)
    processed_root = _resolve(cfg["processed_windows_root"])
    window_run = _select_window_run(processed_root, cfg.get("window_run_name", "latest"))

    train_windows, val_windows, test_windows, train_labels, val_labels, test_labels, source_meta = load_window_run(window_run)

    caps = cfg.get("caps", {})
    train_windows, train_labels = _cap_arrays(train_windows, train_labels, caps.get("max_train_windows"))
    val_windows, val_labels = _cap_arrays(val_windows, val_labels, caps.get("max_val_windows"))
    test_windows, test_labels = _cap_arrays(test_windows, test_labels, caps.get("max_test_windows"))

    vae_cfg = VAEConfig(**cfg.get("vae_config", {}))
    train_cfg = TrainConfig(**cfg.get("train_config", {}))
    threshold_cfg = ThresholdConfig(**cfg.get("threshold_config", {}))

    # When use_val_for_training=false, pass no external val to the training loop
    # so it uses the internal validation_split fraction of training data (all normal).
    # This matches the original notebook setup where val_windows.npy did not exist.
    use_val_for_training = cfg.get("use_val_for_training", True)
    train_val_windows = val_windows if (use_val_for_training and len(val_windows)) else None

    encoder, decoder, history = train_vae(
        train_windows,
        vae_cfg,
        train_cfg,
        val_windows=train_val_windows,
    )

    train_scores = mse_reconstruction_scores(encoder, decoder, train_windows, batch_size=train_cfg.batch_size)
    val_scores = mse_reconstruction_scores(encoder, decoder, val_windows, batch_size=train_cfg.batch_size) if len(val_windows) else np.array([], dtype=np.float32)
    test_scores = mse_reconstruction_scores(encoder, decoder, test_windows, batch_size=train_cfg.batch_size)

    if threshold_cfg.method == "val_f1" and len(val_scores) and np.unique(val_labels).size >= 2:
        threshold, threshold_info = select_threshold(
            train_scores=train_scores,
            cfg=threshold_cfg,
            val_scores=val_scores,
            val_labels=val_labels,
        )
    else:
        fallback_cfg = ThresholdConfig(
            method="percentile",
            percentile=threshold_cfg.percentile,
            std_factor=threshold_cfg.std_factor,
        )
        threshold, threshold_info = select_threshold(train_scores=train_scores, cfg=fallback_cfg)
        threshold_info["fallback_reason"] = "validation labels unavailable or single-class"

    train_preds = classify(train_scores, threshold)
    val_preds = classify(val_scores, threshold) if len(val_scores) else np.array([], dtype=np.int32)
    test_preds = classify(test_scores, threshold)

    run_dir, summary = save_training_artifacts(
        output_root=_resolve(cfg["model_output_root"]),
        encoder=encoder,
        decoder=decoder,
        history=history,
        train_scores=train_scores,
        val_scores=val_scores,
        test_scores=test_scores,
        threshold=threshold,
        train_preds=train_preds,
        val_preds=val_preds,
        test_preds=test_preds,
        train_labels=train_labels,
        val_labels=val_labels,
        test_labels=test_labels,
        vae_cfg=vae_cfg,
        train_cfg=train_cfg,
        threshold_cfg=threshold_cfg,
        threshold_info=threshold_info,
        source_run_dir=window_run,
        source_metadata=source_meta,
        run_config=cfg,
        git_commit=_git_commit(),
    )

    print(f"Saved run: {run_dir}")
    print(f"Threshold: {summary['threshold']:.6f}")
    print(f"Validation metrics: {summary['validation_metrics']}")
    print(f"Test metrics: {summary['metrics']}")


if __name__ == "__main__":
    main()
