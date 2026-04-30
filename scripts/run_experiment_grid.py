"""Run the VAE layer / window / architecture experiment grid (multi-seed capable).

Each entry in ``layer_experiments`` / ``window_experiments`` is repeated for
every seed in ``seeds`` (defaulting to ``[seed]`` for backward compatibility).
The experiment_id stored in the per-run grid metadata is suffixed with the
seed (``<id>_s<seed>``) so a downstream collector can group runs sharing the
same architecture / window.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _resolve(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def _run(args: list[str]) -> None:
    print("$ " + " ".join(args), flush=True)
    subprocess.run(args, cwd=PROJECT_ROOT, check=True)


def _matching_window_run(root: Path, split_run_name: str, window_size: int, stride: int) -> Path | None:
    if not root.exists():
        return None
    for run_dir in sorted([p for p in root.iterdir() if p.is_dir()], reverse=True):
        meta_path = run_dir / "metadata.json"
        if not meta_path.exists():
            continue
        try:
            meta = _load_json(meta_path)
        except Exception:
            continue
        params = meta.get("params", {})
        train_csv = str(params.get("train_csv_path", ""))
        split_match = (
            params.get("split_run_name") == split_run_name
            or split_run_name == "latest"
            or Path(train_csv).parent.name == split_run_name
        )
        same_geom = (
            int(params.get("window_size", -1)) == int(window_size)
            and int(params.get("stride", -1)) == int(stride)
        )
        if split_match and same_geom:
            if params.get("scale_features") is False and params.get("train_normal_only") is True:
                return run_dir
    return None


def _feature_config(base_cfg: dict, window_size: int, stride: int) -> dict:
    return {
        "preprocessed_root": base_cfg.get("preprocessed_root", "dataset/preprocessed"),
        "split_run_name": base_cfg.get("split_run_name", "latest"),
        "output_root": base_cfg.get("processed_windows_root", "dataset/processed_windows"),
        "timestamp_col": "timestamp",
        "label_col": "failure_label",
        "train_normal_only": True,
        "feature_cols": [],
        "point_mode": False,
        "window_size": int(window_size),
        "stride": int(stride),
        "flatten_windows": True,
        "window_label_strategy": "last",
        "window_label_positive_ratio": 0.01,
        "window_label_last_percent": 10.0,
        "scale_features": False,
        "scaler_type": "standard",
    }


def _vae_config(
    base_cfg: dict,
    exp: dict,
    window_run: Path,
    output_root: Path,
    group: str,
    *,
    seed: int,
    base_experiment_id: str,
    experiment_id: str,
) -> dict:
    return {
        "processed_windows_root": str(_resolve(base_cfg.get("processed_windows_root", "dataset/processed_windows")).relative_to(PROJECT_ROOT)),
        "window_run_name": window_run.name,
        "model_output_root": str(output_root.relative_to(PROJECT_ROOT)),
        "use_val_for_training": True,
        "grid_metadata": {
            "study_name": base_cfg["study_name"],
            "experiment_group": group,
            "experiment_id": experiment_id,
            "base_experiment_id": base_experiment_id,
            "window_size": int(exp["window_size"]),
            "stride": int(exp["stride"]),
            "hidden": exp["hidden"],
            "scaling": base_cfg.get("scaling", "none"),
            "seed": int(seed),
        },
        "vae_config": {
            "architecture": exp.get("architecture", "dense"),
            "latent_dim": int(base_cfg["latent_dim"]),
            "hidden_units": exp["hidden"],
            "encoder_use_batchnorm": False,
            "encoder_dropout_rate": 0.0,
        },
        "train_config": {
            "epochs": int(base_cfg["epochs"]),
            "batch_size": int(base_cfg.get("batch_size", 512)),
            "learning_rate": float(base_cfg.get("learning_rate", 1e-3)),
            "beta": float(base_cfg["beta"]),
            "beta_warmup_epochs": int(base_cfg.get("beta_warmup_epochs", 0)),
            "validation_split": 0,
            "gradient_clipnorm": 1.0,
            "random_seed": int(seed),
            "verbose_epoch": True,
            "early_stopping_metric": str(base_cfg.get("early_stopping_metric", "val_roc_auc")),
            "early_stopping_patience": int(base_cfg.get("early_stopping_patience", 4)),
            "early_stopping_min_epochs": int(base_cfg.get("early_stopping_min_epochs", 2)),
            "early_stopping_restore_best": bool(base_cfg.get("early_stopping_restore_best", True)),
        },
        "threshold_config": {
            "method": "percentile",
            "percentile": 98.0,
            "std_factor": 3.0,
        },
        "caps": {
            "max_train_windows": None,
            "max_val_windows": None,
            "max_test_windows": None,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the VAE experiment grid.")
    parser.add_argument("--config", default="configs/experiments/vae_grid.json")
    parser.add_argument("--study-dir", default=None)
    parser.add_argument("--skip-existing", action="store_true", help="Skip an experiment if its output root already has a run.")
    args = parser.parse_args()

    cfg_path = _resolve(args.config)
    cfg = _load_json(cfg_path)
    seeds = [int(s) for s in cfg.get("seeds", [int(cfg.get("seed", 42))])]
    timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    study_id = f"{timestamp}_{cfg['study_name']}"
    study_dir = _resolve(args.study_dir) if args.study_dir else PROJECT_ROOT / "models" / "vae_grid_runs" / study_id
    configs_dir = study_dir / "configs"
    runs_dir = study_dir / "runs"
    tables_dir = study_dir / "tables"
    figures_dir = study_dir / "figures"
    for path in (configs_dir, runs_dir, tables_dir, figures_dir):
        path.mkdir(parents=True, exist_ok=True)

    _write_json(study_dir / "summary.json", {"study_id": study_dir.name, "config": cfg, "status": "running"})
    _write_json(configs_dir / "grid_config.json", cfg)

    processed_root = _resolve(cfg.get("processed_windows_root", "dataset/processed_windows"))
    split_run_name = cfg.get("split_run_name", "latest")
    manifest_rows: list[dict] = []
    experiments = [
        ("layers", exp) for exp in cfg.get("layer_experiments", [])
    ] + [
        ("windows", exp) for exp in cfg.get("window_experiments", [])
    ] + [
        ("architectures", exp) for exp in cfg.get("architecture_experiments", [])
    ]

    for group, exp in experiments:
        base_id = exp["id"]
        window_run = _matching_window_run(processed_root, split_run_name, int(exp["window_size"]), int(exp["stride"]))
        if window_run is None:
            feature_cfg = _feature_config(cfg, int(exp["window_size"]), int(exp["stride"]))
            feature_cfg_path = configs_dir / f"features_{group}_{base_id}.json"
            _write_json(feature_cfg_path, feature_cfg)
            before = set(p.name for p in processed_root.iterdir() if p.is_dir()) if processed_root.exists() else set()
            print(f"Building windows for {base_id}...")
            _run([sys.executable, "scripts/build_windows.py", "--config", str(feature_cfg_path.relative_to(PROJECT_ROOT))])
            after = sorted([p for p in processed_root.iterdir() if p.is_dir() and p.name not in before])
            if not after:
                raise RuntimeError(f"Window build finished but no new run appeared in {processed_root}")
            window_run = after[-1]
            reused = False
        else:
            reused = True
        meta = _load_json(window_run / "metadata.json")

        exp_seeds = [int(s) for s in exp.get("seeds", seeds)]
        for seed in exp_seeds:
            exp_id = f"{base_id}_s{seed}" if len(exp_seeds) > 1 else base_id
            print(f"\n=== {group}/{exp_id} (seed={seed}) ===", flush=True)
            print(f"{'Reusing' if reused else 'Built'} windows: {window_run}")

            run_output_root = runs_dir / group / exp_id
            if args.skip_existing and run_output_root.exists() and any(p.is_dir() for p in run_output_root.iterdir()):
                print(f"Skipping existing run output: {run_output_root}")
                run_dir = sorted(p for p in run_output_root.iterdir() if p.is_dir())[-1]
            else:
                vae_cfg = _vae_config(
                    cfg,
                    exp,
                    window_run,
                    run_output_root,
                    group,
                    seed=seed,
                    base_experiment_id=base_id,
                    experiment_id=exp_id,
                )
                vae_cfg_path = configs_dir / f"vae_{group}_{exp_id}.json"
                _write_json(vae_cfg_path, vae_cfg)
                before = set(p.name for p in run_output_root.iterdir() if p.is_dir()) if run_output_root.exists() else set()
                print(f"Training VAE for {exp_id} (seed {seed})...")
                _run([sys.executable, "scripts/train_vae.py", "--config", str(vae_cfg_path.relative_to(PROJECT_ROOT))])
                after = sorted([p for p in run_output_root.iterdir() if p.is_dir() and p.name not in before])
                if not after:
                    raise RuntimeError(f"VAE training finished but no new run appeared in {run_output_root}")
                run_dir = after[-1]

            manifest_rows.append({
                "study_id": study_dir.name,
                "experiment_group": group,
                "base_experiment_id": base_id,
                "experiment_id": exp_id,
                "seed": int(seed),
                "window_run": str(window_run.relative_to(PROJECT_ROOT)),
                "run_dir": str(run_dir.relative_to(PROJECT_ROOT)),
                "window_size": int(exp["window_size"]),
                "stride": int(exp["stride"]),
                "hidden": json.dumps(exp["hidden"]),
                "architecture": exp.get("architecture", "dense"),
                "reused_windows": reused,
                "n_train_windows": int(meta.get("train_windows_shape", [0])[0]),
                "n_val_windows": int(meta.get("val_windows_shape", [0])[0]),
                "n_test_windows": int(meta.get("test_windows_shape", [0])[0]),
            })
            pd.DataFrame(manifest_rows).to_csv(study_dir / "window_manifest.csv", index=False)

    summary = {"study_id": study_dir.name, "config": cfg, "status": "completed", "experiments": manifest_rows}
    _write_json(study_dir / "summary.json", summary)
    print(f"\nStudy complete: {study_dir}")
    print("Next:")
    print(f"  python scripts/collect_experiment_results.py --study-dir {study_dir.relative_to(PROJECT_ROOT)}")
    print(f"  python scripts/plot_experiment_comparison.py --study-dir {study_dir.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
