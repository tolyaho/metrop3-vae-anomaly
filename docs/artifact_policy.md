# Artifact Policy

Commit source code, configs, documentation, and lightweight report outputs. Do
not commit raw data, processed arrays, model weights, or full run folders.

## Tracked

- `README.md`, `docs/`, and concise project notes
- reusable code in `src/`
- command-line scripts in `scripts/`
- reproducibility configs in `configs/`
- selected tables and figures in `reports/`
- placeholder files such as `.gitkeep`

## Ignored

- `dataset/raw/`, `dataset/preprocessed/`, `dataset/processed_windows/`
- `data/` local data drops, except `data/README.md`
- `models/vae_runs/`, `models/baseline_runs/`, `models/vae_grid_runs/`
- `archive/notebooks_old/*.ipynb` and legacy output folders with absolute paths
- Python environments and caches
- notebook checkpoints
- local IDE/tool state
- temporary outputs under `outputs/`

## Reproducibility

Large artifacts should be regenerated from scripts. The main workflow is:

```bash
python scripts/preprocess.py --config configs/data/archive_like_window60.json
python scripts/build_windows.py --config configs/features/window60_noscale.json
python scripts/train_vae.py --config configs/experiments/dense_window60_beta1_noscale_final.json
python scripts/evaluate.py --run-dir models/vae_runs/<run_id>
```

Wrap-up comparisons are generated with:

```bash
python scripts/train_isolation_forest.py --config configs/baselines/isolation_forest_window60_noscale.json
python scripts/compare_baselines.py --vae-run-dir models/vae_runs/20260424_140621 --if-run-dir models/baseline_runs/isolation_forest/<run_id>
python scripts/run_experiment_grid.py --config configs/experiments/wrapup_grid_layers_windows.json
python scripts/collect_experiment_results.py --study-dir models/vae_grid_runs/<study_id>
python scripts/plot_experiment_comparison.py --study-dir models/vae_grid_runs/<study_id>
```
