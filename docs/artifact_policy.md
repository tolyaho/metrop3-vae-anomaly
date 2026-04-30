# Artifact policy

Commit source code, configs, documentation and the lightweight report
outputs. Do not commit raw data, processed arrays, model weights or full
run folders.

## Tracked

- `README.md`, `docs/`, project notes
- Reusable code in `src/`
- Command-line scripts in `scripts/`
- Configs in `configs/`
- Tables and figures in `reports/`

## Ignored

- `dataset/` (raw, preprocessed and processed windows), except `dataset/README.md`
- `models/` run directories, except `models/README.md`
- Python environments and caches
- Notebook checkpoints
- Local IDE / tool state
- `*.log`, `*.tmp`, `tmp/`

## Reproducibility

Heavy artifacts are regenerated from scripts. Full pipeline in
[`reproducibility.md`](reproducibility.md); short version:

```bash
python scripts/preprocess.py     --config configs/data/metropt3.json
python scripts/build_windows.py  --config configs/features/window30_noscale.json
python scripts/build_windows.py  --config configs/features/window60_noscale.json
python scripts/build_windows.py  --config configs/features/window120_noscale.json

python scripts/run_experiment_grid.py --config configs/experiments/vae_grid.json
python scripts/collect_experiment_results.py --study-dir models/vae_grid_runs/<study_id>
python scripts/plot_experiment_comparison.py --study-dir models/vae_grid_runs/<study_id>

python scripts/train_classical_baseline.py --config configs/baselines/isolation_forest_window60_noscale.json
python scripts/train_classical_baseline.py --config configs/baselines/pca_recon_window60_noscale.json
python scripts/train_classical_baseline.py --config configs/baselines/oc_svm_window60_noscale.json
python scripts/train_classical_baseline.py --config configs/baselines/lof_window60_noscale.json

python scripts/build_canonical_comparison.py --vae-base-id win120
python scripts/plot_models_overlay.py        --vae-grid-dir models/vae_grid_runs/<study_id> --vae-grid-base-id win120
python scripts/plot_threshold_sensitivity.py --vae-run-dir models/vae_grid_runs/<study_id>/runs/windows/win120_s42/<run_ts>
python scripts/analyze_failure_events.py --model "VAE (dense)":vae:<best_vae_run> ...
python scripts/plot_vae_latent_space.py --run-dir <best_vae_run> --split test
```
