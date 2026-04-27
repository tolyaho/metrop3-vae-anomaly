# Reproducibility

## Environment

Install dependencies with:

```bash
pip install -r requirements.txt
```

If multiple Python environments exist, use the one with TensorFlow available:

```bash
python -c "import tensorflow as tf; print(tf.__version__)"
```

## Data

Download MetroPT3 from the UCI repository and place the files in `dataset/raw/`.
The raw dataset is not committed.

## Main VAE Pipeline

```bash
python scripts/preprocess.py --config configs/data/archive_like_window60.json
python scripts/build_windows.py --config configs/features/window60_noscale.json
python scripts/train_vae.py --config configs/experiments/dense_window60_beta1_noscale_final.json
python scripts/evaluate.py --run-dir models/vae_runs/<run_id>
```

`train_p98` is the main threshold: the 98th percentile of training
reconstruction scores. `val_f1` rows are diagnostic because that threshold uses
validation labels.

## Fair Classical Baseline

```bash
python scripts/train_isolation_forest.py --config configs/baselines/isolation_forest_window60_noscale.json
python scripts/compare_baselines.py \
  --vae-run-dir models/vae_runs/20260424_140621 \
  --if-run-dir models/baseline_runs/isolation_forest/<run_id>
```

## Wrap-Up Grid

```bash
python scripts/run_experiment_grid.py --config configs/experiments/wrapup_grid_layers_windows.json
python scripts/collect_experiment_results.py --study-dir models/vae_grid_runs/<study_id>
python scripts/plot_experiment_comparison.py --study-dir models/vae_grid_runs/<study_id>
```

Generated full run directories are ignored. Lightweight summary outputs are
written to `reports/tables/`, `reports/figures/`, and
`reports/wrapup_experiment_summary.md`.
