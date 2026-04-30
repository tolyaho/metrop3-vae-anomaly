# Reproducibility

## Environment

Python 3.11 + TensorFlow 2.17, pinned for H100 / A100 / T4 GPUs (CPU also
works):

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
python -c "import tensorflow as tf; print(tf.__version__, tf.config.list_physical_devices('GPU'))"
```

`requirements.txt` pins NumPy `<2.1`, TensorFlow `==2.17.*` and the data
/ plotting packages to the versions used to produce the figures.

## Data

Download MetroPT3 from UCI and drop the CSV at
`dataset/raw/MetroPT3(AirCompressor).csv`. The raw dataset is not
committed. If the environment cannot resolve `archive.ics.uci.edu`,
provide the IP directly:

```bash
curl --resolve archive.ics.uci.edu:443:128.195.10.252 -L -o /tmp/metropt3.zip \
    https://archive.ics.uci.edu/static/public/791/metropt+3+dataset.zip
```

## End-to-end pipeline

```bash
# 1. Label rows and build chronological splits
python scripts/preprocess.py --config configs/data/metropt3.json

# 2. Build sliding windows for the three reported window sizes
python scripts/build_windows.py --config configs/features/window30_noscale.json
python scripts/build_windows.py --config configs/features/window60_noscale.json
python scripts/build_windows.py --config configs/features/window120_noscale.json

# 3. Multi-seed VAE grid (layers x window-sizes x architectures, 3 seeds)
python scripts/run_experiment_grid.py --config configs/experiments/vae_grid.json
python scripts/collect_experiment_results.py --study-dir models/vae_grid_runs/<study_id>
python scripts/plot_experiment_comparison.py --study-dir models/vae_grid_runs/<study_id>

# 4. Classical baselines (one config per model, all on the window60 run)
python scripts/train_classical_baseline.py --config configs/baselines/isolation_forest_window60_noscale.json
python scripts/train_classical_baseline.py --config configs/baselines/pca_recon_window60_noscale.json
python scripts/train_classical_baseline.py --config configs/baselines/oc_svm_window60_noscale.json
python scripts/train_classical_baseline.py --config configs/baselines/lof_window60_noscale.json

# 5. Canonical comparison table + cross-model figures
python scripts/build_canonical_comparison.py \
    --vae-base-id win120
python scripts/plot_models_overlay.py \
    --vae-grid-dir models/vae_grid_runs/<study_id> \
    --vae-grid-base-id win120
python scripts/plot_threshold_sensitivity.py \
    --vae-run-dir models/vae_grid_runs/<study_id>/runs/windows/win120_s42/<run_ts>

# 6. Per-failure analysis (single --model flag, multiple <label>:<kind>:<dir> entries)
python scripts/analyze_failure_events.py --model \
    "VAE (dense)":vae:<best_vae_run_dir> \
    "Isolation Forest":baseline:<latest_if_run_dir> \
    "PCA reconstruction":baseline:<latest_pca_run_dir> \
    "OC-SVM":baseline:<latest_oc_svm_run_dir> \
    "LOF":baseline:<latest_lof_run_dir>

# 7. Latent-space PCA / t-SNE diagnostic for the best VAE
python scripts/plot_vae_latent_space.py \
    --run-dir <best_vae_run_dir> --split test
```

## Determinism

`src/models/vae_anomaly_detector.py::seed_everything` always seeds Python,
NumPy and TensorFlow. TF op-determinism (`enable_op_determinism`) is off
by default and only turned on when `VAE_DETERMINISM=1` is set; on this
dataset the deterministic cuDNN kernels interact with weight init in a
way that pushes training into the collapse state described in
`docs/methodology.md`.

## Threshold policy

- `train_p98` is the deployable unsupervised threshold and the only one
  quoted as a headline number.
- `val_f1` is a diagnostic upper bound. Don't quote it as the production
  result.

## Output artifacts

Lightweight CSV / Markdown / PNG outputs live under `reports/`. Heavy
per-run artifacts under `models/` and `dataset/` are gitignored; the
pipeline above regenerates them in about 50 minutes on a single H100.

The run that produced the numbers in the report contains 27 VAE runs
(5 layer configs x 3 seed + 3 window sizes x 3 seed + 2 conv1d seed +
1 LSTM seed) plus four classical baseline runs under `models/baseline_runs/`.
