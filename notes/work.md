# Work Log

## 2026-04-23 — Repo refactor

- Reorganized into script-first structure: `scripts/`, `src/`, `configs/`, `archive/`
- Made raw data immutable (no overwrite)
- Added train-only scaling in feature engineering
- Made VAE inference deterministic (sampling layer returns latent mean)
- Moved old notebooks to `archive/notebooks_old/`
- Moved historical runs to `archive/legacy_outputs/vae_runs/`
- Ran preprocessing and feature-building scripts successfully

## 2026-04-24 — Training and evaluation

- Identified root cause of poor recent results: standard scaling was applied, killing anomaly discrimination. Archive runs used unscaled data.
- Built unscaled archive-like window run: `dataset/processed_windows/20260424_135957`
  - 63,526 train windows (normal-only), 21,275 val, 43,910 test
- Trained best model: `models/vae_runs/20260424_140621`
  - Dense [128,64], β=1.0, 20 epochs, seed=42, unscaled, val_f1 → F1=0.792
  - Same model with train_p98 threshold → **F1=0.821** (beats archive F1=0.8105)
- Added `scripts/evaluate_thresholds.py` for multi-threshold evaluation
- Added `beta_warmup_epochs` to `TrainConfig` and wired into training loop
- Fixed `Sampling` class Keras serialization (models now loadable after save)
- Added `train_percentile_val_f1` threshold method to `vae_anomaly_detector.py`
- Added `use_val_for_training` flag to `train_vae.py`

## Current best run

`models/vae_runs/20260424_140621` — see `reports/tables/20260424_140621_threshold_comparison.csv`

| Threshold | F1 | ROC-AUC |
|-----------|----|---------|
| val_f1 | 0.792 | 0.988 |
| train_p98 | **0.821** | 0.988 |
