# Work log

## 2026-04-23

- Moved the workflow from notebooks into scripts.
- Added raw-data labeling and chronological split generation.
- Added feature/window generation with normal-only training windows.
- Made VAE inference deterministic by using the latent mean at scoring.

## 2026-04-24

- Built the raw, unscaled 60-step window run used by the main experiment.
- Trained the dense VAE run `20260424_140621`.
- Added multi-threshold evaluation for `train_pXX` and `val_f1` rules.

## 2026-04-26 to 2026-04-27

- Added an unsupervised Isolation Forest baseline trained on normal
  windows.
- Added the layer/window comparison grid.
- Cleaned report outputs so the active tables and figures match the
  unsupervised comparison.

## 2026-04-30 (rebuild)

- Pinned a fresh Python 3.11 / TensorFlow 2.17 environment in `.venv`,
  re-downloaded the raw MetroPT3 CSV from UCI and re-ran preprocessing
  and windowing for `window_size in {30, 60, 120}`.
- Centralised matplotlib styling in `src/plotting/` (palette, fonts,
  dpi=300, model colour map).
- Added a collapse-guarded early-stopping path to `train_vae`: track
  the best `val_roc_auc` seen so far and restore those weights if the
  AUC peaks above 0.7 and then drops below 0.5. Without the guard the
  dense VAE on this dataset converges to an inverted-score solution
  after ~20 epochs.
- Replaced the bespoke Isolation Forest runner with a unified
  `scripts/train_classical_baseline.py`. Added Isolation Forest, PCA
  reconstruction, One-Class SVM (Nystroem + SGD) and LOF baselines, all
  fed identical normal-only `window_size=60` flattened windows.
- Added multi-seed support to `scripts/run_experiment_grid.py` and
  `scripts/collect_experiment_results.py`. Aggregated results land in
  `grid_aggregated.csv` (mean +/- std across seeds).
- Re-ran the full grid (5 layer x 3 seeds + 3 window x 3 seeds + 2
  conv1d seeds + 1 LSTM seed = 27 runs) plus the four baselines. Best
  configuration: dense VAE, `hidden=[128, 64]`, `window_size=120`,
  F1 = 0.931 +/- 0.006, ROC-AUC = 0.997, PR-AUC = 0.893.
- Added the cross-model PR + ROC overlay (`plot_models_overlay.py`),
  threshold sensitivity (`plot_threshold_sensitivity.py`), per-failure
  analysis (`analyze_failure_events.py`) and the comparison table
  builder (`build_canonical_comparison.py`).
- Updated `README.md`, `docs/methodology.md`, `docs/reproducibility.md`
  and `notes/state.md` with the new numbers and the pipeline; wrote
  `reports/final_report.md` as the writeup.

## 2026-04-30 (cleanup)

- Removed dead configs (`_smoke_*`, `dense_point_*`, the duplicated
  `window60`/`point` feature configs, single-experiment dense/conv1d
  configs superseded by the grid) and the back-compat shim modules in
  `src/` (`data_preprocessing.py`, `feature_engineering.py`,
  `metrics.py`, `vae_anomaly_detector.py`).
- Deleted the empty placeholder dirs (`notebooks/`, `data/`, `outputs/`,
  `archive/legacy_outputs/`, `archive/notebooks_old/`) and dropped the
  now-empty `archive/` after archiving the legacy supervised-baseline
  and Isolation Forest scripts.
- Renamed `wrapup_grid_layers_windows.json` -> `vae_grid.json`,
  `archive_like_window60.json` -> `metropt3.json` and
  `dense_window60_beta1_noscale_final.json` -> `vae_dense_window60.json`.
  The aggregated grid output is now `grid_*` rather than `wrapup_grid_*`.
- Rewrote `.gitignore` from scratch (no stale figure allow-lists).
