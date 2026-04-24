# Project State

Last updated: 2026-04-24

## Current best results

Primary run: `models/vae_runs/20260424_140621`

Config: Dense VAE, hidden=[128,64], latent=16, β=1.0, 20 epochs, seed=42, **no scaling**, archive-like split, window_size=60, stride=10.

| Threshold | Precision | Recall | F1 | ROC-AUC |
|-----------|-----------|--------|----|---------|
| val_f1 (th=13617) | 0.793 | 0.791 | 0.792 | 0.988 |
| **train_p98** (th=13345) | **0.716** | **0.963** | **0.821** | **0.988** |
| Archive best (val_f1) | 0.895 | 0.740 | 0.811 | — |

Metric CSVs: `reports/tables/20260424_140621_threshold_comparison.csv`

## Data

Raw CSV: `dataset/raw/MetroPT3(AirCompressor).csv` (~208 MB, 1.5M rows, not committed).

Active preprocessed split: `dataset/preprocessed/splits/20260424_005227`
- Train: Feb–May 2020, 644,032 rows
- Val: May–Jun 2020, 212,800 rows (contains F2)
- Test: Jun–Aug 2020, 439,152 rows (contains F3, F4)

Active processed windows (unscaled): `dataset/processed_windows/20260424_135957`
- Train: 63,526 windows (normal-only)
- Val: 21,275 windows (237 positive)
- Test: 43,910 windows (1,894 positive)

## What works / what doesn't

**Works:** Dense [128,64] + β=1.0 + no input scaling + seed=42 + external val monitoring during training.

**Doesn't work:** Standard scaling (ROC-AUC → ~0.5), BatchNorm (kills KL explosion), Conv1D (no KL explosion), beta warmup (model memorizes anomalies during β=0 phase), other seeds.

The key mechanism: with unscaled data, the KL term explodes in epoch 1 (~7×10²⁹). Gradient clipping (norm=1.0) lets the smaller reconstruction gradient dominate weight updates, creating useful latent-space regularization. Anything that prevents this explosion kills anomaly detection.

## Pipeline

```bash
# 1. Label and split raw data
python scripts/preprocess.py --config configs/data/archive_like_window60.json

# 2. Build unscaled 60-step windows
python scripts/build_windows.py --config configs/features/window60_noscale.json

# 3. Train VAE and save run
python scripts/train_vae.py --config configs/experiments/dense_window60_beta1_noscale_final.json

# 4. Evaluate with multiple thresholds
python scripts/evaluate_thresholds.py --run-dir models/vae_runs/<run_id>
```

## Archive

Historical runs in `archive/legacy_outputs/vae_runs/` (best: `20260422_013301`, F1=0.8105). These used older format — no val scores/labels saved, paths reference another machine.
