# Project Overview

## What we're doing

Train a VAE on **normal** air-compressor sensor data. At inference, reconstruction error is the anomaly score — windows that don't look like normal operation reconstruct poorly and score high. A threshold converts scores to binary predictions.

## Data

UCI MetroPT3 dataset (~1.5M rows, 15 sensor features, 1-minute resolution). Four labeled AirLeak failures:

| ID | Period | Duration |
|----|--------|----------|
| F1 | Apr 18 2020 | 24 h |
| F2 | May 29–30 2020 | 6.5 h |
| F3 | Jun 5–7 2020 | 52 h |
| F4 | Jul 15 2020 | 4.5 h |

**Split used** (archive-like): Train Feb–May (normal only), Val May–Jun (F2), Test Jun–Aug (F3, F4).

## Model

Dense VAE: flatten 60×15 window → Dense(128) → Dense(64) → latent(16) → decode back.

Loss: `reconstruction_MSE + β × KL_divergence`

β=0 = autoencoder baseline (no KL regularization).  
β=1 = true VAE (latent space regularized toward N(0,1)).

Inference is deterministic: sampling layer returns latent mean, not a sample.

## Threshold strategies

| Method | How |
|--------|-----|
| `val_f1` | Search thresholds, maximize F1 on val labels |
| `train_p98` | 98th percentile of training reconstruction errors |
| `mean_std` | mean + k×std of training scores |

## Key finding

**Input scaling hurts** — standard scaling allows the model to reconstruct anomaly windows as well as normal ones (ROC-AUC collapses to ~0.5). Using raw unscaled sensor values preserves the anomaly signal.

Best result: F1=0.821 (train_p98 threshold), ROC-AUC=0.988.
