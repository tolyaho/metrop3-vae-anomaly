# Project Overview

The project trains reconstruction models on normal MetroPT3 compressor windows
and uses reconstruction error as an anomaly score for later air-leak failures.

## Data

MetroPT3 provides one-minute readings from 15 compressor sensors. The labeled
failure intervals used here are F1-F4 from 2020-04-18 through 2020-07-15.

## Main Setup

- Train on normal windows from February-April 2020.
- Use May 2020 as validation/diagnostic data.
- Evaluate on June-July 2020 failures.
- Build 60-step sliding windows with stride 10 for the main run.
- Label a window positive when the final 10 percent overlaps a failure interval.

## Models

- Dense VAE: `[128, 64]` encoder, `latent_dim=16`, `beta=1.0`.
- Isolation Forest: unsupervised classical baseline on the same normal training
  windows.

## Thresholds

- `train_p98`: 98th percentile of training scores. Main comparison threshold.
- `val_f1`: selected with validation labels. Diagnostic only.

Main VAE result at `train_p98`: F1=0.821, ROC-AUC=0.988.
