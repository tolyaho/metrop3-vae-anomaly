# Project overview

The project trains reconstruction models on normal MetroPT3
air-compressor windows and uses reconstruction error as an unsupervised
anomaly score for later air-leak failures. Numbers below come from the
multi-seed grid run on 2026-04-30. Full discussion in
[`reports/final_report.md`](../reports/final_report.md).

## Data

MetroPT3 provides one-minute readings from 15 compressor sensors.  The
labelled failure intervals used here are F1-F4 from 2020-04-18 through
2020-07-15.

## Main setup

- Train on normal windows from February-April 2020.
- Use May 2020 (F2) only for early-stopping / threshold diagnostics.
- Evaluate on June-July 2020 (F3, F4).
- Headline window: 120-step sliding windows with stride 20.  We also
  ablate window sizes 30 / 60 and architectures conv1d / LSTM-AE.
- Label a window positive when its **last 10 percent** overlaps a
  failure interval.

## Models compared

| Family   | Variant                                              |
|---|---|
| VAE      | dense `[128, 64]` (headline), dense `[64, 32]`, dense `[256, 128]`, dense `[128, 64, 32]`, dense `[256, 128, 64]`, conv1d, LSTM-autoencoder |
| Classical | Isolation Forest, PCA reconstruction (95 % variance), One-Class SVM (Nystroem + SGD), Local Outlier Factor |

Every model is fit only on the normal training windows.

## Thresholds

- `train_pXX` (98 / 99 / 99.5): unsupervised, deployable; `train_p98`
  is the headline.
- `val_f1`: diagnostic upper bound that uses validation labels.

## Headline result

Multi-seed (3 seeds: 42, 7, 123), test split, `train_p98`:

| Metric  | VAE (dense [128, 64], window=120) |
|---|---:|
| F1      | 0.931 +/- 0.006 |
| ROC-AUC | 0.997 +/- 0.000 |
| PR-AUC  | 0.893 +/- 0.013 |

Best classical baseline at the same threshold: Isolation Forest
(F1 = 0.661, ROC-AUC = 0.996, PR-AUC = 0.836).
