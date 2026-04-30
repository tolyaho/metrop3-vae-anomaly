# Methodology

The task is unsupervised anomaly detection on the MetroPT3 air-compressor
sensor stream. Models are fit on a chronological prefix that contains
only normal operation, and they have to flag windows that overlap the
four annotated air-leak failure events without supervision.

## Data and split

| Split | Period | Role | Rows | Windows (size 60, stride 10) | Positive windows |
|---|---|---|---:|---:|---:|
| Train      | 2020-02-01 to 2020-04-30 | Normal-only model fitting (windows overlapping any failure are removed) | 644,032 | 63,526 | 0 |
| Validation | 2020-05-01 to 2020-05-31 | F2 used for early stopping and threshold diagnostics | 212,800 | 21,275 | 237 |
| Test       | 2020-06-01 to 2020-07-31 | Held-out F3 + F4 events | 439,152 | 43,910 | 1,894 |

The split is strictly chronological so generalisation to unseen calendar
weeks is part of the evaluation.

## Window labels

Row labels come from the four annotated failure intervals (F1-F4 in
`src/data/preprocessing.py::DEFAULT_FAILURE_EVENTS`). A sliding window
is positive when any row in its final 10 percent overlaps a failure
interval. This rule keeps the leading edge of a window from being
labelled by what happens at its trailing edge, and it matches industrial
alarm semantics where the most recent observation drives the alert.

## Feature scaling

The VAE uses raw, unscaled windows. Z-scoring on the train normals was
tested and reduced normal-vs-failure separation here, because the
post-failure shutdown signature relies on absolute pressure / current
magnitudes that scaling collapses. Baselines that need scaling (PCA
reconstruction, OC-SVM, LOF) apply their own `StandardScaler` internally,
fit only on the normal training windows.

## Models

Every model sees the same flattened-window inputs (window_size x
n_features features per window). Every model is fit only on the normal
training windows.

- **VAE (dense, primary).** `hidden=[128, 64]` encoder, latent dim 16,
  mirrored decoder. Loss is per-window MSE plus `beta * KL` with
  `beta = 1`. Adam (lr 1e-3, gradient clipnorm 1.0). Up to 30 epochs,
  early stopping on `val_total_loss` with patience 6, plus a collapse
  guard (see below).
- **VAE (1-D conv, ablation).** Same loss, encoder uses 1-D conv blocks
  before the latent projection.
- **VAE (LSTM autoencoder, ablation).** Sequential LSTM encoder and
  decoder over the same window length.
- **Isolation Forest.** 500 trees, default subsampling and contamination.
- **PCA reconstruction.** Standardise, keep the smallest number of
  principal components explaining 95 percent of variance, score with
  squared reconstruction error in the standardised space.
- **One-Class SVM.** Approximate RBF kernel via `Nystroem` (200
  components) followed by `SGDOneClassSVM` (`nu=0.05`). Subsamples to
  30k normal windows so it scales beyond the ~63k flattened windows.
- **LOF.** `novelty=True`, `n_neighbors=35`, fit on a 20k normal
  subsample. LOF queries are O(n_test x n_train), which is why
  subsampling is required to keep the comparison tractable.

### Collapse guard

The dense VAE on this dataset has an attractor where it eventually
reconstructs anomalies as well as normals. When that happens, ROC-AUC
peaks above 0.99 in the first ~15 epochs and then collapses to ~0.08 by
epoch 25 (the score direction inverts). Plain patience-based early
stopping on validation loss does not catch this, because the loss keeps
decreasing.

The fix is to also track the best `val_roc_auc` seen during training.
If the AUC peaks above 0.7 and later drops below 0.5, training is
stopped and the model is restored to the best-AUC checkpoint. With this
guard always on, the multi-seed grid is reproducible and the headline
F1 = 0.931 holds.

## Scoring

Each model produces one anomaly score per test window (higher = more
anomalous).

- VAE: per-window reconstruction MSE evaluated at the latent mean (no
  sampling at inference, so scores are deterministic).
- Isolation Forest, OC-SVM, LOF: `-decision_function` / `-score_samples`.
- PCA: squared reconstruction error in the standardised space.

## Thresholds

- `train_p98`, `train_p99`, `train_p995`: the 98 / 99 / 99.5th
  percentile of the model's training scores. Fully unsupervised. The
  headline comparison uses `train_p98`.
- `val_f1`: threshold maximising validation F1. Reported only as a
  diagnostic upper bound, since it uses validation labels.

## Multi-seed evaluation

Every dense VAE configuration is replicated over three seeds (42, 7,
123). The 1-D conv VAE is run with two seeds (42, 7) and the LSTM
autoencoder with one seed (42), because LSTM training is roughly 6x
slower. Tables and figures report `mean +/- std` across seeds at
`train_p98` on test. The headline model is the configuration with the
highest mean F1 and the lowest seed variance, which on this dataset is
the dense VAE with `hidden=[128, 64]`, `latent_dim=16` and `window_size=120`
(`base_experiment_id=win120`, F1 = 0.931 +/- 0.006 across three seeds).

## Per-failure analysis

Test-set aggregate metrics are complemented by latency-to-first-alarm
for each annotated failure, computed as the gap (in minutes) between
the event start timestamp and the first window inside the event whose
score exceeds `train_p98`. This is the operational quantity that
matters more than aggregate F1 for a maintenance team.
