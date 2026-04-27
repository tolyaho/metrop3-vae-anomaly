# Methodology

The task is unsupervised anomaly detection for MetroPT3 air-compressor failures.
The VAE is fit on normal windows and evaluated on later windows that may overlap
known air-leak intervals.

## Split

| Split | Period | Role |
|---|---|---|
| Train | Feb-Apr 2020 | Normal-only model fitting; F1-overlapping windows filtered |
| Validation | May 2020 | F2 labels for diagnostics and optional threshold analysis |
| Test | Jun-Jul 2020 | Held-out F3/F4 failure evaluation |

## Window Labels

Rows are labeled from the known failure intervals. Sliding windows are then
labeled at the window level. In the main 60-step setting, a window is positive
if the final 10 percent of rows contains any failure label. This avoids marking a
window positive because of an event far from the window endpoint.

## Scaling

The main experiments use raw, unscaled sensor windows. Standard scaling was
tested during development, but it reduced separation between normal and failure
windows for this setup.

## Thresholds

- `train_p98`: primary threshold, computed only from training scores.
- `val_f1`: diagnostic threshold, selected with validation labels.

Report claims should use `train_p98` unless explicitly discussing diagnostics.
