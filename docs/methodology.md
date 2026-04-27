# Methodology Notes

## Problem Setup

The project detects MetroPT3 air-compressor air-leak failures from sensor
windows. The VAE is trained on normal windows only. At inference time,
reconstruction error is the anomaly score.

## Chronological Split

| Split | Period | Role |
|---|---|---|
| Train | Feb-Apr 2020 | Normal-only model fitting; F1-overlapping windows filtered |
| Validation | May 2020 | F2 labels for diagnostics and optional threshold analysis |
| Test | Jun-Jul 2020 | Held-out F3/F4 failure evaluation |

## Window Labels

For a 60-step window, the current main setting labels the window positive if the
last 10 percent of rows contain any failure label. This keeps detection aligned
with the end of each window.

## Scaling

The main experiments use raw, unscaled sensor windows. Earlier experiments found
that standard scaling made anomaly reconstruction too easy and reduced anomaly
separation.

## Thresholds

- `train_p98`: primary fair threshold, computed only from training scores.
- `val_f1`: diagnostic threshold, selected with validation labels.

Report claims should use `train_p98` unless explicitly discussing diagnostics.
