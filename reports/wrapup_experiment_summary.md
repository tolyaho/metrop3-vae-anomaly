# Wrap-Up Experiment Summary

Study directory: `models/vae_grid_runs/20260426_230603_wrapup_layers_windows_noscale_beta1`

Main threshold: `train_p98` from training reconstruction scores.
Validation-label thresholds are kept only as diagnostics.

## Best Test Results By Group

| Group | Experiment | Window | Hidden | Precision | Recall | F1 | ROC-AUC | PR-AUC |
|---|---|---:|---|---:|---:|---:|---:|---:|
| layers | layers_128_64_32 | 60 | `[128, 64, 32]` | 0.903 | 0.863 | 0.883 | 0.996 | 0.859 |
| windows | win120 | 120 | `[128, 64]` | 0.353 | 0.100 | 0.156 | 0.990 | 0.638 |

## All Fresh Grid Runs at train_p98

| Group | Experiment | Window | Hidden | Precision | Recall | F1 | ROC-AUC | PR-AUC |
|---|---|---:|---|---:|---:|---:|---:|---:|
| layers | layers_128_64_32 | 60 | `[128, 64, 32]` | 0.903 | 0.863 | 0.883 | 0.996 | 0.859 |
| layers | layers_64_32 | 60 | `[64, 32]` | 0.251 | 0.069 | 0.108 | 0.537 | 0.069 |
| layers | layers_128_64 | 60 | `[128, 64]` | 0.165 | 0.075 | 0.103 | 0.326 | 0.052 |
| layers | layers_256_128 | 60 | `[256, 128]` | 0.170 | 0.070 | 0.099 | 0.323 | 0.043 |
| layers | layers_256_128_64 | 60 | `[256, 128, 64]` | 0.139 | 0.036 | 0.058 | 0.139 | 0.030 |
| windows | win120 | 120 | `[128, 64]` | 0.353 | 0.100 | 0.156 | 0.990 | 0.638 |
| windows | win60 | 60 | `[128, 64]` | 0.174 | 0.060 | 0.089 | 0.091 | 0.036 |
| windows | win30 | 30 | `[128, 64]` | 0.005 | 0.004 | 0.004 | 0.219 | 0.027 |

## Final Baseline Context

| Model | Run | Precision | Recall | F1 | ROC-AUC | PR-AUC |
|---|---|---:|---:|---:|---:|---:|
| VAE | `20260424_140621` | 0.716 | 0.963 | 0.821 | 0.988 | 0.762 |
| Isolation Forest | `20260426_200420` | 0.496 | 0.994 | 0.661 | 0.996 | 0.836 |

## Notes

- The comparison uses raw, unscaled windows.
- `val_f1` rows are diagnostic because they use validation labels.
- The main fair model-selection threshold is `train_p98`.
