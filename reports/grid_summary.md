# Wrap-up experiment summary

Study directory: `models/vae_grid_runs/20260430_074356_wrapup_layers_windows_noscale_beta1`

Reported numbers are mean +/- std across seed replicates at the `train_p98` threshold on the test split.

## Best test results by group

| Group | Base experiment | Window | Hidden | Architecture | Seeds | F1 | ROC-AUC | PR-AUC | Precision | Recall |
|---|---|---:|---|---|---:|---:|---:|---:|---:|---:|
| architectures | arch_lstm | 60 | `[128, 64]` | lstm_autoencoder | 1 | 0.922+/-0.000 | 0.995+/-0.000 | 0.861+/-0.000 | 0.903+/-0.000 | 0.942+/-0.000 |
| layers | layers_128_64_32 | 60 | `[128, 64, 32]` | dense | 3 | 0.689+/-0.185 | 0.993+/-0.002 | 0.756+/-0.055 | 0.798+/-0.044 | 0.645+/-0.277 |
| windows | win120 | 120 | `[128, 64]` | dense | 3 | 0.931+/-0.006 | 0.997+/-0.000 | 0.893+/-0.013 | 0.890+/-0.012 | 0.977+/-0.001 |

## All grid runs at train_p98

| Group | Base experiment | Window | Hidden | Architecture | Seeds | F1 | ROC-AUC | PR-AUC |
|---|---|---:|---|---|---:|---:|---:|---:|
| architectures | arch_lstm | 60 | `[128, 64]` | lstm_autoencoder | 1 | 0.922+/-0.000 | 0.995+/-0.000 | 0.861+/-0.000 |
| architectures | arch_conv1d | 60 | `[128, 64]` | conv1d | 2 | 0.248+/-0.013 | 0.996+/-0.001 | 0.805+/-0.048 |
| layers | layers_128_64_32 | 60 | `[128, 64, 32]` | dense | 3 | 0.689+/-0.185 | 0.993+/-0.002 | 0.756+/-0.055 |
| layers | layers_64_32 | 60 | `[64, 32]` | dense | 3 | 0.686+/-0.380 | 0.994+/-0.003 | 0.822+/-0.065 |
| layers | layers_128_64 | 60 | `[128, 64]` | dense | 3 | 0.528+/-0.093 | 0.994+/-0.000 | 0.737+/-0.005 |
| layers | layers_256_128_64 | 60 | `[256, 128, 64]` | dense | 3 | 0.368+/-0.251 | 0.994+/-0.001 | 0.742+/-0.031 |
| layers | layers_256_128 | 60 | `[256, 128]` | dense | 3 | 0.352+/-0.306 | 0.994+/-0.001 | 0.755+/-0.027 |
| windows | win120 | 120 | `[128, 64]` | dense | 3 | 0.931+/-0.006 | 0.997+/-0.000 | 0.893+/-0.013 |
| windows | win60 | 60 | `[128, 64]` | dense | 3 | 0.312+/-0.267 | 0.993+/-0.001 | 0.717+/-0.054 |
| windows | win30 | 30 | `[128, 64]` | dense | 3 | 0.190+/-0.062 | 0.972+/-0.029 | 0.512+/-0.218 |
