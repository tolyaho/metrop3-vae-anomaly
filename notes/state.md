# Project State

Last updated: 2026-04-27

## Main Run

`models/vae_runs/20260424_140621`

Dense VAE, hidden `[128, 64]`, latent dimension 16, beta 1.0, 20 epochs,
seed 42, raw unscaled windows, window size 60, stride 10.

| Threshold | Precision | Recall | F1 | ROC-AUC |
|---|---:|---:|---:|---:|
| `train_p98` | 0.716 | 0.963 | 0.821 | 0.988 |
| `val_f1` | 0.793 | 0.791 | 0.792 | 0.988 |

`train_p98` is the main threshold because it is computed only from training
scores. `val_f1` is kept for diagnostic comparison.

## Active Data Artifacts

These are local paths and are ignored by Git:

- Split: `dataset/preprocessed/splits/20260424_005227`
- Windows: `dataset/processed_windows/20260424_135957`

Window counts:

| Split | Windows | Positive windows |
|---|---:|---:|
| Train | 63,526 | 0 |
| Validation | 21,275 | 237 |
| Test | 43,910 | 1,894 |

## Report Outputs

- `reports/tables/20260424_140621_threshold_comparison.csv`
- `reports/tables/baselines/baseline_comparison_primary_train_p98.csv`
- `reports/tables/wrapup_grid_results.csv`
- `reports/wrapup_experiment_summary.md`
