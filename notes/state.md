# Project state

Last updated: 2026-04-30

## Headline run

Multi-seed grid study under
`models/vae_grid_runs/20260430_074356_wrapup_layers_windows_noscale_beta1`
(study id is timestamped at run time; the prefix may differ on a fresh
reproduction).

Best configuration (`base_experiment_id=win120`): dense VAE,
`hidden=[128, 64]`, `latent_dim=16`, `beta=1`, `window_size=120`,
`stride=20`, 30 epochs max with collapse-guarded early stopping, three
seeds (42, 7, 123), raw (unscaled) sensor windows.

Test split, `train_p98` threshold (mean +/- std across seeds):

| Metric        | Value             |
|---|---:|
| F1            | 0.931 +/- 0.006   |
| Precision     | 0.890 +/- 0.012   |
| Recall        | 0.977 +/- 0.001   |
| ROC-AUC       | 0.997 +/- 0.000   |
| PR-AUC        | 0.893 +/- 0.013   |

Per-seed run directories (each contains `summary.json`, `*_scores.npy`,
`*_labels.npy`, `encoder.keras`, `decoder.keras`):

```
models/vae_grid_runs/20260430_074356_wrapup_layers_windows_noscale_beta1/runs/windows/win120_s42/20260430_081342
models/vae_grid_runs/20260430_074356_wrapup_layers_windows_noscale_beta1/runs/windows/win120_s7/20260430_081437
models/vae_grid_runs/20260430_074356_wrapup_layers_windows_noscale_beta1/runs/windows/win120_s123/20260430_081528
```

## Active data artifacts (gitignored)

| Artefact | Path |
|---|---|
| Splits         | `dataset/preprocessed/splits/20260430_061958` |
| Windows (60)   | `dataset/processed_windows/20260430_062154` |
| Windows (30)   | `dataset/processed_windows/20260430_062216` |
| Windows (120)  | `dataset/processed_windows/20260430_062218` |

Window counts at `window_size=60, stride=10` (the size used by the
classical baselines and most ablation runs):

| Split      | Windows | Positive windows |
|---|---:|---:|
| Train      | 63,526  |   0    |
| Validation | 21,275  | 237    |
| Test       | 43,910  | 1,894  |

## Active baseline runs

| Model              | Run dir |
|---|---|
| Isolation Forest    | `models/baseline_runs/isolation_forest/20260430_063747` |
| PCA reconstruction  | `models/baseline_runs/pca_recon/20260430_063858`        |
| OC-SVM              | `models/baseline_runs/oc_svm/20260430_063918`           |
| LOF                 | `models/baseline_runs/lof/20260430_063943`              |

All four use the `window_size=60` flattened normal-only training
windows, identical to the inputs the dense VAE sees on `win60`.

## Report outputs

- [`reports/final_report.md`](../reports/final_report.md) - the writeup.
- [`reports/grid_summary.md`](../reports/grid_summary.md) - grid summary.
- [`reports/tables/grid_aggregated.csv`](../reports/tables/grid_aggregated.csv).
- [`reports/tables/grid_results.csv`](../reports/tables/grid_results.csv) - per-seed rows.
- [`reports/tables/baselines/baseline_comparison_primary_train_p98.csv`](../reports/tables/baselines/baseline_comparison_primary_train_p98.csv).
- [`reports/tables/threshold_sensitivity.csv`](../reports/tables/threshold_sensitivity.csv).
- [`reports/tables/failure_event_metrics.csv`](../reports/tables/failure_event_metrics.csv).
- [`reports/figures/models_overlay_test.png`](../reports/figures/models_overlay_test.png).
- [`reports/figures/threshold_sensitivity.png`](../reports/figures/threshold_sensitivity.png).
- [`reports/figures/failure_events/failure_events_timeline.png`](../reports/figures/failure_events/failure_events_timeline.png).
- [`reports/figures/failure_latency.png`](../reports/figures/failure_latency.png).
- Layer / window / architecture comparison bars under [`reports/figures/`](../reports/figures).

## Notes

- Earlier single-seed VAE numbers (e.g. `models/vae_runs/20260424_140621`,
  F1 ~ 0.82 / 0.88) are superseded by the multi-seed grid above and
  shouldn't be quoted.
- `enable_op_determinism` is off by default. Setting `VAE_DETERMINISM=1`
  reintroduces the cuDNN-deterministic interaction that triggers the
  score-direction inversion documented in `docs/methodology.md`.
