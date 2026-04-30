# MetroPT3 VAE anomaly detection

Unsupervised anomaly detection on the UCI MetroPT3 air-compressor dataset.
A Variational Autoencoder is trained on a chronological prefix that contains
only normal operation, and reconstruction error is used as the anomaly score
for the four annotated air-leak events.

## Dataset

MetroPT3 has one-minute readings from 15 sensors on an industrial air
compressor. The four labelled failure intervals are:

| Failure | Period |
|---|---|
| F1 | 2020-04-18 |
| F2 | 2020-05-29 to 2020-05-30 |
| F3 | 2020-06-05 to 2020-06-07 |
| F4 | 2020-07-15 |

Raw data is not committed. Drop `MetroPT3(AirCompressor).csv` and the
description PDF into `dataset/raw/`.

## Problem setup

The split is chronological:

| Split | Period | Use |
|---|---|---|
| Train      | Feb-Apr 2020 | Fit the VAE and the classical baselines on normal-only windows. |
| Validation | May 2020     | Early stopping, threshold diagnostics. |
| Test       | Jun-Jul 2020 | Held-out evaluation on F3 and F4. |

Sliding windows over the 15 channels are labelled positive when the last
10 percent of the window falls inside any failure interval. Labels are
not used during VAE training; they are only used at evaluation time
(and, separately, for the diagnostic `val_f1` threshold).

## Method

The headline model is a dense VAE:

```text
window x 15 channels  ->  Dense(128) -> Dense(64)  ->  z in R^16  ->  mirrored decoder
```

Anomaly score is per-window MSE, evaluated at the latent mean so that
scoring is deterministic. Loss is `MSE + beta * KL` with `beta = 1`.

Two practical things matter on this dataset:

- **Seeded training.** Python, NumPy and TensorFlow are all seeded.
  TF op-determinism is opt-in via `VAE_DETERMINISM=1` because the
  deterministic cuDNN kernels interact badly with weight init here and
  push the model toward the collapse behaviour described below.
- **Collapse-guarded early stopping.** The VAE on this data has an
  attractor where, after ~20 epochs, it learns to reconstruct anomalies
  *better* than normals and the score direction silently inverts.
  Training watches `val_total_loss` for normal patience-based stopping
  but also tracks the best `val_roc_auc`, and if the AUC peaks above
  0.7 and then drops below 0.5 the best-AUC weights are restored.

Thresholds:

- `train_pXX` (98 / 99 / 99.5): the X-th percentile of training scores.
  Fully unsupervised. `train_p98` is the headline.
- `val_f1`: threshold maximising validation F1. Reported only as a
  diagnostic upper bound because it uses validation labels.

Classical baselines, all fed the same normal-only flattened windows:
Isolation Forest, PCA reconstruction, One-Class SVM (Nystroem + SGD)
and Local Outlier Factor.

## Results

Test set, threshold = `train_p98`. The VAE numbers are mean +/- std
across three random seeds (42, 7, 123).

| Model | F1 | Precision | Recall | ROC-AUC | PR-AUC |
|---|---:|---:|---:|---:|---:|
| **VAE (dense [128, 64], window=120)** | **0.931 +/- 0.006** | **0.890 +/- 0.012** | **0.977 +/- 0.001** | **0.997 +/- 0.000** | **0.893 +/- 0.013** |
| Isolation Forest (window=60, 500 trees) | 0.661 | 0.496 | 0.994 | 0.996 | 0.836 |
| OC-SVM (Nystroem + SGD, nu=0.05) | 0.037 | 0.030 | 0.048 | 0.913 | 0.197 |
| PCA reconstruction (95% variance) | 0.013 | 0.010 | 0.019 | 0.796 | 0.094 |
| LOF (k=35, 20k normal subsample) | 0.008 | 0.006 | 0.011 | 0.740 | 0.072 |

The VAE wins F1 and PR-AUC by a wide margin while matching IF on
ROC-AUC. On the test events themselves the VAE first-alarms F3 at 10.1
minutes and F4 at 3.0 minutes; IF first-alarms at 0.2 / 4.6 minutes.
PCA, OC-SVM and LOF miss F3 entirely under the unsupervised threshold.

The full discussion is in [`reports/final_report.md`](reports/final_report.md).
Other useful artefacts:

- [Grid summary](reports/grid_summary.md) - layer / window / architecture sweep.
- [`reports/figures/models_overlay_test.png`](reports/figures/models_overlay_test.png) - cross-model PR + ROC overlay.
- [`reports/figures/threshold_sensitivity.png`](reports/figures/threshold_sensitivity.png) - threshold robustness.
- [`reports/figures/failure_events/failure_events_timeline.png`](reports/figures/failure_events/failure_events_timeline.png) and [`reports/figures/failure_latency.png`](reports/figures/failure_latency.png) - per-failure view.
- [`reports/tables/grid_aggregated.csv`](reports/tables/grid_aggregated.csv) - mean +/- std across seeds.
- [`reports/tables/baselines/baseline_comparison_primary_train_p98.csv`](reports/tables/baselines/baseline_comparison_primary_train_p98.csv).

## Setup

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
```

If the network can't resolve `archive.ics.uci.edu`, see
[`docs/reproducibility.md`](docs/reproducibility.md) for the
`curl --resolve` workaround.

## Pipeline

```bash
# Preprocess + windowing for all reported window sizes.
python scripts/preprocess.py    --config configs/data/metropt3.json
python scripts/build_windows.py --config configs/features/window30_noscale.json
python scripts/build_windows.py --config configs/features/window60_noscale.json
python scripts/build_windows.py --config configs/features/window120_noscale.json

# Multi-seed VAE grid (layers x window-sizes x architectures, 3 seeds).
python scripts/run_experiment_grid.py        --config configs/experiments/vae_grid.json
python scripts/collect_experiment_results.py --study-dir models/vae_grid_runs/<study_id>
python scripts/plot_experiment_comparison.py --study-dir models/vae_grid_runs/<study_id>

# Four classical baselines, all fit on normal-only window60 features.
python scripts/train_classical_baseline.py --config configs/baselines/isolation_forest_window60_noscale.json
python scripts/train_classical_baseline.py --config configs/baselines/pca_recon_window60_noscale.json
python scripts/train_classical_baseline.py --config configs/baselines/oc_svm_window60_noscale.json
python scripts/train_classical_baseline.py --config configs/baselines/lof_window60_noscale.json

# Comparison table + cross-model figures + per-failure analysis.
python scripts/build_canonical_comparison.py  --vae-base-id win120
python scripts/plot_models_overlay.py         --vae-grid-dir models/vae_grid_runs/<study_id> --vae-grid-base-id win120
python scripts/plot_threshold_sensitivity.py  --vae-run-dir models/vae_grid_runs/<study_id>/runs/windows/win120_s42/<run_ts>
python scripts/analyze_failure_events.py --model \
    "VAE (dense)":vae:<best_vae_run_dir> \
    "Isolation Forest":baseline:<latest_if_run_dir> \
    "PCA reconstruction":baseline:<latest_pca_run_dir> \
    "OC-SVM":baseline:<latest_oc_svm_run_dir> \
    "LOF":baseline:<latest_lof_run_dir>
python scripts/plot_vae_latent_space.py --run-dir <best_vae_run_dir> --split test
```

A single VAE training (no grid) can be launched with the example config
`configs/experiments/vae_dense_window60.json` via `scripts/train_vae.py`.

## Repository layout

```text
configs/   JSON configs for data, features, experiments and baselines
dataset/   raw and derived MetroPT3 data (gitignored)
docs/      methodology, reproducibility, artifact policy
scripts/   command-line pipeline entry points
src/       reusable data / model / baseline / evaluation / plotting code
reports/   committed tables and figures
models/    run outputs (gitignored)
notes/     project log and current state
```

## Notes and limitations

- All headline numbers come from one grid study and one sweep of four
  baselines. Earlier single-seed results in the git history are
  superseded.
- The VAE uses raw, unscaled windows. Standard-scaling globally weakened
  separation here; the baselines that need scaling apply it internally,
  fit only on normal training windows.
- Failure labels are intervals turned into window labels with the
  "last 10 percent" rule. All metrics are window-level.
- `val_f1` looks at validation labels and is reported only as a
  diagnostic, never as the deployed threshold.
- Heavy run artefacts are gitignored. See [`docs/artifact_policy.md`](docs/artifact_policy.md).
