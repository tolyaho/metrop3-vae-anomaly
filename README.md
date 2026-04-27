# VAE Anomaly Detection on MetroPT3

DDA4210 course research project. The goal is to detect MetroPT3 air-compressor
air-leak failures by training a reconstruction model on normal operating windows
only. At inference time, reconstruction error is used as the anomaly score.

The final comparison uses a fair classical unsupervised baseline
(Isolation Forest) and a VAE experiment grid over layer sizes and window sizes.

## Project Claim

A VAE trained on normal compressor behavior can flag later air-leak failures
without seeing failure examples during model training. The main fair threshold is
`train_p98`, the 98th percentile of training anomaly scores.

Validation-label thresholds such as `val_f1` are saved for diagnostics only.

## Setup

```bash
pip install -r requirements.txt
```

Download the UCI MetroPT3 dataset and place it under `dataset/raw/`:

```text
dataset/raw/MetroPT3(AirCompressor).csv
dataset/raw/Data Description_Metro.pdf
```

Raw and processed data are intentionally ignored by Git. See
[dataset/README.md](dataset/README.md).

## Data Split

| Split | Period | Purpose |
|---|---|---|
| Train | Feb-Apr 2020 | Normal-only VAE/IF fitting; F1-overlapping windows filtered |
| Validation | May 2020 | F2 labels for diagnostics and threshold analysis |
| Test | Jun-Jul 2020 | Held-out F3/F4 failure evaluation |

Failure intervals: F1 (Apr 18), F2 (May 29-30), F3 (Jun 5-7), F4 (Jul 15).

## Main Pipeline

Preprocess:

```bash
python scripts/preprocess.py --config configs/data/archive_like_window60.json
```

Build raw, unscaled windows:

```bash
python scripts/build_windows.py --config configs/features/window60_noscale.json
```

Train the main VAE:

```bash
python scripts/train_vae.py --config configs/experiments/dense_window60_beta1_noscale_final.json
```

Evaluate additional thresholds:

```bash
python scripts/evaluate_thresholds.py --run-dir models/vae_runs/<run_id>
```

Run the full main pipeline:

```bash
python scripts/run_experiment.py \
  --data-config configs/data/archive_like_window60.json \
  --feature-config configs/features/window60_noscale.json \
  --experiment-config configs/experiments/dense_window60_beta1_noscale_final.json
```

## Fair Baseline

The final classical baseline is Isolation Forest trained only on the same
normal-only training windows as the VAE.

```bash
python scripts/train_isolation_forest.py \
  --config configs/baselines/isolation_forest_window60_noscale.json

python scripts/compare_baselines.py \
  --vae-run-dir models/vae_runs/20260424_140621 \
  --if-run-dir models/baseline_runs/isolation_forest/<run_id>
```

Primary fair comparison at `train_p98`:

| Model | Precision | Recall | F1 | ROC-AUC | PR-AUC |
|---|---:|---:|---:|---:|---:|
| VAE `20260424_140621` | 0.716 | 0.963 | 0.821 | 0.988 | 0.762 |
| Isolation Forest | 0.496 | 0.994 | 0.661 | 0.996 | 0.836 |

The VAE has higher F1 at the train-only threshold; Isolation Forest reaches high
recall by producing many more false positives.

## Wrap-Up Experiment Grid

Run the layer-size and window-size grid:

```bash
python scripts/run_experiment_grid.py \
  --config configs/experiments/wrapup_grid_layers_windows.json

python scripts/collect_experiment_results.py \
  --study-dir models/vae_grid_runs/<study_id>

python scripts/plot_experiment_comparison.py \
  --study-dir models/vae_grid_runs/<study_id>
```

Latest fresh grid result at `train_p98`:

| Study | Best setting | Precision | Recall | F1 | ROC-AUC | PR-AUC |
|---|---|---:|---:|---:|---:|---:|
| Layer comparison | hidden `[128, 64, 32]`, window 60 | 0.903 | 0.863 | 0.883 | 0.996 | 0.859 |
| Window comparison | window 120, hidden `[128, 64]` | 0.353 | 0.100 | 0.156 | 0.990 | 0.638 |

See:

- [reports/wrapup_experiment_summary.md](reports/wrapup_experiment_summary.md)
- [reports/tables/wrapup_grid_results.csv](reports/tables/wrapup_grid_results.csv)
- [reports/figures/final_model_comparison.png](reports/figures/final_model_comparison.png)

## Repository Layout

```text
configs/      JSON configs for data, features, experiments, and baselines
dataset/      canonical local MetroPT3 raw/processed data path; files ignored
data/         conventional data placeholder; files ignored
docs/         methodology, reproducibility, and artifact policy
scripts/      runnable CLI pipeline entry points
src/          reusable package code
  data/       labeling, splitting, and window construction
  models/     VAE model, training, scoring, and run persistence helpers
  evaluation/ metrics shared by VAE and baselines
reports/      lightweight report tables and figures
models/       local generated model runs; ignored except placeholders/docs
notebooks/    optional exploratory/report notebook placeholders
archive/      legacy notebooks and notes kept for reference
notes/        current project notes
```

## Key Implementation Notes

- Raw unscaled windows are the main experimental input.
- The VAE sampling layer returns the latent mean at inference time, so scoring is
  deterministic.
- Training windows overlapping failure intervals are filtered out.
- A 60-step window is positive if the final 10 percent contains any failure row.
- Runs with `beta=0` are autoencoder baselines, not true VAEs.
- Full model runs, raw data, processed arrays, and local environments are
  ignored by Git. See [docs/artifact_policy.md](docs/artifact_policy.md).
