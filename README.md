# VAE Anomaly Detection on MetroPT3

Course project for unsupervised anomaly detection on the UCI MetroPT3 air
compressor dataset. The model is trained on normal operating windows and scores
later windows by reconstruction error. High reconstruction error is treated as
evidence of an air-leak event.

The code is script-first. Notebooks are kept only for exploration.

## Dataset

MetroPT3 contains one-minute sensor readings from an industrial air compressor.
This repository uses the 15 sensor channels and the four labeled air-leak
intervals supplied with the dataset:

| Failure | Period |
|---|---|
| F1 | 2020-04-18 |
| F2 | 2020-05-29 to 2020-05-30 |
| F3 | 2020-06-05 to 2020-06-07 |
| F4 | 2020-07-15 |

Raw data is not committed. Place the downloaded files under `dataset/raw/`.

## Problem Setup

The main split is chronological:

| Split | Period | Use |
|---|---|---|
| Train | Feb-Apr 2020 | fit VAE and Isolation Forest on normal windows only |
| Validation | May 2020 | diagnostics and optional label-tuned thresholds |
| Test | Jun-Jul 2020 | held-out evaluation on F3/F4 |

Sensor rows are converted into sliding windows. The main setting uses
`window_size=60` and `stride=10`. A window is labeled anomalous if the final
10 percent of the window overlaps a labeled failure interval. The model does not
use these labels during VAE training.

My main contribution in this project was the evaluation setup: failure interval
labeling, normal-only window filtering, train/validation/test splitting,
threshold rules, and rare-event metric reporting.

## Method

The primary model is a dense VAE:

```text
60 x 15 raw window -> Dense(128) -> Dense(64) -> latent_dim=16 -> reconstruction
```

Training uses normal windows only. At inference time, the sampling layer returns
the latent mean, so repeated scoring is deterministic. The anomaly score is the
mean squared reconstruction error.

Thresholds:

- `train_p98`: 98th percentile of training anomaly scores. This is the main
  train-only threshold used for fair comparisons.
- `val_f1`: threshold selected with validation labels. This is useful for
  diagnostics, but it is not treated as the main unsupervised result.

The final classical baseline is Isolation Forest, also trained only on normal
training windows.

## Results

Primary test-set comparison at `train_p98`:

| Model | Precision | Recall | F1 | ROC-AUC | PR-AUC |
|---|---:|---:|---:|---:|---:|
| VAE `20260424_140621` | 0.716 | 0.963 | 0.821 | 0.988 | 0.762 |
| Isolation Forest `20260426_200420` | 0.496 | 0.994 | 0.661 | 0.996 | 0.836 |

The Isolation Forest has slightly higher ranking metrics, but at the fixed
train-only threshold it produces many more false positives. The VAE gives the
better F1 score under the same threshold rule.

The wrap-up grid also retrains VAE variants over layer sizes and window sizes.
The best fresh grid run was `layers_128_64_32`:

| Experiment | Precision | Recall | F1 | ROC-AUC | PR-AUC |
|---|---:|---:|---:|---:|---:|
| hidden `[128, 64, 32]`, window 60 | 0.903 | 0.863 | 0.883 | 0.996 | 0.859 |

Useful report artifacts:

- [Threshold comparison](reports/tables/20260424_140621_threshold_comparison.csv)
- [Baseline comparison table](reports/tables/baselines/baseline_comparison_primary_train_p98.csv)
- [Grid results](reports/tables/wrapup_grid_results.csv)
- [Final comparison plot](reports/figures/final_model_comparison.png)
- [Wrap-up summary](reports/wrapup_experiment_summary.md)

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Expected local data files:

```text
dataset/raw/MetroPT3(AirCompressor).csv
dataset/raw/Data Description_Metro.pdf
```

## Run the Pipeline

Run the main preprocessing, window-building, and VAE training steps:

```bash
python scripts/preprocess.py --config configs/data/archive_like_window60.json
python scripts/build_windows.py --config configs/features/window60_noscale.json
python scripts/train_vae.py --config configs/experiments/dense_window60_beta1_noscale_final.json
python scripts/evaluate.py --run-dir models/vae_runs/<run_id>
```

The combined pipeline uses the same default configs:

```bash
python scripts/run_experiment.py
```

Run the Isolation Forest baseline:

```bash
python scripts/train_isolation_forest.py \
  --config configs/baselines/isolation_forest_window60_noscale.json

python scripts/compare_baselines.py \
  --vae-run-dir models/vae_runs/20260424_140621 \
  --if-run-dir models/baseline_runs/isolation_forest/<run_id>
```

Run the layer/window grid:

```bash
python scripts/run_experiment_grid.py \
  --config configs/experiments/wrapup_grid_layers_windows.json

python scripts/collect_experiment_results.py \
  --study-dir models/vae_grid_runs/<study_id>

python scripts/plot_experiment_comparison.py \
  --study-dir models/vae_grid_runs/<study_id>
```

## Repository Layout

```text
configs/      JSON configs for data, features, models, baselines, and grids
dataset/      local raw/processed MetroPT3 data; ignored except README
docs/         methodology and reproducibility notes
scripts/      command-line pipeline entry points
src/          reusable data, model, and evaluation code
reports/      lightweight tables and plots used in the report
models/       local model outputs; ignored except placeholders
notebooks/    optional exploration notebooks
archive/      old notebooks, supervised-baseline script, and local artifacts
```

## Notes and Limitations

- The main experiments use raw, unscaled windows. Earlier standard-scaling runs
  are kept out of the main results because they weakened anomaly separation.
- Failure labels are interval labels, then converted to window labels. Metrics
  are reported at the window level, not at the original row level.
- `val_f1` uses validation labels and should not be described as a fully
  unsupervised threshold.
- Full model runs, raw data, processed arrays, and local notebook outputs are
  ignored by Git. See [docs/artifact_policy.md](docs/artifact_policy.md).
