# VAE Anomaly Detection — MetroPT3

DDA4210 course project. We train a VAE on **normal** air-compressor behavior and use reconstruction error to detect air-leak failures. No failure examples are seen during training.

---

## Setup

```bash
pip install -r requirements.txt
```

Download the [MetroPT3 dataset](https://archive.ics.uci.edu/dataset/791/metropt+3+dataset) and place under `dataset/raw/`:

```
dataset/raw/MetroPT3(AirCompressor).csv
dataset/raw/Data Description_Metro.pdf
```

---

## Data split

| Split | Period | Contains |
|-------|--------|----------|
| Train | Feb–May 2020 | Normal only (F1 windows filtered out) |
| Val | May–Jun 2020 | F2 air-leak |
| Test | Jun–Aug 2020 | F3, F4 air-leaks |

Four labeled failure events: F1 (Apr 18), F2 (May 29–30), F3 (Jun 5–7), F4 (Jul 15).

---

## Running the pipeline

**Step 1 — Preprocess** (label rows, create splits):

```bash
python scripts/preprocess.py --config configs/data/archive_like_window60.json
```

**Step 2 — Build windows** (unscaled, 60-step sliding windows):

```bash
python scripts/build_windows.py --config configs/features/window60_noscale.json
```

**Step 3 — Train + evaluate**:

```bash
python scripts/train_vae.py --config configs/experiments/dense_window60_beta1_noscale_final.json
```

**Evaluate with multiple threshold strategies** on a saved run:

```bash
python scripts/evaluate_thresholds.py --run-dir models/vae_runs/<run_id>
```

**Generate analysis figures** (score distributions, baselines, time-series):

```bash
python scripts/plot_analysis.py
```

Figures are saved as PDF + PNG to `reports/figures/`. Pass `--run-dir`, `--split-dir`, `--windows-dir` to target a different run.

Or run the full pipeline in one command:

```bash
python scripts/run_experiment.py \
  --data-config configs/data/archive_like_window60.json \
  --feature-config configs/features/window60_noscale.json \
  --experiment-config configs/experiments/dense_window60_beta1_noscale_final.json
```

---

## Best results

Model: Dense VAE — hidden `[128, 64]`, latent dim 16, β=1.0, 20 epochs, seed 42, **unscaled** input, window size 60 / stride 10.

| Threshold | Precision | Recall | F1 | ROC-AUC |
|-----------|-----------|--------|----|---------|
| val_f1 | 0.793 | 0.791 | 0.792 | 0.988 |
| **train_p98** | **0.716** | **0.963** | **0.821** | **0.988** |
| Archive best (val_f1) | 0.895 | 0.740 | 0.811 | — |

**train_p98** = 98th percentile of training reconstruction errors (no validation labels used). Equivalent to a 2 % nominal false-alarm rate on training data.

Saved run: `models/vae_runs/20260424_140621` (excluded from git — see `reports/tables/` for metric CSVs).

---

## Repository layout

```
configs/           JSON configs for data splits, features, and experiments
scripts/           Pipeline entry points (preprocess → build_windows → train_vae → evaluate → plot_analysis)
src/               Core modules: preprocessing, feature engineering, VAE model, metrics
reports/figures/   Publication-ready figures (PDF + PNG) from plot_analysis.py
reports/tables/    Saved metric CSVs from evaluated runs
notes/             Project notes and current state
archive/           Old notebooks and historical runs kept for reference
dataset/           Raw data (not committed) and derived splits (not committed)
models/vae_runs/   Training run artifacts (not committed — too large)
```

---

## Key implementation notes

- **No input scaling** — standard scaling hurts anomaly detection because the model reconstructs scaled anomalies as well as normal data. Raw sensor values are used directly.
- **Deterministic scoring** — the VAE sampling layer returns the latent mean (not a sample) at inference time, so scores are deterministic.
- **Normal-only training** — training windows that overlap any failure interval are filtered out before training.
- **Window labeling** — a 60-step window is labeled positive if the last 6 steps (10 %) contain any failure row.
- Runs with β=0 are autoencoder baselines (no KL regularization), not true VAEs.
