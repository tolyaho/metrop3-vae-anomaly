# Reports

Lightweight outputs committed to git. Full model weights, score arrays
and predictions live under `models/` and are gitignored.

## Top level

- [`final_report.md`](final_report.md) - the writeup, with numbers,
  figures and discussion.
- [`grid_summary.md`](grid_summary.md) - compact summary of the
  multi-seed grid (layers x windows x architectures).

## Tables (`tables/`)

| File | Source | Description |
|---|---|---|
| `grid_aggregated.csv`     | `scripts/collect_experiment_results.py` | Mean +/- std across seeds for every grid configuration. |
| `grid_results.csv`        | `scripts/collect_experiment_results.py` | Per-seed grid rows. |
| `grid_best_by_group.csv`  | `scripts/collect_experiment_results.py` | Best aggregated row per experiment group. |
| `baselines/baseline_comparison_primary_train_p98.csv` | `scripts/build_canonical_comparison.py` | Headline VAE vs each of the four classical baselines. |
| `baselines/baseline_comparison_primary_val_f1.csv`    | `scripts/build_canonical_comparison.py` | Same comparison under the diagnostic label-tuned threshold. |
| `baselines/<kind>/...`    | `scripts/train_classical_baseline.py`   | Per-baseline metrics, predictions and threshold tables. |
| `models_overlay_test.csv` | `scripts/plot_models_overlay.py`        | ROC / PR area-under-curve summary across all models. |
| `threshold_sensitivity.csv` | `scripts/plot_threshold_sensitivity.py` | F1 / precision / recall vs train-score percentile sweep. |
| `failure_event_metrics.csv` | `scripts/analyze_failure_events.py`     | Latency-to-first-alarm and per-event PR per model. |

## Figures (`figures/`)

| File | Source |
|---|---|
| `models_overlay_test.{png,pdf}`                           | `scripts/plot_models_overlay.py` |
| `threshold_sensitivity.{png,pdf}`                         | `scripts/plot_threshold_sensitivity.py` |
| `failure_latency.{png,pdf}` and `failure_events/*.{png,pdf}` | `scripts/analyze_failure_events.py` |
| `vae_latent_space_*_pca.png`, `vae_latent_space_*_tsne.png` | `scripts/plot_vae_latent_space.py` |
| `layer_comparison_*`, `window_comparison_*`, `architecture_comparison_*` | `scripts/plot_experiment_comparison.py` |
| `baselines/<kind>/...`                                    | `scripts/train_classical_baseline.py` |

All matplotlib figures use the shared style in `src/plotting/style.py`
(300 dpi, consistent palette and per-model colour map).
