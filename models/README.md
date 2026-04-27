# Models

Model run directories are generated artifacts and are ignored by Git:

- `vae_runs/`
- `baseline_runs/`
- `vae_grid_runs/`

Each run saves configs, metrics, scores, predictions, and model files locally.
Commit lightweight summaries from `reports/` instead of committing full runs.
