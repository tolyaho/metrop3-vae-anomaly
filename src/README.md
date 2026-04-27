# Source Layout

Reusable project code is grouped by responsibility:

- `data/`: MetroPT3 loading, failure labeling, chronological splitting, and
  window construction.
- `models/`: VAE architecture, training loop, scoring, threshold helpers, and
  run persistence.
- `evaluation/`: shared binary, ROC-AUC, PR-AUC, and anomaly metric helpers.
- `training/`, `visualization/`, `utils/`: package placeholders for reusable
  logic if script-only code later needs to be promoted.

Top-level modules such as `metrics.py` and `vae_anomaly_detector.py` are kept as
thin compatibility imports so older scripts and notes continue to work.
