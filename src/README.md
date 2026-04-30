# Source layout

Reusable project code grouped by responsibility:

- `data/`        - MetroPT3 loading, failure labeling, chronological
  splitting, sliding-window construction.
- `models/`      - VAE architecture, training loop, scoring, threshold
  helpers, run persistence.
- `baselines/`   - Classical anomaly detection baselines (Isolation
  Forest, PCA reconstruction, OC-SVM, LOF) operating on the same
  flattened-window inputs.
- `evaluation/`  - Binary, ROC-AUC, PR-AUC and anomaly metric helpers.
- `plotting/`    - Shared matplotlib style and figure-saving helpers.

All command-line entry points live in `scripts/`.
