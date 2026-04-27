# Legacy Notebooks

These notebooks were the original execution workflow for preprocessing, feature
engineering, and VAE training.

Their main logic has been migrated to:

- `scripts/preprocess.py`
- `scripts/build_windows.py`
- `scripts/train_vae.py`
- `scripts/evaluate_run.py`

The notebooks are kept for local reference only and may contain stale absolute
paths or historical outputs. Notebook files in this folder are intentionally
ignored by Git; keep reusable logic in `scripts/` and `src/`.
