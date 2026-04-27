# Work Log

## 2026-04-23

- Moved the workflow from notebooks into scripts.
- Added raw-data labeling and chronological split generation.
- Added feature/window generation with normal-only training windows.
- Made VAE inference deterministic by using the latent mean at scoring time.

## 2026-04-24

- Built the raw, unscaled 60-step window run used by the main experiment.
- Trained the main dense VAE run: `20260424_140621`.
- Added multi-threshold evaluation for train-percentile and validation-F1
  threshold rules.

## 2026-04-26 to 2026-04-27

- Added an unsupervised Isolation Forest baseline trained on normal windows.
- Added the layer/window comparison grid.
- Cleaned report outputs so the active tables and figures match the final
  unsupervised comparison.
