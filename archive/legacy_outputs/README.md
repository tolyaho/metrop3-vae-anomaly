# Legacy Outputs

This folder contains old model/output artifacts kept for reference.

The archived `vae_runs/` folders include useful historical metrics, but their
summaries reference absolute paths from an earlier machine. They are ignored by
Git and should stay local unless a lightweight summary is intentionally copied
to `reports/`.

The `report_figures/` folder contains older supervised-baseline figures moved
out of active `reports/figures/` so the main report outputs reflect the final
fair unsupervised comparison.

New report-ready runs should be produced through the script pipeline and saved
under `models/vae_runs/`.
