from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _run(args: list[str]) -> None:
    print(" ".join(args))
    subprocess.run(args, cwd=PROJECT_ROOT, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the full MetroPT3 VAE pipeline from configs.")
    parser.add_argument("--data-config", default="configs/data/base.json")
    parser.add_argument("--feature-config", default="configs/features/point.json")
    parser.add_argument("--experiment-config", default="configs/experiments/dense_point_beta0.json")
    parser.add_argument("--skip-preprocess", action="store_true")
    parser.add_argument("--skip-windows", action="store_true")
    args = parser.parse_args()

    python = sys.executable
    if not args.skip_preprocess:
        _run([python, "scripts/preprocess.py", "--config", args.data_config])
    if not args.skip_windows:
        _run([python, "scripts/build_windows.py", "--config", args.feature_config])
    _run([python, "scripts/train_vae.py", "--config", args.experiment_config])


if __name__ == "__main__":
    main()
