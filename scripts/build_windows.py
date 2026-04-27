from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.preprocessing import find_split_run_by_name
from src.data.windows import FeatureEngineeringParams, engineer_and_save_windows


def _resolve(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def load_config(path: str | Path) -> dict:
    with _resolve(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def _find_split_run(preprocessed_root: Path, split_run_name: str) -> Path:
    if split_run_name == "latest":
        split_root = preprocessed_root / "splits"
        runs = sorted(p for p in split_root.iterdir() if p.is_dir())
        if not runs:
            raise FileNotFoundError(f"No split runs found in {split_root}")
        return runs[-1]
    return find_split_run_by_name(preprocessed_root, split_run_name)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build scaled MetroPT3 point/window arrays.")
    parser.add_argument("--config", default="configs/features/point.json", help="Path to feature JSON config.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    preprocessed_root = _resolve(cfg["preprocessed_root"])
    split_dir = _find_split_run(preprocessed_root, cfg.get("split_run_name", "latest"))

    point_mode = bool(cfg.get("point_mode", False))
    window_size = int(cfg.get("window_size", 1 if point_mode else 60))
    stride = int(cfg.get("stride", 1 if point_mode else 10))

    params = FeatureEngineeringParams(
        train_csv_path=str(split_dir / "train_split.csv"),
        val_csv_path=str(split_dir / "val_split.csv"),
        test_csv_path=str(split_dir / "test_split.csv"),
        timestamp_col=cfg.get("timestamp_col", "timestamp"),
        label_col=cfg.get("label_col", "failure_label"),
        train_normal_only=bool(cfg.get("train_normal_only", True)),
        feature_cols=tuple(cfg.get("feature_cols", [])),
        window_size=window_size,
        stride=stride,
        flatten_windows=bool(cfg.get("flatten_windows", True)),
        window_label_strategy=cfg.get("window_label_strategy", "positive_ratio"),
        window_label_positive_ratio=float(cfg.get("window_label_positive_ratio", 0.1)),
        window_label_last_percent=float(cfg.get("window_label_last_percent", 10.0)),
        point_mode=point_mode,
        scale_features=bool(cfg.get("scale_features", True)),
        scaler_type=cfg.get("scaler_type", "standard"),
    )

    artifacts = engineer_and_save_windows(params, _resolve(cfg["output_root"]))

    print(f"Feature run: {artifacts.run_dir.name}")
    print(f"Run directory: {artifacts.run_dir}")
    print(f"Metadata: {artifacts.metadata_path}")
    print(f"Scaler metadata: {artifacts.scaler_metadata_path}")


if __name__ == "__main__":
    main()
