from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT / "src"))

from data_preprocessing import DEFAULT_FAILURE_EVENTS, label_split_and_save


def _resolve(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def load_config(path: str | Path) -> dict:
    with _resolve(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def main() -> None:
    parser = argparse.ArgumentParser(description="Label MetroPT3 data and create chronological splits.")
    parser.add_argument("--config", default="configs/data/base.json", help="Path to preprocessing JSON config.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    split = cfg["split"]

    artifacts = label_split_and_save(
        csv_path=_resolve(cfg["raw_csv_path"]),
        preprocessed_root=_resolve(cfg["preprocessed_root"]),
        timestamp_col=cfg.get("timestamp_col", "timestamp"),
        failure_events=cfg.get("failure_events", DEFAULT_FAILURE_EVENTS),
        train_start=split["train_start"],
        train_end=split["train_end"],
        val_start=split["val_start"],
        val_end=split["val_end"],
        test_start=split["test_start"],
        test_end=split.get("test_end"),
        overwrite_source=False,
    )

    print(f"Preprocessing run: {artifacts.run_id}")
    print(f"Labeled CSV: {artifacts.labeled_csv_path}")
    print(f"Split directory: {artifacts.split_dir}")
    print(f"Metadata: {artifacts.metadata_path}")


if __name__ == "__main__":
    main()
