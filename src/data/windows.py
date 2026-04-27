from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import json
import numpy as np
import pandas as pd

from .preprocessing import default_feature_cols, load_metropt3


@dataclass
class FeatureEngineeringParams:
    train_csv_path: str
    val_csv_path: str
    test_csv_path: str
    timestamp_col: str = "timestamp"
    label_col: str = "failure_label"
    train_normal_only: bool = True
    feature_cols: tuple[str, ...] = ()
    window_size: int = 60
    stride: int = 10
    flatten_windows: bool = True
    window_label_strategy: str = "positive_ratio"  # positive_ratio | last
    window_label_positive_ratio: float = 0.1
    window_label_last_percent: float = 10.0
    point_mode: bool = False
    scale_features: bool = True
    scaler_type: str = "standard"


@dataclass
class FeatureEngineeringArtifacts:
    run_dir: Path
    train_windows_path: Path
    val_windows_path: Path
    test_windows_path: Path
    train_window_labels_path: Path
    val_window_labels_path: Path
    test_window_labels_path: Path
    scaler_npz_path: Path
    scaler_metadata_path: Path
    metadata_path: Path


def _serialize_path(path: str | Path) -> str:
    path = Path(path)
    try:
        return str(path.resolve().relative_to(Path.cwd().resolve()))
    except Exception:
        return str(path)


def _serialize_params(params: FeatureEngineeringParams) -> dict[str, object]:
    out = asdict(params)
    for key in ("train_csv_path", "val_csv_path", "test_csv_path"):
        out[key] = _serialize_path(out[key])
    return out


def _window_shape(n_rows: int, n_features: int, window_size: int, stride: int, flatten: bool) -> tuple[int, ...]:
    if n_rows < window_size:
        return (0, window_size * n_features) if flatten else (0, window_size, n_features)
    n_windows = ((n_rows - window_size) // stride) + 1
    return (n_windows, window_size * n_features) if flatten else (n_windows, window_size, n_features)


def _fit_standard_scaler(data: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if len(data) == 0:
        raise ValueError("Cannot fit scaler on empty training data.")
    mean = np.nanmean(data, axis=0).astype(np.float32)
    scale = np.nanstd(data, axis=0).astype(np.float32)
    scale = np.where(scale < 1e-8, 1.0, scale).astype(np.float32)
    return mean, scale


def _apply_standard_scaler(data: np.ndarray, mean: np.ndarray, scale: np.ndarray) -> np.ndarray:
    return ((data - mean) / scale).astype(np.float32)


def _fit_and_apply_scaler(
    X_train: np.ndarray,
    X_val: np.ndarray,
    X_test: np.ndarray,
    train_row_labels: np.ndarray,
    params: FeatureEngineeringParams,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, object], dict[str, np.ndarray]]:
    if not params.scale_features:
        metadata = {
            "enabled": False,
            "type": None,
            "fit_scope": None,
        }
        return X_train, X_val, X_test, metadata, {}

    if params.scaler_type != "standard":
        raise ValueError(f"Unsupported scaler_type={params.scaler_type!r}. Supported: 'standard'.")

    if params.train_normal_only and len(train_row_labels) == len(X_train):
        fit_mask = train_row_labels == 0
        fit_data = X_train[fit_mask]
        fit_scope = "train_rows_with_label_0"
    else:
        fit_data = X_train
        fit_scope = "all_train_rows"

    mean, scale = _fit_standard_scaler(fit_data)
    metadata = {
        "enabled": True,
        "type": "standard",
        "fit_scope": fit_scope,
        "fit_rows": int(len(fit_data)),
        "zero_scale_replaced_with": 1.0,
    }
    scaler_arrays = {"mean": mean, "scale": scale}
    return (
        _apply_standard_scaler(X_train, mean, scale),
        _apply_standard_scaler(X_val, mean, scale),
        _apply_standard_scaler(X_test, mean, scale),
        metadata,
        scaler_arrays,
    )


def _write_windows_npy(
    data: np.ndarray,
    out_path: Path,
    window_size: int,
    stride: int,
    flatten: bool,
) -> tuple[int, ...]:
    n_rows, n_features = data.shape
    shape = _window_shape(n_rows, n_features, window_size, stride, flatten)

    mem = np.lib.format.open_memmap(
        out_path,
        mode="w+",
        dtype=np.float32,
        shape=shape,
    )

    if shape[0] == 0:
        del mem
        return shape

    for i, s in enumerate(range(0, n_rows - window_size + 1, stride)):
        w = data[s : s + window_size]
        mem[i] = w.reshape(-1) if flatten else w

    mem.flush()
    del mem
    return shape


def _write_windows_npy_from_starts(
    data: np.ndarray,
    out_path: Path,
    window_size: int,
    flatten: bool,
    starts: np.ndarray,
) -> tuple[int, ...]:
    n_rows, n_features = data.shape
    if n_rows < window_size or len(starts) == 0:
        shape = (0, window_size * n_features) if flatten else (0, window_size, n_features)
        mem = np.lib.format.open_memmap(out_path, mode="w+", dtype=np.float32, shape=shape)
        del mem
        return shape

    shape = (len(starts), window_size * n_features) if flatten else (len(starts), window_size, n_features)
    mem = np.lib.format.open_memmap(out_path, mode="w+", dtype=np.float32, shape=shape)

    for i, s in enumerate(starts):
        w = data[s : s + window_size]
        mem[i] = w.reshape(-1) if flatten else w

    mem.flush()
    del mem
    return shape


def _window_starts(n_rows: int, window_size: int, stride: int) -> np.ndarray:
    if n_rows < window_size:
        return np.empty((0,), dtype=np.int64)
    return np.arange(0, n_rows - window_size + 1, stride, dtype=np.int64)


def _window_label_from_slice(
    window_labels: np.ndarray,
    strategy: str,
    positive_ratio: float,
    last_percent: float,
) -> int:
    if strategy == "last":
        tail_fraction = min(1.0, max(0.0, float(last_percent) / 100.0))
        tail_len = max(1, int(np.ceil(len(window_labels) * tail_fraction)))
        tail = window_labels[-tail_len:]
        return int(float(np.mean(tail == 1)) >= positive_ratio)
    if strategy == "positive_ratio":
        return int(float(np.mean(window_labels == 1)) >= positive_ratio)
    return int(np.max(window_labels))


def _build_window_labels(
    labels: np.ndarray,
    window_size: int,
    stride: int,
    strategy: str = "positive_ratio",
    positive_ratio: float = 0.1,
    last_percent: float = 10.0,
) -> np.ndarray:
    """Create one label per window from row-level labels."""
    n_rows = len(labels)
    if n_rows < window_size:
        return np.empty((0,), dtype=np.int32)

    out: list[int] = []
    for s in range(0, n_rows - window_size + 1, stride):
        w = labels[s : s + window_size]
        out.append(_window_label_from_slice(w, strategy, positive_ratio, last_percent))
    return np.asarray(out, dtype=np.int32)


def _build_window_labels_from_starts(
    labels: np.ndarray,
    window_size: int,
    starts: np.ndarray,
    strategy: str = "positive_ratio",
    positive_ratio: float = 0.1,
    last_percent: float = 10.0,
) -> np.ndarray:
    if len(starts) == 0:
        return np.empty((0,), dtype=np.int32)
    out: list[int] = []
    for s in starts:
        w = labels[s : s + window_size]
        out.append(_window_label_from_slice(w, strategy, positive_ratio, last_percent))
    return np.asarray(out, dtype=np.int32)


def engineer_and_save_windows(
    params: FeatureEngineeringParams,
    output_root: str | Path,
) -> FeatureEngineeringArtifacts:
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    train_df = load_metropt3(params.train_csv_path, timestamp_col=params.timestamp_col)
    val_df = load_metropt3(params.val_csv_path, timestamp_col=params.timestamp_col)
    test_df = load_metropt3(params.test_csv_path, timestamp_col=params.timestamp_col)
    train_df = train_df.sort_values(params.timestamp_col).reset_index(drop=True)
    val_df = val_df.sort_values(params.timestamp_col).reset_index(drop=True)
    test_df = test_df.sort_values(params.timestamp_col).reset_index(drop=True)

    feature_source_df = train_df if len(train_df) else (val_df if len(val_df) else test_df)
    feature_cols = list(params.feature_cols) if params.feature_cols else default_feature_cols(feature_source_df, timestamp_col=params.timestamp_col)
    if len(feature_cols) != 15:
        feature_count_warning = f"Expected 15 features, got {len(feature_cols)}"
    else:
        feature_count_warning = ""

    X_train = train_df[feature_cols].to_numpy(dtype=np.float32)
    X_val = val_df[feature_cols].to_numpy(dtype=np.float32)
    X_test = test_df[feature_cols].to_numpy(dtype=np.float32)

    run_id = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    run_dir = output_root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    train_windows_path = run_dir / "train_windows.npy"
    val_windows_path = run_dir / "val_windows.npy"
    test_windows_path = run_dir / "test_windows.npy"
    train_window_labels_path = run_dir / "train_window_labels.npy"
    val_window_labels_path = run_dir / "val_window_labels.npy"
    test_window_labels_path = run_dir / "test_window_labels.npy"
    scaler_npz_path = run_dir / "scaler.npz"
    scaler_metadata_path = run_dir / "scaler_metadata.json"
    metadata_path = run_dir / "metadata.json"

    train_row_labels = train_df[params.label_col].to_numpy(dtype=np.int32) if params.label_col in train_df.columns else np.zeros((len(train_df),), dtype=np.int32)
    val_row_labels = val_df[params.label_col].to_numpy(dtype=np.int32) if params.label_col in val_df.columns else np.zeros((len(val_df),), dtype=np.int32)
    test_row_labels = test_df[params.label_col].to_numpy(dtype=np.int32) if params.label_col in test_df.columns else np.zeros((len(test_df),), dtype=np.int32)

    X_train, X_val, X_test, scaler_metadata, scaler_arrays = _fit_and_apply_scaler(
        X_train,
        X_val,
        X_test,
        train_row_labels,
        params,
    )
    if scaler_arrays:
        np.savez(scaler_npz_path, **scaler_arrays)
    else:
        np.savez(scaler_npz_path)
    scaler_metadata_path.write_text(json.dumps(scaler_metadata, indent=2), encoding="utf-8")

    effective_window_size = 1 if params.point_mode else params.window_size
    effective_stride = 1 if params.point_mode else params.stride

    train_starts_all = _window_starts(len(train_df), effective_window_size, effective_stride)
    val_starts = _window_starts(len(val_df), effective_window_size, effective_stride)
    test_starts = _window_starts(len(test_df), effective_window_size, effective_stride)

    if params.train_normal_only and params.label_col in train_df.columns:
        train_starts = np.asarray(
            [
                int(s)
                for s in train_starts_all
                if np.max(train_row_labels[s : s + effective_window_size]) == 0
            ],
            dtype=np.int64,
        )
    else:
        train_starts = train_starts_all

    train_windows_shape = _write_windows_npy_from_starts(
        X_train,
        train_windows_path,
        window_size=effective_window_size,
        flatten=params.flatten_windows,
        starts=train_starts,
    )
    val_windows_shape = _write_windows_npy_from_starts(
        X_val,
        val_windows_path,
        window_size=effective_window_size,
        flatten=params.flatten_windows,
        starts=val_starts,
    )
    test_windows_shape = _write_windows_npy_from_starts(
        X_test,
        test_windows_path,
        window_size=effective_window_size,
        flatten=params.flatten_windows,
        starts=test_starts,
    )

    if params.point_mode:
        train_window_labels = train_row_labels[train_starts].astype(np.int32)
        val_window_labels = val_row_labels[val_starts].astype(np.int32)
        test_window_labels = test_row_labels[test_starts].astype(np.int32)
    else:
        train_window_labels = _build_window_labels_from_starts(
            train_row_labels,
            effective_window_size,
            train_starts,
            strategy=params.window_label_strategy,
            positive_ratio=params.window_label_positive_ratio,
            last_percent=params.window_label_last_percent,
        )
        val_window_labels = _build_window_labels_from_starts(
            val_row_labels,
            effective_window_size,
            val_starts,
            strategy=params.window_label_strategy,
            positive_ratio=params.window_label_positive_ratio,
            last_percent=params.window_label_last_percent,
        )
        test_window_labels = _build_window_labels_from_starts(
            test_row_labels,
            effective_window_size,
            test_starts,
            strategy=params.window_label_strategy,
            positive_ratio=params.window_label_positive_ratio,
            last_percent=params.window_label_last_percent,
        )

    if params.train_normal_only and np.any(train_window_labels == 1):
        raise ValueError("train_normal_only=True but train windows still contain label 1.")

    np.save(train_window_labels_path, train_window_labels)
    np.save(val_window_labels_path, val_window_labels)
    np.save(test_window_labels_path, test_window_labels)

    metadata = {
        "run_id": run_id,
        "params": _serialize_params(params),
        "feature_cols": feature_cols,
        "feature_count_warning": feature_count_warning,
        "scaler": scaler_metadata,
        "train_rows": int(len(train_df)),
        "val_rows": int(len(val_df)),
        "test_rows": int(len(test_df)),
        "train_candidate_windows": int(len(train_starts_all)),
        "train_kept_windows": int(len(train_starts)),
        "train_windows_shape": list(train_windows_shape),
        "val_windows_shape": list(val_windows_shape),
        "test_windows_shape": list(test_windows_shape),
        "train_period": {
            "start": str(train_df[params.timestamp_col].min()) if len(train_df) else None,
            "end": str(train_df[params.timestamp_col].max()) if len(train_df) else None,
        },
        "test_period": {
            "start": str(test_df[params.timestamp_col].min()) if len(test_df) else None,
            "end": str(test_df[params.timestamp_col].max()) if len(test_df) else None,
        },
        "val_period": {
            "start": str(val_df[params.timestamp_col].min()) if len(val_df) else None,
            "end": str(val_df[params.timestamp_col].max()) if len(val_df) else None,
        },
        "saved_files": {
            "train_windows": _serialize_path(train_windows_path),
            "val_windows": _serialize_path(val_windows_path),
            "test_windows": _serialize_path(test_windows_path),
            "train_window_labels": _serialize_path(train_window_labels_path),
            "val_window_labels": _serialize_path(val_window_labels_path),
            "test_window_labels": _serialize_path(test_window_labels_path),
            "scaler_npz": _serialize_path(scaler_npz_path),
            "scaler_metadata": _serialize_path(scaler_metadata_path),
        },
    }

    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    return FeatureEngineeringArtifacts(
        run_dir=run_dir,
        train_windows_path=train_windows_path,
        val_windows_path=val_windows_path,
        test_windows_path=test_windows_path,
        train_window_labels_path=train_window_labels_path,
        val_window_labels_path=val_window_labels_path,
        test_window_labels_path=test_window_labels_path,
        scaler_npz_path=scaler_npz_path,
        scaler_metadata_path=scaler_metadata_path,
        metadata_path=metadata_path,
    )
