"""Per-failure analysis: latency to first alarm, per-event PR, timeline.

For every annotated failure event that overlaps the test split, compute:
  * latency-to-first-alarm (first window with score > threshold inside the
    event interval, measured in minutes from event start; NaN if missed),
  * recall, precision and false-alarm count restricted to the local
    neighbourhood of the event (event window +/- ``--margin-minutes``),
  * a zoomed timeline plot showing the test score trace, the threshold,
    and the event interval highlighted.

The script accepts one or more ``<label>:<run_dir>`` pairs so VAE and
classical baselines can be compared on the same axes.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.preprocessing import DEFAULT_FAILURE_EVENTS  # noqa: E402
from src.evaluation.metrics import binary_metrics  # noqa: E402
from src.plotting import MODEL_COLORS, PALETTE, apply_paper_style, save_figure, style_axes  # noqa: E402


def _resolve(path):
    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _load_run(run_dir, kind):
    run_dir = Path(run_dir)
    if kind == "vae":
        scores = np.load(run_dir / "test_scores.npy").astype(np.float64)
        labels = np.load(run_dir / "test_labels.npy").astype(np.int32)
        train_scores = np.load(run_dir / "train_scores.npy").astype(np.float64)
        summary = json.loads((run_dir / "summary.json").read_text())
        source_run = summary.get("source_processed_windows_run") or summary.get("source_metadata", {}).get("run_id")
    else:
        data = np.load(run_dir / "scores.npz")
        scores = data["test_scores"].astype(np.float64)
        labels = data["test_labels"].astype(np.int32)
        train_scores = data["train_scores"].astype(np.float64)
        config = json.loads((run_dir / "config.json").read_text())
        source_run = config.get("window_dir")
    return scores, labels, train_scores, source_run


def _resolve_window_dir(source_run):
    if source_run is None:
        return None
    candidate = _resolve(source_run)
    if candidate.exists():
        return candidate
    candidate = _resolve(Path("dataset/processed_windows") / Path(source_run).name)
    return candidate if candidate.exists() else None


def _window_starts(n_rows, window_size, stride):
    if n_rows < window_size:
        return np.empty((0,), dtype=np.int64)
    return np.arange(0, n_rows - window_size + 1, stride, dtype=np.int64)


def _test_window_timestamps(window_dir, n_test_windows):
    metadata = json.loads((window_dir / "metadata.json").read_text())
    params = metadata["params"]
    window_size = 1 if params.get("point_mode", False) else int(params["window_size"])
    stride = 1 if params.get("point_mode", False) else int(params["stride"])
    test_csv = _resolve(params["test_csv_path"]) if Path(params["test_csv_path"]).is_absolute() else _resolve(params["test_csv_path"])
    df = pd.read_csv(test_csv, index_col=0, parse_dates=[params.get("timestamp_col", "timestamp")], engine="python")
    df = df.sort_values(params.get("timestamp_col", "timestamp")).reset_index(drop=True)
    starts = _window_starts(len(df), window_size, stride)[:n_test_windows]
    last_idx = starts + window_size - 1
    last_idx = np.clip(last_idx, 0, len(df) - 1)
    timestamps = df[params.get("timestamp_col", "timestamp")].iloc[last_idx].to_numpy()
    return timestamps, window_size, stride


def _event_metrics(timestamps, scores, threshold, event_start, event_end, margin):
    margin_td = pd.Timedelta(minutes=margin)
    in_event = (timestamps >= event_start) & (timestamps <= event_end)
    near_event = (timestamps >= event_start - margin_td) & (timestamps <= event_end + margin_td)

    preds = (scores > threshold).astype(np.int32)

    tp = int(np.sum(preds[in_event] == 1))
    fn = int(np.sum(preds[in_event] == 0))
    fp_near = int(np.sum((preds == 1) & near_event & (~in_event)))

    in_event_idx = np.flatnonzero(in_event)
    if in_event_idx.size and tp > 0:
        first_alarm_idx = in_event_idx[np.argmax(preds[in_event_idx] == 1)]
        first_alarm_ts = pd.Timestamp(timestamps[first_alarm_idx])
        latency_minutes = (first_alarm_ts - event_start) / pd.Timedelta(minutes=1)
    else:
        first_alarm_ts = pd.NaT
        latency_minutes = float("nan")

    recall = tp / max(1, tp + fn)
    precision_local = tp / max(1, tp + fp_near)

    return {
        "tp": tp,
        "fn": fn,
        "fp_near_event": fp_near,
        "recall_in_event": recall,
        "precision_near_event": precision_local,
        "first_alarm_timestamp": str(first_alarm_ts) if first_alarm_ts is not pd.NaT else "",
        "latency_minutes": latency_minutes,
        "n_windows_in_event": int(in_event.sum()),
    }


def _plot_timelines(events, model_runs, *, threshold_percentile, output_paths):
    fig, axes = plt.subplots(len(events), 1, figsize=(12.0, 3.4 * len(events)), squeeze=False)
    for ax_row, event in zip(axes, events):
        ax = ax_row[0]
        margin_td = pd.Timedelta(minutes=240)
        win_start = event["event_start"] - margin_td
        win_end = event["event_end"] + margin_td
        ax.axvspan(event["event_start"], event["event_end"], color=PALETTE["event"], alpha=0.32, label=f"{event['id']} interval")
        for label, info in model_runs.items():
            timestamps = info["timestamps"]
            mask = (timestamps >= win_start) & (timestamps <= win_end)
            color = MODEL_COLORS.get(label.split(" (")[0], MODEL_COLORS.get(label, PALETTE["primary"]))
            scores = info["scores"][mask]
            if not len(scores):
                continue
            scores_norm = (scores - info["score_min"]) / max(info["score_max"] - info["score_min"], 1e-9)
            ax.plot(timestamps[mask], scores_norm, color=color, linewidth=1.0, label=label)
            threshold_norm = (info["threshold"] - info["score_min"]) / max(info["score_max"] - info["score_min"], 1e-9)
            ax.axhline(threshold_norm, color=color, linestyle="--", linewidth=0.8, alpha=0.7)
        ax.set_title(f"{event['id']} -- {event['failure_type']}  ({event['event_start']:%Y-%m-%d %H:%M} -> {event['event_end']:%H:%M})")
        ax.set_ylabel("Normalized score")
        ax.set_ylim(-0.02, 1.05)
        style_axes(ax, grid_axis="both")
        if ax is axes[0, 0]:
            ax.legend(frameon=False, loc="upper left", fontsize=9)
    axes[-1, 0].set_xlabel("Timestamp")
    fig.suptitle(
        f"Per-failure timeline (threshold = train_p{threshold_percentile})", fontsize=13, fontweight="bold"
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    save_figure(fig, output_paths)


def _plot_latency_bars(latency_df, output_paths):
    pivot = latency_df.pivot_table(index="event_id", columns="model", values="latency_minutes", aggfunc="first")
    if pivot.empty:
        return
    models = list(pivot.columns)
    events = list(pivot.index)
    fig, ax = plt.subplots(figsize=(max(7, 1.8 * len(events)), 5))
    bar_width = 0.8 / max(1, len(models))
    x = np.arange(len(events), dtype=float)
    for i, model in enumerate(models):
        vals = pivot[model].to_numpy(dtype=float)
        plot_vals = np.where(np.isnan(vals), 0.0, vals)
        color = MODEL_COLORS.get(model.split(" (")[0], MODEL_COLORS.get(model, PALETTE["primary"]))
        bars = ax.bar(x + i * bar_width - 0.4 + bar_width / 2, plot_vals, width=bar_width, color=color, label=model)
        for bar, v in zip(bars, vals):
            text = "MISS" if np.isnan(v) else f"{v:.0f}m"
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 5, text, ha="center", va="bottom", fontsize=8.5, color=PALETTE["ink"])
    ax.set_xticks(x)
    ax.set_xticklabels(events)
    ax.set_ylabel("Minutes from event start to first alarm")
    ax.set_title("Latency to first alarm per failure event")
    ax.axhline(0, color=PALETTE["subtle"], linewidth=0.8)
    style_axes(ax, grid_axis="y")
    ax.legend(frameon=False, loc="upper right", fontsize=9)
    fig.tight_layout()
    save_figure(fig, output_paths)


def main():
    parser = argparse.ArgumentParser(description="Per-failure analysis (latency, per-event metrics, timeline).")
    parser.add_argument(
        "--model",
        nargs="+",
        required=True,
        help="One or more <label>:<kind>:<run_dir> entries. kind in {vae, baseline}.",
    )
    parser.add_argument("--threshold-percentile", type=float, default=98.0)
    parser.add_argument("--margin-minutes", type=float, default=240.0)
    parser.add_argument("--output-dir", default="reports/figures/failure_events")
    parser.add_argument("--output-table", default="reports/tables/failure_event_metrics.csv")
    parser.add_argument("--output-latency-figure", default="reports/figures/failure_latency.png")
    args = parser.parse_args()

    apply_paper_style()
    out_dir = _resolve(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model_runs = {}
    timestamps_ref = None
    for entry in args.model:
        parts = entry.split(":")
        if len(parts) != 3:
            raise SystemExit(f"Bad --model spec: {entry!r}; expected <label>:<kind>:<path>")
        label, kind, raw_path = parts
        run_dir = _resolve(raw_path)
        scores, labels, train_scores, source_run = _load_run(run_dir, kind)
        threshold = float(np.percentile(train_scores, args.threshold_percentile))
        window_dir = _resolve_window_dir(source_run)
        if window_dir is None:
            raise SystemExit(f"Could not resolve source window dir for {run_dir} (source_run={source_run!r}).")
        timestamps, _, _ = _test_window_timestamps(window_dir, len(scores))
        timestamps = pd.to_datetime(timestamps)
        if timestamps_ref is None:
            timestamps_ref = timestamps
        model_runs[label] = {
            "run_dir": run_dir,
            "scores": scores,
            "labels": labels,
            "train_scores": train_scores,
            "threshold": threshold,
            "timestamps": timestamps,
            "source_run": source_run,
            "score_min": float(np.min(scores)),
            "score_max": float(np.max(scores)),
        }
        print(
            f"Loaded {label}: scores={scores.shape}, threshold(train_p{args.threshold_percentile})={threshold:.4f}"
        )

    if timestamps_ref is None:
        raise SystemExit("No models loaded.")

    test_min = pd.Timestamp(timestamps_ref.min())
    test_max = pd.Timestamp(timestamps_ref.max())
    events = []
    for ev in DEFAULT_FAILURE_EVENTS:
        ev_start = pd.Timestamp(ev["start"])
        ev_end = pd.Timestamp(ev["end"])
        if ev_end < test_min or ev_start > test_max:
            continue
        events.append(
            {
                "id": ev["id"],
                "failure_type": ev.get("failure_type", "Unknown"),
                "event_start": ev_start,
                "event_end": ev_end,
            }
        )

    if not events:
        print("No annotated failure events fall inside the test split; skipping per-event analysis.")
        return

    rows = []
    latency_rows = []
    for label, info in model_runs.items():
        for event in events:
            metrics = _event_metrics(
                info["timestamps"],
                info["scores"],
                info["threshold"],
                event["event_start"],
                event["event_end"],
                margin=args.margin_minutes,
            )
            row = {
                "model": label,
                "event_id": event["id"],
                "event_start": str(event["event_start"]),
                "event_end": str(event["event_end"]),
                "threshold": info["threshold"],
                **metrics,
            }
            rows.append(row)
            latency_rows.append(
                {
                    "model": label,
                    "event_id": event["id"],
                    "latency_minutes": metrics["latency_minutes"],
                }
            )

    metrics_df = pd.DataFrame(rows)
    out_table = _resolve(args.output_table)
    out_table.parent.mkdir(parents=True, exist_ok=True)
    metrics_df.to_csv(out_table, index=False)
    print(f"Saved metrics: {out_table}")

    timeline_paths = [out_dir / "failure_events_timeline.png", out_dir / "failure_events_timeline.pdf"]
    _plot_timelines(events, model_runs, threshold_percentile=int(args.threshold_percentile), output_paths=timeline_paths)

    latency_paths = [_resolve(args.output_latency_figure), _resolve(args.output_latency_figure).with_suffix(".pdf")]
    _plot_latency_bars(pd.DataFrame(latency_rows), latency_paths)

    print("Per-event metrics:")
    print(metrics_df[["model", "event_id", "tp", "fn", "fp_near_event", "recall_in_event", "precision_near_event", "latency_minutes"]].to_string(index=False))


if __name__ == "__main__":
    main()
