"""Paper-ready matplotlib style for the MetroPT3 VAE anomaly detection report.

All figures across the project should call :func:`apply_paper_style` once
and save through :func:`save_figure` so colors, fonts, line widths, grid
appearance, and DPI stay consistent.

Color choices are based on a colorblind-safe extension of Tableau / Wong
palettes so the figures reproduce well in print and in greyscale.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

import matplotlib as mpl
import matplotlib.pyplot as plt


PALETTE: dict[str, str] = {
    "ink": "#1F2933",
    "muted": "#52606D",
    "subtle": "#9AA5B1",
    "grid": "#D8DEE6",
    "background": "#FFFFFF",
    "panel": "#F5F7FA",
    "primary": "#1F77B4",
    "secondary": "#2CA02C",
    "accent": "#D62728",
    "warm": "#FF7F0E",
    "purple": "#9467BD",
    "teal": "#17BECF",
    "ochre": "#BCBD22",
    "rose": "#E377C2",
    "normal": "#3D7CA9",
    "anomaly": "#D64545",
    "event": "#F4A582",
    "threshold": "#222222",
}

MODEL_COLORS: dict[str, str] = {
    "VAE (dense)": "#1F77B4",
    "VAE (conv1d)": "#9467BD",
    "VAE (LSTM-AE)": "#17BECF",
    "Isolation Forest": "#FF7F0E",
    "PCA reconstruction": "#2CA02C",
    "OC-SVM": "#8C564B",
    "LOF": "#E377C2",
}


def apply_paper_style() -> None:
    """Set rcParams used everywhere in the project for paper-quality plots."""
    mpl.rcParams.update(
        {
            "figure.dpi": 110,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "savefig.facecolor": PALETTE["background"],
            "figure.facecolor": PALETTE["background"],
            "axes.facecolor": PALETTE["background"],
            "axes.edgecolor": PALETTE["muted"],
            "axes.labelcolor": PALETTE["ink"],
            "axes.titlecolor": PALETTE["ink"],
            "axes.titlesize": 13,
            "axes.titleweight": "bold",
            "axes.titlepad": 12,
            "axes.labelsize": 11,
            "axes.labelweight": "regular",
            "axes.linewidth": 0.9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "axes.axisbelow": True,
            "axes.prop_cycle": mpl.cycler(
                color=[
                    PALETTE["primary"],
                    PALETTE["warm"],
                    PALETTE["secondary"],
                    PALETTE["accent"],
                    PALETTE["purple"],
                    PALETTE["teal"],
                    PALETTE["ochre"],
                    PALETTE["rose"],
                ]
            ),
            "grid.color": PALETTE["grid"],
            "grid.linestyle": "--",
            "grid.linewidth": 0.7,
            "grid.alpha": 0.55,
            "xtick.color": PALETTE["muted"],
            "ytick.color": PALETTE["muted"],
            "xtick.labelsize": 9.5,
            "ytick.labelsize": 9.5,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
            "legend.frameon": False,
            "legend.fontsize": 9.5,
            "legend.title_fontsize": 10,
            "legend.handletextpad": 0.6,
            "lines.linewidth": 1.8,
            "lines.markersize": 5.5,
            "patch.linewidth": 0.6,
            "patch.edgecolor": PALETTE["background"],
            "font.family": "DejaVu Sans",
            "font.size": 10.5,
            "mathtext.fontset": "dejavusans",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def style_axes(ax: plt.Axes, *, grid_axis: str = "y") -> None:
    """Apply the project's axes styling (subtle grid, minimal spines)."""
    ax.grid(True, axis=grid_axis, color=PALETTE["grid"], linewidth=0.7, alpha=0.55, linestyle="--")
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(PALETTE["muted"])
        ax.spines[spine].set_linewidth(0.9)
    ax.tick_params(colors=PALETTE["muted"])


def annotate_bars(
    ax: plt.Axes,
    bars: Sequence[mpl.patches.Rectangle],
    fmt: str = "{:.3f}",
    *,
    fontsize: float = 8.5,
    color: str | None = None,
    y_offset: float = 0.012,
) -> None:
    """Place value labels above bars with a consistent style."""
    label_color = color if color is not None else PALETTE["ink"]
    for bar in bars:
        h = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            h + y_offset,
            fmt.format(h),
            ha="center",
            va="bottom",
            fontsize=fontsize,
            color=label_color,
        )


def save_figure(fig: plt.Figure, paths: Iterable[Path | str], *, dpi: int = 300) -> None:
    """Save *fig* to one or more paths with paper-quality settings."""
    for path in paths:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
