"""Centralized matplotlib styling and plotting helpers for paper-ready figures."""

from .style import (
    PALETTE,
    MODEL_COLORS,
    apply_paper_style,
    save_figure,
    style_axes,
    annotate_bars,
)

__all__ = [
    "PALETTE",
    "MODEL_COLORS",
    "apply_paper_style",
    "save_figure",
    "style_axes",
    "annotate_bars",
]
