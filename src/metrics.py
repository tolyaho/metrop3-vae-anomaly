"""Backward-compatible imports for older scripts and notes.

New code should import from :mod:`evaluation.metrics`.
"""

try:
    from src.evaluation.metrics import *  # noqa: F401,F403
except ImportError:
    from evaluation.metrics import *  # noqa: F401,F403
