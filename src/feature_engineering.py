"""Backward-compatible imports for older scripts and notes.

New code should import from :mod:`data.windows`.
"""

try:
    from src.data.windows import *  # noqa: F401,F403
except ImportError:
    from data.windows import *  # noqa: F401,F403
