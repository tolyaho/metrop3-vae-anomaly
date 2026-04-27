"""Backward-compatible imports for older scripts and notes.

New code should import from :mod:`data.preprocessing`.
"""

try:
    from src.data.preprocessing import *  # noqa: F401,F403
except ImportError:
    from data.preprocessing import *  # noqa: F401,F403
