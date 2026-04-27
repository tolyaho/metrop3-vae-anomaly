"""Backward-compatible imports for older scripts and notes.

New code should import from :mod:`models.vae_anomaly_detector`.
"""

try:
    from src.models.vae_anomaly_detector import *  # noqa: F401,F403
except ImportError:
    from models.vae_anomaly_detector import *  # noqa: F401,F403
