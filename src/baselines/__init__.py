"""Classical / statistical anomaly detection baselines used as comparators
to the VAE.  Each baseline is implemented as a small dataclass-style
``train_and_score`` function that operates on flattened window features so
all models see the exact same inputs as the VAE."""

from .classical import (
    BaselineResult,
    score_isolation_forest,
    score_lof,
    score_oc_svm,
    score_pca_reconstruction,
)

__all__ = [
    "BaselineResult",
    "score_isolation_forest",
    "score_lof",
    "score_oc_svm",
    "score_pca_reconstruction",
]
