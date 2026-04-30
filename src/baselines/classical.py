"""Unsupervised classical baselines for the MetroPT3 VAE comparison.

All baselines:
  * are fit on **normal** training windows only (``train_y == 0``),
  * receive flattened window features (same input that the VAE consumes when
    its windows are flattened),
  * return a higher-is-more-anomalous score for every train / val / test
    window so they can share the percentile-thresholding logic with the VAE.

The four baselines mirror the four most common unsupervised anomaly
detection families used as paper baselines:

  * Isolation Forest -- tree ensemble (random partitioning).
  * PCA reconstruction -- linear subspace projection.
  * One-Class SVM -- kernel density boundary, scaled via Nystroem features
    and SGDOneClassSVM so it is tractable on >50k flattened windows.
  * Local Outlier Factor -- local density (``novelty=True`` so we can score
    held-out points; train is subsampled when very large).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.kernel_approximation import Nystroem
from sklearn.linear_model import SGDOneClassSVM
from sklearn.neighbors import LocalOutlierFactor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


@dataclass
class BaselineResult:
    """Train / val / test scores for one fitted baseline."""

    name: str
    train_scores: np.ndarray
    val_scores: np.ndarray
    test_scores: np.ndarray
    extras: dict[str, Any] = field(default_factory=dict)


def _flatten(windows: np.ndarray) -> np.ndarray:
    return windows.reshape(len(windows), -1).astype(np.float32, copy=False)


def _normal_only(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    mask = y == 0
    if not np.any(mask):
        raise ValueError("No normal training windows available; baseline cannot fit.")
    return x[mask]


def score_isolation_forest(
    x_train: np.ndarray,
    x_val: np.ndarray,
    x_test: np.ndarray,
    train_labels: np.ndarray,
    *,
    n_estimators: int = 500,
    max_samples: str | float | int = "auto",
    contamination: str | float = "auto",
    max_features: float = 1.0,
    bootstrap: bool = False,
    random_state: int = 42,
    n_jobs: int = -1,
) -> BaselineResult:
    """Isolation Forest fit on normal windows; score = ``-decision_function``."""
    x_train_f = _flatten(x_train)
    x_val_f = _flatten(x_val)
    x_test_f = _flatten(x_test)
    fit_x = _normal_only(x_train_f, train_labels)

    model = IsolationForest(
        n_estimators=n_estimators,
        max_samples=max_samples,
        contamination=contamination,
        max_features=max_features,
        bootstrap=bootstrap,
        random_state=random_state,
        n_jobs=n_jobs,
        verbose=0,
    )
    model.fit(fit_x)
    return BaselineResult(
        name="Isolation Forest",
        train_scores=-model.decision_function(x_train_f),
        val_scores=-model.decision_function(x_val_f),
        test_scores=-model.decision_function(x_test_f),
        extras={"params": model.get_params()},
    )


def score_pca_reconstruction(
    x_train: np.ndarray,
    x_val: np.ndarray,
    x_test: np.ndarray,
    train_labels: np.ndarray,
    *,
    n_components: int | float | str = 0.95,
    random_state: int = 42,
) -> BaselineResult:
    """Standardize features then project onto top-k PCs; score is the
    squared reconstruction error (sum over features)."""
    x_train_f = _flatten(x_train)
    x_val_f = _flatten(x_val)
    x_test_f = _flatten(x_test)
    fit_x = _normal_only(x_train_f, train_labels)

    if isinstance(n_components, str):
        if n_components.lower() in {"mle"}:
            comps: int | float = "mle"
        else:
            comps = float(n_components)
    else:
        comps = n_components

    pipe = Pipeline(
        [
            ("scaler", StandardScaler(with_mean=True, with_std=True)),
            ("pca", PCA(n_components=comps, random_state=random_state, svd_solver="auto")),
        ]
    )
    pipe.fit(fit_x)
    scaler: StandardScaler = pipe.named_steps["scaler"]
    pca: PCA = pipe.named_steps["pca"]

    def recon_error(x: np.ndarray) -> np.ndarray:
        z = scaler.transform(x)
        z_proj = pca.transform(z)
        z_back = pca.inverse_transform(z_proj)
        diff = z - z_back
        return np.sum(diff * diff, axis=1).astype(np.float32)

    return BaselineResult(
        name="PCA reconstruction",
        train_scores=recon_error(x_train_f),
        val_scores=recon_error(x_val_f),
        test_scores=recon_error(x_test_f),
        extras={
            "n_components": int(pca.n_components_),
            "explained_variance_ratio_sum": float(np.sum(pca.explained_variance_ratio_)),
        },
    )


def score_oc_svm(
    x_train: np.ndarray,
    x_val: np.ndarray,
    x_test: np.ndarray,
    train_labels: np.ndarray,
    *,
    nu: float = 0.05,
    gamma: str | float = "scale",
    n_components: int = 200,
    max_train_samples: int = 30000,
    random_state: int = 42,
) -> BaselineResult:
    """Approximate One-Class SVM via Nystroem RBF features + SGDOneClassSVM.

    The exact ``OneClassSVM`` is O(n^2)-O(n^3) which is intractable on >30k
    windows of dimension ~900-1800, so we use the standard scikit-learn
    recipe of approximating the RBF kernel and fitting the linear OC-SVM via
    SGD.  Fit data is subsampled to ``max_train_samples`` normal windows for
    a reproducible Nystroem basis.
    """
    x_train_f = _flatten(x_train)
    x_val_f = _flatten(x_val)
    x_test_f = _flatten(x_test)
    fit_x = _normal_only(x_train_f, train_labels)

    rng = np.random.default_rng(random_state)
    if len(fit_x) > max_train_samples:
        idx = rng.choice(len(fit_x), size=max_train_samples, replace=False)
        fit_x = fit_x[idx]

    if isinstance(gamma, str):
        if gamma == "scale":
            scale_var = float(np.var(fit_x, axis=0).mean()) or 1.0
            gamma_val = 1.0 / (fit_x.shape[1] * scale_var)
        elif gamma == "auto":
            gamma_val = 1.0 / fit_x.shape[1]
        else:
            raise ValueError(f"Unknown gamma string: {gamma}")
    else:
        gamma_val = float(gamma)

    pipe = Pipeline(
        [
            ("scaler", StandardScaler(with_mean=True, with_std=True)),
            (
                "nystroem",
                Nystroem(
                    kernel="rbf",
                    gamma=gamma_val,
                    n_components=int(n_components),
                    random_state=random_state,
                ),
            ),
            (
                "ocsvm",
                SGDOneClassSVM(
                    nu=nu,
                    shuffle=True,
                    max_iter=2000,
                    tol=1e-4,
                    learning_rate="optimal",
                    random_state=random_state,
                ),
            ),
        ]
    )
    pipe.fit(fit_x)

    def score(x: np.ndarray) -> np.ndarray:
        return -pipe.decision_function(x).astype(np.float32)

    return BaselineResult(
        name="OC-SVM",
        train_scores=score(x_train_f),
        val_scores=score(x_val_f),
        test_scores=score(x_test_f),
        extras={
            "nu": float(nu),
            "gamma": float(gamma_val),
            "n_components": int(n_components),
            "max_train_samples": int(max_train_samples),
            "fit_samples": int(len(fit_x)),
        },
    )


def score_lof(
    x_train: np.ndarray,
    x_val: np.ndarray,
    x_test: np.ndarray,
    train_labels: np.ndarray,
    *,
    n_neighbors: int = 35,
    leaf_size: int = 30,
    metric: str = "euclidean",
    contamination: str | float = "auto",
    max_train_samples: int = 20000,
    standardize: bool = True,
    random_state: int = 42,
    n_jobs: int = -1,
) -> BaselineResult:
    """Local Outlier Factor with ``novelty=True`` so it can score new points.

    Fit data is subsampled (``max_train_samples``) because LOF queries grow
    O(n_test * n_train * d). The score is ``-score_samples(X)`` so higher
    means more anomalous, mirroring IF and OC-SVM convention.
    """
    x_train_f = _flatten(x_train)
    x_val_f = _flatten(x_val)
    x_test_f = _flatten(x_test)
    fit_x = _normal_only(x_train_f, train_labels)

    rng = np.random.default_rng(random_state)
    if len(fit_x) > max_train_samples:
        idx = rng.choice(len(fit_x), size=max_train_samples, replace=False)
        fit_x = fit_x[idx]

    steps: list[tuple[str, object]] = []
    if standardize:
        steps.append(("scaler", StandardScaler(with_mean=True, with_std=True)))
    steps.append(
        (
            "lof",
            LocalOutlierFactor(
                n_neighbors=int(n_neighbors),
                leaf_size=int(leaf_size),
                metric=metric,
                contamination=contamination,
                novelty=True,
                n_jobs=n_jobs,
            ),
        )
    )
    pipe = Pipeline(steps)
    pipe.fit(fit_x)

    def score(x: np.ndarray) -> np.ndarray:
        return -pipe.score_samples(x).astype(np.float32)

    return BaselineResult(
        name="LOF",
        train_scores=score(x_train_f),
        val_scores=score(x_val_f),
        test_scores=score(x_test_f),
        extras={
            "n_neighbors": int(n_neighbors),
            "max_train_samples": int(max_train_samples),
            "fit_samples": int(len(fit_x)),
            "standardize": bool(standardize),
        },
    )
