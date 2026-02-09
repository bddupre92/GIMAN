"""Baseline imputation methods for comparison with GIMIN.

Provides thin wrappers around scikit-learn imputers to match the GIMIN
evaluation interface.  Each baseline class exposes a ``fit_transform``
method that accepts a feature matrix and observation mask and returns
the imputed matrix.

Classes:
    MICEBaseline: Multiple Imputation by Chained Equations using
        scikit-learn's ``IterativeImputer`` with a ``RandomForestRegressor``
        estimator.
    KNNBaseline: k-Nearest-Neighbor imputation via ``KNNImputer``.
    MeanBaseline: Per-feature mean imputation.
    MedianBaseline: Per-feature median imputation.
"""

from __future__ import annotations

import logging

import numpy as np

from ..utils import ArrayLike
from ..utils import prepare_nan_matrix as _prepare_nan_matrix

logger = logging.getLogger(__name__)


def _restore_columns(imputed: np.ndarray, nan_matrix: np.ndarray) -> np.ndarray:
    """Restore original column count if sklearn dropped all-NaN columns."""
    if imputed.shape[1] == nan_matrix.shape[1]:
        return imputed
    full = np.zeros_like(nan_matrix)
    valid_cols = ~np.isnan(nan_matrix).all(axis=0)
    full[:, valid_cols] = imputed
    return full


class MICEBaseline:
    """Multiple Imputation by Chained Equations (MICE) baseline.

    Wraps scikit-learn's ``IterativeImputer`` using a
    ``RandomForestRegressor`` estimator to perform multivariate
    feature-by-feature imputation.

    Args:
        max_iter: Maximum number of imputation rounds.  Default: 10.
        n_estimators: Number of trees in the random forest.  Default: 100.
        random_state: Random seed for reproducibility.  Default: 42.
    """

    def __init__(
        self,
        max_iter: int = 10,
        n_estimators: int = 100,
        random_state: int = 42,
    ) -> None:
        self.max_iter = max_iter
        self.n_estimators = n_estimators
        self.random_state = random_state

    def fit_transform(self, features: ArrayLike, mask: ArrayLike) -> np.ndarray:
        """Impute missing values using MICE with random forest.

        Args:
            features: Feature matrix, shape ``(N, F)``.
            mask: Binary observation mask (1 = observed), shape ``(N, F)``.

        Returns:
            Imputed feature matrix of shape ``(N, F)`` with no NaNs.
        """
        # Lazy imports to avoid hard dependency at module load time.
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.experimental import enable_iterative_imputer  # noqa: F401
        from sklearn.impute import IterativeImputer

        nan_matrix = _prepare_nan_matrix(features, mask)

        estimator = RandomForestRegressor(
            n_estimators=self.n_estimators,
            random_state=self.random_state,
            n_jobs=-1,
        )
        imputer = IterativeImputer(
            estimator=estimator,
            max_iter=self.max_iter,
            random_state=self.random_state,
            verbose=0,
        )

        logger.info(
            "Running MICE baseline (max_iter=%d, n_estimators=%d)",
            self.max_iter,
            self.n_estimators,
        )
        imputed = imputer.fit_transform(nan_matrix)
        return _restore_columns(imputed, nan_matrix)


class KNNBaseline:
    """k-Nearest-Neighbor imputation baseline.

    Wraps scikit-learn's ``KNNImputer`` to fill missing values using
    the mean of each feature's *k* nearest neighbours (in the
    observed-feature subspace).

    Args:
        k: Number of neighbours.  Default: 5.
        weights: Weight function (``"uniform"`` or ``"distance"``).
            Default: ``"distance"``.
    """

    def __init__(
        self,
        k: int = 5,
        weights: str = "distance",
    ) -> None:
        self.k = k
        self.weights = weights

    def fit_transform(
        self,
        features: ArrayLike,
        mask: ArrayLike,
        k: int | None = None,
    ) -> np.ndarray:
        """Impute missing values using kNN.

        Args:
            features: Feature matrix, shape ``(N, F)``.
            mask: Binary observation mask (1 = observed), shape ``(N, F)``.
            k: Override number of neighbours for this call.

        Returns:
            Imputed feature matrix of shape ``(N, F)`` with no NaNs.
        """
        from sklearn.impute import KNNImputer

        n_neighbors = k if k is not None else self.k
        nan_matrix = _prepare_nan_matrix(features, mask)

        imputer = KNNImputer(
            n_neighbors=n_neighbors,
            weights=self.weights,
        )

        logger.info(
            "Running KNN baseline (k=%d, weights='%s')",
            n_neighbors,
            self.weights,
        )
        imputed = imputer.fit_transform(nan_matrix)
        return _restore_columns(imputed, nan_matrix)


class MeanBaseline:
    """Per-feature mean imputation baseline.

    Replaces each missing value with the mean of its feature column
    computed from observed entries.
    """

    def fit_transform(self, features: ArrayLike, mask: ArrayLike) -> np.ndarray:
        """Impute missing values with per-feature means.

        Args:
            features: Feature matrix, shape ``(N, F)``.
            mask: Binary observation mask (1 = observed), shape ``(N, F)``.

        Returns:
            Imputed feature matrix of shape ``(N, F)`` with no NaNs.
        """
        from sklearn.impute import SimpleImputer

        nan_matrix = _prepare_nan_matrix(features, mask)

        imputer = SimpleImputer(strategy="mean")
        logger.info("Running mean imputation baseline.")
        imputed = imputer.fit_transform(nan_matrix)
        return _restore_columns(imputed, nan_matrix)


class MedianBaseline:
    """Per-feature median imputation baseline.

    Replaces each missing value with the median of its feature column
    computed from observed entries.
    """

    def fit_transform(self, features: ArrayLike, mask: ArrayLike) -> np.ndarray:
        """Impute missing values with per-feature medians.

        Args:
            features: Feature matrix, shape ``(N, F)``.
            mask: Binary observation mask (1 = observed), shape ``(N, F)``.

        Returns:
            Imputed feature matrix of shape ``(N, F)`` with no NaNs.
        """
        from sklearn.impute import SimpleImputer

        nan_matrix = _prepare_nan_matrix(features, mask)

        imputer = SimpleImputer(strategy="median")
        logger.info("Running median imputation baseline.")
        imputed = imputer.fit_transform(nan_matrix)
        return _restore_columns(imputed, nan_matrix)
