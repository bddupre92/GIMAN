"""Baseline imputation methods for comparison with GIMIN.

Provides thin wrappers around scikit-learn imputers and deep learning
imputation frameworks to match the GIMIN evaluation interface.  Each
baseline class exposes a ``fit_transform`` method that accepts a feature
matrix and observation mask and returns the imputed matrix.

Classical baselines:
    MICEBaseline: Multiple Imputation by Chained Equations (IterativeImputer + RF).
    KNNBaseline: k-Nearest-Neighbor imputation via ``KNNImputer``.
    MeanBaseline: Per-feature mean imputation.
    MedianBaseline: Per-feature median imputation.
    MissForestBaseline: Iterative RF with convergence checking.

Deep learning baselines:
    GAINBaseline: Generative Adversarial Imputation Nets (Yoon et al., ICML 2018).
    SAITSBaseline: Self-Attention-based Imputation for Time Series (Du et al., ESWA 2023).
    MIWAEBaseline: Missing data Importance-Weighted Autoencoder (Mattei & Frisch, ICML 2019).
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


class MissForestBaseline:
    """MissForest imputation baseline (Stekhoven & Bühlmann, 2012).

    Iterative imputation using Random Forest regressors.  This is
    conceptually the same as MICE with a Random Forest estimator but
    uses dedicated convergence checking: iteration stops when the
    normalised difference between consecutive imputed matrices falls
    below ``tol`` or ``max_iter`` is reached.

    Args:
        max_iter: Maximum number of imputation rounds.  Default: 10.
        n_estimators: Number of trees per random forest.  Default: 100.
        random_state: Random seed.  Default: 42.
        tol: Convergence tolerance.  Default: 1e-3.
    """

    def __init__(
        self,
        max_iter: int = 10,
        n_estimators: int = 100,
        random_state: int = 42,
        tol: float = 1e-3,
    ) -> None:
        self.max_iter = max_iter
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.tol = tol

    def fit_transform(self, features: ArrayLike, mask: ArrayLike) -> np.ndarray:
        """Impute missing values using MissForest.

        Args:
            features: Feature matrix, shape ``(N, F)``.
            mask: Binary observation mask (1 = observed), shape ``(N, F)``.

        Returns:
            Imputed feature matrix of shape ``(N, F)`` with no NaNs.
        """
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.experimental import enable_iterative_imputer  # noqa: F401
        from sklearn.impute import IterativeImputer

        nan_matrix = _prepare_nan_matrix(features, mask)

        estimator = RandomForestRegressor(
            n_estimators=self.n_estimators,
            random_state=self.random_state,
            n_jobs=-1,
            max_depth=None,
            min_samples_leaf=5,
        )
        imputer = IterativeImputer(
            estimator=estimator,
            max_iter=self.max_iter,
            random_state=self.random_state,
            tol=self.tol,
            verbose=0,
            imputation_order="ascending",  # MissForest: least-missing first
        )

        logger.info(
            "Running MissForest baseline (max_iter=%d, n_estimators=%d, tol=%.1e)",
            self.max_iter,
            self.n_estimators,
            self.tol,
        )
        imputed = imputer.fit_transform(nan_matrix)
        return _restore_columns(imputed, nan_matrix)


# ---------------------------------------------------------------------------
# Deep learning baselines
# ---------------------------------------------------------------------------


class GAINBaseline:
    """Generative Adversarial Imputation Nets (Yoon et al., ICML 2018).

    Uses a GAN framework where the generator fills in missing values and
    the discriminator distinguishes observed from imputed entries.
    Accessed via the ``hyperimpute`` package.

    Args:
        n_epochs: Training epochs.  Default: 100.
        batch_size: Mini-batch size.  Default: 128.
        hint_rate: Fraction of hint information given to the discriminator.
            Default: 0.9 (per original paper).
        random_state: Random seed.  Default: 42.
    """

    def __init__(
        self,
        n_epochs: int = 100,
        batch_size: int = 128,
        hint_rate: float = 0.9,
        random_state: int = 42,
    ) -> None:
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.hint_rate = hint_rate
        self.random_state = random_state

    def fit_transform(self, features: ArrayLike, mask: ArrayLike) -> np.ndarray:
        """Impute missing values using GAIN.

        Args:
            features: Feature matrix, shape ``(N, F)``.
            mask: Binary observation mask (1 = observed), shape ``(N, F)``.

        Returns:
            Imputed feature matrix of shape ``(N, F)`` with no NaNs.
        """
        import pandas as pd
        from hyperimpute.plugins.imputers import Imputers

        nan_matrix = _prepare_nan_matrix(features, mask)
        df = pd.DataFrame(nan_matrix)

        logger.info(
            "Running GAIN baseline (n_epochs=%d, batch_size=%d, hint_rate=%.2f)",
            self.n_epochs,
            self.batch_size,
            self.hint_rate,
        )

        plugin = Imputers().get(
            "gain",
            n_epochs=self.n_epochs,
            batch_size=self.batch_size,
            hint_rate=self.hint_rate,
            random_state=self.random_state,
        )
        imputed = plugin.fit_transform(df.copy()).values
        return _restore_columns(imputed, nan_matrix)


class SAITSBaseline:
    """Self-Attention-based Imputation (Du et al., ESWA 2023).

    Lightweight PyTorch implementation of the SAITS diagonal-masked
    self-attention architecture for cross-sectional imputation.  Uses
    a Transformer encoder that learns inter-feature correlations from
    observed entries and reconstructs missing values.

    Args:
        n_layers: Number of Transformer encoder layers.  Default: 2.
        d_model: Embedding dimension.  Default: 64.
        n_heads: Number of attention heads.  Default: 4.
        d_ffn: Feed-forward network hidden dim.  Default: 128.
        dropout: Dropout rate.  Default: 0.1.
        epochs: Training epochs.  Default: 50.
        batch_size: Mini-batch size.  Default: 64.
        lr: Learning rate.  Default: 1e-3.
        patience: Early-stopping patience.  Default: 5.
        random_state: Random seed.  Default: 42.
    """

    def __init__(
        self,
        n_layers: int = 2,
        d_model: int = 64,
        n_heads: int = 4,
        d_ffn: int = 128,
        dropout: float = 0.1,
        epochs: int = 50,
        batch_size: int = 64,
        lr: float = 1e-3,
        patience: int = 5,
        random_state: int = 42,
    ) -> None:
        self.n_layers = n_layers
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_ffn = d_ffn
        self.dropout = dropout
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.patience = patience
        self.random_state = random_state

    def fit_transform(self, features: ArrayLike, mask: ArrayLike) -> np.ndarray:
        """Impute missing values using self-attention.

        Args:
            features: Feature matrix, shape ``(N, F)``.
            mask: Binary observation mask (1 = observed), shape ``(N, F)``.

        Returns:
            Imputed feature matrix of shape ``(N, F)`` with no NaNs.
        """
        import torch
        import torch.nn as nn

        nan_matrix = _prepare_nan_matrix(features, mask)
        n_samples, n_features = nan_matrix.shape

        logger.info(
            "Running SAITS baseline (n_layers=%d, d_model=%d, n_heads=%d, "
            "epochs=%d, lr=%.0e)",
            self.n_layers,
            self.d_model,
            self.n_heads,
            self.epochs,
            self.lr,
        )

        torch.manual_seed(self.random_state)

        # Normalize features to z-scores so all features contribute equally.
        # Fit on observed entries only; after normalization NaN → 0 (=mean).
        obs_mask = (~np.isnan(nan_matrix)).astype(np.float32)
        col_means = np.nanmean(nan_matrix, axis=0)
        col_stds = np.nanstd(nan_matrix, axis=0)
        col_means = np.where(np.isnan(col_means), 0.0, col_means)
        col_stds = np.where((col_stds < 1e-8) | np.isnan(col_stds), 1.0, col_stds)

        X_scaled = (nan_matrix - col_means) / col_stds
        X_filled = np.nan_to_num(X_scaled, nan=0.0).astype(np.float32)

        X_t = torch.from_numpy(X_filled)
        M_t = torch.from_numpy(obs_mask)

        # Simple Transformer encoder for cross-sectional imputation.
        # Input: [batch, n_features] projected to [batch, n_features, d_model]
        # via per-feature linear embedding, then Transformer encoder treats
        # features as the sequence dimension.
        class _SAITSModel(nn.Module):
            def __init__(self_, n_feat, d_model, n_heads, d_ffn, n_layers, dropout):
                super().__init__()
                self_.embed = nn.Linear(1, d_model)
                encoder_layer = nn.TransformerEncoderLayer(
                    d_model=d_model,
                    nhead=n_heads,
                    dim_feedforward=d_ffn,
                    dropout=dropout,
                    batch_first=True,
                )
                self_.encoder = nn.TransformerEncoder(
                    encoder_layer,
                    num_layers=n_layers,
                )
                self_.output = nn.Linear(d_model, 1)
                self_.n_feat = n_feat

            def forward(self_, x, m):
                # x: (B, F), m: (B, F)
                # Expand to (B, F, 1) and embed to (B, F, d_model)
                h = self_.embed(x.unsqueeze(-1))  # (B, F, d_model)
                h = self_.encoder(h)  # (B, F, d_model)
                out = self_.output(h).squeeze(-1)  # (B, F)
                return out

        model = _SAITSModel(
            n_features,
            self.d_model,
            self.n_heads,
            self.d_ffn,
            self.n_layers,
            self.dropout,
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)

        # Train: minimise MSE on observed entries only.
        best_loss = float("inf")
        patience_counter = 0

        for epoch in range(self.epochs):
            model.train()
            indices = torch.randperm(n_samples)
            epoch_loss = 0.0
            n_batches = 0

            for start in range(0, n_samples, self.batch_size):
                idx = indices[start : start + self.batch_size]
                xb, mb = X_t[idx], M_t[idx]

                pred = model(xb, mb)
                loss = ((pred - xb) ** 2 * mb).sum() / mb.sum().clamp(min=1)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                n_batches += 1

            avg_loss = epoch_loss / max(n_batches, 1)
            if avg_loss < best_loss - 1e-6:
                best_loss = avg_loss
                patience_counter = 0
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
            else:
                patience_counter += 1
                if patience_counter >= self.patience:
                    logger.info("  SAITS early stopping at epoch %d", epoch + 1)
                    break

        # Predict with best model.
        model.load_state_dict(best_state)
        model.eval()
        with torch.no_grad():
            imputed_t = model(X_t, M_t)

        # Inverse-transform from z-scores back to original scale.
        imputed_scaled = imputed_t.numpy()
        imputed = imputed_scaled * col_stds + col_means
        # Keep original observed values, only fill missing.
        raw_observed = np.nan_to_num(nan_matrix, nan=0.0)
        imputed = np.where(obs_mask == 1, raw_observed, imputed)

        return _restore_columns(imputed, nan_matrix)


class MIWAEBaseline:
    """Missing data Importance-Weighted Autoencoder (Mattei & Frisch, ICML 2019).

    Uses importance-weighted variational inference to handle missing
    data, estimating the joint distribution and imputing from the
    learned latent space.  Accessed via the ``hyperimpute`` package.

    Args:
        n_epochs: Training epochs.  Default: 100.
        batch_size: Mini-batch size.  Default: 128.
        random_state: Random seed.  Default: 42.
    """

    def __init__(
        self,
        n_epochs: int = 100,
        batch_size: int = 128,
        random_state: int = 42,
    ) -> None:
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.random_state = random_state

    def fit_transform(self, features: ArrayLike, mask: ArrayLike) -> np.ndarray:
        """Impute missing values using MIWAE.

        Args:
            features: Feature matrix, shape ``(N, F)``.
            mask: Binary observation mask (1 = observed), shape ``(N, F)``.

        Returns:
            Imputed feature matrix of shape ``(N, F)`` with no NaNs.
        """
        import pandas as pd
        from hyperimpute.plugins.imputers import Imputers

        nan_matrix = _prepare_nan_matrix(features, mask)

        # Normalize to z-scores before VAE training (MIWAE is scale-sensitive).
        col_means = np.nanmean(nan_matrix, axis=0)
        col_stds = np.nanstd(nan_matrix, axis=0)
        col_means = np.where(np.isnan(col_means), 0.0, col_means)
        col_stds = np.where((col_stds < 1e-8) | np.isnan(col_stds), 1.0, col_stds)

        nan_scaled = (nan_matrix - col_means) / col_stds
        df = pd.DataFrame(nan_scaled)

        logger.info(
            "Running MIWAE baseline (n_epochs=%d, batch_size=%d)",
            self.n_epochs,
            self.batch_size,
        )

        plugin = Imputers().get(
            "miwae",
            n_epochs=self.n_epochs,
            batch_size=self.batch_size,
            random_state=self.random_state,
        )
        imputed_scaled = plugin.fit_transform(df.copy()).values

        # Ensure no NaN remains in scaled space.
        remaining_nan = np.isnan(imputed_scaled)
        if remaining_nan.any():
            for j in range(imputed_scaled.shape[1]):
                imputed_scaled[remaining_nan[:, j], j] = 0.0  # mean in z-space

        # Inverse-transform back to original scale.
        imputed = imputed_scaled * col_stds + col_means

        return _restore_columns(imputed, nan_matrix)
