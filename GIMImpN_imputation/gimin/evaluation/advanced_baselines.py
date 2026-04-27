"""Advanced baseline imputation methods for benchmarking against GIMIN.

Provides three additional state-of-the-art imputation baselines that
complement the simpler methods in ``baselines.py``.  Each class exposes
the same ``fit_transform(features, mask) -> np.ndarray`` interface used
throughout the GIMIN evaluation pipeline.

Classes:
    MissForestBaseline: MissForest imputation using scikit-learn's
        ``IterativeImputer`` with an ``ExtraTreesRegressor`` estimator.
    GAINBaseline: Generative Adversarial Imputation Network
        (Yoon et al., ICML 2018) implemented in pure PyTorch.
    SoftImputeBaseline: Nuclear-norm / SVD-based matrix completion
        via iterative soft-thresholded SVD (Mazumder et al., JMLR 2010).
"""

from __future__ import annotations

import logging

import numpy as np

from ..utils import ArrayLike
from ..utils import prepare_nan_matrix as _prepare_nan_matrix
from ..utils import to_numpy as _to_numpy

logger = logging.getLogger(__name__)

__all__ = [
    "MissForestBaseline",
    "GAINBaseline",
    "SoftImputeBaseline",
]


# ---------------------------------------------------------------------------
# MissForest
# ---------------------------------------------------------------------------


class MissForestBaseline:
    """MissForest imputation baseline.

    Implements the MissForest algorithm (Stekhoven & Buhlmann, 2012) by
    wrapping scikit-learn's ``IterativeImputer`` with an
    ``ExtraTreesRegressor`` estimator.  The original MissForest paper
    uses random forests with extra-trees-style splitting, which is
    precisely what ``ExtraTreesRegressor`` provides.

    Reference:
        Stekhoven, D. J. and Buhlmann, P. (2012).
        "MissForest -- non-parametric missing value imputation for
        mixed-type data."  *Bioinformatics*, 28(1), 112--118.

    Args:
        max_iter: Maximum number of imputation rounds.  Default: 10.
        n_estimators: Number of trees in the extra-trees ensemble.
            Default: 100.
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
        """Impute missing values using the MissForest algorithm.

        Args:
            features: Feature matrix, shape ``(N, F)``.
            mask: Binary observation mask (1 = observed), shape ``(N, F)``.

        Returns:
            Imputed feature matrix of shape ``(N, F)`` with no NaNs.
        """
        # Lazy imports to avoid hard dependency at module load time.
        from sklearn.ensemble import ExtraTreesRegressor
        from sklearn.experimental import enable_iterative_imputer  # noqa: F401
        from sklearn.impute import IterativeImputer

        nan_matrix = _prepare_nan_matrix(features, mask)

        estimator = ExtraTreesRegressor(
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
            "Running MissForest baseline (max_iter=%d, n_estimators=%d)",
            self.max_iter,
            self.n_estimators,
        )
        imputed = imputer.fit_transform(nan_matrix)
        return imputed


# ---------------------------------------------------------------------------
# GAIN
# ---------------------------------------------------------------------------


class GAINBaseline:
    """Generative Adversarial Imputation Network (GAIN) baseline.

    A pure-PyTorch implementation of the GAIN framework, which trains a
    generator to impute missing values while a discriminator tries to
    distinguish observed entries from imputed ones.  A *hint mechanism*
    reveals partial mask information to the discriminator, preventing
    mode collapse.

    Reference:
        Yoon, J., Jordon, J., and van der Schaar, M. (2018).
        "GAIN: Missing Data Imputation using Generative Adversarial Nets."
        *Proceedings of the 35th International Conference on Machine
        Learning (ICML)*, PMLR 80:5689--5698.

    Args:
        hint_rate: Fraction of mask entries revealed to the discriminator
            via the hint vector.  Default: 0.9.
        alpha: Weight for the reconstruction loss in the generator
            objective.  Default: 100.
        iterations: Number of training iterations.  Default: 300.
        lr: Learning rate for both generator and discriminator Adam
            optimizers.  Default: 1e-3.
        batch_size: Mini-batch size.  If ``None``, the full dataset is
            used as a single batch.  Default: ``None``.
        random_state: Random seed for reproducibility.  Default: 42.
    """

    def __init__(
        self,
        hint_rate: float = 0.9,
        alpha: float = 100.0,
        iterations: int = 300,
        lr: float = 1e-3,
        batch_size: int | None = None,
        random_state: int = 42,
    ) -> None:
        self.hint_rate = hint_rate
        self.alpha = alpha
        self.iterations = iterations
        self.lr = lr
        self.batch_size = batch_size
        self.random_state = random_state

    # -- internal network builders ------------------------------------------

    @staticmethod
    def _build_generator(input_dim: int, torch_nn):
        """Build the generator network.

        Architecture:
            [input_dim * 2] -> 256 (ReLU) -> 256 (ReLU) -> input_dim (Sigmoid)

        The generator receives the concatenation of the corrupted features
        (observed values + noise at missing positions) and the mask.
        """
        return torch_nn.Sequential(
            torch_nn.Linear(input_dim * 2, 256),
            torch_nn.ReLU(),
            torch_nn.Linear(256, 256),
            torch_nn.ReLU(),
            torch_nn.Linear(256, input_dim),
            torch_nn.Sigmoid(),
        )

    @staticmethod
    def _build_discriminator(input_dim: int, torch_nn):
        """Build the discriminator network.

        Architecture:
            [input_dim * 2] -> 256 (ReLU) -> 256 (ReLU) -> input_dim (Sigmoid)

        The discriminator receives the concatenation of the (possibly
        imputed) feature matrix and the hint vector.
        """
        return torch_nn.Sequential(
            torch_nn.Linear(input_dim * 2, 256),
            torch_nn.ReLU(),
            torch_nn.Linear(256, 256),
            torch_nn.ReLU(),
            torch_nn.Linear(256, input_dim),
            torch_nn.Sigmoid(),
        )

    # -- public interface ---------------------------------------------------

    def fit_transform(self, features: ArrayLike, mask: ArrayLike) -> np.ndarray:
        """Impute missing values using GAIN.

        Trains the generator and discriminator from scratch, then returns
        the imputed feature matrix.

        Args:
            features: Feature matrix, shape ``(N, F)``.
            mask: Binary observation mask (1 = observed), shape ``(N, F)``.

        Returns:
            Imputed feature matrix of shape ``(N, F)`` with no NaNs.
        """
        # Lazy import to avoid hard dependency at module load time.
        import torch
        import torch.nn as nn

        features_np = _to_numpy(features).copy()
        mask_np = _to_numpy(mask).astype(np.float64)

        N, F = features_np.shape  # noqa: N806

        # Normalise features to [0, 1] per column for sigmoid output.
        col_min = np.nanmin(np.where(mask_np == 1, features_np, np.nan), axis=0)
        col_max = np.nanmax(np.where(mask_np == 1, features_np, np.nan), axis=0)
        col_range = col_max - col_min
        col_range[col_range == 0] = 1.0  # avoid division by zero

        normed = (features_np - col_min) / col_range
        normed = np.nan_to_num(normed, nan=0.0)

        # Convert to tensors.
        torch.manual_seed(self.random_state)
        device = torch.device("cpu")

        X = torch.tensor(normed, dtype=torch.float32, device=device)  # noqa: N806
        M = torch.tensor(mask_np, dtype=torch.float32, device=device)  # noqa: N806

        # Build networks.
        G = self._build_generator(F, nn).to(device)  # noqa: N806
        D = self._build_discriminator(F, nn).to(device)  # noqa: N806

        opt_G = torch.optim.Adam(G.parameters(), lr=self.lr)
        opt_D = torch.optim.Adam(D.parameters(), lr=self.lr)

        bce = nn.BCELoss(reduction="none")
        rng = np.random.default_rng(self.random_state)

        logger.info(
            "Running GAIN baseline (iterations=%d, hint_rate=%.2f, "
            "alpha=%.1f, lr=%.1e)",
            self.iterations,
            self.hint_rate,
            self.alpha,
            self.lr,
        )

        batch_size = self.batch_size if self.batch_size is not None else N

        for it in range(self.iterations):
            # -- sample a mini-batch ----------------------------------------
            idx = rng.choice(N, size=batch_size, replace=False)
            X_mb = X[idx]  # noqa: N806
            M_mb = M[idx]  # noqa: N806

            # Noise for missing entries.
            Z = torch.tensor(  # noqa: N806
                rng.uniform(0, 0.01, size=(batch_size, F)),
                dtype=torch.float32,
                device=device,
            )

            # Hint vector: reveal mask at hint_rate fraction of positions.
            B = torch.tensor(  # noqa: N806
                (rng.random((batch_size, F)) < self.hint_rate).astype(np.float32),
                dtype=torch.float32,
                device=device,
            )
            H = B * M_mb + 0.5 * (1 - B)  # noqa: N806

            # Generator input: observed values + noise at missing spots.
            G_input = X_mb * M_mb + Z * (1 - M_mb)  # noqa: N806
            G_input = torch.cat([G_input, M_mb], dim=1)

            # -- Generator forward ------------------------------------------
            G_sample = G(G_input)  # noqa: N806

            # Imputed matrix: keep observed, fill missing with generated.
            X_hat = X_mb * M_mb + G_sample * (1 - M_mb)  # noqa: N806

            # -- Discriminator forward --------------------------------------
            D_input = torch.cat([X_hat.detach(), H], dim=1)  # noqa: N806
            D_prob = D(D_input)  # noqa: N806

            # -- Discriminator loss -----------------------------------------
            D_loss = -torch.mean(  # noqa: N806
                M_mb * torch.log(D_prob + 1e-8)
                + (1 - M_mb) * torch.log(1 - D_prob + 1e-8)
            )

            opt_D.zero_grad()
            D_loss.backward()
            opt_D.step()

            # -- Generator loss ---------------------------------------------
            D_input_g = torch.cat([X_hat, H], dim=1)  # noqa: N806
            D_prob_g = D(D_input_g)  # noqa: N806

            # Adversarial loss: fool D on missing entries.
            G_loss_adv = -torch.mean(  # noqa: N806
                (1 - M_mb) * torch.log(D_prob_g + 1e-8)
            )
            # Reconstruction loss: match observed entries.
            G_loss_rec = torch.mean(  # noqa: N806
                M_mb * (X_mb - G_sample) ** 2
            )
            G_loss = G_loss_adv + self.alpha * G_loss_rec  # noqa: N806

            opt_G.zero_grad()
            G_loss.backward()
            opt_G.step()

            if (it + 1) % 100 == 0:
                logger.debug(
                    "GAIN iter %d/%d  D_loss=%.4f  G_loss=%.4f",
                    it + 1,
                    self.iterations,
                    D_loss.item(),
                    G_loss.item(),
                )

        # -- Final imputation -----------------------------------------------
        with torch.no_grad():
            Z_final = torch.tensor(  # noqa: N806
                rng.uniform(0, 0.01, size=(N, F)),
                dtype=torch.float32,
                device=device,
            )
            G_input_final = X * M + Z_final * (1 - M)  # noqa: N806
            G_input_final = torch.cat([G_input_final, M], dim=1)
            G_sample_final = G(G_input_final)  # noqa: N806

            imputed_normed = X * M + G_sample_final * (1 - M)

        imputed_normed = imputed_normed.cpu().numpy().astype(np.float64)

        # De-normalise back to original scale.
        imputed = imputed_normed * col_range + col_min

        logger.info("GAIN baseline imputation complete.")
        return imputed


# ---------------------------------------------------------------------------
# SoftImpute
# ---------------------------------------------------------------------------


class SoftImputeBaseline:
    """SoftImpute / SVD-based matrix-completion baseline.

    A pure NumPy/SciPy implementation of the SoftImpute algorithm, which
    iteratively replaces missing values with predictions from a
    low-rank SVD approximation of the matrix until convergence.

    Reference:
        Mazumder, R., Hastie, T., and Tibshirani, R. (2010).
        "Spectral Regularization Algorithms for Learning Large Incomplete
        Matrices."  *Journal of Machine Learning Research*, 11,
        2287--2322.

    Args:
        max_rank: Maximum rank for the truncated SVD approximation.
            Default: 10.
        max_iter: Maximum number of iterations.  Default: 100.
        convergence_threshold: Relative change in Frobenius norm below
            which convergence is declared.  Default: 1e-5.
    """

    def __init__(
        self,
        max_rank: int = 10,
        max_iter: int = 100,
        convergence_threshold: float = 1e-5,
    ) -> None:
        self.max_rank = max_rank
        self.max_iter = max_iter
        self.convergence_threshold = convergence_threshold

    def fit_transform(self, features: ArrayLike, mask: ArrayLike) -> np.ndarray:
        """Impute missing values using iterative SVD completion.

        Args:
            features: Feature matrix, shape ``(N, F)``.
            mask: Binary observation mask (1 = observed), shape ``(N, F)``.

        Returns:
            Imputed feature matrix of shape ``(N, F)`` with no NaNs.
        """
        # Lazy import to avoid hard dependency at module load time.
        from scipy.linalg import svd

        features_np = _to_numpy(features).copy()
        mask_np = _to_numpy(mask).astype(bool)

        N, F = features_np.shape  # noqa: N806
        rank = min(self.max_rank, N, F)

        # Initialise missing entries with column means of observed values.
        col_means = np.zeros(F, dtype=np.float64)
        for j in range(F):
            observed = features_np[mask_np[:, j], j]
            col_means[j] = observed.mean() if len(observed) > 0 else 0.0
        filled = features_np.copy()
        filled[~mask_np] = np.take(col_means, np.where(~mask_np)[1])

        logger.info(
            "Running SoftImpute baseline (max_rank=%d, max_iter=%d, tol=%.1e)",
            rank,
            self.max_iter,
            self.convergence_threshold,
        )

        prev_norm = np.linalg.norm(filled, "fro")

        for it in range(self.max_iter):
            # Step 1-2: Compute truncated SVD.
            U, s, Vt = svd(filled, full_matrices=False)  # noqa: N806
            U = U[:, :rank]  # noqa: N806
            s = s[:rank]
            Vt = Vt[:rank, :]  # noqa: N806

            # Step 3: Reconstruct from truncated SVD.
            reconstructed = U @ np.diag(s) @ Vt

            # Step 4: Keep observed values, update only missing positions.
            filled[~mask_np] = reconstructed[~mask_np]

            # Step 5: Check convergence.
            current_norm = np.linalg.norm(filled, "fro")
            if prev_norm > 0:
                rel_change = np.linalg.norm(filled - reconstructed, "fro") / prev_norm
            else:
                rel_change = 0.0

            if (it + 1) % 20 == 0:
                logger.debug(
                    "SoftImpute iter %d/%d  rel_change=%.2e",
                    it + 1,
                    self.max_iter,
                    rel_change,
                )

            if rel_change < self.convergence_threshold:
                logger.info(
                    "SoftImpute converged at iteration %d (rel_change=%.2e < %.2e)",
                    it + 1,
                    rel_change,
                    self.convergence_threshold,
                )
                break

            prev_norm = current_norm

        logger.info("SoftImpute baseline imputation complete.")
        return filled
