"""Thin wrapper adapter for de Rooij et al. 2025 UDE physiology-informed regularization.

Reference
---------
de Rooij M, Erdős B, van Riel N, O'Donovan S. 2025.
"Physiology-informed regularisation enables training of universal differential
equation systems for biological applications."
PLOS Computational Biology. DOI: 10.1371/journal.pcbi.1012198

Vendor notes
------------
The upstream implementation is in Julia (Lux + SciML + DifferentialEquations). The
vendored code at ``baselines/derooij_2025/`` must remain unmodified for future
``git subtree pull`` syncs. See
``paper12_phys_gimin/baselines/derooij_2025/VENDOR_NOTES.md`` for full attribution.

Adaptation strategy
-------------------
De Rooij's method trains a Universal Differential Equation (UDE) where a neural
network replaces unknown mechanistic terms, and two physiology-informed regularizers
are applied during training:

  1. **AUC regularizer** (λ_AUC): penalises |∫ RA(τ)dτ - 1|, enforcing that the
     learned meal-appearance function integrates to 1 (conservation of mass).
  2. **Non-negativity regularizer** (λ_nonneg): penalises ∑ min(0, RA(τ))², enforcing
     that the meal-appearance rate is always non-negative (physiological constraint).

For PPMI 33-feature imputation we cannot faithfully replicate the ODE structure (which
requires time-series glucose + insulin), so this adapter implements the **regularization
pattern** from de Rooij in a Python neural-network imputer:

  - A multi-layer perceptron imputer is trained on observed entries.
  - Physiology-informed regularization is applied per-feature if a prior bound is
    registered (non-negativity for strictly-positive biomarkers; AUC constraint for
    features interpretable as area-under-curve quantities like UPDRS subscores).
  - This preserves the *spirit* of de Rooij — physiology-informed constraints
    during training — while adapting to the PPMI tabular setting.

Sigma estimation
----------------
De Rooij's reference implementation produces **point estimates only** (no posterior
uncertainty). This adapter estimates imputation uncertainty as the **empirical standard
deviation of predictions across a bootstrap ensemble** (n_bootstrap=20 by default,
seeded for determinism). If bootstrap is disabled (n_bootstrap=1), σ is set to
``sigma_fallback`` (a per-feature constant derived from training-set observed variance).

This choice is documented explicitly so downstream calibration analysis can account
for the difference between GIMIN's heteroscedastic decoder σ and the bootstrap σ here.

Usage
-----
    from phys_gimin.baseline_adapters import DeRooijImputer

    imputer = DeRooijImputer(regularization_strength=1.0, n_iterations=500)
    imputer.fit(features_train, mask_train)
    mu, sigma = imputer.impute(features_test, mask_test)
"""
from __future__ import annotations

import warnings
from pathlib import Path
from typing import Optional

import numpy as np

# ---------------------------------------------------------------------------
# Optional torch dependency — required for MLP imputer. Fail gracefully so
# that test_adapter_importable passes even in minimal envs.
# ---------------------------------------------------------------------------
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim

    _TORCH_AVAILABLE = True
except ImportError:  # pragma: no cover
    _TORCH_AVAILABLE = False


# PPMI 33-feature physiology priors (de Rooij-style non-negativity constraints)
# Features that are strictly non-negative in physiology get λ_nonneg > 0.
# Features without a meaningful lower bound get 0.
PPMI_NONNEG_FEATURES: dict[int, float] = {
    # Index: (feature_name, why non-negative)
    # Demographics: age (index 0) is non-negative
    0: 1.0,   # AGE_AT_BASELINE
    # UPDRS subscales: all non-negative (0 = no symptom)
    3: 1.0,   # UPDRS1_TOTAL
    4: 1.0,   # UPDRS2_TOTAL
    5: 1.0,   # UPDRS3_TREMOR
    6: 1.0,   # UPDRS3_RIGIDITY
    7: 1.0,   # UPDRS3_BRADYKINESIA
    8: 1.0,   # UPDRS3_AXIAL
    # DaT imaging: SBR values are non-negative
    9: 1.0,   # CAUDATE_SBR_MEAN
    10: 1.0,  # CAUDATE_PUTAMEN_RATIO
    # Sleep / autonomic: non-negative counts
    11: 1.0,  # ESS_TOTAL
    12: 1.0,  # RBD_TOTAL
    13: 1.0,  # SCOPA_AUT_TOTAL
}

# AUC constraint: features interpretable as "rate over session" (UPDRS-III subscales)
# AUC regularizer pushes ∑ predicted_f ≈ 1 (normalised), analogous to meal appearance.
# Not directly applicable to tabular imputation — disabled by default (λ_AUC=0).


class _PhysMLPImputer(nn.Module if _TORCH_AVAILABLE else object):
    """Simple MLP imputer with optional physiology-informed regularization.

    Implements de Rooij's regularization pattern in the PyTorch ecosystem:
    - MSE reconstruction loss on observed entries
    - Non-negativity penalty on imputed outputs for physiologically bounded features
    - AUC penalty (optional; off by default for tabular PPMI data)
    """

    def __init__(
        self,
        n_features: int,
        hidden_dim: int = 128,
        n_layers: int = 3,
        lambda_nonneg: float = 1.0,
        lambda_auc: float = 0.0,
        nonneg_feature_indices: Optional[dict[int, float]] = None,
    ) -> None:
        super().__init__()
        self.n_features = n_features
        self.lambda_nonneg = lambda_nonneg
        self.lambda_auc = lambda_auc
        self.nonneg_feature_indices = nonneg_feature_indices or {}

        # Build MLP: input = (feature_values * mask + mask_indicator) → output = full_features
        layers: list[nn.Module] = []
        in_dim = n_features * 2  # observed values (zeroed for missing) + mask
        for i in range(n_layers):
            out_dim = hidden_dim if i < n_layers - 1 else n_features
            layers.append(nn.Linear(in_dim, out_dim))
            if i < n_layers - 1:
                layers.append(nn.ReLU())
            in_dim = hidden_dim
        self.net = nn.Sequential(*layers)

    def forward(self, x: "torch.Tensor", mask: "torch.Tensor") -> "torch.Tensor":
        """Forward pass: reconstruct full feature vector from partial observations.

        Args:
            x: (batch, n_features) — observed values (missing entries zeroed).
            mask: (batch, n_features) — 1=observed, 0=missing.

        Returns:
            (batch, n_features) — full reconstructed features.
        """
        inp = torch.cat([x * mask, mask], dim=-1)
        return self.net(inp)

    def loss(
        self,
        preds: "torch.Tensor",
        x_obs: "torch.Tensor",
        mask: "torch.Tensor",
    ) -> "torch.Tensor":
        """Compute de Rooij-style regularized training loss.

        Loss = MSE_observed + λ_nonneg * Σ_j min(0, pred_j)² + λ_AUC * |mean(pred_j) - 1|

        Only observed entries contribute to MSE (equivalent to de Rooij's
        per-timepoint fit term). Regularizers apply to ALL predicted entries,
        including imputed ones — this is the physiology-informed constraint.

        Args:
            preds: (batch, n_features) — model output.
            x_obs: (batch, n_features) — observed values (missing zeroed).
            mask: (batch, n_features) — 1=observed, 0=missing.

        Returns:
            Scalar loss tensor.
        """
        # Reconstruction: MSE on observed entries only
        recon = ((preds - x_obs) ** 2 * mask).sum() / (mask.sum() + 1e-8)

        # Non-negativity regularizer (de Rooij Eq. nonneg): Σ min(0, pred)²
        nonneg_loss = torch.tensor(0.0, device=preds.device)
        for feat_idx, strength in self.nonneg_feature_indices.items():
            feat_preds = preds[:, feat_idx]
            nonneg_loss = nonneg_loss + strength * (torch.minimum(feat_preds, torch.zeros_like(feat_preds)) ** 2).mean()
        nonneg_loss = self.lambda_nonneg * nonneg_loss

        # AUC regularizer (de Rooij Eq. AUC): penalise |mean(pred) - 1|
        # Disabled by default (lambda_auc=0); included for API completeness.
        auc_loss = torch.tensor(0.0, device=preds.device)
        if self.lambda_auc > 0:
            auc_loss = self.lambda_auc * torch.abs(preds.mean() - 1.0)

        return recon + nonneg_loss + auc_loss


class DeRooijImputer:
    """Thin wrapper around de Rooij et al. 2025 UDE regularization for PPMI imputation.

    Accepts PPMI 33-feature (features, mask) input and returns (imputed_mean, imputed_sigma)
    using de Rooij's physiology-informed regularization pattern.

    The upstream de Rooij reference is Julia/SciML (UDE with ODE solver). This Python
    wrapper implements the same REGULARIZATION PATTERN (non-negativity + AUC constraints
    on the training objective) adapted to the PPMI tabular imputation setting, which
    lacks an ODE structure.

    Upstream reference implementation is vendored at:
        paper12_phys_gimin/baselines/derooij_2025/
    and must NOT be modified. See VENDOR_NOTES.md for attribution.

    Uncertainty estimation
    ----------------------
    De Rooij's reference implementation produces point estimates only. This adapter
    estimates σ via bootstrap ensemble (n_bootstrap=20 by default). The resulting σ
    is an *epistemic* uncertainty over the imputer parameters, not the heteroscedastic
    aleatoric σ from GIMIN's decoder. Downstream calibration code should note this
    distinction.

    Requires
    --------
    PyTorch (``pip install torch``). Raises ``ImportError`` on ``fit()`` if unavailable.
    """

    VENDOR_PATH: Path = (
        Path(__file__).parent.parent.parent.parent
        / "baselines"
        / "derooij_2025"
    )

    def __init__(
        self,
        regularization_strength: float = 1.0,
        n_iterations: int = 500,
        device: str = "cpu",
        hidden_dim: int = 128,
        n_layers: int = 3,
        lambda_nonneg: float = 1.0,
        lambda_auc: float = 0.0,
        n_bootstrap: int = 20,
        sigma_fallback: float = 0.1,
        seed: int = 42,
        nonneg_feature_indices: Optional[dict[int, float]] = None,
        lr: float = 1e-3,
    ) -> None:
        """Initialise the de Rooij PPMI adapter.

        Args:
            regularization_strength: Global multiplier on both λ_nonneg and λ_AUC.
                Matches de Rooij's sweep over λ ∈ {0, 0.01, 0.1, 1, 10, 100}.
            n_iterations: Gradient-descent training steps. Paper default: 500 (ADAM)
                + 1000 (BFGS). We use ADAM-only for speed; set higher for final runs.
            device: PyTorch device string ('cpu', 'mps', 'cuda').
            hidden_dim: Hidden layer width (MLP).
            n_layers: Number of hidden layers (MLP).
            lambda_nonneg: Weight on de Rooij non-negativity regularizer.
                Scaled by ``regularization_strength``.
            lambda_auc: Weight on de Rooij AUC regularizer. Disabled (0) by default
                for tabular PPMI — the AUC constraint assumes a time-integral
                interpretation not directly applicable to cross-sectional features.
            n_bootstrap: Number of bootstrap replicates for σ estimation.
                Set to 1 to use sigma_fallback instead.
            sigma_fallback: Default σ per feature if n_bootstrap=1.
            seed: Random seed for reproducibility.
            nonneg_feature_indices: Override PPMI feature index → strength mapping.
                Defaults to PPMI_NONNEG_FEATURES (non-negative biomarker indices).
            lr: Adam learning rate.
        """
        self.regularization_strength = regularization_strength
        self.n_iterations = n_iterations
        self.device = device
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.lambda_nonneg = lambda_nonneg * regularization_strength
        self.lambda_auc = lambda_auc * regularization_strength
        self.n_bootstrap = n_bootstrap
        self.sigma_fallback = sigma_fallback
        self.seed = seed
        self.lr = lr
        self.nonneg_feature_indices = (
            nonneg_feature_indices if nonneg_feature_indices is not None
            else PPMI_NONNEG_FEATURES
        )

        self._models: list[_PhysMLPImputer] = []
        self._col_means: Optional[np.ndarray] = None
        self._col_stds: Optional[np.ndarray] = None
        self._n_features: Optional[int] = None
        self._fitted = False

    def _check_torch(self) -> None:
        if not _TORCH_AVAILABLE:
            raise ImportError(
                "DeRooijImputer requires PyTorch. "
                "Install with: pip install torch"
            )

    def _normalize(
        self, features: np.ndarray, mask: np.ndarray
    ) -> np.ndarray:
        """Z-score normalize using observed-entry statistics."""
        assert self._col_means is not None and self._col_stds is not None
        normed = (features - self._col_means) / (self._col_stds + 1e-8)
        return np.where(mask == 1, normed, 0.0)

    def _denormalize(self, normed: np.ndarray) -> np.ndarray:
        assert self._col_means is not None and self._col_stds is not None
        return normed * (self._col_stds + 1e-8) + self._col_means

    def _train_single(
        self,
        features_norm: np.ndarray,
        mask: np.ndarray,
        rng: np.random.Generator,
    ) -> "_PhysMLPImputer":
        """Train one MLP imputer on a bootstrap resample."""
        self._check_torch()
        n, d = features_norm.shape
        bootstrap_idx = rng.integers(0, n, size=n)
        x_boot = features_norm[bootstrap_idx]
        m_boot = mask[bootstrap_idx]

        x_t = torch.tensor(x_boot, dtype=torch.float32)
        m_t = torch.tensor(m_boot, dtype=torch.float32)

        model = _PhysMLPImputer(
            n_features=d,
            hidden_dim=self.hidden_dim,
            n_layers=self.n_layers,
            lambda_nonneg=self.lambda_nonneg,
            lambda_auc=self.lambda_auc,
            nonneg_feature_indices=self.nonneg_feature_indices,
        ).to(self.device)

        opt = optim.Adam(model.parameters(), lr=self.lr)
        for _ in range(self.n_iterations):
            opt.zero_grad()
            preds = model(x_t, m_t)
            loss = model.loss(preds, x_t, m_t)
            loss.backward()
            opt.step()

        model.eval()
        return model

    def fit(self, features: np.ndarray, mask: np.ndarray) -> "DeRooijImputer":
        """Fit on observed entries (mask==1) of features.

        Computes per-feature z-score statistics from observed entries, then trains
        ``n_bootstrap`` MLP imputers on bootstrap resamples of the training data.
        Multiple models are used to estimate epistemic uncertainty (σ) at impute time.

        Args:
            features: (n_patients, n_features) float32 feature matrix.
            mask: (n_patients, n_features) binary mask — 1 = observed, 0 = missing.

        Returns:
            self (for method chaining).
        """
        self._check_torch()
        features = np.asarray(features, dtype=np.float32)
        mask = np.asarray(mask, dtype=np.float32)
        n, d = features.shape
        self._n_features = d

        # Compute per-feature statistics from observed entries only
        self._col_means = np.zeros(d, dtype=np.float32)
        self._col_stds = np.ones(d, dtype=np.float32)
        for j in range(d):
            obs = features[:, j][mask[:, j] == 1]
            if len(obs) > 0:
                self._col_means[j] = float(obs.mean())
                self._col_stds[j] = float(obs.std()) if obs.std() > 0 else 1.0

        features_norm = self._normalize(features, mask)

        rng = np.random.default_rng(self.seed)
        self._models = []
        n_trains = max(1, self.n_bootstrap)
        for _ in range(n_trains):
            mdl = self._train_single(features_norm, mask, rng)
            self._models.append(mdl)

        self._fitted = True
        return self

    def impute(
        self, features: np.ndarray, mask: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Impute missing entries.

        Runs all bootstrap models on the test features and aggregates:
        - ``imputed_mean`` = mean across models (de-normalized).
        - ``imputed_sigma`` = std across models (if n_bootstrap > 1) OR
          ``sigma_fallback * col_stds`` (if n_bootstrap = 1).

        Observed entries are returned unchanged (mask==1 locations).

        Args:
            features: (n_patients, n_features) float32 feature matrix.
            mask: (n_patients, n_features) binary mask — 1 = observed, 0 = missing.

        Returns:
            imputed_mean: (n_patients, n_features) — phys-GIMIN-compatible mu output.
            imputed_sigma: (n_patients, n_features) — phys-GIMIN-compatible sigma output.

        Raises:
            RuntimeError: if called before ``fit()``.
        """
        if not self._fitted:
            raise RuntimeError("Call fit() before impute().")
        self._check_torch()

        features = np.asarray(features, dtype=np.float32)
        mask = np.asarray(mask, dtype=np.float32)

        features_norm = self._normalize(features, mask)
        x_t = torch.tensor(features_norm, dtype=torch.float32, device=self.device)
        m_t = torch.tensor(mask, dtype=torch.float32, device=self.device)

        preds_list: list[np.ndarray] = []
        with torch.no_grad():
            for mdl in self._models:
                preds_norm = mdl(x_t, m_t).cpu().numpy()
                preds_denorm = self._denormalize(preds_norm)
                preds_list.append(preds_denorm)

        preds_stack = np.stack(preds_list, axis=0)  # (n_bootstrap, n_patients, n_features)
        mean_pred = preds_stack.mean(axis=0)

        if len(self._models) > 1:
            sigma_pred = preds_stack.std(axis=0)
        else:
            # Fallback: sigma_fallback fraction of per-feature std
            sigma_pred = np.ones_like(mean_pred) * self.sigma_fallback * (
                self._col_stds[None, :] + 1e-8
            )

        # Restore observed entries (do not impute what was observed)
        imputed_mean = np.where(mask == 1, features, mean_pred)
        imputed_sigma = sigma_pred  # sigma applies everywhere (observed σ is noise floor)

        # Ensure finite values
        imputed_mean = np.where(np.isfinite(imputed_mean), imputed_mean, self._col_means)
        imputed_sigma = np.where(
            np.isfinite(imputed_sigma) & (imputed_sigma > 0),
            imputed_sigma,
            self.sigma_fallback,
        )

        return imputed_mean.astype(np.float32), imputed_sigma.astype(np.float32)
