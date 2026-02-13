"""Composite loss functions for GIMIN training.

The total GIMIN training objective combines three complementary losses:

1. **Reconstruction loss** -- Gaussian negative log-likelihood computed only
   on artificially masked observed values (self-supervised signal).
2. **Distribution loss** -- Per-feature KL divergence between the imputed
   and observed marginal distributions, encouraging distributional fidelity.
3. **Cross-modal consistency loss** -- MSE penalty between features that
   should agree across modalities (e.g., FreeSurfer vs. DICOM volumes).

The losses are combined as:

    L_total = L_recon + lambda_dist * L_dist + lambda_cross * L_cross
"""

from __future__ import annotations

import logging

import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812

logger = logging.getLogger(__name__)


class GIMINLoss(nn.Module):
    """Composite loss for GIMIN training.

    Combines reconstruction, distribution-matching, and cross-modal
    consistency objectives into a single differentiable loss.

    Args:
        lambda_dist: Weight for the distribution-matching loss term.
            Default: 0.1.
        lambda_cross: Weight for the cross-modal consistency loss term.
            Default: 0.05.
        cross_modal_pairs: Optional list of (feature_idx_a, feature_idx_b)
            tuples identifying pairs of features that should agree across
            modalities.  When ``None``, the cross-modal loss is skipped.
        eps: Small constant for numerical stability. Default: 1e-6.
    """

    def __init__(
        self,
        lambda_dist: float = 0.1,
        lambda_cross: float = 0.05,
        cross_modal_pairs: list[tuple[int, int]] | None = None,
        binary_feature_indices: list[int] | None = None,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.lambda_dist = lambda_dist
        self.lambda_cross = lambda_cross
        self.cross_modal_pairs = cross_modal_pairs
        self.binary_feature_indices = set(binary_feature_indices or [])
        self.eps = eps

    # ------------------------------------------------------------------
    # Individual loss components
    # ------------------------------------------------------------------

    def reconstruction_loss(
        self,
        pred_mean: torch.Tensor,
        pred_log_var: torch.Tensor,
        true_values: torch.Tensor,
        target_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Heterogeneous reconstruction loss.

        BCE for binary features, Gaussian NLL for continuous.

        Only positions where ``target_mask == 1`` contribute to the loss.
        Binary features use BCE on sigmoid(pred_mean).  Continuous features
        use the heteroscedastic Gaussian NLL formulation.

        Args:
            pred_mean: Predicted mean values (raw logits for binary features),
                shape ``(N, F)``.
            pred_log_var: Predicted log-variance, shape ``(N, F)``.
                Ignored for binary features.
            true_values: Ground-truth feature values, shape ``(N, F)``.
            target_mask: Binary mask (1 = artificially masked position where
                loss is computed), shape ``(N, F)``.

        Returns:
            Scalar loss tensor.
        """
        num_targets = target_mask.sum()
        if num_targets < 1.0:
            return (pred_mean * 0.0).sum()

        num_features = pred_mean.shape[1]
        total_loss = torch.zeros(1, device=pred_mean.device)

        # Build boolean masks for binary vs continuous feature columns.
        binary_cols = torch.zeros(
            num_features, dtype=torch.bool, device=pred_mean.device
        )
        for idx in self.binary_feature_indices:
            if idx < num_features:
                binary_cols[idx] = True
        continuous_cols = ~binary_cols

        # --- Binary features: BCE loss on sigmoid(pred_mean) ---
        if binary_cols.any() and self.binary_feature_indices:
            bin_mask = target_mask[:, binary_cols]
            bin_count = bin_mask.sum()
            if bin_count > 0:
                bin_targets = true_values[:, binary_cols]
                bin_logits = pred_mean[:, binary_cols]
                bce = F.binary_cross_entropy_with_logits(
                    bin_logits, bin_targets, reduction="none"
                )
                total_loss = total_loss + (bce * bin_mask).sum() / bin_count

        # --- Continuous features: Gaussian NLL ---
        if continuous_cols.any():
            cont_mask = target_mask[:, continuous_cols]
            cont_count = cont_mask.sum()
            if cont_count > 0:
                cont_log_var = torch.clamp(
                    pred_log_var[:, continuous_cols], min=-10.0, max=10.0
                )
                variance = torch.exp(cont_log_var) + self.eps
                squared_error = (
                    true_values[:, continuous_cols] - pred_mean[:, continuous_cols]
                ) ** 2
                nll = 0.5 * (cont_log_var + squared_error / variance)
                total_loss = total_loss + (nll * cont_mask).sum() / cont_count

        return total_loss.squeeze()

    def distribution_loss(
        self,
        imputed_values: torch.Tensor,
        observed_values: torch.Tensor,
        mask: torch.Tensor,
        feature_dim: int,
    ) -> torch.Tensor:
        """Per-feature KL divergence between imputed and observed distributions.

        Uses a Gaussian approximation: for each feature, compute mean and
        variance of the observed and imputed marginals, then compute the
        closed-form KL divergence between the two Gaussians.

        .. math::

            \\text{KL}(q \\| p) = \\log\\frac{\\sigma_p}{\\sigma_q}
            + \\frac{\\sigma_q^2 + (\\mu_q - \\mu_p)^2}{2\\sigma_p^2}
            - 0.5

        Args:
            imputed_values: Imputed feature matrix, shape ``(N, F)``.
            observed_values: Original feature matrix, shape ``(N, F)``.
            mask: Binary observation mask (1 = observed), shape ``(N, F)``.
            feature_dim: Total number of features ``F``.

        Returns:
            Mean per-feature KL divergence (scalar).
        """
        mask_bool = mask.bool()  # (N, F)
        counts = mask_bool.sum(dim=0).float()  # (F,)
        valid = counts >= 2  # features with enough observations

        if valid.sum() == 0:
            return (imputed_values * 0.0).sum()

        # Masked means: zero out non-observed, then divide by count.
        obs_masked = observed_values * mask  # (N, F)
        imp_masked = imputed_values * mask  # (N, F)

        counts_safe = counts.clamp(min=1)  # avoid div-by-zero
        mu_p = obs_masked.sum(dim=0) / counts_safe  # (F,)
        mu_q = imp_masked.sum(dim=0) / counts_safe  # (F,)

        # Masked variances (unbiased).
        obs_diff2 = ((observed_values - mu_p.unsqueeze(0)) ** 2) * mask
        imp_diff2 = ((imputed_values - mu_q.unsqueeze(0)) ** 2) * mask
        counts_var = (counts - 1).clamp(min=1)
        var_p = obs_diff2.sum(dim=0) / counts_var + self.eps  # (F,)
        var_q = imp_diff2.sum(dim=0) / counts_var + self.eps  # (F,)

        # KL(q || p) in closed form for univariate Gaussians, per feature.
        kl = (
            0.5 * torch.log(var_p / var_q)
            + (var_q + (mu_q - mu_p) ** 2) / (2.0 * var_p)
            - 0.5
        )  # (F,)

        return kl[valid].mean()

    def cross_modal_consistency_loss(
        self,
        imputed_values: torch.Tensor,
        modality_dims: list[int] | None = None,
    ) -> torch.Tensor:
        """MSE between features that should be correlated across modalities.

        For each pair in ``self.cross_modal_pairs``, compute the MSE between
        the two feature columns in the imputed matrix.  This encourages, for
        example, FreeSurfer-derived volumes and DICOM-derived volumes to
        agree after imputation.

        Args:
            imputed_values: Imputed feature matrix, shape ``(N, F)``.
            modality_dims: List of per-modality feature counts (unused in
                the current implementation but reserved for future
                modality-aware pair selection).

        Returns:
            Mean pairwise MSE (scalar).  Returns 0 if no cross-modal pairs
            are configured.
        """
        if self.cross_modal_pairs is None or len(self.cross_modal_pairs) == 0:
            return (imputed_values * 0.0).sum()

        total_mse = torch.zeros(1, device=imputed_values.device)
        num_pairs = 0

        for idx_a, idx_b in self.cross_modal_pairs:
            if idx_a >= imputed_values.shape[1] or idx_b >= imputed_values.shape[1]:
                logger.warning(
                    "Cross-modal pair (%d, %d) exceeds feature dimension %d; skipping.",
                    idx_a,
                    idx_b,
                    imputed_values.shape[1],
                )
                continue

            feat_a = imputed_values[:, idx_a]
            feat_b = imputed_values[:, idx_b]
            total_mse = total_mse + F.mse_loss(feat_a, feat_b)
            num_pairs += 1

        if num_pairs == 0:
            return (imputed_values * 0.0).sum()

        return (total_mse / num_pairs).squeeze()

    # ------------------------------------------------------------------
    # Combined forward
    # ------------------------------------------------------------------

    def forward(
        self,
        model_output: dict[str, torch.Tensor],
        true_values: torch.Tensor,
        target_mask: torch.Tensor,
        observed_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Compute the composite GIMIN loss.

        Args:
            model_output: Dictionary produced by :class:`~gimin.model.gimin_core.GIMIN`
                containing at least:

                - ``"pred_mean"``: predicted means, shape ``(N, F)``
                - ``"pred_log_var"``: predicted log-variances, shape ``(N, F)``
                - ``"imputed"``: fully-imputed feature matrix, shape ``(N, F)``
            true_values: Original (uncorrupted) feature values, shape ``(N, F)``.
            target_mask: Binary mask marking artificially masked positions
                (1 = was masked for self-supervision), shape ``(N, F)``.
            observed_mask: Binary observation mask for the *original* data
                (1 = genuinely observed), shape ``(N, F)``.

        Returns:
            Dictionary with keys ``"total"``, ``"reconstruction"``,
            ``"distribution"``, and ``"cross_modal"``, each mapping to a
            scalar loss tensor.
        """
        pred_mean = model_output["pred_mean"]
        pred_log_var = model_output["pred_log_var"]
        imputed = model_output["imputed"]

        feature_dim = pred_mean.shape[1]

        # 1. Reconstruction loss (primary self-supervised signal).
        l_recon = self.reconstruction_loss(
            pred_mean, pred_log_var, true_values, target_mask
        )

        # 2. Distribution-matching loss.
        l_dist = self.distribution_loss(
            imputed, true_values, observed_mask, feature_dim
        )

        # 3. Cross-modal consistency loss.
        l_cross = self.cross_modal_consistency_loss(imputed)

        # Weighted combination.
        l_total = l_recon + self.lambda_dist * l_dist + self.lambda_cross * l_cross

        return {
            "total": l_total,
            "reconstruction": l_recon,
            "distribution": l_dist,
            "cross_modal": l_cross,
        }
