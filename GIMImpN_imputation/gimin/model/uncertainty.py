"""Uncertainty quantification via MC dropout and heteroscedastic output.

GIMIN produces both point estimates and uncertainty estimates for every
imputed value.  Uncertainty is captured in two complementary ways:

1. **Heteroscedastic variance**: The decoder outputs a mean and a log-variance
   for each feature, modelling *aleatoric* (data-inherent) uncertainty.
2. **MC Dropout**: At inference time, dropout is kept active and multiple
   forward passes are performed.  The variance across these stochastic
   passes captures *epistemic* (model) uncertainty.

Functions and classes:
    MCDropoutWrapper: Utility that enables dropout during inference and
        aggregates predictions over multiple stochastic forward passes.
    calibrate_uncertainty: Compute calibration metrics comparing predicted
        uncertainty to observed imputation error.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import torch
import torch.nn as nn

_LOG_2PI = math.log(2.0 * math.pi)


class MCDropoutWrapper:
    """Enable MC Dropout at inference time and aggregate stochastic predictions.

    Usage::

        wrapper = MCDropoutWrapper(model, num_samples=50)
        results = wrapper.predict_with_uncertainty(data)
        mean = results["mean"]     # (N, F) point estimate
        std  = results["std"]      # (N, F) total uncertainty

    The wrapper temporarily switches dropout layers to training mode while
    keeping batch-normalisation and layer-normalisation in eval mode, then
    runs the model *num_samples* times with different dropout masks.

    Args:
        model: A GIMIN model instance (or any ``nn.Module`` whose
            ``forward`` returns a dict with ``"imputed_mean"`` and
            ``"imputed_log_var"`` keys).
        num_samples: Number of stochastic forward passes. Default: 50.
    """

    def __init__(self, model: nn.Module, num_samples: int = 50) -> None:
        self.model = model
        self.num_samples = num_samples

    @staticmethod
    def _enable_mc_dropout(model: nn.Module) -> None:
        """Set all Dropout layers to training mode (stochastic)."""
        for module in model.modules():
            if isinstance(module, (nn.Dropout, nn.Dropout2d, nn.Dropout3d)):
                module.train()

    @staticmethod
    def _restore_eval(model: nn.Module) -> None:
        """Restore the entire model to eval mode."""
        model.eval()

    @torch.no_grad()
    def predict_with_uncertainty(
        self,
        features: torch.Tensor,
        mask: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor,
        overlap_frac: torch.Tensor,
        modality_dims: list[int],
    ) -> dict[str, torch.Tensor]:
        """Run MC Dropout inference and aggregate results.

        Args:
            features: Patient feature matrix, shape (N, F).
            mask: Binary observation mask, shape (N, F).
            edge_index: Graph edge connectivity, shape (2, E).
            edge_weight: Edge weights, shape (E,).
            overlap_frac: Per-edge feature overlap fractions, shape (E,).
            modality_dims: List of per-modality feature counts.

        Returns:
            Dictionary with keys:

            - ``"mean"``: Averaged imputed values across MC samples, (N, F).
            - ``"std"``: Total standard deviation (epistemic + aleatoric),
              (N, F).
            - ``"epistemic_std"``: Standard deviation of the means across MC
              samples (epistemic uncertainty), (N, F).
            - ``"aleatoric_std"``: Mean of the per-sample predicted standard
              deviations (aleatoric uncertainty), (N, F).
        """
        self.model.eval()
        self._enable_mc_dropout(self.model)

        means = []
        aleatoric_vars = []

        for _ in range(self.num_samples):
            output = self.model(
                features=features,
                mask=mask,
                edge_index=edge_index,
                edge_weight=edge_weight,
                overlap_frac=overlap_frac,
                modality_dims=modality_dims,
            )
            means.append(output["imputed_mean"])
            # Convert log_var to variance for averaging.
            aleatoric_vars.append(torch.exp(output["imputed_log_var"]))

        self._restore_eval(self.model)

        # Stack samples: (S, N, F)
        means_stack = torch.stack(means, dim=0)
        aleatoric_vars_stack = torch.stack(aleatoric_vars, dim=0)

        # Epistemic uncertainty: variance of means across samples.
        epistemic_var = means_stack.var(dim=0)  # (N, F)

        # Aleatoric uncertainty: average predicted variance.
        mean_aleatoric_var = aleatoric_vars_stack.mean(dim=0)  # (N, F)

        # Total uncertainty: sum of epistemic + aleatoric variance.
        total_var = epistemic_var + mean_aleatoric_var  # (N, F)

        # Point estimate: average of means.
        mean_prediction = means_stack.mean(dim=0)  # (N, F)

        return {
            "mean": mean_prediction,
            "std": torch.sqrt(total_var.clamp(min=1e-8)),
            "epistemic_std": torch.sqrt(epistemic_var.clamp(min=1e-8)),
            "aleatoric_std": torch.sqrt(mean_aleatoric_var.clamp(min=1e-8)),
        }


def calibrate_uncertainty(
    predicted_means: torch.Tensor,
    predicted_stds: torch.Tensor,
    true_values: torch.Tensor,
    mask: torch.Tensor,
    quantile_bins: int = 10,
) -> dict[str, Any]:
    """Evaluate calibration of uncertainty estimates.

    A well-calibrated model should have its predicted confidence intervals
    contain the true values at the advertised rate.  For example, the
    predicted 90% interval should contain ~90% of ground-truth values.

    This function computes:

    1. **Expected Calibration Error (ECE)**: Mean absolute difference
       between expected and observed coverage across quantile bins.
    2. **Per-bin coverage**: Observed fraction of true values falling
       within the predicted interval at each confidence level.
    3. **Negative Log-Likelihood (NLL)**: Average Gaussian NLL of true
       values under the predicted (mean, std).

    Only features marked as missing in *mask* (i.e., ``mask == 0``) are
    evaluated, since those are the ones being imputed.

    Args:
        predicted_means: Model's imputed means, shape (N, F).
        predicted_stds: Model's predicted standard deviations, shape (N, F).
        true_values: Ground-truth feature values, shape (N, F).
        mask: Binary observation mask, shape (N, F). Entries with
            ``mask == 0`` are the imputed (and therefore evaluated) features.
        quantile_bins: Number of quantile bins for the calibration curve.
            Default: 10.

    Returns:
        Dictionary with keys:

        - ``"ece"``: Expected Calibration Error (float).
        - ``"expected_coverage"``: Array of expected coverage levels, shape
          (quantile_bins,).
        - ``"observed_coverage"``: Array of observed coverage fractions,
          shape (quantile_bins,).
        - ``"nll"``: Mean Gaussian negative log-likelihood (float).
    """
    # Select only imputed (missing) features for evaluation.
    imputed_mask = mask == 0

    if imputed_mask.sum() == 0:
        return {
            "ece": 0.0,
            "expected_coverage": np.linspace(0.1, 1.0, quantile_bins),
            "observed_coverage": np.ones(quantile_bins),
            "nll": 0.0,
        }

    pred_mean = predicted_means[imputed_mask]  # (K,)
    pred_std = predicted_stds[imputed_mask].clamp(min=1e-8)  # (K,)
    true_val = true_values[imputed_mask]  # (K,)

    # Standardized residuals.
    z_scores = ((true_val - pred_mean) / pred_std).abs()  # (K,)

    # --- Calibration curve ---
    expected_coverages = np.linspace(1.0 / quantile_bins, 1.0, quantile_bins)
    observed_coverages = []

    for p in expected_coverages:
        # For a Gaussian, the interval [-z_p, z_p] covers fraction p of mass.
        # z_p = Phi^{-1}((1 + p) / 2)
        z_threshold = torch.erfinv(torch.tensor(p, dtype=torch.float32)) * (2.0**0.5)
        fraction_inside = (z_scores <= z_threshold).float().mean().item()
        observed_coverages.append(fraction_inside)

    observed_coverages_arr = np.array(observed_coverages)
    ece = float(np.mean(np.abs(expected_coverages - observed_coverages_arr)))

    # --- Gaussian NLL ---
    # NLL = 0.5 * (log(2*pi) + log(var) + (x - mu)^2 / var)
    variance = pred_std**2
    nll = 0.5 * (
        _LOG_2PI + torch.log(variance) + (true_val - pred_mean) ** 2 / variance
    )
    mean_nll = float(nll.mean().item())

    return {
        "ece": ece,
        "expected_coverage": expected_coverages,
        "observed_coverage": observed_coverages_arr,
        "nll": mean_nll,
    }
