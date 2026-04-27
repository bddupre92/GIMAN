"""Imputation quality metrics for GIMIN evaluation.

All metric functions follow a consistent interface:

- ``pred``, ``true``: arrays of the same shape ``(N, F)`` or ``(N,)``.
- ``mask``: binary array of the same shape; **1 = the position is
  evaluated** (i.e., the value was held out or is of interest).

This convention means metrics are computed *only* at masked positions,
which is the standard protocol for imputation benchmarks.

Functions fall into three categories:

1. **Point metrics** -- RMSE, MAE, R-squared, NRMSE.
2. **Distributional metrics** -- per-feature Kolmogorov--Smirnov test.
3. **Calibration metrics** -- quantile coverage and expected calibration
   error (ECE) for uncertainty-aware models.
"""

from __future__ import annotations

import logging

import numpy as np
from scipy import stats

from ..utils import ArrayLike
from ..utils import to_numpy as _to_numpy

logger = logging.getLogger(__name__)


# ======================================================================
# Point metrics
# ======================================================================


def rmse(
    pred: ArrayLike,
    true: ArrayLike,
    mask: ArrayLike,
) -> float:
    """Root mean squared error on masked positions.

    Args:
        pred: Predicted values, shape ``(N, F)`` or ``(N,)``.
        true: Ground-truth values, same shape as *pred*.
        mask: Binary evaluation mask (1 = evaluate), same shape.

    Returns:
        Scalar RMSE value.

    Raises:
        ValueError: If no positions are selected by *mask*.
    """
    pred, true, mask = _to_numpy(pred), _to_numpy(true), _to_numpy(mask)
    mask_bool = mask.astype(bool)

    if mask_bool.sum() == 0:
        raise ValueError("RMSE undefined: mask selects zero positions.")

    errors = (pred[mask_bool] - true[mask_bool]) ** 2
    return float(np.sqrt(errors.mean()))


def mae(
    pred: ArrayLike,
    true: ArrayLike,
    mask: ArrayLike,
) -> float:
    """Mean absolute error on masked positions.

    Args:
        pred: Predicted values, shape ``(N, F)`` or ``(N,)``.
        true: Ground-truth values, same shape as *pred*.
        mask: Binary evaluation mask (1 = evaluate), same shape.

    Returns:
        Scalar MAE value.

    Raises:
        ValueError: If no positions are selected by *mask*.
    """
    pred, true, mask = _to_numpy(pred), _to_numpy(true), _to_numpy(mask)
    mask_bool = mask.astype(bool)

    if mask_bool.sum() == 0:
        raise ValueError("MAE undefined: mask selects zero positions.")

    return float(np.abs(pred[mask_bool] - true[mask_bool]).mean())


def r_squared(
    pred: ArrayLike,
    true: ArrayLike,
    mask: ArrayLike,
) -> float:
    """Coefficient of determination (R-squared) on masked positions.

    .. math::

        R^2 = 1 - \\frac{\\sum (y - \\hat{y})^2}{\\sum (y - \\bar{y})^2}

    Args:
        pred: Predicted values.
        true: Ground-truth values.
        mask: Binary evaluation mask (1 = evaluate).

    Returns:
        Scalar R-squared value.  Can be negative if predictions are
        worse than predicting the mean.

    Raises:
        ValueError: If no positions are selected or variance is zero.
    """
    pred, true, mask = _to_numpy(pred), _to_numpy(true), _to_numpy(mask)
    mask_bool = mask.astype(bool)

    if mask_bool.sum() == 0:
        raise ValueError("R-squared undefined: mask selects zero positions.")

    y = true[mask_bool]
    y_hat = pred[mask_bool]

    ss_res = ((y - y_hat) ** 2).sum()
    ss_tot = ((y - y.mean()) ** 2).sum()

    if ss_tot < 1e-12:
        logger.warning("R-squared: total variance is near zero.")
        return 0.0

    return float(1.0 - ss_res / ss_tot)


def nrmse(
    pred: ArrayLike,
    true: ArrayLike,
    mask: ArrayLike,
) -> float:
    """Normalized RMSE (divided by per-feature range) on masked positions.

    For 2-D inputs the RMSE is computed per feature, normalized by the
    range of observed (masked) true values for that feature, then
    averaged.  For 1-D inputs, a single normalised value is returned.

    Args:
        pred: Predicted values, shape ``(N, F)`` or ``(N,)``.
        true: Ground-truth values, same shape as *pred*.
        mask: Binary evaluation mask (1 = evaluate), same shape.

    Returns:
        Scalar NRMSE value in ``[0, inf)``.

    Raises:
        ValueError: If no positions are selected by *mask*.
    """
    pred, true, mask = _to_numpy(pred), _to_numpy(true), _to_numpy(mask)

    if pred.ndim == 1:
        pred = pred[:, np.newaxis]
        true = true[:, np.newaxis]
        mask = mask[:, np.newaxis]

    mask_bool = mask.astype(bool)
    if mask_bool.sum() == 0:
        raise ValueError("NRMSE undefined: mask selects zero positions.")

    num_features = pred.shape[1]
    nrmse_values: list[float] = []

    for f in range(num_features):
        f_mask = mask_bool[:, f]
        if f_mask.sum() == 0:
            continue

        f_true = true[f_mask, f]
        f_pred = pred[f_mask, f]

        f_range = f_true.max() - f_true.min()
        if f_range < 1e-12:
            # Constant feature -- RMSE / range is undefined; skip.
            logger.debug("NRMSE: feature %d has near-zero range; skipping.", f)
            continue

        f_rmse = float(np.sqrt(((f_pred - f_true) ** 2).mean()))
        nrmse_values.append(f_rmse / f_range)

    if len(nrmse_values) == 0:
        raise ValueError("NRMSE: no features with non-zero range.")

    return float(np.mean(nrmse_values))


# ======================================================================
# Distributional metrics
# ======================================================================


def ks_test_per_feature(
    imputed: ArrayLike,
    observed: ArrayLike,
    mask: ArrayLike,
    feature_names: list[str] | None = None,
) -> dict[str, dict[str, float]]:
    """Per-feature two-sample Kolmogorov--Smirnov test.

    For each feature, compares the distribution of imputed values
    (at missing positions) with the observed values to assess
    distributional fidelity.

    Args:
        imputed: Imputed feature matrix, shape ``(N, F)``.
        observed: Original feature matrix, shape ``(N, F)`` (only
            observed positions are used).
        mask: Binary observation mask (1 = observed), shape ``(N, F)``.
        feature_names: Optional list of length ``F`` giving feature
            names.  If ``None``, integer indices are used.

    Returns:
        Dictionary mapping feature name/index to a dict with keys
        ``"ks_statistic"`` and ``"p_value"``.
    """
    imputed = _to_numpy(imputed)
    observed = _to_numpy(observed)
    mask = _to_numpy(mask)

    if imputed.ndim == 1:
        imputed = imputed[:, np.newaxis]
        observed = observed[:, np.newaxis]
        mask = mask[:, np.newaxis]

    num_features = imputed.shape[1]
    if feature_names is None:
        feature_names = [str(i) for i in range(num_features)]

    results: dict[str, dict[str, float]] = {}

    for f in range(num_features):
        obs_mask = mask[:, f].astype(bool)
        miss_mask = ~obs_mask

        obs_vals = observed[obs_mask, f]
        imp_vals = imputed[miss_mask, f]

        if len(obs_vals) < 2 or len(imp_vals) < 2:
            logger.debug(
                "KS test: feature '%s' has insufficient samples "
                "(obs=%d, imp=%d); skipping.",
                feature_names[f],
                len(obs_vals),
                len(imp_vals),
            )
            continue

        ks_stat, p_value = stats.ks_2samp(obs_vals, imp_vals)
        results[feature_names[f]] = {
            "ks_statistic": float(ks_stat),
            "p_value": float(p_value),
        }

    return results


# ======================================================================
# Calibration metrics
# ======================================================================


def calibration_metrics(
    pred_means: ArrayLike,
    pred_stds: ArrayLike,
    true_values: ArrayLike,
    mask: ArrayLike,
    quantiles: list[float] | None = None,
) -> dict[str, float]:
    """Quantile coverage calibration metrics.

    For each nominal quantile level *q*, compute the fraction of true
    values that fall within the predicted ``q``-confidence interval.
    A perfectly calibrated model yields coverage equal to *q*.

    The confidence interval for quantile level *q* is:

    .. math::

        [\\mu - z_q \\cdot \\sigma, \\; \\mu + z_q \\cdot \\sigma]

    where :math:`z_q` is the standard-normal quantile corresponding to
    a two-tailed coverage of *q*.

    Args:
        pred_means: Predicted means, shape ``(N, F)`` or ``(N,)``.
        pred_stds: Predicted standard deviations, same shape.
        true_values: Ground-truth values, same shape.
        mask: Binary evaluation mask (1 = evaluate), same shape.
        quantiles: Coverage levels to check.  Default:
            ``[0.5, 0.8, 0.9, 0.95]``.

    Returns:
        Dictionary mapping ``"coverage_{q}"`` to the empirical coverage
        fraction, and ``"avg_calibration_error"`` to the mean absolute
        difference between nominal and empirical coverage.
    """
    if quantiles is None:
        quantiles = [0.5, 0.8, 0.9, 0.95]

    pred_means = _to_numpy(pred_means)
    pred_stds = _to_numpy(pred_stds)
    true_values = _to_numpy(true_values)
    mask = _to_numpy(mask)
    mask_bool = mask.astype(bool)

    if mask_bool.sum() == 0:
        raise ValueError("Calibration metrics undefined: mask selects zero positions.")

    mu = pred_means[mask_bool]
    sigma = pred_stds[mask_bool]
    y = true_values[mask_bool]

    results: dict[str, float] = {}
    total_cal_error = 0.0

    for q in quantiles:
        # z-score for two-tailed interval with coverage q.
        z = stats.norm.ppf(0.5 + q / 2.0)
        lower = mu - z * sigma
        upper = mu + z * sigma

        covered = ((y >= lower) & (y <= upper)).mean()
        results[f"coverage_{q:.2f}"] = float(covered)
        total_cal_error += abs(covered - q)

    results["avg_calibration_error"] = total_cal_error / len(quantiles)
    return results


def expected_calibration_error(
    pred_means: ArrayLike,
    pred_stds: ArrayLike,
    true_values: ArrayLike,
    mask: ArrayLike,
    n_bins: int = 10,
) -> float:
    """Expected calibration error (ECE) for probabilistic predictions.

    Bins predictions by predicted confidence (inverse of predicted
    standard deviation), then measures the gap between expected and
    observed accuracy within each bin.  The ECE is the weighted average
    of per-bin calibration gaps.

    We define "accuracy" as the fraction of true values falling within
    one predicted standard deviation of the mean, and "confidence" as
    the inverse of the predicted standard deviation (higher confidence
    = smaller predicted uncertainty).

    Args:
        pred_means: Predicted means, shape ``(N, F)`` or ``(N,)``.
        pred_stds: Predicted standard deviations, same shape.
        true_values: Ground-truth values, same shape.
        mask: Binary evaluation mask (1 = evaluate), same shape.
        n_bins: Number of confidence bins. Default: 10.

    Returns:
        Scalar ECE value in ``[0, 1]``.

    Raises:
        ValueError: If no positions are selected by *mask*.
    """
    pred_means = _to_numpy(pred_means)
    pred_stds = _to_numpy(pred_stds)
    true_values = _to_numpy(true_values)
    mask = _to_numpy(mask)
    mask_bool = mask.astype(bool)

    if mask_bool.sum() == 0:
        raise ValueError("ECE undefined: mask selects zero positions.")

    mu = pred_means[mask_bool]
    sigma = np.maximum(pred_stds[mask_bool], 1e-8)
    y = true_values[mask_bool]

    # Standardised residuals.
    z_scores = np.abs((y - mu) / sigma)

    # Expected coverage at 1 sigma for a standard normal: ~68.27%.
    within_1sigma = (z_scores <= 1.0).astype(float)

    # Confidence = 1 / sigma (higher = more confident).
    confidence = 1.0 / sigma

    # Bin by confidence.
    bin_edges = np.linspace(confidence.min(), confidence.max() + 1e-8, n_bins + 1)
    ece = 0.0
    total_samples = len(confidence)

    for b in range(n_bins):
        in_bin = (confidence >= bin_edges[b]) & (confidence < bin_edges[b + 1])
        n_in_bin = in_bin.sum()
        if n_in_bin == 0:
            continue

        # Empirical accuracy in this bin.
        bin_accuracy = within_1sigma[in_bin].mean()
        # Expected coverage: fraction of a N(0,1) within 1 sigma = 0.6827.
        expected_coverage = 0.6827
        ece += (n_in_bin / total_samples) * abs(bin_accuracy - expected_coverage)

    return float(ece)
