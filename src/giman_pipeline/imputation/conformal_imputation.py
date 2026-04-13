"""Stage-conditioned conformal calibration for imputation uncertainty.

Provides distribution-free prediction intervals for imputed values using
conformal prediction, calibrated per NSD-ISS biological stage. This
addresses the key Paper 2 insight: uncertainty should be stage-aware
because biomarker variability differs across disease stages.

Three calibration modes:

1. **Per-feature absolute residual**: Each feature gets its own conformal
   quantile based on absolute residuals. No model std needed. Produces
   feature-appropriate interval widths but intervals are constant for a
   given feature (not locally adaptive).

2. **Normalized-space heteroscedastic**: Uses model's predicted std in
   normalized space to compute residual scores = |y_norm - ŷ_norm| / σ̂_norm.
   Single quantile across features; intervals locally adaptive. Calibrate
   in normalized space to avoid Jacobian blow-up on inverse transform.
   Coverage is invariant to monotone transformations.

3. **Per-stage**: Either mode 1 or 2, stratified by NSD-ISS stage for
   conditional coverage guarantees.

Key metrics reported:
    - Coverage: Fraction of true values inside prediction interval (target: 90%)
    - MPIW: Mean Prediction Interval Width (absolute units)
    - NMPIW: Normalized MPIW (divided by feature range, comparable across features)
    - Sharpness: Median interval width (more robust than mean)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class ConformalResult:
    """Container for conformal calibration results."""

    coverage_target: float
    observed_coverage: float
    mean_interval_width: float
    median_interval_width: float
    quantile_threshold: float | np.ndarray  # scalar or per-feature
    n_calibration: int
    per_feature_coverage: np.ndarray | None = None
    per_feature_width: np.ndarray | None = None
    nmpiw: float | None = None  # Normalized mean prediction interval width


@dataclass
class StageConformalResult:
    """Container for per-stage conformal results."""

    marginal: ConformalResult
    per_stage: dict[int, ConformalResult] = field(default_factory=dict)
    stage_names: dict[int, str] = field(default_factory=dict)


def _conformal_quantile(n: int, alpha: float) -> float:
    """Compute finite-sample conformal quantile level.

    Uses the standard formula: q = ceil((n+1) * alpha) / n
    which guarantees marginal coverage >= alpha.
    """
    return min(np.ceil((n + 1) * alpha) / n, 1.0)


class ConformalImputation:
    """Stage-conditioned conformal prediction for imputation intervals.

    Uses split conformal prediction with nonconformity scores to construct
    prediction intervals for imputed values.

    The key innovation for Paper 2: per-stage calibration ensures that
    coverage guarantees hold within each NSD-ISS stage, not just
    marginally across the population.

    Args:
        coverage_target: Desired coverage probability. Default: 0.90.
        mode: Calibration mode. Options:
            - "per_feature": Per-feature absolute residual (recommended)
            - "normalized": Normalized residuals using model std
            - "global": Single global threshold (legacy, not recommended)
            Default: "per_feature".
    """

    def __init__(
        self,
        coverage_target: float = 0.90,
        mode: str = "per_feature",
    ) -> None:
        self.coverage_target = coverage_target
        self.mode = mode
        self._marginal_quantile: float | np.ndarray | None = None
        self._stage_quantiles: dict[int, float | np.ndarray] = {}
        self._feature_names: list[str] | None = None

    def calibrate_marginal(
        self,
        predicted_means: np.ndarray,
        true_values: np.ndarray,
        mask: np.ndarray,
        predicted_stds: np.ndarray | None = None,
    ) -> ConformalResult:
        """Compute marginal conformal quantile threshold.

        For mode="per_feature": each feature gets its own quantile based
        on absolute residuals at imputed positions. This is the recommended
        approach for heterogeneous features (e.g., brain volumes vs clinical
        scores).

        For mode="normalized": uses |y - ŷ| / σ̂ as nonconformity score
        with a single global quantile. Requires predicted_stds.

        Args:
            predicted_means: Model's imputed means (N, F).
            true_values: Ground-truth values (N, F).
            mask: Binary mask where 0 = imputed position to evaluate (N, F).
            predicted_stds: Optional predicted std devs (N, F). Required for
                mode="normalized".

        Returns:
            ConformalResult with calibration statistics.
        """
        imputed_mask = mask == 0
        n_imputed = imputed_mask.sum()
        N, F = predicted_means.shape

        if n_imputed == 0:
            logger.warning("No imputed positions to calibrate.")
            return ConformalResult(
                coverage_target=self.coverage_target,
                observed_coverage=1.0,
                mean_interval_width=0.0,
                median_interval_width=0.0,
                quantile_threshold=0.0,
                n_calibration=0,
            )

        residuals = np.abs(true_values - predicted_means)

        if self.mode == "per_feature":
            return self._calibrate_per_feature(
                predicted_means, true_values, residuals, mask, imputed_mask
            )
        elif self.mode == "normalized":
            return self._calibrate_normalized(
                predicted_means,
                true_values,
                residuals,
                mask,
                imputed_mask,
                predicted_stds,
            )
        else:
            # Global mode (legacy)
            return self._calibrate_global(
                predicted_means,
                true_values,
                residuals,
                mask,
                imputed_mask,
                predicted_stds,
            )

    def _calibrate_per_feature(
        self,
        predicted_means: np.ndarray,
        true_values: np.ndarray,
        residuals: np.ndarray,
        mask: np.ndarray,
        imputed_mask: np.ndarray,
    ) -> ConformalResult:
        """Per-feature conformal: each feature j gets quantile q_j.

        Interval for feature j: [ŷ_j - q_j, ŷ_j + q_j].
        """
        N, F = predicted_means.shape
        per_feature_q = np.zeros(F)
        per_feature_cov = np.zeros(F)
        per_feature_width = np.zeros(F)
        total_covered = 0
        total_imputed = 0

        for j in range(F):
            # Imputed positions for feature j
            feat_imputed = imputed_mask[:, j]
            n_j = feat_imputed.sum()

            if n_j < 2:
                per_feature_q[j] = 0.0
                per_feature_cov[j] = 1.0
                per_feature_width[j] = 0.0
                continue

            scores_j = residuals[feat_imputed, j]
            q_level = _conformal_quantile(int(n_j), self.coverage_target)
            q_j = float(np.quantile(scores_j, q_level))
            per_feature_q[j] = q_j

            # Coverage for this feature
            lower_j = predicted_means[feat_imputed, j] - q_j
            upper_j = predicted_means[feat_imputed, j] + q_j
            covered_j = (true_values[feat_imputed, j] >= lower_j) & (
                true_values[feat_imputed, j] <= upper_j
            )
            per_feature_cov[j] = float(covered_j.mean())
            per_feature_width[j] = 2 * q_j
            total_covered += covered_j.sum()
            total_imputed += n_j

        self._marginal_quantile = per_feature_q

        # Overall coverage (weighted by number of imputed positions per feature)
        overall_coverage = total_covered / max(total_imputed, 1)

        # NMPIW: normalize widths by feature range
        feature_ranges = np.ptp(true_values, axis=0)  # max - min per feature
        feature_ranges = np.maximum(feature_ranges, 1e-8)
        nmpiw = float(np.mean(per_feature_width / feature_ranges))

        return ConformalResult(
            coverage_target=self.coverage_target,
            observed_coverage=float(overall_coverage),
            mean_interval_width=float(np.mean(per_feature_width)),
            median_interval_width=float(np.median(per_feature_width)),
            quantile_threshold=per_feature_q,
            n_calibration=int(total_imputed),
            per_feature_coverage=per_feature_cov,
            per_feature_width=per_feature_width,
            nmpiw=nmpiw,
        )

    def _calibrate_normalized(
        self,
        predicted_means: np.ndarray,
        true_values: np.ndarray,
        residuals: np.ndarray,
        mask: np.ndarray,
        imputed_mask: np.ndarray,
        predicted_stds: np.ndarray | None = None,
    ) -> ConformalResult:
        """Normalized residual conformal: score = |y - ŷ| / σ̂.

        Uses a single quantile q. Interval: [ŷ - q⋅σ̂, ŷ + q⋅σ̂].
        Locally adaptive — wider where model is uncertain.
        """
        if predicted_stds is None:
            logger.warning(
                "mode='normalized' requires predicted_stds, falling back to per_feature"
            )
            return self._calibrate_per_feature(
                predicted_means, true_values, residuals, mask, imputed_mask
            )

        safe_stds = np.maximum(predicted_stds, 1e-8)
        scores = residuals[imputed_mask] / safe_stds[imputed_mask]

        n = len(scores)
        q_level = _conformal_quantile(n, self.coverage_target)
        q = float(np.quantile(scores, q_level))
        self._marginal_quantile = q

        # Compute intervals
        intervals = safe_stds * q
        lower = predicted_means - intervals
        upper = predicted_means + intervals
        covered = (true_values >= lower) & (true_values <= upper)
        overall_coverage = float(covered[imputed_mask].mean())

        widths = (2 * intervals)[imputed_mask]

        # Per-feature coverage
        N, F = predicted_means.shape
        per_feature_cov = np.zeros(F)
        per_feature_width = np.zeros(F)
        for j in range(F):
            feat_imputed = imputed_mask[:, j]
            if feat_imputed.sum() > 0:
                per_feature_cov[j] = float(covered[feat_imputed, j].mean())
                per_feature_width[j] = float((2 * intervals)[feat_imputed, j].mean())

        # NMPIW
        feature_ranges = np.ptp(true_values, axis=0)
        feature_ranges = np.maximum(feature_ranges, 1e-8)
        nmpiw = float(np.mean(per_feature_width / feature_ranges))

        return ConformalResult(
            coverage_target=self.coverage_target,
            observed_coverage=overall_coverage,
            mean_interval_width=float(widths.mean()),
            median_interval_width=float(np.median(widths)),
            quantile_threshold=q,
            n_calibration=n,
            per_feature_coverage=per_feature_cov,
            per_feature_width=per_feature_width,
            nmpiw=nmpiw,
        )

    def _calibrate_global(
        self,
        predicted_means: np.ndarray,
        true_values: np.ndarray,
        residuals: np.ndarray,
        mask: np.ndarray,
        imputed_mask: np.ndarray,
        predicted_stds: np.ndarray | None = None,
    ) -> ConformalResult:
        """Global single-threshold conformal (legacy, not recommended)."""
        if predicted_stds is not None:
            safe_stds = np.maximum(predicted_stds, 1e-8)
            scores = residuals[imputed_mask] / safe_stds[imputed_mask]
        else:
            scores = residuals[imputed_mask]

        n = len(scores)
        q_level = _conformal_quantile(n, self.coverage_target)
        q = float(np.quantile(scores, q_level))
        self._marginal_quantile = q

        if predicted_stds is not None:
            intervals = safe_stds * q
        else:
            intervals = np.full_like(predicted_means, q)

        lower = predicted_means - intervals
        upper = predicted_means + intervals
        covered = (true_values >= lower) & (true_values <= upper)
        overall_coverage = float(covered[imputed_mask].mean())
        widths = (2 * intervals)[imputed_mask]

        return ConformalResult(
            coverage_target=self.coverage_target,
            observed_coverage=overall_coverage,
            mean_interval_width=float(widths.mean()),
            median_interval_width=float(np.median(widths)),
            quantile_threshold=q,
            n_calibration=n,
        )

    def calibrate_per_stage(
        self,
        predicted_means: np.ndarray,
        true_values: np.ndarray,
        mask: np.ndarray,
        stages: np.ndarray,
        predicted_stds: np.ndarray | None = None,
        stage_names: dict[int, str] | None = None,
    ) -> StageConformalResult:
        """Compute per-stage conformal quantile thresholds.

        Calibrates separate conformal prediction intervals for each
        NSD-ISS stage, ensuring conditional coverage guarantees.

        Args:
            predicted_means: Model's imputed means (N, F).
            true_values: Ground-truth values (N, F).
            mask: Binary mask where 0 = imputed position (N, F).
            stages: NSD-ISS stage indices (N,).
            predicted_stds: Optional predicted std devs (N, F).
            stage_names: Optional mapping from stage ID to name.

        Returns:
            StageConformalResult with marginal and per-stage calibration.
        """
        if stage_names is None:
            stage_names = {
                0: "Stage 0",
                1: "Stage 1",
                2: "Stage 2B",
                3: "Stage 3",
                4: "Stage 4",
                5: "Unknown",
            }

        # First compute marginal
        marginal = self.calibrate_marginal(
            predicted_means, true_values, mask, predicted_stds
        )

        # Then per-stage
        unique_stages = np.unique(stages)
        per_stage = {}

        for stage_id in unique_stages:
            stage_mask_patients = stages == stage_id
            n_patients = stage_mask_patients.sum()

            if n_patients < 5:
                logger.warning(
                    "Stage %s has only %d patients, skipping per-stage calibration.",
                    stage_names.get(int(stage_id), str(stage_id)),
                    n_patients,
                )
                continue

            stage_pred = predicted_means[stage_mask_patients]
            stage_true = true_values[stage_mask_patients]
            stage_obs_mask = mask[stage_mask_patients]
            stage_stds = (
                predicted_stds[stage_mask_patients]
                if predicted_stds is not None
                else None
            )

            result = self.calibrate_marginal(
                stage_pred, stage_true, stage_obs_mask, stage_stds
            )
            per_stage[int(stage_id)] = result
            self._stage_quantiles[int(stage_id)] = result.quantile_threshold

            logger.info(
                "Stage %s: coverage=%.3f, NMPIW=%.4f, mean_width=%.2f (n=%d)",
                stage_names.get(int(stage_id), str(stage_id)),
                result.observed_coverage,
                result.nmpiw if result.nmpiw is not None else 0.0,
                result.mean_interval_width,
                result.n_calibration,
            )

        return StageConformalResult(
            marginal=marginal,
            per_stage=per_stage,
            stage_names=stage_names,
        )

    def predict_intervals(
        self,
        predicted_means: np.ndarray,
        predicted_stds: np.ndarray | None = None,
        stages: np.ndarray | None = None,
        use_per_stage: bool = True,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Generate prediction intervals for new data.

        Args:
            predicted_means: Model's imputed means (N, F).
            predicted_stds: Optional predicted std devs (N, F).
            stages: Optional stage indices for per-stage intervals (N,).
            use_per_stage: If True and stages provided, use per-stage
                quantiles. Otherwise use marginal. Default: True.

        Returns:
            Tuple of (lower_bounds, upper_bounds), each (N, F).
        """
        N, F = predicted_means.shape

        if self.mode == "per_feature":
            return self._predict_per_feature(predicted_means, stages, use_per_stage)
        elif self.mode == "normalized":
            return self._predict_normalized(
                predicted_means, predicted_stds, stages, use_per_stage
            )
        else:
            return self._predict_global(
                predicted_means, predicted_stds, stages, use_per_stage
            )

    def _predict_per_feature(
        self,
        predicted_means: np.ndarray,
        stages: np.ndarray | None = None,
        use_per_stage: bool = True,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Generate per-feature intervals."""
        N, F = predicted_means.shape

        if use_per_stage and stages is not None and self._stage_quantiles:
            # Per-stage, per-feature
            half_widths = np.zeros_like(predicted_means)
            for i in range(N):
                stage = int(stages[i])
                if stage in self._stage_quantiles:
                    q = self._stage_quantiles[stage]
                elif self._marginal_quantile is not None:
                    q = self._marginal_quantile
                else:
                    q = np.ones(F)
                # q is per-feature array
                if isinstance(q, np.ndarray):
                    half_widths[i, :] = q[:F]
                else:
                    half_widths[i, :] = q
        else:
            if self._marginal_quantile is not None:
                q = self._marginal_quantile
                if isinstance(q, np.ndarray):
                    half_widths = np.tile(q[:F], (N, 1))
                else:
                    half_widths = np.full((N, F), q)
            else:
                half_widths = np.ones((N, F))

        return predicted_means - half_widths, predicted_means + half_widths

    def _predict_normalized(
        self,
        predicted_means: np.ndarray,
        predicted_stds: np.ndarray | None = None,
        stages: np.ndarray | None = None,
        use_per_stage: bool = True,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Generate normalized (locally adaptive) intervals."""
        N, F = predicted_means.shape
        safe_stds = (
            np.maximum(predicted_stds, 1e-8)
            if predicted_stds is not None
            else np.ones((N, F))
        )

        if use_per_stage and stages is not None and self._stage_quantiles:
            quantiles = np.zeros(N)
            for i in range(N):
                stage = int(stages[i])
                if stage in self._stage_quantiles:
                    quantiles[i] = self._stage_quantiles[stage]
                elif self._marginal_quantile is not None:
                    quantiles[i] = self._marginal_quantile
                else:
                    quantiles[i] = 2.0
            half_widths = safe_stds * quantiles[:, None]
        else:
            q = self._marginal_quantile if self._marginal_quantile is not None else 2.0
            half_widths = safe_stds * q

        return predicted_means - half_widths, predicted_means + half_widths

    def _predict_global(
        self,
        predicted_means: np.ndarray,
        predicted_stds: np.ndarray | None = None,
        stages: np.ndarray | None = None,
        use_per_stage: bool = True,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Generate global intervals (legacy)."""
        return self._predict_normalized(
            predicted_means, predicted_stds, stages, use_per_stage
        )

    def evaluate_coverage(
        self,
        predicted_means: np.ndarray,
        true_values: np.ndarray,
        mask: np.ndarray,
        stages: np.ndarray,
        predicted_stds: np.ndarray | None = None,
        coverages: list[float] | None = None,
    ) -> dict[str, dict[float, float]]:
        """Evaluate coverage at multiple confidence levels.

        Produces calibration curve data: for each target coverage (50%, 70%,
        80%, 90%, 95%), reports observed coverage marginally and per-stage.
        A well-calibrated model has observed ≈ target at each level.

        Args:
            predicted_means: Model's imputed means (N, F).
            true_values: Ground-truth values (N, F).
            mask: Binary mask (N, F).
            stages: Stage indices (N,).
            predicted_stds: Optional predicted std devs (N, F).
            coverages: List of target coverage levels.

        Returns:
            Nested dict: {stage_name: {coverage_target: observed_coverage}}.
        """
        if coverages is None:
            coverages = [0.50, 0.70, 0.80, 0.90, 0.95]

        results = {}
        unique_stages = np.unique(stages)

        for coverage in coverages:
            cal = ConformalImputation(coverage_target=coverage, mode=self.mode)

            # Marginal
            marginal = cal.calibrate_marginal(
                predicted_means, true_values, mask, predicted_stds
            )
            results.setdefault("marginal", {})[coverage] = marginal.observed_coverage

            # Per-stage
            for stage_id in unique_stages:
                stage_mask = stages == stage_id
                if stage_mask.sum() < 5:
                    continue

                stage_result = cal.calibrate_marginal(
                    predicted_means[stage_mask],
                    true_values[stage_mask],
                    mask[stage_mask],
                    predicted_stds[stage_mask] if predicted_stds is not None else None,
                )
                stage_name = f"stage_{int(stage_id)}"
                results.setdefault(stage_name, {})[coverage] = (
                    stage_result.observed_coverage
                )

        return results
