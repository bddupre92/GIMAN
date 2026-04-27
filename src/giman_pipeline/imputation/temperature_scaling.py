"""Per-feature temperature scaling for GIMIN uncertainty calibration.

A lightweight post-hoc calibration wrapper that learns one scalar T_f per feature
such that T_f * sigma_f matches the empirical residual spread at a target coverage
level. Does NOT require retraining the base GIMIN model.

This addresses the empirical finding that GIMIN's raw heteroscedastic + MC-dropout
uncertainty is correctly *ranked* but scale-miscalibrated — intervals are too
narrow, producing ~71% observed coverage at 90% nominal on PPMI imputation
residuals. A per-feature scalar T_f applied post-hoc closes most of this gap
without touching the trained model.

See Guo et al., "On Calibration of Modern Neural Networks," ICML 2017 for the
single-scalar (Platt-style) version. Our per-feature extension accounts for
heterogeneous feature scales (clinical scores: 0-30 vs. brain volumes: 1000-8000).

Usage:
    >>> from giman_pipeline.imputation.temperature_scaling import PerFeatureTemperatureScaler
    >>> scaler = PerFeatureTemperatureScaler(target_coverage=0.90)
    >>> scaler.fit(mean_cal, std_cal, truth_cal, mask_cal)
    >>> T = scaler.temperatures_         # shape (n_features,)
    >>> calibrated_std = scaler.transform(std_test)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy.stats import norm as scipy_norm


@dataclass
class PerFeatureTemperatureScaler:
    """Learns one scalar temperature per feature to calibrate Gaussian intervals.

    For each feature f, fits T_f such that |residual_f| / (T_f * std_f) matches
    the nominal Gaussian quantile at the target coverage level on held-out
    calibration data. Inference intervals become [mean - z * T_f * std,
    mean + z * T_f * std] for nominal coverage gamma with z = Phi^-1((1+gamma)/2).

    Parameters
    ----------
    target_coverage : float, default 0.90
        The nominal coverage level at which T_f is fit. Coverage at other levels
        will also improve but is most accurate at this target.
    min_calibration_samples : int, default 5
        Features with fewer than this many calibration samples retain T=1.0
        (fall back to raw std).
    min_temperature : float, default 1e-3
        Lower clamp on T_f to avoid pathological scale-to-zero.
    max_temperature : float, default 1e6
        Upper clamp on T_f to prevent runaway widths when stds are miscalibrated
        by many orders of magnitude.
    """

    target_coverage: float = 0.90
    min_calibration_samples: int = 5
    min_temperature: float = 1e-3
    max_temperature: float = 1e6

    temperatures_: Optional[np.ndarray] = None
    n_features_: Optional[int] = None

    def fit(
        self,
        mean: np.ndarray,
        std: np.ndarray,
        truth: np.ndarray,
        mask: np.ndarray,
    ) -> "PerFeatureTemperatureScaler":
        """Learn per-feature temperatures T_f on calibration data.

        Parameters
        ----------
        mean : array (N, F)
            Predicted mean per patient per feature.
        std : array (N, F)
            Predicted standard deviation per patient per feature (raw GIMIN output).
        truth : array (N, F)
            Ground-truth feature values (only meaningful where mask == 1).
        mask : array (N, F)
            Binary mask: 1 = position is a calibration sample, 0 = ignore.
        """
        assert mean.shape == std.shape == truth.shape == mask.shape, (
            f"shape mismatch: mean {mean.shape}, std {std.shape}, "
            f"truth {truth.shape}, mask {mask.shape}"
        )
        n, f = mean.shape
        self.n_features_ = f
        T = np.ones(f)
        z_target = scipy_norm.ppf(0.5 + self.target_coverage / 2)

        std_safe = np.maximum(std, 1e-12)
        residuals = np.abs(truth - mean)

        for j in range(f):
            cal_positions = np.flatnonzero(mask[:, j] > 0)
            if len(cal_positions) < self.min_calibration_samples:
                continue
            # Empirical ratio |residual|/std on calibration set
            ratio = residuals[cal_positions, j] / std_safe[cal_positions, j]
            # Want T_f such that z_target * T_f matches the target-coverage quantile
            empirical_q = np.quantile(ratio, self.target_coverage)
            T_f = empirical_q / z_target
            T[j] = np.clip(T_f, self.min_temperature, self.max_temperature)

        self.temperatures_ = T
        return self

    def transform(self, std: np.ndarray) -> np.ndarray:
        """Return T_f * std (per-feature-scaled)."""
        if self.temperatures_ is None:
            raise RuntimeError("Must call .fit() before .transform()")
        return std * self.temperatures_[np.newaxis, :]

    def fit_transform(
        self,
        mean: np.ndarray,
        std: np.ndarray,
        truth: np.ndarray,
        mask: np.ndarray,
    ) -> np.ndarray:
        """Fit then return calibrated std."""
        return self.fit(mean, std, truth, mask).transform(std)

    def evaluate_coverage(
        self,
        mean: np.ndarray,
        std_calibrated: np.ndarray,
        truth: np.ndarray,
        mask: np.ndarray,
        coverage_targets: Optional[list[float]] = None,
    ) -> dict:
        """Compute empirical coverage at multiple nominal levels using scaled std.

        Returns dict mapping nominal target -> observed coverage.
        """
        if coverage_targets is None:
            coverage_targets = [0.50, 0.70, 0.80, 0.90, 0.95]

        std_safe = np.maximum(std_calibrated, 1e-12)
        residuals = np.abs(truth - mean)
        eval_bool = mask > 0

        out = {"n_eval": int(eval_bool.sum())}
        for gamma in coverage_targets:
            z = scipy_norm.ppf(0.5 + gamma / 2)
            inside = residuals[eval_bool] <= z * std_safe[eval_bool]
            width = float((2 * z * std_safe[eval_bool]).mean())
            out[f"gamma_{gamma:.2f}"] = {
                "target": float(gamma),
                "observed_coverage": float(inside.mean()),
                "mean_interval_width": width,
            }
        return out

    def to_dict(self) -> dict:
        return {
            "target_coverage": self.target_coverage,
            "min_calibration_samples": self.min_calibration_samples,
            "min_temperature": self.min_temperature,
            "max_temperature": self.max_temperature,
            "n_features": self.n_features_,
            "temperatures": (
                self.temperatures_.tolist()
                if self.temperatures_ is not None
                else None
            ),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "PerFeatureTemperatureScaler":
        obj = cls(
            target_coverage=d["target_coverage"],
            min_calibration_samples=d.get("min_calibration_samples", 5),
            min_temperature=d.get("min_temperature", 1e-3),
            max_temperature=d.get("max_temperature", 1e6),
        )
        if d.get("temperatures") is not None:
            obj.temperatures_ = np.array(d["temperatures"])
            obj.n_features_ = d["n_features"]
        return obj
