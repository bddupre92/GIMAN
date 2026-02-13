"""Masked-value reconstruction benchmark for GIMIN.

Implements the standard evaluation protocol for imputation models:

1. Take a fully-observed (or partially-observed) dataset.
2. Artificially mask additional fractions of observed values.
3. Impute the masked values using the model under evaluation.
4. Compare the imputed values against the held-out ground truth.

The experiment is repeated over multiple random seeds and mask fractions,
producing a comprehensive results dictionary with per-modality breakdowns.
"""

from __future__ import annotations

import logging
import time
from collections import defaultdict
from typing import Any

import numpy as np
import torch

from ..config import GIMINConfig
from ..utils import ArrayLike
from ..utils import to_numpy as _to_numpy
from . import metrics as M  # noqa: N812

logger = logging.getLogger(__name__)


class MaskedValueExperiment:
    """Run masked-value reconstruction benchmarks.

    For each combination of mask fraction and random seed, the experiment:

    1. Randomly hides a fraction of observed values.
    2. Passes the corrupted data to the imputation model (and optionally
       to a set of baseline methods).
    3. Evaluates RMSE, MAE, R-squared, NRMSE, and (if uncertainty is
       available) calibration metrics at the held-out positions.
    4. Optionally computes per-modality breakdowns.

    Args:
        config: GIMIN configuration (used for modality definitions and
            evaluation parameters).  If ``None``, modality breakdowns
            are skipped and default parameters are used.
    """

    def __init__(self, config: GIMINConfig | None = None) -> None:
        self.config = config

    # ------------------------------------------------------------------
    # Masking utility
    # ------------------------------------------------------------------

    @staticmethod
    def _create_evaluation_mask(
        mask: np.ndarray,
        fraction: float,
        rng: np.random.Generator,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Artificially mask a fraction of observed values.

        Args:
            mask: Original binary observation mask ``(N, F)``.
            fraction: Fraction of observed values to hide.
            rng: NumPy random generator for reproducibility.

        Returns:
            Tuple of ``(corrupted_mask, target_mask)`` where
            ``corrupted_mask`` is the mask after hiding values and
            ``target_mask`` marks the newly hidden positions (loss/metrics
            are computed here).
        """
        observed_indices = np.argwhere(mask.astype(bool))
        num_to_mask = max(1, int(len(observed_indices) * fraction))
        chosen = rng.choice(len(observed_indices), size=num_to_mask, replace=False)
        chosen_positions = observed_indices[chosen]

        target_mask = np.zeros_like(mask)
        target_mask[chosen_positions[:, 0], chosen_positions[:, 1]] = 1.0

        corrupted_mask = mask.copy()
        corrupted_mask[target_mask.astype(bool)] = 0.0

        return corrupted_mask, target_mask

    # ------------------------------------------------------------------
    # GIMIN model imputation
    # ------------------------------------------------------------------

    def _impute_with_model(
        self,
        model: torch.nn.Module,
        features: np.ndarray,
        corrupted_mask: np.ndarray,
        edge_index: torch.Tensor | None,
        edge_weight: torch.Tensor | None,
        scaler: object | None = None,
    ) -> dict[str, np.ndarray]:
        """Run the GIMIN model on corrupted data.

        If a scaler is provided, input features are normalized before
        the model forward pass and the output is inverse-transformed
        back to the original clinical scale.

        Returns:
            Dictionary with ``"imputed"``, and optionally
            ``"pred_mean"``, ``"pred_std"`` arrays.
        """
        device = next(model.parameters()).device
        feat_t = torch.from_numpy(features.astype(np.float32)).to(device)
        mask_t = torch.from_numpy(corrupted_mask.astype(np.float32)).to(device)

        # Normalize input if scaler is available.
        if scaler is not None:
            feat_t = scaler.transform(feat_t, mask_t)

        # Zero out hidden positions.
        feat_t = feat_t * mask_t

        ei = (
            edge_index.to(device)
            if edge_index is not None
            else torch.zeros((2, 0), dtype=torch.long, device=device)
        )
        ew = (
            edge_weight.to(device)
            if edge_weight is not None
            else torch.ones(ei.shape[1], device=device)
        )
        of = torch.ones(ei.shape[1], device=device)

        # Get modality_dims from config or model attribute
        if self.config is not None:
            modality_dims = self.config.modality_dims
        elif hasattr(model, "modality_dims"):
            modality_dims = model.modality_dims
        else:
            modality_dims = [feat_t.shape[1]]

        model_input: dict[str, Any] = {
            "features": feat_t,
            "mask": mask_t,
            "edge_index": ei,
            "edge_weight": ew,
            "overlap_frac": of,
            "modality_dims": modality_dims,
        }

        model.eval()
        with torch.no_grad():
            output = model(**model_input)

        imputed_t = output["imputed"]

        # Inverse-transform back to clinical scale if scaler is available.
        if scaler is not None:
            imputed_t = scaler.inverse_transform(imputed_t)

        result: dict[str, np.ndarray] = {
            "imputed": imputed_t.cpu().numpy()
            if isinstance(imputed_t, torch.Tensor)
            else imputed_t,
        }
        if "pred_mean" in output:
            pred_mean_t = output["pred_mean"]
            if scaler is not None:
                pred_mean_t = scaler.inverse_transform(pred_mean_t)
            result["pred_mean"] = (
                pred_mean_t.cpu().numpy()
                if isinstance(pred_mean_t, torch.Tensor)
                else pred_mean_t
            )
        if "pred_log_var" in output:
            pred_std = np.exp(0.5 * output["pred_log_var"].cpu().numpy())
            result["pred_std"] = pred_std

        return result

    # ------------------------------------------------------------------
    # Baseline imputation
    # ------------------------------------------------------------------

    @staticmethod
    def _impute_with_baseline(
        baseline: Any,
        features: np.ndarray,
        corrupted_mask: np.ndarray,
    ) -> np.ndarray:
        """Run a baseline imputer on corrupted data."""
        corrupted_features = features.copy()
        corrupted_features[~corrupted_mask.astype(bool)] = 0.0
        return baseline.fit_transform(corrupted_features, corrupted_mask)

    # ------------------------------------------------------------------
    # Metric computation
    # ------------------------------------------------------------------

    def _compute_metrics(
        self,
        imputed: np.ndarray,
        true_values: np.ndarray,
        target_mask: np.ndarray,
        pred_mean: np.ndarray | None = None,
        pred_std: np.ndarray | None = None,
    ) -> dict[str, float]:
        """Compute all relevant metrics at held-out positions."""
        result: dict[str, float] = {
            "rmse": M.rmse(imputed, true_values, target_mask),
            "mae": M.mae(imputed, true_values, target_mask),
            "r_squared": M.r_squared(imputed, true_values, target_mask),
        }

        try:
            result["nrmse"] = M.nrmse(imputed, true_values, target_mask)
        except ValueError:
            result["nrmse"] = float("nan")

        # Calibration metrics (only if uncertainty is available).
        if pred_mean is not None and pred_std is not None:
            try:
                cal = M.calibration_metrics(
                    pred_mean, pred_std, true_values, target_mask
                )
                result.update(cal)
                result["ece"] = M.expected_calibration_error(
                    pred_mean, pred_std, true_values, target_mask
                )
            except ValueError:
                logger.debug("Calibration metrics could not be computed.")

        return result

    def _compute_modality_metrics(
        self,
        imputed: np.ndarray,
        true_values: np.ndarray,
        target_mask: np.ndarray,
    ) -> dict[str, dict[str, float]]:
        """Compute per-modality metric breakdowns."""
        if self.config is None:
            return {}

        modality_results: dict[str, dict[str, float]] = {}
        col_start = 0

        for mod in self.config.modalities:
            num_features = len(mod.features)
            col_end = col_start + num_features

            mod_mask = target_mask[:, col_start:col_end]
            if mod_mask.sum() == 0:
                col_start = col_end
                continue

            mod_imputed = imputed[:, col_start:col_end]
            mod_true = true_values[:, col_start:col_end]

            try:
                modality_results[mod.name] = {
                    "rmse": M.rmse(mod_imputed, mod_true, mod_mask),
                    "mae": M.mae(mod_imputed, mod_true, mod_mask),
                    "r_squared": M.r_squared(mod_imputed, mod_true, mod_mask),
                }
            except ValueError:
                logger.debug(
                    "Modality '%s' has no valid evaluation positions.",
                    mod.name,
                )

            col_start = col_end

        return modality_results

    # ------------------------------------------------------------------
    # Main experiment runner
    # ------------------------------------------------------------------

    def run(
        self,
        model: torch.nn.Module,
        features: ArrayLike,
        mask: ArrayLike,
        mask_fractions: list[float] | None = None,
        num_runs: int = 10,
        baselines: dict[str, Any] | None = None,
        edge_index: torch.Tensor | None = None,
        edge_weight: torch.Tensor | None = None,
        random_seed: int = 42,
        scaler: object | None = None,
    ) -> dict[str, Any]:
        """Run the full masked-value reconstruction benchmark.

        For each ``(fraction, run)`` combination:

        1. Mask a fraction of observed values.
        2. Impute with the GIMIN model (and each baseline).
        3. Compute metrics at held-out positions.

        Args:
            model: Trained GIMIN model.
            features: Original feature matrix, shape ``(N, F)``.
            mask: Binary observation mask, shape ``(N, F)``.
            mask_fractions: List of mask fractions to test.
                Default: ``[0.1, 0.2, 0.3, 0.5]``.
            num_runs: Number of random repetitions per fraction.
                Default: 10.
            baselines: Optional dictionary mapping baseline name to a
                baseline imputer object (must have ``fit_transform``).
            edge_index: Graph edge indices for the GIMIN model.
            edge_weight: Graph edge weights for the GIMIN model.
            random_seed: Base random seed. Default: 42.

        Returns:
            Nested results dictionary with structure::

                {
                    "gimin": {
                        fraction: {
                            "runs": [metrics_dict, ...],
                            "mean": aggregated_metrics,
                            "std": aggregated_stds,
                            "modality_breakdown": {modality: metrics},
                        },
                        ...
                    },
                    "baseline_name": { ... },
                    ...
                    "summary": { ... }
                }
        """
        if mask_fractions is None:
            if self.config is not None:
                mask_fractions = self.config.evaluation.eval_mask_fractions
            else:
                mask_fractions = [0.1, 0.2, 0.3, 0.5]

        features_np = _to_numpy(features)
        mask_np = _to_numpy(mask)
        baselines = baselines or {}

        all_results: dict[str, Any] = {"gimin": {}}
        for bname in baselines:
            all_results[bname] = {}

        total_experiments = len(mask_fractions) * num_runs
        logger.info(
            "Starting masked-value experiment: %d fractions x %d runs = "
            "%d total experiments",
            len(mask_fractions),
            num_runs,
            total_experiments,
        )

        for frac in mask_fractions:
            frac_key = f"{frac:.2f}"
            gimin_runs: list[dict[str, float]] = []
            gimin_modality_runs: list[dict[str, dict[str, float]]] = []
            baseline_runs: dict[str, list[dict[str, float]]] = {
                bname: [] for bname in baselines
            }

            for run_idx in range(num_runs):
                seed = random_seed + run_idx
                rng = np.random.default_rng(seed)

                corrupted_mask, target_mask = self._create_evaluation_mask(
                    mask_np, frac, rng
                )

                # -- GIMIN model --
                t0 = time.time()
                model_result = self._impute_with_model(
                    model,
                    features_np,
                    corrupted_mask,
                    edge_index,
                    edge_weight,
                    scaler=scaler,
                )
                gimin_time = time.time() - t0

                run_metrics = self._compute_metrics(
                    imputed=model_result["imputed"],
                    true_values=features_np,
                    target_mask=target_mask,
                    pred_mean=model_result.get("pred_mean"),
                    pred_std=model_result.get("pred_std"),
                )
                run_metrics["time_seconds"] = gimin_time
                gimin_runs.append(run_metrics)

                mod_metrics = self._compute_modality_metrics(
                    model_result["imputed"], features_np, target_mask
                )
                gimin_modality_runs.append(mod_metrics)

                # -- Baselines --
                for bname, bmodel in baselines.items():
                    t0 = time.time()
                    b_imputed = self._impute_with_baseline(
                        bmodel, features_np, corrupted_mask
                    )
                    b_time = time.time() - t0

                    b_metrics = self._compute_metrics(
                        imputed=b_imputed,
                        true_values=features_np,
                        target_mask=target_mask,
                    )
                    b_metrics["time_seconds"] = b_time
                    baseline_runs[bname].append(b_metrics)

                if (run_idx + 1) % 5 == 0:
                    logger.info(
                        "  frac=%.2f  run %d/%d  GIMIN RMSE=%.4f",
                        frac,
                        run_idx + 1,
                        num_runs,
                        run_metrics["rmse"],
                    )

            # Aggregate GIMIN results for this fraction.
            all_results["gimin"][frac_key] = self._aggregate_runs(
                gimin_runs, gimin_modality_runs
            )

            # Aggregate baseline results.
            for bname in baselines:
                all_results[bname][frac_key] = self._aggregate_runs(
                    baseline_runs[bname]
                )

        # Build summary.
        all_results["summary"] = self._build_summary(all_results, mask_fractions)

        return all_results

    # ------------------------------------------------------------------
    # Aggregation helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _aggregate_runs(
        runs: list[dict[str, float]],
        modality_runs: list[dict[str, dict[str, float]]] | None = None,
    ) -> dict[str, Any]:
        """Aggregate per-run metrics into mean/std summaries."""
        if not runs:
            return {"runs": [], "mean": {}, "std": {}}

        keys = runs[0].keys()
        mean_dict: dict[str, float] = {}
        std_dict: dict[str, float] = {}

        for k in keys:
            values = [r[k] for r in runs if not np.isnan(r.get(k, float("nan")))]
            if values:
                mean_dict[k] = float(np.mean(values))
                std_dict[k] = float(np.std(values))
            else:
                mean_dict[k] = float("nan")
                std_dict[k] = float("nan")

        result: dict[str, Any] = {
            "runs": runs,
            "mean": mean_dict,
            "std": std_dict,
        }

        # Aggregate modality breakdowns.
        if modality_runs:
            mod_agg: dict[str, dict[str, float]] = defaultdict(
                lambda: defaultdict(list)
            )
            for mod_dict in modality_runs:
                for mod_name, mod_metrics in mod_dict.items():
                    for mk, mv in mod_metrics.items():
                        mod_agg[mod_name][mk].append(mv)

            result["modality_breakdown"] = {
                mod_name: {
                    mk: float(np.mean(mv_list))
                    for mk, mv_list in mod_metrics_dict.items()
                }
                for mod_name, mod_metrics_dict in mod_agg.items()
            }

        return result

    @staticmethod
    def _build_summary(
        all_results: dict[str, Any],
        mask_fractions: list[float],
    ) -> dict[str, Any]:
        """Build a high-level summary comparing methods across fractions."""
        summary: dict[str, Any] = {}

        methods = [k for k in all_results if k != "summary"]
        for method in methods:
            method_summary: dict[str, float] = {}
            rmses = []
            for frac in mask_fractions:
                frac_key = f"{frac:.2f}"
                frac_data = all_results[method].get(frac_key, {})
                mean_metrics = frac_data.get("mean", {})
                if "rmse" in mean_metrics:
                    rmses.append(mean_metrics["rmse"])
                    method_summary[f"rmse@{frac_key}"] = mean_metrics["rmse"]

            if rmses:
                method_summary["avg_rmse"] = float(np.mean(rmses))
            summary[method] = method_summary

        return summary
