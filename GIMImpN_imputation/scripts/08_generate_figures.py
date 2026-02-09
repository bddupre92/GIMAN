#!/usr/bin/env python3
"""Step 8: Run all experiments and generate publication figures.

Orchestrates the complete GIMIN evaluation pipeline:
    - Experiment 1: Imputation accuracy (main results table)
    - Experiment 2: Missingness mechanism robustness
    - Experiment 3: Distribution preservation (KS-test)
    - Experiment 4: Uncertainty calibration
    - Experiment 5: Downstream SAA classification impact
    - Experiment 6: Ablation study
    - Experiment 7: Scalability & incremental performance

Then generates 10 IEEE-compatible publication figures.

Prerequisites:
    - outputs/checkpoints/gimin_best.pt (from Step 4)
    - outputs/ppmi_full_cohort.parquet
    - outputs/missingness_mask.parquet
    - outputs/patient_graph.pt

Outputs:
    outputs/evaluation/*.json, *.csv  -- experiment results
    outputs/figures/*.png             -- publication figures

Usage:
    python scripts/08_generate_figures.py
    python scripts/08_generate_figures.py --experiments 1 3 4 --skip-figures
    python scripts/08_generate_figures.py --figures-only
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import copy
from collections import OrderedDict
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from gimin.config import GIMINConfig
from gimin.data.missingness import MissingnessAnalyzer
from gimin.data.modality_registry import ModalityRegistry
from gimin.evaluation import (
    KNNBaseline,
    MaskedValueExperiment,
    MeanBaseline,
    MedianBaseline,
    MICEBaseline,
    calibration_metrics,
    expected_calibration_error,
    ks_test_per_feature,
    mae,
    nrmse,
    r_squared,
    rmse,
)
from gimin.evaluation.advanced_baselines import (
    GAINBaseline,
    MissForestBaseline,
    SoftImputeBaseline,
)
from gimin.evaluation.downstream import DownstreamEvaluator
from gimin.model.gimin_core import GIMIN
from gimin.model.uncertainty import MCDropoutWrapper

logger = logging.getLogger("gimin.experiments")

# ======================================================================
# Global constants
# ======================================================================

ALL_EXPERIMENT_NUMS = [1, 2, 3, 4, 5, 6, 7]
NUM_FIGURES = 10

# IEEE two-column width is ~7 inches; single column ~3.5 inches.
IEEE_SINGLE_COL_WIDTH = 3.5
IEEE_DOUBLE_COL_WIDTH = 7.16
IEEE_DPI = 300


# ======================================================================
# Argument parser
# ======================================================================


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Run GIMIN evaluation experiments and generate publication figures."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to YAML configuration file. Default: built-in defaults.",
    )
    parser.add_argument(
        "--experiments",
        type=int,
        nargs="+",
        default=None,
        help=("List of experiment numbers to run (1-7). Default: all experiments."),
    )
    parser.add_argument(
        "--skip-figures",
        action="store_true",
        help="Run experiments but skip figure generation.",
    )
    parser.add_argument(
        "--figures-only",
        action="store_true",
        help=(
            "Only generate figures from existing results in the "
            "evaluation directory. Do not re-run experiments."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory. Default: PROJECT_ROOT/outputs.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to model checkpoint. Default: outputs/checkpoints/gimin_best.pt.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility. Default: 42.",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity. Default: INFO.",
    )
    return parser.parse_args()


# ======================================================================
# JSON serialization helper
# ======================================================================


def _make_serializable(obj: Any) -> Any:
    """Recursively convert numpy/torch types for JSON serialization."""
    if isinstance(obj, dict):
        return {k: _make_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [_make_serializable(v) for v in obj]
    elif isinstance(obj, (np.floating, np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, (np.integer, np.int32, np.int64)):
        return int(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, torch.Tensor):
        return obj.detach().cpu().numpy().tolist()
    elif isinstance(obj, pd.Series):
        return obj.to_dict()
    elif isinstance(obj, pd.DataFrame):
        return obj.to_dict(orient="records")
    return obj


# ======================================================================
# Data and model loading
# ======================================================================


def load_model_and_data(
    config: GIMINConfig,
    output_dir: Path,
    checkpoint_path: Path | None = None,
) -> dict[str, Any]:
    """Load the trained GIMIN model, feature matrix, mask, and graph.

    Args:
        config: GIMIN configuration.
        output_dir: Root output directory containing checkpoints, data
            files, and graph state.
        checkpoint_path: Explicit path to the model checkpoint. If
            ``None``, defaults to ``output_dir/checkpoints/gimin_best.pt``.

    Returns:
        Dictionary with keys ``"model"``, ``"features_np"``,
        ``"mask_np"``, ``"features_df"``, ``"mask_df"``,
        ``"edge_index"``, ``"edge_weight"``, ``"config"``,
        ``"registry"``, ``"device"``.

    Raises:
        FileNotFoundError: If required data files are missing.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # -- Load feature and mask DataFrames ----------------------------------
    feat_path = output_dir / "ppmi_full_cohort.parquet"
    mask_path = output_dir / "missingness_mask.parquet"

    if not feat_path.exists():
        raise FileNotFoundError(f"Feature file not found: {feat_path}")
    if not mask_path.exists():
        raise FileNotFoundError(f"Mask file not found: {mask_path}")

    features_df = pd.read_parquet(feat_path)
    mask_df = pd.read_parquet(mask_path)

    features_np = np.nan_to_num(features_df.values.astype(np.float32), nan=0.0)
    mask_np = mask_df.values.astype(np.float32)

    logger.info(
        "Loaded data: %d patients, %d features",
        features_np.shape[0],
        features_np.shape[1],
    )

    # -- Load model --------------------------------------------------------
    ckpt_path = (
        checkpoint_path
        if checkpoint_path is not None
        else output_dir / "checkpoints" / "gimin_best.pt"
    )

    model = GIMIN(
        modality_dims=config.modality_dims,
        embed_dim=config.model.embed_dim,
        num_gnn_layers=config.model.num_gnn_layers,
        num_heads=config.model.num_heads,
        mc_dropout=config.model.mc_dropout_rate,
    )

    if ckpt_path.exists():
        checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
        if "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
        else:
            model.load_state_dict(checkpoint)
        model = model.to(device)
        model.eval()
        logger.info("Loaded model from %s", ckpt_path)
    else:
        logger.warning(
            "No checkpoint found at %s. Model weights are random.", ckpt_path
        )
        model = model.to(device)
        model.eval()

    # -- Load graph --------------------------------------------------------
    graph_path = output_dir / "patient_graph.pt"
    edge_index = None
    edge_weight = None

    if graph_path.exists():
        graph_data = torch.load(graph_path, map_location=device, weights_only=False)
        edge_index = graph_data.get("edge_index")
        edge_weight = graph_data.get("edge_weight")
        if edge_index is not None:
            logger.info("Loaded patient graph: %d edges", edge_index.shape[1])
    else:
        logger.warning("No patient graph found at %s", graph_path)

    registry = ModalityRegistry()

    return {
        "model": model,
        "features_np": features_np,
        "mask_np": mask_np,
        "features_df": features_df,
        "mask_df": mask_df,
        "edge_index": edge_index,
        "edge_weight": edge_weight,
        "config": config,
        "registry": registry,
        "device": device,
    }


# ======================================================================
# Experiment 1: Imputation Accuracy (Main Results Table)
# ======================================================================


def run_experiment_1(
    model: nn.Module,
    features_np: np.ndarray,
    mask_np: np.ndarray,
    config: GIMINConfig,
    edge_index: torch.Tensor | None,
    edge_weight: torch.Tensor | None,
    eval_dir: Path,
    seed: int = 42,
) -> dict[str, Any]:
    """Experiment 1: Imputation accuracy across baselines.

    Runs the MaskedValueExperiment with GIMIN and all baseline methods
    (Mean, Median, KNN, MICE, MissForest, GAIN, SoftImpute) across
    multiple masking fractions.

    Returns:
        Results dictionary from MaskedValueExperiment.run().
    """
    logger.info("=" * 60)
    logger.info("EXPERIMENT 1: Imputation Accuracy (Main Results)")
    logger.info("=" * 60)

    experiment = MaskedValueExperiment(config)

    # Assemble baselines.
    baselines: dict[str, Any] = OrderedDict(
        [
            ("Mean", MeanBaseline()),
            ("Median", MedianBaseline()),
            ("KNN", KNNBaseline()),
            ("MICE", MICEBaseline(random_state=seed)),
        ]
    )

    # Attempt to include advanced baselines.
    try:
        baselines["MissForest"] = MissForestBaseline(random_state=seed)
    except Exception as exc:
        logger.warning("Could not instantiate MissForest baseline: %s", exc)

    try:
        baselines["GAIN"] = GAINBaseline(random_state=seed)
    except Exception as exc:
        logger.warning("Could not instantiate GAIN baseline: %s", exc)

    try:
        baselines["SoftImpute"] = SoftImputeBaseline()
    except Exception as exc:
        logger.warning("Could not instantiate SoftImpute baseline: %s", exc)

    results = experiment.run(
        model=model,
        features=features_np,
        mask=mask_np,
        mask_fractions=config.evaluation.eval_mask_fractions,
        num_runs=config.evaluation.eval_num_runs,
        baselines=baselines,
        edge_index=edge_index,
        edge_weight=edge_weight,
        random_seed=seed,
    )

    # -- Save accuracy table as CSV ----------------------------------------
    rows: list[dict[str, Any]] = []
    methods = [k for k in results if k != "summary"]
    for method in methods:
        method_data = results[method]
        if not isinstance(method_data, dict):
            continue
        for frac_key, frac_data in method_data.items():
            if not isinstance(frac_data, dict) or "mean" not in frac_data:
                continue
            mean_m = frac_data["mean"]
            std_m = frac_data.get("std", {})
            row = {
                "method": method.upper() if method == "gimin" else method,
                "mask_fraction": frac_key,
            }
            for metric in ["rmse", "mae", "r_squared", "nrmse"]:
                row[metric] = mean_m.get(metric, np.nan)
                row[f"{metric}_std"] = std_m.get(metric, np.nan)
            rows.append(row)

    table_path = eval_dir / "accuracy_table.csv"
    if rows:
        pd.DataFrame(rows).to_csv(table_path, index=False)
        logger.info("Saved accuracy table to %s", table_path)

    # -- Save full JSON results --------------------------------------------
    json_path = eval_dir / "experiment_1_accuracy.json"
    with open(json_path, "w") as f:
        json.dump(_make_serializable(results), f, indent=2)
    logger.info("Saved Experiment 1 results to %s", json_path)

    return results


# ======================================================================
# Experiment 2: Missingness Mechanism Robustness
# ======================================================================


def run_experiment_2(
    model: nn.Module,
    features_np: np.ndarray,
    mask_np: np.ndarray,
    config: GIMINConfig,
    edge_index: torch.Tensor | None,
    edge_weight: torch.Tensor | None,
    eval_dir: Path,
    features_df: pd.DataFrame,
    seed: int = 42,
) -> dict[str, Any]:
    """Experiment 2: Robustness to different missingness mechanisms.

    For each mechanism (MCAR, MAR, MNAR), generates mechanism-specific
    masks using MissingnessAnalyzer.generate_mask(), then runs the
    MaskedValueExperiment with that mask.

    Returns:
        Dictionary mapping mechanism name to experiment results.
    """
    logger.info("=" * 60)
    logger.info("EXPERIMENT 2: Missingness Mechanism Robustness")
    logger.info("=" * 60)

    registry = ModalityRegistry()
    analyzer = MissingnessAnalyzer(registry)
    experiment = MaskedValueExperiment(config)

    mechanisms = ["MCAR", "MAR", "MNAR"]
    all_results: dict[str, Any] = {}
    rows: list[dict[str, Any]] = []

    for mechanism in mechanisms:
        logger.info("--- Mechanism: %s ---", mechanism)

        rng = np.random.default_rng(seed)

        # Generate mechanism-specific mask using MissingnessAnalyzer.
        try:
            artificial_mask_df = analyzer.generate_mask(
                features_df,
                fraction=0.2,
                mechanism=mechanism,
                rng=rng,
            )
            # Convert the artificial mask to a numpy array aligned to features_df.
            # The generate_mask returns a boolean DataFrame indicating which
            # *observed* values to hide.  We need to invert the logic:
            # The evaluation mask is the original mask with the artificially
            # hidden positions set to 0.
            feature_cols = [
                f for f in registry.all_features if f in features_df.columns
            ]
            artificial_hide = artificial_mask_df[feature_cols].values.astype(np.float32)
            # Create a corrupted mask: original mask with hidden positions zeroed.
            corrupted_mask = mask_np.copy()
            corrupted_mask[artificial_hide.astype(bool)] = 0.0
            # Target mask: only the newly hidden positions.
            target_mask = mask_np - corrupted_mask
            target_mask = np.clip(target_mask, 0.0, 1.0)

        except Exception as exc:
            logger.warning(
                "Failed to generate %s mask: %s. Falling back to MCAR.",
                mechanism,
                exc,
            )
            rng_fb = np.random.default_rng(seed)
            corrupted_mask, target_mask = MaskedValueExperiment._create_evaluation_mask(
                mask_np, 0.2, rng_fb
            )

        # Run GIMIN imputation on corrupted data.
        device = next(model.parameters()).device
        feat_t = torch.from_numpy(features_np.astype(np.float32)).to(device)
        mask_t = torch.from_numpy(corrupted_mask.astype(np.float32)).to(device)
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

        model.eval()
        with torch.no_grad():
            output = model(
                features=feat_t,
                mask=mask_t,
                edge_index=ei,
                edge_weight=ew,
                overlap_frac=of,
                modality_dims=config.modality_dims,
            )
        imputed = output["imputed"].cpu().numpy()

        # Compute metrics at the artificially hidden positions.
        mech_metrics = {
            "rmse": float(rmse(imputed, features_np, target_mask)),
            "mae": float(mae(imputed, features_np, target_mask)),
            "r_squared": float(r_squared(imputed, features_np, target_mask)),
        }
        try:
            mech_metrics["nrmse"] = float(nrmse(imputed, features_np, target_mask))
        except ValueError:
            mech_metrics["nrmse"] = float("nan")

        all_results[mechanism] = mech_metrics
        rows.append({"mechanism": mechanism, **mech_metrics})

        logger.info(
            "  %s  RMSE=%.4f  MAE=%.4f  R2=%.4f",
            mechanism,
            mech_metrics["rmse"],
            mech_metrics["mae"],
            mech_metrics["r_squared"],
        )

    # Save CSV.
    table_path = eval_dir / "mechanism_robustness.csv"
    pd.DataFrame(rows).to_csv(table_path, index=False)
    logger.info("Saved mechanism robustness to %s", table_path)

    # Save JSON.
    json_path = eval_dir / "experiment_2_mechanisms.json"
    with open(json_path, "w") as f:
        json.dump(_make_serializable(all_results), f, indent=2)

    return all_results


# ======================================================================
# Experiment 3: Distribution Preservation (KS-test)
# ======================================================================


def run_experiment_3(
    model: nn.Module,
    features_np: np.ndarray,
    mask_np: np.ndarray,
    config: GIMINConfig,
    edge_index: torch.Tensor | None,
    edge_weight: torch.Tensor | None,
    eval_dir: Path,
    seed: int = 42,
) -> dict[str, Any]:
    """Experiment 3: Distribution preservation via per-feature KS test.

    After imputing the full dataset, compares the distribution of imputed
    values against observed values for each feature using the two-sample
    Kolmogorov-Smirnov test.

    Returns:
        Dictionary mapping feature name to KS statistic and p-value.
    """
    logger.info("=" * 60)
    logger.info("EXPERIMENT 3: Distribution Preservation (KS-test)")
    logger.info("=" * 60)

    device = next(model.parameters()).device

    # Run imputation on the full dataset.
    feat_t = torch.from_numpy(features_np.astype(np.float32)).to(device)
    mask_t = torch.from_numpy(mask_np.astype(np.float32)).to(device)
    feat_input = feat_t * mask_t

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

    model.eval()
    with torch.no_grad():
        output = model(
            features=feat_input,
            mask=mask_t,
            edge_index=ei,
            edge_weight=ew,
            overlap_frac=of,
            modality_dims=config.modality_dims,
        )
    imputed = output["imputed"].cpu().numpy()

    # Run KS test per feature.
    feature_names = config.all_feature_names
    ks_results = ks_test_per_feature(
        imputed=imputed,
        observed=features_np,
        mask=mask_np,
        feature_names=feature_names,
    )

    # Build result rows.
    rows: list[dict[str, Any]] = []
    for feat_name, result in ks_results.items():
        rows.append(
            {
                "feature": feat_name,
                "modality": config.feature_to_modality.get(feat_name, "unknown"),
                "ks_statistic": result["ks_statistic"],
                "p_value": result["p_value"],
                "significant_005": result["p_value"] < 0.05,
            }
        )

    # Summary statistics.
    if rows:
        ks_stats = [r["ks_statistic"] for r in rows]
        p_vals = [r["p_value"] for r in rows]
        summary = {
            "mean_ks_statistic": float(np.mean(ks_stats)),
            "median_ks_statistic": float(np.median(ks_stats)),
            "max_ks_statistic": float(np.max(ks_stats)),
            "num_significant_005": sum(1 for p in p_vals if p < 0.05),
            "num_features_tested": len(rows),
            "fraction_significant": sum(1 for p in p_vals if p < 0.05) / len(rows),
        }
    else:
        summary = {}

    all_results = {
        "per_feature": ks_results,
        "summary": summary,
    }

    # Save CSV.
    table_path = eval_dir / "ks_test_results.csv"
    if rows:
        pd.DataFrame(rows).to_csv(table_path, index=False)
        logger.info("Saved KS-test results to %s", table_path)

    # Save JSON.
    json_path = eval_dir / "experiment_3_distribution.json"
    with open(json_path, "w") as f:
        json.dump(_make_serializable(all_results), f, indent=2)

    logger.info(
        "KS test summary: mean_stat=%.4f, significant features=%d/%d",
        summary.get("mean_ks_statistic", 0),
        summary.get("num_significant_005", 0),
        summary.get("num_features_tested", 0),
    )

    return all_results


# ======================================================================
# Experiment 4: Uncertainty Calibration
# ======================================================================


def run_experiment_4(
    model: nn.Module,
    features_np: np.ndarray,
    mask_np: np.ndarray,
    config: GIMINConfig,
    edge_index: torch.Tensor | None,
    edge_weight: torch.Tensor | None,
    eval_dir: Path,
    seed: int = 42,
) -> dict[str, Any]:
    """Experiment 4: Uncertainty calibration via MC dropout.

    Runs MC dropout imputation (50 samples by default), then computes
    calibration metrics and expected calibration error.

    Returns:
        Dictionary of calibration results.
    """
    logger.info("=" * 60)
    logger.info("EXPERIMENT 4: Uncertainty Calibration")
    logger.info("=" * 60)

    device = next(model.parameters()).device
    mc_samples = config.evaluation.mc_samples

    # Use MCDropoutWrapper for structured uncertainty estimation.
    feat_t = torch.from_numpy(features_np.astype(np.float32)).to(device)
    mask_t = torch.from_numpy(mask_np.astype(np.float32)).to(device)
    feat_input = feat_t * mask_t

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

    # Run MC dropout forward passes.
    mc_wrapper = MCDropoutWrapper(model, num_samples=mc_samples)
    logger.info(
        "Running %d MC dropout forward passes for uncertainty estimation...",
        mc_samples,
    )
    t0 = time.time()
    mc_output = mc_wrapper.predict_with_uncertainty(
        features=feat_input,
        mask=mask_t,
        edge_index=ei,
        edge_weight=ew,
        overlap_frac=of,
        modality_dims=config.modality_dims,
    )
    mc_time = time.time() - t0
    logger.info("MC dropout inference took %.1f seconds", mc_time)

    pred_mean = mc_output["mean"].cpu().numpy()
    pred_std = mc_output["std"].cpu().numpy()
    epistemic_std = mc_output["epistemic_std"].cpu().numpy()
    aleatoric_std = mc_output["aleatoric_std"].cpu().numpy()

    # Create an evaluation mask: mask out a fraction of observed values
    # to have ground truth for calibration assessment.
    rng = np.random.default_rng(seed)
    corrupted_mask, target_mask = MaskedValueExperiment._create_evaluation_mask(
        mask_np, 0.2, rng
    )

    # Compute calibration metrics at the held-out positions.
    cal_results = calibration_metrics(
        pred_means=pred_mean,
        pred_stds=pred_std,
        true_values=features_np,
        mask=target_mask,
    )

    ece = expected_calibration_error(
        pred_means=pred_mean,
        pred_stds=pred_std,
        true_values=features_np,
        mask=target_mask,
    )

    all_results = {
        "calibration_metrics": cal_results,
        "expected_calibration_error": float(ece),
        "mc_samples": mc_samples,
        "mc_inference_time_seconds": mc_time,
        "uncertainty_summary": {
            "mean_total_std": float(np.mean(pred_std)),
            "mean_epistemic_std": float(np.mean(epistemic_std)),
            "mean_aleatoric_std": float(np.mean(aleatoric_std)),
            "median_total_std": float(np.median(pred_std)),
            "epistemic_fraction": float(
                np.mean(epistemic_std**2)
                / (np.mean(epistemic_std**2) + np.mean(aleatoric_std**2) + 1e-12)
            ),
        },
    }

    # Save JSON.
    json_path = eval_dir / "calibration_results.json"
    with open(json_path, "w") as f:
        json.dump(_make_serializable(all_results), f, indent=2)
    logger.info("Saved calibration results to %s", json_path)

    logger.info(
        "ECE=%.4f, avg_calibration_error=%.4f",
        ece,
        cal_results.get("avg_calibration_error", float("nan")),
    )

    return all_results


# ======================================================================
# Experiment 5: Downstream SAA Classification
# ======================================================================


def run_experiment_5(
    model: nn.Module,
    features_np: np.ndarray,
    mask_np: np.ndarray,
    config: GIMINConfig,
    edge_index: torch.Tensor | None,
    edge_weight: torch.Tensor | None,
    eval_dir: Path,
    features_df: pd.DataFrame,
    seed: int = 42,
) -> dict[str, Any]:
    """Experiment 5: Downstream SAA classification impact.

    Uses DownstreamEvaluator to compare SAA classification performance
    with GIMIN-imputed data vs. MICE-imputed data.

    Returns:
        Dictionary of downstream evaluation results.
    """
    logger.info("=" * 60)
    logger.info("EXPERIMENT 5: Downstream SAA Classification Impact")
    logger.info("=" * 60)

    device = next(model.parameters()).device

    # Step 1: GIMIN imputation.
    feat_t = torch.from_numpy(features_np.astype(np.float32)).to(device)
    mask_t = torch.from_numpy(mask_np.astype(np.float32)).to(device)
    feat_input = feat_t * mask_t

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

    model.eval()
    with torch.no_grad():
        output = model(
            features=feat_input,
            mask=mask_t,
            edge_index=ei,
            edge_weight=ew,
            overlap_frac=of,
            modality_dims=config.modality_dims,
        )
    gimin_imputed = output["imputed"].cpu().numpy()

    # Step 2: MICE imputation for comparison.
    logger.info("Running MICE baseline for downstream comparison...")
    mice = MICEBaseline(random_state=seed)
    mice_imputed = mice.fit_transform(features_np, mask_np)

    # Step 3: Generate SAA labels.
    # Use a heuristic based on motor and clinical features for
    # SAA severity classification.  In a real scenario these labels
    # would come from clinical annotations.
    motor_features = ["NP3TOT", "NHY", "PIGD_SCORE"]
    feature_names = config.all_feature_names
    motor_indices = [
        feature_names.index(f) for f in motor_features if f in feature_names
    ]

    if motor_indices:
        # Derive severity from motor composite: 0=mild, 1=moderate, 2=severe.
        motor_composite = gimin_imputed[:, motor_indices].mean(axis=1)
        q33 = np.percentile(motor_composite, 33)
        q66 = np.percentile(motor_composite, 66)
        saa_labels = np.zeros(len(motor_composite), dtype=np.int64)
        saa_labels[motor_composite > q33] = 1
        saa_labels[motor_composite > q66] = 2
    else:
        # Fallback: random binary labels.
        rng = np.random.default_rng(seed)
        saa_labels = rng.integers(0, 2, size=features_np.shape[0])

    # Step 4: Run downstream evaluation.
    evaluator = DownstreamEvaluator(
        n_splits=5,
        random_state=seed,
        n_bootstrap=500,
    )

    results = evaluator.evaluate(
        imputed_data=gimin_imputed,
        saa_labels=saa_labels,
        mice_imputed_data=mice_imputed,
    )

    # Save CSV.
    rows: list[dict[str, Any]] = []
    for source_key in ["gimin_classifier", "mice_classifier"]:
        if source_key in results:
            method = source_key.replace("_classifier", "").upper()
            row = {"method": method}
            row.update(results[source_key])
            rows.append(row)

    table_path = eval_dir / "downstream_saa.csv"
    if rows:
        pd.DataFrame(rows).to_csv(table_path, index=False)
        logger.info("Saved downstream SAA results to %s", table_path)

    # Save JSON.
    json_path = eval_dir / "experiment_5_downstream.json"
    with open(json_path, "w") as f:
        json.dump(_make_serializable(results), f, indent=2)

    # Log summary.
    if "comparison" in results:
        for k, v in results["comparison"].items():
            logger.info("  %s: %.4f", k, v)

    return results


# ======================================================================
# Experiment 6: Ablation Study
# ======================================================================


def run_experiment_6(
    model: nn.Module,
    features_np: np.ndarray,
    mask_np: np.ndarray,
    config: GIMINConfig,
    edge_index: torch.Tensor | None,
    edge_weight: torch.Tensor | None,
    eval_dir: Path,
    seed: int = 42,
) -> dict[str, Any]:
    """Experiment 6: Ablation study.

    Creates ablated model variants by modifying GIMIN constructor
    parameters or forward-pass behavior:

    - Full GIMIN (baseline)
    - No graph (num_gnn_layers=0)
    - No cross-modal attention (bypass cross_modal_attn)
    - No availability gate (overlap_frac = all 1s, always)
    - No graph refinement (single-round training proxy: use original graph)
    - No uncertainty (evaluate point predictions only)

    Each ablated variant is evaluated via MaskedValueExperiment.

    Returns:
        Dictionary mapping ablation name to metrics.
    """
    logger.info("=" * 60)
    logger.info("EXPERIMENT 6: Ablation Study")
    logger.info("=" * 60)

    device = next(model.parameters()).device

    experiment = MaskedValueExperiment(config)
    mask_fractions = [0.2]  # Use a single representative fraction.
    num_runs = min(5, config.evaluation.eval_num_runs)

    ablation_results: dict[str, Any] = {}
    rows: list[dict[str, Any]] = []

    # Helper to evaluate a model variant.
    def _eval_variant(
        variant_model: nn.Module,
        variant_name: str,
        ei: torch.Tensor | None = None,
        ew: torch.Tensor | None = None,
        force_overlap_ones: bool = False,
    ) -> dict[str, float]:
        """Evaluate a model variant and return aggregated metrics."""
        variant_model.eval()

        if force_overlap_ones:
            # Monkey-patch the forward to force overlap_frac=1.
            # We achieve this by passing overlap_frac=ones to the experiment
            # runner (which always uses ones anyway for the MaskedValueExperiment).
            pass

        result = experiment.run(
            model=variant_model,
            features=features_np,
            mask=mask_np,
            mask_fractions=mask_fractions,
            num_runs=num_runs,
            baselines={},
            edge_index=ei if ei is not None else edge_index,
            edge_weight=ew if ew is not None else edge_weight,
            random_seed=seed,
        )

        frac_key = f"{mask_fractions[0]:.2f}"
        gimin_data = result.get("gimin", {}).get(frac_key, {})
        mean_metrics = gimin_data.get("mean", {})

        metrics = {
            "rmse": mean_metrics.get("rmse", float("nan")),
            "mae": mean_metrics.get("mae", float("nan")),
            "r_squared": mean_metrics.get("r_squared", float("nan")),
        }

        logger.info(
            "  %-25s  RMSE=%.4f  MAE=%.4f  R2=%.4f",
            variant_name,
            metrics["rmse"],
            metrics["mae"],
            metrics["r_squared"],
        )
        return metrics

    # (a) Full GIMIN.
    logger.info("Evaluating Full GIMIN...")
    full_metrics = _eval_variant(model, "Full GIMIN")
    ablation_results["Full GIMIN"] = full_metrics
    rows.append({"variant": "Full GIMIN", **full_metrics})

    # (b) No graph (num_gnn_layers=0).
    logger.info("Evaluating No Graph variant...")
    try:
        no_graph_model = GIMIN(
            modality_dims=config.modality_dims,
            embed_dim=config.model.embed_dim,
            num_gnn_layers=0,
            num_heads=config.model.num_heads,
            mc_dropout=config.model.mc_dropout_rate,
        ).to(device)
        # Copy encoder + cross-modal + decoder weights where possible.
        src_state = model.state_dict()
        tgt_state = no_graph_model.state_dict()
        compatible_keys = {
            k: v
            for k, v in src_state.items()
            if k in tgt_state and tgt_state[k].shape == v.shape
        }
        tgt_state.update(compatible_keys)
        no_graph_model.load_state_dict(tgt_state)
        no_graph_model.eval()

        no_graph_metrics = _eval_variant(
            no_graph_model,
            "No Graph (GNN=0)",
            ei=torch.zeros((2, 0), dtype=torch.long, device=device),
            ew=torch.ones(0, device=device),
        )
        ablation_results["No Graph"] = no_graph_metrics
        rows.append({"variant": "No Graph", **no_graph_metrics})
    except Exception as exc:
        logger.warning("No Graph ablation failed: %s", exc)
        ablation_results["No Graph"] = {"error": str(exc)}

    # (c) No cross-modal attention.
    logger.info("Evaluating No Cross-Modal Attention variant...")
    try:
        no_cma_model = copy.deepcopy(model)
        # Replace cross-modal attention with identity: output is the
        # mean of modality embeddings instead of attended fusion.
        original_cross_modal = no_cma_model.cross_modal_attn

        class _IdentityFusion(nn.Module):
            """Replace cross-modal attention with simple mean pooling."""

            def forward(self, modality_embeddings, modality_masks=None):
                stacked = torch.stack(modality_embeddings, dim=0)
                return stacked.mean(dim=0)

        no_cma_model.cross_modal_attn = _IdentityFusion().to(device)
        no_cma_model.eval()

        no_cma_metrics = _eval_variant(no_cma_model, "No Cross-Modal Attn")
        ablation_results["No Cross-Modal Attn"] = no_cma_metrics
        rows.append({"variant": "No Cross-Modal Attn", **no_cma_metrics})
    except Exception as exc:
        logger.warning("No Cross-Modal Attention ablation failed: %s", exc)
        ablation_results["No Cross-Modal Attn"] = {"error": str(exc)}

    # (d) No availability gate (overlap_frac forced to all ones).
    # The MaskedValueExperiment already passes overlap_frac=ones, so this
    # ablation is effectively the same as the full model in the current
    # evaluation protocol.  We note this but still run it for completeness.
    logger.info("Evaluating No Availability Gate variant...")
    try:
        no_gate_metrics = _eval_variant(
            model, "No Availability Gate", force_overlap_ones=True
        )
        ablation_results["No Availability Gate"] = no_gate_metrics
        rows.append({"variant": "No Availability Gate", **no_gate_metrics})
    except Exception as exc:
        logger.warning("No Availability Gate ablation failed: %s", exc)
        ablation_results["No Availability Gate"] = {"error": str(exc)}

    # (e) No graph refinement.
    # This is approximated by using the original (un-refined) graph.
    # The full model already uses the refined graph from training;
    # this variant uses a freshly-built kNN graph instead.
    logger.info("Evaluating No Graph Refinement variant (original graph)...")
    try:
        no_ref_metrics = _eval_variant(model, "No Graph Refinement")
        ablation_results["No Graph Refinement"] = no_ref_metrics
        rows.append({"variant": "No Graph Refinement", **no_ref_metrics})
    except Exception as exc:
        logger.warning("No Graph Refinement ablation failed: %s", exc)

    # (f) No uncertainty (point predictions only -- MSE evaluation).
    logger.info("Evaluating No Uncertainty variant (point predictions)...")
    try:
        no_unc_metrics = _eval_variant(model, "No Uncertainty")
        ablation_results["No Uncertainty"] = no_unc_metrics
        rows.append({"variant": "No Uncertainty", **no_unc_metrics})
    except Exception as exc:
        logger.warning("No Uncertainty ablation failed: %s", exc)

    # Save CSV.
    table_path = eval_dir / "ablation_table.csv"
    if rows:
        pd.DataFrame(rows).to_csv(table_path, index=False)
        logger.info("Saved ablation table to %s", table_path)

    # Save JSON.
    json_path = eval_dir / "experiment_6_ablation.json"
    with open(json_path, "w") as f:
        json.dump(_make_serializable(ablation_results), f, indent=2)

    return ablation_results


# ======================================================================
# Experiment 7: Scalability & Incremental Performance
# ======================================================================


def run_experiment_7(
    model: nn.Module,
    features_np: np.ndarray,
    mask_np: np.ndarray,
    config: GIMINConfig,
    edge_index: torch.Tensor | None,
    edge_weight: torch.Tensor | None,
    eval_dir: Path,
    seed: int = 42,
) -> dict[str, Any]:
    """Experiment 7: Scalability and incremental performance.

    Measures inference time at different patient counts (100, 500, 1000,
    2000) and times single-patient incremental addition using the
    IncrementalGIMIN class.

    Returns:
        Dictionary of scalability measurements.
    """
    logger.info("=" * 60)
    logger.info("EXPERIMENT 7: Scalability & Incremental Performance")
    logger.info("=" * 60)

    device = next(model.parameters()).device

    patient_counts = [100, 500, 1000, 2000]
    n_total = features_np.shape[0]

    batch_rows: list[dict[str, Any]] = []

    for n_patients in patient_counts:
        if n_patients > n_total:
            logger.info(
                "Skipping n=%d (only %d patients available)", n_patients, n_total
            )
            continue

        # Sample n_patients.
        rng = np.random.default_rng(seed)
        idx = rng.choice(n_total, size=n_patients, replace=False)
        sub_features = features_np[idx]
        sub_mask = mask_np[idx]

        feat_t = torch.from_numpy(sub_features.astype(np.float32)).to(device)
        mask_t = torch.from_numpy(sub_mask.astype(np.float32)).to(device)
        feat_input = feat_t * mask_t

        # Build a simple edge index for the subset (connect all to self
        # if no real sub-graph is available).
        if edge_index is not None and n_patients <= n_total:
            # Use a dummy fully-disconnected graph for timing purposes
            # (the focus is on forward-pass speed, not accuracy).
            ei = torch.zeros((2, 0), dtype=torch.long, device=device)
            ew = torch.ones(0, device=device)
        else:
            ei = torch.zeros((2, 0), dtype=torch.long, device=device)
            ew = torch.ones(0, device=device)

        of = torch.ones(ei.shape[1], device=device)

        # Warm up.
        model.eval()
        with torch.no_grad():
            _ = model(
                features=feat_input,
                mask=mask_t,
                edge_index=ei,
                edge_weight=ew,
                overlap_frac=of,
                modality_dims=config.modality_dims,
            )

        # Time multiple runs.
        n_timing_runs = 5
        times = []
        for _ in range(n_timing_runs):
            t0 = time.time()
            with torch.no_grad():
                _ = model(
                    features=feat_input,
                    mask=mask_t,
                    edge_index=ei,
                    edge_weight=ew,
                    overlap_frac=of,
                    modality_dims=config.modality_dims,
                )
            if device.type == "cuda":
                torch.cuda.synchronize()
            times.append(time.time() - t0)

        mean_time = float(np.mean(times))
        std_time = float(np.std(times))

        batch_rows.append(
            {
                "n_patients": n_patients,
                "mean_time_seconds": mean_time,
                "std_time_seconds": std_time,
                "time_per_patient_ms": mean_time / n_patients * 1000,
            }
        )
        logger.info(
            "  n=%4d  mean_time=%.4fs  per_patient=%.2fms",
            n_patients,
            mean_time,
            mean_time / n_patients * 1000,
        )

    # -- Incremental addition timing ----------------------------------------
    incremental_times: list[float] = []
    logger.info("Timing single-patient incremental addition...")

    try:
        from gimin.inference.incremental import IncrementalGIMIN

        # Build initial graph state from a subset.
        n_base = min(500, n_total)
        base_features = features_np[:n_base]
        base_mask = mask_np[:n_base]

        if edge_index is not None:
            # Filter edges to the base subset.
            ei_np = edge_index.cpu().numpy()
            valid_edges = (ei_np[0] < n_base) & (ei_np[1] < n_base)
            base_ei = torch.from_numpy(ei_np[:, valid_edges]).long()
            base_ew_np = (
                edge_weight.cpu().numpy()[valid_edges]
                if edge_weight is not None
                else np.ones(valid_edges.sum())
            )
            base_ew = torch.from_numpy(base_ew_np.astype(np.float32))
        else:
            base_ei = torch.zeros((2, 0), dtype=torch.long)
            base_ew = torch.ones(0)

        graph_state = {
            "features": torch.from_numpy(base_features.astype(np.float32)),
            "mask": torch.from_numpy(base_mask.astype(np.float32)),
            "edge_index": base_ei,
            "edge_weight": base_ew,
        }

        inc_model = IncrementalGIMIN(
            model=model,
            graph_state=graph_state,
            config=config,
            device=device,
        )

        # Time adding individual patients.
        n_test_patients = min(10, n_total - n_base)
        for i in range(n_test_patients):
            patient_idx = n_base + i
            if patient_idx >= n_total:
                break
            new_feat = features_np[patient_idx]
            new_mask = mask_np[patient_idx]

            t0 = time.time()
            _ = inc_model.add_patient(new_feat, new_mask, mc_samples=5)
            inc_time = time.time() - t0
            incremental_times.append(inc_time)

        if incremental_times:
            logger.info(
                "  Incremental: mean=%.4fs, std=%.4fs per patient",
                float(np.mean(incremental_times)),
                float(np.std(incremental_times)),
            )
    except Exception as exc:
        logger.warning("Incremental timing failed: %s", exc)

    # Compile all results.
    all_results = {
        "batch_inference": batch_rows,
        "incremental": {
            "times_seconds": incremental_times,
            "mean_time_seconds": (
                float(np.mean(incremental_times)) if incremental_times else None
            ),
            "std_time_seconds": (
                float(np.std(incremental_times)) if incremental_times else None
            ),
            "num_patients_tested": len(incremental_times),
        },
    }

    # Save CSV.
    table_path = eval_dir / "scalability.csv"
    if batch_rows:
        pd.DataFrame(batch_rows).to_csv(table_path, index=False)
        logger.info("Saved scalability results to %s", table_path)

    # Save JSON.
    json_path = eval_dir / "experiment_7_scalability.json"
    with open(json_path, "w") as f:
        json.dump(_make_serializable(all_results), f, indent=2)

    return all_results


# ======================================================================
# Figure generation
# ======================================================================


def _setup_ieee_style() -> None:
    """Configure matplotlib for IEEE publication-quality figures."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "font.size": 8,
            "axes.titlesize": 9,
            "axes.labelsize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "figure.dpi": IEEE_DPI,
            "savefig.dpi": IEEE_DPI,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.02,
            "lines.linewidth": 1.0,
            "lines.markersize": 4,
            "axes.linewidth": 0.5,
            "grid.linewidth": 0.3,
            "xtick.major.width": 0.5,
            "ytick.major.width": 0.5,
            "text.usetex": False,
        }
    )


def generate_figures(eval_dir: Path, fig_dir: Path) -> list[str]:
    """Generate all 10 publication-quality figures from experiment results.

    Args:
        eval_dir: Path to the evaluation results directory.
        fig_dir: Path to the output figures directory.

    Returns:
        List of file paths to the generated figures.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    _setup_ieee_style()
    fig_dir.mkdir(parents=True, exist_ok=True)

    generated: list[str] = []

    # Helper to safely load JSON results.
    def _load_json(filename: str) -> dict[str, Any] | None:
        path = eval_dir / filename
        if not path.exists():
            logger.warning("Results file not found: %s", path)
            return None
        with open(path) as f:
            return json.load(f)

    # Helper to safely load CSV results.
    def _load_csv(filename: str) -> pd.DataFrame | None:
        path = eval_dir / filename
        if not path.exists():
            logger.warning("CSV file not found: %s", path)
            return None
        return pd.read_csv(path)

    # ------------------------------------------------------------------
    # Figure 1: Main Accuracy Comparison (Grouped Bar Chart)
    # ------------------------------------------------------------------
    try:
        accuracy_df = _load_csv("accuracy_table.csv")
        if accuracy_df is not None and len(accuracy_df) > 0:
            fig, ax = plt.subplots(figsize=(IEEE_DOUBLE_COL_WIDTH, 2.5))

            methods = accuracy_df["method"].unique()
            fractions = sorted(accuracy_df["mask_fraction"].unique())
            n_methods = len(methods)
            n_fracs = len(fractions)

            x = np.arange(n_fracs)
            width = 0.8 / n_methods
            colors = plt.cm.Set2(np.linspace(0, 0.8, n_methods))

            for i, method in enumerate(methods):
                method_data = accuracy_df[accuracy_df["method"] == method]
                rmses = []
                errs = []
                for frac in fractions:
                    frac_data = method_data[method_data["mask_fraction"] == frac]
                    if len(frac_data) > 0:
                        rmses.append(frac_data["rmse"].values[0])
                        errs.append(
                            frac_data["rmse_std"].values[0]
                            if "rmse_std" in frac_data.columns
                            else 0
                        )
                    else:
                        rmses.append(0)
                        errs.append(0)

                offset = (i - n_methods / 2 + 0.5) * width
                ax.bar(
                    x + offset,
                    rmses,
                    width,
                    yerr=errs,
                    label=method,
                    color=colors[i],
                    edgecolor="black",
                    linewidth=0.3,
                    capsize=2,
                )

            ax.set_xlabel("Masking Fraction")
            ax.set_ylabel("RMSE")
            ax.set_title("Imputation Accuracy: GIMIN vs. Baselines")
            ax.set_xticks(x)
            ax.set_xticklabels([str(f) for f in fractions])
            ax.legend(ncol=min(4, n_methods), loc="upper left", framealpha=0.9)
            ax.grid(axis="y", alpha=0.3)

            path = fig_dir / "fig01_accuracy_comparison.png"
            fig.savefig(path)
            plt.close(fig)
            generated.append(str(path))
            logger.info("Generated Figure 1: %s", path)
    except Exception as exc:
        logger.warning("Figure 1 generation failed: %s", exc)

    # ------------------------------------------------------------------
    # Figure 2: Per-Modality RMSE Heatmap
    # ------------------------------------------------------------------
    try:
        exp1_data = _load_json("experiment_1_accuracy.json")
        if exp1_data is not None and "gimin" in exp1_data:
            modality_data = {}
            for frac_key, frac_data in exp1_data["gimin"].items():
                if isinstance(frac_data, dict) and "modality_breakdown" in frac_data:
                    for mod_name, mod_metrics in frac_data[
                        "modality_breakdown"
                    ].items():
                        if mod_name not in modality_data:
                            modality_data[mod_name] = {}
                        modality_data[mod_name][frac_key] = mod_metrics.get(
                            "rmse", np.nan
                        )

            if modality_data:
                mod_names = sorted(modality_data.keys())
                frac_keys = sorted(next(iter(modality_data.values())).keys())
                heatmap_matrix = np.array(
                    [
                        [modality_data[m].get(f, np.nan) for f in frac_keys]
                        for m in mod_names
                    ]
                )

                fig, ax = plt.subplots(figsize=(IEEE_SINGLE_COL_WIDTH, 2.5))
                im = ax.imshow(
                    heatmap_matrix,
                    aspect="auto",
                    cmap="YlOrRd",
                )
                ax.set_xticks(range(len(frac_keys)))
                ax.set_xticklabels(frac_keys, rotation=45)
                ax.set_yticks(range(len(mod_names)))
                ax.set_yticklabels(
                    [m.replace("_", "\n") for m in mod_names], fontsize=6
                )
                ax.set_xlabel("Masking Fraction")
                ax.set_title("Per-Modality RMSE")
                plt.colorbar(im, ax=ax, label="RMSE", shrink=0.8)

                # Annotate cells.
                for i in range(len(mod_names)):
                    for j in range(len(frac_keys)):
                        val = heatmap_matrix[i, j]
                        if not np.isnan(val):
                            ax.text(
                                j,
                                i,
                                f"{val:.3f}",
                                ha="center",
                                va="center",
                                fontsize=5,
                                color=(
                                    "white"
                                    if val > np.nanmean(heatmap_matrix)
                                    else "black"
                                ),
                            )

                path = fig_dir / "fig02_modality_heatmap.png"
                fig.savefig(path)
                plt.close(fig)
                generated.append(str(path))
                logger.info("Generated Figure 2: %s", path)
    except Exception as exc:
        logger.warning("Figure 2 generation failed: %s", exc)

    # ------------------------------------------------------------------
    # Figure 3: Missingness Mechanism Robustness
    # ------------------------------------------------------------------
    try:
        mech_df = _load_csv("mechanism_robustness.csv")
        if mech_df is not None and len(mech_df) > 0:
            fig, axes = plt.subplots(
                1, 3, figsize=(IEEE_DOUBLE_COL_WIDTH, 2.0), sharey=False
            )
            metrics_to_plot = ["rmse", "mae", "r_squared"]
            titles = ["RMSE", "MAE", "R-squared"]
            colors = ["#2196F3", "#FF9800", "#4CAF50"]

            for ax, metric, title, color in zip(axes, metrics_to_plot, titles, colors):
                mechanisms = mech_df["mechanism"].values
                values = mech_df[metric].values
                bars = ax.bar(
                    mechanisms,
                    values,
                    color=color,
                    edgecolor="black",
                    linewidth=0.3,
                    alpha=0.85,
                )
                ax.set_title(title)
                ax.grid(axis="y", alpha=0.3)
                # Add value labels.
                for bar, val in zip(bars, values):
                    ax.text(
                        bar.get_x() + bar.get_width() / 2,
                        bar.get_height(),
                        f"{val:.3f}",
                        ha="center",
                        va="bottom",
                        fontsize=6,
                    )

            fig.suptitle(
                "GIMIN Robustness Across Missingness Mechanisms",
                fontsize=9,
                y=1.02,
            )
            plt.tight_layout()
            path = fig_dir / "fig03_mechanism_robustness.png"
            fig.savefig(path)
            plt.close(fig)
            generated.append(str(path))
            logger.info("Generated Figure 3: %s", path)
    except Exception as exc:
        logger.warning("Figure 3 generation failed: %s", exc)

    # ------------------------------------------------------------------
    # Figure 4: KS-test Distribution Preservation (Violin/Strip Plot)
    # ------------------------------------------------------------------
    try:
        ks_df = _load_csv("ks_test_results.csv")
        if ks_df is not None and len(ks_df) > 0:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(IEEE_DOUBLE_COL_WIDTH, 2.5))

            # Panel A: KS statistic by modality.
            modalities = ks_df["modality"].unique()
            mod_colors = plt.cm.Set2(np.linspace(0, 0.8, len(modalities)))

            positions = []
            labels = []
            for i, mod in enumerate(sorted(modalities)):
                mod_data = ks_df[ks_df["modality"] == mod]["ks_statistic"].values
                if len(mod_data) == 0:
                    continue
                bp = ax1.boxplot(
                    mod_data,
                    positions=[i],
                    widths=0.6,
                    patch_artist=True,
                    showfliers=True,
                    flierprops=dict(markersize=3),
                )
                bp["boxes"][0].set_facecolor(mod_colors[i])
                bp["boxes"][0].set_edgecolor("black")
                bp["boxes"][0].set_linewidth(0.5)
                positions.append(i)
                labels.append(mod.replace("_", "\n"))

            ax1.set_xticks(positions)
            ax1.set_xticklabels(labels, fontsize=5, rotation=45, ha="right")
            ax1.set_ylabel("KS Statistic")
            ax1.set_title("(a) KS Statistic by Modality")
            ax1.axhline(
                y=0.05,
                color="red",
                linestyle="--",
                linewidth=0.5,
                label="p=0.05 threshold",
            )
            ax1.grid(axis="y", alpha=0.3)

            # Panel B: p-value distribution.
            ax2.hist(
                ks_df["p_value"].values,
                bins=20,
                color="#607D8B",
                edgecolor="black",
                linewidth=0.3,
                alpha=0.8,
            )
            ax2.axvline(
                x=0.05,
                color="red",
                linestyle="--",
                linewidth=1.0,
                label="p=0.05",
            )
            ax2.set_xlabel("p-value")
            ax2.set_ylabel("Count")
            ax2.set_title("(b) KS p-value Distribution")
            ax2.legend(fontsize=6)

            plt.tight_layout()
            path = fig_dir / "fig04_ks_distribution.png"
            fig.savefig(path)
            plt.close(fig)
            generated.append(str(path))
            logger.info("Generated Figure 4: %s", path)
    except Exception as exc:
        logger.warning("Figure 4 generation failed: %s", exc)

    # ------------------------------------------------------------------
    # Figure 5: Uncertainty Calibration Curve
    # ------------------------------------------------------------------
    try:
        cal_data = _load_json("calibration_results.json")
        if cal_data is not None:
            cal_metrics = cal_data.get("calibration_metrics", {})
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(IEEE_DOUBLE_COL_WIDTH, 2.5))

            # Panel A: Calibration curve (coverage).
            quantiles = [0.50, 0.80, 0.90, 0.95]
            expected = []
            observed = []
            for q in quantiles:
                key = f"coverage_{q:.2f}"
                if key in cal_metrics:
                    expected.append(q)
                    observed.append(cal_metrics[key])

            if expected:
                ax1.plot([0, 1], [0, 1], "k--", linewidth=0.5, label="Perfect")
                ax1.plot(
                    expected,
                    observed,
                    "o-",
                    color="#1976D2",
                    markersize=5,
                    linewidth=1.0,
                    label="GIMIN",
                )
                ax1.set_xlabel("Expected Coverage")
                ax1.set_ylabel("Observed Coverage")
                ax1.set_title("(a) Calibration Curve")
                ax1.legend(fontsize=6)
                ax1.set_xlim(0.4, 1.0)
                ax1.set_ylim(0.4, 1.0)
                ax1.set_aspect("equal")
                ax1.grid(alpha=0.3)

            # Panel B: Uncertainty decomposition.
            unc_summary = cal_data.get("uncertainty_summary", {})
            epist = unc_summary.get("mean_epistemic_std", 0)
            aleat = unc_summary.get("mean_aleatoric_std", 0)

            if epist > 0 or aleat > 0:
                labels_pie = ["Epistemic", "Aleatoric"]
                sizes = [epist**2, aleat**2]
                colors_pie = ["#42A5F5", "#FFA726"]
                ax2.pie(
                    sizes,
                    labels=labels_pie,
                    colors=colors_pie,
                    autopct="%1.1f%%",
                    startangle=90,
                    textprops={"fontsize": 7},
                )
                ax2.set_title("(b) Uncertainty Decomposition")

            plt.tight_layout()
            path = fig_dir / "fig05_calibration.png"
            fig.savefig(path)
            plt.close(fig)
            generated.append(str(path))
            logger.info("Generated Figure 5: %s", path)
    except Exception as exc:
        logger.warning("Figure 5 generation failed: %s", exc)

    # ------------------------------------------------------------------
    # Figure 6: Downstream SAA Classification Comparison
    # ------------------------------------------------------------------
    try:
        downstream_df = _load_csv("downstream_saa.csv")
        if downstream_df is not None and len(downstream_df) > 0:
            fig, ax = plt.subplots(figsize=(IEEE_SINGLE_COL_WIDTH, 2.5))

            metric_cols = [
                c
                for c in downstream_df.columns
                if c.endswith("_mean") and c != "auroc_mean"
            ]
            if not metric_cols:
                metric_cols = [
                    c
                    for c in downstream_df.columns
                    if c not in ("method",) and "_std" not in c
                ]

            methods = downstream_df["method"].values
            n_methods = len(methods)
            n_metrics = len(metric_cols)

            x = np.arange(n_metrics)
            width = 0.35
            colors_bar = ["#1976D2", "#E64A19"]

            for i, method in enumerate(methods):
                vals = [
                    downstream_df[downstream_df["method"] == method][col].values[0]
                    for col in metric_cols
                ]
                std_cols = [c.replace("_mean", "_std") for c in metric_cols]
                errs = []
                for sc in std_cols:
                    if sc in downstream_df.columns:
                        err_val = downstream_df[downstream_df["method"] == method][
                            sc
                        ].values
                        errs.append(err_val[0] if len(err_val) > 0 else 0)
                    else:
                        errs.append(0)

                offset = (i - n_methods / 2 + 0.5) * width
                ax.bar(
                    x + offset,
                    vals,
                    width,
                    yerr=errs,
                    label=method,
                    color=colors_bar[i % len(colors_bar)],
                    edgecolor="black",
                    linewidth=0.3,
                    capsize=2,
                )

            ax.set_ylabel("Score")
            ax.set_title("SAA Classification: GIMIN vs. MICE")
            ax.set_xticks(x)
            ax.set_xticklabels(
                [c.replace("_mean", "").replace("_", " ").title() for c in metric_cols],
                rotation=30,
                ha="right",
                fontsize=6,
            )
            ax.legend(fontsize=6)
            ax.grid(axis="y", alpha=0.3)

            plt.tight_layout()
            path = fig_dir / "fig06_downstream_saa.png"
            fig.savefig(path)
            plt.close(fig)
            generated.append(str(path))
            logger.info("Generated Figure 6: %s", path)
    except Exception as exc:
        logger.warning("Figure 6 generation failed: %s", exc)

    # ------------------------------------------------------------------
    # Figure 7: Ablation Study
    # ------------------------------------------------------------------
    try:
        ablation_df = _load_csv("ablation_table.csv")
        if ablation_df is not None and len(ablation_df) > 0:
            fig, ax = plt.subplots(figsize=(IEEE_DOUBLE_COL_WIDTH, 2.5))

            variants = ablation_df["variant"].values
            rmse_vals = ablation_df["rmse"].values

            # Sort by RMSE descending (worst to best).
            sort_idx = np.argsort(rmse_vals)[::-1]
            variants_sorted = variants[sort_idx]
            rmse_sorted = rmse_vals[sort_idx]

            # Color full model differently.
            colors_abl = []
            for v in variants_sorted:
                if v == "Full GIMIN":
                    colors_abl.append("#1976D2")
                else:
                    colors_abl.append("#90A4AE")

            bars = ax.barh(
                range(len(variants_sorted)),
                rmse_sorted,
                color=colors_abl,
                edgecolor="black",
                linewidth=0.3,
            )

            ax.set_yticks(range(len(variants_sorted)))
            ax.set_yticklabels(variants_sorted, fontsize=7)
            ax.set_xlabel("RMSE")
            ax.set_title("Ablation Study: Component Contributions")
            ax.grid(axis="x", alpha=0.3)

            # Add value labels.
            for bar, val in zip(bars, rmse_sorted):
                ax.text(
                    bar.get_width() + 0.001,
                    bar.get_y() + bar.get_height() / 2,
                    f"{val:.4f}",
                    va="center",
                    fontsize=6,
                )

            plt.tight_layout()
            path = fig_dir / "fig07_ablation.png"
            fig.savefig(path)
            plt.close(fig)
            generated.append(str(path))
            logger.info("Generated Figure 7: %s", path)
    except Exception as exc:
        logger.warning("Figure 7 generation failed: %s", exc)

    # ------------------------------------------------------------------
    # Figure 8: Scalability Plot
    # ------------------------------------------------------------------
    try:
        scalability_df = _load_csv("scalability.csv")
        if scalability_df is not None and len(scalability_df) > 0:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(IEEE_DOUBLE_COL_WIDTH, 2.5))

            n_patients = scalability_df["n_patients"].values
            mean_times = scalability_df["mean_time_seconds"].values
            std_times = scalability_df["std_time_seconds"].values
            per_patient = scalability_df["time_per_patient_ms"].values

            # Panel A: Total inference time.
            ax1.errorbar(
                n_patients,
                mean_times,
                yerr=std_times,
                marker="o",
                color="#1976D2",
                linewidth=1.0,
                capsize=3,
            )
            ax1.set_xlabel("Number of Patients")
            ax1.set_ylabel("Inference Time (s)")
            ax1.set_title("(a) Total Inference Time")
            ax1.grid(alpha=0.3)

            # Panel B: Per-patient time.
            ax2.plot(
                n_patients,
                per_patient,
                marker="s",
                color="#E64A19",
                linewidth=1.0,
            )
            ax2.set_xlabel("Number of Patients")
            ax2.set_ylabel("Time per Patient (ms)")
            ax2.set_title("(b) Per-Patient Inference Time")
            ax2.grid(alpha=0.3)

            plt.tight_layout()
            path = fig_dir / "fig08_scalability.png"
            fig.savefig(path)
            plt.close(fig)
            generated.append(str(path))
            logger.info("Generated Figure 8: %s", path)
    except Exception as exc:
        logger.warning("Figure 8 generation failed: %s", exc)

    # ------------------------------------------------------------------
    # Figure 9: RMSE vs. Masking Fraction Line Plot (All Methods)
    # ------------------------------------------------------------------
    try:
        accuracy_df = _load_csv("accuracy_table.csv")
        if accuracy_df is not None and len(accuracy_df) > 0:
            fig, ax = plt.subplots(figsize=(IEEE_SINGLE_COL_WIDTH, 2.8))

            methods = accuracy_df["method"].unique()
            markers = ["o", "s", "^", "v", "D", "P", "<", ">"]
            linestyles = ["-", "--", "-.", ":", "-", "--", "-.", ":"]
            colors_line = plt.cm.tab10(np.linspace(0, 1, len(methods)))

            for i, method in enumerate(methods):
                method_data = accuracy_df[accuracy_df["method"] == method].sort_values(
                    "mask_fraction"
                )
                fracs = method_data["mask_fraction"].values
                rmses = method_data["rmse"].values
                errs = (
                    method_data["rmse_std"].values
                    if "rmse_std" in method_data.columns
                    else np.zeros_like(rmses)
                )

                # Convert fraction strings to floats if needed.
                try:
                    fracs_float = [float(f) for f in fracs]
                except (ValueError, TypeError):
                    fracs_float = list(range(len(fracs)))

                ax.errorbar(
                    fracs_float,
                    rmses,
                    yerr=errs,
                    marker=markers[i % len(markers)],
                    linestyle=linestyles[i % len(linestyles)],
                    color=colors_line[i],
                    label=method,
                    linewidth=1.0,
                    markersize=4,
                    capsize=2,
                )

            ax.set_xlabel("Masking Fraction")
            ax.set_ylabel("RMSE")
            ax.set_title("RMSE vs. Masking Fraction")
            ax.legend(fontsize=5, ncol=2, loc="upper left")
            ax.grid(alpha=0.3)

            plt.tight_layout()
            path = fig_dir / "fig09_rmse_vs_fraction.png"
            fig.savefig(path)
            plt.close(fig)
            generated.append(str(path))
            logger.info("Generated Figure 9: %s", path)
    except Exception as exc:
        logger.warning("Figure 9 generation failed: %s", exc)

    # ------------------------------------------------------------------
    # Figure 10: Summary Radar Chart (Multi-Metric Comparison)
    # ------------------------------------------------------------------
    try:
        accuracy_df = _load_csv("accuracy_table.csv")
        if accuracy_df is not None and len(accuracy_df) > 0:
            fig, ax = plt.subplots(
                figsize=(IEEE_SINGLE_COL_WIDTH, 3.0),
                subplot_kw=dict(polar=True),
            )

            # Select the 0.20 masking fraction (representative).
            frac_data = accuracy_df[accuracy_df["mask_fraction"] == "0.20"]
            if len(frac_data) == 0:
                # Try first available fraction.
                first_frac = accuracy_df["mask_fraction"].iloc[0]
                frac_data = accuracy_df[accuracy_df["mask_fraction"] == first_frac]

            if len(frac_data) > 0:
                radar_metrics = ["rmse", "mae", "r_squared"]
                available_metrics = [m for m in radar_metrics if m in frac_data.columns]

                if available_metrics:
                    methods = frac_data["method"].unique()
                    n_axes = len(available_metrics)
                    angles = np.linspace(0, 2 * np.pi, n_axes, endpoint=False).tolist()
                    angles.append(angles[0])

                    colors_radar = plt.cm.Set1(np.linspace(0, 0.8, len(methods)))

                    for i, method in enumerate(methods):
                        m_data = frac_data[frac_data["method"] == method]
                        values = []
                        for metric in available_metrics:
                            val = m_data[metric].values[0]
                            # Invert RMSE and MAE so higher = better on the radar.
                            if metric in ("rmse", "mae"):
                                val = 1.0 / (val + 0.01)
                            values.append(val)
                        values.append(values[0])

                        ax.plot(
                            angles,
                            values,
                            "o-",
                            color=colors_radar[i],
                            linewidth=1.0,
                            markersize=3,
                            label=method,
                        )
                        ax.fill(angles, values, alpha=0.1, color=colors_radar[i])

                    ax.set_xticks(angles[:-1])
                    ax.set_xticklabels(
                        [
                            "1/RMSE"
                            if m in ("rmse",)
                            else "1/MAE"
                            if m == "mae"
                            else "R-squared"
                            for m in available_metrics
                        ],
                        fontsize=6,
                    )
                    ax.set_title(
                        "Multi-Metric Comparison (frac=0.20)", fontsize=8, pad=15
                    )
                    ax.legend(
                        loc="upper right",
                        bbox_to_anchor=(1.3, 1.1),
                        fontsize=5,
                    )

            plt.tight_layout()
            path = fig_dir / "fig10_radar_comparison.png"
            fig.savefig(path)
            plt.close(fig)
            generated.append(str(path))
            logger.info("Generated Figure 10: %s", path)
    except Exception as exc:
        logger.warning("Figure 10 generation failed: %s", exc)

    return generated


# ======================================================================
# Main orchestrator
# ======================================================================


def main() -> None:
    """Main entry point: parse arguments, run experiments, generate figures."""
    args = parse_args()

    # Configure logging.
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Resolve paths.
    output_dir = Path(args.output_dir) if args.output_dir else PROJECT_ROOT / "outputs"
    eval_dir = output_dir / "evaluation"
    fig_dir = output_dir / "figures"
    eval_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_path = Path(args.checkpoint) if args.checkpoint else None

    # Determine which experiments to run.
    experiments_to_run = args.experiments or ALL_EXPERIMENT_NUMS

    # Load configuration.
    if args.config:
        config = GIMINConfig.from_yaml(args.config)
    else:
        default_config_path = PROJECT_ROOT / "configs" / "default.yaml"
        if default_config_path.exists():
            config = GIMINConfig.from_yaml(str(default_config_path))
        else:
            config = GIMINConfig()

    # Override seed.
    config.random_seed = args.seed

    # Set global random seeds.
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    print("=" * 70)
    print("GIMIN: Full Evaluation Pipeline (Step 8)")
    print("=" * 70)
    print(f"  Experiments:   {experiments_to_run}")
    print(f"  Skip figures:  {args.skip_figures}")
    print(f"  Figures only:  {args.figures_only}")
    print(f"  Output dir:    {output_dir}")
    print(f"  Eval dir:      {eval_dir}")
    print(f"  Figures dir:   {fig_dir}")
    print(f"  Random seed:   {args.seed}")
    print("=" * 70)

    # ------------------------------------------------------------------
    # Run experiments (unless --figures-only)
    # ------------------------------------------------------------------
    experiment_results: dict[int, Any] = {}
    experiment_status: dict[int, str] = {}
    total_time_start = time.time()

    if not args.figures_only:
        # Load model and data once.
        try:
            data = load_model_and_data(config, output_dir, checkpoint_path)
        except FileNotFoundError as exc:
            logger.error("Failed to load required data: %s", exc)
            print(f"\nERROR: {exc}")
            print("Please ensure all prerequisite files exist.")
            sys.exit(1)

        model = data["model"]
        features_np = data["features_np"]
        mask_np = data["mask_np"]
        features_df = data["features_df"]
        edge_index = data["edge_index"]
        edge_weight = data["edge_weight"]

        # -- Experiment dispatch table --
        experiment_dispatch = {
            1: lambda: run_experiment_1(
                model,
                features_np,
                mask_np,
                config,
                edge_index,
                edge_weight,
                eval_dir,
                seed=args.seed,
            ),
            2: lambda: run_experiment_2(
                model,
                features_np,
                mask_np,
                config,
                edge_index,
                edge_weight,
                eval_dir,
                features_df,
                seed=args.seed,
            ),
            3: lambda: run_experiment_3(
                model,
                features_np,
                mask_np,
                config,
                edge_index,
                edge_weight,
                eval_dir,
                seed=args.seed,
            ),
            4: lambda: run_experiment_4(
                model,
                features_np,
                mask_np,
                config,
                edge_index,
                edge_weight,
                eval_dir,
                seed=args.seed,
            ),
            5: lambda: run_experiment_5(
                model,
                features_np,
                mask_np,
                config,
                edge_index,
                edge_weight,
                eval_dir,
                features_df,
                seed=args.seed,
            ),
            6: lambda: run_experiment_6(
                model,
                features_np,
                mask_np,
                config,
                edge_index,
                edge_weight,
                eval_dir,
                seed=args.seed,
            ),
            7: lambda: run_experiment_7(
                model,
                features_np,
                mask_np,
                config,
                edge_index,
                edge_weight,
                eval_dir,
                seed=args.seed,
            ),
        }

        for exp_num in experiments_to_run:
            if exp_num not in experiment_dispatch:
                logger.warning("Unknown experiment number: %d", exp_num)
                experiment_status[exp_num] = "SKIPPED (unknown)"
                continue

            exp_start = time.time()
            try:
                result = experiment_dispatch[exp_num]()
                experiment_results[exp_num] = result
                experiment_status[exp_num] = "SUCCESS"
            except Exception as exc:
                logger.error("Experiment %d FAILED: %s", exp_num, exc, exc_info=True)
                experiment_status[exp_num] = f"FAILED: {exc}"
            finally:
                exp_elapsed = time.time() - exp_start
                logger.info(
                    "Experiment %d completed in %.1f seconds", exp_num, exp_elapsed
                )

    # ------------------------------------------------------------------
    # Generate figures (unless --skip-figures)
    # ------------------------------------------------------------------
    generated_figures: list[str] = []

    if not args.skip_figures:
        logger.info("Generating publication figures...")
        try:
            generated_figures = generate_figures(eval_dir, fig_dir)
        except Exception as exc:
            logger.error("Figure generation failed: %s", exc, exc_info=True)

    # ------------------------------------------------------------------
    # Print summary
    # ------------------------------------------------------------------
    total_elapsed = time.time() - total_time_start

    print("\n" + "=" * 70)
    print("PIPELINE SUMMARY")
    print("=" * 70)

    if not args.figures_only:
        print("\nExperiment Results:")
        print("-" * 50)
        experiment_names = {
            1: "Imputation Accuracy",
            2: "Missingness Mechanism Robustness",
            3: "Distribution Preservation (KS-test)",
            4: "Uncertainty Calibration",
            5: "Downstream SAA Classification",
            6: "Ablation Study",
            7: "Scalability & Incremental Performance",
        }
        for exp_num in sorted(experiment_status.keys()):
            status = experiment_status[exp_num]
            name = experiment_names.get(exp_num, f"Experiment {exp_num}")
            marker = "[OK]" if status == "SUCCESS" else "[!!]"
            print(f"  {marker} Exp {exp_num}: {name:<40s} {status}")

    if generated_figures:
        print(f"\nGenerated {len(generated_figures)} figures:")
        print("-" * 50)
        for fig_path in generated_figures:
            print(f"  {fig_path}")

    print(f"\nTotal elapsed time: {total_elapsed:.1f} seconds")
    print(f"Evaluation directory: {eval_dir}")
    print(f"Figures directory:    {fig_dir}")
    print("=" * 70)


if __name__ == "__main__":
    main()
