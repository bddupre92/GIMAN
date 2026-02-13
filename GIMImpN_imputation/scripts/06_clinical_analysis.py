#!/usr/bin/env python3
"""Step 6: Clinical superiority analysis for GIMIN.

Demonstrates GIMIN's advantages over baseline imputation methods
for PPMI Parkinson's disease data through five analyses:

1. Cross-modal transfer -- GIMIN imputes CSF biomarkers from imaging
2. Uncertainty quantification -- MC dropout confidence intervals
3. Clinical constraint preservation -- NHY <-> Putamen SBR correlation
4. High-missingness robustness -- degradation curves at 50/70/90%
5. Per-patient clinical coherence profiles

Prerequisites:
    - outputs/checkpoints/gimin_best.pt  (from Step 4)
    - outputs/ppmi_full_cohort.parquet
    - outputs/missingness_mask.parquet
    - outputs/patient_graph.pt

Outputs:
    outputs/clinical_analysis/cross_modal_transfer.json
    outputs/clinical_analysis/uncertainty_analysis.json
    outputs/clinical_analysis/clinical_correlations.json
    outputs/clinical_analysis/robustness_curves.json
    outputs/clinical_analysis/patient_coherence.json
    outputs/clinical_analysis/figures/  (publication-ready plots)

Usage:
    python scripts/06_clinical_analysis.py [--config configs/default.yaml]
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Clinical superiority analysis for GIMIN."
    )
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    return parser.parse_args()


def _make_serializable(obj):
    """Recursively convert numpy types for JSON serialization."""
    if isinstance(obj, dict):
        return {k: _make_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_make_serializable(v) for v in obj]
    elif isinstance(obj, (np.floating, np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, (np.integer, np.int32, np.int64)):
        return int(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


# ======================================================================
# Analysis 1: Cross-Modal Transfer
# ======================================================================
def run_cross_modal_transfer(
    model,
    features_np,
    mask_np,
    config,
    edge_index,
    edge_weight,
    scaler,
    baselines,
    rng,
    logger,
):
    """Mask all CSF biomarkers for patients with imaging data.

    GIMIN can use graph neighbors' CSF values while baselines
    can only use within-patient features.
    """
    logger.info("=== Analysis 1: Cross-Modal Transfer ===")

    # CSF biomarker indices: 19-22 (ALPHA_SYNUCLEIN, TOTAL_TAU, ABETA42, PTAU181)
    csf_start, csf_end = 19, 23
    # Imaging indices: 7-18 (structural_imaging + spect_sbr)
    imaging_indices = list(range(7, 19))

    csf_names = ["ALPHA_SYNUCLEIN", "TOTAL_TAU", "ABETA42", "PTAU181"]

    # Find patients with both CSF and at least some imaging observed
    csf_observed = mask_np[:, csf_start:csf_end].sum(axis=1) >= 2
    imaging_observed = mask_np[:, imaging_indices].sum(axis=1) >= 4
    eligible = csf_observed & imaging_observed
    n_eligible = eligible.sum()
    logger.info("  %d patients with both CSF + imaging data", n_eligible)

    if n_eligible < 5:
        logger.warning("  Too few eligible patients; skipping cross-modal transfer.")
        return {"status": "skipped", "reason": "too_few_patients"}

    # Create corrupted mask: hide all CSF for eligible patients
    corrupted_mask = mask_np.copy()
    corrupted_mask[eligible, csf_start:csf_end] = 0.0

    target_mask = np.zeros_like(mask_np)
    target_mask[eligible, csf_start:csf_end] = 1.0

    from gimin.evaluation import metrics as M  # noqa: N812
    from gimin.evaluation.masked_experiment import MaskedValueExperiment

    results = {"n_eligible": int(n_eligible), "per_feature": {}, "overall": {}}

    # GIMIN imputation
    exp = MaskedValueExperiment(config)
    gimin_result = exp._impute_with_model(
        model,
        features_np,
        corrupted_mask,
        edge_index,
        edge_weight,
        scaler=scaler,
    )
    gimin_imputed = gimin_result["imputed"]

    # Per-CSF-feature metrics for GIMIN
    for fi, fname in enumerate(csf_names):
        col = csf_start + fi
        col_mask = target_mask[:, col : col + 1]
        if col_mask.sum() == 0:
            continue
        rmse = M.rmse(
            gimin_imputed[:, col : col + 1],
            features_np[:, col : col + 1],
            col_mask,
        )
        mae = M.mae(
            gimin_imputed[:, col : col + 1],
            features_np[:, col : col + 1],
            col_mask,
        )
        results["per_feature"][fname] = {"GIMIN": {"rmse": rmse, "mae": mae}}

    # Overall GIMIN
    csf_mask = target_mask[:, csf_start:csf_end]
    results["overall"]["GIMIN"] = {
        "rmse": M.rmse(
            gimin_imputed[:, csf_start:csf_end],
            features_np[:, csf_start:csf_end],
            csf_mask,
        ),
        "r_squared": M.r_squared(
            gimin_imputed[:, csf_start:csf_end],
            features_np[:, csf_start:csf_end],
            csf_mask,
        ),
    }

    # Baseline imputation
    for bname, bmodel in baselines.items():
        corrupted_features = features_np.copy()
        corrupted_features[~corrupted_mask.astype(bool)] = 0.0
        b_imputed = bmodel.fit_transform(corrupted_features, corrupted_mask)

        for fi, fname in enumerate(csf_names):
            col = csf_start + fi
            col_mask = target_mask[:, col : col + 1]
            if col_mask.sum() == 0:
                continue
            rmse = M.rmse(
                b_imputed[:, col : col + 1],
                features_np[:, col : col + 1],
                col_mask,
            )
            mae = M.mae(
                b_imputed[:, col : col + 1],
                features_np[:, col : col + 1],
                col_mask,
            )
            if fname not in results["per_feature"]:
                results["per_feature"][fname] = {}
            results["per_feature"][fname][bname] = {"rmse": rmse, "mae": mae}

        results["overall"][bname] = {
            "rmse": M.rmse(
                b_imputed[:, csf_start:csf_end],
                features_np[:, csf_start:csf_end],
                csf_mask,
            ),
            "r_squared": M.r_squared(
                b_imputed[:, csf_start:csf_end],
                features_np[:, csf_start:csf_end],
                csf_mask,
            ),
        }

    logger.info(
        "  Cross-modal transfer: GIMIN CSF RMSE=%.4f, R²=%.4f",
        results["overall"]["GIMIN"]["rmse"],
        results["overall"]["GIMIN"]["r_squared"],
    )
    return results


# ======================================================================
# Analysis 2: Uncertainty Quantification
# ======================================================================
def run_uncertainty_analysis(
    model,
    features_np,
    mask_np,
    config,
    edge_index,
    edge_weight,
    scaler,
    rng,
    logger,
    mc_samples=50,
):
    """MC dropout uncertainty quantification and calibration."""
    import torch

    logger.info("=== Analysis 2: Uncertainty Quantification ===")

    # Create evaluation mask: mask 20% of observed values
    from gimin.evaluation.masked_experiment import MaskedValueExperiment

    exp = MaskedValueExperiment(config)
    corrupted_mask, target_mask = exp._create_evaluation_mask(mask_np, 0.2, rng)

    device = next(model.parameters()).device
    feat_t = torch.from_numpy(features_np.astype(np.float32)).to(device)
    mask_t = torch.from_numpy(corrupted_mask.astype(np.float32)).to(device)

    if scaler is not None:
        feat_t = scaler.transform(feat_t, mask_t)
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

    # MC dropout: multiple forward passes
    model.train()  # Enable dropout
    mc_predictions = []

    with torch.no_grad():
        for _ in range(mc_samples):
            output = model(
                features=feat_t,
                mask=mask_t,
                edge_index=ei,
                edge_weight=ew,
                overlap_frac=of,
                modality_dims=config.modality_dims,
            )
            imputed_t = output["imputed"]
            if scaler is not None:
                imputed_t = scaler.inverse_transform(imputed_t)
            mc_predictions.append(imputed_t.cpu().numpy())

    model.eval()

    mc_stack = np.stack(mc_predictions, axis=0)  # (S, N, F)
    pred_mean = mc_stack.mean(axis=0)
    pred_std = mc_stack.std(axis=0)

    # Calibration analysis at held-out positions
    target_positions = target_mask.astype(bool)
    true_at_target = features_np[target_positions]
    mean_at_target = pred_mean[target_positions]
    std_at_target = pred_std[target_positions]

    # Compute coverage at various confidence levels
    confidence_levels = [0.50, 0.80, 0.90, 0.95]
    from scipy.stats import norm

    calibration = {}
    for cl in confidence_levels:
        z = norm.ppf(0.5 + cl / 2)
        lower = mean_at_target - z * std_at_target
        upper = mean_at_target + z * std_at_target
        coverage = ((true_at_target >= lower) & (true_at_target <= upper)).mean()
        calibration[f"{cl:.2f}"] = {
            "expected": cl,
            "observed": float(coverage),
        }
        logger.info("  CI %.0f%%: expected=%.2f, observed=%.4f", cl * 100, cl, coverage)

    # Per-modality uncertainty
    modality_uncertainty = {}
    col_start = 0
    for mod in config.modalities:
        num_features = len(mod.features)
        col_end = col_start + num_features
        mod_target = target_mask[:, col_start:col_end]
        if mod_target.sum() > 0:
            mod_std = pred_std[:, col_start:col_end]
            modality_uncertainty[mod.name] = {
                "mean_std": float(mod_std[mod_target.astype(bool)].mean()),
                "median_std": float(np.median(mod_std[mod_target.astype(bool)])),
            }
        col_start = col_end

    results = {
        "mc_samples": mc_samples,
        "n_target_positions": int(target_positions.sum()),
        "calibration": calibration,
        "modality_uncertainty": modality_uncertainty,
        "overall_mean_std": float(std_at_target.mean()),
    }

    return results


# ======================================================================
# Analysis 3: Clinical Constraint Preservation
# ======================================================================
def run_clinical_correlations(
    model,
    features_np,
    mask_np,
    config,
    edge_index,
    edge_weight,
    scaler,
    baselines,
    rng,
    logger,
):
    """Check known PD clinical correlations after full imputation."""
    logger.info("=== Analysis 3: Clinical Constraint Preservation ===")

    # Known PD correlation pairs (feature_idx_a, feature_idx_b, expected_direction)
    # Feature indices for 33-feature vector (after dropping 6 zero-var cols):
    #  0:SEX 1:AGE 2:NP3TOT 3:NHY 4:PIGD 5:TREMOR 6:MCATOT
    #  7:CAUDATE_L_VOL 8:CAUDATE_R_VOL 9:PUTAMEN_L_VOL 10:PUTAMEN_R_VOL
    # 11:HIPPOCAMPUS_L_VOL 12:HIPPOCAMPUS_R_VOL
    # 13:CAUDATE_L_SBR 14:CAUDATE_R_SBR 15:PUTAMEN_L_SBR 16:PUTAMEN_R_SBR
    # 17:CAUDATE_ASYM 18:PUTAMEN_ASYM
    # 19:ALPHA_SYN 20:TOTAL_TAU 21:ABETA42 22:PTAU181
    # 23:UPSIT 24:RBD 25:SCOPA 26:ESS
    # 27-32: cortical thickness
    correlation_pairs = [
        (3, 15, "negative", "NHY vs PUTAMEN_L_SBR"),  # NHY vs PUTAMEN_L_SBR
        (7, 13, "positive", "CAUDATE_L_VOL vs CAUDATE_L_SBR"),  # Structure-function
        (2, 3, "positive", "NP3TOT vs NHY"),  # Motor severity
        (6, 1, "negative", "MCATOT vs AGE_AT_VISIT"),  # Cognition-age
    ]

    from gimin.evaluation.masked_experiment import MaskedValueExperiment

    exp = MaskedValueExperiment(config)

    # Impute with GIMIN (full dataset, no artificial masking)
    gimin_result = exp._impute_with_model(
        model,
        features_np,
        mask_np,
        edge_index,
        edge_weight,
        scaler=scaler,
    )
    gimin_imputed = gimin_result["imputed"]

    # Impute with baselines
    baseline_imputed = {}
    for bname, bmodel in baselines.items():
        corrupted_features = features_np.copy()
        corrupted_features[~mask_np.astype(bool)] = 0.0
        baseline_imputed[bname] = bmodel.fit_transform(corrupted_features, mask_np)

    # Compute ground-truth correlations
    # (using only patients with both features observed)
    results = {"pairs": []}

    for idx_a, idx_b, expected_dir, pair_name in correlation_pairs:
        both_observed = mask_np[:, idx_a].astype(bool) & mask_np[:, idx_b].astype(bool)
        n_observed = both_observed.sum()

        pair_result = {
            "pair": pair_name,
            "idx_a": idx_a,
            "idx_b": idx_b,
            "expected_direction": expected_dir,
            "n_both_observed": int(n_observed),
            "correlations": {},
        }

        if n_observed >= 10:
            gt_corr = np.corrcoef(
                features_np[both_observed, idx_a],
                features_np[both_observed, idx_b],
            )[0, 1]
            pair_result["ground_truth_correlation"] = float(gt_corr)
        else:
            pair_result["ground_truth_correlation"] = None

        # GIMIN correlation (using all patients)
        gimin_corr = np.corrcoef(gimin_imputed[:, idx_a], gimin_imputed[:, idx_b])[0, 1]
        pair_result["correlations"]["GIMIN"] = float(gimin_corr)

        # Baseline correlations
        for bname, b_imp in baseline_imputed.items():
            b_corr = np.corrcoef(b_imp[:, idx_a], b_imp[:, idx_b])[0, 1]
            pair_result["correlations"][bname] = float(b_corr)

        # Compute preservation ratio
        if pair_result["ground_truth_correlation"] is not None:
            gt_r = pair_result["ground_truth_correlation"]
            for method, method_r in pair_result["correlations"].items():
                if abs(gt_r) > 0.01:
                    ratio = method_r / gt_r
                else:
                    ratio = float("nan")
                pair_result.setdefault("preservation_ratio", {})[method] = float(ratio)

        results["pairs"].append(pair_result)

        logger.info(
            "  %s: GT=%.3f, GIMIN=%.3f",
            pair_name,
            pair_result.get("ground_truth_correlation", 0) or 0,
            gimin_corr,
        )

    return results


# ======================================================================
# Analysis 4: High-Missingness Robustness
# ======================================================================
def run_robustness_curves(
    model,
    features_np,
    mask_np,
    config,
    edge_index,
    edge_weight,
    scaler,
    baselines,
    seed,
    logger,
):
    """Extend masked experiment to extreme missingness fractions."""
    logger.info("=== Analysis 4: High-Missingness Robustness ===")

    from gimin.evaluation.masked_experiment import MaskedValueExperiment

    fractions = [0.1, 0.2, 0.3, 0.5, 0.7, 0.9]
    num_runs = 5  # Fewer runs for speed at extreme fractions

    experiment = MaskedValueExperiment(config)
    results = experiment.run(
        model=model,
        features=features_np,
        mask=mask_np,
        mask_fractions=fractions,
        num_runs=num_runs,
        baselines=baselines,
        edge_index=edge_index,
        edge_weight=edge_weight,
        random_seed=seed,
        scaler=scaler,
    )

    # Extract degradation curves
    curves = {}
    methods = [k for k in results if k not in ("summary", "metadata")]
    for method in methods:
        method_data = results[method]
        curve = []
        for frac in fractions:
            frac_key = f"{frac:.2f}"
            frac_data = method_data.get(frac_key, {})
            mean_metrics = frac_data.get("mean", {})
            curve.append(
                {
                    "fraction": frac,
                    "rmse": mean_metrics.get("rmse", float("nan")),
                    "r_squared": mean_metrics.get("r_squared", float("nan")),
                }
            )
        curves[method] = curve

    logger.info("  Robustness curves computed for %d methods", len(curves))
    for method, curve in curves.items():
        for pt in curve:
            if pt["fraction"] in [0.1, 0.5, 0.9]:
                logger.info(
                    "    %s @ %.0f%%: RMSE=%.2f, R²=%.4f",
                    method,
                    pt["fraction"] * 100,
                    pt["rmse"],
                    pt["r_squared"],
                )

    return {"fractions": fractions, "curves": curves}


# ======================================================================
# Analysis 5: Per-Patient Clinical Coherence
# ======================================================================
def run_patient_coherence(
    model,
    features_np,
    mask_np,
    config,
    edge_index,
    edge_weight,
    scaler,
    baselines,
    logger,
):
    """Check clinical coherence rules on imputed patient profiles."""
    logger.info("=== Analysis 5: Per-Patient Clinical Coherence ===")

    from gimin.evaluation.masked_experiment import MaskedValueExperiment

    exp = MaskedValueExperiment(config)

    # GIMIN imputation
    gimin_result = exp._impute_with_model(
        model,
        features_np,
        mask_np,
        edge_index,
        edge_weight,
        scaler=scaler,
    )
    gimin_imputed = gimin_result["imputed"]

    # Mean baseline imputation
    corrupted_features = features_np.copy()
    corrupted_features[~mask_np.astype(bool)] = 0.0

    baseline_imputed = {}
    for bname, bmodel in baselines.items():
        baseline_imputed[bname] = bmodel.fit_transform(corrupted_features, mask_np)

    # Clinical coherence rules (on clinical-scale values)
    # 33-feature indices: NHY=3, PUTAMEN_L_SBR=15, MCATOT=6,
    # HIPPOCAMPUS_L_VOL=11, NP3TOT=2, TREMOR_SCORE=5
    rules = [
        {
            "name": "Advanced PD implies low putamen SBR",
            "condition_idx": 3,  # NHY
            "condition_fn": lambda v: v >= 3.0,
            "check_idx": 15,  # PUTAMEN_L_SBR
            "check_fn": lambda v: v < 1.5,
        },
        {
            "name": "Normal cognition implies preserved hippocampus",
            "condition_idx": 6,  # MCATOT
            "condition_fn": lambda v: v >= 26.0,
            "check_idx": 11,  # HIPPOCAMPUS_L_VOL
            "check_fn": lambda v: v > 2000.0,
        },
        {
            "name": "Tremor present implies motor deficit",
            "condition_idx": 5,  # TREMOR_SCORE
            "condition_fn": lambda v: v > 0.5,
            "check_idx": 2,  # NP3TOT
            "check_fn": lambda v: v > 0.5,
        },
    ]

    # Select patients with moderate missingness (30-70% missing)
    per_patient_miss = 1.0 - mask_np.mean(axis=1)
    moderate_miss = (per_patient_miss >= 0.3) & (per_patient_miss <= 0.7)
    candidate_indices = np.where(moderate_miss)[0]
    logger.info("  %d patients with 30-70%% missingness", len(candidate_indices))

    results = {"rules": [], "n_candidates": int(len(candidate_indices))}

    for rule in rules:
        rule_result = {
            "name": rule["name"],
            "violations": {},
            "n_applicable": 0,
        }

        for method_name, imputed in [("GIMIN", gimin_imputed)] + list(
            baseline_imputed.items()
        ):
            violations = 0
            applicable = 0

            for pi in candidate_indices:
                cond_val = imputed[pi, rule["condition_idx"]]
                if rule["condition_fn"](cond_val):
                    applicable += 1
                    check_val = imputed[pi, rule["check_idx"]]
                    if not rule["check_fn"](check_val):
                        violations += 1

            if method_name == "GIMIN":
                rule_result["n_applicable"] = applicable

            rule_result["violations"][method_name] = {
                "count": violations,
                "rate": violations / max(applicable, 1),
            }

        results["rules"].append(rule_result)
        logger.info(
            "  Rule '%s': applicable=%d, GIMIN violations=%d",
            rule["name"],
            rule_result["n_applicable"],
            rule_result["violations"]["GIMIN"]["count"],
        )

    return results


# ======================================================================
# Plotting
# ======================================================================
def create_figures(analysis_results, fig_dir, logger):
    """Create publication-ready figures from analysis results."""
    fig_dir.mkdir(parents=True, exist_ok=True)

    # Figure 1: Cross-Modal Transfer Bar Chart
    if (
        "cross_modal" in analysis_results
        and analysis_results["cross_modal"].get("status") != "skipped"
    ):
        cm = analysis_results["cross_modal"]
        fig, ax = plt.subplots(figsize=(8, 5))
        methods = list(cm["overall"].keys())
        rmses = [cm["overall"][m]["rmse"] for m in methods]
        colors = ["#2196F3" if m == "GIMIN" else "#9E9E9E" for m in methods]
        ax.bar(methods, rmses, color=colors)
        ax.set_ylabel("RMSE (CSF Biomarkers)")
        ax.set_title("Cross-Modal Transfer: CSF Imputation from Imaging Data")
        plt.tight_layout()
        plt.savefig(fig_dir / "cross_modal_transfer.png", dpi=150)
        plt.close()
        logger.info("  Saved cross_modal_transfer.png")

    # Figure 2: Calibration Diagram
    if "uncertainty" in analysis_results:
        unc = analysis_results["uncertainty"]
        cal = unc["calibration"]
        fig, ax = plt.subplots(figsize=(6, 6))
        expected = [cal[k]["expected"] for k in sorted(cal.keys())]
        observed = [cal[k]["observed"] for k in sorted(cal.keys())]
        ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Perfect calibration")
        ax.plot(expected, observed, "o-", color="#2196F3", label="GIMIN MC Dropout")
        ax.set_xlabel("Expected Coverage")
        ax.set_ylabel("Observed Coverage")
        ax.set_title("Uncertainty Calibration Diagram")
        ax.legend()
        ax.set_xlim(0.4, 1.0)
        ax.set_ylim(0.4, 1.0)
        plt.tight_layout()
        plt.savefig(fig_dir / "calibration_diagram.png", dpi=150)
        plt.close()
        logger.info("  Saved calibration_diagram.png")

    # Figure 3: Robustness Curves
    if "robustness" in analysis_results:
        rob = analysis_results["robustness"]
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        for method, curve in rob["curves"].items():
            fracs = [pt["fraction"] for pt in curve]
            rmses = [pt["rmse"] for pt in curve]
            r2s = [pt["r_squared"] for pt in curve]
            style = "-o" if method == "gimin" else "--s"
            lw = 2.5 if method == "gimin" else 1.5
            label = method.upper() if method == "gimin" else method
            ax1.plot(fracs, rmses, style, label=label, linewidth=lw)
            ax2.plot(fracs, r2s, style, label=label, linewidth=lw)

        ax1.set_xlabel("Mask Fraction")
        ax1.set_ylabel("RMSE")
        ax1.set_title("Imputation Accuracy vs Missingness")
        ax1.legend()

        ax2.set_xlabel("Mask Fraction")
        ax2.set_ylabel("R²")
        ax2.set_title("R² vs Missingness")
        ax2.legend()

        plt.tight_layout()
        plt.savefig(fig_dir / "robustness_curves.png", dpi=150)
        plt.close()
        logger.info("  Saved robustness_curves.png")

    # Figure 4: Clinical Correlation Preservation
    if "correlations" in analysis_results:
        corr = analysis_results["correlations"]
        fig, ax = plt.subplots(figsize=(10, 6))

        pair_names = [p["pair"] for p in corr["pairs"]]
        methods = set()
        for p in corr["pairs"]:
            methods.update(p.get("preservation_ratio", {}).keys())
        methods = sorted(methods)

        x = np.arange(len(pair_names))
        width = 0.15
        for i, method in enumerate(methods):
            ratios = []
            for p in corr["pairs"]:
                pr = p.get("preservation_ratio", {})
                ratios.append(pr.get(method, float("nan")))
            offset = (i - len(methods) / 2 + 0.5) * width
            color = "#2196F3" if method == "GIMIN" else None
            ax.bar(x + offset, ratios, width, label=method, color=color)

        ax.axhline(
            y=1.0, color="k", linestyle="--", alpha=0.5, label="Perfect (ratio=1)"
        )
        ax.set_xticks(x)
        ax.set_xticklabels(pair_names, rotation=15, ha="right", fontsize=9)
        ax.set_ylabel("Correlation Preservation Ratio")
        ax.set_title("Clinical Correlation Preservation After Imputation")
        ax.legend(fontsize=8)
        plt.tight_layout()
        plt.savefig(fig_dir / "correlation_preservation.png", dpi=150)
        plt.close()
        logger.info("  Saved correlation_preservation.png")


# ======================================================================
# Main
# ======================================================================
def main() -> None:
    args = parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    logger = logging.getLogger("gimin.clinical_analysis")

    import pandas as pd
    import torch

    from gimin.config import GIMINConfig
    from gimin.data.scaler import ModalityAwareScaler

    # Load config
    if args.config:
        config = GIMINConfig.from_yaml(args.config)
    else:
        config = GIMINConfig()

    output_dir = Path(args.output_dir) if args.output_dir else PROJECT_ROOT / "outputs"
    analysis_dir = output_dir / "clinical_analysis"
    fig_dir = analysis_dir / "figures"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("GIMIN: Clinical Superiority Analysis (Step 6)")
    print("=" * 70)

    # Load data
    feat_path = output_dir / "ppmi_full_cohort.parquet"
    mask_path = output_dir / "missingness_mask.parquet"
    graph_path = output_dir / "patient_graph.pt"

    features_df = pd.read_parquet(feat_path)
    mask_df = pd.read_parquet(mask_path)

    # Filter to only the features defined in config (drops zero-variance cols)
    keep_cols = config.all_feature_names
    dropped = [c for c in features_df.columns if c not in keep_cols]
    if dropped:
        logger.info(
            "Dropping %d zero-variance/unused columns: %s", len(dropped), dropped
        )
    features_df = features_df[keep_cols]
    mask_df = mask_df[keep_cols]

    # Filter to eligible patients
    graph_data = (
        torch.load(graph_path, weights_only=False) if graph_path.exists() else None
    )
    eligible_idx = graph_data.get("eligible_indices") if graph_data else None
    if eligible_idx is not None:
        features_df = features_df.iloc[eligible_idx].reset_index(drop=True)
        mask_df = mask_df.iloc[eligible_idx].reset_index(drop=True)

    features_np = np.nan_to_num(features_df.values.astype(np.float32), nan=0.0)
    mask_np = mask_df.values.astype(np.float32)

    logger.info("Dataset: %d patients, %d features", *features_np.shape)

    # Load model
    from gimin.model.gimin_core import GIMIN

    model = GIMIN(
        modality_dims=config.modality_dims,
        embed_dim=config.model.embed_dim,
        num_gnn_layers=config.model.num_gnn_layers,
        num_heads=config.model.num_heads,
        mc_dropout=config.model.mc_dropout_rate,
        binary_feature_indices=getattr(config, "binary_feature_indices", None),
    )

    ckpt_path = (
        Path(args.checkpoint)
        if args.checkpoint
        else output_dir / "checkpoints" / "gimin_best.pt"
    )
    scaler = None
    if ckpt_path.exists():
        checkpoint = torch.load(ckpt_path, weights_only=False)
        if "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
        else:
            model.load_state_dict(checkpoint)
        model.eval()
        logger.info("Loaded model from %s", ckpt_path)

        if "scaler_state_dict" in checkpoint:
            scaler = ModalityAwareScaler()
            scaler.load_state_dict(checkpoint["scaler_state_dict"])
            logger.info("Loaded scaler from checkpoint.")
    else:
        logger.error("No checkpoint found at %s", ckpt_path)
        sys.exit(1)

    # Load graph
    edge_index = graph_data.get("edge_index") if graph_data else None
    edge_weight = graph_data.get("edge_weight") if graph_data else None

    # Create baselines
    from gimin.evaluation.baselines import (
        KNNBaseline,
        MeanBaseline,
        MICEBaseline,
    )

    baselines = {
        "MICE": MICEBaseline(),
        "KNN": KNNBaseline(),
        "Mean": MeanBaseline(),
    }

    rng = np.random.default_rng(args.seed)

    # ---- Run all analyses ----
    all_results = {}
    t_start = time.time()

    # Analysis 1: Cross-Modal Transfer
    t0 = time.time()
    all_results["cross_modal"] = run_cross_modal_transfer(
        model,
        features_np,
        mask_np,
        config,
        edge_index,
        edge_weight,
        scaler,
        baselines,
        rng,
        logger,
    )
    logger.info("  Analysis 1 completed in %.1fs", time.time() - t0)

    # Analysis 2: Uncertainty Quantification
    t0 = time.time()
    all_results["uncertainty"] = run_uncertainty_analysis(
        model,
        features_np,
        mask_np,
        config,
        edge_index,
        edge_weight,
        scaler,
        rng,
        logger,
        mc_samples=30,
    )
    logger.info("  Analysis 2 completed in %.1fs", time.time() - t0)

    # Analysis 3: Clinical Correlations
    t0 = time.time()
    all_results["correlations"] = run_clinical_correlations(
        model,
        features_np,
        mask_np,
        config,
        edge_index,
        edge_weight,
        scaler,
        baselines,
        rng,
        logger,
    )
    logger.info("  Analysis 3 completed in %.1fs", time.time() - t0)

    # Analysis 4: Robustness Curves
    t0 = time.time()
    all_results["robustness"] = run_robustness_curves(
        model,
        features_np,
        mask_np,
        config,
        edge_index,
        edge_weight,
        scaler,
        baselines,
        args.seed,
        logger,
    )
    logger.info("  Analysis 4 completed in %.1fs", time.time() - t0)

    # Analysis 5: Patient Coherence
    t0 = time.time()
    all_results["coherence"] = run_patient_coherence(
        model,
        features_np,
        mask_np,
        config,
        edge_index,
        edge_weight,
        scaler,
        baselines,
        logger,
    )
    logger.info("  Analysis 5 completed in %.1fs", time.time() - t0)

    total_time = time.time() - t_start
    logger.info("All analyses completed in %.1fs", total_time)

    # ---- Create figures ----
    logger.info("Creating figures...")
    create_figures(all_results, fig_dir, logger)

    # ---- Save results ----
    for name, data in all_results.items():
        result_path = analysis_dir / f"{name}.json"
        with open(result_path, "w") as f:
            json.dump(_make_serializable(data), f, indent=2)
        logger.info("Saved %s", result_path)

    # Combined summary
    summary_path = analysis_dir / "clinical_analysis_summary.json"
    with open(summary_path, "w") as f:
        json.dump(_make_serializable(all_results), f, indent=2)

    print(f"\n{'=' * 70}")
    print("Clinical Analysis Complete")
    print(f"{'=' * 70}")
    print(f"  Total time:    {total_time:.1f}s")
    print(f"  Output dir:    {analysis_dir}")
    print(f"  Figures dir:   {fig_dir}")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
