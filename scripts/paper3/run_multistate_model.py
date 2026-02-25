#!/usr/bin/env python3
"""
Step 4: Fit Multi-State Markov Model for NSD-ISS Stage Transitions.

Fits both a homogeneous (no covariates) and a covariate-adjusted continuous-time
Markov chain to the longitudinal NSD-ISS staging data.

Usage:
    python scripts/paper3/run_multistate_model.py [--no-bootstrap] [--nsd-only]

Outputs:
    outputs/paper3_markov/
        markov_results.json       — Full results (Q, sojourn, probs, HRs, CIs)
        intensity_matrix_Q.csv    — Intensity matrix
        transition_prob_*.csv     — P(t) at 1yr, 2yr, 5yr, 10yr
        trajectory_predictions.csv — Stage occupancy predictions
        expected_transition_times.csv — First-passage time estimates
        covariate_hazard_ratios.csv — HR for each covariate x transition
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

# Project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from giman_pipeline.paper3.multistate_markov import (
    CORE_TRANSITIONS,
    STAGE_LABELS,
    STAGE_TO_IDX,
    bootstrap_ci,
    build_allowed_transitions,
    compute_expected_transition_times,
    fit_homogeneous,
    fit_with_covariates,
    format_Q_matrix,
    format_transition_probs,
    predict_trajectory,
    save_results,
)


DATA_DIR = PROJECT_ROOT / "data"
FEATURES_PATH = DATA_DIR / "07_paper3_features" / "longitudinal_features.csv"
COHORT_PATH = DATA_DIR / "06_longitudinal_staging" / "cohort_summary.json"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "paper3_markov"


def load_data(nsd_only: bool = False) -> tuple[pd.DataFrame, dict]:
    """Load longitudinal features and cohort summary."""
    print("Loading data...")
    features = pd.read_csv(FEATURES_PATH, low_memory=False)
    print(f"  Loaded {len(features)} observations from "
          f"{features['PATNO'].nunique()} patients")

    with open(COHORT_PATH) as f:
        cohort = json.load(f)

    if nsd_only:
        # Restrict to NSD+ patients (stages 1-6 at baseline)
        baseline = features[features["months_from_baseline"] == 0.0]
        nsd_patients = baseline[
            baseline["nsd_stage"].isin(["1", "2B", "3", "4", "5", "6"])
        ]["PATNO"].unique()
        features = features[features["PATNO"].isin(nsd_patients)]
        print(f"  NSD+ filter: {len(features)} observations from "
              f"{features['PATNO'].nunique()} patients")

    stage_dist = features["nsd_stage"].value_counts()
    print(f"  Stage distribution:\n{stage_dist.to_string()}")

    return features, cohort


def run_homogeneous_model(features: pd.DataFrame, cohort: dict) -> dict:
    """Fit and evaluate the homogeneous (no covariates) CTMC."""
    print("\n" + "=" * 70)
    print("MODEL 1: Homogeneous CTMC (no covariates)")
    print("=" * 70)

    # Build allowed transitions from empirical data
    trans_matrix = cohort.get("transition_matrix", None)
    allowed = build_allowed_transitions(trans_matrix, min_count=5)

    t0 = time.time()
    result = fit_homogeneous(
        features, allowed=allowed, transition_matrix=trans_matrix, verbose=True
    )
    elapsed = time.time() - t0
    print(f"\nFit time: {elapsed:.1f}s")

    # Print results
    print(f"\n{format_Q_matrix(result.Q)}")

    print("\nMean Sojourn Times (years in each stage before any transition):")
    for stage, t in sorted(result.sojourn_times.items(),
                           key=lambda x: STAGE_TO_IDX.get(x[0], 99)):
        if np.isfinite(t):
            print(f"  Stage {stage}: {t:.2f} years")
        else:
            print(f"  Stage {stage}: absorbing (no transitions out)")

    for horizon, P in result.transition_probs.items():
        print(f"\n{format_transition_probs(P, horizon)}")

    # Expected first-passage times for key transitions
    print("\nExpected First-Passage Times:")
    key_transitions = [("2B", "3"), ("3", "4"), ("4", "5"), ("2B", "4")]
    fpt_results = []
    for from_s, to_s in tqdm(key_transitions, desc="Computing first-passage times",
                              unit="transition"):
        fpt = compute_expected_transition_times(result.Q, from_s, to_s)
        fpt_results.append(fpt)
        tqdm.write(f"  {from_s} -> {to_s}: median {fpt['median_years']:.2f}yr, "
                   f"mean {fpt['mean_years']:.2f}yr, "
                   f"P(5yr)={fpt['prob_at_5yr']:.3f}, "
                   f"P(10yr)={fpt['prob_at_10yr']:.3f}")

    # Compare to Simuni reference
    simuni_ref = cohort.get("simuni_2025_reference", {})
    print("\nComparison to Simuni et al. (2025) KM medians:")
    comparisons = [
        ("2B", "3", simuni_ref.get("2B_to_3_median_years")),
        ("3", "4", simuni_ref.get("3_to_4_median_years")),
        ("4", "5", simuni_ref.get("4_to_5_median_years")),
    ]
    for from_s, to_s, simuni_val in tqdm(comparisons, desc="Simuni comparison",
                                          unit="transition"):
        fpt = compute_expected_transition_times(result.Q, from_s, to_s)
        if simuni_val:
            tqdm.write(f"  {from_s}->{to_s}: Markov median={fpt['median_years']:.2f}yr, "
                       f"Simuni KM={simuni_val:.2f}yr")

    return {
        "result": result,
        "fpt_results": fpt_results,
    }


def run_covariate_model(features: pd.DataFrame, cohort: dict) -> dict:
    """Fit the proportional-intensities covariate model.

    Uses 6 core transitions x 4 covariates = 30 params (vs 91 with full model)
    to keep optimization tractable with finite-difference gradients.
    """
    print("\n" + "=" * 70)
    print("MODEL 2: Covariate CTMC (proportional intensities)")
    print("=" * 70)

    # Use only 4 key covariates to keep parameter count manageable
    # 6 transitions x 4 covariates = 24 betas + 6 base = 30 total params
    covariate_candidates = [
        ("age_at_baseline", "Age at baseline"),
        ("sex", "Sex (1=Male)"),
        ("gba_carrier", "GBA carrier"),
        ("lrrk2_carrier", "LRRK2 carrier"),
    ]

    available_covariates = []
    for col, desc in covariate_candidates:
        if col in features.columns:
            coverage = features[col].notna().mean()
            if coverage >= 0.80:
                available_covariates.append((col, desc))
                print(f"  {col} ({desc}): {coverage:.1%} coverage")
            else:
                print(f"  {col} ({desc}): SKIPPED ({coverage:.1%} coverage < 80%)")

    covariate_names = [c for c, _ in available_covariates]
    if not covariate_names:
        print("  No covariates with sufficient coverage. Skipping.")
        return {}

    df_cov = features.dropna(subset=covariate_names).copy()
    print(f"\n  After dropping missing covariates: {len(df_cov)} observations "
          f"from {df_cov['PATNO'].nunique()} patients")

    # Use 6 core transitions only (3 forward + 3 backward)
    allowed = list(CORE_TRANSITIONS)
    print(f"  Using {len(allowed)} core transitions (3 forward + 3 backward)")

    t0 = time.time()
    result = fit_with_covariates(
        df_cov,
        covariate_names=covariate_names,
        allowed=allowed,
        max_iter=500,
        verbose=True,
    )
    elapsed = time.time() - t0
    print(f"\nFit time: {elapsed:.1f}s")

    print(f"\n{format_Q_matrix(result.Q)}")

    print("\nHazard Ratios (per unit increase in covariate):")
    hr_rows = []
    for trans_label, hr_dict in result.covariate_hazard_ratios.items():
        for cov_name, hr_val in hr_dict.items():
            desc = dict(available_covariates).get(cov_name, cov_name)
            sig = "*" if abs(np.log(hr_val)) > 0.1 else ""
            print(f"  {trans_label}: {desc} HR={hr_val:.4f}{sig}")
            hr_rows.append({
                "transition": trans_label,
                "covariate": cov_name,
                "description": desc,
                "hazard_ratio": hr_val,
                "log_hr": np.log(hr_val),
            })

    return {
        "result": result,
        "hr_table": pd.DataFrame(hr_rows),
        "covariate_names": covariate_names,
    }


def run_bootstrap(features: pd.DataFrame, allowed: list, n_bootstrap: int = 200):
    """Run bootstrap confidence intervals for the homogeneous model."""
    print("\n" + "=" * 70)
    print(f"BOOTSTRAP: {n_bootstrap} resamples for 95% CIs")
    print("=" * 70)

    t0 = time.time()
    ci = bootstrap_ci(features, allowed, n_bootstrap=n_bootstrap, verbose=True)
    elapsed = time.time() - t0
    print(f"\nBootstrap time: {elapsed:.1f}s ({ci.get('n_successful', 0)} successful)")

    if ci.get("intensity_ci"):
        print("\nIntensity Parameter 95% CIs:")
        for trans, vals in ci["intensity_ci"].items():
            print(f"  {trans}: {vals['mean']:.5f} "
                  f"[{vals['ci_lower']:.5f}, {vals['ci_upper']:.5f}]")

    if ci.get("sojourn_ci"):
        print("\nSojourn Time 95% CIs (years):")
        for stage, vals in ci["sojourn_ci"].items():
            print(f"  Stage {stage}: {vals['mean_years']:.2f} "
                  f"[{vals['ci_lower']:.2f}, {vals['ci_upper']:.2f}]")

    return ci


def generate_trajectory_predictions(Q, output_dir: Path):
    """Generate and save stage occupancy predictions for key starting stages."""
    print("\nGenerating trajectory predictions...")
    horizons = list(range(0, 181, 3))  # 0 to 15 years in 3-month steps

    all_preds = []
    for start_stage in tqdm(["2B", "3", "4"], desc="Predicting trajectories",
                            unit="stage"):
        preds = predict_trajectory(Q, start_stage, horizons)
        preds["start_stage"] = start_stage
        all_preds.append(preds)

    preds_df = pd.concat(all_preds, ignore_index=True)
    preds_df.to_csv(output_dir / "trajectory_predictions.csv", index=False)
    print(f"  Saved {len(preds_df)} predictions to trajectory_predictions.csv")

    # Print summary for key start stages
    for start_stage in ["2B", "3", "4"]:
        subset = preds_df[preds_df["start_stage"] == start_stage]
        print(f"\n  Starting at Stage {start_stage}:")
        for yr in [1, 2, 5, 10]:
            row = subset[subset["years"] == yr]
            if len(row) == 1:
                probs = row.iloc[0]
                top3 = sorted(
                    [(s, probs[s]) for s in STAGE_LABELS],
                    key=lambda x: -x[1],
                )[:3]
                top3_str = ", ".join(f"Stage {s}={p:.1%}" for s, p in top3)
                print(f"    At {yr}yr: {top3_str}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Fit Multi-State Markov Model")
    parser.add_argument("--no-bootstrap", action="store_true",
                        help="Skip bootstrap CIs (faster)")
    parser.add_argument("--nsd-only", action="store_true",
                        help="Restrict to NSD+ patients only")
    parser.add_argument("--n-bootstrap", type=int, default=200,
                        help="Number of bootstrap resamples")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    features, cohort = load_data(nsd_only=args.nsd_only)

    # Model 1: Homogeneous CTMC
    homo_results = run_homogeneous_model(features, cohort)
    homo_result = homo_results["result"]

    # Save homogeneous results
    save_results(homo_result, OUTPUT_DIR)

    # Generate trajectory predictions
    generate_trajectory_predictions(homo_result.Q, OUTPUT_DIR)

    # Save expected first-passage times
    fpt_df = pd.DataFrame(homo_results["fpt_results"])
    fpt_df.to_csv(OUTPUT_DIR / "expected_transition_times.csv", index=False)

    # Model 2: Covariate CTMC
    cov_results = run_covariate_model(features, cohort)
    if cov_results:
        cov_result = cov_results["result"]
        # Save covariate results
        save_results(cov_result, OUTPUT_DIR / "covariate_model")
        if "hr_table" in cov_results:
            cov_results["hr_table"].to_csv(
                OUTPUT_DIR / "covariate_hazard_ratios.csv", index=False
            )

    # Bootstrap CIs (homogeneous model)
    if not args.no_bootstrap:
        ci = run_bootstrap(
            features, homo_result.allowed_transitions, n_bootstrap=args.n_bootstrap
        )
        homo_result.bootstrap_ci = ci
        save_results(homo_result, OUTPUT_DIR)  # Re-save with CIs
    else:
        print("\nSkipping bootstrap (--no-bootstrap flag)")

    # Final summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Homogeneous model: {'CONVERGED' if homo_result.converged else 'DID NOT CONVERGE'}")
    print(f"  Log-likelihood: {homo_result.log_likelihood:.1f}")
    print(f"  Observations: {homo_result.n_observations}")
    print(f"  Transitions: {homo_result.n_transitions}")
    print(f"  Allowed transitions: {len(homo_result.allowed_transitions)}")

    print("\nKey Sojourn Times (mean years in stage):")
    for stage in ["2B", "3", "4", "5"]:
        t = homo_result.sojourn_times.get(stage, float("inf"))
        if np.isfinite(t):
            print(f"  Stage {stage}: {t:.2f} years")

    if cov_results:
        cov_result = cov_results["result"]
        print(f"\nCovariate model: {'CONVERGED' if cov_result.converged else 'DID NOT CONVERGE'}")
        print(f"  Log-likelihood: {cov_result.log_likelihood:.1f}")

    print(f"\nResults saved to: {OUTPUT_DIR}")
    print("Done!")


if __name__ == "__main__":
    main()
