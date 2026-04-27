#!/usr/bin/env python3
"""
Phase 4 — Confounding-by-Indication Control Analysis

Critical analysis for Paper 9 academic review:
  The N(t)×LEDD interaction (p=0.011) in the ON-OFF gap model might be
  entirely explained by confounding — sicker patients get MORE LEDD AND
  have BIGGER gaps. If we control for disease severity (OFF-state UPDRS-III),
  and the interaction STILL survives, it's mechanistic. If it vanishes,
  it's confounding.

Three analyses:
  1. Add OFF-UPDRS as severity covariate (+ severity×LEDD interaction)
  2. Within-patient first-difference model (eliminates all time-invariant confounders)
  3. Lagged LEDD (Granger-style temporal precedence)

Output:
  outputs/mechanistic_twin/phase4/phase4_confounding_control.json
"""

from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

# ── Project root ──────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "mechanistic_twin"))
from _reproducibility import (
    capture_provenance,
    write_run_manifest,
)

UPDRS3_RAW = PROJECT_ROOT / "data" / "00_raw" / "MDS-UPDRS Part IV" / "MDS-UPDRS_Part_III_12Apr2026.csv"
ASSEMBLED  = PROJECT_ROOT / "outputs" / "mechanistic_twin" / "phase4" / "phase4_assembled_data.parquet"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "mechanistic_twin" / "phase4"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

RNG_SEED = 42
np.random.seed(RNG_SEED)
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


# ======================================================================
# Step 1: Build the analysis dataset with paired ON-OFF + LEDD + N(t)/N0
# ======================================================================
def build_analysis_dataset() -> pd.DataFrame:
    """Build dataset with gap, updrs3_off, n_frac, ledd_scaled."""
    print("=" * 70)
    print("Step 1: Building analysis dataset")
    print("=" * 70)

    # --- Load raw UPDRS-III for paired ON-OFF ---
    updrs = pd.read_csv(UPDRS3_RAW, low_memory=False)
    updrs["updrs3_total"] = pd.to_numeric(updrs["NP3TOT"], errors="coerce")
    updrs["PATNO"] = updrs["PATNO"].astype(str)

    # Split ON/OFF
    on_df = (
        updrs[updrs["PDSTATE"] == "ON"]
        [["PATNO", "EVENT_ID", "updrs3_total"]]
        .rename(columns={"updrs3_total": "updrs3_on"})
        .dropna(subset=["updrs3_on"])
    )
    off_df = (
        updrs[updrs["PDSTATE"] == "OFF"]
        [["PATNO", "EVENT_ID", "updrs3_total"]]
        .rename(columns={"updrs3_total": "updrs3_off"})
        .dropna(subset=["updrs3_off"])
    )

    print(f"  ON assessments: {len(on_df):,}")
    print(f"  OFF assessments: {len(off_df):,}")

    # Pair on PATNO + EVENT_ID (same visit)
    paired = on_df.merge(off_df, on=["PATNO", "EVENT_ID"], how="inner")
    paired["gap"] = paired["updrs3_off"] - paired["updrs3_on"]
    print(f"  Paired ON-OFF visits: {len(paired):,} ({paired['PATNO'].nunique():,} patients)")

    # --- Merge with assembled data for LEDD and N(t)/N0 ---
    assembled = pd.read_parquet(ASSEMBLED)
    assembled["PATNO"] = assembled["PATNO"].astype(str)

    asm_cols = ["PATNO", "EVENT_ID", "ledd_total", "pct_loss_per_yr_median",
                "years_from_baseline", "has_posterior", "months_from_baseline"]
    asm = assembled[asm_cols].copy()

    df = paired.merge(asm, on=["PATNO", "EVENT_ID"], how="inner")
    print(f"  After merge with assembled: {len(df):,} rows")

    # Compute N(t)/N0
    df["pct_loss_per_yr_median"] = pd.to_numeric(df["pct_loss_per_yr_median"], errors="coerce")
    df["years_from_baseline"] = pd.to_numeric(df["years_from_baseline"], errors="coerce")
    df["ledd_total"] = pd.to_numeric(df["ledd_total"], errors="coerce")
    df["months_from_baseline"] = pd.to_numeric(df["months_from_baseline"], errors="coerce")

    has_nfrac = df["pct_loss_per_yr_median"].notna() & df["years_from_baseline"].notna()
    df["n_frac"] = np.nan
    df.loc[has_nfrac, "n_frac"] = np.exp(
        -(df.loc[has_nfrac, "pct_loss_per_yr_median"] / 100.0)
        * df.loc[has_nfrac, "years_from_baseline"]
    )

    # Scale LEDD
    df["ledd_scaled"] = df["ledd_total"] / 500.0

    # Filter: need gap + LEDD > 0 + N(t)/N0 + updrs3_off
    mask = (
        df["gap"].notna()
        & (df["ledd_total"] > 0)
        & df["n_frac"].notna()
        & df["updrs3_off"].notna()
    )
    df_full = df[mask].copy()

    # Filter to patients with >= 2 obs for mixed-effects
    pat_counts = df_full["PATNO"].value_counts()
    pats_ge2 = pat_counts[pat_counts >= 2].index
    df_me = df_full[df_full["PATNO"].isin(pats_ge2)].copy()

    print(f"\n  Analysis dataset (gap + LEDD>0 + N/N0 + OFF-UPDRS, >=2 obs/patient):")
    print(f"    Rows: {len(df_me):,}")
    print(f"    Patients: {df_me['PATNO'].nunique():,}")
    print(f"\n  Variable distributions:")
    for col in ["gap", "updrs3_off", "n_frac", "ledd_scaled"]:
        print(f"    {col}: mean={df_me[col].mean():.3f}, "
              f"std={df_me[col].std():.3f}, "
              f"min={df_me[col].min():.3f}, max={df_me[col].max():.3f}")

    return df_me


# ======================================================================
# Step 2: Center variables for interaction models
# ======================================================================
def center_variables(df: pd.DataFrame) -> pd.DataFrame:
    """Grand-mean center all continuous predictors."""
    df = df.copy()
    for col, centered_col in [
        ("n_frac", "n_frac_c"),
        ("ledd_scaled", "ledd_c"),
        ("updrs3_off", "updrs3_off_c"),
    ]:
        mean_val = df[col].mean()
        df[centered_col] = df[col] - mean_val
        print(f"  Centered {col}: mean={mean_val:.4f}")
    return df


# ======================================================================
# Step 3: Fit mixed-effects models
# ======================================================================
def extract_model_info(fit, model_name: str) -> dict:
    """Extract coefficients, p-values, AIC from a fitted mixedlm."""
    info = {
        "model_name": model_name,
        "n_obs": int(fit.nobs),
        "n_groups": int(fit.nobs - fit.df_resid),  # approximate
        "converged": bool(fit.converged),
    }

    # Fixed effects
    coefs = {}
    for param_name in fit.fe_params.index:
        coefs[param_name] = {
            "coef": float(fit.fe_params[param_name]),
            "se": float(fit.bse_fe[param_name]) if param_name in fit.bse_fe.index else None,
            "z": float(fit.tvalues[param_name]) if param_name in fit.tvalues.index else None,
            "p": float(fit.pvalues[param_name]) if param_name in fit.pvalues.index else None,
        }
    info["fixed_effects"] = coefs

    # Random effect variance
    if hasattr(fit.cov_re, "iloc"):
        info["random_intercept_var"] = float(fit.cov_re.iloc[0, 0])
    elif hasattr(fit.cov_re, "item"):
        info["random_intercept_var"] = float(fit.cov_re.item())

    # Residual variance
    info["residual_var"] = float(fit.scale)

    # AIC (statsmodels mixedlm doesn't always have .aic)
    try:
        info["aic"] = float(fit.aic)
    except Exception:
        # Manual AIC: -2*loglike + 2*k
        try:
            ll = fit.llf
            k = len(fit.fe_params) + 1  # fixed + random intercept var
            info["aic"] = float(-2 * ll + 2 * k)
        except Exception:
            info["aic"] = None

    # Log-likelihood
    try:
        info["llf"] = float(fit.llf)
    except Exception:
        info["llf"] = None

    return info


def fit_mixed_models(df: pd.DataFrame) -> dict:
    """Fit the three mixed-effects models."""
    print("\n" + "=" * 70)
    print("Step 3: Fitting mixed-effects models (confounding control)")
    print("=" * 70)

    results = {}

    # ------------------------------------------------------------------
    # Model 1: Original interaction (no severity control)
    # gap ~ n_frac_c + ledd_c + n_frac_c:ledd_c + (1|PATNO)
    # ------------------------------------------------------------------
    print("\n  --- Model 1: Original (no severity control) ---")
    print("      gap ~ n_frac_c * ledd_c + (1|PATNO)")
    m1 = smf.mixedlm(
        "gap ~ n_frac_c * ledd_c",
        data=df, groups=df["PATNO"], re_formula="~1",
    )
    fit1 = m1.fit(reml=True)
    print(fit1.summary())

    m1_info = extract_model_info(fit1, "M1_original")
    interaction_key = "n_frac_c:ledd_c"
    m1_info["interaction_coef"] = m1_info["fixed_effects"][interaction_key]["coef"]
    m1_info["interaction_p"] = m1_info["fixed_effects"][interaction_key]["p"]
    results["model1_original"] = m1_info

    print(f"\n  >> Interaction: coef={m1_info['interaction_coef']:.4f}, "
          f"p={m1_info['interaction_p']:.6f}")

    # ------------------------------------------------------------------
    # Model 2: Severity-controlled (add updrs3_off as covariate)
    # gap ~ n_frac_c + ledd_c + n_frac_c:ledd_c + updrs3_off_c + (1|PATNO)
    # ------------------------------------------------------------------
    print("\n  --- Model 2: Severity-controlled (+ OFF-UPDRS) ---")
    print("      gap ~ n_frac_c * ledd_c + updrs3_off_c + (1|PATNO)")
    m2 = smf.mixedlm(
        "gap ~ n_frac_c * ledd_c + updrs3_off_c",
        data=df, groups=df["PATNO"], re_formula="~1",
    )
    fit2 = m2.fit(reml=True)
    print(fit2.summary())

    m2_info = extract_model_info(fit2, "M2_severity_controlled")
    m2_info["interaction_coef"] = m2_info["fixed_effects"][interaction_key]["coef"]
    m2_info["interaction_p"] = m2_info["fixed_effects"][interaction_key]["p"]
    m2_info["interaction_survives"] = m2_info["interaction_p"] < 0.05
    results["model2_severity_controlled"] = m2_info

    print(f"\n  >> Interaction: coef={m2_info['interaction_coef']:.4f}, "
          f"p={m2_info['interaction_p']:.6f}")
    print(f"  >> OFF-UPDRS: coef={m2_info['fixed_effects']['updrs3_off_c']['coef']:.4f}, "
          f"p={m2_info['fixed_effects']['updrs3_off_c']['p']:.6f}")
    surv = "SURVIVES" if m2_info["interaction_survives"] else "VANISHES"
    print(f"  >> Interaction {surv} after controlling for OFF-UPDRS severity")

    # ------------------------------------------------------------------
    # Model 3: Severity + severity x LEDD interaction
    # gap ~ n_frac_c + ledd_c + n_frac_c:ledd_c + updrs3_off_c + updrs3_off_c:ledd_c + (1|PATNO)
    # ------------------------------------------------------------------
    print("\n  --- Model 3: Severity + severity x LEDD ---")
    print("      gap ~ n_frac_c * ledd_c + updrs3_off_c * ledd_c + (1|PATNO)")
    # Note: n_frac_c * ledd_c expands to n_frac_c + ledd_c + n_frac_c:ledd_c
    # updrs3_off_c * ledd_c expands to updrs3_off_c + ledd_c + updrs3_off_c:ledd_c
    # ledd_c is shared, so: gap ~ n_frac_c + ledd_c + n_frac_c:ledd_c + updrs3_off_c + updrs3_off_c:ledd_c
    m3 = smf.mixedlm(
        "gap ~ n_frac_c + ledd_c + n_frac_c:ledd_c + updrs3_off_c + updrs3_off_c:ledd_c",
        data=df, groups=df["PATNO"], re_formula="~1",
    )
    fit3 = m3.fit(reml=True)
    print(fit3.summary())

    m3_info = extract_model_info(fit3, "M3_severity_x_ledd")
    m3_info["interaction_coef"] = m3_info["fixed_effects"][interaction_key]["coef"]
    m3_info["interaction_p"] = m3_info["fixed_effects"][interaction_key]["p"]
    m3_info["interaction_survives"] = m3_info["interaction_p"] < 0.05
    results["model3_severity_x_ledd"] = m3_info

    sev_ledd_key = "updrs3_off_c:ledd_c"
    print(f"\n  >> N(t)xLEDD interaction: coef={m3_info['interaction_coef']:.4f}, "
          f"p={m3_info['interaction_p']:.6f}")
    print(f"  >> Severity x LEDD: coef={m3_info['fixed_effects'][sev_ledd_key]['coef']:.4f}, "
          f"p={m3_info['fixed_effects'][sev_ledd_key]['p']:.6f}")
    surv = "SURVIVES" if m3_info["interaction_survives"] else "VANISHES"
    print(f"  >> N(t)xLEDD interaction {surv} even after severity x LEDD control")

    return results


# ======================================================================
# Step 4: First-difference within-patient analysis
# ======================================================================
def first_difference_analysis(df: pd.DataFrame) -> dict:
    """Eliminates all time-invariant patient confounders via differencing."""
    print("\n" + "=" * 70)
    print("Step 4: First-difference (within-patient) analysis")
    print("=" * 70)

    # Sort by patient and time
    df_sorted = df.sort_values(["PATNO", "months_from_baseline"]).copy()

    # Compute first differences within each patient
    diffs = []
    for patno, grp in df_sorted.groupby("PATNO"):
        if len(grp) < 3:
            continue  # Need 3+ visits for meaningful differences
        grp = grp.sort_values("months_from_baseline")
        for i in range(1, len(grp)):
            row_prev = grp.iloc[i - 1]
            row_curr = grp.iloc[i]
            dt_months = row_curr["months_from_baseline"] - row_prev["months_from_baseline"]
            if dt_months <= 0:
                continue
            diffs.append({
                "PATNO": patno,
                "delta_gap": row_curr["gap"] - row_prev["gap"],
                "delta_n_frac": row_curr["n_frac"] - row_prev["n_frac"],
                "delta_ledd": row_curr["ledd_scaled"] - row_prev["ledd_scaled"],
                "delta_updrs3_off": row_curr["updrs3_off"] - row_prev["updrs3_off"],
                "dt_months": dt_months,
            })

    df_diff = pd.DataFrame(diffs)
    print(f"  Patients with 3+ visits: {df_diff['PATNO'].nunique():,}")
    print(f"  First differences: {len(df_diff):,}")

    if len(df_diff) < 30:
        print("  WARNING: Too few differences for reliable analysis")
        return {
            "error": "insufficient_data",
            "n_patients": int(df_diff["PATNO"].nunique()),
            "n_differences": len(df_diff),
        }

    # Compute interaction term
    df_diff["delta_n_frac_x_delta_ledd"] = df_diff["delta_n_frac"] * df_diff["delta_ledd"]

    # Center
    for col in ["delta_n_frac", "delta_ledd", "delta_updrs3_off"]:
        df_diff[col + "_c"] = df_diff[col] - df_diff[col].mean()
    df_diff["delta_interaction_c"] = df_diff["delta_n_frac_c"] * df_diff["delta_ledd_c"]

    print(f"\n  Variable distributions (first differences):")
    for col in ["delta_gap", "delta_n_frac", "delta_ledd", "delta_updrs3_off"]:
        print(f"    {col}: mean={df_diff[col].mean():.4f}, std={df_diff[col].std():.4f}")

    # Fit: delta_gap ~ delta_n_frac + delta_ledd + delta_n_frac x delta_ledd
    # Use OLS since differencing already removes patient fixed effects
    import statsmodels.api as sm

    X_cols = ["delta_n_frac_c", "delta_ledd_c", "delta_interaction_c"]
    X = sm.add_constant(df_diff[X_cols])
    y = df_diff["delta_gap"]

    ols_fit = sm.OLS(y, X).fit(cov_type="cluster", cov_kwds={"groups": df_diff["PATNO"]})
    print("\n  First-difference OLS (cluster-robust SE by PATNO):")
    print(ols_fit.summary())

    fd_result = {
        "n_patients": int(df_diff["PATNO"].nunique()),
        "n_differences": int(len(df_diff)),
        "coefficients": {},
    }
    for param in ols_fit.params.index:
        fd_result["coefficients"][param] = {
            "coef": float(ols_fit.params[param]),
            "se": float(ols_fit.bse[param]),
            "t": float(ols_fit.tvalues[param]),
            "p": float(ols_fit.pvalues[param]),
        }

    # Extract interaction result
    if "delta_interaction_c" in ols_fit.params.index:
        fd_result["delta_interaction_coef"] = float(ols_fit.params["delta_interaction_c"])
        fd_result["delta_interaction_p"] = float(ols_fit.pvalues["delta_interaction_c"])
        fd_result["delta_interaction_survives"] = fd_result["delta_interaction_p"] < 0.05
        print(f"\n  >> Delta interaction: coef={fd_result['delta_interaction_coef']:.4f}, "
              f"p={fd_result['delta_interaction_p']:.4f}")
    else:
        fd_result["delta_interaction_coef"] = None
        fd_result["delta_interaction_p"] = None

    return fd_result


# ======================================================================
# Step 5: Lagged LEDD (Granger-style)
# ======================================================================
def lagged_ledd_analysis(df: pd.DataFrame) -> dict:
    """Does LAST visit's LEDD predict THIS visit's gap change?"""
    print("\n" + "=" * 70)
    print("Step 5: Lagged LEDD (Granger-style temporal precedence)")
    print("=" * 70)

    df_sorted = df.sort_values(["PATNO", "months_from_baseline"]).copy()

    # Build lagged dataset
    lagged_rows = []
    for patno, grp in df_sorted.groupby("PATNO"):
        if len(grp) < 3:
            continue
        grp = grp.sort_values("months_from_baseline")
        for i in range(1, len(grp)):
            row_prev = grp.iloc[i - 1]
            row_curr = grp.iloc[i]
            dt_months = row_curr["months_from_baseline"] - row_prev["months_from_baseline"]
            if dt_months <= 0:
                continue
            lagged_rows.append({
                "PATNO": patno,
                "gap_t": row_curr["gap"],
                "gap_t1": row_prev["gap"],
                "n_frac_t": row_curr["n_frac"],
                "ledd_t1": row_prev["ledd_scaled"],  # lagged LEDD
                "ledd_t": row_curr["ledd_scaled"],    # concurrent LEDD
                "updrs3_off_t": row_curr["updrs3_off"],
            })

    df_lag = pd.DataFrame(lagged_rows)
    print(f"  Patients with 3+ visits: {df_lag['PATNO'].nunique():,}")
    print(f"  Lagged observations: {len(df_lag):,}")

    if len(df_lag) < 30:
        print("  WARNING: Too few observations for lagged analysis")
        return {
            "error": "insufficient_data",
            "n_patients": int(df_lag["PATNO"].nunique()),
            "n_observations": len(df_lag),
        }

    # Center
    for col in ["gap_t1", "n_frac_t", "ledd_t1"]:
        df_lag[col + "_c"] = df_lag[col] - df_lag[col].mean()
    df_lag["n_frac_t_x_ledd_t1_c"] = df_lag["n_frac_t_c"] * df_lag["ledd_t1_c"]

    # Fit: gap_t ~ gap_{t-1} + n_frac_t + ledd_{t-1} + n_frac_t x ledd_{t-1} + (1|PATNO)
    print("\n  Lagged mixed-effects model:")
    print("    gap_t ~ gap_t1_c + n_frac_t_c + ledd_t1_c + n_frac_t_c:ledd_t1_c + (1|PATNO)")

    # Check if enough groups
    pat_counts = df_lag["PATNO"].value_counts()
    pats_ge2 = pat_counts[pat_counts >= 2].index
    df_lag_me = df_lag[df_lag["PATNO"].isin(pats_ge2)].copy()
    print(f"  Patients with >=2 lagged obs: {df_lag_me['PATNO'].nunique():,}, "
          f"rows: {len(df_lag_me):,}")

    if len(df_lag_me) < 30 or df_lag_me["PATNO"].nunique() < 10:
        # Fall back to OLS with clustered SE
        print("  Falling back to OLS with clustered SE (too few for mixedlm)")
        import statsmodels.api as sm

        X_cols = ["gap_t1_c", "n_frac_t_c", "ledd_t1_c", "n_frac_t_x_ledd_t1_c"]
        X = sm.add_constant(df_lag[X_cols])
        y = df_lag["gap_t"]
        ols_fit = sm.OLS(y, X).fit(cov_type="cluster", cov_kwds={"groups": df_lag["PATNO"]})
        print(ols_fit.summary())

        lag_result = {
            "method": "OLS_clustered",
            "n_patients": int(df_lag["PATNO"].nunique()),
            "n_observations": int(len(df_lag)),
            "coefficients": {},
        }
        for param in ols_fit.params.index:
            lag_result["coefficients"][param] = {
                "coef": float(ols_fit.params[param]),
                "se": float(ols_fit.bse[param]),
                "t": float(ols_fit.tvalues[param]),
                "p": float(ols_fit.pvalues[param]),
            }
        if "n_frac_t_x_ledd_t1_c" in ols_fit.params.index:
            lag_result["lag_interaction_coef"] = float(ols_fit.params["n_frac_t_x_ledd_t1_c"])
            lag_result["lag_interaction_p"] = float(ols_fit.pvalues["n_frac_t_x_ledd_t1_c"])
        return lag_result

    # Mixed-effects
    m_lag = smf.mixedlm(
        "gap_t ~ gap_t1_c + n_frac_t_c + ledd_t1_c + n_frac_t_x_ledd_t1_c",
        data=df_lag_me, groups=df_lag_me["PATNO"], re_formula="~1",
    )
    fit_lag = m_lag.fit(reml=True)
    print(fit_lag.summary())

    lag_result = extract_model_info(fit_lag, "Lagged_LEDD_Granger")

    interaction_col = "n_frac_t_x_ledd_t1_c"
    if interaction_col in lag_result["fixed_effects"]:
        lag_result["lag_interaction_coef"] = lag_result["fixed_effects"][interaction_col]["coef"]
        lag_result["lag_interaction_p"] = lag_result["fixed_effects"][interaction_col]["p"]
        print(f"\n  >> Lagged interaction: coef={lag_result['lag_interaction_coef']:.4f}, "
              f"p={lag_result['lag_interaction_p']:.4f}")

    return lag_result


# ======================================================================
# Step 6: Verdict
# ======================================================================
def determine_verdict(results: dict) -> str:
    """Determine overall verdict: MECHANISTIC, CONFOUNDING, or AMBIGUOUS."""
    print("\n" + "=" * 70)
    print("Step 6: Determining verdict")
    print("=" * 70)

    m2 = results.get("model2_severity_controlled", {})
    m3 = results.get("model3_severity_x_ledd", {})
    fd = results.get("first_difference", {})

    # Check Model 2: does interaction survive after controlling for OFF-UPDRS?
    m2_survives = m2.get("interaction_survives", False)
    m2_p = m2.get("interaction_p", 1.0)

    # Check Model 3: does interaction survive after severity x LEDD?
    m3_survives = m3.get("interaction_survives", False)
    m3_p = m3.get("interaction_p", 1.0)

    # Check first-difference
    fd_survives = fd.get("delta_interaction_survives", False)
    fd_p = fd.get("delta_interaction_p", 1.0)
    fd_has_data = "error" not in fd

    print(f"\n  Model 2 (+ OFF-UPDRS):    interaction p = {m2_p:.6f} -> {'SURVIVES' if m2_survives else 'VANISHES'}")
    print(f"  Model 3 (+ sev x LEDD):   interaction p = {m3_p:.6f} -> {'SURVIVES' if m3_survives else 'VANISHES'}")
    if fd_has_data:
        print(f"  First-difference:         interaction p = {fd_p:.6f} -> {'SURVIVES' if fd_survives else 'VANISHES'}")
    else:
        print(f"  First-difference:         insufficient data")

    # Decision logic
    if m2_survives and m3_survives:
        if fd_has_data and fd_survives:
            verdict = "MECHANISTIC"
            rationale = (
                "Interaction survives all three controls: "
                "(1) OFF-UPDRS severity covariate, "
                "(2) severity x LEDD interaction, "
                "(3) within-patient first-difference. "
                "The N(t)xLEDD interaction is independent of confounding-by-indication."
            )
        elif fd_has_data and not fd_survives:
            verdict = "AMBIGUOUS"
            rationale = (
                "Interaction survives cross-sectional severity controls (Models 2, 3) "
                "but NOT the within-patient first-difference analysis. "
                "Time-invariant confounders may be partially controlled, "
                "but time-varying confounders remain possible."
            )
        else:
            verdict = "AMBIGUOUS"
            rationale = (
                "Interaction survives cross-sectional severity controls "
                "but first-difference analysis had insufficient data."
            )
    elif m2_survives and not m3_survives:
        verdict = "AMBIGUOUS"
        rationale = (
            "Interaction survives basic severity control (Model 2) "
            "but vanishes when severity x LEDD interaction is added (Model 3). "
            "The mechanism may be partially confounded."
        )
    elif not m2_survives:
        if m2_p > 0.10:
            verdict = "CONFOUNDING"
            rationale = (
                f"Interaction vanishes (p={m2_p:.4f} > 0.10) after controlling for "
                "OFF-state UPDRS-III. The original interaction was capturing "
                "disease severity, not a mechanistic N(t)->DA conversion effect."
            )
        else:
            verdict = "AMBIGUOUS"
            rationale = (
                f"Interaction weakens (p={m2_p:.4f}) after controlling for "
                "OFF-state UPDRS-III. Evidence is inconclusive."
            )
    else:
        verdict = "AMBIGUOUS"
        rationale = "Mixed evidence across analyses."

    print(f"\n  *** VERDICT: {verdict} ***")
    print(f"  Rationale: {rationale}")

    return verdict, rationale


# ======================================================================
# Step 7: Coefficient attenuation analysis
# ======================================================================
def coefficient_attenuation(results: dict) -> dict:
    """Measure how much the interaction coefficient attenuates across models."""
    m1 = results.get("model1_original", {})
    m2 = results.get("model2_severity_controlled", {})
    m3 = results.get("model3_severity_x_ledd", {})

    coef_m1 = m1.get("interaction_coef", None)
    coef_m2 = m2.get("interaction_coef", None)
    coef_m3 = m3.get("interaction_coef", None)

    attenuation = {}
    if coef_m1 is not None and coef_m2 is not None and coef_m1 != 0:
        pct_change_m2 = 100 * (coef_m2 - coef_m1) / abs(coef_m1)
        attenuation["m1_to_m2_pct_change"] = float(pct_change_m2)
        print(f"\n  Coefficient attenuation M1 -> M2: {pct_change_m2:+.1f}%")
        print(f"    M1: {coef_m1:.4f}, M2: {coef_m2:.4f}")

    if coef_m1 is not None and coef_m3 is not None and coef_m1 != 0:
        pct_change_m3 = 100 * (coef_m3 - coef_m1) / abs(coef_m1)
        attenuation["m1_to_m3_pct_change"] = float(pct_change_m3)
        print(f"  Coefficient attenuation M1 -> M3: {pct_change_m3:+.1f}%")
        print(f"    M1: {coef_m1:.4f}, M3: {coef_m3:.4f}")

    # Interpretation guide
    if attenuation.get("m1_to_m2_pct_change") is not None:
        change = abs(attenuation["m1_to_m2_pct_change"])
        if change < 10:
            attenuation["interpretation"] = "Minimal attenuation (<10%): interaction is robust to severity control"
        elif change < 30:
            attenuation["interpretation"] = "Moderate attenuation (10-30%): partial confounding but interaction persists"
        else:
            attenuation["interpretation"] = "Substantial attenuation (>30%): confounding explains much of the interaction"

    return attenuation


# ======================================================================
# Main
# ======================================================================
def main():
    print("Phase 4 — Confounding-by-Indication Control Analysis")
    print("=" * 70)
    print("Paper 9: Critical test — does N(t)xLEDD interaction survive")
    print("controlling for disease severity (OFF-state UPDRS-III)?")
    print("=" * 70)

    # Provenance
    provenance = capture_provenance(
        script_path=Path(__file__).resolve(),
        repo_root=PROJECT_ROOT,
        input_files=[UPDRS3_RAW, ASSEMBLED],
        extra={"seed": RNG_SEED, "analysis": "confounding_by_indication_control"},
    )

    # Step 1: Build dataset
    df = build_analysis_dataset()

    # Step 2: Center variables
    print("\n" + "=" * 70)
    print("Step 2: Centering variables")
    print("=" * 70)
    df = center_variables(df)

    # Step 3: Mixed-effects models
    results = fit_mixed_models(df)

    # Step 4: First-difference
    fd_result = first_difference_analysis(df)
    results["first_difference"] = fd_result

    # Step 5: Lagged LEDD
    lag_result = lagged_ledd_analysis(df)
    results["lagged_ledd"] = lag_result

    # Step 6: Verdict
    verdict, rationale = determine_verdict(results)
    results["verdict"] = verdict
    results["rationale"] = rationale

    # Step 7: Coefficient attenuation
    attenuation = coefficient_attenuation(results)
    results["coefficient_attenuation"] = attenuation

    # Save
    output = {
        "confounding_control_analysis": True,
        "description": (
            "Tests whether the N(t)/N0 x LEDD interaction in the ON-OFF gap model "
            "survives controlling for disease severity (OFF-state UPDRS-III). "
            "If it does, the interaction is mechanistic (fewer neurons -> less DA conversion). "
            "If it vanishes, it was confounding-by-indication."
        ),
        "results": results,
        "seed": RNG_SEED,
        "_provenance": provenance,
    }

    out_path = OUTPUT_DIR / "phase4_confounding_control.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\n  Results saved to: {out_path}")

    # Write run manifest
    write_run_manifest(
        manifest_path=OUTPUT_DIR / "phase4_confounding_control_RUN_MANIFEST.md",
        step_name="Phase 4 Confounding-by-Indication Control",
        provenance=provenance,
        summary_metrics={
            "verdict": verdict,
            "M1_interaction_p": results["model1_original"].get("interaction_p"),
            "M2_interaction_p": results["model2_severity_controlled"].get("interaction_p"),
            "M3_interaction_p": results["model3_severity_x_ledd"].get("interaction_p"),
            "first_diff_interaction_p": fd_result.get("delta_interaction_p"),
        },
    )

    # Final summary
    print("\n" + "=" * 70)
    print("SUMMARY — CONFOUNDING CONTROL ANALYSIS")
    print("=" * 70)
    print(f"\n  Model 1 (original):           interaction p = "
          f"{results['model1_original'].get('interaction_p', 'N/A'):.6f}")
    print(f"  Model 2 (+ OFF-UPDRS):        interaction p = "
          f"{results['model2_severity_controlled'].get('interaction_p', 'N/A'):.6f}")
    print(f"  Model 3 (+ sev x LEDD):       interaction p = "
          f"{results['model3_severity_x_ledd'].get('interaction_p', 'N/A'):.6f}")
    if "delta_interaction_p" in fd_result:
        print(f"  First-difference:             interaction p = "
              f"{fd_result['delta_interaction_p']:.6f}")
    if "lag_interaction_p" in lag_result:
        print(f"  Lagged LEDD (Granger):        interaction p = "
              f"{lag_result['lag_interaction_p']:.6f}")

    print(f"\n  *** VERDICT: {verdict} ***")
    print(f"  {rationale}")


if __name__ == "__main__":
    main()
