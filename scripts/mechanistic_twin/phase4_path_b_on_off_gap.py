#!/usr/bin/env python3
"""
Phase 4 — Path B: ON-OFF UPDRS-III Gap Model

Predicts the medication benefit (GAP = UPDRS_OFF - UPDRS_ON) as a function
of N(t)/N₀ and LEDD using a Hill dose-response model.

Scientific rationale:
  - UPDRS_OFF = unmasked disease burden ≈ f(N(t))
  - UPDRS_ON  = disease burden MINUS medication benefit
  - GAP = OFF - ON = medication benefit
  - As N(t) declines → fewer surviving neurons → less AADC enzyme
    → less DA from levodopa → GAP shrinks
  - Hill model: GAP = G_max × (ρ × LEDD × N/N₀)^h / (1 + (ρ × LEDD × N/N₀)^h)

Output:
  outputs/mechanistic_twin/phase4/phase4_path_b_results.json
"""

from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize

# ── Project root ──────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

UPDRS3_RAW = PROJECT_ROOT / "data" / "00_raw" / "MDS-UPDRS Part IV" / "MDS-UPDRS_Part_III_12Apr2026.csv"
ASSEMBLED  = PROJECT_ROOT / "outputs" / "mechanistic_twin" / "phase4" / "phase4_assembled_data.parquet"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "mechanistic_twin" / "phase4"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Reproducibility
RNG_SEED = 42
np.random.seed(RNG_SEED)
warnings.filterwarnings("ignore", category=RuntimeWarning)


# ======================================================================
# Step 1: Extract paired ON-OFF assessments
# ======================================================================
def extract_paired_on_off() -> pd.DataFrame:
    """Extract paired ON-OFF UPDRS-III assessments from raw data."""
    print("=" * 70)
    print("Step 1: Extracting paired ON-OFF assessments")
    print("=" * 70)

    updrs = pd.read_csv(UPDRS3_RAW, low_memory=False)
    print(f"  Raw UPDRS-III rows: {len(updrs):,}")

    # Compute total score from NP3TOT
    updrs["updrs3_total"] = pd.to_numeric(updrs["NP3TOT"], errors="coerce")
    print(f"  Valid NP3TOT scores: {updrs['updrs3_total'].notna().sum():,}")

    # PDSTATE distribution
    print(f"  PDSTATE distribution: {updrs['PDSTATE'].value_counts().to_dict()}")
    print(f"  PDSTATE missing: {updrs['PDSTATE'].isna().sum():,}")

    # Cast PATNO to str for consistent merging
    updrs["PATNO"] = updrs["PATNO"].astype(str)

    # Split ON/OFF — keep INFODT for date matching
    on_df = (
        updrs[updrs["PDSTATE"] == "ON"]
        [["PATNO", "EVENT_ID", "INFODT", "updrs3_total"]]
        .rename(columns={"updrs3_total": "updrs3_on", "INFODT": "date_on"})
        .dropna(subset=["updrs3_on"])
    )
    off_df = (
        updrs[updrs["PDSTATE"] == "OFF"]
        [["PATNO", "EVENT_ID", "INFODT", "updrs3_total"]]
        .rename(columns={"updrs3_total": "updrs3_off_raw", "INFODT": "date_off"})
        .dropna(subset=["updrs3_off_raw"])
    )

    print(f"  ON assessments with valid score: {len(on_df):,} ({on_df['PATNO'].nunique():,} patients)")
    print(f"  OFF assessments with valid score: {len(off_df):,} ({off_df['PATNO'].nunique():,} patients)")

    # Merge on PATNO + EVENT_ID (same visit)
    paired = on_df.merge(off_df, on=["PATNO", "EVENT_ID"], how="inner")
    paired["gap"] = paired["updrs3_off_raw"] - paired["updrs3_on"]

    print(f"\n  Paired ON-OFF visits: {len(paired):,}")
    print(f"  Patients with paired assessments: {paired['PATNO'].nunique():,}")

    # Gap distribution
    print(f"\n  Gap distribution (OFF - ON):")
    print(f"    Mean:   {paired['gap'].mean():.2f}")
    print(f"    Median: {paired['gap'].median():.2f}")
    print(f"    Std:    {paired['gap'].std():.2f}")
    print(f"    Min:    {paired['gap'].min():.1f}")
    print(f"    Max:    {paired['gap'].max():.1f}")
    print(f"    Negative gaps (ON > OFF): {(paired['gap'] < 0).sum()} "
          f"({100*(paired['gap'] < 0).mean():.1f}%)")

    return paired


# ======================================================================
# Step 2: Merge with LEDD and N(t)/N₀
# ======================================================================
def merge_with_ledd_and_nfrac(paired: pd.DataFrame) -> pd.DataFrame:
    """Merge paired ON-OFF data with LEDD and N(t)/N₀ from assembled data."""
    print("\n" + "=" * 70)
    print("Step 2: Merging with LEDD and N(t)/N₀")
    print("=" * 70)

    assembled = pd.read_parquet(ASSEMBLED)
    assembled["PATNO"] = assembled["PATNO"].astype(str)
    print(f"  Assembled data: {len(assembled):,} rows, {assembled['PATNO'].nunique():,} patients")

    # Select relevant columns from assembled
    asm_cols = ["PATNO", "EVENT_ID", "ledd_total", "pct_loss_per_yr_median",
                "years_from_baseline", "has_posterior", "nsd_stage_numeric"]
    asm = assembled[asm_cols].copy()

    # Merge
    df = paired.merge(asm, on=["PATNO", "EVENT_ID"], how="inner")
    print(f"  After merge with assembled: {len(df):,} rows, {df['PATNO'].nunique():,} patients")

    # Compute N(t)/N₀ correctly using pct_loss_per_yr_median
    # (not T_tox_median which is in per-second units)
    df["pct_loss_per_yr_median"] = pd.to_numeric(df["pct_loss_per_yr_median"], errors="coerce")
    df["years_from_baseline"] = pd.to_numeric(df["years_from_baseline"], errors="coerce")
    df["ledd_total"] = pd.to_numeric(df["ledd_total"], errors="coerce")

    has_nfrac = df["pct_loss_per_yr_median"].notna() & df["years_from_baseline"].notna()
    df["n_frac"] = np.nan
    # Compound decay: pct_loss_per_yr is percentage of REMAINING neurons lost per year
    df.loc[has_nfrac, "n_frac"] = (
        (1 - df.loc[has_nfrac, "pct_loss_per_yr_median"] / 100.0)
        ** df.loc[has_nfrac, "years_from_baseline"]
    )

    # Scale LEDD for numerical stability
    df["ledd_scaled"] = df["ledd_total"] / 500.0

    print(f"  Rows with N(t)/N₀: {df['n_frac'].notna().sum():,}")
    print(f"  Rows with LEDD > 0: {(df['ledd_total'] > 0).sum():,}")
    print(f"  Rows with LEDD > 0 AND N(t)/N₀: "
          f"{((df['ledd_total'] > 0) & df['n_frac'].notna()).sum():,}")

    # Full analysis dataset: need gap, LEDD, n_frac all present
    full_mask = df["gap"].notna() & (df["ledd_total"] > 0) & df["n_frac"].notna()
    df_full = df[full_mask].copy()
    print(f"\n  Full analysis dataset (gap + LEDD>0 + N(t)/N₀):")
    print(f"    Rows: {len(df_full):,}")
    print(f"    Patients: {df_full['PATNO'].nunique():,}")

    if len(df_full) > 0:
        print(f"\n  N(t)/N₀ distribution:")
        print(f"    Mean:   {df_full['n_frac'].mean():.4f}")
        print(f"    Median: {df_full['n_frac'].median():.4f}")
        print(f"    Min:    {df_full['n_frac'].min():.4f}")
        print(f"    Max:    {df_full['n_frac'].max():.4f}")
        print(f"    Std:    {df_full['n_frac'].std():.4f}")
        print(f"\n  LEDD distribution (analysis subset):")
        print(f"    Mean:   {df_full['ledd_total'].mean():.1f}")
        print(f"    Median: {df_full['ledd_total'].median():.1f}")
        print(f"    Max:    {df_full['ledd_total'].max():.1f}")
        print(f"\n  Gap distribution (analysis subset):")
        print(f"    Mean:   {df_full['gap'].mean():.2f}")
        print(f"    Median: {df_full['gap'].median():.2f}")

    # Also prepare LEDD-only dataset (no N(t)/N₀ required)
    ledd_mask = df["gap"].notna() & (df["ledd_total"] > 0)
    df_ledd = df[ledd_mask].copy()
    print(f"\n  LEDD-only dataset (gap + LEDD>0, no N(t)/N₀ required):")
    print(f"    Rows: {len(df_ledd):,}")
    print(f"    Patients: {df_ledd['PATNO'].nunique():,}")

    return df, df_full, df_ledd


# ======================================================================
# Step 3: Fit models
# ======================================================================
def compute_aic(n: int, k: int, rss: float) -> float:
    """AIC from residual sum of squares assuming Gaussian errors."""
    if rss <= 0 or n <= k:
        return np.inf
    ll = -n / 2 * (np.log(2 * np.pi * rss / n) + 1)
    return 2 * k - 2 * ll


def compute_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """R² score."""
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    if ss_tot == 0:
        return 0.0
    return 1.0 - ss_res / ss_tot


def fit_models(df_full: pd.DataFrame, df_ledd: pd.DataFrame) -> dict:
    """Fit all 5 models and compare."""
    print("\n" + "=" * 70)
    print("Step 3: Fitting models")
    print("=" * 70)

    results = {}

    gap = df_full["gap"].values
    n_frac = df_full["n_frac"].values
    ledd = df_full["ledd_total"].values
    ledd_s = df_full["ledd_scaled"].values
    n = len(gap)

    print(f"\n  Analysis dataset: {n} observations, {df_full['PATNO'].nunique()} patients")

    # ------------------------------------------------------------------
    # B0: Intercept-only (null model)
    # ------------------------------------------------------------------
    print("\n  --- Model B0: Intercept-only (null) ---")
    gap_mean = gap.mean()
    pred_b0 = np.full_like(gap, gap_mean)
    rss_b0 = np.sum((gap - pred_b0) ** 2)
    r2_b0 = 0.0
    aic_b0 = compute_aic(n, 1, rss_b0)
    results["B0_null"] = {
        "description": "Intercept-only (null model)",
        "n_obs": n,
        "n_params": 1,
        "R2": r2_b0,
        "AIC": aic_b0,
        "RSS": float(rss_b0),
        "intercept": float(gap_mean),
    }
    print(f"    R² = {r2_b0:.4f}, AIC = {aic_b0:.1f}")
    print(f"    Intercept (mean gap) = {gap_mean:.2f}")

    # ------------------------------------------------------------------
    # B1: Gap ~ N(t)/N₀ (linear)
    # ------------------------------------------------------------------
    print("\n  --- Model B1: Gap ~ N(t)/N₀ ---")
    from numpy.polynomial.polynomial import polyfit as np_polyfit
    X_b1 = np.column_stack([np.ones(n), n_frac])
    beta_b1, res_b1, _, _ = np.linalg.lstsq(X_b1, gap, rcond=None)
    pred_b1 = X_b1 @ beta_b1
    rss_b1 = np.sum((gap - pred_b1) ** 2)
    r2_b1 = compute_r2(gap, pred_b1)
    aic_b1 = compute_aic(n, 2, rss_b1)
    results["B1_nfrac"] = {
        "description": "Gap ~ N(t)/N0 (linear)",
        "n_obs": n,
        "n_params": 2,
        "R2": float(r2_b1),
        "AIC": float(aic_b1),
        "RSS": float(rss_b1),
        "intercept": float(beta_b1[0]),
        "beta_nfrac": float(beta_b1[1]),
    }
    print(f"    R² = {r2_b1:.4f}, AIC = {aic_b1:.1f}")
    print(f"    β₀ = {beta_b1[0]:.3f}, β_nfrac = {beta_b1[1]:.3f}")

    # ------------------------------------------------------------------
    # B2: Gap ~ LEDD (linear) — uses full LEDD dataset
    # ------------------------------------------------------------------
    print("\n  --- Model B2: Gap ~ LEDD (on full paired+LEDD dataset) ---")
    gap_l = df_ledd["gap"].values
    ledd_l = df_ledd["ledd_total"].values
    ledd_sl = df_ledd["ledd_scaled"].values
    n_l = len(gap_l)

    X_b2 = np.column_stack([np.ones(n_l), ledd_sl])
    beta_b2, _, _, _ = np.linalg.lstsq(X_b2, gap_l, rcond=None)
    pred_b2 = X_b2 @ beta_b2
    rss_b2 = np.sum((gap_l - pred_b2) ** 2)
    r2_b2 = compute_r2(gap_l, pred_b2)
    aic_b2 = compute_aic(n_l, 2, rss_b2)
    results["B2_ledd"] = {
        "description": "Gap ~ LEDD (linear, larger dataset)",
        "n_obs": int(n_l),
        "n_params": 2,
        "R2": float(r2_b2),
        "AIC": float(aic_b2),
        "RSS": float(rss_b2),
        "intercept": float(beta_b2[0]),
        "beta_ledd_scaled": float(beta_b2[1]),
        "note": "LEDD scaled by /500",
    }
    print(f"    N = {n_l}")
    print(f"    R² = {r2_b2:.4f}, AIC = {aic_b2:.1f}")
    print(f"    β₀ = {beta_b2[0]:.3f}, β_ledd_scaled = {beta_b2[1]:.3f}")

    # B2 on same dataset as others for fair comparison
    X_b2f = np.column_stack([np.ones(n), ledd_s])
    beta_b2f, _, _, _ = np.linalg.lstsq(X_b2f, gap, rcond=None)
    pred_b2f = X_b2f @ beta_b2f
    rss_b2f = np.sum((gap - pred_b2f) ** 2)
    r2_b2f = compute_r2(gap, pred_b2f)
    aic_b2f = compute_aic(n, 2, rss_b2f)
    results["B2_ledd_matched"] = {
        "description": "Gap ~ LEDD (linear, matched N(t)/N0 dataset for fair AIC comparison)",
        "n_obs": n,
        "n_params": 2,
        "R2": float(r2_b2f),
        "AIC": float(aic_b2f),
        "RSS": float(rss_b2f),
        "intercept": float(beta_b2f[0]),
        "beta_ledd_scaled": float(beta_b2f[1]),
    }
    print(f"    (Matched dataset: R² = {r2_b2f:.4f}, AIC = {aic_b2f:.1f})")

    # ------------------------------------------------------------------
    # B3: Gap ~ N(t)/N₀ × LEDD (interaction)
    # ------------------------------------------------------------------
    print("\n  --- Model B3: Gap ~ N(t)/N₀ + LEDD + N(t)/N₀ × LEDD ---")
    interaction = n_frac * ledd_s
    X_b3 = np.column_stack([np.ones(n), n_frac, ledd_s, interaction])
    beta_b3, _, _, _ = np.linalg.lstsq(X_b3, gap, rcond=None)
    pred_b3 = X_b3 @ beta_b3
    rss_b3 = np.sum((gap - pred_b3) ** 2)
    r2_b3 = compute_r2(gap, pred_b3)
    aic_b3 = compute_aic(n, 4, rss_b3)
    results["B3_interaction"] = {
        "description": "Gap ~ N(t)/N0 + LEDD + N(t)/N0 x LEDD (interaction)",
        "n_obs": n,
        "n_params": 4,
        "R2": float(r2_b3),
        "AIC": float(aic_b3),
        "RSS": float(rss_b3),
        "intercept": float(beta_b3[0]),
        "beta_nfrac": float(beta_b3[1]),
        "beta_ledd_scaled": float(beta_b3[2]),
        "beta_interaction": float(beta_b3[3]),
    }
    print(f"    R² = {r2_b3:.4f}, AIC = {aic_b3:.1f}")
    print(f"    β₀ = {beta_b3[0]:.3f}, β_nfrac = {beta_b3[1]:.3f}, "
          f"β_ledd = {beta_b3[2]:.3f}, β_interact = {beta_b3[3]:.3f}")

    # ------------------------------------------------------------------
    # B4: Hill mechanistic model
    # GAP = G_max × (ρ × LEDD × N/N₀)^h / (1 + (ρ × LEDD × N/N₀)^h)
    # ------------------------------------------------------------------
    print("\n  --- Model B4: Hill mechanistic model ---")

    def hill_gap(params, ledd_arr, nfrac_arr, h_fixed=None):
        if h_fixed is not None:
            g_max, rho = params
            h = h_fixed
        else:
            g_max, rho, h = params
        x = rho * ledd_arr * nfrac_arr
        # Clamp to prevent overflow
        xh = np.clip(x, 0, 1e6) ** np.clip(h, 0.1, 10)
        return g_max * xh / (1.0 + xh)

    def hill_objective(params, ledd_arr, nfrac_arr, gap_obs, h_fixed=None):
        pred = hill_gap(params, ledd_arr, nfrac_arr, h_fixed)
        return np.sum((gap_obs - pred) ** 2)

    # B4a: h fixed at 2
    print("    B4a: h=2 (fixed)")
    best_b4a = None
    best_rss_b4a = np.inf
    for g0 in [5, 10, 20, 40]:
        for r0 in [0.0001, 0.001, 0.005, 0.01, 0.05]:
            res = minimize(
                hill_objective, x0=[g0, r0],
                args=(ledd, n_frac, gap, 2),
                method="Nelder-Mead",
                options={"maxiter": 50000, "xatol": 1e-10, "fatol": 1e-10},
            )
            if res.fun < best_rss_b4a:
                best_rss_b4a = res.fun
                best_b4a = res

    pred_b4a = hill_gap(best_b4a.x, ledd, n_frac, h_fixed=2)
    r2_b4a = compute_r2(gap, pred_b4a)
    aic_b4a = compute_aic(n, 2, best_rss_b4a)  # 2 params: G_max, rho
    results["B4a_hill_h2"] = {
        "description": "Hill model (h=2 fixed): GAP = G_max * (rho*LEDD*N/N0)^2 / (1 + (rho*LEDD*N/N0)^2)",
        "n_obs": n,
        "n_params": 2,
        "R2": float(r2_b4a),
        "AIC": float(aic_b4a),
        "RSS": float(best_rss_b4a),
        "G_max": float(best_b4a.x[0]),
        "rho": float(best_b4a.x[1]),
        "h": 2,
        "converged": bool(best_b4a.success),
    }
    print(f"    R² = {r2_b4a:.4f}, AIC = {aic_b4a:.1f}")
    print(f"    G_max = {best_b4a.x[0]:.3f}, ρ = {best_b4a.x[1]:.6f}")

    # B4b: h free
    print("    B4b: h free")
    best_b4b = None
    best_rss_b4b = np.inf
    for g0 in [5, 10, 20, 40]:
        for r0 in [0.0001, 0.001, 0.005, 0.01, 0.05]:
            for h0 in [0.5, 1.0, 2.0, 3.0]:
                res = minimize(
                    hill_objective, x0=[g0, r0, h0],
                    args=(ledd, n_frac, gap, None),
                    method="Nelder-Mead",
                    options={"maxiter": 50000, "xatol": 1e-10, "fatol": 1e-10},
                )
                if res.fun < best_rss_b4b:
                    best_rss_b4b = res.fun
                    best_b4b = res

    pred_b4b = hill_gap(best_b4b.x, ledd, n_frac)
    r2_b4b = compute_r2(gap, pred_b4b)
    aic_b4b = compute_aic(n, 3, best_rss_b4b)  # 3 params: G_max, rho, h
    results["B4b_hill_hfree"] = {
        "description": "Hill model (h free): GAP = G_max * (rho*LEDD*N/N0)^h / (1 + (rho*LEDD*N/N0)^h)",
        "n_obs": n,
        "n_params": 3,
        "R2": float(r2_b4b),
        "AIC": float(aic_b4b),
        "RSS": float(best_rss_b4b),
        "G_max": float(best_b4b.x[0]),
        "rho": float(best_b4b.x[1]),
        "h": float(best_b4b.x[2]),
        "converged": bool(best_b4b.success),
    }
    print(f"    R² = {r2_b4b:.4f}, AIC = {aic_b4b:.1f}")
    print(f"    G_max = {best_b4b.x[0]:.3f}, ρ = {best_b4b.x[1]:.6f}, h = {best_b4b.x[2]:.3f}")

    # ------------------------------------------------------------------
    # B5: Mixed-effects model (random intercept per patient)
    # ------------------------------------------------------------------
    print("\n  --- Model B5: Mixed-effects (random intercept per patient) ---")
    try:
        import statsmodels.formula.api as smf

        df_me = df_full[["PATNO", "gap", "n_frac", "ledd_scaled"]].copy()
        df_me = df_me.dropna()

        # Filter to patients with >= 2 observations
        pat_counts = df_me["PATNO"].value_counts()
        pats_ge2 = pat_counts[pat_counts >= 2].index
        df_me2 = df_me[df_me["PATNO"].isin(pats_ge2)].copy()
        print(f"    Patients with >=2 paired obs: {df_me2['PATNO'].nunique()}, "
              f"rows: {len(df_me2)}")

        if len(df_me2) > 20 and df_me2["PATNO"].nunique() > 5:
            model_me = smf.mixedlm(
                "gap ~ n_frac * ledd_scaled",
                data=df_me2,
                groups=df_me2["PATNO"],
                re_formula="~1",
            )
            fit_me = model_me.fit(reml=True)
            print(fit_me.summary())

            # Extract metrics
            pred_me = fit_me.fittedvalues
            gap_me = df_me2.loc[pred_me.index, "gap"]
            rss_me = np.sum((gap_me.values - pred_me.values) ** 2)
            r2_me = compute_r2(gap_me.values, pred_me.values)
            aic_me = float(fit_me.aic) if hasattr(fit_me, "aic") else compute_aic(len(gap_me), 5, rss_me)

            results["B5_mixed_effects"] = {
                "description": "Mixed-effects: gap ~ n_frac * ledd_scaled + (1|PATNO)",
                "n_obs": int(len(df_me2)),
                "n_patients": int(df_me2["PATNO"].nunique()),
                "n_params": 5,  # 4 fixed + 1 random variance
                "R2_conditional": float(r2_me),
                "AIC": float(aic_me),
                "RSS": float(rss_me),
                "fixed_effects": {
                    k: float(v) for k, v in fit_me.fe_params.items()
                },
                "random_effect_var": float(fit_me.cov_re.iloc[0, 0]) if hasattr(fit_me.cov_re, "iloc") else None,
                "converged": bool(fit_me.converged),
            }
            print(f"\n    R² (conditional) = {r2_me:.4f}, AIC = {aic_me:.1f}")
        else:
            print(f"    Too few observations for mixed-effects model")
            results["B5_mixed_effects"] = {"error": "insufficient data", "n_obs": len(df_me2)}

    except Exception as e:
        print(f"    Mixed-effects failed: {e}")
        results["B5_mixed_effects"] = {"error": str(e)}

    return results


# ======================================================================
# Step 4: Compare and generate figure data
# ======================================================================
def compare_models(results: dict, df_full: pd.DataFrame) -> dict:
    """Compare all models and generate figure data."""
    print("\n" + "=" * 70)
    print("Step 4: Model comparison")
    print("=" * 70)

    # Print comparison table
    print(f"\n  {'Model':<30s} {'n_obs':>6s} {'k':>3s} {'R²':>8s} {'AIC':>10s}")
    print("  " + "-" * 60)

    comparable_models = {}
    for name, res in results.items():
        if "R2" in res or "R2_conditional" in res:
            r2 = res.get("R2", res.get("R2_conditional", None))
            aic = res.get("AIC", None)
            k = res.get("n_params", "?")
            nobs = res.get("n_obs", "?")
            print(f"  {name:<30s} {nobs:>6} {k:>3} {r2:>8.4f} {aic:>10.1f}")
            comparable_models[name] = {"R2": r2, "AIC": aic, "n_obs": nobs}

    # Delta AIC relative to best
    # Only compare models on same dataset (same n_obs)
    main_n = results.get("B0_null", {}).get("n_obs", 0)
    same_n = {k: v for k, v in comparable_models.items() if v["n_obs"] == main_n}
    if same_n:
        best_aic = min(v["AIC"] for v in same_n.values())
        print(f"\n  ΔAIC (relative to best, same-N models only, N={main_n}):")
        for name, v in sorted(same_n.items(), key=lambda x: x[1]["AIC"]):
            delta = v["AIC"] - best_aic
            print(f"    {name:<30s} ΔAIC = {delta:>8.1f}  {'*** BEST' if delta == 0 else ''}")

    # Generate figure data: gap vs N(t)/N₀ colored by LEDD quartile
    figure_data = {}
    if len(df_full) > 0:
        df_fig = df_full[["gap", "n_frac", "ledd_total"]].dropna().copy()
        df_fig["ledd_quartile"] = pd.qcut(
            df_fig["ledd_total"], q=4, labels=["Q1 (low)", "Q2", "Q3", "Q4 (high)"]
        )
        quartile_stats = {}
        for q in ["Q1 (low)", "Q2", "Q3", "Q4 (high)"]:
            sub = df_fig[df_fig["ledd_quartile"] == q]
            quartile_stats[q] = {
                "n": int(len(sub)),
                "ledd_range": [float(sub["ledd_total"].min()), float(sub["ledd_total"].max())],
                "gap_mean": float(sub["gap"].mean()),
                "gap_median": float(sub["gap"].median()),
                "gap_std": float(sub["gap"].std()),
                "n_frac_mean": float(sub["n_frac"].mean()),
                "n_frac_median": float(sub["n_frac"].median()),
            }
        figure_data["gap_vs_nfrac_by_ledd_quartile"] = quartile_stats
        print(f"\n  Gap by LEDD quartile:")
        for q, s in quartile_stats.items():
            print(f"    {q}: n={s['n']}, LEDD=[{s['ledd_range'][0]:.0f}-{s['ledd_range'][1]:.0f}], "
                  f"gap_mean={s['gap_mean']:.2f}, N/N₀_mean={s['n_frac_mean']:.4f}")

    return figure_data


# ======================================================================
# Main
# ======================================================================
def main():
    print("Phase 4 — Path B: ON-OFF UPDRS-III Gap Model")
    print("=" * 70)

    # Step 1
    paired = extract_paired_on_off()

    # Step 2
    df_all, df_full, df_ledd = merge_with_ledd_and_nfrac(paired)

    # Step 3
    if len(df_full) < 10:
        print("\n  WARNING: Too few observations with all three variables.")
        print("  Fitting LEDD-only models on larger dataset instead.")
        # Even if we can't get n_frac for many, report what we have
        model_results = {}
    else:
        model_results = fit_models(df_full, df_ledd)

    # Step 4
    figure_data = compare_models(model_results, df_full)

    # Save results
    output = {
        "path_b_on_off_gap_analysis": True,
        "paired_on_off": {
            "total_paired_visits": int(len(paired)),
            "total_patients": int(paired["PATNO"].nunique()),
            "gap_mean": float(paired["gap"].mean()),
            "gap_median": float(paired["gap"].median()),
            "gap_std": float(paired["gap"].std()),
            "gap_min": float(paired["gap"].min()),
            "gap_max": float(paired["gap"].max()),
            "negative_gaps_n": int((paired["gap"] < 0).sum()),
            "negative_gaps_pct": float(100 * (paired["gap"] < 0).mean()),
        },
        "analysis_dataset": {
            "full_dataset_rows": int(len(df_full)),
            "full_dataset_patients": int(df_full["PATNO"].nunique()),
            "ledd_only_rows": int(len(df_ledd)),
            "ledd_only_patients": int(df_ledd["PATNO"].nunique()),
        },
        "model_results": model_results,
        "figure_data": figure_data,
        "seed": RNG_SEED,
    }

    out_path = OUTPUT_DIR / "phase4_path_b_results.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\n  Results saved to: {out_path}")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"  Paired ON-OFF visits: {len(paired):,}")
    print(f"  Full analysis (gap + LEDD + N/N₀): {len(df_full):,} rows, "
          f"{df_full['PATNO'].nunique():,} patients")
    if model_results:
        best = min(
            ((k, v) for k, v in model_results.items()
             if "AIC" in v and v.get("n_obs") == len(df_full)),
            key=lambda x: x[1]["AIC"],
            default=(None, None),
        )
        if best[0]:
            print(f"  Best model (by AIC): {best[0]} (AIC={best[1]['AIC']:.1f}, "
                  f"R²={best[1].get('R2', best[1].get('R2_conditional', 'N/A')):.4f})")


if __name__ == "__main__":
    main()
