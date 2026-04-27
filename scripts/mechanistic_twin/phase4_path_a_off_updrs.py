#!/usr/bin/env python
"""Phase 4 Path A: OFF-state UPDRS-III predicted from N(t)/N₀.

Models:
  A1 — Pooled OLS (UPDRS3_OFF ~ n_frac)
  A2 — Linear mixed-effects (random intercept + slope per patient)
  A3 — Hill-type mechanistic (Emax sigmoid)
  A4 — Time-only OLS baseline (UPDRS3_OFF ~ years_from_baseline)
  A5 — Time mixed-effects baseline

Key comparison: does per-patient neurodegeneration trajectory N(t)/N₀
outperform a simple time trend for predicting OFF-state motor score?

Usage:
  .venv/bin/python scripts/mechanistic_twin/phase4_path_a_off_updrs.py
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy.optimize import minimize

# ── repo anchors ──────────────────────────────────────────────────────
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from scripts.mechanistic_twin._reproducibility import (
    capture_provenance,
    write_run_manifest,
)

INPUT_PATH = REPO_ROOT / "outputs" / "mechanistic_twin" / "phase4" / "phase4_assembled_data.parquet"
OUTPUT_DIR = REPO_ROOT / "outputs" / "mechanistic_twin" / "phase4"
OUTPUT_JSON = OUTPUT_DIR / "phase4_path_a_results.json"
MANIFEST_PATH = OUTPUT_DIR / "phase4_path_a_RUN_MANIFEST.md"


# ── helpers ───────────────────────────────────────────────────────────

def _aic_from_sse(sse: float, n: int, k: int) -> float:
    """AIC from sum-of-squared-errors for OLS/nonlinear LS (Gaussian)."""
    return n * np.log(sse / n) + 2 * k


def _compute_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - ss_res / ss_tot


# ── data loading ──────────────────────────────────────────────────────

def load_data() -> pd.DataFrame:
    """Load assembled parquet and return analysis-ready subset."""
    df = pd.read_parquet(INPUT_PATH)

    # Filter: must have posteriors + non-null UPDRS3-OFF + months
    mask = (
        (df["has_posterior"] == 1)
        & df["updrs3_off"].notna()
        & df["months_from_baseline"].notna()
        & df["pct_loss_per_yr_median"].notna()
    )
    sub = df.loc[mask].copy()

    # ── CRITICAL FIX: recompute n_frac from pct_loss_per_yr_median ──
    # The parquet's n_frac was computed from T_tox_median (per-second units,
    # gives n_frac ≈ 1.0, useless). Use pct_loss_per_yr_median instead.
    sub["years_from_baseline"] = sub["months_from_baseline"] / 12.0
    sub["n_frac"] = (1 - sub["pct_loss_per_yr_median"] / 100.0) ** sub["years_from_baseline"]

    # Rename for formula convenience
    sub["updrs3_total"] = sub["updrs3_off"]

    # Drop rows where n_frac is non-finite (shouldn't happen, safety)
    sub = sub[np.isfinite(sub["n_frac"])]

    print(f"Analysis subset: {len(sub)} visits, {sub['PATNO'].nunique()} patients")
    print(f"  n_frac range: [{sub['n_frac'].min():.4f}, {sub['n_frac'].max():.4f}]")
    print(f"  UPDRS3-OFF range: [{sub['updrs3_total'].min():.0f}, {sub['updrs3_total'].max():.0f}]")
    print(f"  Years from BL range: [{sub['years_from_baseline'].min():.1f}, {sub['years_from_baseline'].max():.1f}]")
    return sub


# ── Model A1: Pooled OLS ─────────────────────────────────────────────

def fit_model_a1(df: pd.DataFrame) -> dict:
    """Pooled OLS: UPDRS3 ~ β₀ + β₁ * n_frac."""
    X = sm.add_constant(df["n_frac"].values)
    y = df["updrs3_total"].values
    ols = sm.OLS(y, X).fit()

    print("\n═══ Model A1: Pooled OLS (UPDRS3 ~ n_frac) ═══")
    print(ols.summary().tables[1])
    print(f"  R² = {ols.rsquared:.4f}, Adj R² = {ols.rsquared_adj:.4f}")
    print(f"  AIC = {ols.aic:.1f}")

    return {
        "r2": round(float(ols.rsquared), 5),
        "r2_adj": round(float(ols.rsquared_adj), 5),
        "aic": round(float(ols.aic), 1),
        "coefs": {
            "intercept": round(float(ols.params[0]), 4),
            "n_frac": round(float(ols.params[1]), 4),
        },
        "pvalues": {
            "intercept": float(ols.pvalues[0]),
            "n_frac": float(ols.pvalues[1]),
        },
        "n_obs": int(ols.nobs),
        "rmse": round(float(np.sqrt(ols.mse_resid)), 3),
    }


# ── Model A2: Linear Mixed-Effects ───────────────────────────────────

def fit_model_a2(df: pd.DataFrame) -> dict:
    """LME: UPDRS3 ~ n_frac, random intercept + slope per patient."""
    # Try random intercept + slope first
    converged = False
    re_formula = "~n_frac"
    model_desc = "random intercept + slope"

    for attempt, (re_form, method, desc) in enumerate([
        ("~n_frac", ["lbfgs"], "random intercept + slope (lbfgs)"),
        ("~n_frac", ["powell"], "random intercept + slope (powell)"),
        ("~1", ["lbfgs"], "random intercept only (lbfgs)"),
    ]):
        try:
            model = smf.mixedlm(
                "updrs3_total ~ n_frac",
                data=df,
                groups=df["PATNO"],
                re_formula=re_form,
            )
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                result = model.fit(method=method, maxiter=500)
            if result.converged:
                re_formula = re_form
                model_desc = desc
                converged = True
                break
        except Exception as e:
            print(f"  Attempt {attempt}: {desc} failed — {e}")
            continue

    if not converged:
        print("  WARNING: Mixed model did not converge, using last result")

    print(f"\n═══ Model A2: Linear Mixed-Effects ({model_desc}) ═══")
    print(result.summary().tables[1])

    # Extract fixed effects
    fe = result.fe_params
    fe_pval = result.pvalues

    # Random effects variance (extract early — needed for AIC fallback)
    re_cov = result.cov_re
    re_var = {}
    if hasattr(re_cov, "values"):
        labels = re_cov.index.tolist()
        for i, lab in enumerate(labels):
            re_var[str(lab)] = round(float(re_cov.values[i, i]), 4)
    else:
        re_var["intercept_var"] = round(float(re_cov), 4)

    # Compute marginal R² (fixed effects only) and conditional R² (fixed + random)
    y = df["updrs3_total"].values
    y_pred_marginal = result.predict()  # fixed effects only by default
    # For conditional: fixed + random
    y_pred_cond = result.fittedvalues

    marginal_r2 = _compute_r2(y, y_pred_marginal)
    conditional_r2 = _compute_r2(y, y_pred_cond)

    rmse = float(np.sqrt(np.mean((y - y_pred_cond) ** 2)))

    # statsmodels mixedlm .aic can be NaN; use llf-based computation
    aic_val = result.aic
    if np.isnan(aic_val):
        # AIC = -2*llf + 2*k
        n_fe = len(result.fe_params)
        n_re = re_cov.size if hasattr(re_cov, 'size') else 1
        n_params = n_fe + n_re + 1  # +1 for residual variance
        aic_val = -2 * result.llf + 2 * n_params

    print(f"  Marginal R² = {marginal_r2:.4f}")
    print(f"  Conditional R² = {conditional_r2:.4f}")
    print(f"  AIC = {aic_val:.1f}")
    print(f"  Converged = {result.converged}")

    return {
        "marginal_r2": round(float(marginal_r2), 5),
        "conditional_r2": round(float(conditional_r2), 5),
        "aic": round(float(aic_val), 1),
        "converged": bool(result.converged),
        "model_desc": model_desc,
        "re_formula": re_formula,
        "fixed_effects": {k: round(float(v), 4) for k, v in fe.items()},
        "fixed_pvalues": {k: float(v) for k, v in fe_pval.items()},
        "random_effects_var": re_var,
        "residual_var": round(float(result.scale), 4),
        "n_obs": int(result.nobs),
        "n_groups": int(df["PATNO"].nunique()),
        "rmse": round(rmse, 3),
    }


# ── Model A3: Hill-type mechanistic ──────────────────────────────────

def fit_model_a3(df: pd.DataFrame) -> dict:
    """Hill model: UPDRS3 = U_max * (1 - n_frac^h / (K^h + n_frac^h))."""
    n = df["n_frac"].values
    y = df["updrs3_total"].values

    def hill_predict(params, n_frac):
        u_max, k = params
        h = 2.0  # fixed for identifiability
        return u_max * (1 - n_frac**h / (k**h + n_frac**h))

    def hill_loss(params):
        pred = hill_predict(params, n)
        return np.sum((y - pred) ** 2)

    # Grid search for initial conditions
    best_loss = np.inf
    best_x0 = [80.0, 0.5]
    for u_try in [50, 80, 100, 150]:
        for k_try in [0.3, 0.5, 0.7, 0.9]:
            try:
                res = minimize(hill_loss, [u_try, k_try], method="Nelder-Mead",
                               options={"maxiter": 5000})
                if res.fun < best_loss:
                    best_loss = res.fun
                    best_x0 = list(res.x)
            except Exception:
                pass

    # Refine with L-BFGS-B
    result = minimize(
        hill_loss,
        best_x0,
        method="L-BFGS-B",
        bounds=[(1, 300), (0.01, 2.0)],
        options={"maxiter": 10000},
    )
    u_max, k = result.x
    h = 2.0
    y_pred = hill_predict(result.x, n)
    r2 = _compute_r2(y, y_pred)
    rmse = float(np.sqrt(np.mean((y - y_pred) ** 2)))
    sse = float(np.sum((y - y_pred) ** 2))
    aic = _aic_from_sse(sse, len(y), 2)  # 2 free params (U_max, K; h fixed)

    print(f"\n═══ Model A3: Hill-type Mechanistic ═══")
    print(f"  U_max = {u_max:.2f}, K = {k:.4f}, h = {h:.0f} (fixed)")
    print(f"  R² = {r2:.4f}, RMSE = {rmse:.3f}")
    print(f"  AIC = {aic:.1f}")
    print(f"  When N/N₀=1 (healthy): UPDRS3 = {hill_predict(result.x, np.array([1.0]))[0]:.1f}")
    print(f"  When N/N₀=0.5: UPDRS3 = {hill_predict(result.x, np.array([0.5]))[0]:.1f}")
    print(f"  When N/N₀=0 (total loss): UPDRS3 = {u_max:.1f}")

    return {
        "params": {
            "U_max": round(float(u_max), 3),
            "K": round(float(k), 5),
            "h": h,
        },
        "r2": round(float(r2), 5),
        "rmse": round(rmse, 3),
        "aic": round(aic, 1),
        "n_obs": len(y),
        "optimizer_success": bool(result.success),
        "predictions_at_key_points": {
            "n_frac_1.0_healthy": round(float(hill_predict(result.x, np.array([1.0]))[0]), 2),
            "n_frac_0.8": round(float(hill_predict(result.x, np.array([0.8]))[0]), 2),
            "n_frac_0.5": round(float(hill_predict(result.x, np.array([0.5]))[0]), 2),
            "n_frac_0.2": round(float(hill_predict(result.x, np.array([0.2]))[0]), 2),
            "n_frac_0.0_total_loss": round(float(u_max), 2),
        },
    }


# ── Model A4: Time-only OLS ──────────────────────────────────────────

def fit_model_a4(df: pd.DataFrame) -> dict:
    """OLS: UPDRS3 ~ years_from_baseline (simple time trend)."""
    X = sm.add_constant(df["years_from_baseline"].values)
    y = df["updrs3_total"].values
    ols = sm.OLS(y, X).fit()

    print(f"\n═══ Model A4: Time-only OLS (UPDRS3 ~ years) ═══")
    print(ols.summary().tables[1])
    print(f"  R² = {ols.rsquared:.4f}")
    print(f"  AIC = {ols.aic:.1f}")

    return {
        "r2": round(float(ols.rsquared), 5),
        "r2_adj": round(float(ols.rsquared_adj), 5),
        "aic": round(float(ols.aic), 1),
        "coefs": {
            "intercept": round(float(ols.params[0]), 4),
            "years_from_baseline": round(float(ols.params[1]), 4),
        },
        "pvalues": {
            "intercept": float(ols.pvalues[0]),
            "years_from_baseline": float(ols.pvalues[1]),
        },
        "n_obs": int(ols.nobs),
        "rmse": round(float(np.sqrt(ols.mse_resid)), 3),
    }


# ── Model A5: Time-only Mixed-Effects ────────────────────────────────

def fit_model_a5(df: pd.DataFrame) -> dict:
    """LME: UPDRS3 ~ years_from_baseline, random intercept + slope."""
    converged = False
    for attempt, (re_form, method, desc) in enumerate([
        ("~years_from_baseline", ["lbfgs"], "random intercept + slope (lbfgs)"),
        ("~years_from_baseline", ["powell"], "random intercept + slope (powell)"),
        ("~1", ["lbfgs"], "random intercept only (lbfgs)"),
    ]):
        try:
            model = smf.mixedlm(
                "updrs3_total ~ years_from_baseline",
                data=df,
                groups=df["PATNO"],
                re_formula=re_form,
            )
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                result = model.fit(method=method, maxiter=500)
            if result.converged:
                model_desc = desc
                converged = True
                break
        except Exception as e:
            print(f"  A5 attempt {attempt}: {desc} failed — {e}")
            continue

    if not converged:
        print("  WARNING: A5 did not converge, using last result")
        model_desc = desc

    print(f"\n═══ Model A5: Time-only Mixed-Effects ({model_desc}) ═══")
    print(result.summary().tables[1])

    y = df["updrs3_total"].values
    y_pred_cond = result.fittedvalues
    marginal_r2 = _compute_r2(y, result.predict())
    conditional_r2 = _compute_r2(y, y_pred_cond)
    rmse = float(np.sqrt(np.mean((y - y_pred_cond) ** 2)))

    # AIC fallback (same as A2)
    aic_val = result.aic
    if np.isnan(aic_val):
        re_cov_a5 = result.cov_re
        n_fe = len(result.fe_params)
        n_re = re_cov_a5.size if hasattr(re_cov_a5, 'size') else 1
        n_params = n_fe + n_re + 1
        aic_val = -2 * result.llf + 2 * n_params

    print(f"  Marginal R² = {marginal_r2:.4f}")
    print(f"  Conditional R² = {conditional_r2:.4f}")
    print(f"  AIC = {aic_val:.1f}")

    fe = result.fe_params
    return {
        "marginal_r2": round(float(marginal_r2), 5),
        "conditional_r2": round(float(conditional_r2), 5),
        "aic": round(float(aic_val), 1),
        "converged": bool(result.converged),
        "model_desc": model_desc,
        "fixed_effects": {k: round(float(v), 4) for k, v in fe.items()},
        "n_obs": int(result.nobs),
        "rmse": round(rmse, 3),
    }


# ── main ──────────────────────────────────────────────────────────────

def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    provenance = capture_provenance(
        script_path=Path(__file__).resolve(),
        repo_root=REPO_ROOT,
        input_files=[INPUT_PATH],
    )

    df = load_data()

    # Fit all models
    a1 = fit_model_a1(df)
    a2 = fit_model_a2(df)
    a3 = fit_model_a3(df)
    a4 = fit_model_a4(df)
    a5 = fit_model_a5(df)

    # ── Key comparisons ───────────────────────────────────────────────
    print("\n" + "═" * 60)
    print("KEY COMPARISONS")
    print("═" * 60)

    # A2 vs A5: N(t)/N₀ mixed model vs time-only mixed model
    delta_aic_a2_vs_a5 = a2["aic"] - a5["aic"]
    print(f"\n  A2 (N(t) LME) vs A5 (time LME):")
    print(f"    ΔAIC = {delta_aic_a2_vs_a5:.1f}  (negative = A2 wins)")
    print(f"    Cond R²: A2={a2['conditional_r2']:.4f}  vs  A5={a5['conditional_r2']:.4f}")
    print(f"    Marg R²: A2={a2['marginal_r2']:.4f}  vs  A5={a5['marginal_r2']:.4f}")

    # A1 vs A4: N(t) OLS vs time-only OLS
    delta_aic_a1_vs_a4 = a1["aic"] - a4["aic"]
    print(f"\n  A1 (N(t) OLS) vs A4 (time OLS):")
    print(f"    ΔAIC = {delta_aic_a1_vs_a4:.1f}  (negative = A1 wins)")
    print(f"    R²:   A1={a1['r2']:.4f}  vs  A4={a4['r2']:.4f}")

    # A3 vs A1: Hill vs linear
    delta_aic_a3_vs_a1 = a3["aic"] - a1["aic"]
    print(f"\n  A3 (Hill) vs A1 (linear OLS):")
    print(f"    ΔAIC = {delta_aic_a3_vs_a1:.1f}  (negative = A3 wins)")
    print(f"    R²:   A3={a3['r2']:.4f}  vs  A1={a1['r2']:.4f}")

    # Headline number
    headline_r2 = a2["conditional_r2"]
    print(f"\n  ★ HEADLINE: Conditional R² (Model A2) = {headline_r2:.4f}")

    comparison = {
        "n_frac_lme_vs_time_lme_delta_aic": round(delta_aic_a2_vs_a5, 1),
        "n_frac_ols_vs_time_ols_delta_aic": round(delta_aic_a1_vs_a4, 1),
        "hill_vs_linear_delta_aic": round(delta_aic_a3_vs_a1, 1),
        "headline_conditional_r2": round(headline_r2, 5),
        "a2_wins_over_a5": delta_aic_a2_vs_a5 < 0,
        "interpretation": (
            "N(t)/N₀ mixed model beats time-only"
            if delta_aic_a2_vs_a5 < -2
            else (
                "Time-only mixed model beats N(t)/N₀"
                if delta_aic_a2_vs_a5 > 2
                else "N(t)/N₀ and time-only are comparable (|ΔAIC| < 2)"
            )
        ),
    }

    # ── Save results ──────────────────────────────────────────────────
    results = {
        "model_a1_ols": a1,
        "model_a2_mixed": a2,
        "model_a3_hill": a3,
        "model_a4_time_only": a4,
        "model_a5_time_mixed": a5,
        "comparison": comparison,
        "n_patients": int(df["PATNO"].nunique()),
        "n_visits": len(df),
        "_provenance": provenance,
    }

    with open(OUTPUT_JSON, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {OUTPUT_JSON}")

    write_run_manifest(
        manifest_path=MANIFEST_PATH,
        step_name="Phase 4 Path A: OFF-state UPDRS-III ~ N(t)/N₀",
        provenance=provenance,
        summary_metrics={
            "n_patients": df["PATNO"].nunique(),
            "n_visits": len(df),
            "A1_OLS_R2": a1["r2"],
            "A2_LME_conditional_R2": a2["conditional_r2"],
            "A3_Hill_R2": a3["r2"],
            "A4_time_OLS_R2": a4["r2"],
            "A5_time_LME_conditional_R2": a5["conditional_r2"],
            "headline": f"Conditional R² = {headline_r2:.4f}",
        },
    )
    print(f"Manifest saved to {MANIFEST_PATH}")


if __name__ == "__main__":
    main()
