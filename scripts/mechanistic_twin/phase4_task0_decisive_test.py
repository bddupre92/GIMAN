"""Phase 4 Task 0: Decisive test — does LEDD add independent information
beyond SBR-only decay rate?

Wave 1 (original): Spearman correlations between Phase 2 IS-weighted posterior
parameters (pct_loss_per_yr_median, T_tox_median) and per-patient mean LEDD.
Result: rho=-0.026, p=0.651 — null. But this tested the WRONG hypothesis
(main effect instead of interaction).

Wave 2 (corrected): Three tests of whether LEDD adds information beyond N(t)
for predicting UPDRS-III, using the visit-level assembled data.
  Test A: Partial correlation (residuals of UPDRS~N(t) correlated with LEDD)
  Test B: Mixed-effects interaction (random intercept per patient)
  Test C: Model comparison (N(t)-only vs N(t)+LEDD, AIC/R²)

Decision criteria (Wave 1):
  |rho| > 0.15 AND p < 0.05  => PASS (proceed to NLME)
  |rho| < 0.10 OR  p > 0.10  => FAIL (LEDD adds nothing beyond SBR trajectory)
  Otherwise                   => BORDERLINE (coordinator decides)

Decision criteria (Wave 2):
  2 of 3 corrected tests pass => PASS
  1 of 3                      => BORDERLINE
  0 of 3                      => FAIL

Output: outputs/mechanistic_twin/phase4/phase4_task0_decisive_test.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

# ── Project root and reproducibility ──────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
from scripts.mechanistic_twin._reproducibility import (
    capture_provenance,
    write_run_manifest,
)

import numpy as np
import pandas as pd
from scipy import stats

import statsmodels.api as sm
import statsmodels.formula.api as smf

# ── Paths ─────────────────────────────────────────────────────────────────
POSTERIORS_PATH = PROJECT_ROOT / "outputs" / "mechanistic_twin" / "data" / "posteriors" / "phase2_coupled_is_step26v4.csv"
LEDD_PATH = PROJECT_ROOT / "data" / "00_raw" / "LEDD_Concomitant_Medication_Log_12Apr2026.csv"
ASSEMBLED_PATH = PROJECT_ROOT / "outputs" / "mechanistic_twin" / "phase4" / "phase4_assembled_data.parquet"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "mechanistic_twin" / "phase4"
OUTPUT_JSON = OUTPUT_DIR / "phase4_task0_decisive_test.json"
MANIFEST_PATH = OUTPUT_DIR / "phase4_task0_RUN_MANIFEST.md"


def decide_verdict(rho: float, p: float) -> str:
    """Apply the decision gate to a single Spearman result."""
    abs_rho = abs(rho)
    if abs_rho > 0.15 and p < 0.05:
        return "PASS"
    elif abs_rho < 0.10 or p > 0.10:
        return "FAIL"
    else:
        return "BORDERLINE"


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ── Provenance ────────────────────────────────────────────────────────
    provenance = capture_provenance(
        script_path=Path(__file__).resolve(),
        repo_root=PROJECT_ROOT,
        input_files=[POSTERIORS_PATH, LEDD_PATH],
        extra={"task": "Phase 4 Task 0 decisive test", "decision_thresholds": {
            "pass": "|rho| > 0.15 AND p < 0.05",
            "fail": "|rho| < 0.10 OR p > 0.10",
        }},
    )

    # ── Load posteriors ───────────────────────────────────────────────────
    print("Loading Phase 2 IS posteriors...")
    posteriors = pd.read_csv(POSTERIORS_PATH)
    print(f"  Posteriors: {len(posteriors)} patients")
    print(f"  pct_loss_per_yr_median: median={posteriors['pct_loss_per_yr_median'].median():.2f}%/yr")

    # ── Load LEDD ─────────────────────────────────────────────────────────
    print("\nLoading LEDD data...")
    ledd_raw = pd.read_csv(LEDD_PATH)
    print(f"  Raw LEDD rows: {len(ledd_raw)}, unique patients: {ledd_raw['PATNO'].nunique()}")

    # Convert LEDD to numeric, coercing non-numeric entries (e.g. "LD x 0.33")
    ledd_raw["LEDD_numeric"] = pd.to_numeric(ledd_raw["LEDD"], errors="coerce")
    n_non_numeric = ledd_raw["LEDD_numeric"].isna().sum() - ledd_raw["LEDD"].isna().sum()
    print(f"  Non-numeric LEDD values dropped: {n_non_numeric}")

    # Compute per-patient mean LEDD (only numeric values)
    patient_ledd = (
        ledd_raw.dropna(subset=["LEDD_numeric"])
        .groupby("PATNO")["LEDD_numeric"]
        .agg(["mean", "median", "count"])
        .rename(columns={"mean": "mean_ledd", "median": "median_ledd", "count": "n_records"})
        .reset_index()
    )
    print(f"  Patients with numeric LEDD: {len(patient_ledd)}")

    # ── Merge ─────────────────────────────────────────────────────────────
    print("\nMerging posteriors + LEDD...")
    # Ensure PATNO types match (both should be int)
    posteriors["PATNO"] = posteriors["PATNO"].astype(int)
    patient_ledd["PATNO"] = patient_ledd["PATNO"].astype(int)

    merged = posteriors.merge(patient_ledd, on="PATNO", how="inner")
    print(f"  Overlap: {len(merged)} patients")
    print(f"  Posteriors-only (no LEDD): {len(posteriors) - len(merged)}")
    print(f"  LEDD-only (no posterior): {len(patient_ledd) - len(merged)}")

    if len(merged) < 10:
        print("\nERROR: Too few overlapping patients for meaningful correlation.")
        result = {
            "error": f"Only {len(merged)} overlapping patients",
            "verdict": "FAIL",
            "_provenance": provenance,
        }
        OUTPUT_JSON.write_text(json.dumps(result, indent=2))
        return

    # ── Summary stats ─────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("MERGED DATASET SUMMARY")
    print(f"{'='*60}")
    print(f"  N patients: {len(merged)}")
    print(f"  pct_loss_per_yr_median: mean={merged['pct_loss_per_yr_median'].mean():.2f}, "
          f"median={merged['pct_loss_per_yr_median'].median():.2f}")
    print(f"  T_tox_median: mean={merged['T_tox_median'].mean():.2e}, "
          f"median={merged['T_tox_median'].median():.2e}")
    print(f"  mean_ledd: mean={merged['mean_ledd'].mean():.1f}, "
          f"median={merged['mean_ledd'].median():.1f}")
    print(f"  LEDD records per patient: mean={merged['n_records'].mean():.1f}, "
          f"median={merged['n_records'].median():.0f}")

    # ── Spearman correlations ─────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("SPEARMAN CORRELATIONS")
    print(f"{'='*60}")

    # Test 1: pct_loss_per_yr_median vs mean_ledd
    rho1, p1 = stats.spearmanr(merged["pct_loss_per_yr_median"], merged["mean_ledd"])
    verdict1 = decide_verdict(rho1, p1)
    print(f"\n  Test 1: pct_loss_per_yr_median vs mean_ledd")
    print(f"    rho  = {rho1:.4f}")
    print(f"    p    = {p1:.4e}")
    print(f"    verdict = {verdict1}")

    # Test 2: T_tox_median vs mean_ledd
    rho2, p2 = stats.spearmanr(merged["T_tox_median"], merged["mean_ledd"])
    verdict2 = decide_verdict(rho2, p2)
    print(f"\n  Test 2: T_tox_median vs mean_ledd")
    print(f"    rho  = {rho2:.4f}")
    print(f"    p    = {p2:.4e}")
    print(f"    verdict = {verdict2}")

    # Also compute with median_ledd for robustness
    rho3, p3 = stats.spearmanr(merged["pct_loss_per_yr_median"], merged["median_ledd"])
    verdict3 = decide_verdict(rho3, p3)
    print(f"\n  Test 3 (robustness): pct_loss_per_yr_median vs median_ledd")
    print(f"    rho  = {rho3:.4f}")
    print(f"    p    = {p3:.4e}")
    print(f"    verdict = {verdict3}")

    # ── Overall verdict ───────────────────────────────────────────────────
    # Primary test is Test 1 (pct_loss vs mean_ledd)
    overall_verdict = verdict1
    print(f"\n{'='*60}")
    print(f"OVERALL VERDICT (based on primary test): {overall_verdict}")
    print(f"{'='*60}")

    if overall_verdict == "PASS":
        print("  => LEDD adds independent information beyond SBR decay rate.")
        print("  => Proceed to Phase 4 NLME model (Task 4).")
    elif overall_verdict == "FAIL":
        print("  => LEDD does NOT add independent information beyond SBR decay rate.")
        print("  => Phase 4 NLME model would be pointless.")
    else:
        print("  => Borderline result. Coordinator should review.")

    # ── Build result dict ─────────────────────────────────────────────────
    result = {
        "task": "Phase 4 Task 0: LEDD decisive test",
        "n_patients_overlap": int(len(merged)),
        "n_posteriors": int(len(posteriors)),
        "n_ledd_patients": int(len(patient_ledd)),
        "primary_test": {
            "variables": ["pct_loss_per_yr_median", "mean_ledd"],
            "spearman_rho": float(rho1),
            "p_value": float(p1),
            "verdict": verdict1,
        },
        "secondary_test_ttox": {
            "variables": ["T_tox_median", "mean_ledd"],
            "spearman_rho": float(rho2),
            "p_value": float(p2),
            "verdict": verdict2,
        },
        "robustness_test_median_ledd": {
            "variables": ["pct_loss_per_yr_median", "median_ledd"],
            "spearman_rho": float(rho3),
            "p_value": float(p3),
            "verdict": verdict3,
        },
        "overall_verdict": overall_verdict,
        "decision_criteria": {
            "pass": "|rho| > 0.15 AND p < 0.05",
            "fail": "|rho| < 0.10 OR p > 0.10",
            "borderline": "otherwise",
        },
        "summary_stats": {
            "pct_loss_per_yr_median": {
                "mean": float(merged["pct_loss_per_yr_median"].mean()),
                "median": float(merged["pct_loss_per_yr_median"].median()),
                "std": float(merged["pct_loss_per_yr_median"].std()),
            },
            "T_tox_median": {
                "mean": float(merged["T_tox_median"].mean()),
                "median": float(merged["T_tox_median"].median()),
                "std": float(merged["T_tox_median"].std()),
            },
            "mean_ledd": {
                "mean": float(merged["mean_ledd"].mean()),
                "median": float(merged["mean_ledd"].median()),
                "std": float(merged["mean_ledd"].std()),
            },
        },
        "_provenance": provenance,
    }

    # ── Wave 2: Corrected decisive tests ────────────────────────────────
    print(f"\n{'='*60}")
    print("WAVE 2: CORRECTED DECISIVE TESTS (interaction, not main effect)")
    print(f"{'='*60}")

    corrected = run_corrected_tests()
    result["corrected_test_A"] = corrected["test_A"]
    result["corrected_test_B"] = corrected["test_B"]
    result["corrected_test_C"] = corrected["test_C"]
    result["overall_corrected_verdict"] = corrected["overall_verdict"]
    result["corrected_data_summary"] = corrected["data_summary"]

    print(f"\n{'='*60}")
    print(f"CORRECTED OVERALL VERDICT: {corrected['overall_verdict']}")
    print(f"{'='*60}")

    # ── Write outputs ─────────────────────────────────────────────────────
    OUTPUT_JSON.write_text(json.dumps(result, indent=2))
    print(f"\nResults saved to: {OUTPUT_JSON}")

    # Write run manifest
    write_run_manifest(
        manifest_path=MANIFEST_PATH,
        step_name="Phase 4 Task 0: LEDD Decisive Test (Wave 1 + 2)",
        provenance=provenance,
        gate_results={
            "primary_test (pct_loss vs mean_ledd)": verdict1,
            "secondary_test (T_tox vs mean_ledd)": verdict2,
            "robustness_test (pct_loss vs median_ledd)": verdict3,
            "corrected_test_A (partial correlation)": corrected["test_A"]["verdict"],
            "corrected_test_B (mixed-effects interaction)": corrected["test_B"]["verdict"],
            "corrected_test_C (model comparison)": corrected["test_C"]["verdict"],
            "corrected_overall": corrected["overall_verdict"],
        },
        summary_metrics={
            "n_patients_overlap": len(merged),
            "primary_rho": f"{rho1:.4f}",
            "primary_p": f"{p1:.4e}",
            "overall_verdict_wave1": overall_verdict,
            "overall_verdict_wave2": corrected["overall_verdict"],
        },
    )
    print(f"Manifest saved to: {MANIFEST_PATH}")


def run_corrected_tests() -> dict:
    """Wave 2: Three corrected tests using visit-level assembled data.

    Tests the INTERACTION hypothesis: N(t) MODULATES response to LEDD,
    not the main-effect hypothesis that LEDD correlates with neuron death rate.

    Uses pct_loss_per_yr_median (not T_tox_median) to compute n_frac because
    T_tox_median is in per-second units (~1e-6) yielding n_frac~1.0 with no
    variance; pct_loss_per_yr_median (%/yr) gives biologically meaningful decay.
    """
    print("\nLoading assembled visit-level data...")
    df_all = pd.read_parquet(ASSEMBLED_PATH)
    print(f"  Total rows: {len(df_all)}, patients: {df_all['PATNO'].nunique()}")

    # ── Filter: LEDD > 0, UPDRS-III not null, posterior available ─────
    df_all["updrs3_off"] = pd.to_numeric(df_all["updrs3_off"], errors="coerce")
    df_all["ledd_total"] = pd.to_numeric(df_all["ledd_total"], errors="coerce")
    df_all["pct_loss_per_yr_median"] = pd.to_numeric(df_all["pct_loss_per_yr_median"], errors="coerce")
    df_all["years_from_baseline"] = pd.to_numeric(df_all["years_from_baseline"], errors="coerce")

    mask = (
        (df_all["ledd_total"] > 0)
        & df_all["updrs3_off"].notna()
        & df_all["pct_loss_per_yr_median"].notna()
        & df_all["years_from_baseline"].notna()
    )
    df = df_all[mask].copy()
    print(f"  After filtering (LEDD>0, UPDRS not null, posterior available): "
          f"{len(df)} rows, {df['PATNO'].nunique()} patients")

    # ── Compute N(t)/N₀ from pct_loss_per_yr ──────────────────────────
    # Compound decay: pct_loss_per_yr is percentage of REMAINING neurons lost per year
    df["n_frac"] = (1 - df["pct_loss_per_yr_median"] / 100.0) ** df["years_from_baseline"]
    # Clamp to (0, 1] for safety
    df["n_frac"] = df["n_frac"].clip(lower=1e-6, upper=1.0)

    # Replace any inf/nan
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=["n_frac", "updrs3_off", "ledd_total"])
    print(f"  After NaN/inf cleanup: {len(df)} rows, {df['PATNO'].nunique()} patients")

    data_summary = {
        "n_rows": int(len(df)),
        "n_patients": int(df["PATNO"].nunique()),
        "n_frac_mean": float(df["n_frac"].mean()),
        "n_frac_std": float(df["n_frac"].std()),
        "n_frac_min": float(df["n_frac"].min()),
        "n_frac_max": float(df["n_frac"].max()),
        "updrs3_off_mean": float(df["updrs3_off"].mean()),
        "ledd_total_mean": float(df["ledd_total"].mean()),
        "years_from_baseline_mean": float(df["years_from_baseline"].mean()),
    }

    # ── Test A: Partial Correlation ────────────────────────────────────
    print("\n--- Test A: Partial Correlation ---")
    print("  Regress UPDRS-III on N(t)/N₀, correlate residuals with LEDD")

    X_a = sm.add_constant(df["n_frac"].values)
    ols_a = sm.OLS(df["updrs3_off"].values, X_a).fit()
    residuals_a = ols_a.resid

    rho_a, p_a = stats.spearmanr(residuals_a, df["ledd_total"].values)
    pass_a = abs(rho_a) > 0.10 and p_a < 0.05

    print(f"  OLS(updrs3 ~ n_frac): R²={ols_a.rsquared:.4f}")
    print(f"  Spearman(residuals, LEDD): rho={rho_a:.4f}, p={p_a:.2e}")
    print(f"  Criterion: |rho|>0.10 AND p<0.05 => {'PASS' if pass_a else 'FAIL'}")

    test_A = {
        "test_name": "Partial Correlation: residuals(UPDRS~N(t)) vs LEDD",
        "ols_r_squared": float(ols_a.rsquared),
        "ols_n_frac_coef": float(ols_a.params[1]),
        "ols_n_frac_pvalue": float(ols_a.pvalues[1]),
        "spearman_rho_partial": float(rho_a),
        "spearman_p_partial": float(p_a),
        "criterion": "|rho_partial| > 0.10 AND p < 0.05",
        "verdict": "PASS" if pass_a else "FAIL",
    }

    # ── Test B: Mixed-Effects Interaction ──────────────────────────────
    print("\n--- Test B: Within-Patient Mixed-Effects Interaction ---")

    df_me = df[["PATNO", "updrs3_off", "n_frac", "ledd_total"]].copy()
    df_me["n_frac_c"] = df_me["n_frac"] - df_me["n_frac"].mean()
    df_me["ledd_c"] = df_me["ledd_total"] - df_me["ledd_total"].mean()

    try:
        me_model = smf.mixedlm(
            "updrs3_off ~ n_frac_c + ledd_c + n_frac_c:ledd_c",
            data=df_me,
            groups=df_me["PATNO"],
            re_formula="~1",
        )
        me_result = me_model.fit(reml=True)

        print(me_result.summary())

        # Extract interaction term
        interaction_key = "n_frac_c:ledd_c"
        interaction_coef = float(me_result.fe_params[interaction_key])
        interaction_p = float(me_result.pvalues[interaction_key])
        pass_b = interaction_p < 0.05

        print(f"\n  Interaction (n_frac_c:ledd_c): coef={interaction_coef:.6f}, p={interaction_p:.2e}")
        print(f"  Criterion: interaction p < 0.05 => {'PASS' if pass_b else 'FAIL'}")

        test_B = {
            "test_name": "Mixed-Effects Interaction: updrs3 ~ n_frac_c + ledd_c + n_frac_c:ledd_c | (1|PATNO)",
            "fixed_effects": {
                k: {"coef": float(me_result.fe_params[k]), "p_value": float(me_result.pvalues[k])}
                for k in me_result.fe_params.index
            },
            "interaction_coef": interaction_coef,
            "interaction_p_value": interaction_p,
            "random_intercept_var": float(me_result.cov_re.iloc[0, 0]),
            "n_groups": int(me_result.nobs),
            "criterion": "interaction p < 0.05",
            "verdict": "PASS" if pass_b else "FAIL",
        }
    except Exception as exc:
        print(f"  Mixed-effects model FAILED: {exc}")
        print("  Falling back to OLS interaction (ignoring within-patient correlation)")
        ols_b = smf.ols("updrs3_off ~ n_frac_c + ledd_c + n_frac_c:ledd_c", data=df_me).fit()
        interaction_key = "n_frac_c:ledd_c"
        interaction_coef = float(ols_b.params[interaction_key])
        interaction_p = float(ols_b.pvalues[interaction_key])
        pass_b = interaction_p < 0.05

        print(f"  OLS Interaction (n_frac_c:ledd_c): coef={interaction_coef:.6f}, p={interaction_p:.2e}")
        print(f"  Criterion: interaction p < 0.05 => {'PASS' if pass_b else 'FAIL'}")

        test_B = {
            "test_name": "OLS Interaction (fallback): updrs3 ~ n_frac_c + ledd_c + n_frac_c:ledd_c",
            "note": f"Mixed-effects failed ({exc}); OLS fallback ignores within-patient correlation",
            "coefficients": {
                k: {"coef": float(ols_b.params[k]), "p_value": float(ols_b.pvalues[k])}
                for k in ols_b.params.index
            },
            "interaction_coef": interaction_coef,
            "interaction_p_value": interaction_p,
            "r_squared": float(ols_b.rsquared),
            "criterion": "interaction p < 0.05",
            "verdict": "PASS" if pass_b else "FAIL",
        }

    # ── Test C: Model Comparison (N(t)-only vs N(t)+LEDD) ─────────────
    print("\n--- Test C: Model Comparison (AIC) ---")

    X1 = sm.add_constant(df["n_frac"].values)
    ols_m1 = sm.OLS(df["updrs3_off"].values, X1).fit()

    X2_df = df[["n_frac", "ledd_total"]].copy()
    X2_df["n_frac_x_ledd"] = X2_df["n_frac"] * X2_df["ledd_total"]
    X2 = sm.add_constant(X2_df.values)
    ols_m2 = sm.OLS(df["updrs3_off"].values, X2).fit()

    delta_aic = ols_m1.aic - ols_m2.aic  # positive = coupled model better
    delta_r2 = ols_m2.rsquared - ols_m1.rsquared
    pass_c = delta_aic > 10

    print(f"  Model 1 (N(t)-only):    R²={ols_m1.rsquared:.4f}, AIC={ols_m1.aic:.1f}")
    print(f"  Model 2 (N(t)+LEDD):    R²={ols_m2.rsquared:.4f}, AIC={ols_m2.aic:.1f}")
    print(f"  Delta AIC (M1 - M2):    {delta_aic:.1f}  (positive = coupled better)")
    print(f"  Delta R²:               {delta_r2:.4f}")
    print(f"  Criterion: ΔAIC > 10 favoring coupled model => {'PASS' if pass_c else 'FAIL'}")

    test_C = {
        "test_name": "Model Comparison: N(t)-only vs N(t)+LEDD+interaction",
        "model_1_n_only": {
            "formula": "updrs3 ~ n_frac",
            "r_squared": float(ols_m1.rsquared),
            "aic": float(ols_m1.aic),
            "bic": float(ols_m1.bic),
        },
        "model_2_coupled": {
            "formula": "updrs3 ~ n_frac + ledd + n_frac*ledd",
            "r_squared": float(ols_m2.rsquared),
            "aic": float(ols_m2.aic),
            "bic": float(ols_m2.bic),
        },
        "delta_aic": float(delta_aic),
        "delta_r_squared": float(delta_r2),
        "criterion": "delta_AIC > 10 favoring coupled model",
        "verdict": "PASS" if pass_c else "FAIL",
    }

    # ── Overall corrected verdict ─────────────────────────────────────
    n_pass = sum([pass_a, pass_b, pass_c])
    if n_pass >= 2:
        overall = "PASS"
    elif n_pass == 1:
        overall = "BORDERLINE"
    else:
        overall = "FAIL"

    print(f"\n  Tests passed: {n_pass}/3 => {overall}")

    return {
        "test_A": test_A,
        "test_B": test_B,
        "test_C": test_C,
        "overall_verdict": overall,
        "data_summary": data_summary,
    }


if __name__ == "__main__":
    main()
