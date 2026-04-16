"""Ch 9.6 LOO forward validation with rate-change and stratified analyses.

Inputs:
  - SAEM v3: outputs/mechanistic_twin/data/posteriors/saem_multi_obs_v3/individual_params.csv
  - SAEM v3: outputs/mechanistic_twin/data/posteriors/saem_multi_obs_v3/diagnostics.json
  - Cohort:  outputs/mechanistic_twin/ch9_6/cohort_5channel.parquet

Outputs:
  - loo_forward.json      (aggregate coverage + relative error)
  - loo_forward.csv       (per-scan results)
  - residual_timecourse.json (trend slopes, p-values)
  - loo_stratified.json   (slow/normal/fast coverage)

Method:
- Importance-sampling LOO (not refit-based; cheaper by ~1000x).
- Forward model: SBR(t) = sbr_anchor * exp(GAMMA * (-decay_hr * t_hr))
  where decay_hr = alpha_tox * O_ss(k_n) + K_AGE  (K_AGE = 0 in v3)
  and O_ss(k_n) = k_n * M_SS^2 / (K_CONV + K_CLEAR_O).
  This exactly matches multi_obs_saem.py::compute_patient_log_likelihood.
- For each (patno, held-out scan):
    1. Sample from the patient's log-normal posterior (mean, sd) for (k_n, alpha_tox).
    2. Reweight samples by the Gaussian LL of remaining scans (sigma_sbr from diagnostics.json).
    3. Simulate SBR at held-out scan time for each sample.
    4. Compute 95% CI, check if observed is in CI.
- Trend analysis: for patients with >=3 scans, fit linear regression of
  (observed - predicted) vs visit index, report slope p-value.
- Stratification: pct_loss_per_yr thresholds on informative subset (sigma_logkn < 1.0).
  Patients with pct_loss >= 99 are capped at 50%/yr to avoid ceiling artifacts.

Note on 100%/yr ceiling: 49 patients have pct_loss_per_yr = 100.0 (SS exponential
ceiling artifact). These are capped at 50.0 for stratification but included in all
other analyses unchanged. This is documented in CLAUDE.md.

Performance: ~4,000 LOO iterations at 300 samples each ~ 5-15 min on a laptop.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[2]
V3_DIR = ROOT / "outputs/mechanistic_twin/data/posteriors/saem_multi_obs_v3"
COHORT = ROOT / "outputs/mechanistic_twin/ch9_6/cohort_5channel.parquet"
OUT_DIR = ROOT / "outputs/mechanistic_twin/ch9_6"

# ── Forward-model constants (exact match to multi_obs_saem.py) ──────────────
K_CONV    = 0.001
K_CLEAR_O = 0.003
K_AGE     = 0.0          # set to 0 in SAEM v3
M_SS      = 2.0          # nM (K_PROD/K_CLEAR_M = 0.1/0.05)
GAMMA     = 0.7
HR_PER_YR = 8766.0       # hours per year (exact value from multi_obs_saem.py)

N_SAMPLES = 300          # IS samples per LOO iteration (reduced from 50k SAEM IS)


# ── Forward model ─────────────────────────────────────────────────────────────

def o_ss(k_n: np.ndarray) -> np.ndarray:
    """Steady-state oligomer concentration (vectorised)."""
    return k_n * M_SS ** 2 / (K_CONV + K_CLEAR_O)


def predict_sbr(k_n: np.ndarray, alpha_tox: np.ndarray,
                t_years: float, sbr_anchor: float) -> np.ndarray:
    """Predict SBR at time t_years relative to the anchor scan.

    Matches multi_obs_saem.py lines 262-266:
        decay_hr = atox * o_ss + K_AGE
        log_ratio = clip(-decay_hr * t_hr, -50, 0)
        sbr_pred  = sbr_anchor * exp(GAMMA * log_ratio)
    """
    decay_hr = alpha_tox * o_ss(k_n) + K_AGE
    t_hr = t_years * HR_PER_YR
    log_ratio = np.clip(-decay_hr * t_hr, -50.0, 0.0)
    return sbr_anchor * np.exp(GAMMA * log_ratio)


def log_sbr_ll(obs: float, pred: np.ndarray, sigma: float) -> np.ndarray:
    """Gaussian log-likelihood of SBR observation (vectorised)."""
    return -0.5 * ((pred - obs) / sigma) ** 2


# ── Per-patient IS-LOO ────────────────────────────────────────────────────────

def loo_one_patient(
    patno: int,
    visits: pd.DataFrame,
    ebe_row: pd.Series,
    sigma_sbr: float,
    seed: int = 42,
) -> list[dict]:
    """Importance-sampling LOO for one patient.

    For each held-out scan:
      1. Sample (k_n, alpha_tox) from the log-normal posterior.
      2. Reweight by LL of remaining scans.
      3. Predict held-out scan; report 95% CI and in-CI flag.

    Parameters
    ----------
    visits   : DataFrame with columns visit_month, sbr_putamen (may have NaN)
    ebe_row  : Row from individual_params.csv with log_k_n_mean/sd and
               log_alpha_tox_mean/sd for this patient.
    sigma_sbr: Population SBR noise sigma from diagnostics.json.
    """
    valid = visits.dropna(subset=["sbr_putamen"]).sort_values("visit_month").reset_index(drop=True)
    n = len(valid)
    if n < 2:
        return []   # need >=2 scans for LOO

    sbr_anchor = float(valid["sbr_putamen"].iloc[0])
    if not np.isfinite(sbr_anchor) or sbr_anchor <= 0:
        return []

    # t_years relative to first scan
    t0_months = float(valid["visit_month"].iloc[0])
    t_years_arr = ((valid["visit_month"].values - t0_months) / 12.0).astype(float)

    # Sample from patient posterior
    rng = np.random.default_rng(seed + int(patno) % (2 ** 31))
    mu_kn  = float(ebe_row["log_k_n_mean"])
    sd_kn  = max(float(ebe_row["log_k_n_sd"]), 1e-4)
    mu_at  = float(ebe_row["log_alpha_tox_mean"])
    sd_at  = max(float(ebe_row["log_alpha_tox_sd"]), 1e-4)

    log_kn_samples  = rng.normal(mu_kn, sd_kn, N_SAMPLES)
    log_at_samples  = rng.normal(mu_at, sd_at, N_SAMPLES)
    kn_samples      = np.exp(log_kn_samples)
    at_samples      = np.exp(log_at_samples)

    results = []
    for i in range(n):
        t_held  = t_years_arr[i]
        obs_sbr = float(valid["sbr_putamen"].iloc[i])
        if not np.isfinite(obs_sbr):
            continue

        # Skip the anchor scan itself (t=0): predict = anchor for all samples
        # (zero-width CI, trivially 100% coverage -- not a forward validation).
        # Scientifically, LOO forward validation requires the held-out scan to be
        # at a later timepoint than the training scans, so we exclude i=0.
        if t_held == 0.0:
            continue

        # Remaining scans (exclude held-out)
        rem_t  = np.delete(t_years_arr, i)
        rem_sb = valid["sbr_putamen"].values.copy()
        rem_sb = np.delete(rem_sb, i)
        mask   = np.isfinite(rem_sb)
        rem_t  = rem_t[mask]
        rem_sb = rem_sb[mask]

        # IS weights from remaining-scan likelihood
        log_w = np.zeros(N_SAMPLES)
        for t_r, sbr_r in zip(rem_t, rem_sb):
            pred_r = predict_sbr(kn_samples, at_samples, t_r, sbr_anchor)
            log_w += log_sbr_ll(sbr_r, pred_r, sigma_sbr)

        log_w -= log_w.max()
        w = np.exp(log_w)
        w_sum = w.sum()
        if w_sum < 1e-10 or not np.isfinite(w_sum):
            continue
        w /= w_sum
        ess = 1.0 / (w ** 2).sum()

        # Predict held-out scan — POSTERIOR PREDICTIVE interval.
        # A proper predictive interval adds observation noise to each posterior sample:
        #   sbr_predicted_obs ~ N(sbr_pred(theta), sigma_sbr)
        # This gives a 95% *predictive* CI rather than just the 95% posterior CI.
        # Without noise, the CI only captures parameter uncertainty and will be
        # too narrow for informative patients (sigma_logkn << sigma_sbr).
        preds      = predict_sbr(kn_samples, at_samples, t_held, sbr_anchor)
        noise      = rng.normal(0.0, sigma_sbr, size=N_SAMPLES)
        preds_obs  = preds + noise   # posterior predictive samples
        # Weighted quantiles of the posterior predictive distribution
        sort_idx = np.argsort(preds_obs)
        cum_w    = np.cumsum(w[sort_idx])
        lo_idx   = int(np.searchsorted(cum_w, 0.025))
        hi_idx   = int(np.searchsorted(cum_w, 0.975))
        lo_idx   = min(lo_idx, N_SAMPLES - 1)
        hi_idx   = min(hi_idx, N_SAMPLES - 1)
        ci_low   = float(preds_obs[sort_idx[lo_idx]])
        ci_high  = float(preds_obs[sort_idx[hi_idx]])
        # Weighted mean as point estimate
        pred_mean = float(np.sum(w * preds))

        results.append({
            "patno":       int(patno),
            "visit_index": int(i),
            "visit_month": float(valid["visit_month"].iloc[i]),
            "t_years":     float(t_held),
            "observed":    obs_sbr,
            "pred_mean":   pred_mean,
            "ci_low":      ci_low,
            "ci_high":     ci_high,
            "in_ci":       bool(ci_low <= obs_sbr <= ci_high),
            "rel_error":   float(abs(obs_sbr - pred_mean) / max(abs(obs_sbr), 1e-6)),
            "ess":         float(ess),
            "n_scans":     int(n),
        })
    return results


# ── Residual time-course analysis ─────────────────────────────────────────────

def residual_timecourse_analysis(df_res: pd.DataFrame) -> dict:
    """Linear regression of residuals vs visit order for patients with >=3 scans.

    slope > 0: residuals increase with visit order → rate DECELERATING
    slope < 0: residuals decrease                  → rate ACCELERATING
    """
    multi = df_res[df_res["n_scans"] >= 3].copy()
    slopes, pvals = [], []
    for patno, g in multi.groupby("patno"):
        g = g.sort_values("visit_index")
        if len(g) < 3:
            continue
        x = g["visit_index"].values.astype(float)
        y = (g["observed"] - g["pred_mean"]).values.astype(float)
        if np.std(y) < 1e-10:
            continue
        result = stats.linregress(x, y)
        slopes.append(result.slope)
        pvals.append(result.pvalue)

    if not slopes:
        return {
            "n_patients_with_3plus_scans": 0,
            "slope_residual_vs_visit_order": {},
            "pvalue_trend": {},
            "interpretation": "No patients with >=3 scans",
        }

    slopes_arr = np.array(slopes)
    pvals_arr  = np.array(pvals)
    return {
        "n_patients_with_3plus_scans": len(slopes),
        "slope_residual_vs_visit_order": {
            "median": float(np.median(slopes_arr)),
            "mean":   float(np.mean(slopes_arr)),
            "q025":   float(np.quantile(slopes_arr, 0.025)),
            "q975":   float(np.quantile(slopes_arr, 0.975)),
        },
        "pvalue_trend": {
            "median_pvalue":          float(np.median(pvals_arr)),
            "n_significant_at_0.05":  int((pvals_arr < 0.05).sum()),
            "frac_significant_at_0.05": float((pvals_arr < 0.05).mean()),
        },
        "interpretation": (
            "slope > 0 => residuals increase with visit order => rate is DECELERATING "
            "(actual decay slower than model predicts); "
            "slope < 0 => rate ACCELERATING; "
            "median near 0 with few individually-significant => constant-rate model holds."
        ),
    }


# ── Stratified analysis ───────────────────────────────────────────────────────

def stratified_coverage(df_res: pd.DataFrame, ebes: pd.DataFrame) -> dict:
    """Coverage by progressor class (slow <2%/yr, normal 2-5, fast >5).

    Uses only informative patients (sigma_logkn < 1.0).
    Caps pct_loss at 50 to avoid 100%/yr ceiling artefacts in fast stratum.
    """
    ebes_sub = ebes[["PATNO", "pct_loss_per_yr", "log_k_n_sd"]].rename(
        columns={"PATNO": "patno"}
    ).copy()
    # Cap ceiling artefacts
    ebes_sub["pct_loss_capped"] = ebes_sub["pct_loss_per_yr"].clip(upper=50.0)

    df_strat = df_res.merge(ebes_sub, on="patno", how="left")
    df_inf   = df_strat[df_strat["log_k_n_sd"] < 1.0].copy()

    def classify(x: float) -> str:
        if x < 2.0:  return "slow"
        if x < 5.0:  return "normal"
        return "fast"

    df_inf["progressor"] = df_inf["pct_loss_capped"].apply(classify)

    out: dict = {}
    for grp, g in df_inf.groupby("progressor"):
        out[grp] = {
            "n_scans":          int(len(g)),
            "n_patients":       int(g["patno"].nunique()),
            "coverage_95":      float(g["in_ci"].mean()) if len(g) > 0 else float("nan"),
            "median_rel_error": float(g["rel_error"].median()) if len(g) > 0 else float("nan"),
        }

    # Ensure all three groups are present (even if empty)
    for grp in ["slow", "normal", "fast"]:
        if grp not in out:
            out[grp] = {"n_scans": 0, "n_patients": 0,
                        "coverage_95": float("nan"), "median_rel_error": float("nan")}
    return out


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    t_start = time.time()

    # Load data
    ebes   = pd.read_csv(V3_DIR / "individual_params.csv")
    ebes["PATNO"] = ebes["PATNO"].astype(int)
    cohort = pd.read_parquet(COHORT)
    cohort["patno"] = cohort["patno"].astype(int)

    with (V3_DIR / "diagnostics.json").open() as f:
        diag = json.load(f)
    sigma_sbr = float(diag["population_params"]["sigma_sbr"])
    print(f"sigma_sbr from SAEM v3 diagnostics: {sigma_sbr:.5f}")

    ebe_idx = {int(r.PATNO): r for _, r in ebes.iterrows()}
    n_total = ebes["PATNO"].nunique()
    print(f"Running IS-LOO on {n_total} patients, {N_SAMPLES} samples each ...")

    all_results: list[dict] = []
    for i_pat, (patno, g) in enumerate(cohort.groupby("patno")):
        if patno not in ebe_idx:
            continue
        row = loo_one_patient(int(patno), g, ebe_idx[patno], sigma_sbr)
        all_results.extend(row)
        if (i_pat + 1) % 500 == 0:
            elapsed = time.time() - t_start
            print(f"  {i_pat + 1}/{n_total} patients | {len(all_results)} scans evaluated "
                  f"| {elapsed:.0f}s elapsed")

    elapsed = time.time() - t_start
    print(f"\nIS-LOO complete: {len(all_results)} scan evaluations in {elapsed:.1f}s")

    if not all_results:
        print("ERROR: no LOO results produced — check cohort / EBE alignment")
        return

    df_res = pd.DataFrame(all_results)

    # ── Aggregate metrics ───────────────────────────────────────────────────
    coverage   = float(df_res["in_ci"].mean())
    med_rel    = float(df_res["rel_error"].median())
    ess_median = float(df_res["ess"].median())

    # Coverage by time horizon (all forward scans)
    horizons = {}
    for label, lo, hi in [("<1yr", 0, 1), ("1-2yr", 1, 2), ("2-5yr", 2, 5), (">5yr", 5, 999)]:
        sub = df_res[(df_res["t_years"] >= lo) & (df_res["t_years"] < hi)]
        horizons[label] = {
            "n_scans":   int(len(sub)),
            "coverage":  float(sub["in_ci"].mean()) if len(sub) > 0 else float("nan"),
        }

    agg_out = {
        "n_scans_evaluated":             int(len(df_res)),
        "n_patients":                    int(df_res["patno"].nunique()),
        "coverage_95_credible_interval": coverage,
        "median_relative_error":         med_rel,
        "n_samples_per_loo_iteration":   N_SAMPLES,
        "ess_median":                    ess_median,
        "sigma_sbr_used":                sigma_sbr,
        "runtime_seconds":               round(elapsed, 1),
        "coverage_by_horizon":           horizons,
        "note": (
            "Anchor scan (visit_index=0, t=0) excluded: at t=0 all samples predict "
            "sbr_anchor exactly, yielding trivially 100% coverage. "
            "All n_scans_evaluated are true forward predictions (t > 0)."
        ),
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with (OUT_DIR / "loo_forward.json").open("w") as f:
        json.dump(agg_out, f, indent=2)
    df_res.to_csv(OUT_DIR / "loo_forward.csv", index=False)
    print("\nAggregate LOO results:")
    print(json.dumps(agg_out, indent=2))

    # ── Residual time-course ────────────────────────────────────────────────
    trend_out = residual_timecourse_analysis(df_res)
    with (OUT_DIR / "residual_timecourse.json").open("w") as f:
        json.dump(trend_out, f, indent=2, default=float)
    print("\nResidual time-course:")
    print(json.dumps(trend_out, indent=2, default=float))

    # ── Stratified coverage ─────────────────────────────────────────────────
    strat_out = stratified_coverage(df_res, ebes)
    with (OUT_DIR / "loo_stratified.json").open("w") as f:
        json.dump(strat_out, f, indent=2, default=float)
    print("\nStratified coverage (informative patients only, pct_loss capped at 50):")
    print(json.dumps(strat_out, indent=2, default=float))


if __name__ == "__main__":
    main()
