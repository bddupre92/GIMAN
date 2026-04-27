#!/usr/bin/env python3
"""
Phase 2 Step 2.6v4 — Importance-weighted posterior over (k_n, alpha_tox, T_tox)
================================================================================

Purpose
-------
Replace the failed single-chain NUTS sampling of Step 2.6v3 with an
importance-weighted posterior computed directly from the Variant B
slow-fast-collapse closed-form SBR decay. Audit 4 (2026-04-09, at
/tmp/phase2_step27_audit4_importance_weighted.py) showed IS recovers the
predicted sloppy ridge (cor(log k_n, log alpha_tox) ≈ -0.98 for PATNO 3002,
T_tox SD 40× tighter than NUTS) while NUTS reported prior-dominated chains
with false-positive single-chain R-hat.

This script runs IS for all 304 Wave A patients and produces:
  - per-patient summary CSV (ESS, posterior means/quantiles, correlations,
    T_tox-derived %/yr neuron loss)
  - per-patient weighted chain parquets (replaces Step 2.6v3 chains for
    downstream Step 2.7 / 2.8 / S1 analyses)
  - aggregate JSON summary + stratified distributions
  - diagnostic figure (ESS distribution, T_tox vs alpha_tox tightening,
    implied %/yr stratified by ESS)

Method
------
1. Draw N_PRIOR = 50,000 samples from the priors (LogNormal(log(1e-4), 1.5)
   for k_n, LogNormal(log(1.8e-5), 2.0) for alpha_tox).
2. For each patient, compute the closed-form SBR decay under the slow-fast
   collapse (M, O at quasi-steady-state, log-N state solution):

       SBR(t) = SBR_0 * exp(gamma * log(N(t)/N_0))
       log(N(t)/N_0) = -(alpha_tox * O_ss + k_age) * t_hr
       O_ss          = k_n * M_ss^2 / (k_conv + k_clear_O)

   Pinned constants match calibrate_phase2_coupled.jl exactly (M_ss from
   K_PROD/K_CLEAR_M, K_CONV, K_CLEAR_O, GAMMA=0.7, K_AGE=0).

3. Compute Gaussian log-likelihood vs observed SBR with sigma = 0.20
   (matches Step 2.6v3 posterior mean sigma median). Normalize and importance-weight.

4. Persist full (logk, loga, logt, weight) chain per patient to
   chains_is/PATNO_XXXX.parquet, plus summary row to
   phase2_coupled_is_step26v4.csv.

Gate criteria (Step 2.6v4 acceptance)
-------------------------------------
(a) Median ESS fraction >= 15% across cohort (data is meaningfully informative)
(b) Median cor(log k_n, log alpha_tox) <= -0.50 on high-ESS subset (ESS < 20%)
    -- demonstrates sloppy-ridge structure predicted by the T_tox reframe
(c) High-ESS-subset median T_tox posterior SD <= 0.5 log10 decades
(d) HONESTY gate: report implied %/yr distribution as-is, do NOT filter
    to match Fearnley & Lees 1991. If distribution runs high, that is a
    real scientific finding to investigate in Step 2.8 (Phase 1 vs Phase 2
    head-to-head) and paper7, not rationalize away.

Citations
---------
Liu & Chen 1998 J Am Stat Assoc 93:1032  -- importance sampling / ESS
Vehtari et al. 2017 Stat Comput 27:1413   -- PSIS diagnostics (k-hat)
Raue et al. 2009 Bioinformatics 25:1923   -- profile-likelihood framework
Fearnley & Lees 1991 Brain 114:2283       -- 2-5%/yr canonical range
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Reproducibility header helper — LOCKED 2026-04-09 by closed-loop v1.0
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _reproducibility import capture_provenance, write_run_manifest  # noqa: E402

# ------------------------------------------------------------------
# Configuration (matches calibrate_phase2_coupled.jl Variant B)
# ------------------------------------------------------------------
REPO_ROOT    = Path(__file__).resolve().parents[2]
DAT_PATH     = REPO_ROOT / "outputs/mechanistic_twin/data/dat_spect_longitudinal.parquet"
PROG_CSV     = REPO_ROOT / "outputs/mechanistic_twin/data/posteriors/phase2_coupled_progress_step26v3_ttox_full.csv"
OUT_DIR      = REPO_ROOT / "outputs/mechanistic_twin/phase2"
POST_DIR     = REPO_ROOT / "outputs/mechanistic_twin/data/posteriors"
CHAINS_IS_DIR = POST_DIR / "chains_is"
OUT_DIR.mkdir(parents=True, exist_ok=True)
CHAINS_IS_DIR.mkdir(parents=True, exist_ok=True)

# Priors (calibrate_phase2_coupled.jl lines 187-188)
PRIOR_MU_KN = np.log(1e-4);    PRIOR_SD_KN = 1.5
PRIOR_MU_AL = np.log(1.8e-5);  PRIOR_SD_AL = 2.0

# ODE constants (pinned, Variant B)
K_PROD    = 0.1
K_CLEAR_M = 0.05
K_CONV    = 0.001
K_CLEAR_O = 0.003
K_AGE     = 0.0
M_SS      = K_PROD / K_CLEAR_M
GAMMA     = 0.7
HR_PER_YR = 8766.0
T_TOX_CONST = M_SS**2 / (K_CONV + K_CLEAR_O)

# Observation sigma (Step 2.6v3 posterior mean median ~0.20)
SBR_SIGMA = 0.20

# IS configuration
N_PRIOR = 50_000
RNG_SEED = 202604091
# Save only a thinned weighted subsample per patient to keep disk use modest
SAVE_N_PER_PATIENT = 5_000


def is_posterior_one_patient(patno: int, t_obs: np.ndarray, sbr_obs: np.ndarray,
                               k_n_draws: np.ndarray, alpha_draws: np.ndarray,
                               O_ss_draws: np.ndarray, log_prior_kn: np.ndarray,
                               log_prior_al: np.ndarray) -> dict:
    """Return summary dict + the weighted samples for one patient."""
    SBR_0 = float(sbr_obs[0])
    decay_hr = alpha_draws * O_ss_draws + K_AGE
    t_hr = t_obs * HR_PER_YR

    log_ratio = -decay_hr[:, None] * t_hr[None, :]
    log_ratio = np.clip(log_ratio, -50.0, 0.0)
    sbr_pred = SBR_0 * np.exp(GAMMA * log_ratio)

    resid = sbr_pred - sbr_obs[None, :]
    log_lik = -0.5 * np.sum((resid / SBR_SIGMA) ** 2, axis=1)

    # Stabilized softmax weights
    log_lik_shift = log_lik - np.max(log_lik)
    w = np.exp(log_lik_shift)
    w_sum = w.sum()
    if not np.isfinite(w_sum) or w_sum <= 0:
        return {"patno": patno, "ess": 0.0, "error": "degenerate-weights"}
    w /= w_sum
    ess = float(1.0 / np.sum(w ** 2))

    # Convergence diagnostic: IS ESS fraction (Liu & Chen 1998). This is the
    # IS-native analog to MCMC R̂ / ESS. Per Vehtari, Gelman, Gabry 2017 §2.2,
    # the primary reliability diagnostic for IS is ESS / N_draws; Pareto-
    # smoothed k̂ is a supplementary diagnostic. We opted not to ship k̂ in
    # this script because a correct implementation (Zhang & Stephens 2009)
    # requires `arviz` or a hand-rolled MLE that was not available in-venv
    # on 2026-04-09, and a naive moment-based k̂ proxy failed pytest
    # correctness tests. The ESS-fraction stratification (HIGH <20%, MOD
    # 20-50%, LOW ≥50%) is the canonical diagnostic for this pipeline and
    # is what paper7 §3.5.3 reports.

    logk = np.log(k_n_draws)
    loga = np.log(alpha_draws)
    T_tox = alpha_draws * k_n_draws * T_TOX_CONST  # hr^-1
    logt = np.log(T_tox)

    def _wmean(x):
        return float(np.sum(w * x))

    def _wvar(x):
        m = _wmean(x)
        return float(np.sum(w * (x - m) ** 2))

    def _wq(x, q):
        order = np.argsort(x)
        cw = np.cumsum(w[order])
        idx = int(np.searchsorted(cw, q))
        idx = min(max(idx, 0), len(x) - 1)
        return float(x[order[idx]])

    sd_logk = float(np.sqrt(_wvar(logk)))
    sd_loga = float(np.sqrt(_wvar(loga)))
    sd_logt = float(np.sqrt(_wvar(logt)))

    mk, ma = _wmean(logk), _wmean(loga)
    cov_ka = float(np.sum(w * (logk - mk) * (loga - ma)))
    cor_ka = cov_ka / (sd_logk * sd_loga + 1e-30)

    pct_loss = (1.0 - np.exp(-T_tox * HR_PER_YR)) * 100.0

    return {
        "PATNO": patno,
        "ess": ess,
        "ess_frac": ess / N_PRIOR,
        # k_n posterior (weighted)
        "k_n_mean": float(np.exp(_wmean(logk))),
        "k_n_median": float(np.exp(_wq(logk, 0.5))),
        "k_n_q025": float(np.exp(_wq(logk, 0.025))),
        "k_n_q975": float(np.exp(_wq(logk, 0.975))),
        "log_k_n_sd": sd_logk,
        "log_k_n_sd_prior_ratio": sd_logk / PRIOR_SD_KN,
        # alpha_tox posterior (weighted)
        "alpha_tox_mean": float(np.exp(_wmean(loga))),
        "alpha_tox_median": float(np.exp(_wq(loga, 0.5))),
        "alpha_tox_q025": float(np.exp(_wq(loga, 0.025))),
        "alpha_tox_q975": float(np.exp(_wq(loga, 0.975))),
        "log_alpha_tox_sd": sd_loga,
        "log_alpha_tox_sd_prior_ratio": sd_loga / PRIOR_SD_AL,
        # T_tox posterior (weighted)
        "T_tox_mean": float(np.exp(_wmean(logt))),
        "T_tox_median": float(np.exp(_wq(logt, 0.5))),
        "T_tox_q025": float(np.exp(_wq(logt, 0.025))),
        "T_tox_q975": float(np.exp(_wq(logt, 0.975))),
        "log_T_tox_sd": sd_logt,
        "log_T_tox_sd_log10": sd_logt / np.log(10),
        # Sloppy-ridge diagnostic
        "cor_logk_logalpha": cor_ka,
        # Neuron-loss implied rate (weighted quantiles of the %/yr transform)
        "pct_loss_per_yr_median": float(_wq(pct_loss, 0.5)),
        "pct_loss_per_yr_q025": float(_wq(pct_loss, 0.025)),
        "pct_loss_per_yr_q975": float(_wq(pct_loss, 0.975)),
        # Raw weighted samples (downstream can resample)
        "_weights": w,
        "_logk": logk,
        "_loga": loga,
        "_logt": logt,
    }


def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


def main() -> int:
    # ------------------------------------------------------------------
    # Reproducibility header — LOCKED by closed-loop v1.0 (2026-04-09)
    # ------------------------------------------------------------------
    provenance = capture_provenance(
        script_path=Path(__file__).resolve(),
        repo_root=REPO_ROOT,
        input_files=[DAT_PATH, PROG_CSV],
        extra={
            "rng_seed": RNG_SEED,
            "n_prior_samples": N_PRIOR,
            "save_n_per_patient": SAVE_N_PER_PATIENT,
            "sbr_sigma": SBR_SIGMA,
            "prior": {
                "k_n_logmean": float(PRIOR_MU_KN), "k_n_logsd": float(PRIOR_SD_KN),
                "alpha_tox_logmean": float(PRIOR_MU_AL), "alpha_tox_logsd": float(PRIOR_SD_AL),
            },
            "ode_constants": {
                "K_PROD": K_PROD, "K_CLEAR_M": K_CLEAR_M, "K_CONV": K_CONV,
                "K_CLEAR_O": K_CLEAR_O, "K_AGE": K_AGE, "M_SS": M_SS,
                "GAMMA": GAMMA, "T_TOX_CONST": T_TOX_CONST,
            },
        },
    )

    print("=" * 76)
    print("Phase 2 Step 2.6v4 — Importance-weighted posterior over (k_n, alpha_tox, T_tox)")
    print("=" * 76)
    git = provenance["git"]
    print(f"Git:     {git.get('sha','?')}  {'(DIRTY)' if git.get('dirty') else '(clean)'}  "
          f"branch={git.get('branch','?')}")
    print(f"Python:  {provenance['python']['version']}  "
          f"numpy={provenance['packages']['numpy']}  "
          f"pandas={provenance['packages']['pandas']}")
    print(f"Script:  {provenance['script']['sha256'][:16]}...")
    for f in provenance["input_files"]:
        if f.get("status") == "MISSING":
            print(f"Input:   {f['path']} MISSING")
        else:
            print(f"Input:   {f['path']} rows={f.get('row_count','?')} "
                  f"sha={f['sha256'][:16]}...")
    print()
    print(f"Prior k_n       ~ LogNormal({PRIOR_MU_KN:+.3f}, {PRIOR_SD_KN})   "
          f"-> median 1e-4, 95% CI [{np.exp(PRIOR_MU_KN-1.96*PRIOR_SD_KN):.1e}, "
          f"{np.exp(PRIOR_MU_KN+1.96*PRIOR_SD_KN):.1e}]")
    print(f"Prior alpha_tox ~ LogNormal({PRIOR_MU_AL:+.3f}, {PRIOR_SD_AL})   "
          f"-> median 1.8e-5, 95% CI [{np.exp(PRIOR_MU_AL-1.96*PRIOR_SD_AL):.1e}, "
          f"{np.exp(PRIOR_MU_AL+1.96*PRIOR_SD_AL):.1e}]")
    print(f"N_prior samples per patient: {N_PRIOR}")
    print(f"Observation sigma:           {SBR_SIGMA}")
    print()

    dat = pd.read_parquet(DAT_PATH)
    prog = pd.read_csv(PROG_CSV)
    # Sort patient list for deterministic run order — reproducibility-critical
    # because per-patient rng.choice() resampling consumes rng state.
    wave_a_patients = sorted(prog.PATNO.astype(int).tolist())
    print(f"Patients in Step 2.6v3 Wave A progress CSV: {len(wave_a_patients)} (sorted)")
    print()

    # Pre-draw the prior once (shared across all patients for efficiency,
    # since the prior is patient-independent)
    rng = np.random.default_rng(RNG_SEED)
    k_n_draws   = np.exp(rng.normal(PRIOR_MU_KN, PRIOR_SD_KN, N_PRIOR))
    alpha_draws = np.exp(rng.normal(PRIOR_MU_AL, PRIOR_SD_AL, N_PRIOR))
    O_ss_draws  = k_n_draws * M_SS ** 2 / (K_CONV + K_CLEAR_O)
    log_prior_kn = -np.log(k_n_draws) - 0.5 * ((np.log(k_n_draws) - PRIOR_MU_KN) / PRIOR_SD_KN) ** 2
    log_prior_al = -np.log(alpha_draws) - 0.5 * ((np.log(alpha_draws) - PRIOR_MU_AL) / PRIOR_SD_AL) ** 2

    # Run IS per patient
    rows = []
    skipped = []
    t_start = _now_utc()
    for i, patno in enumerate(wave_a_patients, start=1):
        pat = dat[dat.PATNO == patno].sort_values("t_years")
        if len(pat) < 2:
            skipped.append({"patno": patno, "reason": "<2-scans"})
            continue
        t_obs = pat.t_years.to_numpy(dtype=float)
        sbr_obs = pat.sbr_putamen_mean.to_numpy(dtype=float)
        if np.any(~np.isfinite(t_obs)) or np.any(~np.isfinite(sbr_obs)):
            skipped.append({"patno": patno, "reason": "non-finite"})
            continue

        res = is_posterior_one_patient(
            patno, t_obs, sbr_obs,
            k_n_draws, alpha_draws, O_ss_draws,
            log_prior_kn, log_prior_al,
        )
        if "error" in res:
            skipped.append({"patno": patno, "reason": res["error"]})
            continue

        # Persist a thinned weighted sample (resampling step)
        w = res.pop("_weights")
        logk = res.pop("_logk")
        loga = res.pop("_loga")
        logt = res.pop("_logt")
        idx = rng.choice(N_PRIOR, size=SAVE_N_PER_PATIENT, replace=True, p=w)
        pd.DataFrame({
            "k_n": np.exp(logk[idx]),
            "alpha_tox": np.exp(loga[idx]),
            "T_tox": np.exp(logt[idx]),
        }).to_parquet(CHAINS_IS_DIR / f"PATNO_{patno}.parquet")

        res["n_scans"] = int(len(pat))
        rows.append(res)

        if i % 25 == 0 or i == len(wave_a_patients):
            elapsed = (_now_utc() - t_start).total_seconds()
            rate = i / elapsed if elapsed > 0 else 0.0
            eta = (len(wave_a_patients) - i) / rate if rate > 0 else 0.0
            print(f"  [{i:3d}/{len(wave_a_patients)}] ESS_frac="
                  f"{res['ess_frac']:.2%}, cor={res['cor_logk_logalpha']:+.2f}, "
                  f"T_tox SD={res['log_T_tox_sd_log10']:.2f} decades, "
                  f"%/yr median={res['pct_loss_per_yr_median']:.2f}  "
                  f"({rate:.1f} pt/s, ETA {eta:.0f}s)")

    df = pd.DataFrame(rows).sort_values("PATNO").reset_index(drop=True)
    print()
    print(f"Completed {len(df)} / {len(wave_a_patients)} patients. Skipped: {len(skipped)}")

    # -----------------------------------------------------
    # Summary + gate evaluation
    # -----------------------------------------------------
    print()
    print("=" * 76)
    print("COHORT SUMMARY")
    print("=" * 76)
    print(f"ESS fraction          median = {df.ess_frac.median():.2%}  "
          f"p25-p75 = [{df.ess_frac.quantile(0.25):.2%}, {df.ess_frac.quantile(0.75):.2%}]")
    print(f"cor(log k_n, log α)   median = {df.cor_logk_logalpha.median():+.3f}  "
          f"p25-p75 = [{df.cor_logk_logalpha.quantile(0.25):+.3f}, "
          f"{df.cor_logk_logalpha.quantile(0.75):+.3f}]")
    print(f"T_tox SD (log10)      median = {df.log_T_tox_sd_log10.median():.2f} dec  "
          f"p25-p75 = [{df.log_T_tox_sd_log10.quantile(0.25):.2f}, "
          f"{df.log_T_tox_sd_log10.quantile(0.75):.2f}]")
    print(f"α_tox SD / prior SD   median = {df.log_alpha_tox_sd_prior_ratio.median():.3f}")
    print(f"k_n   SD / prior SD   median = {df.log_k_n_sd_prior_ratio.median():.3f}")
    print(f"Implied %/yr (median) median = {df.pct_loss_per_yr_median.median():.2f}%/yr  "
          f"p25-p75 = [{df.pct_loss_per_yr_median.quantile(0.25):.2f}, "
          f"{df.pct_loss_per_yr_median.quantile(0.75):.2f}]")
    print()

    # Stratify by ESS fraction (informative = ESS < 20% of prior = likelihood rejects >80%)
    high_info = df[df.ess_frac < 0.20]
    mod_info  = df[(df.ess_frac >= 0.20) & (df.ess_frac < 0.50)]
    low_info  = df[df.ess_frac >= 0.50]
    print(f"HIGH-INFO (ESS<20%):   N={len(high_info):3d}  "
          f"cor median={high_info.cor_logk_logalpha.median():+.3f}  "
          f"T_tox SD={high_info.log_T_tox_sd_log10.median():.2f} dec  "
          f"%/yr={high_info.pct_loss_per_yr_median.median():.2f}")
    print(f"MOD-INFO  (20<=ESS<50%): N={len(mod_info):3d}  "
          f"cor median={mod_info.cor_logk_logalpha.median():+.3f}  "
          f"T_tox SD={mod_info.log_T_tox_sd_log10.median():.2f} dec  "
          f"%/yr={mod_info.pct_loss_per_yr_median.median():.2f}")
    print(f"LOW-INFO  (ESS>=50%):  N={len(low_info):3d}  "
          f"cor median={low_info.cor_logk_logalpha.median():+.3f}  "
          f"T_tox SD={low_info.log_T_tox_sd_log10.median():.2f} dec  "
          f"%/yr={low_info.pct_loss_per_yr_median.median():.2f}")
    print()

    # Gate evaluation
    gate_a = df.ess_frac.median() >= 0.15
    gate_b = (high_info.cor_logk_logalpha.median() <= -0.50) if len(high_info) > 0 else False
    gate_c = (high_info.log_T_tox_sd_log10.median() <= 0.5) if len(high_info) > 0 else False
    gates_all = gate_a and gate_b and gate_c
    print("=" * 76)
    print("STEP 2.6v4 GATE EVALUATION")
    print("=" * 76)
    print(f"(a) median ESS fraction >= 15%:                       "
          f"{'PASS' if gate_a else 'FAIL'} ({df.ess_frac.median():.2%})")
    print(f"(b) HIGH-INFO subset median cor <= -0.50:             "
          f"{'PASS' if gate_b else 'FAIL'} "
          f"({high_info.cor_logk_logalpha.median():+.3f} on N={len(high_info)})")
    print(f"(c) HIGH-INFO subset median T_tox SD <= 0.5 decades:  "
          f"{'PASS' if gate_c else 'FAIL'} "
          f"({high_info.log_T_tox_sd_log10.median():.2f} on N={len(high_info)})")
    print(f"OVERALL: {'ALL GATES PASS' if gates_all else 'ONE OR MORE GATES FAIL'}")
    print()

    # Persist
    out_csv = POST_DIR / "phase2_coupled_is_step26v4.csv"
    df.to_csv(out_csv, index=False)
    print(f"Per-patient summary: {out_csv}")

    summary = {
        "_provenance": provenance,
        "datetime_utc": _now_utc().isoformat(),
        "n_patients": int(len(df)),
        "n_skipped": int(len(skipped)),
        "n_prior_samples": int(N_PRIOR),
        "sbr_sigma": float(SBR_SIGMA),
        "prior": {
            "k_n_logmean": float(PRIOR_MU_KN), "k_n_logsd": float(PRIOR_SD_KN),
            "alpha_tox_logmean": float(PRIOR_MU_AL), "alpha_tox_logsd": float(PRIOR_SD_AL),
        },
        "constants": {
            "K_PROD": K_PROD, "K_CLEAR_M": K_CLEAR_M, "K_CONV": K_CONV,
            "K_CLEAR_O": K_CLEAR_O, "K_AGE": K_AGE, "M_SS": M_SS,
            "GAMMA": GAMMA, "T_TOX_CONST": T_TOX_CONST,
        },
        "medians": {
            "ess_frac": float(df.ess_frac.median()),
            "cor_logk_logalpha": float(df.cor_logk_logalpha.median()),
            "log_T_tox_sd_log10": float(df.log_T_tox_sd_log10.median()),
            "log_alpha_tox_sd_prior_ratio": float(df.log_alpha_tox_sd_prior_ratio.median()),
            "log_k_n_sd_prior_ratio": float(df.log_k_n_sd_prior_ratio.median()),
            "pct_loss_per_yr_median": float(df.pct_loss_per_yr_median.median()),
        },
        "stratified": {
            "high_info_N":   int(len(high_info)),
            "high_info_cor": float(high_info.cor_logk_logalpha.median()) if len(high_info) > 0 else None,
            "high_info_T_tox_sd": float(high_info.log_T_tox_sd_log10.median()) if len(high_info) > 0 else None,
            "high_info_pct_yr":    float(high_info.pct_loss_per_yr_median.median()) if len(high_info) > 0 else None,
            "mod_info_N":    int(len(mod_info)),
            "low_info_N":    int(len(low_info)),
        },
        "gates": {
            "gate_a_ess_ge_15pct": bool(gate_a),
            "gate_b_high_info_cor_le_m050": bool(gate_b),
            "gate_c_high_info_T_tox_sd_le_05dec": bool(gate_c),
            "overall_pass": bool(gates_all),
        },
        "skipped": skipped,
    }
    summary_json = OUT_DIR / "step_2_6_v4_is_summary.json"
    summary_json.write_text(json.dumps(summary, indent=2))
    print(f"Summary JSON:         {summary_json}")
    print(f"Per-patient chains:   {CHAINS_IS_DIR}/PATNO_*.parquet  ({len(df)} files, "
          f"{SAVE_N_PER_PATIENT} weighted resamples each)")

    # -----------------------------------------------------
    # Diagnostic figure (2x2)
    # -----------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))

    ax = axes[0, 0]
    ax.hist(df.ess_frac, bins=50, color="tab:blue", alpha=0.75, edgecolor="k")
    ax.axvline(0.20, color="tab:red", linestyle="--", alpha=0.7, label="20% (high-info threshold)")
    ax.axvline(df.ess_frac.median(), color="tab:orange", linestyle="-", alpha=0.8,
               label=f"median = {df.ess_frac.median():.2%}")
    ax.set_xlabel("ESS fraction")
    ax.set_ylabel("# patients")
    ax.set_title("(a) Importance-sampling ESS fraction per patient\nLower = data more informative")
    ax.legend(fontsize=9)

    ax = axes[0, 1]
    ax.scatter(df.ess_frac, df.cor_logk_logalpha, s=14, alpha=0.6, color="tab:purple")
    ax.axhline(0, color="k", linestyle="-", alpha=0.3)
    ax.axhline(-0.5, color="tab:red", linestyle="--", alpha=0.6, label="-0.5 sloppy-ridge threshold")
    ax.set_xlabel("ESS fraction")
    ax.set_ylabel("cor(log k_n, log α_tox)")
    ax.set_title("(b) Sloppy-ridge signature vs likelihood informativeness")
    ax.legend(fontsize=9)

    ax = axes[1, 0]
    ax.scatter(df.log_alpha_tox_sd / np.log(10), df.log_T_tox_sd_log10,
               s=14, alpha=0.6,
               c=df.ess_frac, cmap="viridis_r")
    lim = max(df.log_alpha_tox_sd.max() / np.log(10), df.log_T_tox_sd_log10.max())
    ax.plot([0, lim], [0, lim], "k--", alpha=0.4, label="y = x (equal)")
    ax.set_xlabel("α_tox posterior SD (log10 decades)")
    ax.set_ylabel("T_tox posterior SD (log10 decades)")
    ax.set_title("(c) T_tox vs α_tox posterior widths\nBelow y=x = T_tox tighter (stiff direction)")
    plt.colorbar(ax.collections[0], ax=ax, label="ESS fraction")
    ax.legend(fontsize=9)

    ax = axes[1, 1]
    bins = np.logspace(np.log10(0.01), np.log10(200), 60)
    ax.hist(high_info.pct_loss_per_yr_median, bins=bins, alpha=0.65,
            color="tab:blue", label=f"HIGH-INFO (N={len(high_info)})")
    ax.hist(mod_info.pct_loss_per_yr_median, bins=bins, alpha=0.55,
            color="tab:orange", label=f"MOD-INFO (N={len(mod_info)})")
    ax.hist(low_info.pct_loss_per_yr_median, bins=bins, alpha=0.45,
            color="tab:red", label=f"LOW-INFO (N={len(low_info)})")
    ax.axvspan(2, 5, alpha=0.2, color="tab:green",
               label="Fearnley & Lees 1991\n2-5%/yr canonical")
    ax.set_xscale("log")
    ax.set_xlabel("Implied neuron loss rate (%/yr), weighted median")
    ax.set_ylabel("# patients")
    ax.set_title("(d) T_tox-implied neuron loss stratified by ESS")
    ax.legend(fontsize=8)

    fig.suptitle(f"Phase 2 Step 2.6v4 — IS-weighted posterior (N = {len(df)})",
                 fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig_path = OUT_DIR / "step_2_6_v4_is_diagnostic.png"
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    print(f"Figure:               {fig_path}")

    # ------------------------------------------------------------------
    # Reproducibility lock-in: hash all output files + write RUN_MANIFEST
    # ------------------------------------------------------------------
    import hashlib as _hashlib

    def _sha256(p: Path) -> str:
        h = _hashlib.sha256()
        with p.open("rb") as f:
            for chunk in iter(lambda: f.read(2**20), b""):
                h.update(chunk)
        return h.hexdigest()

    # Bitwise-reproducibility anchor: sort chain files by PATNO, hash each,
    # hash the concatenation. Same-seed reruns must produce the same combined hash.
    chain_files = sorted(CHAINS_IS_DIR.glob("PATNO_*.parquet"))
    per_chain_hashes = {cf.name: _sha256(cf) for cf in chain_files}
    combined_hash = _hashlib.sha256(
        "\n".join(f"{k}:{v}" for k, v in per_chain_hashes.items()).encode()
    ).hexdigest()

    output_hashes = {
        "phase2_coupled_is_step26v4.csv": _sha256(out_csv),
        "step_2_6_v4_is_diagnostic.png": _sha256(fig_path),
        "chains_is_combined_sha256": combined_hash,
        "n_chain_files": len(chain_files),
    }
    summary["_output_hashes"] = output_hashes

    # Re-write summary JSON with output hashes embedded
    summary_json.write_text(json.dumps(summary, indent=2, default=str))
    # Now compute the summary JSON's own hash (after the final write)
    output_hashes["step_2_6_v4_is_summary.json"] = _sha256(summary_json)

    # Write the companion RUN_MANIFEST.md
    manifest_path = OUT_DIR / "step_2_6_v4_RUN_MANIFEST.md"
    write_run_manifest(
        manifest_path=manifest_path,
        step_name="Phase 2 Step 2.6v4 — IS-weighted posterior",
        provenance=provenance,
        gate_results={
            "gate_a_ess_ge_15pct": gate_a,
            "gate_b_high_info_cor_le_m050": gate_b,
            "gate_c_high_info_T_tox_sd_le_05dec": gate_c,
            "overall_pass": gates_all,
        },
        summary_metrics={
            "n_patients": len(df),
            "median_ess_frac": f"{df.ess_frac.median():.4f}",
            "median_cor_logk_logalpha": f"{df.cor_logk_logalpha.median():+.4f}",
            "median_T_tox_sd_log10_decades": f"{df.log_T_tox_sd_log10.median():.4f}",
            "median_implied_pct_yr": f"{df.pct_loss_per_yr_median.median():.4f}",
            "high_info_N": len(high_info),
            "high_info_cor": f"{high_info.cor_logk_logalpha.median():+.4f}" if len(high_info) > 0 else "N/A",
            "high_info_T_tox_sd_log10": f"{high_info.log_T_tox_sd_log10.median():.4f}" if len(high_info) > 0 else "N/A",
            "high_info_pct_yr": f"{high_info.pct_loss_per_yr_median.median():.4f}" if len(high_info) > 0 else "N/A",
            "chains_is_combined_sha256": combined_hash,
            "n_chain_files": len(chain_files),
        },
    )
    print(f"RUN_MANIFEST:         {manifest_path}")
    print(f"Combined chains SHA:  {combined_hash[:16]}...  ({len(chain_files)} files)")

    return 0 if gates_all else 2


if __name__ == "__main__":
    raise SystemExit(main())
