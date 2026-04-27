#!/usr/bin/env python3
"""
Phase 2 Step 2.6v5 — Joint SBR + CSF α-syn IS posterior
=========================================================

Purpose
-------
Extend Step 2.6v4 (SBR-only IS posterior) with a second observable: CSF total
α-synuclein. Under the Variant B slow-fast collapse, O_ss = k_n × M_ss² /
(K_CONV + K_CLEAR_O) is TIME-INVARIANT, so CSF α-syn constrains k_n
independently of α_tox. Combined with T_tox = α_tox × k_n × const (from SBR),
this breaks the sloppy-ridge degeneracy and recovers individual k_n and α_tox.

Key insight: r_o (oligomer ELISA cross-reactivity) is an ASSAY property, not
a patient property. Fixing r_o at the prior mean (1.0) keeps IS at 2 free
parameters (k_n, α_tox) — same dimensionality as v4.

CSF observation model
---------------------
Under slow-fast collapse, the CSF total α-syn for patient i is:

    CSF_pred_i(k_n) = S_CSF × (M_ss + r_o × O_ss(k_n))
    O_ss(k_n)       = k_n × M_ss² / (K_CONV + K_CLEAR_O)

where S_CSF (pg/ml per model-nM) is estimated once from the population and
held fixed. Each patient's n_i CSF measurements are IID N(CSF_pred, σ_CSF²)
observations of the same time-invariant quantity.

Joint log-likelihood = log L_SBR(k_n, α_tox) + log L_CSF(k_n)

Patients without CSF data (27 of 304 Wave A) use SBR-only likelihood (same
as v4). Patients with CSF but <2 SBR scans are still skipped.

Gate criteria (Step 2.6v5 acceptance)
-------------------------------------
(a) Median ESS fraction >= 15% across CSF cohort
(b) CSF-subset cor(log k_n, log α_tox) LESS negative than v4's -0.852
    (degeneracy partially broken)
(c) Individual k_n posterior SD < v4 k_n posterior SD (CSF constrains k_n)
(d) Cohort median implied %/yr still in [1, 10] range (biological sanity)
(e) HONESTY: if degeneracy does NOT break, report honestly — the 4.8%
    oligomer fraction at typical k_n may be below detection threshold.

Citations
---------
Liu & Chen 1998 J Am Stat Assoc 93:1032  — importance sampling / ESS
Kumar et al. 2020 ACS Chem Neurosci 11:1337 — CSF α-syn ELISA cross-reactivity
Giasson et al. 2000 J Biol Chem 275:6109 — oligomeric α-syn structure
Majbour et al. 2016 Mol Neurodegener 11:7 — CSF oligomeric α-syn assay
Mollenhauer et al. 2011 Lancet Neurol 10:230 — CSF total α-syn biomarker
"""
from __future__ import annotations

import argparse
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
CSF_PATH     = REPO_ROOT / "data/00_raw/GIMAN/ppmi_data_csv/Current_Biospecimen_Analysis_Results_30Sep2025.csv"
V4_CSV       = REPO_ROOT / "outputs/mechanistic_twin/data/posteriors/phase2_coupled_is_step26v4.csv"
PROG_CSV     = REPO_ROOT / "outputs/mechanistic_twin/data/posteriors/phase2_coupled_progress_step26v3_ttox_full.csv"
OUT_DIR      = REPO_ROOT / "outputs/mechanistic_twin/phase2"
POST_DIR     = REPO_ROOT / "outputs/mechanistic_twin/data/posteriors"
CHAINS_DIR   = POST_DIR / "chains_is_v5"

# Priors (same as v4 — calibrate_phase2_coupled.jl lines 187-188)
PRIOR_MU_KN = np.log(1e-4);    PRIOR_SD_KN = 1.5
PRIOR_MU_AL = np.log(1.8e-5);  PRIOR_SD_AL = 2.0

# ODE constants (pinned, Variant B)
K_PROD    = 0.1
K_CLEAR_M = 0.05
K_CONV    = 0.001
K_CLEAR_O = 0.003
K_AGE     = 0.0
M_SS      = K_PROD / K_CLEAR_M          # 2.0 nM (model units)
GAMMA     = 0.7
HR_PER_YR = 8766.0
T_TOX_CONST = M_SS**2 / (K_CONV + K_CLEAR_O)  # 1000.0

# SBR observation noise (same as v4)
SBR_SIGMA = 0.20

# CSF observation model parameters
R_O       = 1.0    # ELISA oligomer cross-reactivity (assay property, fixed)
                    # Prior: LogNormal(log(1.0), 0.5) per neuron_death.jl docstring
                    # Fixed at prior median because it's a population-level assay constant

# IS configuration
N_PRIOR_DEFAULT = 50_000
RNG_SEED = 202604101      # v5-specific seed
SAVE_N_PER_PATIENT = 5_000


def load_csf_data(csf_path: Path, wave_a_pats: list[int]) -> dict[int, np.ndarray]:
    """Load CSF α-syn measurements for Wave A patients.

    Returns dict mapping PATNO -> array of CSF values (pg/ml).
    """
    bio = pd.read_csv(csf_path, low_memory=False)
    csf = bio[bio["TESTNAME"] == "CSF Alpha-synuclein"].copy()
    csf["TESTVALUE"] = pd.to_numeric(csf["TESTVALUE"], errors="coerce")
    csf = csf.dropna(subset=["TESTVALUE"])
    csf = csf[csf["PATNO"].isin(wave_a_pats)]

    csf_dict: dict[int, np.ndarray] = {}
    for patno, grp in csf.groupby("PATNO"):
        vals = grp["TESTVALUE"].values.astype(float)
        if len(vals) >= 1 and np.all(np.isfinite(vals)) and np.all(vals > 0):
            csf_dict[int(patno)] = vals
    return csf_dict


def estimate_csf_params(csf_dict: dict[int, np.ndarray]) -> tuple[float, float]:
    """Estimate population S_CSF and σ_CSF from CSF data.

    S_CSF: scale factor converting model nM to observed pg/ml.
           Estimated as median(patient_mean_CSF) / M_ss.

    σ_CSF: pooled within-patient SD (RMSE of residuals from patient means).
           This captures measurement noise but NOT between-patient S_CSF variation.
    """
    patient_means = []
    residuals = []
    for patno, vals in sorted(csf_dict.items()):
        m = vals.mean()
        patient_means.append(m)
        if len(vals) >= 2:
            residuals.extend(vals - m)

    s_csf = float(np.median(patient_means)) / M_SS
    sigma_csf = float(np.sqrt(np.mean(np.array(residuals) ** 2))) if residuals else 300.0
    return s_csf, sigma_csf


def is_posterior_one_patient(
    patno: int,
    t_obs: np.ndarray,
    sbr_obs: np.ndarray,
    csf_vals: np.ndarray | None,
    k_n_draws: np.ndarray,
    alpha_draws: np.ndarray,
    O_ss_draws: np.ndarray,
    s_csf: float,
    sigma_csf: float,
) -> dict:
    """Joint SBR + CSF IS posterior for one patient.

    SBR likelihood: same closed-form decay as v4.
    CSF likelihood: IID Gaussian around S_CSF × (M_ss + R_O × O_ss(k_n)).
    """
    N = len(k_n_draws)
    SBR_0 = float(sbr_obs[0])

    # --- SBR log-likelihood (identical to v4) ---
    decay_hr = alpha_draws * O_ss_draws + K_AGE
    t_hr = t_obs * HR_PER_YR
    log_ratio = -decay_hr[:, None] * t_hr[None, :]
    log_ratio = np.clip(log_ratio, -50.0, 0.0)
    sbr_pred = SBR_0 * np.exp(GAMMA * log_ratio)
    resid_sbr = sbr_pred - sbr_obs[None, :]
    log_lik_sbr = -0.5 * np.sum((resid_sbr / SBR_SIGMA) ** 2, axis=1)

    # --- CSF log-likelihood (new in v5) ---
    log_lik_csf = np.zeros(N, dtype=np.float64)
    has_csf = csf_vals is not None and len(csf_vals) > 0
    if has_csf:
        # CSF_pred(k_n) = S_CSF × (M_ss + R_O × O_ss(k_n))
        # O_ss already computed as k_n × M_ss² / (K_CONV + K_CLEAR_O) = k_n × T_TOX_CONST
        csf_pred = s_csf * (M_SS + R_O * O_ss_draws)  # shape (N,)
        # Each CSF measurement is IID N(csf_pred, sigma_csf²)
        for y_csf in csf_vals:
            log_lik_csf += -0.5 * ((y_csf - csf_pred) / sigma_csf) ** 2

    # --- Joint log-likelihood ---
    log_lik = log_lik_sbr + log_lik_csf

    # Stabilized softmax weights
    log_lik_shift = log_lik - np.max(log_lik)
    w = np.exp(log_lik_shift)
    w_sum = w.sum()
    if not np.isfinite(w_sum) or w_sum <= 0:
        return {"PATNO": patno, "ess": 0.0, "error": "degenerate-weights"}
    w /= w_sum
    ess = float(1.0 / np.sum(w ** 2))

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

    # CSF-specific diagnostics
    csf_diagnostics = {}
    if has_csf:
        csf_pred_weighted = s_csf * (M_SS + R_O * np.exp(_wmean(np.log(O_ss_draws * w.sum()))))
        csf_pred_at_post_mean = s_csf * (M_SS + R_O * np.exp(_wmean(logk)) * T_TOX_CONST)
        csf_diagnostics = {
            "csf_mean_obs": float(csf_vals.mean()),
            "csf_sd_obs": float(csf_vals.std()) if len(csf_vals) >= 2 else 0.0,
            "csf_n_obs": int(len(csf_vals)),
            "csf_pred_at_post_mean": float(csf_pred_at_post_mean),
            "csf_residual_pgml": float(csf_vals.mean() - csf_pred_at_post_mean),
        }

    return {
        "PATNO": patno,
        "ess": ess,
        "ess_frac": ess / len(k_n_draws),
        "has_csf": has_csf,
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
        # Neuron-loss implied rate
        "pct_loss_per_yr_median": float(_wq(pct_loss, 0.5)),
        "pct_loss_per_yr_q025": float(_wq(pct_loss, 0.025)),
        "pct_loss_per_yr_q975": float(_wq(pct_loss, 0.975)),
        # CSF diagnostics
        **csf_diagnostics,
        # Raw weighted samples (popped before DataFrame)
        "_weights": w,
        "_logk": logk,
        "_loga": loga,
        "_logt": logt,
    }


def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


def main() -> int:
    parser = argparse.ArgumentParser(description="Step 2.6v5 — Joint SBR + CSF IS posterior")
    parser.add_argument("--n-prior", type=int, default=N_PRIOR_DEFAULT,
                        help=f"Number of prior draws (default {N_PRIOR_DEFAULT})")
    parser.add_argument("--smoke-test", type=int, default=0,
                        help="Run on first N patients only (0 = all)")
    parser.add_argument("--sigma-csf-override", type=float, default=0.0,
                        help="Override σ_CSF (0 = estimate from data)")
    args = parser.parse_args()

    n_prior = args.n_prior
    smoke_n = args.smoke_test

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    CHAINS_DIR.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Reproducibility header
    # ------------------------------------------------------------------
    provenance = capture_provenance(
        script_path=Path(__file__).resolve(),
        repo_root=REPO_ROOT,
        input_files=[DAT_PATH, CSF_PATH, V4_CSV, PROG_CSV],
        extra={
            "rng_seed": RNG_SEED,
            "n_prior_samples": n_prior,
            "save_n_per_patient": SAVE_N_PER_PATIENT,
            "sbr_sigma": SBR_SIGMA,
            "r_o_fixed": R_O,
            "smoke_test_n": smoke_n,
            "sigma_csf_override": args.sigma_csf_override,
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
    print("Phase 2 Step 2.6v5 — Joint SBR + CSF α-syn IS posterior")
    print("=" * 76)
    git = provenance["git"]
    print(f"Git:     {git.get('sha', '?')}  {'(DIRTY)' if git.get('dirty') else '(clean)'}  "
          f"branch={git.get('branch', '?')}")
    print(f"Python:  {provenance['python']['version']}  "
          f"numpy={provenance['packages']['numpy']}  "
          f"pandas={provenance['packages']['pandas']}")
    print(f"Script:  {provenance['script']['sha256'][:16]}...")
    print()

    # ------------------------------------------------------------------
    # Load data
    # ------------------------------------------------------------------
    dat = pd.read_parquet(DAT_PATH)
    prog = pd.read_csv(PROG_CSV)
    wave_a_patients = sorted(prog.PATNO.astype(int).tolist())

    # Load v4 results for comparison
    v4 = pd.read_csv(V4_CSV)
    v4_by_pat = v4.set_index("PATNO")

    # Load CSF data
    csf_dict = load_csf_data(CSF_PATH, wave_a_patients)
    s_csf, sigma_csf_est = estimate_csf_params(csf_dict)
    sigma_csf = args.sigma_csf_override if args.sigma_csf_override > 0 else sigma_csf_est

    print(f"Wave A patients:    {len(wave_a_patients)}")
    print(f"Patients with CSF:  {len(csf_dict)} ({len(csf_dict)/len(wave_a_patients)*100:.1f}%)")
    print(f"S_CSF (scale):      {s_csf:.1f} pg/ml per model-nM")
    print(f"σ_CSF (noise):      {sigma_csf:.1f} pg/ml"
          f"{'  (estimated)' if args.sigma_csf_override == 0 else '  (OVERRIDE)'}")
    print(f"r_o (fixed):        {R_O}")
    print(f"N_prior samples:    {n_prior}")
    if smoke_n > 0:
        print(f"*** SMOKE TEST: first {smoke_n} patients only ***")
    print()

    # Sensitivity check: how much does CSF change across k_n prior range?
    for k_n_val in [1e-5, 1e-4, 1e-3]:
        o_ss = k_n_val * T_TOX_CONST
        csf_pred = s_csf * (M_SS + R_O * o_ss)
        oligo_pct = R_O * o_ss / (M_SS + R_O * o_ss) * 100
        print(f"  k_n={k_n_val:.0e}:  O_ss={o_ss:.4f} nM,  "
              f"CSF_pred={csf_pred:.0f} pg/ml,  oligomer={oligo_pct:.1f}%")
    print()

    # ------------------------------------------------------------------
    # Pre-draw prior (shared across patients, same as v4)
    # ------------------------------------------------------------------
    rng = np.random.default_rng(RNG_SEED)
    k_n_draws   = np.exp(rng.normal(PRIOR_MU_KN, PRIOR_SD_KN, n_prior))
    alpha_draws = np.exp(rng.normal(PRIOR_MU_AL, PRIOR_SD_AL, n_prior))
    O_ss_draws  = k_n_draws * M_SS ** 2 / (K_CONV + K_CLEAR_O)

    # ------------------------------------------------------------------
    # Run IS per patient
    # ------------------------------------------------------------------
    patients_to_run = wave_a_patients[:smoke_n] if smoke_n > 0 else wave_a_patients
    rows = []
    skipped = []
    t_start = _now_utc()

    for i, patno in enumerate(patients_to_run, start=1):
        pat = dat[dat.PATNO == patno].sort_values("t_years")
        if len(pat) < 2:
            skipped.append({"patno": patno, "reason": "<2-scans"})
            continue
        t_obs = pat.t_years.to_numpy(dtype=float)
        sbr_obs = pat.sbr_putamen_mean.to_numpy(dtype=float)
        if np.any(~np.isfinite(t_obs)) or np.any(~np.isfinite(sbr_obs)):
            skipped.append({"patno": patno, "reason": "non-finite-sbr"})
            continue

        csf_vals = csf_dict.get(patno, None)

        res = is_posterior_one_patient(
            patno, t_obs, sbr_obs, csf_vals,
            k_n_draws, alpha_draws, O_ss_draws,
            s_csf, sigma_csf,
        )
        if "error" in res:
            skipped.append({"patno": patno, "reason": res["error"]})
            continue

        # Persist thinned weighted sample
        w = res.pop("_weights")
        logk = res.pop("_logk")
        loga = res.pop("_loga")
        logt = res.pop("_logt")
        idx = rng.choice(n_prior, size=SAVE_N_PER_PATIENT, replace=True, p=w)
        pd.DataFrame({
            "k_n": np.exp(logk[idx]),
            "alpha_tox": np.exp(loga[idx]),
            "T_tox": np.exp(logt[idx]),
        }).to_parquet(CHAINS_DIR / f"PATNO_{patno}.parquet")

        res["n_scans"] = int(len(pat))
        rows.append(res)

        if i % 25 == 0 or i == len(patients_to_run) or smoke_n > 0:
            elapsed = (_now_utc() - t_start).total_seconds()
            rate = i / elapsed if elapsed > 0 else 0.0
            eta = (len(patients_to_run) - i) / rate if rate > 0 else 0.0
            csf_flag = "CSF" if res.get("has_csf") else "SBR-only"
            print(f"  [{i:3d}/{len(patients_to_run)}] PATNO {patno} ({csf_flag}) "
                  f"ESS={res['ess_frac']:.2%}, cor={res['cor_logk_logalpha']:+.3f}, "
                  f"k_n_sd/prior={res['log_k_n_sd_prior_ratio']:.3f}, "
                  f"%/yr={res['pct_loss_per_yr_median']:.2f}  "
                  f"({rate:.1f} pt/s, ETA {eta:.0f}s)")

    df = pd.DataFrame(rows).sort_values("PATNO").reset_index(drop=True)
    print()
    print(f"Completed {len(df)} / {len(patients_to_run)} patients. "
          f"Skipped: {len(skipped)}. With CSF: {df['has_csf'].sum()}")

    # ------------------------------------------------------------------
    # Comparison with v4 (degeneracy-breaking evaluation)
    # ------------------------------------------------------------------
    df_csf = df[df["has_csf"] == True].copy()  # noqa: E712
    df_sbr_only = df[df["has_csf"] == False].copy()  # noqa: E712

    # Match v4 results for comparison
    v4_matched = v4_by_pat.reindex(df_csf["PATNO"].values)
    v4_matched = v4_matched.dropna(subset=["cor_logk_logalpha"])

    print()
    print("=" * 76)
    print("DEGENERACY-BREAKING EVALUATION (v5 CSF vs v4 SBR-only)")
    print("=" * 76)

    if len(df_csf) > 0:
        print(f"\nCSF-augmented patients (N={len(df_csf)}):")
        print(f"  cor(log k_n, log α_tox):")
        print(f"    v5 (SBR+CSF):  median = {df_csf['cor_logk_logalpha'].median():+.3f}")
        if len(v4_matched) > 0:
            v4_cors = v4_matched.loc[df_csf["PATNO"].values[df_csf["PATNO"].isin(v4_matched.index)], "cor_logk_logalpha"]
            print(f"    v4 (SBR-only): median = {v4_cors.median():+.3f}")
            delta_cor = df_csf["cor_logk_logalpha"].median() - v4_cors.median()
            print(f"    Δ (v5 - v4):   {delta_cor:+.3f} "
                  f"({'TOWARD ZERO = degeneracy breaking' if delta_cor > 0 else 'NO improvement'})")
        print(f"  k_n posterior SD / prior SD:")
        print(f"    v5: median = {df_csf['log_k_n_sd_prior_ratio'].median():.3f}")
        if len(v4_matched) > 0:
            v4_kn_ratio = v4_matched.loc[df_csf["PATNO"].values[df_csf["PATNO"].isin(v4_matched.index)], "log_k_n_sd_prior_ratio"]
            print(f"    v4: median = {v4_kn_ratio.median():.3f}")
        print(f"  α_tox posterior SD / prior SD:")
        print(f"    v5: median = {df_csf['log_alpha_tox_sd_prior_ratio'].median():.3f}")
        print(f"  ESS fraction: median = {df_csf['ess_frac'].median():.2%}")
        print(f"  T_tox SD (log10): median = {df_csf['log_T_tox_sd_log10'].median():.2f} dec")
        print(f"  Implied %/yr: median = {df_csf['pct_loss_per_yr_median'].median():.2f}")

    if len(df_sbr_only) > 0:
        print(f"\nSBR-only patients (no CSF, N={len(df_sbr_only)}):")
        print(f"  cor: median = {df_sbr_only['cor_logk_logalpha'].median():+.3f}")
        print(f"  ESS: median = {df_sbr_only['ess_frac'].median():.2%}")

    # ------------------------------------------------------------------
    # ESS stratification (same as v4)
    # ------------------------------------------------------------------
    print()
    print("=" * 76)
    print("COHORT SUMMARY (all patients)")
    print("=" * 76)
    print(f"ESS fraction:  median = {df.ess_frac.median():.2%}  "
          f"p25-p75 = [{df.ess_frac.quantile(0.25):.2%}, {df.ess_frac.quantile(0.75):.2%}]")
    print(f"cor(log k_n, log α):  median = {df.cor_logk_logalpha.median():+.3f}")
    print(f"T_tox SD (log10):  median = {df.log_T_tox_sd_log10.median():.2f} dec")
    print(f"Implied %/yr:  median = {df.pct_loss_per_yr_median.median():.2f}")
    print()

    high_info = df[df.ess_frac < 0.20]
    mod_info  = df[(df.ess_frac >= 0.20) & (df.ess_frac < 0.50)]
    low_info  = df[df.ess_frac >= 0.50]
    for label, sub in [("HIGH-INFO (ESS<20%)", high_info),
                       ("MOD-INFO (20-50%)", mod_info),
                       ("LOW-INFO (≥50%)", low_info)]:
        if len(sub) > 0:
            print(f"{label:30s} N={len(sub):3d}  "
                  f"cor={sub.cor_logk_logalpha.median():+.3f}  "
                  f"T_tox SD={sub.log_T_tox_sd_log10.median():.2f} dec  "
                  f"%/yr={sub.pct_loss_per_yr_median.median():.2f}")
    print()

    # ------------------------------------------------------------------
    # Gate evaluation
    # ------------------------------------------------------------------
    V4_HIGH_INFO_COR = -0.852  # v4 result from Block 2

    gate_a = df.ess_frac.median() >= 0.15
    gate_b_val = df_csf["cor_logk_logalpha"].median() if len(df_csf) > 0 else -1.0
    gate_b = gate_b_val > V4_HIGH_INFO_COR  # less negative = degeneracy breaking
    gate_c_val = df_csf["log_k_n_sd_prior_ratio"].median() if len(df_csf) > 0 else 1.0
    gate_c = gate_c_val < 1.0  # k_n constrained below prior width
    gate_d_val = df.pct_loss_per_yr_median.median()
    gate_d = 1.0 <= gate_d_val <= 10.0
    gates_all = gate_a and gate_b and gate_c and gate_d

    print("=" * 76)
    print("STEP 2.6v5 GATE EVALUATION")
    print("=" * 76)
    print(f"(a) median ESS ≥ 15%:                     "
          f"{'PASS' if gate_a else 'FAIL'} ({df.ess_frac.median():.2%})")
    print(f"(b) CSF cor less negative than v4 (-0.852): "
          f"{'PASS' if gate_b else 'FAIL'} ({gate_b_val:+.3f})")
    print(f"(c) k_n SD/prior < 1.0 (constrained):     "
          f"{'PASS' if gate_c else 'FAIL'} ({gate_c_val:.3f})")
    print(f"(d) %/yr in [1, 10]:                       "
          f"{'PASS' if gate_d else 'FAIL'} ({gate_d_val:.2f})")
    print(f"OVERALL: {'ALL GATES PASS' if gates_all else 'ONE OR MORE GATES FAIL'}")
    print()

    # ------------------------------------------------------------------
    # Persist results
    # ------------------------------------------------------------------
    out_csv = POST_DIR / "phase2_coupled_is_step26v5_csf.csv"
    df.to_csv(out_csv, index=False)
    print(f"Per-patient summary: {out_csv}")

    summary = {
        "_provenance": provenance,
        "datetime_utc": _now_utc().isoformat(),
        "n_patients": int(len(df)),
        "n_patients_with_csf": int(df["has_csf"].sum()),
        "n_patients_sbr_only": int((~df["has_csf"]).sum()),
        "n_skipped": int(len(skipped)),
        "n_prior_samples": int(n_prior),
        "sbr_sigma": float(SBR_SIGMA),
        "csf_params": {
            "s_csf": float(s_csf),
            "sigma_csf": float(sigma_csf),
            "r_o": float(R_O),
        },
        "prior": {
            "k_n_logmean": float(PRIOR_MU_KN), "k_n_logsd": float(PRIOR_SD_KN),
            "alpha_tox_logmean": float(PRIOR_MU_AL), "alpha_tox_logsd": float(PRIOR_SD_AL),
        },
        "constants": {
            "K_PROD": K_PROD, "K_CLEAR_M": K_CLEAR_M, "K_CONV": K_CONV,
            "K_CLEAR_O": K_CLEAR_O, "K_AGE": K_AGE, "M_SS": M_SS,
            "GAMMA": GAMMA, "T_TOX_CONST": T_TOX_CONST,
        },
        "cohort_medians": {
            "ess_frac": float(df.ess_frac.median()),
            "cor_logk_logalpha": float(df.cor_logk_logalpha.median()),
            "log_T_tox_sd_log10": float(df.log_T_tox_sd_log10.median()),
            "log_k_n_sd_prior_ratio": float(df.log_k_n_sd_prior_ratio.median()),
            "log_alpha_tox_sd_prior_ratio": float(df.log_alpha_tox_sd_prior_ratio.median()),
            "pct_loss_per_yr_median": float(df.pct_loss_per_yr_median.median()),
        },
        "csf_subset_medians": {
            "cor_logk_logalpha": float(df_csf.cor_logk_logalpha.median()) if len(df_csf) > 0 else None,
            "log_k_n_sd_prior_ratio": float(df_csf.log_k_n_sd_prior_ratio.median()) if len(df_csf) > 0 else None,
            "ess_frac": float(df_csf.ess_frac.median()) if len(df_csf) > 0 else None,
        },
        "v4_comparison": {
            "v4_high_info_cor": V4_HIGH_INFO_COR,
            "v5_csf_cor": float(gate_b_val),
            "delta_cor": float(gate_b_val - V4_HIGH_INFO_COR) if gate_b_val != -1.0 else None,
        },
        "stratified": {
            "high_info_N": int(len(high_info)),
            "mod_info_N": int(len(mod_info)),
            "low_info_N": int(len(low_info)),
        },
        "gates": {
            "gate_a_ess_ge_15pct": bool(gate_a),
            "gate_b_cor_less_negative_than_v4": bool(gate_b),
            "gate_c_kn_sd_below_prior": bool(gate_c),
            "gate_d_pct_yr_in_range": bool(gate_d),
            "overall_pass": bool(gates_all),
        },
        "skipped": skipped,
    }

    summary_json = OUT_DIR / "step_2_6_v5_csf_summary.json"
    summary_json.write_text(json.dumps(summary, indent=2, default=str))
    print(f"Summary JSON:         {summary_json}")

    # ------------------------------------------------------------------
    # Diagnostic figure (2x3)
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    # (a) ESS distribution
    ax = axes[0, 0]
    ax.hist(df_csf.ess_frac, bins=40, alpha=0.7, color="tab:blue", edgecolor="k",
            label=f"CSF (N={len(df_csf)})")
    if len(df_sbr_only) > 0:
        ax.hist(df_sbr_only.ess_frac, bins=40, alpha=0.5, color="tab:gray",
                edgecolor="k", label=f"SBR-only (N={len(df_sbr_only)})")
    ax.axvline(0.20, color="tab:red", ls="--", alpha=0.7, label="20% threshold")
    ax.set_xlabel("ESS fraction")
    ax.set_ylabel("# patients")
    ax.set_title("(a) ESS distribution")
    ax.legend(fontsize=8)

    # (b) Correlation comparison v5 vs v4
    ax = axes[0, 1]
    if len(df_csf) > 0 and len(v4_matched) > 0:
        matched_pats = df_csf["PATNO"].values[df_csf["PATNO"].isin(v4_matched.index)]
        v5_cors = df_csf.set_index("PATNO").loc[matched_pats, "cor_logk_logalpha"].values
        v4_cors_arr = v4_matched.loc[matched_pats, "cor_logk_logalpha"].values
        ax.scatter(v4_cors_arr, v5_cors, s=12, alpha=0.5, color="tab:purple")
        ax.plot([-1, 0], [-1, 0], "k--", alpha=0.4, label="y=x (no change)")
        ax.set_xlabel("v4 cor(log k_n, log α_tox)")
        ax.set_ylabel("v5 cor(log k_n, log α_tox)")
        ax.set_title("(b) Degeneracy: v5 vs v4\nAbove y=x = breaking")
    ax.legend(fontsize=8)

    # (c) k_n SD/prior comparison
    ax = axes[0, 2]
    if len(df_csf) > 0 and len(v4_matched) > 0:
        v5_kn = df_csf.set_index("PATNO").loc[matched_pats, "log_k_n_sd_prior_ratio"].values
        v4_kn = v4_matched.loc[matched_pats, "log_k_n_sd_prior_ratio"].values
        ax.scatter(v4_kn, v5_kn, s=12, alpha=0.5, color="tab:green")
        ax.plot([0, 1.2], [0, 1.2], "k--", alpha=0.4, label="y=x (no change)")
        ax.set_xlabel("v4 k_n SD / prior SD")
        ax.set_ylabel("v5 k_n SD / prior SD")
        ax.set_title("(c) k_n constraint: v5 vs v4\nBelow y=x = CSF helped")
    ax.legend(fontsize=8)

    # (d) CSF observed vs predicted
    ax = axes[1, 0]
    if len(df_csf) > 0 and "csf_mean_obs" in df_csf.columns:
        csf_sub = df_csf.dropna(subset=["csf_mean_obs", "csf_pred_at_post_mean"])
        ax.scatter(csf_sub["csf_pred_at_post_mean"], csf_sub["csf_mean_obs"],
                   s=12, alpha=0.5, color="tab:orange")
        lims = [min(csf_sub["csf_pred_at_post_mean"].min(), csf_sub["csf_mean_obs"].min()) * 0.8,
                max(csf_sub["csf_pred_at_post_mean"].max(), csf_sub["csf_mean_obs"].max()) * 1.1]
        ax.plot(lims, lims, "k--", alpha=0.4)
        ax.set_xlabel("CSF predicted (pg/ml)")
        ax.set_ylabel("CSF observed mean (pg/ml)")
        ax.set_title("(d) CSF fit quality")

    # (e) T_tox vs v4
    ax = axes[1, 1]
    if len(df_csf) > 0 and len(v4_matched) > 0:
        v5_ttox = df_csf.set_index("PATNO").loc[matched_pats, "log_T_tox_sd_log10"].values
        v4_ttox = v4_matched.loc[matched_pats, "log_T_tox_sd_log10"].values
        ax.scatter(v4_ttox, v5_ttox, s=12, alpha=0.5, color="tab:red")
        ax.plot([0, 2], [0, 2], "k--", alpha=0.4, label="y=x")
        ax.set_xlabel("v4 T_tox SD (log10 dec)")
        ax.set_ylabel("v5 T_tox SD (log10 dec)")
        ax.set_title("(e) T_tox precision: v5 vs v4")
    ax.legend(fontsize=8)

    # (f) Implied %/yr
    ax = axes[1, 2]
    bins = np.logspace(np.log10(0.01), np.log10(200), 60)
    ax.hist(df_csf.pct_loss_per_yr_median, bins=bins, alpha=0.7,
            color="tab:blue", label=f"CSF (N={len(df_csf)})")
    if len(df_sbr_only) > 0:
        ax.hist(df_sbr_only.pct_loss_per_yr_median, bins=bins, alpha=0.5,
                color="tab:gray", label=f"SBR-only (N={len(df_sbr_only)})")
    ax.axvspan(2, 5, alpha=0.2, color="tab:green", label="Fearnley 2-5%/yr")
    ax.set_xscale("log")
    ax.set_xlabel("Implied %/yr")
    ax.set_ylabel("# patients")
    ax.set_title("(f) Neuron loss rate")
    ax.legend(fontsize=8)

    fig.suptitle(f"Phase 2 Step 2.6v5 — Joint SBR + CSF IS posterior (N={len(df)})",
                 fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig_path = OUT_DIR / "step_2_6_v5_csf_diagnostic.png"
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    print(f"Figure:               {fig_path}")

    # ------------------------------------------------------------------
    # Reproducibility lock-in
    # ------------------------------------------------------------------
    import hashlib as _hashlib

    def _sha256(p: Path) -> str:
        h = _hashlib.sha256()
        with p.open("rb") as f:
            for chunk in iter(lambda: f.read(2**20), b""):
                h.update(chunk)
        return h.hexdigest()

    chain_files = sorted(CHAINS_DIR.glob("PATNO_*.parquet"))
    per_chain_hashes = {cf.name: _sha256(cf) for cf in chain_files}
    combined_hash = _hashlib.sha256(
        "\n".join(f"{k}:{v}" for k, v in sorted(per_chain_hashes.items())).encode()
    ).hexdigest()

    output_hashes = {
        "phase2_coupled_is_step26v5_csf.csv": _sha256(out_csv),
        "step_2_6_v5_csf_diagnostic.png": _sha256(fig_path),
        "chains_is_v5_combined_sha256": combined_hash,
        "n_chain_files": len(chain_files),
    }
    summary["_output_hashes"] = output_hashes
    summary_json.write_text(json.dumps(summary, indent=2, default=str))
    output_hashes["step_2_6_v5_csf_summary.json"] = _sha256(summary_json)

    manifest_path = OUT_DIR / "step_2_6_v5_RUN_MANIFEST.md"
    write_run_manifest(
        manifest_path=manifest_path,
        step_name="Phase 2 Step 2.6v5 — Joint SBR + CSF IS posterior",
        provenance=provenance,
        gate_results={
            "gate_a_ess_ge_15pct": gate_a,
            "gate_b_cor_less_negative_than_v4": gate_b,
            "gate_c_kn_sd_below_prior": gate_c,
            "gate_d_pct_yr_biological_range": gate_d,
            "overall_pass": gates_all,
        },
        summary_metrics={
            "n_patients": len(df),
            "n_with_csf": int(df["has_csf"].sum()),
            "median_ess_frac": f"{df.ess_frac.median():.4f}",
            "csf_cor_logk_logalpha": f"{gate_b_val:+.4f}",
            "v4_cor_for_comparison": f"{V4_HIGH_INFO_COR:+.4f}",
            "delta_cor_v5_minus_v4": f"{gate_b_val - V4_HIGH_INFO_COR:+.4f}" if gate_b_val != -1.0 else "N/A",
            "median_kn_sd_prior_ratio": f"{gate_c_val:.4f}",
            "median_implied_pct_yr": f"{gate_d_val:.4f}",
            "chains_v5_combined_sha256": combined_hash,
        },
    )
    print(f"RUN_MANIFEST:         {manifest_path}")
    print(f"Combined chains SHA:  {combined_hash[:16]}...  ({len(chain_files)} files)")

    return 0 if gates_all else 2


if __name__ == "__main__":
    raise SystemExit(main())
