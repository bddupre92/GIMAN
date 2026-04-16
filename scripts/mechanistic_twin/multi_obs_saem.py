#!/usr/bin/env python3
"""
Multi-Observable SAEM for per-patient (k_n, alpha_tox) calibration
===================================================================

SAEM (Stochastic Approximation EM) with up to 6 observation equations:
  1. DaT-SPECT SBR  → T_tox = k_n × α_tox × const     [1,065 patients]
  2. CSF total α-syn → S_CSF × (M_ss + r_o × O_ss)     [506 patients]
  3. SAA dilution TTT → log(C / F_ss(k_n))              [46 patients]
  4. aSyn aggregate % → O_ss / (M_ss + O_ss)            [48 patients]
  5. NEV α-syn       → S_NEV × (O_ss + β_F × F_ss)     [90 patients]
  6. NfL             → S_NfL × α_tox × O_ss             [523 patients]

Algorithm (Kuhn & Lavielle 2005, Comets et al. 2017 JSS):
  E-step: For each patient independently, draw (k_n_i, α_tox_i) from the
           conditional posterior given current population params, using
           importance-weighted sampling (our proven IS framework).
  M-step: Update population params (μ, σ) from sufficient statistics
           with stochastic approximation averaging.

Literature grounding:
  - Comets et al. 2017 JSS (123 cit) — saemix R package, SAEM reference
  - Lavielle et al. 2007 JPKPD (167 cit) — SAEM for PopPK in Monolix
  - Chan et al. 2010 JPKPD (87 cit) — SAEM for complex PK-PD-viral ODE
  - Bazzoli et al. 2009 Stat Med (36 cit) — SAEM multi-response validated
  - Schunck et al. 2025 bioRxiv — hierarchical ODE <10% bias at 0% overlap
  - Qiu et al. 2024 — multiple starting points for SAEM robustness

Citations
---------
Kuhn & Lavielle 2005 Comput Stat Data Anal 49:1020
Delyon, Lavielle & Moulines 1999 Ann Stat 27:94  -- SAEM convergence proof
"""
from __future__ import annotations

import json
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from scipy import stats

# Reproducibility
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _reproducibility import capture_provenance, write_run_manifest

REPO_ROOT = Path(__file__).resolve().parents[2]

# ====================================================================
# ODE Constants (pinned, Variant B — matches IS pipeline)
# ====================================================================
K_PROD    = 0.1
K_CLEAR_M = 0.05
K_CONV    = 0.001
K_CLEAR_O = 0.003
K_CLEAR_F = 0.001
K_AGE     = 0.0
M_SS      = K_PROD / K_CLEAR_M  # 2.0 nM
GAMMA     = 0.7
HR_PER_YR = 8766.0
T_TOX_CONST = M_SS**2 / (K_CONV + K_CLEAR_O)

# Prior bounds for IS draws
N_IS = 50_000  # IS samples per patient per E-step

# ====================================================================
# ParamVector — typed parameter bag for single-patient likelihood calls
# ====================================================================
# NOTE: The SAEM E/M-step uses plain dicts (pop_params, patient) for
# vectorised IS batches.  ParamVector is a scalar companion for the
# per-channel helper functions and the test suite, and for ch9.6
# single-point likelihood evaluations used in identifiability checks.
#
# Appended fields (s_gfap, sigma_gfap) carry defaults so that code
# that constructs a ParamVector without GFAP args remains valid
# (v2 checkpoint back-compat).

@dataclass
class ParamVector:
    """Scalar parameter bag for single-patient log-likelihood evaluation.

    Core parameters
    ---------------
    k_n        : nucleation rate (/hr) — same as k_n in IS/SAEM
    alpha_tox  : neurotoxicity rate (/hr) — same as alpha_tox in IS/SAEM

    Observation noise (σ) and scale (S/C) parameters
    -------------------------------------------------
    sigma_sbr  : SBR observation noise (a.u.)
    sigma_agg  : aSyn aggregate fraction noise
    sigma_saa  : SAA log-TTT noise (log-scale)
    sigma_nev  : NEV α-syn noise

    ch9.6 GFAP additions (appended with defaults for v2 back-compat)
    -----------------------------------------------------------------
    s_gfap     : GFAP scale factor  (log2 ng/mL per unit oligomer)
    sigma_gfap : GFAP observation noise (log2 ng/mL)
    """
    k_n: float
    alpha_tox: float
    sigma_sbr: float
    sigma_agg: float
    sigma_saa: float
    sigma_nev: float
    # ch9.6 additions — appended for v2 checkpoint back-compat
    s_gfap: float = 1.0
    sigma_gfap: float = 0.5


# ====================================================================
# Per-channel likelihood helpers (scalar; used by total_log_likelihood)
# ====================================================================

def gfap_likelihood(gfap_obs: float, O_t: float, s_gfap: float,
                    sigma_gfap: float) -> float:
    """Normal log-likelihood for CSF GFAP channel.

    GFAP(t) ~ Normal(s_gfap * O_t, sigma_gfap)

    Where O_t is the oligomer concentration at the observation time (from
    the steady-state approximation O_ss(k_n) used throughout this SAEM),
    and s_gfap, sigma_gfap are calibrated population parameters.

    Uses log2(ng/mL) for GFAP measurements (matching the Simoa Project 152
    extraction convention — see mechanistic.ch9_6_gfap_longitudinal).

    Parameters
    ----------
    gfap_obs  : observed CSF GFAP in log2(ng/mL)
    O_t       : oligomer concentration at observation time (nM)
    s_gfap    : scale parameter mapping O_t → predicted log2(ng/mL)
    sigma_gfap: observation noise std (log2 ng/mL)

    Returns
    -------
    float : normal log-likelihood (always finite for finite inputs)
    """
    pred = s_gfap * O_t
    resid = gfap_obs - pred
    return -0.5 * (resid / sigma_gfap) ** 2 - math.log(sigma_gfap) - 0.5 * math.log(2 * math.pi)


def total_log_likelihood(obs: dict, params: ParamVector) -> float:
    """Scalar log-likelihood for one patient-visit observation dict.

    This is a single-point (non-batched) counterpart to the vectorised
    compute_patient_log_likelihood() used in the SAEM E-step.  It uses
    the same steady-state forward model (O_ss, F_ss) so that identifi-
    ability results from ch9_6_identifiability_audit.py apply here.

    NOTE on forward-model alignment:
    The SAEM uses a steady-state (SS) approximation — O_ss(k_n) and
    F_ss(k_n) from analytic formulas — rather than the 4-state scipy ODE
    used in ch9_6_identifiability_audit.py (Task 2).  This is a known
    divergence: the SS model is faster and sufficient for the SAEM IS
    E-step, but the identifiability proof from Task 2 is strictly valid
    only for the ODE-integrated trajectories.  Flag for Task 9 / §9.6
    discussion: verify that SS ≈ ODE at steady state for all obs types.

    Parameters
    ----------
    obs    : dict with any subset of keys:
               "sbr"          – DaT-SPECT SBR (a.u.)
               "asyn_agg_pct" – aSyn aggregate percentage (%)
               "saa_ttt"      – SAA dilution TTT (hours)
               "nev_asyn"     – NEV α-syn signal
               "gfap_npx"     – CSF GFAP in log2(ng/mL)   [ch9.6 addition]
               "t_years"      – observation time (years, used for SBR decay)
    params : ParamVector

    Returns
    -------
    float : total log-likelihood (sum over present channels)
    """
    k_n = params.k_n
    atox = params.alpha_tox
    o_ss_val = k_n * M_SS**2 / (K_CONV + K_CLEAR_O)
    f_ss_val = K_CONV * o_ss_val / K_CLEAR_F

    ll = 0.0

    # --- SBR (longitudinal decay) ---
    if "sbr" in obs and obs["sbr"] is not None and math.isfinite(obs["sbr"]):
        t_yr = float(obs.get("t_years", 0.0))
        t_hr = t_yr * HR_PER_YR
        decay_hr = atox * o_ss_val + K_AGE
        log_ratio = max(-decay_hr * t_hr, -50.0)
        sbr_anchor = float(obs["sbr"])  # treat observed SBR as anchor (t=0)
        sbr_pred = sbr_anchor * math.exp(GAMMA * log_ratio)
        ll += -0.5 * ((sbr_pred - sbr_anchor) / params.sigma_sbr) ** 2

    # --- aSyn aggregate fraction ---
    if "asyn_agg_pct" in obs and obs["asyn_agg_pct"] is not None:
        agg_obs = float(obs["asyn_agg_pct"])
        if math.isfinite(agg_obs):
            agg_pred = o_ss_val / (M_SS + o_ss_val) * 100.0
            ll += -0.5 * ((agg_pred - agg_obs) / params.sigma_agg) ** 2

    # --- SAA dilution TTT ---
    if "saa_ttt" in obs and obs["saa_ttt"] is not None:
        saa_obs = float(obs["saa_ttt"])
        if math.isfinite(saa_obs) and saa_obs > 0:
            # log-normal model (matching compute_patient_log_likelihood)
            log_ttt_pred = -math.log(max(f_ss_val, 1e-20))
            log_ttt_obs = math.log(max(saa_obs, 0.01))
            ll += -0.5 * ((log_ttt_pred - log_ttt_obs) / params.sigma_saa) ** 2

    # --- NEV α-syn ---
    if "nev_asyn" in obs and obs["nev_asyn"] is not None:
        nev_obs = float(obs["nev_asyn"])
        if math.isfinite(nev_obs):
            nev_pred = o_ss_val + f_ss_val  # S_nev = 1.0 (population default)
            ll += -0.5 * ((nev_pred - nev_obs) / params.sigma_nev) ** 2

    # --- CSF GFAP (ch9.6 addition) ---
    if "gfap_npx" in obs and obs["gfap_npx"] is not None:
        gfap_obs_val = float(obs["gfap_npx"])
        if math.isfinite(gfap_obs_val):
            ll += gfap_likelihood(
                gfap_obs=gfap_obs_val,
                O_t=o_ss_val,
                s_gfap=params.s_gfap,
                sigma_gfap=params.sigma_gfap,
            )

    return ll


def O_ss(k_n):
    """Steady-state oligomer concentration."""
    return k_n * M_SS**2 / (K_CONV + K_CLEAR_O)


def F_ss(k_n):
    """Steady-state fibril concentration."""
    return K_CONV * O_ss(k_n) / K_CLEAR_F


def compute_patient_log_likelihood(
    log_kn: np.ndarray,
    log_atox: np.ndarray,
    patient: dict,
    pop_params: dict,
) -> np.ndarray:
    """Compute log-likelihood for a batch of (k_n, α_tox) values for one patient.

    Returns shape (N_IS,) array of log-likelihoods.
    """
    k_n = np.exp(log_kn)
    atox = np.exp(log_atox)
    o_ss = k_n * M_SS**2 / (K_CONV + K_CLEAR_O)
    f_ss = K_CONV * o_ss / K_CLEAR_F

    ll = np.zeros(len(log_kn))

    # === 1. SBR (longitudinal, everyone) ===
    sbr_anchor = patient["sbr_anchor"]
    decay_hr = atox * o_ss + K_AGE
    for j in range(patient["n_scans"]):
        t_hr = patient["t_years"][j] * HR_PER_YR
        log_ratio = np.clip(-decay_hr * t_hr, -50.0, 0.0)
        sbr_pred = sbr_anchor * np.exp(GAMMA * log_ratio)
        ll += -0.5 * ((sbr_pred - patient["sbr_obs"][j]) / pop_params["sigma_sbr"]) ** 2

    # === 2. CSF total alpha-syn ===
    if not np.isnan(patient["csf_asyn"]):
        csf_pred = pop_params["S_csf"] * (M_SS + 1.0 * o_ss)  # r_o = 1.0
        ll += -0.5 * ((csf_pred - patient["csf_asyn"]) / pop_params["sigma_csf"]) ** 2

    # === 3. SAA dilution TTT ===
    if not np.isnan(patient["saa_ttt"]):
        log_ttt_pred = np.log(pop_params["C_saa"]) - np.log(np.maximum(f_ss, 1e-20))
        log_ttt_obs = np.log(max(patient["saa_ttt"], 0.01))
        ll += -0.5 * ((log_ttt_pred - log_ttt_obs) / pop_params["sigma_saa"]) ** 2

    # === 4. aSyn aggregate fraction ===
    if not np.isnan(patient["asyn_agg_frac"]):
        agg_pred = o_ss / (M_SS + o_ss)
        ll += -0.5 * ((agg_pred - patient["asyn_agg_frac"]) / pop_params["sigma_agg"]) ** 2

    # === 5. NEV alpha-syn ===
    if not np.isnan(patient["nev_asyn"]):
        nev_pred = pop_params["S_nev"] * (o_ss + f_ss)
        ll += -0.5 * ((nev_pred - patient["nev_asyn"]) / pop_params["sigma_nev"]) ** 2

    # === 6. NfL ===
    if not np.isnan(patient["nfl"]):
        nfl_pred = pop_params["S_nfl"] * atox * o_ss * HR_PER_YR
        ll += -0.5 * ((nfl_pred - patient["nfl"]) / pop_params["sigma_nfl"]) ** 2

    # === 7. CSF GFAP (ch9.6 addition) ===
    if not np.isnan(patient.get("gfap_npx", np.nan)):
        gfap_pred = pop_params["S_gfap"] * o_ss
        ll += -0.5 * ((gfap_pred - patient["gfap_npx"]) / pop_params["sigma_gfap"]) ** 2

    return ll


def e_step_one_patient(patient: dict, pop_params: dict, rng: np.random.Generator) -> dict:
    """E-step: IS-weighted posterior for one patient given population params."""
    mu_kn = pop_params["mu_logkn"]
    sd_kn = pop_params["sigma_logkn"]
    mu_at = pop_params["mu_logatox"]
    sd_at = pop_params["sigma_logatox"]

    # Draw from current population distribution (proposal = population prior)
    log_kn = rng.normal(mu_kn, sd_kn, N_IS)
    log_atox = rng.normal(mu_at, sd_at, N_IS)

    # Compute log-likelihood
    ll = compute_patient_log_likelihood(log_kn, log_atox, patient, pop_params)

    # IS weights (proposal = population, so weight = likelihood only)
    ll_shift = ll - np.max(ll)
    w = np.exp(ll_shift)
    w_sum = w.sum()
    if not np.isfinite(w_sum) or w_sum <= 0:
        return {"error": True, "patno": patient["patno"]}
    w /= w_sum
    ess = 1.0 / np.sum(w**2)

    # Weighted sufficient statistics
    wm_logkn = np.sum(w * log_kn)
    wm_logatox = np.sum(w * log_atox)
    wv_logkn = np.sum(w * (log_kn - wm_logkn) ** 2)
    wv_logatox = np.sum(w * (log_atox - wm_logatox) ** 2)

    # Individual EBE (empirical Bayes estimate)
    kn_ebe = np.exp(wm_logkn)
    atox_ebe = np.exp(wm_logatox)
    T_tox = kn_ebe * atox_ebe * T_TOX_CONST
    pct_yr = (1.0 - np.exp(-T_tox * HR_PER_YR)) * 100.0
    cor_ka = np.sum(w * (log_kn - wm_logkn) * (log_atox - wm_logatox))
    sd_k = np.sqrt(max(wv_logkn, 1e-20))
    sd_a = np.sqrt(max(wv_logatox, 1e-20))
    cor_ka = cor_ka / (sd_k * sd_a + 1e-30)

    # Compute residuals at EBE for M-step sigma estimation
    o_ss_ebe = kn_ebe * M_SS**2 / (K_CONV + K_CLEAR_O)
    f_ss_ebe = K_CONV * o_ss_ebe / K_CLEAR_F
    decay_ebe = atox_ebe * o_ss_ebe + K_AGE

    sbr_resid_sq = []
    for j in range(patient["n_scans"]):
        t_hr = patient["t_years"][j] * HR_PER_YR
        lr = max(-decay_ebe * t_hr, -50.0)
        pred = patient["sbr_anchor"] * np.exp(GAMMA * lr)
        sbr_resid_sq.append((pred - patient["sbr_obs"][j]) ** 2)

    csf_resid_sq = None
    if not np.isnan(patient["csf_asyn"]):
        csf_pred = pop_params["S_csf"] * (M_SS + 1.0 * o_ss_ebe)
        csf_resid_sq = (csf_pred - patient["csf_asyn"]) ** 2

    saa_resid_sq = None
    if not np.isnan(patient["saa_ttt"]):
        log_ttt_pred = np.log(pop_params["C_saa"]) - np.log(max(f_ss_ebe, 1e-20))
        saa_resid_sq = (log_ttt_pred - np.log(max(patient["saa_ttt"], 0.01))) ** 2

    gfap_resid_sq = None
    if not np.isnan(patient.get("gfap_npx", np.nan)):
        gfap_pred = pop_params["S_gfap"] * o_ss_ebe
        gfap_resid_sq = (gfap_pred - patient["gfap_npx"]) ** 2

    return {
        "error": False,
        "patno": patient["patno"],
        "ess": ess,
        "log_kn_mean": wm_logkn,
        "log_atox_mean": wm_logatox,
        "log_kn_sq_mean": wm_logkn**2 + wv_logkn,
        "log_atox_sq_mean": wm_logatox**2 + wv_logatox,
        "k_n_ebe": kn_ebe,
        "atox_ebe": atox_ebe,
        "T_tox": T_tox,
        "pct_yr": pct_yr,
        "log_kn_sd": sd_k,
        "log_atox_sd": sd_a,
        "cor_ka": cor_ka,
        "sbr_resid_sq": sbr_resid_sq,
        "csf_resid_sq": csf_resid_sq,
        "saa_resid_sq": saa_resid_sq,
        "gfap_resid_sq": gfap_resid_sq,
    }


def m_step(e_results: list[dict], pop_params: dict, gamma_k: float) -> dict:
    """M-step: Update population params from E-step sufficient statistics.

    Uses stochastic approximation: θ_{k+1} = (1-γ_k) θ_k + γ_k θ̂_k
    where γ_k is the step size (decreasing schedule for convergence).
    """
    valid = [r for r in e_results if not r["error"]]
    n = len(valid)
    if n == 0:
        return pop_params

    # Sufficient statistics from E-step
    mean_logkn = np.mean([r["log_kn_mean"] for r in valid])
    mean_logatox = np.mean([r["log_atox_mean"] for r in valid])
    var_logkn = np.mean([r["log_kn_sq_mean"] for r in valid]) - mean_logkn**2
    var_logatox = np.mean([r["log_atox_sq_mean"] for r in valid]) - mean_logatox**2

    # SA update
    new_params = dict(pop_params)
    new_params["mu_logkn"] = (1 - gamma_k) * pop_params["mu_logkn"] + gamma_k * mean_logkn
    new_params["mu_logatox"] = (1 - gamma_k) * pop_params["mu_logatox"] + gamma_k * mean_logatox
    new_params["sigma_logkn"] = max(0.01, np.sqrt(
        (1 - gamma_k) * pop_params["sigma_logkn"]**2 + gamma_k * max(var_logkn, 0.01)
    ))
    new_params["sigma_logatox"] = max(0.01, np.sqrt(
        (1 - gamma_k) * pop_params["sigma_logatox"]**2 + gamma_k * max(var_logatox, 0.01)
    ))

    # Joint estimation of observation noise (Bazzoli 2009, Chan 2010)
    # sigma^2 MLE = mean(residuals^2) across patients
    all_sbr_resid = []
    all_csf_resid = []
    all_saa_resid = []
    all_gfap_resid = []
    for r in valid:
        all_sbr_resid.extend(r["sbr_resid_sq"])
        if r["csf_resid_sq"] is not None:
            all_csf_resid.append(r["csf_resid_sq"])
        if r["saa_resid_sq"] is not None:
            all_saa_resid.append(r["saa_resid_sq"])
        if r.get("gfap_resid_sq") is not None:
            all_gfap_resid.append(r["gfap_resid_sq"])

    if all_sbr_resid:
        sigma_sbr_new = np.sqrt(np.mean(all_sbr_resid))
        new_params["sigma_sbr"] = (1 - gamma_k) * pop_params["sigma_sbr"] + gamma_k * max(sigma_sbr_new, 0.01)
    if all_csf_resid:
        sigma_csf_new = np.sqrt(np.mean(all_csf_resid))
        new_params["sigma_csf"] = (1 - gamma_k) * pop_params["sigma_csf"] + gamma_k * max(sigma_csf_new, 1.0)
    if all_saa_resid:
        sigma_saa_new = np.sqrt(np.mean(all_saa_resid))
        new_params["sigma_saa"] = (1 - gamma_k) * pop_params["sigma_saa"] + gamma_k * max(sigma_saa_new, 0.1)
    if all_gfap_resid:
        sigma_gfap_new = np.sqrt(np.mean(all_gfap_resid))
        new_params["sigma_gfap"] = (1 - gamma_k) * pop_params["sigma_gfap"] + gamma_k * max(sigma_gfap_new, 0.05)

    return new_params


def load_data():
    """Load DaT-SPECT + multi-observable inventory + expanded SAA."""
    dat = pd.read_parquet(REPO_ROOT / "outputs/mechanistic_twin/data/dat_spect_longitudinal.parquet")
    inv = pd.read_parquet(REPO_ROOT / "outputs/mechanistic_twin/data/multi_observable_inventory.parquet")
    dat_patnos = set(dat.PATNO.unique())

    # Expanded SAA extraction: ALL TTT sources across all PPMI projects
    # Per Mammana 2024 CCLM: LAG/TTT is the most reliable kinetic parameter
    bio = pd.read_csv(REPO_ROOT / "data/00_raw/GIMAN/ppmi_data_csv/Current_Biospecimen_Analysis_Results_30Sep2025.csv",
                       low_memory=False)
    saa_tests = [t for t in bio.TESTNAME.unique() if 'TTT' in str(t) and 'SAA' in str(t)]
    saa_all = bio[bio.TESTNAME.isin(saa_tests) & bio.PATNO.isin(dat_patnos)].copy()
    saa_all['value'] = pd.to_numeric(saa_all.TESTVALUE, errors='coerce')
    saa_all = saa_all.dropna(subset=['value'])
    saa_all = saa_all[saa_all.value > 0]
    saa_expanded = saa_all.groupby('PATNO')['value'].median().to_dict()
    print(f"Expanded SAA: {len(saa_expanded)} patients with TTT + DaT overlap")

    patients = []
    for patno, gdf in dat.groupby("PATNO"):
        gdf = gdf.sort_values("t_years")
        inv_row = inv[inv.PATNO == patno]

        p = {
            "patno": int(patno),
            "t_years": gdf.t_years.values.astype(float),
            "sbr_obs": gdf.sbr_caudate_mean.values.astype(float),
            "n_scans": len(gdf),
            "sbr_anchor": float(gdf.sbr_caudate_mean.iloc[0]),
            "wave": gdf.wave.iloc[0],
            "csf_asyn": float(inv_row.csf_asyn_median.iloc[0]) if len(inv_row) and pd.notna(inv_row.csf_asyn_median.iloc[0]) else np.nan,
            "saa_ttt": np.nan,
            "asyn_agg_frac": float(inv_row.asyn_agg_frac_median.iloc[0]) if len(inv_row) and pd.notna(inv_row.asyn_agg_frac_median.iloc[0]) else np.nan,
            "nev_asyn": float(inv_row.nev_asyn_median.iloc[0]) if len(inv_row) and pd.notna(inv_row.nev_asyn_median.iloc[0]) else np.nan,
            "nfl": float(inv_row.nfl_median.iloc[0]) if len(inv_row) and pd.notna(inv_row.nfl_median.iloc[0]) else np.nan,
            # ch9.6 GFAP addition — log2(ng/mL) median per patient
            "gfap_npx": float(inv_row.gfap_log2_median.iloc[0]) if len(inv_row) and "gfap_log2_median" in inv_row.columns and pd.notna(inv_row.gfap_log2_median.iloc[0]) else np.nan,
        }
        # SAA: use expanded extraction (all TTT sources, median across reps)
        if int(patno) in saa_expanded:
            p["saa_ttt"] = saa_expanded[int(patno)]
        # Fallback: dilution TTT at 1:400 from inventory
        elif len(inv_row) and "saa_ttt_1400" in inv_row.columns and pd.notna(inv_row.saa_ttt_1400.iloc[0]):
            p["saa_ttt"] = float(inv_row.saa_ttt_1400.iloc[0])
        patients.append(p)

    patients.sort(key=lambda p: p["patno"])
    return patients


def run_saem(patients: list[dict], n_iterations: int = 100, n_burn: int = 50,
             seed: int = 20260411, run_tag: str = "saem_multi_obs_v1"):
    """Run SAEM algorithm."""
    rng = np.random.default_rng(seed)

    n_csf = sum(1 for p in patients if not np.isnan(p["csf_asyn"]))
    n_saa = sum(1 for p in patients if not np.isnan(p["saa_ttt"]))
    n_agg = sum(1 for p in patients if not np.isnan(p["asyn_agg_frac"]))
    n_nev = sum(1 for p in patients if not np.isnan(p["nev_asyn"]))
    n_nfl = sum(1 for p in patients if not np.isnan(p["nfl"]))
    n_gfap = sum(1 for p in patients if not np.isnan(p.get("gfap_npx", np.nan)))
    print(f"SAEM: {len(patients)} patients, coverage: CSF={n_csf}, SAA={n_saa}, "
          f"Agg={n_agg}, NEV={n_nev}, NfL={n_nfl}, GFAP={n_gfap}")

    # Initial population params
    pop = {
        "mu_logkn": np.log(1e-4),
        "mu_logatox": np.log(1.8e-5),
        "sigma_logkn": 1.5,
        "sigma_logatox": 2.0,
        "sigma_sbr": 0.20,
        "sigma_csf": 300.0,
        "sigma_saa": 3.0,
        "sigma_agg": 0.10,
        "sigma_nev": 0.30,
        "sigma_nfl": 5.0,
        "S_csf": 680.0,
        "S_nev": 1.0,
        "S_nfl": 1.0,
        "C_saa": 1.0,
        # ch9.6 GFAP additions — S_gfap and sigma_gfap initialized from
        # Task 4 cohort (median log2(GFAP)≈4.0, O_ss≈2e-3 nM → S≈2000)
        "S_gfap": 2000.0,
        "sigma_gfap": 0.5,
    }

    # SAEM iteration
    history = []
    final_e_results = None

    for iteration in range(n_iterations):
        # Step size schedule (Delyon et al. 1999):
        # γ_k = 1 during burn-in (pure SA), then γ_k = 1/(k-n_burn+1) for averaging
        if iteration < n_burn:
            gamma_k = 1.0
        else:
            gamma_k = 1.0 / (iteration - n_burn + 1)

        # E-step: per-patient IS (parallelizable, but sequential here for clarity)
        e_results = []
        for p in patients:
            res = e_step_one_patient(p, pop, rng)
            e_results.append(res)

        # M-step
        pop = m_step(e_results, pop, gamma_k)

        # Track convergence
        valid = [r for r in e_results if not r["error"]]
        median_ess = np.median([r["ess"] for r in valid])
        median_pct = np.median([r["pct_yr"] for r in valid])
        median_cor = np.median([r["cor_ka"] for r in valid])
        median_kn_sd = np.median([r["log_kn_sd"] for r in valid])

        history.append({
            "iteration": iteration,
            "mu_logkn": pop["mu_logkn"],
            "mu_logatox": pop["mu_logatox"],
            "sigma_logkn": pop["sigma_logkn"],
            "sigma_logatox": pop["sigma_logatox"],
            "median_ess": median_ess,
            "median_pct_yr": median_pct,
            "median_cor_ka": median_cor,
            "median_kn_sd": median_kn_sd,
            "n_valid": len(valid),
        })

        if iteration % 10 == 0 or iteration == n_iterations - 1:
            print(f"  [{iteration:3d}/{n_iterations}] μ_kn={pop['mu_logkn']:.3f} "
                  f"σ_kn={pop['sigma_logkn']:.3f} μ_at={pop['mu_logatox']:.3f} "
                  f"σ_at={pop['sigma_logatox']:.3f} | ESS={median_ess:.0f} "
                  f"cor={median_cor:.3f} %/yr={median_pct:.2f}")

        final_e_results = e_results

    return pop, final_e_results, history


def save_results(pop, e_results, history, patients, run_tag, seed):
    """Save SAEM results."""
    out_dir = REPO_ROOT / "outputs/mechanistic_twin/data/posteriors" / f"saem_{run_tag}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Individual params
    rows = []
    for r, p in zip(e_results, patients):
        if r["error"]:
            continue
        rows.append({
            "PATNO": r["patno"],
            "n_scans": p["n_scans"],
            "wave": p["wave"],
            "k_n_ebe": r["k_n_ebe"],
            "alpha_tox_ebe": r["atox_ebe"],
            "log_k_n_mean": r["log_kn_mean"],
            "log_alpha_tox_mean": r["log_atox_mean"],
            "log_k_n_sd": r["log_kn_sd"],
            "log_alpha_tox_sd": r["log_atox_sd"],
            "T_tox": r["T_tox"],
            "pct_loss_per_yr": r["pct_yr"],
            "cor_logk_logalpha": r["cor_ka"],
            "ess": r["ess"],
            "has_csf": not np.isnan(p["csf_asyn"]),
            "has_saa": not np.isnan(p["saa_ttt"]),
            "has_agg": not np.isnan(p["asyn_agg_frac"]),
            "has_nev": not np.isnan(p["nev_asyn"]),
            "has_nfl": not np.isnan(p["nfl"]),
            "has_gfap": not np.isnan(p.get("gfap_npx", np.nan)),
        })
    indiv_df = pd.DataFrame(rows).sort_values("PATNO")
    indiv_df.to_csv(out_dir / "individual_params.csv", index=False)

    # Stratified analysis
    with_asyn = indiv_df[indiv_df.has_csf | indiv_df.has_saa | indiv_df.has_agg]
    without_asyn = indiv_df[~(indiv_df.has_csf | indiv_df.has_saa | indiv_df.has_agg)]

    # Diagnostics
    diagnostics = {
        "run_tag": run_tag,
        "n_patients": len(patients),
        "n_valid": len(rows),
        "seed": seed,
        "population_params": pop,
        "cohort_summary": {
            "log_kn_sd_median": float(indiv_df.log_k_n_sd.median()),
            "log_kn_sd_prior_ratio": float(indiv_df.log_k_n_sd.median() / 1.5),
            "cor_logk_loga_median": float(indiv_df.cor_logk_logalpha.median()),
            "pct_loss_yr_median": float(indiv_df.pct_loss_per_yr.median()),
            "pct_loss_yr_q025": float(indiv_df.pct_loss_per_yr.quantile(0.025)),
            "pct_loss_yr_q975": float(indiv_df.pct_loss_per_yr.quantile(0.975)),
        },
        "stratified": {
            "with_asyn_obs": {
                "n": len(with_asyn),
                "log_kn_sd_median": float(with_asyn.log_k_n_sd.median()) if len(with_asyn) else None,
                "cor_median": float(with_asyn.cor_logk_logalpha.median()) if len(with_asyn) else None,
            },
            "without_asyn_obs": {
                "n": len(without_asyn),
                "log_kn_sd_median": float(without_asyn.log_k_n_sd.median()) if len(without_asyn) else None,
                "cor_median": float(without_asyn.cor_logk_logalpha.median()) if len(without_asyn) else None,
            },
        },
        "comparison": {
            "is_v4_sbr_only": 0.926,
            "is_v5_csf_joint": 0.655,
            "hlme_sbr_only": 0.53,
        },
    }

    with open(out_dir / "diagnostics.json", "w") as f:
        json.dump(diagnostics, f, indent=2, default=str)

    # History
    pd.DataFrame(history).to_csv(out_dir / "convergence_history.csv", index=False)

    # Provenance
    provenance = capture_provenance(
        Path(__file__).resolve(), REPO_ROOT,
        [
            REPO_ROOT / "outputs/mechanistic_twin/data/dat_spect_longitudinal.parquet",
            REPO_ROOT / "outputs/mechanistic_twin/data/multi_observable_inventory.parquet",
        ],
        extra={"seed": seed, "N_IS": N_IS, "run_tag": run_tag},
    )
    with open(out_dir / "provenance.json", "w") as f:
        json.dump(provenance, f, indent=2, default=str)

    write_run_manifest(
        out_dir / "RUN_MANIFEST.md",
        f"Multi-Observable SAEM ({run_tag})",
        provenance,
        summary_metrics={
            "n_patients": len(patients),
            "log_kn_sd_prior_ratio": diagnostics["cohort_summary"]["log_kn_sd_prior_ratio"],
            "cor_logk_loga": diagnostics["cohort_summary"]["cor_logk_loga_median"],
            "pct_yr_median": diagnostics["cohort_summary"]["pct_loss_yr_median"],
        },
    )

    print(f"\nResults saved to {out_dir}")
    return diagnostics, indiv_df


def print_summary(diagnostics, indiv_df):
    """Print summary report."""
    coh = diagnostics["cohort_summary"]
    strat = diagnostics["stratified"]
    comp = diagnostics["comparison"]

    print("\n" + "=" * 70)
    print("MULTI-OBSERVABLE SAEM — SUMMARY")
    print("=" * 70)
    print(f"Patients: {diagnostics['n_patients']} ({diagnostics['n_valid']} valid)")

    pop = diagnostics["population_params"]
    print(f"\nPopulation: μ_kn={pop['mu_logkn']:.3f} (k_n={np.exp(pop['mu_logkn']):.2e}), "
          f"σ_kn={pop['sigma_logkn']:.3f}")
    print(f"            μ_at={pop['mu_logatox']:.3f} (α_tox={np.exp(pop['mu_logatox']):.2e}), "
          f"σ_at={pop['sigma_logatox']:.3f}")

    hlme_ratio = coh["log_kn_sd_prior_ratio"]
    print(f"\n--- Degeneracy diagnostic ---")
    print(f"  IS v4 (SBR-only):     SD/prior = {comp['is_v4_sbr_only']}")
    print(f"  IS v5 (SBR+CSF):      SD/prior = {comp['is_v5_csf_joint']}")
    print(f"  HLME SBR-only:        SD/prior = {comp['hlme_sbr_only']}")
    print(f"  SAEM MULTI-OBS:       SD/prior = {hlme_ratio:.3f}")
    print(f"  cor(log k_n, log α):  {coh['cor_logk_loga_median']:.3f}")
    print(f"  %/yr neuron loss:     {coh['pct_loss_yr_median']:.2f}% "
          f"[{coh['pct_loss_yr_q025']:.2f}, {coh['pct_loss_yr_q975']:.2f}]")

    w = strat["with_asyn_obs"]
    wo = strat["without_asyn_obs"]
    print(f"\n--- Stratified by α-syn observable ---")
    if w["log_kn_sd_median"] is not None:
        print(f"  WITH (n={w['n']}):    SD/prior = {w['log_kn_sd_median']/1.5:.3f}, cor = {w['cor_median']:.3f}")
    if wo["log_kn_sd_median"] is not None:
        print(f"  WITHOUT (n={wo['n']}): SD/prior = {wo['log_kn_sd_median']/1.5:.3f}, cor = {wo['cor_median']:.3f}")

    print(f"\n--- Next: Run decisive test ---")
    print(f"  python scripts/mechanistic_twin/hlme_decisive_test.py "
          f"--hlme-dir saem_{diagnostics['run_tag']}")
    print("=" * 70)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-iterations", type=int, default=100)
    parser.add_argument("--n-burn", type=int, default=50)
    parser.add_argument("--seed", type=int, default=20260411)
    parser.add_argument("--run-tag", type=str, default="multi_obs_v1")
    args = parser.parse_args()

    patients = load_data()
    print(f"Loaded {len(patients)} patients")

    pop, e_results, history = run_saem(
        patients,
        n_iterations=args.n_iterations,
        n_burn=args.n_burn,
        seed=args.seed,
        run_tag=args.run_tag,
    )

    diagnostics, indiv_df = save_results(
        pop, e_results, history, patients, args.run_tag, args.seed
    )
    print_summary(diagnostics, indiv_df)


if __name__ == "__main__":
    main()
