#!/usr/bin/env python3
"""
Block 5 — Strict Leave-Last-Scan-Out Forward Validation (Step 2.8v5-LOO)
=========================================================================

Purpose
-------
Replace the training-data PPC (99.5% coverage, known to be over-wide) with
an honest out-of-sample validation: for each patient with >=3 scans, fit the
IS posterior using only the FIRST N-1 scans, then predict the LAST scan and
check whether the observed SBR falls within the posterior predictive interval.

This follows the Leave-Future-Out CV approach recommended by Bürkner et al.
(2019 J Stat Comp Sim) for Bayesian time series models, adapted to our
closed-form IS framework. Since our IS is fast (~0.04s per patient), we do
exact refitting rather than PSIS approximation (Vehtari et al. 2017).

Methodology
-----------
For each patient i with scans {s_1, ..., s_N}:
  1. Fit IS posterior on scans {s_1, ..., s_{N-1}} (train set)
  2. For each IS draw, predict SBR at time t_N using the train posterior
  3. Add observation noise N(0, σ_SBR²) to get predictive distribution
  4. Check: does observed SBR(t_N) fall within the 95% predictive interval?

Joint SBR + CSF likelihood is used (same as Step 2.6v5). CSF data is included
in the train set (all CSF timepoints, since CSF is time-invariant and not
being predicted).

Gate criteria
-------------
  (L1) LOO coverage >= 80% at 95% PI (honest out-of-sample, lower than
       training-data PPC is expected)
  (L2) LOO coverage >= Phase 1 LOO coverage (93.75%) - 10pp = 83.75%
       (non-regression with generous tolerance for the harder LOO task)
  (L3) Median absolute z-score in [0.5, 2.0] (not too sharp, not too wide)

Citations
---------
Vehtari et al. 2017 J Mach Learn Res 25:1-48 — PSIS
Bürkner et al. 2019 J Stat Comp Sim 89:2338-2354 — LFO-CV
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _reproducibility import capture_provenance, write_run_manifest  # noqa: E402

REPO_ROOT    = Path(__file__).resolve().parents[2]
DAT_PATH     = REPO_ROOT / "outputs/mechanistic_twin/data/dat_spect_longitudinal.parquet"
CSF_PATH     = REPO_ROOT / "data/00_raw/GIMAN/ppmi_data_csv/Current_Biospecimen_Analysis_Results_30Sep2025.csv"
V5_CSV       = REPO_ROOT / "outputs/mechanistic_twin/data/posteriors/phase2_coupled_is_step26v5_csf.csv"
OUT_DIR      = REPO_ROOT / "outputs/mechanistic_twin/phase2"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Constants (must match step_2_6_v5)
PRIOR_MU_KN = np.log(1e-4);    PRIOR_SD_KN = 1.5
PRIOR_MU_AL = np.log(1.8e-5);  PRIOR_SD_AL = 2.0
K_PROD    = 0.1
K_CLEAR_M = 0.05
K_CONV    = 0.001
K_CLEAR_O = 0.003
K_AGE     = 0.0
M_SS      = K_PROD / K_CLEAR_M
GAMMA     = 0.7
HR_PER_YR = 8766.0
T_TOX_CONST = M_SS**2 / (K_CONV + K_CLEAR_O)
SBR_SIGMA = 0.20
R_O       = 1.0
N_PRIOR   = 50_000
RNG_SEED  = 202604103  # distinct from v5 seed


def load_csf_data(csf_path: Path, wave_a_pats: list[int]) -> dict[int, np.ndarray]:
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


def loo_one_patient(
    patno: int,
    t_obs_train: np.ndarray,
    sbr_obs_train: np.ndarray,
    t_test: float,
    sbr_test: float,
    csf_vals: np.ndarray | None,
    k_n_draws: np.ndarray,
    alpha_draws: np.ndarray,
    O_ss_draws: np.ndarray,
    s_csf: float,
    sigma_csf: float,
    rng: np.random.Generator,
) -> dict:
    """Fit IS on train scans, predict test scan, return coverage info."""
    N = len(k_n_draws)
    SBR_0 = float(sbr_obs_train[0])

    # --- SBR log-likelihood on TRAIN scans only ---
    decay_hr = alpha_draws * O_ss_draws + K_AGE
    t_hr_train = t_obs_train * HR_PER_YR
    log_ratio = -decay_hr[:, None] * t_hr_train[None, :]
    log_ratio = np.clip(log_ratio, -50.0, 0.0)
    sbr_pred_train = SBR_0 * np.exp(GAMMA * log_ratio)
    resid = sbr_pred_train - sbr_obs_train[None, :]
    log_lik_sbr = -0.5 * np.sum((resid / SBR_SIGMA) ** 2, axis=1)

    # --- CSF log-likelihood (all timepoints, not held out) ---
    log_lik_csf = np.zeros(N, dtype=np.float64)
    if csf_vals is not None and len(csf_vals) > 0:
        csf_pred = s_csf * (M_SS + R_O * O_ss_draws)
        for y_csf in csf_vals:
            log_lik_csf += -0.5 * ((y_csf - csf_pred) / sigma_csf) ** 2

    # --- Joint weights ---
    log_lik = log_lik_sbr + log_lik_csf
    log_lik_shift = log_lik - np.max(log_lik)
    w = np.exp(log_lik_shift)
    w_sum = w.sum()
    if not np.isfinite(w_sum) or w_sum <= 0:
        return {"PATNO": patno, "error": "degenerate"}
    w /= w_sum

    # --- Predict test scan ---
    t_hr_test = t_test * HR_PER_YR
    T_tox = alpha_draws * O_ss_draws + K_AGE
    log_ratio_test = np.clip(-T_tox * t_hr_test, -50.0, 0.0)
    sbr_pred_test = SBR_0 * np.exp(GAMMA * log_ratio_test)

    # Resample to get predictive distribution
    idx = rng.choice(N, size=5000, replace=True, p=w)
    sbr_pred_samples = sbr_pred_test[idx] + rng.normal(0, SBR_SIGMA, size=5000)

    # Predictive intervals
    pi_025, pi_975 = np.quantile(sbr_pred_samples, [0.025, 0.975])
    pi_25, pi_75 = np.quantile(sbr_pred_samples, [0.25, 0.75])
    pred_median = np.median(sbr_pred_samples)

    covered_95 = pi_025 <= sbr_test <= pi_975
    covered_50 = pi_25 <= sbr_test <= pi_75
    z_score = abs(sbr_test - pred_median) / max(SBR_SIGMA, 1e-10)

    return {
        "PATNO": patno,
        "n_train_scans": len(t_obs_train),
        "t_test_yr": float(t_test),
        "sbr_test_obs": float(sbr_test),
        "sbr_pred_median": float(pred_median),
        "pi_025": float(pi_025),
        "pi_975": float(pi_975),
        "covered_95": bool(covered_95),
        "covered_50": bool(covered_50),
        "z_score": float(z_score),
        "has_csf": csf_vals is not None and len(csf_vals) > 0,
    }


def main() -> int:
    provenance = capture_provenance(
        script_path=Path(__file__).resolve(),
        repo_root=REPO_ROOT,
        input_files=[DAT_PATH, CSF_PATH, V5_CSV],
        extra={"rng_seed": RNG_SEED, "n_prior": N_PRIOR, "sbr_sigma": SBR_SIGMA},
    )

    print("=" * 76)
    print("Block 5 — Strict Leave-Last-Scan-Out Forward Validation")
    print("=" * 76)
    git = provenance["git"]
    print(f"Git: {git.get('sha', '?')}{' (DIRTY)' if git.get('dirty') else ''}")
    print()

    dat = pd.read_parquet(DAT_PATH)
    v5 = pd.read_csv(V5_CSV)
    wave_a_pats = sorted(v5.PATNO.astype(int).tolist())

    csf_dict = load_csf_data(CSF_PATH, wave_a_pats)
    s_csf, sigma_csf = estimate_csf_params(csf_dict)
    print(f"Patients: {len(wave_a_pats)}, CSF: {len(csf_dict)}")
    print(f"S_CSF={s_csf:.1f}, σ_CSF={sigma_csf:.1f}")
    print()

    rng = np.random.default_rng(RNG_SEED)
    k_n_draws = np.exp(rng.normal(PRIOR_MU_KN, PRIOR_SD_KN, N_PRIOR))
    alpha_draws = np.exp(rng.normal(PRIOR_MU_AL, PRIOR_SD_AL, N_PRIOR))
    O_ss_draws = k_n_draws * M_SS**2 / (K_CONV + K_CLEAR_O)

    rows = []
    skipped = 0
    for i, patno in enumerate(wave_a_pats, 1):
        pat = dat[dat.PATNO == patno].sort_values("t_years")
        if len(pat) < 3:  # need >=2 train + 1 test
            skipped += 1
            continue

        t_all = pat.t_years.to_numpy(dtype=float)
        sbr_all = pat.sbr_putamen_mean.to_numpy(dtype=float)
        if np.any(~np.isfinite(t_all)) or np.any(~np.isfinite(sbr_all)):
            skipped += 1
            continue

        # Leave last scan out
        t_train = t_all[:-1]
        sbr_train = sbr_all[:-1]
        t_test = float(t_all[-1])
        sbr_test = float(sbr_all[-1])

        csf_vals = csf_dict.get(patno, None)

        res = loo_one_patient(
            patno, t_train, sbr_train, t_test, sbr_test,
            csf_vals, k_n_draws, alpha_draws, O_ss_draws,
            s_csf, sigma_csf, rng,
        )
        if "error" not in res:
            rows.append(res)

        if i % 50 == 0:
            print(f"  [{i:3d}/{len(wave_a_pats)}] processed")

    df = pd.DataFrame(rows).sort_values("PATNO").reset_index(drop=True)
    print(f"\nAnalyzed: {len(df)} patients (skipped {skipped} with <3 scans)")

    # --- Results ---
    cov_95 = df.covered_95.mean() * 100
    cov_50 = df.covered_50.mean() * 100
    med_z = df.z_score.median()

    print()
    print("=" * 76)
    print("LEAVE-LAST-SCAN-OUT FORWARD VALIDATION RESULTS")
    print("=" * 76)
    print(f"LOO 95% PI coverage:  {cov_95:.1f}%  (target L1 >= 80%)")
    print(f"LOO 50% PI coverage:  {cov_50:.1f}%  (well-calibrated ~ 50%)")
    print(f"Median |z-score|:     {med_z:.3f}  (target L3 in [0.5, 2.0])")
    print()

    # Compare with training PPC
    print(f"Training-data PPC:    99.5% (Step 2.8v5)")
    print(f"LOO - Training delta: {cov_95 - 99.5:+.1f} pp")
    print()

    # Phase 1 comparison
    phase1_loo = 93.75
    print(f"Phase 1 LOO:          {phase1_loo}%")
    print(f"Phase 2 LOO - Phase 1: {cov_95 - phase1_loo:+.1f} pp")
    print()

    # Stratify by CSF
    csf_sub = df[df.has_csf == True]  # noqa
    sbr_sub = df[df.has_csf == False]  # noqa
    if len(csf_sub) > 0:
        print(f"CSF patients (N={len(csf_sub)}): LOO coverage = {csf_sub.covered_95.mean()*100:.1f}%")
    if len(sbr_sub) > 0:
        print(f"SBR-only (N={len(sbr_sub)}): LOO coverage = {sbr_sub.covered_95.mean()*100:.1f}%")
    print()

    # Gates
    gate_l1 = cov_95 >= 80.0
    gate_l2 = cov_95 >= (phase1_loo - 10.0)
    gate_l3 = 0.5 <= med_z <= 2.0
    gates_all = gate_l1 and gate_l2 and gate_l3

    print("=" * 76)
    print("GATE EVALUATION")
    print("=" * 76)
    print(f"(L1) LOO coverage >= 80%:           {'PASS' if gate_l1 else 'FAIL'} ({cov_95:.1f}%)")
    print(f"(L2) LOO >= Phase1 - 10pp (83.75%): {'PASS' if gate_l2 else 'FAIL'} ({cov_95:.1f}%)")
    print(f"(L3) Median |z| in [0.5, 2.0]:      {'PASS' if gate_l3 else 'FAIL'} ({med_z:.3f})")
    print(f"OVERALL: {'ALL GATES PASS' if gates_all else 'ONE OR MORE GATES FAIL'}")

    # --- Persist ---
    csv_out = OUT_DIR / "step_2_8_v5_loo_forward.csv"
    df.to_csv(csv_out, index=False)

    summary = {
        "_provenance": provenance,
        "n_patients": int(len(df)),
        "n_skipped": int(skipped),
        "coverage_95": float(cov_95),
        "coverage_50": float(cov_50),
        "median_z_score": float(med_z),
        "training_ppc_95": 99.5,
        "phase1_loo_95": phase1_loo,
        "gates": {
            "L1_coverage_ge_80": bool(gate_l1),
            "L2_coverage_ge_phase1_minus_10": bool(gate_l2),
            "L3_median_z_in_range": bool(gate_l3),
            "overall_pass": bool(gates_all),
        },
    }
    json_out = OUT_DIR / "step_2_8_v5_loo_forward_summary.json"
    json_out.write_text(json.dumps(summary, indent=2, default=str))
    print(f"\nCSV: {csv_out}")
    print(f"JSON: {json_out}")

    # --- Figure ---
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    ax = axes[0]
    ax.scatter(df.sbr_test_obs, df.sbr_pred_median, s=12, alpha=0.5, color="tab:blue")
    lims = [0, max(df.sbr_test_obs.max(), df.sbr_pred_median.max()) * 1.1]
    ax.plot(lims, lims, "k--", alpha=0.4)
    ax.set_xlabel("Observed SBR (held-out last scan)")
    ax.set_ylabel("Predicted SBR (median)")
    ax.set_title(f"A. LOO Prediction (N={len(df)})\nCoverage: {cov_95:.1f}%")

    ax = axes[1]
    ax.hist(df.z_score, bins=40, color="tab:green", alpha=0.75, edgecolor="white")
    ax.axvline(med_z, color="tab:red", linewidth=2, label=f"Median |z| = {med_z:.2f}")
    ax.set_xlabel("|z-score|")
    ax.set_ylabel("# patients")
    ax.set_title("B. Calibration sharpness")
    ax.legend()

    ax = axes[2]
    bars = ax.bar(["Training PPC\n(Step 2.8v5)", "LOO Forward\n(Block 5)", "Phase 1\nLOO"],
                  [99.5, cov_95, phase1_loo],
                  color=["tab:orange", "tab:blue", "tab:gray"], alpha=0.8)
    ax.axhline(95, color="k", linestyle="--", alpha=0.4, label="95% target")
    ax.axhline(80, color="tab:red", linestyle=":", alpha=0.4, label="80% minimum")
    ax.set_ylabel("95% PI Coverage (%)")
    ax.set_title("C. Coverage comparison")
    ax.set_ylim(0, 105)
    ax.legend(fontsize=8)

    fig.suptitle("Block 5 — Leave-Last-Scan-Out Forward Validation",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig_out = OUT_DIR / "step_2_8_v5_loo_forward.png"
    fig.savefig(fig_out, dpi=300, bbox_inches="tight")
    print(f"FIG: {fig_out}")

    # RUN_MANIFEST
    manifest_out = OUT_DIR / "step_2_8_v5_loo_forward_RUN_MANIFEST.md"
    write_run_manifest(
        manifest_path=manifest_out,
        step_name="Block 5 — Leave-Last-Scan-Out Forward Validation",
        provenance=provenance,
        gate_results={
            "L1_coverage_ge_80pct": gate_l1,
            "L2_non_regression": gate_l2,
            "L3_calibration_sharpness": gate_l3,
            "overall_pass": gates_all,
        },
        summary_metrics={
            "n_patients": len(df),
            "loo_coverage_95pct": f"{cov_95:.1f}",
            "loo_coverage_50pct": f"{cov_50:.1f}",
            "median_z_score": f"{med_z:.3f}",
            "training_ppc": "99.5",
            "phase1_loo": f"{phase1_loo}",
        },
    )
    print(f"MANIFEST: {manifest_out}")

    return 0 if gates_all else 2


if __name__ == "__main__":
    raise SystemExit(main())
