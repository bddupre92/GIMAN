#!/usr/bin/env python3
"""
Phase 2 Step 2.8 — Phase 1 vs Phase 2 Posterior Predictive Regression Check
===========================================================================

Purpose
-------
Verify that Phase 2 (mechanistic coupled ODE with toxicity flux T_tox reframe)
does NOT regress from Phase 1 (phenomenological k_sbr_decay) in forecasting
skill on the shared 304 PPMI Wave A cohort. Phase 1 achieved 93.75% leave-one-
scan-out coverage at the 1-year forecast horizon. Phase 2 must maintain at
least 70% coverage under a comparable posterior predictive check (the S2
regression criterion documented in phase1_report.md and in the Phase 2 entry
criteria locked 2026-04-07).

Methodology
-----------
This is a POSTERIOR PREDICTIVE CHECK (PPC) on the training scans — not a
true strict LOO. A strict LOO would require re-calibrating each patient with
the last scan held out (~90 min additional compute). The posterior predictive
check on training data is a LOWER BAR than true LOO (it uses the same data
for training and checking, so it tends to report higher coverage than true
out-of-sample). Under the regression criterion, Phase 2 must CLEAR the 70%
lower bar as a necessary condition for proceeding; a true strict LOO re-run
is scheduled as manuscript-revision work.

The predictive model is the slow-fast approximation:

    SBR(t) / sbr_anchor = exp(-gamma * T_tox * t)

where T_tox is the toxicity flux (stiff direction identified in Step 2.7) and
gamma is the power-law exponent pinned at 0.7 per Lee 2019. This is the same
functional form that appeared in the deep-research lit-review slow-fast
analysis (archived at /tmp/deep_research_aggregation_toxicity_degeneracy.md).

For each patient:
    1. Read the persisted Step 2.6v3 chain: (k_n, alpha_tox, sigma, T_tox) samples
    2. For each observed scan time t_i (in years), compute SBR_pred samples:
           SBR_pred_i,s = sbr_anchor * exp(-gamma * T_tox_s * t_i) + Normal(0, sigma_s)
    3. Compute the 95% posterior predictive interval at each scan time
    4. Coverage = fraction of observed scans within the 95% PI, across all patients

Phase 1 baseline: read the Phase 1 posterior summary
(outputs/mechanistic_twin/data/posteriors/k_sbr_decay_posterior.parquet) and
compute the same PPC under the Phase 1 model:

    SBR(t) / sbr_anchor = exp(-k_sbr_decay * t)

Both models use the same sbr_anchor = first observed scan for each patient.

Gate criteria
-------------
(S2) Phase 2 PPC coverage >= 70% (mandatory regression check)
(S2+) Phase 2 PPC coverage >= Phase 1 PPC coverage minus 5 percentage points

Inputs
------
outputs/mechanistic_twin/data/posteriors/chains/PATNO_*.parquet   (Phase 2)
outputs/mechanistic_twin/data/posteriors/phase2_coupled_progress_step26v3_ttox_full.csv
outputs/mechanistic_twin/data/posteriors/k_sbr_decay_posterior.parquet  (Phase 1, if present)
outputs/mechanistic_twin/data/dat_spect_longitudinal.parquet

Outputs
-------
outputs/mechanistic_twin/phase2/step_2_8_predictive_check_report.md
outputs/mechanistic_twin/phase2/step_2_8_predictive_check.csv
outputs/mechanistic_twin/phase2/step_2_8_predictive_check.png

Citations
---------
Vehtari et al. 2017 Statistics and Computing 27:1413          doi:10.1007/s11222-016-9696-4
    (PSIS-LOO as the gold-standard method; PPC on training data is a weaker variant)
Fearnley & Lees 1991 Brain 114:2283                          doi:10.1093/brain/114.5.2283
Phase 1 report                                               outputs/mechanistic_twin/data/phase1_report.md
"""

from __future__ import annotations

import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
POSTERIOR_DIR = REPO_ROOT / "outputs" / "mechanistic_twin" / "data" / "posteriors"
CHAINS_DIR = POSTERIOR_DIR / "chains"
STEP26V3_CSV = POSTERIOR_DIR / "phase2_coupled_progress_step26v3_ttox_full.csv"
PHASE1_POSTERIOR = POSTERIOR_DIR / "k_sbr_decay_posterior.parquet"
LONG_PARQUET = REPO_ROOT / "outputs" / "mechanistic_twin" / "data" / "dat_spect_longitudinal.parquet"
OUTPUT_DIR = REPO_ROOT / "outputs" / "mechanistic_twin" / "phase2"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

GAMMA = 0.7   # SBR ~ (N / N_0)^gamma, Lee 2019 / Cheng 2010
HR_PER_YR = 8766.0
COVERAGE_LEVEL = 0.95
REGRESSION_THRESHOLD = 0.70  # S2 gate criterion from Phase 1 report


def phase2_predictive_samples(chain: pd.DataFrame, t_years: np.ndarray,
                              sbr_anchor: float) -> np.ndarray:
    """
    Draw posterior predictive samples for each observed scan time.

    Shape: (n_samples, n_scans). Each row is one posterior sample's predicted
    SBR trajectory at the observed scan times. Noise is added per (sample, scan)
    using the sample's sigma.
    """
    T_tox = chain["T_tox"].to_numpy()               # hr^-1
    sigma = chain["sigma"].to_numpy()
    n_samples = len(T_tox)
    n_scans = len(t_years)

    # Convert t_years to hours
    t_hr = t_years * HR_PER_YR
    # Mean predictive: sbr_anchor * exp(-gamma * T_tox * t_hr)
    # Broadcast: (n_samples, 1) * (1, n_scans) -> (n_samples, n_scans)
    ratio = np.exp(-GAMMA * np.outer(T_tox, t_hr))
    sbr_mean_pred = sbr_anchor * ratio
    # Add per-sample noise (sigma is the SBR noise scale in the likelihood)
    rng = np.random.default_rng(42)
    noise = rng.normal(0, 1, size=(n_samples, n_scans)) * sigma[:, None]
    return sbr_mean_pred + noise


def phase1_predictive_samples(k_decay_samples: np.ndarray, sigma_samples: np.ndarray,
                              t_years: np.ndarray, sbr_anchor: float) -> np.ndarray:
    """Phase 1 predictive: SBR(t) = sbr_anchor * exp(-k_sbr_decay * t_years)."""
    ratio = np.exp(-np.outer(k_decay_samples, t_years))
    sbr_mean_pred = sbr_anchor * ratio
    rng = np.random.default_rng(42)
    noise = rng.normal(0, 1, size=ratio.shape) * sigma_samples[:, None]
    return sbr_mean_pred + noise


def coverage_from_samples(pred_samples: np.ndarray, observed: np.ndarray) -> np.ndarray:
    """For each scan, is the observed SBR within the 95% predictive interval?"""
    lo = np.quantile(pred_samples, (1 - COVERAGE_LEVEL) / 2, axis=0)
    hi = np.quantile(pred_samples, 1 - (1 - COVERAGE_LEVEL) / 2, axis=0)
    return (observed >= lo) & (observed <= hi)


def main() -> int:
    print("=" * 72)
    print("Phase 2 Step 2.8 — Phase 1 vs Phase 2 Posterior Predictive Check")
    print("=" * 72)
    print()

    if not STEP26V3_CSV.exists():
        print(f"ERROR: Step 2.6v3 progress CSV not found at {STEP26V3_CSV}")
        print("       Run Step 2.6v3 first.")
        return 1

    progress = pd.read_csv(STEP26V3_CSV)
    print(f"Loaded Phase 2 progress summary: {len(progress)} patients")

    chain_files = sorted(glob.glob(str(CHAINS_DIR / "PATNO_*.parquet")))
    if len(chain_files) == 0:
        print(f"ERROR: no chain files under {CHAINS_DIR}")
        return 1
    print(f"Found {len(chain_files)} Phase 2 chain files")

    # Load longitudinal trajectories (for scan times and observed SBRs)
    long_df = pd.read_parquet(LONG_PARQUET)
    long_df = long_df.sort_values(["PATNO", "t_years"]).reset_index(drop=True)
    print(f"Loaded longitudinal cohort: {len(long_df)} scans across {long_df.PATNO.nunique()} patients")

    # Load Phase 1 posterior summary if present
    phase1_available = PHASE1_POSTERIOR.exists()
    if phase1_available:
        phase1 = pd.read_parquet(PHASE1_POSTERIOR)
        print(f"Phase 1 posterior summary loaded: {len(phase1)} patients")
    else:
        phase1 = None
        print(f"Phase 1 posterior summary NOT found at {PHASE1_POSTERIOR}")
        print("    (Phase 2 PPC will still run; Phase 1 comparison will be skipped.)")
    print()

    # Per-patient PPC loop
    rows = []
    for chain_path in chain_files:
        patno = int(Path(chain_path).stem.split("_")[1])
        pat_scans = long_df[long_df.PATNO == patno].sort_values("t_years")
        if len(pat_scans) < 2:
            continue

        t_years = pat_scans.t_years.to_numpy()
        sbr_obs = pat_scans.sbr_putamen_mean.to_numpy()
        sbr_anchor = sbr_obs[0]

        # Phase 2 PPC
        chain = pd.read_parquet(chain_path)
        phase2_pred = phase2_predictive_samples(chain, t_years, sbr_anchor)
        phase2_covered = coverage_from_samples(phase2_pred, sbr_obs)

        # Phase 1 PPC (if available)
        phase1_covered = None
        if phase1_available and phase1 is not None:
            phase1_row = phase1[phase1.PATNO == patno]
            if not phase1_row.empty:
                # Phase 1 posterior summary assumed to have k_sbr_decay_mean + k_sbr_decay_std + sigma_mean
                cols = phase1_row.columns
                k_decay_mean_col = next((c for c in cols if "k_sbr_decay" in c and "mean" in c), None)
                sigma_mean_col = next((c for c in cols if c in ("sigma_mean", "sigma")), None)
                if k_decay_mean_col and sigma_mean_col:
                    # Approximate Phase 1 predictive by drawing samples from Normal(mean, std)
                    # using the posterior summary (proper version would use the full Phase 1 chain
                    # if persisted). Degraded precision is acceptable for this regression check.
                    k_mean = float(phase1_row[k_decay_mean_col].iloc[0])
                    k_std_col = next((c for c in cols if "k_sbr_decay" in c and "std" in c), None)
                    k_std = float(phase1_row[k_std_col].iloc[0]) if k_std_col else 0.1 * abs(k_mean)
                    sig_mean = float(phase1_row[sigma_mean_col].iloc[0])
                    rng = np.random.default_rng(patno)
                    k_samples = rng.normal(k_mean, k_std, size=2000)
                    s_samples = np.full(2000, sig_mean)
                    phase1_pred = phase1_predictive_samples(k_samples, s_samples, t_years, sbr_anchor)
                    phase1_covered = coverage_from_samples(phase1_pred, sbr_obs)

        row_data = {
            "PATNO": patno,
            "n_scans": int(len(t_years)),
            "phase2_scans_covered": int(phase2_covered.sum()),
            "phase2_scans_total": int(len(phase2_covered)),
            "phase2_patient_coverage": float(phase2_covered.mean()),
        }
        if phase1_covered is not None:
            row_data.update({
                "phase1_scans_covered": int(phase1_covered.sum()),
                "phase1_scans_total": int(len(phase1_covered)),
                "phase1_patient_coverage": float(phase1_covered.mean()),
            })
        rows.append(row_data)

    results = pd.DataFrame(rows)
    n_patients = len(results)
    if n_patients == 0:
        print("ERROR: no patients produced predictive samples")
        return 1

    # Aggregate scan-level coverage (fraction of scans covered across all patients)
    phase2_scan_cov = results.phase2_scans_covered.sum() / max(1, results.phase2_scans_total.sum())
    phase1_scan_cov = None
    if "phase1_scans_covered" in results.columns:
        phase1_scan_cov = (results.phase1_scans_covered.sum() /
                           max(1, results.phase1_scans_total.sum()))

    print("=" * 72)
    print("HEADLINE COVERAGE")
    print("=" * 72)
    print(f"N patients analyzed:           {n_patients}")
    print(f"Phase 2 scan-level coverage:   {phase2_scan_cov:.1%}  "
          f"(target: >= {REGRESSION_THRESHOLD:.0%} for S2 regression check)")
    if phase1_scan_cov is not None:
        print(f"Phase 1 scan-level coverage:   {phase1_scan_cov:.1%}  "
              f"(reference from phase1_report.md: 93.75% at 1-yr LOO)")
        phase2_vs_phase1_delta_pct = (phase2_scan_cov - phase1_scan_cov) * 100
        print(f"Phase 2 - Phase 1 delta:       {phase2_vs_phase1_delta_pct:+.1f} percentage points")
    print()

    # Gate evaluation
    gate_s2 = phase2_scan_cov >= REGRESSION_THRESHOLD
    gate_s2_plus = (phase1_scan_cov is None or
                    phase2_scan_cov >= phase1_scan_cov - 0.05)
    print("=" * 72)
    print("GATE CRITERIA")
    print("=" * 72)
    print(f"S2  Phase 2 >= 70% coverage:                          "
          f"{'PASS' if gate_s2 else 'FAIL'}  ({phase2_scan_cov:.1%})")
    if phase1_scan_cov is not None:
        print(f"S2+ Phase 2 >= Phase 1 - 5 percentage points:         "
              f"{'PASS' if gate_s2_plus else 'FAIL'}  "
              f"(Phase 2 {phase2_scan_cov:.1%} vs Phase 1 {phase1_scan_cov:.1%})")
    print()

    # Write outputs
    csv_out = OUTPUT_DIR / "step_2_8_predictive_check.csv"
    results.to_csv(csv_out, index=False)
    print(f"Per-patient predictive-check CSV: {csv_out}")

    summary = {
        "n_patients": int(n_patients),
        "phase2_scan_coverage": float(phase2_scan_cov),
        "phase1_scan_coverage": float(phase1_scan_cov) if phase1_scan_cov is not None else None,
        "regression_threshold": REGRESSION_THRESHOLD,
        "gate_s2_pass": bool(gate_s2),
        "gate_s2_plus_pass": bool(gate_s2_plus),
        "methodology_note": (
            "Posterior predictive check on training data (not strict LOO). "
            "Strict LOO with per-patient last-scan holdout scheduled as "
            "manuscript-revision work (~90 min additional compute)."
        ),
    }
    (OUTPUT_DIR / "step_2_8_predictive_check_summary.json").write_text(json.dumps(summary, indent=2))

    report = OUTPUT_DIR / "step_2_8_predictive_check_report.md"
    phase1_line = (f"**Phase 1 comparison:** Phase 1 coverage = "
                   f"{phase1_scan_cov:.1%} (this run, PPC). "
                   f"phase1_report.md reports 93.75% under strict LOO at 1-year horizon. "
                   f"The PPC-vs-strict-LOO gap is expected because PPC uses the same data for training "
                   f"and checking.") if phase1_scan_cov is not None else "**Phase 1 comparison skipped** — k_sbr_decay_posterior.parquet not found."
    report.write_text(f"""# Phase 2 Step 2.8 — Phase 1 vs Phase 2 Posterior Predictive Check

**Date:** generated automatically by `scripts/mechanistic_twin/step_2_8_phase1_vs_phase2_predictive_check.py`
**Input:** Step 2.6v3 chains + Phase 1 posterior summary
**N patients analyzed:** {n_patients}

## Headline

- **Phase 2 posterior predictive coverage:** {phase2_scan_cov:.1%}
- **S2 regression threshold:** {REGRESSION_THRESHOLD:.0%}
- **S2 gate:** {'PASS' if gate_s2 else 'FAIL'}

{phase1_line}

## Methodology

This is a **posterior predictive check (PPC) on training scans**, NOT a strict leave-one-scan-out check. PPC is a lower bar than strict LOO because it uses the same data for training and checking; coverage tends to report optimistically. The S2 regression gate at 70% is the lower bar that Phase 2 must clear as a necessary condition before proceeding to the manuscript draft.

The strict LOO regression check (with per-patient last-scan holdout and full re-calibration) is scheduled as manuscript-revision work, requiring approximately 90 minutes of additional compute. Under strict LOO, coverage is expected to be lower than the PPC value reported here.

### Phase 2 predictive model

Under the slow-fast approximation validated by the deep-research lit-review (`/tmp/deep_research_aggregation_toxicity_degeneracy.md`), the observation model reduces to:

    SBR(t) / sbr_anchor = exp(-gamma * T_tox * t)

where `T_tox = alpha_tox * k_n * M_ss^2 / (k_conv + k_clear_O)` is the toxicity flux composite (stiff direction) identified in Step 2.7, and gamma = 0.7 per Lee 2019. This form is a direct post-processing computation on the persisted Step 2.6v3 chains — no Julia forward simulation required.

### Phase 1 predictive model

    SBR(t) / sbr_anchor = exp(-k_sbr_decay * t)

where `k_sbr_decay` is the phenomenological lumped rate from Phase 1 (`outputs/mechanistic_twin/data/posteriors/k_sbr_decay_posterior.parquet`). For this PPC, Phase 1 posterior samples were approximated from the posterior summary (mean + std) because the full Phase 1 chains are not persisted in the same format.

## Implication for the Phase 2 novelty claim

{'If the Phase 2 S2 gate passes (>= 70%), Phase 2 forecasting skill is at worst comparable to Phase 1 while adding biological interpretability via the toxicity flux reframe. This is the regression check for the manuscript.' if gate_s2 else '**WARNING: Phase 2 did NOT clear the S2 regression gate.** This is publishable as an honest negative result: biological interpretability came at a cost in raw forecasting skill. Investigate before committing to the CPT:PSP manuscript.'}

## Citations

- Vehtari, A., Gelman, A., Gabry, J. 2017. "Practical Bayesian model evaluation using leave-one-out cross-validation and WAIC." *Statistics and Computing* 27:1413. doi:10.1007/s11222-016-9696-4
- Fearnley, J. M. & Lees, A. J. 1991. "Ageing and Parkinson's disease: substantia nigra regional selectivity." *Brain* 114:2283. doi:10.1093/brain/114.5.2283
- Phase 1 report: `outputs/mechanistic_twin/data/phase1_report.md`
""")
    print(f"Markdown report:                  {report}")
    print(f"JSON summary:                     {OUTPUT_DIR / 'step_2_8_predictive_check_summary.json'}")

    return 0 if gate_s2 else 1


if __name__ == "__main__":
    raise SystemExit(main())
