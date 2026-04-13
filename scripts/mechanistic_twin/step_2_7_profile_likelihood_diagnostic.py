#!/usr/bin/env python3
"""
Phase 2 Step 2.7 — Profile-Likelihood Practical Identifiability Diagnostic
===========================================================================

Purpose
-------
Diagnose the practical identifiability of (alpha_tox, k_n) in the Variant B
coupled alpha-synuclein aggregation + dopaminergic neuron death ODE, as
calibrated in Phase 2 Step 2.6v3 from longitudinal DaT-SPECT.

Methodology
-----------
The canonical diagnostic is the profile likelihood of Raue et al. 2009
*Bioinformatics* 25:1923, doi:10.1093/bioinformatics/btp358. A FLAT profile
along alpha_tox (with k_n re-optimized at each grid point) indicates
practical non-identifiability, even when the parameter is globally
structurally identifiable in the sense of Villaverde et al. 2016 *PLoS Comp
Biol* 12:e1005153.

Because Step 2.6v3 persists the full joint (k_n, alpha_tox, sigma, T_tox)
chains per patient, we can approximate the Raue 2009 profile via the
marginal posterior width of log(alpha_tox) and the joint posterior
correlation cor(log alpha_tox, log k_n). Under the slow-fast timescale
degeneracy identified in the deep-research lit-review (archived at
/tmp/deep_research_aggregation_toxicity_degeneracy.md), we predict:

  1. The marginal posterior on log(alpha_tox) spans multiple decades for
     the 54/304 Step 2.6v2 convergence failures (all but 3 are 4-scan
     patients) — because the data does not constrain alpha_tox individually.
  2. The joint posterior correlation cor(log alpha_tox, log k_n) is either
     (a) strongly negative (sloppy ridge) or (b) near zero (prior-dominated
     isotropic diffusion). Either pattern is consistent with practical non-
     identifiability.
  3. The marginal posterior on log(T_tox) is MUCH TIGHTER than on
     log(alpha_tox) because T_tox is the stiff direction in the
     Gutenkunst/Transtrum sloppy-models sense. The 54/304 under-converged
     patients are expected to converge CLEANLY on T_tox even though they
     are under-converged on alpha_tox individually.

Gate criteria
-------------
The Step 2.7 diagnostic PASSES if:

  (i)  T_tox posterior width is SMALLER than alpha_tox posterior width
       (measured as log-scale 95% CI span) for >= 90% of patients
  (ii) The converged-subset (max R-hat < 1.05 on k_n, alpha_tox, sigma)
       median T_tox implied neuron loss rate lands inside [2, 5] %/yr
       (Fearnley & Lees 1991 canonical range)
 (iii) For the 54/304 Step 2.6v2 under-converged patients, the T_tox
       posterior is informative (log-scale 95% CI span < 2 decades) even
       though their alpha_tox posteriors are diffuse

Inputs
------
outputs/mechanistic_twin/data/posteriors/chains/PATNO_XXXX.parquet
    Per-patient joint (k_n, alpha_tox, sigma, T_tox) chain files from Step 2.6v3.
outputs/mechanistic_twin/data/posteriors/phase2_coupled_progress_step26v3_ttox_full.csv
    Per-patient posterior summary with convergence diagnostics.
outputs/mechanistic_twin/data/dat_spect_longitudinal.parquet
    n_scans per patient for stratification.

Outputs
-------
outputs/mechanistic_twin/phase2/step_2_7_profile_likelihood_report.md
    Markdown report with headline findings.
outputs/mechanistic_twin/phase2/step_2_7_profile_likelihood.csv
    Per-patient diagnostic metrics (95% CI spans, correlations, ess).
outputs/mechanistic_twin/phase2/step_2_7_profile_likelihood.png
    2x2 figure: (a) alpha_tox vs T_tox posterior width scatter,
                (b) cor(log alpha, log k_n) histogram,
                (c) T_tox implied %/yr distribution stratified by convergence,
                (d) marginal posterior width ratio log(alpha_tox_ci/T_tox_ci).

Citations
---------
Raue et al. 2009 Bioinformatics 25:1923           doi:10.1093/bioinformatics/btp358
Gutenkunst et al. 2007 PLoS Comp Biol 3:e189      doi:10.1371/journal.pcbi.0030189
Transtrum et al. 2015 J Chem Phys 143:010901     doi:10.1063/1.4923066
Villaverde et al. 2016 PLoS Comp Biol 12:e1005153 doi:10.1371/journal.pcbi.1005153
Fearnley & Lees 1991 Brain 114:2283              doi:10.1093/brain/114.5.2283
"""

from __future__ import annotations

import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd

# ------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[2]
POSTERIOR_DIR = REPO_ROOT / "outputs" / "mechanistic_twin" / "data" / "posteriors"
CHAINS_DIR = POSTERIOR_DIR / "chains"
PROGRESS_CSV = POSTERIOR_DIR / "phase2_coupled_progress_step26v3_ttox_full.csv"
LONG_PARQUET = REPO_ROOT / "outputs" / "mechanistic_twin" / "data" / "dat_spect_longitudinal.parquet"
OUTPUT_DIR = REPO_ROOT / "outputs" / "mechanistic_twin" / "phase2"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Per-year conversion factor for T_tox (hr^-1) -> fractional neuron loss per year
HR_PER_YR = 8766.0
# Gate criterion ranges
FEARNLEY_LEES_LOW = 2.0   # %/yr
FEARNLEY_LEES_HIGH = 5.0  # %/yr
# T_tox informative threshold (log-scale 95% CI span)
T_TOX_INFORMATIVE_DECADES = 2.0


# ------------------------------------------------------------------
# Per-patient diagnostic computation
# ------------------------------------------------------------------
def _log_ci_span(samples: np.ndarray) -> float:
    """Return log10 width of the 95% credible interval, skipping non-positive samples."""
    pos = samples[samples > 0]
    if len(pos) < 2:
        return np.nan
    lo = np.quantile(pos, 0.025)
    hi = np.quantile(pos, 0.975)
    if lo <= 0 or hi <= 0:
        return np.nan
    return float(np.log10(hi / lo))


def patient_diagnostics(chain: pd.DataFrame, patno: int, n_scans: int,
                        max_rhat: float) -> dict:
    """Compute profile-likelihood-analog diagnostics for one patient."""
    k_n = chain["k_n"].to_numpy()
    alpha = chain["alpha_tox"].to_numpy()
    T_tox = chain["T_tox"].to_numpy()

    # Marginal posterior widths (95% CI in log10 decades)
    k_n_ci = _log_ci_span(k_n)
    alpha_ci = _log_ci_span(alpha)
    T_tox_ci = _log_ci_span(T_tox)

    # Posterior correlation on log scale (meaningful for positive-constrained params)
    logk = np.log(k_n[k_n > 0])
    loga = np.log(alpha[alpha > 0])
    n_common = min(len(logk), len(loga))
    if n_common > 10:
        log_cor = float(np.corrcoef(
            np.log(k_n[:n_common][k_n[:n_common] > 0]),
            np.log(alpha[:n_common][alpha[:n_common] > 0]),
        )[0, 1])
    else:
        log_cor = np.nan

    # T_tox implied annual neuron loss rate
    T_tox_median_hr = float(np.median(T_tox))
    T_tox_mean_hr = float(np.mean(T_tox))
    pct_loss_median = (1.0 - np.exp(-T_tox_median_hr * HR_PER_YR)) * 100.0
    pct_loss_mean = (1.0 - np.exp(-T_tox_mean_hr * HR_PER_YR)) * 100.0

    # Stiff/sloppy ratio: how much tighter is T_tox than alpha_tox (in log decades)?
    stiff_sloppy_ratio = (alpha_ci - T_tox_ci) if (not np.isnan(alpha_ci) and not np.isnan(T_tox_ci)) else np.nan

    return {
        "PATNO": patno,
        "n_scans": n_scans,
        "max_rhat": max_rhat,
        "converged": max_rhat < 1.05,
        "n_samples": len(chain),
        # Marginal posterior 95% CI spans (decades on log10 scale)
        "k_n_ci_decades": k_n_ci,
        "alpha_tox_ci_decades": alpha_ci,
        "T_tox_ci_decades": T_tox_ci,
        # Degeneracy diagnostic: strong negative = sloppy ridge; near 0 = prior-dominated
        "log_kn_alpha_correlation": log_cor,
        # Stiff direction identification
        "T_tox_median_hr": T_tox_median_hr,
        "T_tox_mean_hr": T_tox_mean_hr,
        "pct_loss_per_yr_median": pct_loss_median,
        "pct_loss_per_yr_mean": pct_loss_mean,
        # Raue 2009 sloppy - stiff signature: alpha_ci >> T_tox_ci
        "stiff_minus_sloppy_decades": stiff_sloppy_ratio,
        # Gate criterion check per patient
        "T_tox_informative": (T_tox_ci < T_TOX_INFORMATIVE_DECADES) if not np.isnan(T_tox_ci) else False,
    }


# ------------------------------------------------------------------
# Main analysis
# ------------------------------------------------------------------
def main() -> int:
    print("=" * 72)
    print("Phase 2 Step 2.7 — Profile-Likelihood Practical Identifiability Diagnostic")
    print("=" * 72)
    print()

    if not PROGRESS_CSV.exists():
        print(f"ERROR: progress CSV not found at {PROGRESS_CSV}")
        print("       Run Step 2.6v3 first via calibrate_phase2_coupled.jl with --run-tag step26v3_ttox_full")
        return 1

    progress = pd.read_csv(PROGRESS_CSV)
    progress["max_rhat"] = progress[["k_n_rhat", "alpha_tox_rhat", "sigma_rhat"]].max(axis=1)
    print(f"Loaded {len(progress)} patients from Step 2.6v3 progress CSV")

    chain_files = sorted(glob.glob(str(CHAINS_DIR / "PATNO_*.parquet")))
    print(f"Found {len(chain_files)} persisted chain Parquet files")
    print()

    if len(chain_files) == 0:
        print(f"ERROR: no chain files found under {CHAINS_DIR}")
        return 1

    # Per-patient diagnostics
    rows = []
    for chain_path in chain_files:
        patno = int(Path(chain_path).stem.split("_")[1])
        row = progress[progress.PATNO == patno]
        if row.empty:
            continue
        n_scans = int(row.n_scans.iloc[0])
        max_rhat = float(row.max_rhat.iloc[0])
        chain = pd.read_parquet(chain_path)
        rows.append(patient_diagnostics(chain, patno, n_scans, max_rhat))

    diag = pd.DataFrame(rows).sort_values("PATNO").reset_index(drop=True)
    n_total = len(diag)
    n_converged = int(diag.converged.sum())
    print(f"Computed diagnostics for {n_total} patients ({n_converged} converged at R-hat<1.05)")
    print()

    # ------------------------------------------------------------------
    # Gate criterion evaluation
    # ------------------------------------------------------------------
    # (i) T_tox tighter than alpha_tox for >= 90% of patients
    tighter_frac = (diag.stiff_minus_sloppy_decades > 0).sum() / max(1, n_total)
    gate_i = tighter_frac >= 0.90

    # (ii) converged-subset median implied pct/yr in [2, 5]
    conv_median_pct = diag[diag.converged].pct_loss_per_yr_median.median()
    gate_ii = FEARNLEY_LEES_LOW <= conv_median_pct <= FEARNLEY_LEES_HIGH

    # (iii) T_tox informative even for under-converged patients
    underconv = diag[~diag.converged]
    underconv_informative_frac = (underconv.T_tox_informative.sum() / max(1, len(underconv))) if len(underconv) > 0 else 1.0
    gate_iii = underconv_informative_frac >= 0.80

    gate_all = gate_i and gate_ii and gate_iii

    print("=" * 72)
    print("GATE CRITERIA (Step 2.7 profile-likelihood diagnostic)")
    print("=" * 72)
    print(f"(i)   T_tox_ci < alpha_tox_ci for >= 90% of patients: "
          f"{'PASS' if gate_i else 'FAIL'} ({tighter_frac:.1%})")
    print(f"(ii)  Converged median pct/yr in [2, 5]%: "
          f"{'PASS' if gate_ii else 'FAIL'} ({conv_median_pct:.2f}%/yr)")
    print(f"(iii) T_tox informative (<2 decades) for >= 80% of R-hat>1.05 patients: "
          f"{'PASS' if gate_iii else 'FAIL'} ({underconv_informative_frac:.1%} of {len(underconv)})")
    print()
    print(f"OVERALL: {'ALL GATES PASS' if gate_all else 'ONE OR MORE GATES FAIL'}")
    print()

    # ------------------------------------------------------------------
    # Stratified summaries
    # ------------------------------------------------------------------
    print("=" * 72)
    print("STRATIFIED SUMMARIES")
    print("=" * 72)

    for label, subset in [("ALL", diag), ("CONVERGED (max R-hat < 1.05)", diag[diag.converged]),
                          ("UNDER-CONVERGED", diag[~diag.converged])]:
        if len(subset) == 0:
            continue
        print(f"\n{label}  (N = {len(subset)})")
        print(f"  alpha_tox 95% CI span  median = {subset.alpha_tox_ci_decades.median():.2f} decades  "
              f"p25-p75 = [{subset.alpha_tox_ci_decades.quantile(0.25):.2f}, {subset.alpha_tox_ci_decades.quantile(0.75):.2f}]")
        print(f"  k_n 95% CI span        median = {subset.k_n_ci_decades.median():.2f} decades  "
              f"p25-p75 = [{subset.k_n_ci_decades.quantile(0.25):.2f}, {subset.k_n_ci_decades.quantile(0.75):.2f}]")
        print(f"  T_tox 95% CI span      median = {subset.T_tox_ci_decades.median():.2f} decades  "
              f"p25-p75 = [{subset.T_tox_ci_decades.quantile(0.25):.2f}, {subset.T_tox_ci_decades.quantile(0.75):.2f}]")
        print(f"  log k_n-alpha correl.  median = {subset.log_kn_alpha_correlation.median():+.3f}  "
              f"p25-p75 = [{subset.log_kn_alpha_correlation.quantile(0.25):+.3f}, {subset.log_kn_alpha_correlation.quantile(0.75):+.3f}]")
        print(f"  T_tox pct/yr (median)  median = {subset.pct_loss_per_yr_median.median():.2f}%/yr  "
              f"p25-p75 = [{subset.pct_loss_per_yr_median.quantile(0.25):.2f}, {subset.pct_loss_per_yr_median.quantile(0.75):.2f}]%/yr")
    print()

    # ------------------------------------------------------------------
    # Write outputs
    # ------------------------------------------------------------------
    csv_out = OUTPUT_DIR / "step_2_7_profile_likelihood.csv"
    diag.to_csv(csv_out, index=False)
    print(f"Per-patient diagnostics written to: {csv_out}")

    report_out = OUTPUT_DIR / "step_2_7_profile_likelihood_report.md"
    with report_out.open("w") as f:
        f.write(f"""# Phase 2 Step 2.7 — Profile-Likelihood Practical Identifiability Diagnostic

**Date:** generated automatically by `scripts/mechanistic_twin/step_2_7_profile_likelihood_diagnostic.py`
**Input:** Step 2.6v3 joint chains (`outputs/mechanistic_twin/data/posteriors/chains/PATNO_*.parquet`)
**N patients analyzed:** {n_total}
**N converged (R-hat < 1.05):** {n_converged}

## Gate criteria

| Gate | Criterion | Result | Value |
|---|---|---|---|
| (i)   | T_tox 95% CI tighter than alpha_tox for >= 90% of patients | {'PASS' if gate_i else 'FAIL'} | {tighter_frac:.1%} |
| (ii)  | Converged-subset median implied neuron loss in [2%, 5%]/yr (Fearnley & Lees 1991 canonical range) | {'PASS' if gate_ii else 'FAIL'} | {conv_median_pct:.2f}%/yr |
| (iii) | T_tox informative (<2 decades log CI) for >= 80% of R-hat>1.05 patients | {'PASS' if gate_iii else 'FAIL'} | {underconv_informative_frac:.1%} of {len(underconv)} under-converged |

**Overall verdict:** {'ALL GATES PASS' if gate_all else 'ONE OR MORE GATES FAIL'}

## Converged-subset headline

- N = {n_converged} patients with max R-hat < 1.05
- Median T_tox implied neuron loss rate: **{conv_median_pct:.2f}%/yr**
- p25-p75: [{diag[diag.converged].pct_loss_per_yr_median.quantile(0.25):.2f}%, {diag[diag.converged].pct_loss_per_yr_median.quantile(0.75):.2f}%]/yr
- Fearnley & Lees 1991 canonical range: 2-5%/yr for active PD

## Sloppy-direction diagnostic

The median posterior correlation cor(log alpha_tox, log k_n) across all patients is **{diag.log_kn_alpha_correlation.median():+.3f}**.
- Strong negative (<-0.5) would indicate a textbook sloppy-ridge degeneracy along the T_tox = const manifold.
- Near zero would indicate prior-dominated isotropic diffusion — both parameters individually weakly identified.

## Stiff direction (T_tox) posterior tightness

Median across all patients: **{diag.T_tox_ci_decades.median():.2f} decades (95% log CI)** vs alpha_tox median **{diag.alpha_tox_ci_decades.median():.2f} decades**.

A ratio > 1 confirms T_tox is the **stiff direction** per Gutenkunst et al. 2007 / Transtrum et al. 2015 sloppy-models framework.

## Citations supporting the diagnostic

- Raue, A. et al. 2009. "Structural and practical identifiability analysis of partially observed dynamical models by exploiting the profile likelihood." *Bioinformatics* 25:1923. doi:10.1093/bioinformatics/btp358
- Gutenkunst, R. N. et al. 2007. "Universally sloppy parameter sensitivities in systems biology models." *PLoS Comp Biol* 3:e189. doi:10.1371/journal.pcbi.0030189
- Transtrum, M. K. et al. 2015. "Perspective: Sloppiness and emergent theories in physics, biology, and beyond." *J Chem Phys* 143:010901. doi:10.1063/1.4923066
- Villaverde, A. F. et al. 2016. "Structural Identifiability of Dynamic Systems Biology Models." *PLoS Comp Biol* 12:e1005153. doi:10.1371/journal.pcbi.1005153
- Fearnley, J. M. & Lees, A. J. 1991. "Ageing and Parkinson's disease: substantia nigra regional selectivity." *Brain* 114:2283. doi:10.1093/brain/114.5.2283
""")
    print(f"Report written to: {report_out}")

    # Try to produce the 2x2 figure (non-fatal if matplotlib missing)
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 2, figsize=(12, 9))

        ax = axes[0, 0]
        ax.scatter(diag.alpha_tox_ci_decades, diag.T_tox_ci_decades, s=20, alpha=0.6,
                   c=["tab:blue" if c else "tab:red" for c in diag.converged])
        lim = max(diag.alpha_tox_ci_decades.max(), diag.T_tox_ci_decades.max()) * 1.05
        ax.plot([0, lim], [0, lim], "k--", alpha=0.4, label="y = x (equal width)")
        ax.set_xlabel("alpha_tox 95% CI width (log10 decades)")
        ax.set_ylabel("T_tox 95% CI width (log10 decades)")
        ax.set_title("(a) Stiff (T_tox) vs sloppy (alpha_tox) posterior widths\nBelow y=x = T_tox tighter (good)")
        ax.legend(loc="upper left")

        ax = axes[0, 1]
        ax.hist(diag.log_kn_alpha_correlation.dropna(), bins=40, color="tab:purple", alpha=0.7, edgecolor="k")
        ax.axvline(0, color="k", linestyle="--", alpha=0.5, label="0 (uncorrelated)")
        ax.axvline(-0.5, color="tab:red", linestyle=":", alpha=0.5, label="-0.5 (sloppy ridge)")
        ax.set_xlabel("cor(log alpha_tox, log k_n)")
        ax.set_ylabel("Number of patients")
        ax.set_title("(b) Degeneracy signature\nStrong negative = sloppy ridge")
        ax.legend(loc="upper left")

        ax = axes[1, 0]
        conv_rates = diag[diag.converged].pct_loss_per_yr_median.dropna()
        under_rates = diag[~diag.converged].pct_loss_per_yr_median.dropna()
        bins = np.logspace(np.log10(0.1), np.log10(200), 40)
        ax.hist(conv_rates, bins=bins, alpha=0.6, label=f"Converged (N={len(conv_rates)})", color="tab:blue")
        if len(under_rates) > 0:
            ax.hist(under_rates, bins=bins, alpha=0.6, label=f"Under-converged (N={len(under_rates)})", color="tab:red")
        ax.axvspan(FEARNLEY_LEES_LOW, FEARNLEY_LEES_HIGH, alpha=0.2, color="tab:green",
                   label="Fearnley & Lees 1991\ncanonical 2-5%/yr")
        ax.set_xscale("log")
        ax.set_xlabel("Implied neuron loss rate (%/yr), T_tox median")
        ax.set_ylabel("Number of patients")
        ax.set_title("(c) T_tox implied neuron loss, stratified by convergence")
        ax.legend(loc="upper right", fontsize=8)

        ax = axes[1, 1]
        stiff_sloppy = diag.stiff_minus_sloppy_decades.dropna()
        ax.hist(stiff_sloppy, bins=40, color="tab:green", alpha=0.7, edgecolor="k")
        ax.axvline(0, color="k", linestyle="--", alpha=0.5, label="0 (equal width)")
        median_ss = stiff_sloppy.median()
        ax.axvline(median_ss, color="tab:orange", linestyle="-", alpha=0.8,
                   label=f"median = {median_ss:+.2f}")
        ax.set_xlabel("alpha_tox CI - T_tox CI (log10 decades)\npositive = T_tox tighter (stiff direction)")
        ax.set_ylabel("Number of patients")
        ax.set_title("(d) Stiff-direction advantage\n(how much tighter T_tox is than alpha_tox)")
        ax.legend(loc="upper right")

        fig.suptitle(f"Phase 2 Step 2.7 — Profile-Likelihood Diagnostic (N = {n_total})",
                     fontsize=14, fontweight="bold")
        fig.tight_layout()
        fig_path = OUTPUT_DIR / "step_2_7_profile_likelihood.png"
        fig.savefig(fig_path, dpi=150, bbox_inches="tight")
        print(f"Figure saved to: {fig_path}")
    except ImportError:
        print("matplotlib not available; skipping figure generation")

    # JSON summary for programmatic access
    summary = {
        "n_total": int(n_total),
        "n_converged": int(n_converged),
        "gate_i_pass": bool(gate_i),
        "gate_i_value": float(tighter_frac),
        "gate_ii_pass": bool(gate_ii),
        "gate_ii_value_pct_yr": float(conv_median_pct),
        "gate_iii_pass": bool(gate_iii),
        "gate_iii_value": float(underconv_informative_frac),
        "overall_verdict": "PASS" if gate_all else "FAIL",
        "median_log_kn_alpha_cor": float(diag.log_kn_alpha_correlation.median()),
        "median_alpha_ci_decades": float(diag.alpha_tox_ci_decades.median()),
        "median_T_tox_ci_decades": float(diag.T_tox_ci_decades.median()),
    }
    (OUTPUT_DIR / "step_2_7_profile_likelihood_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"JSON summary written to: {OUTPUT_DIR / 'step_2_7_profile_likelihood_summary.json'}")

    return 0 if gate_all else 1


if __name__ == "__main__":
    raise SystemExit(main())
