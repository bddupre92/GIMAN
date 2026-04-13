#!/usr/bin/env python3
"""
Phase 2 Step 2.8v4 — Phase 1 vs Phase 2 Posterior Predictive Check (IS-posterior edition)
==========================================================================================

Purpose
-------
Verify that Phase 2 (mechanistic coupled ODE with the slow-fast T_tox
reframe, calibrated via IS on the Variant B decay in Step 2.6v4) does not
regress from Phase 1 (phenomenological k_sbr_decay, Phase 1 93.75% LOO
coverage) in forecasting skill on the shared 304 PPMI Wave A cohort.

This is the v4 rerun of the original step_2_8 script that read Step 2.6v3
NUTS chains. The 2.6v3 chains were audited on 2026-04-09 and found to be
prior-dominated (sampler-failure), so the original 2.8 output is invalid.
This v4 reads the IS-weighted chains at `chains_is/PATNO_*.parquet`.

Gate criteria (S2 regression check from phase1_report.md)
---------------------------------------------------------
  (S2)  Phase 2 PPC scan-level coverage >= 70% (mandatory)
  (S2+) Phase 2 PPC coverage >= Phase 1 PPC coverage - 5 percentage points
        (non-degradation relative to the Phase 1 baseline)
  (S2++) HIGH-INFO subset PPC coverage >= 80%
        (the data-informed subset should achieve stronger coverage since
        the posterior is not prior-dominated)

Methodology
-----------
Posterior predictive check on training scans (not strict LOO). For each
patient:
  1. Read IS-resampled chain (k_n, alpha_tox, T_tox) — 5000 weighted draws
  2. For each observed scan time t_i, compute predictive samples:
         SBR_pred_i,s = sbr_anchor * exp(-GAMMA * T_tox_s * t_i_hr) + N(0, SBR_SIGMA)
     where SBR_SIGMA=0.20 is the fixed observation noise from Step 2.6v4.
  3. 95% posterior predictive interval at each scan time.
  4. Coverage = observed SBR in the PI.

Reproducibility
---------------
Deterministic given fixed RNG seed + IS chains. Follows the closed-loop
v1.0 reproducibility rule: provenance header, RUN_MANIFEST companion,
output hashes embedded in JSON.
"""
from __future__ import annotations

import hashlib
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

REPO_ROOT = Path(__file__).resolve().parents[2]
POST_DIR  = REPO_ROOT / "outputs/mechanistic_twin/data/posteriors"
CHAINS_IS = POST_DIR / "chains_is"
V4_SUMMARY = POST_DIR / "phase2_coupled_is_step26v4.csv"
PHASE1_POSTERIOR = POST_DIR / "k_sbr_decay_posterior.parquet"
LONG_PARQUET = REPO_ROOT / "outputs/mechanistic_twin/data/dat_spect_longitudinal.parquet"
OUT_DIR = REPO_ROOT / "outputs/mechanistic_twin/phase2"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Physical / pipeline constants (must match step_2_6_v4)
GAMMA = 0.7
HR_PER_YR = 8766.0
SBR_SIGMA = 0.20
# Phase 1 uses Truncated(Normal(0.15, 0.1), 0.01, 0.5) for sigma (sampled per
# patient), but calibrate_neuron_death.jl does NOT persist the sigma posterior
# per-patient — only k_sbr_decay columns land in the parquet. We approximate
# Phase 1's per-patient sigma by the prior mode for this comparison, which is
# a reasonable proxy for a well-fit patient. Acknowledged approximation.
PHASE1_SIGMA_PRIOR_MEAN = 0.15
COVERAGE_LEVEL = 0.95
REGRESSION_THRESHOLD = 0.70
PHASE1_MINUS_5_TOLERANCE = 0.05
HIGH_INFO_COVERAGE_THRESHOLD = 0.80
HIGH_INFO_ESS_CUTOFF = 0.20
RNG_SEED = 202604092  # distinct from step_2_6_v4's seed


def _sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(2**20), b""):
            h.update(chunk)
    return h.hexdigest()


def phase2_ppc_samples(chain: pd.DataFrame, t_years: np.ndarray,
                        sbr_anchor: float, rng: np.random.Generator) -> np.ndarray:
    """Phase 2 predictive: SBR(t) = sbr_anchor * exp(-γ·T_tox·t_hr) + N(0,σ)."""
    T_tox = chain["T_tox"].to_numpy()
    n_samples = len(T_tox)
    t_hr = t_years * HR_PER_YR
    decay = np.clip(-GAMMA * np.outer(T_tox, t_hr), -50.0, 0.0)
    ratio = np.exp(decay)
    sbr_mean = sbr_anchor * ratio
    noise = rng.normal(0.0, SBR_SIGMA, size=sbr_mean.shape)
    return sbr_mean + noise


def phase1_ppc_samples(k_decay_mean: float, k_decay_std: float, sigma_mean: float,
                        t_years: np.ndarray, sbr_anchor: float,
                        rng: np.random.Generator, n_draws: int = 2000) -> np.ndarray:
    """Phase 1 predictive: SBR(t) = sbr_anchor * exp(-k_sbr_decay·t_years) + N(0,σ).

    Samples k_sbr_decay from a Normal(mean, std) proxy for the Phase 1
    posterior summary (degraded precision vs the full Phase 1 chain, but
    adequate for the S2 regression check).
    """
    k_draws = rng.normal(k_decay_mean, max(k_decay_std, 1e-6), size=n_draws)
    k_draws = np.clip(k_draws, 0.001, 2.0)  # physically valid range
    ratio = np.exp(-np.outer(k_draws, t_years))
    sbr_mean = sbr_anchor * ratio
    noise = rng.normal(0.0, sigma_mean, size=sbr_mean.shape)
    return sbr_mean + noise


def coverage_from_samples(pred_samples: np.ndarray, observed: np.ndarray,
                            level: float = COVERAGE_LEVEL) -> np.ndarray:
    lo = np.quantile(pred_samples, (1 - level) / 2, axis=0)
    hi = np.quantile(pred_samples, 1 - (1 - level) / 2, axis=0)
    return (observed >= lo) & (observed <= hi)


def calibration_z_ratio(pred_samples: np.ndarray, observed: np.ndarray) -> float:
    """Ratio of empirical residual RMS to predictive SD.

    A well-calibrated model has ratio ≈ 1.0. Ratio ≪ 1 → bands too wide
    (uninformative coverage). Ratio ≫ 1 → bands too narrow (bad fit).
    """
    mean_pred = pred_samples.mean(axis=0)
    sd_pred   = pred_samples.std(axis=0)
    resid = observed - mean_pred
    if np.any(sd_pred < 1e-12):
        return np.nan
    z = resid / sd_pred
    return float(np.sqrt(np.mean(z ** 2)))


def main() -> int:
    print("=" * 76)
    print("Phase 2 Step 2.8v4 — PPC on IS posterior (Phase 1 vs Phase 2)")
    print("=" * 76)

    # --- Provenance header (closed-loop v1.0 rule) ---
    input_files = [V4_SUMMARY, LONG_PARQUET]
    if PHASE1_POSTERIOR.exists():
        input_files.append(PHASE1_POSTERIOR)
    # Also hash the first 3 IS chains as a sentinel for chain-tree consistency
    sample_chains = sorted(CHAINS_IS.glob("PATNO_*.parquet"))[:3]
    input_files += sample_chains

    provenance = capture_provenance(
        script_path=Path(__file__).resolve(),
        repo_root=REPO_ROOT,
        input_files=input_files,
        extra={
            "rng_seed": RNG_SEED,
            "gamma": GAMMA,
            "sbr_sigma": SBR_SIGMA,
            "coverage_level": COVERAGE_LEVEL,
            "thresholds": {
                "S2_mandatory":    REGRESSION_THRESHOLD,
                "S2plus_tolerance": PHASE1_MINUS_5_TOLERANCE,
                "S2plusplus_high_info": HIGH_INFO_COVERAGE_THRESHOLD,
                "high_info_ess_cutoff": HIGH_INFO_ESS_CUTOFF,
            },
            "chains_dir": str(CHAINS_IS.relative_to(REPO_ROOT)),
        },
    )
    git = provenance["git"]
    print(f"Git:     {git.get('sha','?')}{' (DIRTY)' if git.get('dirty') else ''}")
    print(f"Python:  {provenance['python']['version']}  numpy={provenance['packages']['numpy']}")
    print(f"Script:  {provenance['script']['sha256'][:16]}...")
    print()

    if not V4_SUMMARY.exists():
        print(f"ERROR: {V4_SUMMARY} not found — run step_2_6_v4 first.")
        return 1

    summary_v4 = pd.read_csv(V4_SUMMARY)
    long_df = pd.read_parquet(LONG_PARQUET).sort_values(["PATNO", "t_years"]).reset_index(drop=True)
    chain_files = sorted(CHAINS_IS.glob("PATNO_*.parquet"))
    print(f"IS chains:     {len(chain_files)}")
    print(f"v4 summary:    {len(summary_v4)} patients")
    print(f"Long scans:    {len(long_df)} across {long_df.PATNO.nunique()} patients")

    phase1_available = PHASE1_POSTERIOR.exists()
    phase1 = None
    if phase1_available:
        phase1 = pd.read_parquet(PHASE1_POSTERIOR)
        print(f"Phase 1 post:  {len(phase1)} patients")
    else:
        print(f"Phase 1 post:  NOT FOUND at {PHASE1_POSTERIOR} — phase1 comparison skipped")
    print()

    # --- Per-patient PPC loop ---
    rows = []
    for cf in chain_files:
        patno = int(cf.stem.split("_")[1])
        pat_scans = long_df[long_df.PATNO == patno].sort_values("t_years")
        if len(pat_scans) < 2:
            continue

        t_years = pat_scans.t_years.to_numpy(dtype=float)
        sbr_obs = pat_scans.sbr_putamen_mean.to_numpy(dtype=float)
        sbr_anchor = float(sbr_obs[0])

        # Phase 2 PPC (IS chain)
        chain = pd.read_parquet(cf)
        rng2 = np.random.default_rng(RNG_SEED + patno)
        phase2_pred = phase2_ppc_samples(chain, t_years, sbr_anchor, rng2)
        phase2_covered_95 = coverage_from_samples(phase2_pred, sbr_obs, level=0.95)
        phase2_covered_50 = coverage_from_samples(phase2_pred, sbr_obs, level=0.50)
        phase2_zratio = calibration_z_ratio(phase2_pred, sbr_obs)

        row = {
            "PATNO": patno,
            "n_scans": int(len(t_years)),
            "phase2_scans_covered": int(phase2_covered_95.sum()),
            "phase2_scans_total":   int(len(phase2_covered_95)),
            "phase2_patient_coverage": float(phase2_covered_95.mean()),
            "phase2_scans_covered_50": int(phase2_covered_50.sum()),
            "phase2_patient_coverage_50": float(phase2_covered_50.mean()),
            "phase2_zratio": phase2_zratio,
        }

        # ess_frac from v4 summary (for HIGH/MOD/LOW stratification)
        sr = summary_v4[summary_v4.PATNO == patno]
        if not sr.empty:
            row["ess_frac"] = float(sr.ess_frac.iloc[0])

        # Phase 1 PPC (if available). Phase 1 parquet schema:
        # PATNO, wave, n_scans, k_sbr_decay_mean, k_sbr_decay_std,
        # k_sbr_decay_q025, k_sbr_decay_q975, n_eff, r_hat, prior_mu
        # NO sigma column — we use PHASE1_SIGMA_PRIOR_MEAN as a proxy.
        if phase1_available and phase1 is not None:
            p1row = phase1[phase1.PATNO == patno]
            if not p1row.empty:
                k_mean = float(p1row["k_sbr_decay_mean"].iloc[0])
                k_std  = float(p1row["k_sbr_decay_std"].iloc[0])
                rng1 = np.random.default_rng(RNG_SEED + 100000 + patno)
                phase1_pred = phase1_ppc_samples(k_mean, k_std, PHASE1_SIGMA_PRIOR_MEAN,
                                                   t_years, sbr_anchor, rng1)
                phase1_covered_95 = coverage_from_samples(phase1_pred, sbr_obs, level=0.95)
                phase1_covered_50 = coverage_from_samples(phase1_pred, sbr_obs, level=0.50)
                phase1_zratio = calibration_z_ratio(phase1_pred, sbr_obs)
                row.update({
                    "phase1_scans_covered": int(phase1_covered_95.sum()),
                    "phase1_scans_total":   int(len(phase1_covered_95)),
                    "phase1_patient_coverage": float(phase1_covered_95.mean()),
                    "phase1_scans_covered_50": int(phase1_covered_50.sum()),
                    "phase1_patient_coverage_50": float(phase1_covered_50.mean()),
                    "phase1_zratio": phase1_zratio,
                })

        rows.append(row)

    df = pd.DataFrame(rows).sort_values("PATNO").reset_index(drop=True)
    n_pat = len(df)
    if n_pat == 0:
        print("ERROR: no patients produced predictive samples")
        return 1

    # --- Aggregate coverage ---
    phase2_scan_cov = df.phase2_scans_covered.sum() / max(1, df.phase2_scans_total.sum())
    phase2_scan_cov_50 = df.phase2_scans_covered_50.sum() / max(1, df.phase2_scans_total.sum())
    phase2_zratio_median = float(df.phase2_zratio.median())
    phase1_scan_cov = None
    phase1_scan_cov_50 = None
    phase1_zratio_median = None
    if "phase1_scans_covered" in df.columns:
        mask = df.phase1_scans_covered.notna() & df.phase1_scans_total.notna()
        if mask.any():
            phase1_scan_cov = (df.loc[mask, "phase1_scans_covered"].sum() /
                                max(1, df.loc[mask, "phase1_scans_total"].sum()))
            phase1_scan_cov_50 = (df.loc[mask, "phase1_scans_covered_50"].sum() /
                                    max(1, df.loc[mask, "phase1_scans_total"].sum()))
            phase1_zratio_median = float(df.loc[mask, "phase1_zratio"].median())

    # --- Stratified coverage by ESS bin ---
    if "ess_frac" in df.columns:
        high = df[df.ess_frac < HIGH_INFO_ESS_CUTOFF]
        mod  = df[(df.ess_frac >= HIGH_INFO_ESS_CUTOFF) & (df.ess_frac < 0.5)]
        low  = df[df.ess_frac >= 0.5]
    else:
        high = mod = low = df.iloc[0:0]

    def _stratum_cov(sub):
        if len(sub) == 0:
            return 0.0, 0, 0
        cov = sub.phase2_scans_covered.sum() / max(1, sub.phase2_scans_total.sum())
        return float(cov), int(sub.phase2_scans_covered.sum()), int(sub.phase2_scans_total.sum())

    h_cov, h_num, h_den = _stratum_cov(high)
    m_cov, m_num, m_den = _stratum_cov(mod)
    l_cov, l_num, l_den = _stratum_cov(low)

    print("=" * 76)
    print("HEADLINE COVERAGE")
    print("=" * 76)
    print(f"N patients analyzed:           {n_pat}")
    print(f"Phase 2 (IS) 95% PI coverage:  {phase2_scan_cov:.1%}  "
          f"(target S2 >= {REGRESSION_THRESHOLD:.0%})")
    print(f"Phase 2 (IS) 50% PI coverage:  {phase2_scan_cov_50:.1%}  "
          f"(well-calibrated target ≈ 50%)")
    print(f"Phase 2 z-ratio (median):      {phase2_zratio_median:.3f}  "
          f"(well-calibrated target ≈ 1.0)")
    if phase1_scan_cov is not None:
        delta = (phase2_scan_cov - phase1_scan_cov) * 100
        print()
        print(f"Phase 1      95% PI coverage:  {phase1_scan_cov:.1%}  "
              f"(phase1_report.md LOO baseline: 93.75%)")
        print(f"Phase 1      50% PI coverage:  {phase1_scan_cov_50:.1%}")
        print(f"Phase 1 z-ratio (median):      {phase1_zratio_median:.3f}")
        print(f"Phase 2 - Phase 1 delta (95%): {delta:+.1f} pp")
    print()
    print("Stratified by ESS fraction (data informativeness):")
    print(f"  HIGH (<{HIGH_INFO_ESS_CUTOFF:.0%}):   N={len(high):3d}  "
          f"coverage={h_cov:.1%}  ({h_num}/{h_den} scans)")
    print(f"  MOD  (20-50%): N={len(mod):3d}  coverage={m_cov:.1%}  ({m_num}/{m_den})")
    print(f"  LOW  (>=50%):  N={len(low):3d}  coverage={l_cov:.1%}  ({l_num}/{l_den})")
    print()

    # --- Gates ---
    gate_s2 = phase2_scan_cov >= REGRESSION_THRESHOLD
    gate_s2_plus = (phase1_scan_cov is None or
                     phase2_scan_cov >= phase1_scan_cov - PHASE1_MINUS_5_TOLERANCE)
    gate_s2_pp = h_cov >= HIGH_INFO_COVERAGE_THRESHOLD if len(high) > 0 else False
    gates_all = gate_s2 and gate_s2_plus and gate_s2_pp

    print("=" * 76)
    print("GATES")
    print("=" * 76)
    print(f"(S2)   Phase 2 >= 70% coverage:                    "
          f"{'PASS' if gate_s2 else 'FAIL'}  ({phase2_scan_cov:.1%})")
    if phase1_scan_cov is not None:
        print(f"(S2+)  Phase 2 >= Phase 1 - 5 pp:                  "
              f"{'PASS' if gate_s2_plus else 'FAIL'}  "
              f"(Phase 2 {phase2_scan_cov:.1%} vs Phase 1 {phase1_scan_cov:.1%})")
    else:
        print(f"(S2+)  Phase 1 not available — skipping comparison gate")
    print(f"(S2++) HIGH-INFO Phase 2 >= 80% coverage:          "
          f"{'PASS' if gate_s2_pp else 'FAIL'}  ({h_cov:.1%} on N={len(high)})")
    print(f"OVERALL: {'ALL GATES PASS' if gates_all else 'ONE OR MORE GATES FAIL'}")
    print()

    # --- Persist ---
    csv_out  = OUT_DIR / "step_2_8_v4_ppc.csv"
    json_out = OUT_DIR / "step_2_8_v4_ppc_summary.json"
    md_out   = OUT_DIR / "step_2_8_v4_ppc_report.md"
    fig_out  = OUT_DIR / "step_2_8_v4_ppc.png"
    manifest_out = OUT_DIR / "step_2_8_v4_RUN_MANIFEST.md"

    df.to_csv(csv_out, index=False)

    # --- Diagnostic figure ---
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    ax = axes[0]
    bins = np.linspace(0, 1, 21)
    ax.hist(df.phase2_patient_coverage, bins=bins, alpha=0.65, color="tab:blue",
            label=f"Phase 2 IS (scan cov={phase2_scan_cov:.1%})", edgecolor="k")
    if phase1_scan_cov is not None:
        ax.hist(df.phase1_patient_coverage.dropna(), bins=bins, alpha=0.55,
                color="tab:orange",
                label=f"Phase 1 (scan cov={phase1_scan_cov:.1%})", edgecolor="k")
    ax.axvline(REGRESSION_THRESHOLD, color="tab:red", linestyle="--",
               label=f"S2 gate ({REGRESSION_THRESHOLD:.0%})")
    ax.set_xlabel("Per-patient coverage fraction")
    ax.set_ylabel("# patients")
    ax.set_title("(a) Per-patient predictive coverage")
    ax.legend(fontsize=9)

    ax = axes[1]
    strata = [("HIGH (<20%)", high, "tab:blue"),
              ("MOD (20-50%)", mod,  "tab:orange"),
              ("LOW (>=50%)",  low,  "tab:red")]
    xs, covs, ns = [], [], []
    for lab, sub, _ in strata:
        c, num, den = _stratum_cov(sub)
        xs.append(lab); covs.append(c); ns.append(len(sub))
    bars = ax.bar(xs, covs, color=[s[2] for s in strata], alpha=0.75, edgecolor="k")
    for bar, n in zip(bars, ns):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f"N={n}", ha="center", fontsize=9)
    ax.axhline(REGRESSION_THRESHOLD, color="tab:red", linestyle="--",
               label=f"S2 ({REGRESSION_THRESHOLD:.0%})")
    ax.axhline(HIGH_INFO_COVERAGE_THRESHOLD, color="tab:green", linestyle=":",
               label=f"S2++ ({HIGH_INFO_COVERAGE_THRESHOLD:.0%})")
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("Scan-level coverage")
    ax.set_title("(b) Coverage by ESS stratum")
    ax.legend(fontsize=9)

    fig.suptitle(f"Phase 2 Step 2.8v4 — PPC on IS posterior (N={n_pat})",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(fig_out, dpi=150, bbox_inches="tight")

    # --- Hashes + JSON ---
    summary = {
        "_provenance": provenance,
        "n_patients": int(n_pat),
        "phase2_scan_coverage": float(phase2_scan_cov),
        "phase1_scan_coverage": float(phase1_scan_cov) if phase1_scan_cov is not None else None,
        "phase1_available": bool(phase1_available),
        "stratified": {
            "high_info": {"N": int(len(high)), "coverage": h_cov,
                           "scans_covered": h_num, "scans_total": h_den},
            "mod_info":  {"N": int(len(mod)),  "coverage": m_cov,
                           "scans_covered": m_num, "scans_total": m_den},
            "low_info":  {"N": int(len(low)),  "coverage": l_cov,
                           "scans_covered": l_num, "scans_total": l_den},
        },
        "gates": {
            "s2_ge_70pct":             bool(gate_s2),
            "s2_plus_ge_phase1_m5pp":  bool(gate_s2_plus),
            "s2_plus_plus_high_info_ge_80pct": bool(gate_s2_pp),
            "overall_pass":            bool(gates_all),
        },
        "_output_hashes": {
            "step_2_8_v4_ppc.csv": _sha256(csv_out),
            "step_2_8_v4_ppc.png": _sha256(fig_out),
        },
    }
    json_out.write_text(json.dumps(summary, indent=2, default=str))
    summary["_output_hashes"]["step_2_8_v4_ppc_summary.json"] = _sha256(json_out)
    print(f"CSV:  {csv_out}")
    print(f"JSON: {json_out}")
    print(f"FIG:  {fig_out}")

    # --- Markdown report ---
    md_lines = [
        "# Phase 2 Step 2.8v4 — Posterior Predictive Check (IS-posterior)",
        "",
        f"**Date:** {provenance['datetime_utc']}",
        f"**Git:** `{git.get('sha','?')}`{' DIRTY' if git.get('dirty') else ''}",
        f"**Input:** IS chains from `chains_is/` + Phase 1 `{PHASE1_POSTERIOR.name}`"
        + (" (found)" if phase1_available else " (NOT FOUND — Phase 1 comparison skipped)"),
        f"**Run manifest:** [step_2_8_v4_RUN_MANIFEST.md](step_2_8_v4_RUN_MANIFEST.md)",
        "",
        f"## Headline — {'ALL GATES PASS' if gates_all else 'NOT ALL GATES PASS'}",
        "",
        f"Phase 2 (IS-weighted posterior) achieves **{phase2_scan_cov:.1%}** "
        f"scan-level 95% PPC coverage on N={n_pat} Wave A patients, "
        f"{'exceeding' if gate_s2 else 'failing to reach'} the 70% S2 regression threshold.",
        "",
        "## Gates",
        "",
        "| Gate | Criterion | Value | Result |",
        "|---|---|---|---|",
        f"| S2   | Phase 2 ≥ 70% coverage | {phase2_scan_cov:.1%} | {'PASS' if gate_s2 else 'FAIL'} |",
    ]
    if phase1_scan_cov is not None:
        md_lines.append(f"| S2+  | Phase 2 ≥ Phase 1 − 5pp | P2 {phase2_scan_cov:.1%} vs P1 {phase1_scan_cov:.1%} | {'PASS' if gate_s2_plus else 'FAIL'} |")
    md_lines.append(f"| S2++ | HIGH-INFO ≥ 80% | {h_cov:.1%} on N={len(high)} | {'PASS' if gate_s2_pp else 'FAIL'} |")
    md_lines += [
        "",
        "## Stratified coverage",
        "",
        "| Stratum | N | Coverage | Scans (covered/total) |",
        "|---|---|---|---|",
        f"| HIGH (ESS<20%) | {len(high)} | {h_cov:.1%} | {h_num}/{h_den} |",
        f"| MOD  (20-50%)  | {len(mod)} |  {m_cov:.1%} | {m_num}/{m_den} |",
        f"| LOW  (≥50%)    | {len(low)} |  {l_cov:.1%} | {l_num}/{l_den} |",
        f"| ALL            | {n_pat} |  {phase2_scan_cov:.1%} | {df.phase2_scans_covered.sum()}/{df.phase2_scans_total.sum()} |",
        "",
        "## Interpretation",
        "",
        "The scan-level PPC coverage under Phase 2 should be interpreted alongside the ESS "
        "stratification from Step 2.6v4: the HIGH-INFO subset is where the T_tox reframe is "
        "empirically grounded, so that stratum's PPC coverage is the primary figure-of-merit. "
        "LOW-INFO patients have IS posteriors ≈ prior, so their PPC wide bands trivially cover "
        "the data — this inflates the cohort-level coverage above what a strict LOO would report.",
        "",
        "A strict leave-one-scan-out rerun (matching the Phase 1 Addendum A2 protocol) is "
        "scheduled as manuscript-revision work per the original plan §2.8.",
        "",
        "## Citations",
        "",
        "- Vehtari A., Gelman A., Gabry J. 2017. \"Practical Bayesian model evaluation using leave-one-out cross-validation and WAIC.\" *Statistics and Computing* 27:1413. doi:10.1007/s11222-016-9696-4",
        "- Fearnley J.M. & Lees A.J. 1991. *Brain* 114:2283. doi:10.1093/brain/114.5.2283",
        "- Phase 1 report (LOO baseline 93.75%): `outputs/mechanistic_twin/data/phase1_report.md`",
        "",
    ]
    md_out.write_text("\n".join(md_lines))
    print(f"MD:   {md_out}")

    # --- RUN_MANIFEST ---
    write_run_manifest(
        manifest_path=manifest_out,
        step_name="Phase 2 Step 2.8v4 — PPC on IS posterior",
        provenance=provenance,
        gate_results={
            "s2_ge_70pct": gate_s2,
            "s2_plus_ge_phase1_m5pp": gate_s2_plus,
            "s2_plus_plus_high_info_ge_80pct": gate_s2_pp,
            "overall_pass": gates_all,
        },
        summary_metrics={
            "n_patients": n_pat,
            "phase2_scan_coverage": f"{phase2_scan_cov:.4f}",
            "phase1_scan_coverage": f"{phase1_scan_cov:.4f}" if phase1_scan_cov is not None else "N/A",
            "high_info_coverage": f"{h_cov:.4f} on N={len(high)}",
            "mod_info_coverage":  f"{m_cov:.4f} on N={len(mod)}",
            "low_info_coverage":  f"{l_cov:.4f} on N={len(low)}",
        },
    )
    print(f"MANIFEST: {manifest_out}")

    return 0 if gates_all else 2


if __name__ == "__main__":
    raise SystemExit(main())
