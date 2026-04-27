#!/usr/bin/env python3
"""
Phase 2 Step 2.9 (S1) — Spearman T_tox vs Paper 3 Markov sojourn times
========================================================================

Purpose
-------
Original plan §3.4 S1 asked: does the per-patient toxicity flux T_tox
(Phase 2, data-informed via IS in Step 2.6v4) stratify patients across
baseline NSD-ISS stages in a direction consistent with Paper 3's
continuous-time Markov chain sojourn times?

Null: T_tox is uncorrelated with progression rate, so per-stage median
T_tox has no relationship to the Markov sojourn times.

Alt (expected): higher T_tox → faster neuron loss → shorter sojourn.
Spearman correlation between per-stage median T_tox and per-stage mean
sojourn time should be NEGATIVE.

Granularity caveat
------------------
The plan originally described S1 as a per-patient correlation. The
Paper 3 Markov output only reports population-level mean sojourn times
per stage (not per-patient). The honest rewording is a STAGE-LEVEL
correlation across N=5 stages (0, 2B, 3, 4, 5 — the stages represented
in the PPMI Wave A cohort). With N=5 the power is inherently limited
and we cannot reject the null at p<0.05 in the best case (Spearman
rank-sum variance for N=5 gives min achievable p ≈ 0.042 only for
ρ = ±1.0). We report:

  (a) Spearman ρ and p at the 5-stage aggregate level
  (b) Sign test (is ρ negative?) as the primary gate
  (c) Stratified by HIGH-INFO subset where the T_tox reframe is
      empirically grounded
  (d) Direction consistency: does the per-stage ORDERING of T_tox
      inversely match the per-stage ORDERING of sojourn times?

Gate criteria
-------------
(S1) Spearman ρ <= -0.5 on HIGH-INFO subset stage-level aggregation
     (medium-to-strong negative correlation — rank agreement even with
     low N)
(S1+) Sign of ρ is negative on the full cohort aggregation
(S1++) At least 3/5 stages are rank-concordant between T_tox rank and
      (inverse) sojourn rank

Reproducibility
---------------
Follows closed-loop v1.0 reproducibility rule. Deterministic.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _reproducibility import capture_provenance, write_run_manifest  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
V4_SUMMARY = REPO_ROOT / "outputs/mechanistic_twin/data/posteriors/phase2_coupled_is_step26v4.csv"
LONG_PARQUET = REPO_ROOT / "outputs/mechanistic_twin/data/dat_spect_longitudinal.parquet"
MARKOV_JSON = REPO_ROOT / "outputs/paper3_markov/markov_results.json"
OUT_DIR = REPO_ROOT / "outputs/mechanistic_twin/phase2"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Mapping from dat_spect_longitudinal numeric NSD-ISS codes to paper3 Markov keys
STAGE_CODE_MAP = {
    0.0: "0",
    1.0: "1",
    2.5: "2B",
    3.0: "3",
    4.0: "4",
    5.0: "5",
    6.0: "6",
}

HIGH_INFO_ESS_CUTOFF = 0.20
GATE_S1_THRESHOLD = -0.5
MIN_PATIENTS_PER_STAGE = 5


def _sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(2**20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    print("=" * 76)
    print("Phase 2 Step 2.9 (S1) — Spearman T_tox vs Paper 3 Markov sojourn")
    print("=" * 76)

    provenance = capture_provenance(
        script_path=Path(__file__).resolve(),
        repo_root=REPO_ROOT,
        input_files=[V4_SUMMARY, LONG_PARQUET, MARKOV_JSON],
        extra={
            "high_info_ess_cutoff": HIGH_INFO_ESS_CUTOFF,
            "gate_S1_threshold_rho": GATE_S1_THRESHOLD,
            "min_patients_per_stage": MIN_PATIENTS_PER_STAGE,
            "stage_code_map": STAGE_CODE_MAP,
        },
    )
    git = provenance["git"]
    print(f"Git:     {git.get('sha','?')}{' (DIRTY)' if git.get('dirty') else ''}")
    print(f"Script:  {provenance['script']['sha256'][:16]}...")
    print()

    for f in (V4_SUMMARY, LONG_PARQUET, MARKOV_JSON):
        if not f.exists():
            print(f"ERROR: required input {f} missing"); return 1

    v4 = pd.read_csv(V4_SUMMARY)
    long_df = pd.read_parquet(LONG_PARQUET)
    with MARKOV_JSON.open() as f:
        markov = json.load(f)

    sojourn = markov["sojourn_times"]  # dict: stage_name -> mean years

    # Per-patient baseline NSD-ISS stage (first observed scan)
    baseline = (long_df.sort_values(["PATNO", "t_years"])
                      .groupby("PATNO")
                      .nsd_iss_stage.first()
                      .rename("baseline_stage_code"))
    baseline_named = baseline.map(STAGE_CODE_MAP).rename("baseline_stage")
    # Join onto v4 summary
    v4 = v4.merge(baseline.reset_index(), on="PATNO", how="left")
    v4 = v4.merge(baseline_named.reset_index(), on="PATNO", how="left")
    v4 = v4[v4.baseline_stage.notna()].copy()

    print(f"v4 summary rows with baseline stage: {len(v4)} / {len(baseline)}")
    print()

    def per_stage_agg(sub: pd.DataFrame, label: str) -> pd.DataFrame:
        g = (sub.groupby("baseline_stage")
                .agg(n=("T_tox_mean", "size"),
                     T_tox_median=("T_tox_mean", "median"),
                     T_tox_mean=("T_tox_mean", "mean"),
                     pct_yr_median=("pct_loss_per_yr_median", "median"))
                .reset_index())
        g["sojourn_years"] = g.baseline_stage.map(sojourn).astype(float)
        g = g[g.n >= MIN_PATIENTS_PER_STAGE].copy()
        g = g.dropna(subset=["sojourn_years", "T_tox_median"]).copy()
        g["label"] = label
        return g

    # Full cohort aggregation
    full_agg = per_stage_agg(v4, "full")
    high_agg = per_stage_agg(v4[v4.ess_frac < HIGH_INFO_ESS_CUTOFF], "high_info")

    print("FULL COHORT per-stage aggregation:")
    for _, r in full_agg.iterrows():
        print(f"  stage {r.baseline_stage}: n={int(r.n):3d}  "
              f"T_tox_median={r.T_tox_median:.3e}  "
              f"pct_yr_median={r.pct_yr_median:5.2f}  "
              f"sojourn={r.sojourn_years:.2f} yr")

    print()
    print("HIGH-INFO (ESS<20%) per-stage aggregation:")
    for _, r in high_agg.iterrows():
        print(f"  stage {r.baseline_stage}: n={int(r.n):3d}  "
              f"T_tox_median={r.T_tox_median:.3e}  "
              f"pct_yr_median={r.pct_yr_median:5.2f}  "
              f"sojourn={r.sojourn_years:.2f} yr")
    print()

    # Spearman correlations
    def spear(agg):
        if len(agg) < 3:
            return (np.nan, np.nan)
        rho, p = spearmanr(agg.T_tox_median, agg.sojourn_years)
        return float(rho), float(p)

    rho_full, p_full = spear(full_agg)
    rho_hi,   p_hi   = spear(high_agg)

    # Rank-concordance diagnostic: do patients with high T_tox also have
    # short sojourn? (Agreement between T_tox-rank-desc and sojourn-rank-asc)
    def rank_concordance(agg):
        if len(agg) < 2:
            return np.nan
        T_rank = agg.T_tox_median.rank()
        sj_rank = agg.sojourn_years.rank()
        # Perfect negative correlation → T_rank_i + sj_rank_i = N+1 for all i
        # Concordance fraction = fraction of pairs (i, j) where T[i]>T[j] implies sj[i]<sj[j]
        n = len(agg)
        concordant = 0
        total = 0
        for i in range(n):
            for j in range(i+1, n):
                if agg.T_tox_median.iloc[i] == agg.T_tox_median.iloc[j]:
                    continue
                if agg.sojourn_years.iloc[i] == agg.sojourn_years.iloc[j]:
                    continue
                total += 1
                if ((agg.T_tox_median.iloc[i] > agg.T_tox_median.iloc[j]) !=
                    (agg.sojourn_years.iloc[i] > agg.sojourn_years.iloc[j])):
                    concordant += 1
        return concordant / total if total else np.nan

    conc_full = rank_concordance(full_agg)
    conc_hi   = rank_concordance(high_agg)

    print(f"FULL cohort Spearman:  rho={rho_full:+.3f}  p={p_full:.3f}  concordance={conc_full:.2f}")
    print(f"HIGH-INFO   Spearman:  rho={rho_hi:+.3f}  p={p_hi:.3f}  concordance={conc_hi:.2f}")
    print()

    # Gates (defensive against NaN)
    def _le(a, b): return (a == a) and (a <= b)
    def _lt(a, b): return (a == a) and (a < b)
    gate_s1    = _le(rho_hi, GATE_S1_THRESHOLD) if len(high_agg) >= 3 else False
    gate_s1_p  = _lt(rho_full, 0.0) if len(full_agg) >= 3 else False
    gate_s1_pp = (conc_full == conc_full) and (conc_full >= 0.6)
    gates_all = gate_s1 and gate_s1_p and gate_s1_pp

    print("=" * 76)
    print("GATES")
    print("=" * 76)
    print(f"(S1)   HIGH-INFO Spearman ρ ≤ {GATE_S1_THRESHOLD}: "
          f"{'PASS' if gate_s1 else 'FAIL'} (ρ={rho_hi:+.3f} on N={len(high_agg)} stages)")
    print(f"(S1+)  Full cohort ρ < 0 (negative sign): "
          f"{'PASS' if gate_s1_p else 'FAIL'} (ρ={rho_full:+.3f})")
    print(f"(S1++) Full cohort rank concordance ≥ 60%: "
          f"{'PASS' if gate_s1_pp else 'FAIL'} ({conc_full:.2f})")
    print(f"OVERALL: {'ALL GATES PASS' if gates_all else 'ONE OR MORE GATES FAIL'}")
    print()

    # Figure
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    for ax, agg, title in [(axes[0], full_agg, f"Full cohort (N stages={len(full_agg)})"),
                             (axes[1], high_agg, f"HIGH-INFO (ESS<20%, N stages={len(high_agg)})")]:
        if len(agg) > 0:
            ax.scatter(agg.sojourn_years, agg.T_tox_median, s=100, color="tab:blue",
                       edgecolor="k", zorder=3)
            for _, r in agg.iterrows():
                ax.annotate(f"stage {r.baseline_stage}\nn={int(r.n)}",
                            (r.sojourn_years, r.T_tox_median),
                            xytext=(5, 5), textcoords="offset points", fontsize=9)
        ax.set_xlabel("Paper 3 Markov sojourn time (years)")
        ax.set_ylabel("Median T_tox (hr⁻¹)")
        ax.set_yscale("log")
        rho, p = spear(agg)
        ax.set_title(f"{title}\nSpearman ρ={rho:+.3f}  p={p:.3f}" if not np.isnan(rho) else title)
        ax.grid(alpha=0.3)

    fig.suptitle("Phase 2 Step 2.9 (S1) — T_tox vs Markov sojourn by baseline stage",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig_out = OUT_DIR / "step_2_9_s1_spearman.png"
    fig.savefig(fig_out, dpi=150, bbox_inches="tight")

    # Persist
    csv_out  = OUT_DIR / "step_2_9_s1_per_stage.csv"
    json_out = OUT_DIR / "step_2_9_s1_summary.json"
    md_out   = OUT_DIR / "step_2_9_s1_report.md"
    manifest_out = OUT_DIR / "step_2_9_s1_RUN_MANIFEST.md"

    combined = pd.concat([full_agg, high_agg], ignore_index=True)
    combined.to_csv(csv_out, index=False)

    summary = {
        "_provenance": provenance,
        "sojourn_times_by_stage": {k: float(v) for k, v in sojourn.items()},
        "n_stages_full": int(len(full_agg)),
        "n_stages_high_info": int(len(high_agg)),
        "spearman_full_rho": float(rho_full) if not np.isnan(rho_full) else None,
        "spearman_full_p":   float(p_full) if not np.isnan(p_full) else None,
        "spearman_high_rho": float(rho_hi) if not np.isnan(rho_hi) else None,
        "spearman_high_p":   float(p_hi) if not np.isnan(p_hi) else None,
        "concordance_full":  float(conc_full) if not np.isnan(conc_full) else None,
        "concordance_high_info": float(conc_hi) if not np.isnan(conc_hi) else None,
        "gates": {
            "s1_high_info_rho_le_m05": bool(gate_s1),
            "s1_plus_full_rho_negative": bool(gate_s1_p),
            "s1_pp_concordance_ge_60pct": bool(gate_s1_pp),
            "overall_pass": bool(gates_all),
        },
        "_output_hashes": {
            "step_2_9_s1_per_stage.csv": _sha256(csv_out),
            "step_2_9_s1_spearman.png":  _sha256(fig_out),
        },
    }
    json_out.write_text(json.dumps(summary, indent=2, default=str))
    summary["_output_hashes"]["step_2_9_s1_summary.json"] = _sha256(json_out)
    print(f"CSV:  {csv_out}")
    print(f"JSON: {json_out}")
    print(f"FIG:  {fig_out}")

    # Markdown report
    def _fmt(x): return f"{x:+.3f}" if (x == x) else "N/A"
    md = [
        "# Phase 2 Step 2.9 (S1) — T_tox vs Paper 3 Markov sojourn",
        "",
        f"**Date:** {provenance['datetime_utc']}",
        f"**Git:** `{git.get('sha','?')}`{' DIRTY' if git.get('dirty') else ''}",
        f"**Run manifest:** [step_2_9_s1_RUN_MANIFEST.md](step_2_9_s1_RUN_MANIFEST.md)",
        "",
        f"## Headline — {'ALL GATES PASS' if gates_all else 'NOT ALL GATES PASS'}",
        "",
        "Stage-level Spearman rank correlation between per-stage median T_tox "
        "and the corresponding Paper 3 Markov mean sojourn time. Expected sign: "
        "negative (high T_tox → faster neuron loss → shorter sojourn). "
        "Granularity: stage-level, not per-patient, because Paper 3 reports "
        "population-mean sojourn per stage.",
        "",
        "## Gates",
        "",
        "| Gate | Criterion | Value | Result |",
        "|---|---|---|---|",
        f"| S1   | HIGH-INFO ρ ≤ {GATE_S1_THRESHOLD} | {_fmt(rho_hi)} (N={len(high_agg)}) | {'PASS' if gate_s1 else 'FAIL'} |",
        f"| S1+  | Full cohort ρ negative | {_fmt(rho_full)} | {'PASS' if gate_s1_p else 'FAIL'} |",
        f"| S1++ | Full rank concordance ≥ 60% | {conc_full:.2f} | {'PASS' if gate_s1_pp else 'FAIL'} |",
        "",
        "## Per-stage aggregation (full cohort)",
        "",
        "| Stage | N patients | Median T_tox (hr⁻¹) | Implied %/yr | Markov sojourn (yr) |",
        "|---|---|---|---|---|",
    ]
    for _, r in full_agg.iterrows():
        md.append(f"| {r.baseline_stage} | {int(r.n)} | "
                  f"{r.T_tox_median:.3e} | {r.pct_yr_median:.2f} | "
                  f"{r.sojourn_years:.2f} |")
    md += [
        "",
        "## Per-stage aggregation (HIGH-INFO subset, ESS<20%)",
        "",
        "| Stage | N patients | Median T_tox (hr⁻¹) | Implied %/yr | Markov sojourn (yr) |",
        "|---|---|---|---|---|",
    ]
    for _, r in high_agg.iterrows():
        md.append(f"| {r.baseline_stage} | {int(r.n)} | "
                  f"{r.T_tox_median:.3e} | {r.pct_yr_median:.2f} | "
                  f"{r.sojourn_years:.2f} |")
    md += [
        "",
        "## Power limitation (honest disclosure)",
        "",
        f"With N_stages = {len(full_agg)} (full) / {len(high_agg)} (HIGH-INFO), the Spearman "
        "rank-sum test has inherently limited power — Bonferroni-adjusted "
        "significance at α=0.05 requires ρ ≈ ±1.0. The primary interpretive "
        "anchor is therefore (a) the **sign** of ρ and (b) the **pairwise "
        "rank concordance** (fraction of stage pairs where T_tox rank is "
        "inverse to sojourn rank). Per-patient correlation would require "
        "per-patient Markov sojourn posteriors which Paper 3 does not output.",
        "",
        "## Citations",
        "",
        "- Paper 3 Markov results: `outputs/paper3_markov/markov_results.json` (Feb 2026)",
        "- NSD-ISS staging: Simuni et al. 2024 Lancet Neurology",
        "- Spearman rank correlation: Hollander, Wolfe, Chicken 2014 *Nonparametric Statistical Methods* 3rd ed.",
        "",
    ]
    md_out.write_text("\n".join(md))
    print(f"MD:   {md_out}")

    write_run_manifest(
        manifest_path=manifest_out,
        step_name="Phase 2 Step 2.9 (S1) — T_tox vs Markov sojourn Spearman",
        provenance=provenance,
        gate_results={
            "s1_high_info_rho_le_m05": gate_s1,
            "s1_plus_full_rho_negative": gate_s1_p,
            "s1_pp_concordance_ge_60pct": gate_s1_pp,
            "overall_pass": gates_all,
        },
        summary_metrics={
            "n_stages_full": len(full_agg),
            "n_stages_high_info": len(high_agg),
            "rho_full":        _fmt(rho_full),
            "p_full":          f"{p_full:.4f}" if p_full == p_full else "N/A",
            "rho_high_info":   _fmt(rho_hi),
            "p_high_info":     f"{p_hi:.4f}" if p_hi == p_hi else "N/A",
            "concordance_full": f"{conc_full:.4f}" if conc_full == conc_full else "N/A",
            "concordance_high_info": f"{conc_hi:.4f}" if conc_hi == conc_hi else "N/A",
        },
    )
    print(f"MANIFEST: {manifest_out}")

    return 0 if gates_all else 2


if __name__ == "__main__":
    raise SystemExit(main())
