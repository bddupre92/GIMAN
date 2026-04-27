#!/usr/bin/env python3
"""Phase 5 Task 3: External cross-sectional baseline SBR validation (LCC).

**Second pivot (2026-04-13):** Initial pivot to LCC cross-sectional
revealed that LCC's 43 DaT-SPECT patients are ALL healthy controls
("No PD Nor Other Neurological Disorder" in amp_pd_case_control).
Healthy controls have preserved SBR — the 50% higher LCC values vs
PPMI PD baseline are EXPECTED, not a model failure.

**Final scope: dual cross-sectional comparison.**

1. HC-vs-HC: PPMI HC (N=343) vs LCC HC (N=43)
   - Same-population comparison — tests scanner/protocol consistency
   - Expected: distributions similar within ~10-20%

2. HC-vs-PD: LCC HC (N=43) vs PPMI PD baseline (N=2,080)
   - Tests the model's HC/PD discrimination assumption
   - Expected: LCC HC > PPMI PD (validates SBR framework)

**What this DOES validate:**
- PPMI's SBR measurement framework produces expected HC/PD gap
- LCC SBR values fall within PPMI's range at cohort level

**What this DOES NOT validate:**
- Longitudinal decay rate (requires ≥2 scans/patient — LCC has 1)
- Cross-cohort generalization of 3.29%/yr rate (requires harmonization)
- Paper 10 claim strength: weak cross-sectional reference only

Documented as NASEM audit limitation. Full longitudinal external validation
pending SURE-PD3 BioSEND DUA or DeNoPa (Paper 11 / future work).

Run:
    .venv/bin/python scripts/mechanistic_twin/phase5_external_validation_lcc.py

Output:
    outputs/mechanistic_twin/paper10_mech_vs_giman/external_validation_lcc.json

Author: Blair Dupre
Date: 2026-04-13
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp, mannwhitneyu

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.mechanistic_twin._reproducibility import capture_provenance

PPMI_DAT = PROJECT_ROOT / "data/00_raw/DaTScan_SBR_Analysis_08Feb2026.csv"
PPMI_PARTICIPANT = PROJECT_ROOT / "data/00_raw/Participant_Status_07Feb2026.csv"
LCC_DAT = PROJECT_ROOT / "data/00_raw/LCC/DaTSCAN_SBR.csv"
LCC_CASE = PROJECT_ROOT / "data/00_raw/LCC/amp_pd_case_control.csv"
OUTPUT_DIR = PROJECT_ROOT / "outputs/mechanistic_twin/paper10_mech_vs_giman"


def load_ppmi_baseline_sbr(cohort: str | None = None) -> pd.DataFrame:
    """Load PPMI baseline DaT-SPECT SBR (screening visit).

    Args:
        cohort: Optional filter — "Healthy Control", "Parkinson's Disease",
                "Prodromal", or None for all.
    """
    df = pd.read_csv(PPMI_DAT)
    # Filter to baseline (SC = screening) with analyzed scans
    baseline = df[
        (df["EVENT_ID"].isin(["SC", "ST"]))
        & (df["DATSCAN_ANALYZED"] == "Yes")
    ].copy()
    baseline = baseline.dropna(
        subset=[
            "DATSCAN_CAUDATE_R", "DATSCAN_CAUDATE_L",
            "DATSCAN_PUTAMEN_R", "DATSCAN_PUTAMEN_L",
        ]
    )
    # Rename to match LCC schema
    baseline = baseline.rename(
        columns={
            "DATSCAN_CAUDATE_R": "sbr_caudate_r",
            "DATSCAN_CAUDATE_L": "sbr_caudate_l",
            "DATSCAN_PUTAMEN_R": "sbr_putamen_r",
            "DATSCAN_PUTAMEN_L": "sbr_putamen_l",
        }
    )
    # Apply cohort filter if requested
    if cohort is not None:
        ps = pd.read_csv(PPMI_PARTICIPANT)
        target_patnos = set(
            ps[ps["COHORT_DEFINITION"] == cohort]["PATNO"].astype(int)
        )
        baseline = baseline[baseline["PATNO"].isin(target_patnos)]
    return baseline[
        ["PATNO", "EVENT_ID",
         "sbr_caudate_r", "sbr_caudate_l",
         "sbr_putamen_r", "sbr_putamen_l"]
    ]


def load_lcc_baseline_sbr() -> pd.DataFrame:
    """Load LCC baseline DaT-SPECT SBR."""
    df = pd.read_csv(LCC_DAT)
    # LCC already has cleaned column names; all entries are baseline (M0)
    df = df.dropna(
        subset=["sbr_caudate_r", "sbr_caudate_l", "sbr_putamen_r", "sbr_putamen_l"]
    )
    return df


def compare_distributions(
    ppmi: pd.DataFrame, lcc: pd.DataFrame, region: str
) -> dict:
    """Compare PPMI and LCC distributions for a given SBR region."""
    ppmi_vals = ppmi[region].values
    lcc_vals = lcc[region].values

    ks_stat, ks_p = ks_2samp(ppmi_vals, lcc_vals)
    try:
        mw_stat, mw_p = mannwhitneyu(ppmi_vals, lcc_vals, alternative="two-sided")
    except ValueError:
        mw_stat, mw_p = np.nan, np.nan

    return {
        "region": region,
        "ppmi_n": len(ppmi_vals),
        "ppmi_mean": float(np.mean(ppmi_vals)),
        "ppmi_median": float(np.median(ppmi_vals)),
        "ppmi_std": float(np.std(ppmi_vals)),
        "ppmi_min": float(np.min(ppmi_vals)),
        "ppmi_max": float(np.max(ppmi_vals)),
        "lcc_n": len(lcc_vals),
        "lcc_mean": float(np.mean(lcc_vals)),
        "lcc_median": float(np.median(lcc_vals)),
        "lcc_std": float(np.std(lcc_vals)),
        "lcc_min": float(np.min(lcc_vals)),
        "lcc_max": float(np.max(lcc_vals)),
        "ks_statistic": float(ks_stat),
        "ks_p": float(ks_p),
        "mw_statistic": float(mw_stat) if not np.isnan(mw_stat) else None,
        "mw_p": float(mw_p) if not np.isnan(mw_p) else None,
        "lcc_within_ppmi_range": bool(
            np.min(lcc_vals) >= np.min(ppmi_vals)
            and np.max(lcc_vals) <= np.max(ppmi_vals)
        ),
    }


def run_comparison(
    a_df: pd.DataFrame, b_df: pd.DataFrame, a_name: str, b_name: str,
) -> dict:
    """Run full per-region + combined comparison between two cohorts."""
    regions = ["sbr_caudate_r", "sbr_caudate_l", "sbr_putamen_r", "sbr_putamen_l"]
    region_results = {}
    for region in regions:
        # Reuse compare_distributions but relabel
        res = compare_distributions(a_df.rename(columns={}), b_df.rename(columns={}), region)
        # Remap keys to A/B labels
        labeled = {
            f"{a_name}_n": res["ppmi_n"],
            f"{a_name}_mean": res["ppmi_mean"],
            f"{a_name}_median": res["ppmi_median"],
            f"{a_name}_std": res["ppmi_std"],
            f"{a_name}_range": [res["ppmi_min"], res["ppmi_max"]],
            f"{b_name}_n": res["lcc_n"],
            f"{b_name}_mean": res["lcc_mean"],
            f"{b_name}_median": res["lcc_median"],
            f"{b_name}_std": res["lcc_std"],
            f"{b_name}_range": [res["lcc_min"], res["lcc_max"]],
            "ks_statistic": res["ks_statistic"],
            "ks_p": res["ks_p"],
            "mw_p": res["mw_p"],
            "b_within_a_range": res["lcc_within_ppmi_range"],
            "relative_diff_pct": (res["lcc_mean"] - res["ppmi_mean"]) / res["ppmi_mean"] * 100,
        }
        region_results[region] = labeled
    return region_results


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    prov = capture_provenance(
        script_path=Path(__file__),
        repo_root=PROJECT_ROOT,
        input_files=[PPMI_DAT, PPMI_PARTICIPANT, LCC_DAT, LCC_CASE],
    )

    print("Loading cohorts...")
    ppmi_all = load_ppmi_baseline_sbr()
    ppmi_hc = load_ppmi_baseline_sbr(cohort="Healthy Control")
    ppmi_pd = load_ppmi_baseline_sbr(cohort="Parkinson's Disease")
    lcc = load_lcc_baseline_sbr()
    print(f"  PPMI all:  {len(ppmi_all):,} scans, {ppmi_all['PATNO'].nunique():,} patients")
    print(f"  PPMI HC:   {len(ppmi_hc):,} scans, {ppmi_hc['PATNO'].nunique():,} patients")
    print(f"  PPMI PD:   {len(ppmi_pd):,} scans, {ppmi_pd['PATNO'].nunique():,} patients")
    print(f"  LCC HC:    {len(lcc):,} scans (all healthy controls per amp_pd_case_control)")

    # Comparison 1: LCC HC vs PPMI HC (same-population scanner/protocol check)
    print("\n" + "="*60)
    print("COMPARISON 1: LCC HC (n=43) vs PPMI HC (n=343)")
    print("Question: Do scanner/protocol effects cause SBR differences even in HC?")
    print("="*60)
    hc_vs_hc = run_comparison(ppmi_hc, lcc, "ppmi_hc", "lcc_hc")
    for region, res in hc_vs_hc.items():
        print(f"\n{region}:")
        print(f"  PPMI HC: {res['ppmi_hc_mean']:.2f} ± {res['ppmi_hc_std']:.2f}")
        print(f"  LCC HC:  {res['lcc_hc_mean']:.2f} ± {res['lcc_hc_std']:.2f}")
        print(f"  Relative diff: {res['relative_diff_pct']:+.1f}%")
        print(f"  KS p: {res['ks_p']:.4e}")

    # Comparison 2: LCC HC vs PPMI PD (HC/PD discrimination validation)
    print("\n" + "="*60)
    print("COMPARISON 2: LCC HC (n=43) vs PPMI PD baseline (n=2,049)")
    print("Question: Does the SBR framework produce expected HC > PD gap?")
    print("="*60)
    hc_vs_pd = run_comparison(ppmi_pd, lcc, "ppmi_pd", "lcc_hc")
    for region, res in hc_vs_pd.items():
        print(f"\n{region}:")
        print(f"  PPMI PD: {res['ppmi_pd_mean']:.2f} ± {res['ppmi_pd_std']:.2f}")
        print(f"  LCC HC:  {res['lcc_hc_mean']:.2f} ± {res['lcc_hc_std']:.2f}")
        print(f"  Gap: LCC HC is {res['relative_diff_pct']:+.1f}% vs PPMI PD")
        print(f"  (Expected: HC should be 40-80% higher than PD in striatum)")

    # Summary
    hc_hc_relative_diffs = [r["relative_diff_pct"] for r in hc_vs_hc.values()]
    hc_pd_relative_diffs = [r["relative_diff_pct"] for r in hc_vs_pd.values()]

    # Verdict
    mean_hc_hc_diff = np.mean([abs(d) for d in hc_hc_relative_diffs])
    mean_hc_pd_diff = np.mean(hc_pd_relative_diffs)  # signed

    if mean_hc_hc_diff < 10:
        hc_verdict = "STRONG: HC distributions consistent across cohorts (<10% diff)"
    elif mean_hc_hc_diff < 20:
        hc_verdict = (
            f"PARTIAL: HC distributions differ by {mean_hc_hc_diff:.1f}% on average. "
            "Likely scanner/protocol effects. Cross-cohort SBR comparisons require "
            "harmonization (ComBat, Wakasugi 2024)."
        )
    else:
        hc_verdict = (
            f"LIMITED: HC distributions differ by {mean_hc_hc_diff:.1f}% on average. "
            "Significant scanner/protocol differences detected. Direct cross-cohort "
            "comparison not recommended without harmonization."
        )

    if mean_hc_pd_diff > 40:
        pd_verdict = (
            f"Expected HC>PD gap confirmed: LCC HC {mean_hc_pd_diff:.1f}% higher than "
            "PPMI PD. SBR framework produces correct diagnostic discrimination."
        )
    else:
        pd_verdict = (
            f"HC/PD gap smaller than expected ({mean_hc_pd_diff:.1f}%). Investigate."
        )

    summary = {
        "scope": (
            "Cross-sectional baseline SBR validation. LCC N=43 are ALL healthy "
            "controls (amp_pd_case_control 'No PD Nor Other Neurological Disorder'). "
            "NOT external PD decay validation. Longitudinal external validation "
            "requires SURE-PD3/DeNoPa (Paper 11 future work)."
        ),
        "pdbp_spect_finding": (
            "PDBP SPECT data exists only in 2 DLB studies (Leverenz DLB Consortium "
            "N=259, Kantarci Longitudinal Imaging in DLB N=167). PDBP has NO "
            "standard-PD DaT-SPECT. Confirmed via pdbp.ninds.nih.gov Query Tool."
        ),
        "cohorts": {
            "ppmi_all_baseline": {"n_scans": len(ppmi_all), "n_patients": int(ppmi_all["PATNO"].nunique())},
            "ppmi_hc": {"n_scans": len(ppmi_hc), "n_patients": int(ppmi_hc["PATNO"].nunique())},
            "ppmi_pd": {"n_scans": len(ppmi_pd), "n_patients": int(ppmi_pd["PATNO"].nunique())},
            "lcc_hc": {"n_scans": len(lcc)},
        },
        "comparison_1_hc_vs_hc": {
            "description": "LCC HC vs PPMI HC — scanner/protocol consistency check",
            "per_region": hc_vs_hc,
            "mean_abs_relative_diff_pct": float(mean_hc_hc_diff),
            "verdict": hc_verdict,
        },
        "comparison_2_hc_vs_pd": {
            "description": "LCC HC vs PPMI PD — HC/PD discrimination validation",
            "per_region": hc_vs_pd,
            "mean_relative_diff_pct": float(mean_hc_pd_diff),
            "verdict": pd_verdict,
        },
        "limitations": [
            "Not external PD decay validation — LCC has no PD patients with DaT",
            "Not longitudinal — LCC has 1 scan per patient",
            "Scanner/protocol differences likely confound direct comparison",
            "Pending: SURE-PD3 via BioSEND DUA for longitudinal PD external validation",
            "Pending: DeNoPa via Mollenhauer collaboration for oligomeric α-syn validation",
        ],
        "_provenance": prov,
    }

    out_path = OUTPUT_DIR / "external_validation_lcc.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)

    print("\n" + "="*60)
    print("=== FINAL VERDICT ===")
    print("="*60)
    print(f"\nHC-vs-HC: {hc_verdict}")
    print(f"\nHC-vs-PD: {pd_verdict}")
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
