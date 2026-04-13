#!/usr/bin/env python3
"""
Hierarchical NLME — Decisive Test (Change 5, Closed-Loop v1.3)
================================================================

Purpose: non-circular test of whether hierarchical NLME population structure
breaks the k_n / alpha_tox degeneracy. Computes Spearman(posterior k_n from
HLME, SAA TTT) for the 59 SAA+ patients in the 1,065-patient cohort.

CRITICAL: this test uses HLME posterior k_n (which did NOT see SAA data)
against SAA TTT. This is the Change 5 decisive test — non-circular by
construction because SAA was never in the HLME likelihood.

The test PASSES if rho < -0.2 and p < 0.05 (negative because higher
nucleation k_n should correlate with faster SAA kinetics = lower TTT).

If the test FAILS (rho ~ 0), it means population structure alone cannot
separate k_n from alpha_tox — SAA must be included as a covariate (Phase B)
or the honest conclusion is that SBR cannot identify individual k_n.

Literature grounding:
  - Change 5: docs/lessons/2026-04-10-identifiability-validation-protocol.md
  - IS decisive test result: rho = -0.01, p = 0.95 (FAILED)
  - ITS decisive test result: rho = -0.06, p = 0.72 (FAILED)
  - This test asks: does hierarchical NLME do better?

Citations
---------
Closed-Loop Methodology v1.3, Change 5
Schunck et al. 2025 bioRxiv — hierarchical ODE <10% bias under sparsity
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

# Reproducibility header
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _reproducibility import capture_provenance, write_run_manifest  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]


def main():
    import argparse

    parser = argparse.ArgumentParser(description="HLME decisive test: k_n vs SAA TTT")
    parser.add_argument("--hlme-dir", type=str, required=True,
                        help="Path to HLME output dir (e.g., hlme_wave_a_v1)")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output dir for results (defaults to hlme-dir)")
    args = parser.parse_args()

    hlme_dir = Path(args.hlme_dir)
    if not hlme_dir.is_absolute():
        hlme_dir = REPO_ROOT / "outputs/mechanistic_twin/data/posteriors" / hlme_dir
    out_dir = Path(args.output_dir) if args.output_dir else hlme_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load HLME individual posteriors
    indiv_path = hlme_dir / "individual_params.csv"
    assert indiv_path.exists(), f"Missing {indiv_path}"
    indiv = pd.read_csv(indiv_path)
    print(f"Loaded {len(indiv)} HLME individual posteriors from {indiv_path}")

    # Load SAA kinetics
    saa_path = REPO_ROOT / "outputs/mechanistic_twin/data/saa_kinetics_extracted.csv"
    assert saa_path.exists(), f"Missing {saa_path}"
    saa = pd.read_csv(saa_path)
    saa_pos = saa[saa.saa_positive & saa.in_dat_spect].copy()
    print(f"SAA+ patients with DaT-SPECT: {len(saa_pos)}")

    # Merge on PATNO
    merged = pd.merge(indiv, saa_pos[["PATNO", "median_TTT", "median_Fmax"]],
                       on="PATNO", how="inner")
    print(f"Overlap (HLME x SAA+): {len(merged)} patients")

    if len(merged) < 10:
        print("WARNING: insufficient overlap for meaningful Spearman test")
        result = {
            "test": "decisive_hlme_vs_saa_ttt",
            "n_overlap": len(merged),
            "verdict": "INSUFFICIENT_DATA",
        }
    else:
        # Decisive test: Spearman(HLME posterior k_n, SAA TTT)
        # We test k_n_median (posterior median from HLME)
        # Support both HLME (k_n_median) and SAEM (k_n_ebe) column names
        kn_col = "k_n_median" if "k_n_median" in merged.columns else "k_n_ebe"
        ttox_col = "T_tox_median" if "T_tox_median" in merged.columns else "T_tox"
        atox_col = "alpha_tox_median" if "alpha_tox_median" in merged.columns else "alpha_tox_ebe"

        rho_kn, p_kn = stats.spearmanr(
            np.log(merged[kn_col].values),
            np.log(merged["median_TTT"].values)
        )

        # Also test T_tox as a sanity check (should correlate more strongly)
        rho_ttox, p_ttox = stats.spearmanr(
            np.log(merged[ttox_col].values),
            np.log(merged["median_TTT"].values)
        )

        # Also compare with alpha_tox
        rho_atox, p_atox = stats.spearmanr(
            np.log(merged[atox_col].values),
            np.log(merged["median_TTT"].values)
        )

        # Gate: rho_kn < -0.2 AND p < 0.05
        gate_pass = rho_kn < -0.2 and p_kn < 0.05

        result = {
            "test": "decisive_hlme_vs_saa_ttt",
            "n_overlap": len(merged),
            "hlme_run_tag": hlme_dir.name,
            "spearman_kn_vs_ttt": {
                "rho": round(float(rho_kn), 4),
                "p": round(float(p_kn), 4),
                "gate_threshold": "rho < -0.2 AND p < 0.05",
                "gate_pass": bool(gate_pass),
            },
            "spearman_ttox_vs_ttt": {
                "rho": round(float(rho_ttox), 4),
                "p": round(float(p_ttox), 4),
            },
            "spearman_atox_vs_ttt": {
                "rho": round(float(rho_atox), 4),
                "p": round(float(p_atox), 4),
            },
            "comparison_to_is": {
                "is_v4_sbr_only_kn_vs_ttt_rho": -0.01,
                "is_v4_sbr_only_kn_vs_ttt_p": 0.95,
                "its_hierarchical_kn_vs_ttt_rho": -0.06,
                "its_hierarchical_kn_vs_ttt_p": 0.72,
            },
            "verdict": "PASS — hierarchical structure breaks degeneracy" if gate_pass
                       else "FAIL — hierarchical structure insufficient for per-patient k_n",
        }

        print(f"\n{'='*60}")
        print("DECISIVE TEST: HLME posterior k_n vs SAA TTT")
        print(f"{'='*60}")
        print(f"N overlap:  {len(merged)}")
        print(f"rho(k_n):   {rho_kn:.4f}  (p={p_kn:.4f})")
        print(f"rho(T_tox): {rho_ttox:.4f}  (p={p_ttox:.4f})")
        print(f"rho(a_tox): {rho_atox:.4f}  (p={p_atox:.4f})")
        print(f"\nIS v4 SBR-only:  rho = -0.01  (p=0.95)")
        print(f"ITS hierarchical: rho = -0.06  (p=0.72)")
        print(f"\nGate (rho < -0.2, p < 0.05): {'PASS' if gate_pass else 'FAIL'}")
        print(f"{'='*60}")

    # Save results
    result_path = out_dir / "decisive_hlme_vs_saa.json"
    provenance = capture_provenance(
        Path(__file__).resolve(), REPO_ROOT,
        [indiv_path, saa_path],
        extra={"hlme_dir": str(hlme_dir)},
    )
    result["_provenance"] = provenance
    with open(result_path, "w") as f:
        json.dump(result, f, indent=2, default=str)
    print(f"\nSaved to {result_path}")

    # Write RUN_MANIFEST
    write_run_manifest(
        out_dir / "decisive_hlme_RUN_MANIFEST.md",
        "Decisive Test: HLME k_n vs SAA TTT",
        provenance,
        gate_results={"rho_kn < -0.2 AND p < 0.05": result.get("spearman_kn_vs_ttt", {}).get("gate_pass", False)},
        summary_metrics={
            "rho_kn": result.get("spearman_kn_vs_ttt", {}).get("rho", "N/A"),
            "p_kn": result.get("spearman_kn_vs_ttt", {}).get("p", "N/A"),
            "n_overlap": result.get("n_overlap", 0),
        },
    )


if __name__ == "__main__":
    main()
