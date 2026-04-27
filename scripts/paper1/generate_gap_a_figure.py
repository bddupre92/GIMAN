"""Paper 1 Gap A figure: SAA coverage by stage + Venuto-imputed S+ rate.

Two-panel publication-grade figure for the IEEE JBHI submission:
  Panel (a): SAA observation coverage stratified by NSD-ISS stage —
            stacked bars (SAA+, SAA−, SAA-untested) showing the structural
            sparsity that motivates Gap A's construct-validity concern.
  Panel (b): Venuto-2025-imputed S+ probability distribution among the
            SAA-missing NSD+ subset, with the imputed S+ rate (91.5%) and
            literature reference bands (Siderowf 88%, Venuto sporadic-PD 93%).

Outputs PNG (300 DPI) and vector PDF for IEEE submission.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from giman_pipeline.data.db import read_sql

OUT_DIR = Path("outputs/mechanistic_twin/paper1_submission/ieee-jbhi/figures")
OUT_DIR.mkdir(parents=True, exist_ok=True)
JSON_IN = Path("outputs/paper1_r2_responses/q_gap_a_venuto_saa.json")

# Okabe–Ito colorblind-safe palette
C_SAA_POS = "#0072B2"
C_SAA_NEG = "#D55E00"
C_SAA_MISS = "#999999"
C_VENUTO = "#009E73"
C_LIT_BAND = "#E69F00"


def panel_a_data() -> pd.DataFrame:
    """SAA status × stage counts."""
    df = read_sql(
        """
        SELECT
            COALESCE(nsd_iss_stage::text, 'NA') AS stage,
            CASE
                WHEN s_positive = TRUE THEN 'SAA+'
                WHEN s_positive = FALSE THEN 'SAA−'
                ELSE 'SAA-untested'
            END AS saa_status,
            COUNT(*) AS n
        FROM features.paper1_features_extended_33
        GROUP BY 1, 2
        ORDER BY 1, 2;
        """
    )
    return df


def main() -> None:
    coverage = panel_a_data()
    payload = json.loads(JSON_IN.read_text())
    sub_b = payload["sub_b_imputation_on_saa_missing"]
    sub_c = payload["sub_c_stage_redistribution"]["nsd_positive_subset_directly_relevant_to_gap_a"]

    fig, (ax_a, ax_b) = plt.subplots(
        1, 2, figsize=(8.5, 3.6), gridspec_kw={"width_ratios": [1.0, 1.0]}
    )

    # ----- Panel A: SAA coverage stacked bars per stage -----
    stage_order = ["0", "1", "2B", "3", "4", "unclassified"]
    stage_labels = ["Stage 0\n(HC + no\nbiomarker)", "Stage 1", "Stage 2B", "Stage 3", "Stage 4", "Uncl."]
    pivot = coverage.pivot(index="stage", columns="saa_status", values="n").fillna(0)
    pivot = pivot.reindex(stage_order, fill_value=0)
    bottoms = np.zeros(len(stage_order))
    width = 0.7
    x = np.arange(len(stage_order))

    pos = pivot.get("SAA+", pd.Series(0, index=stage_order)).values
    neg = pivot.get("SAA−", pd.Series(0, index=stage_order)).values
    miss = pivot.get("SAA-untested", pd.Series(0, index=stage_order)).values

    ax_a.bar(x, pos, width, color=C_SAA_POS, label="SAA+", edgecolor="white", linewidth=0.5)
    bottoms = pos.copy()
    ax_a.bar(x, neg, width, bottom=bottoms, color=C_SAA_NEG, label="SAA−", edgecolor="white", linewidth=0.5)
    bottoms += neg
    ax_a.bar(
        x,
        miss,
        width,
        bottom=bottoms,
        color=C_SAA_MISS,
        label="SAA-untested",
        edgecolor="white",
        linewidth=0.5,
        hatch="///",
    )

    totals = pos + neg + miss
    for xi, total in zip(x, totals):
        ax_a.text(xi, total + 18, f"n={int(total)}", ha="center", va="bottom", fontsize=7.5)

    ax_a.set_xticks(x)
    ax_a.set_xticklabels(stage_labels, fontsize=8)
    ax_a.set_ylabel("Number of patients", fontsize=9)
    ax_a.set_title("(a) SAA observation coverage by NSD-ISS stage", fontsize=9.5, loc="left")
    ax_a.legend(fontsize=7.5, loc="upper right", framealpha=0.95)
    ax_a.grid(axis="y", alpha=0.25, linewidth=0.5)
    ax_a.set_ylim(0, max(totals) * 1.12)
    ax_a.spines["top"].set_visible(False)
    ax_a.spines["right"].set_visible(False)

    # ----- Panel B: Venuto-imputed S+ probability among SAA-missing NSD+ -----
    q25, q50, q75 = (
        sub_b["venuto_prob_quartiles"]["q25"],
        sub_b["venuto_prob_quartiles"]["q50"],
        sub_b["venuto_prob_quartiles"]["q75"],
    )
    imputed_rate = sub_c["imputed_s_positive_rate_among_missing"]
    siderowf_rate = sub_c["literature_s_positive_rate_in_ppmi_pd"]
    venuto_rate = sub_c["venuto_2025_ppmi_pd_rate"]
    obs_rate = sub_c["observed_s_positive_rate_among_tested"]

    bars = [
        ("Observed S+\n(132 SAA-tested\nNSD+ in PPMI)", obs_rate, C_SAA_POS),
        ("Venuto-imputed S+\n(258 SAA-missing\nNSD+ with UPSIT)", imputed_rate, C_VENUTO),
        ("Siderowf 2023\nPPMI PD\n(literature)", siderowf_rate, C_LIT_BAND),
        ("Venuto 2025\nsporadic PD\n(literature)", venuto_rate, C_LIT_BAND),
    ]
    bx = np.arange(len(bars))
    bvals = [b[1] for b in bars]
    bcolors = [b[2] for b in bars]
    blabels = [b[0] for b in bars]

    ax_b.bar(bx, bvals, color=bcolors, edgecolor="white", linewidth=0.6, width=0.6)
    for xi, v in zip(bx, bvals):
        ax_b.text(xi, v + 0.015, f"{v:.0%}", ha="center", va="bottom", fontsize=8, fontweight="bold")

    ax_b.axhspan(siderowf_rate, venuto_rate, color=C_LIT_BAND, alpha=0.12, zorder=0,
                 label=f"Literature band {siderowf_rate:.0%}–{venuto_rate:.0%}")
    ax_b.set_xticks(bx)
    ax_b.set_xticklabels(blabels, fontsize=7)
    ax_b.set_ylabel("S+ rate among PD", fontsize=9)
    ax_b.set_title("(b) NSD+ S-anchor positivity: observed, imputed, literature", fontsize=9.5, loc="left")
    ax_b.set_ylim(0, 1.05)
    ax_b.legend(fontsize=7, loc="lower right", framealpha=0.95)
    ax_b.grid(axis="y", alpha=0.25, linewidth=0.5)
    ax_b.spines["top"].set_visible(False)
    ax_b.spines["right"].set_visible(False)

    fig.suptitle(
        "S-anchor coverage and Venuto-2025 construct-validity check",
        fontsize=10.5,
        y=1.02,
    )
    fig.tight_layout()

    png_path = OUT_DIR / "fig_gap_a_saa_coverage.png"
    pdf_path = OUT_DIR / "fig_gap_a_saa_coverage.pdf"
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    print(f"Wrote {png_path}")
    print(f"Wrote {pdf_path}")


if __name__ == "__main__":
    main()
