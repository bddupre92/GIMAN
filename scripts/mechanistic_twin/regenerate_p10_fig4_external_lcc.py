#!/usr/bin/env python3
"""Regenerate p10_fig4_external_lcc.pdf from external_validation_lcc.json.

The original figure (9.7 KB) was blank because the generator in
scripts/mechanistic_twin/phase5_generate_figures.py looked for JSON keys
(\"ppmi_hc.mean_sbr\") that do not exist in the actual JSON schema.
The real keys live under \"comparison_1_hc_vs_hc.per_region.*\" and
\"comparison_2_hc_vs_pd.per_region.*\". This script produces a 2-panel
figure using the correct schema.

Panel (A): Mean baseline SBR (mean +/- SD) by region for LCC HC, PPMI HC,
           and PPMI PD.
Panel (B): Relative percent difference summary: HC-vs-HC (scanner/protocol
           effect) and LCC-HC-vs-PPMI-PD (biological gap) per region.

Output:
  outputs/dissertation/figures/p10_fig4_external_lcc.pdf
  outputs/dissertation/figures/p10_fig4_external_lcc.png
  outputs/mechanistic_twin/paper10_submission/npj-pd/figures/p10_fig4_external_lcc.pdf

Usage:
  .venv/bin/python scripts/mechanistic_twin/regenerate_p10_fig4_external_lcc.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA = PROJECT_ROOT / "outputs/mechanistic_twin/paper10_mech_vs_giman/external_validation_lcc.json"

# Output destinations (keep both dissertation + submission packages in sync)
OUT_FILES = [
    PROJECT_ROOT / "outputs/dissertation/figures/p10_fig4_external_lcc.pdf",
    PROJECT_ROOT / "outputs/dissertation/figures/p10_fig4_external_lcc.png",
    PROJECT_ROOT / "outputs/mechanistic_twin/paper10_submission/npj-pd/figures/p10_fig4_external_lcc.pdf",
]

# Okabe-Ito palette (colorblind-safe)
COLOR_LCC_HC  = "#0072B2"  # blue     (external healthy control)
COLOR_PPMI_HC = "#009E73"  # green    (reference healthy control)
COLOR_PPMI_PD = "#D55E00"  # vermilion (PD cohort)

REGIONS = ["sbr_caudate_l", "sbr_caudate_r", "sbr_putamen_l", "sbr_putamen_r"]
REGION_LABELS = ["Caudate L", "Caudate R", "Putamen L", "Putamen R"]


def main() -> None:
    with DATA.open() as f:
        d = json.load(f)

    hc_hc = d["comparison_1_hc_vs_hc"]["per_region"]
    hc_pd = d["comparison_2_hc_vs_pd"]["per_region"]

    lcc_mean = [hc_hc[r]["lcc_hc_mean"] for r in REGIONS]
    lcc_std  = [hc_hc[r]["lcc_hc_std"]  for r in REGIONS]
    ppmi_hc_mean = [hc_hc[r]["ppmi_hc_mean"] for r in REGIONS]
    ppmi_hc_std  = [hc_hc[r]["ppmi_hc_std"]  for r in REGIONS]
    ppmi_pd_mean = [hc_pd[r]["ppmi_pd_mean"] for r in REGIONS]
    ppmi_pd_std  = [hc_pd[r]["ppmi_pd_std"]  for r in REGIONS]

    hc_hc_gap = [hc_hc[r]["relative_diff_pct"] for r in REGIONS]   # scanner effect
    hc_pd_gap = [hc_pd[r]["relative_diff_pct"] for r in REGIONS]   # biological gap

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))

    # ---- Panel A: per-region group means ----
    x = np.arange(len(REGIONS))
    w = 0.27
    ax1.bar(x - w, lcc_mean, w, yerr=lcc_std, capsize=3,
            color=COLOR_LCC_HC, edgecolor="black", linewidth=0.5,
            label=f"LCC HC (n={d['cohorts']['lcc_hc']['n_scans']})")
    ax1.bar(x,     ppmi_hc_mean, w, yerr=ppmi_hc_std, capsize=3,
            color=COLOR_PPMI_HC, edgecolor="black", linewidth=0.5,
            label=f"PPMI HC (n={d['cohorts']['ppmi_hc']['n_patients']})")
    ax1.bar(x + w, ppmi_pd_mean, w, yerr=ppmi_pd_std, capsize=3,
            color=COLOR_PPMI_PD, edgecolor="black", linewidth=0.5,
            label=f"PPMI PD (n={d['cohorts']['ppmi_pd']['n_patients']})")
    ax1.set_xticks(x)
    ax1.set_xticklabels(REGION_LABELS, fontsize=9)
    ax1.set_ylabel("DaT-SPECT SBR  (mean $\\pm$ SD)")
    ax1.set_title("(a) Baseline SBR by cohort and region", fontsize=11)
    ax1.legend(loc="upper right", fontsize=8, frameon=False)
    ax1.grid(axis="y", alpha=0.3)
    ax1.set_ylim(0, None)

    # ---- Panel B: gap summary ----
    w2 = 0.35
    ax2.bar(x - w2/2, hc_hc_gap, w2,
            color=COLOR_PPMI_HC, edgecolor="black", linewidth=0.5, alpha=0.85,
            label="LCC HC vs. PPMI HC  (scanner/protocol effect)")
    ax2.bar(x + w2/2, hc_pd_gap, w2,
            color=COLOR_PPMI_PD, edgecolor="black", linewidth=0.5, alpha=0.85,
            label="LCC HC vs. PPMI PD  (biological HC/PD gap)")
    # Annotate bar heights
    for i, v in enumerate(hc_hc_gap):
        ax2.text(i - w2/2, v + 3, f"{v:.0f}%", ha="center", fontsize=8)
    for i, v in enumerate(hc_pd_gap):
        ax2.text(i + w2/2, v + 3, f"{v:.0f}%", ha="center", fontsize=8)
    ax2.set_xticks(x)
    ax2.set_xticklabels(REGION_LABELS, fontsize=9)
    ax2.set_ylabel("Relative difference (%)")
    ax2.set_title("(b) Cross-cohort percent difference", fontsize=11)
    ax2.legend(loc="upper left", fontsize=8, frameon=False)
    ax2.grid(axis="y", alpha=0.3)
    ax2.set_ylim(0, max(hc_pd_gap) * 1.15)

    # Context text box
    note = (
        f"Scanner effect: HC--HC mean gap {d['comparison_1_hc_vs_hc']['mean_abs_relative_diff_pct']:.1f}\\%\n"
        f"Biological effect: HC--PD mean gap {d['comparison_2_hc_vs_pd']['mean_relative_diff_pct']:.1f}\\%"
    )
    fig.suptitle(
        "External validation on the Longitudinal Clinical Core (LCC) cohort",
        y=1.02, fontsize=12,
    )
    fig.text(0.5, -0.04, note, ha="center", fontsize=8, style="italic")
    fig.tight_layout()

    for out in OUT_FILES:
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, bbox_inches="tight", dpi=300 if out.suffix == ".png" else None)
        print(f"Saved: {out}")


if __name__ == "__main__":
    main()
