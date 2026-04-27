#!/usr/bin/env python3
"""Generate Paper 6 Fig. 6: per-patient N(t)/N_0 distribution.

Two-panel histogram of the mechanistic-twin-derived dopaminergic-neuron
fraction surfaced per patient by the v2 pipeline.

Panel A: all 1,065 patients with a Phase-2 calibrated posterior.
Panel B: long-follow-up subset (>= 10 years from baseline, n=241).

Writes PDF + PNG to outputs/mechanistic_twin/paper6_submission/jamia/figures/
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SUMMARY_PATH = (
    ROOT / "outputs" / "paper6" / "pipeline_results" / "v2_full_cohort"
    / "pipeline_summary.json"
)
FIG_DIR = (
    ROOT / "outputs" / "mechanistic_twin" / "paper6_submission" / "jamia" / "figures"
)

LONG_FU_YEARS = 10.0

OKABE_ITO = {
    "orange": "#E69F00",
    "skyblue": "#56B4E9",
    "green": "#009E73",
    "yellow": "#F0E442",
    "blue": "#0072B2",
    "vermillion": "#D55E00",
    "purple": "#CC79A7",
    "grey": "#999999",
}


def main():
    with open(SUMMARY_PATH) as f:
        data = json.load(f)

    all_n = []
    long_n = []
    for k, r in data.items():
        if k == "summary" or "error" in r:
            continue
        m = r.get("mechanistic", {})
        if not m.get("available"):
            continue
        n_frac = m.get("n_frac_median")
        years = m.get("years_from_baseline", 0.0)
        if n_frac is None or not np.isfinite(n_frac):
            continue
        all_n.append(n_frac)
        if years >= LONG_FU_YEARS:
            long_n.append(n_frac)

    all_n = np.array(all_n)
    long_n = np.array(long_n)

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.2), sharey=False)

    # Panel A: all patients with posterior
    ax = axes[0]
    ax.hist(
        all_n, bins=40, range=(0.0, 1.0), color=OKABE_ITO["blue"],
        edgecolor="white", linewidth=0.5, alpha=0.9,
    )
    ax.axvline(np.median(all_n), color=OKABE_ITO["vermillion"], linestyle="--",
               linewidth=1.5, label=f"Median = {np.median(all_n):.2f}")
    ax.axvline(np.percentile(all_n, 25), color=OKABE_ITO["grey"], linestyle=":",
               linewidth=1.2, label=f"IQR [{np.percentile(all_n,25):.2f}, {np.percentile(all_n,75):.2f}]")
    ax.axvline(np.percentile(all_n, 75), color=OKABE_ITO["grey"], linestyle=":",
               linewidth=1.2)
    ax.set_xlabel(r"$N(t)/N_0$ at current-visit horizon")
    ax.set_ylabel("Patients (count)")
    ax.set_title(f"(a) All calibrated (n={len(all_n):,})",
                 fontsize=10, loc="left")
    ax.set_xlim(0.0, 1.0)
    ax.legend(fontsize=8, loc="upper left", frameon=False)

    # Panel B: long follow-up subset
    ax = axes[1]
    ax.hist(
        long_n, bins=30, range=(0.0, 1.0), color=OKABE_ITO["orange"],
        edgecolor="white", linewidth=0.5, alpha=0.9,
    )
    ax.axvline(np.median(long_n), color=OKABE_ITO["vermillion"], linestyle="--",
               linewidth=1.5, label=f"Median = {np.median(long_n):.2f}")
    ax.axvline(np.percentile(long_n, 25), color=OKABE_ITO["grey"], linestyle=":",
               linewidth=1.2, label=f"IQR [{np.percentile(long_n,25):.2f}, {np.percentile(long_n,75):.2f}]")
    ax.axvline(np.percentile(long_n, 75), color=OKABE_ITO["grey"], linestyle=":",
               linewidth=1.2)
    ax.set_xlabel(r"$N(t)/N_0$ at current-visit horizon")
    ax.set_ylabel("Patients (count)")
    ax.set_title(f"(b) Long follow-up $\\geq$ {LONG_FU_YEARS:.0f} yr (n={len(long_n)})",
                 fontsize=10, loc="left")
    ax.set_xlim(0.0, 1.0)
    ax.legend(fontsize=8, loc="upper left", frameon=False)

    plt.tight_layout()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    pdf = FIG_DIR / "fig6_n_frac_distribution.pdf"
    png = FIG_DIR / "fig6_n_frac_distribution.png"
    plt.savefig(pdf, dpi=300, bbox_inches="tight")
    plt.savefig(png, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Wrote: {pdf}")
    print(f"Wrote: {png}")
    print(
        f"Stats: all n={len(all_n):,} median={np.median(all_n):.3f} IQR="
        f"[{np.percentile(all_n,25):.3f}, {np.percentile(all_n,75):.3f}]"
    )
    print(
        f"       long n={len(long_n)} median={np.median(long_n):.3f} IQR="
        f"[{np.percentile(long_n,25):.3f}, {np.percentile(long_n,75):.3f}]"
    )


if __name__ == "__main__":
    main()
