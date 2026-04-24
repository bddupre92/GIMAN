"""Regenerate Fig 4 (feature ablation) with 21-feat paired-bootstrap numbers.

Replaces the old 22-feat vs 12-feat bars with 21-feat vs 12-feat under strict
circularity. Uses Okabe-Ito-compatible NSD palette.

Output: outputs/mechanistic_twin/paper1_submission/ieee-jbhi/figures/fig3_feature_ablation.pdf
        (and .png for legacy compatibility)
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path("/Users/blair.dupre/Projects/CSCI-FALL-2025")
SRC = ROOT / "outputs/paper1_r2_responses/q_r2_w3_ablation_21feat.json"
OUT_DIR = ROOT / "outputs/mechanistic_twin/paper1_submission/ieee-jbhi/figures"

# Okabe-Ito palette (colourblind-safe)
C_PRIMARY = "#0072B2"  # blue: 21-feat Path 3 primary
C_CLINICAL = "#E69F00"  # orange: 12-feat clinical-only
C_TEXT = "#333333"

TARGET_DISPLAY = {
    "binary": "Binary NSD+",
    "3class": "Three-class",
    "full_ordinal": "Full ordinal",
    "nsd_positive": "NSD+ sub-staging",
}


def main() -> None:
    d = json.loads(SRC.read_text())
    targets = list(TARGET_DISPLAY.keys())
    labels = [TARGET_DISPLAY[t] for t in targets]
    auc21 = [d["per_target"][t]["spec_21"]["pooled_auc"] for t in targets]
    ci21_lo = [d["per_target"][t]["spec_21"]["pooled_ci95"][0] for t in targets]
    ci21_hi = [d["per_target"][t]["spec_21"]["pooled_ci95"][1] for t in targets]
    auc12 = [d["per_target"][t]["spec_12"]["pooled_auc"] for t in targets]
    ci12_lo = [d["per_target"][t]["spec_12"]["pooled_ci95"][0] for t in targets]
    ci12_hi = [d["per_target"][t]["spec_12"]["pooled_ci95"][1] for t in targets]
    deltas = [d["per_target"][t]["delta_percentage_points"] for t in targets]
    p_values = [d["per_target"][t]["paired_bootstrap_21_minus_12"]["p_two_sided"] for t in targets]

    err21 = [[a - lo for a, lo in zip(auc21, ci21_lo)], [hi - a for a, hi in zip(auc21, ci21_hi)]]
    err12 = [[a - lo for a, lo in zip(auc12, ci12_lo)], [hi - a for a, hi in zip(auc12, ci12_hi)]]

    fig, ax = plt.subplots(figsize=(7.2, 4.5), dpi=300)

    x = np.arange(len(targets))
    w = 0.38
    bars1 = ax.bar(x - w/2, auc21, w, yerr=err21, capsize=4,
                    color=C_PRIMARY, edgecolor="black", linewidth=0.6,
                    label="21-feature strict-circularity primary")
    bars2 = ax.bar(x + w/2, auc12, w, yerr=err12, capsize=4,
                    color=C_CLINICAL, edgecolor="black", linewidth=0.6,
                    label="12-feature clinical-only")

    # Annotate delta-pp + p-value above bar pair
    for i, (d_pp, p) in enumerate(zip(deltas, p_values)):
        top = max(auc21[i], auc12[i]) + max(err21[1][i], err12[1][i]) + 0.02
        if p < 0.001:
            p_str = "$p{<}0.001$"
        elif p < 0.05:
            p_str = f"$p{{=}}{p:.3f}$"
        else:
            p_str = f"$p{{=}}{p:.2f}$"
        sign = "$-$" if d_pp < 0 else "$+$"
        sig_marker = "***" if p < 0.001 else ("*" if p < 0.05 else "n.s.")
        ax.text(i, top, f"{sign}{abs(d_pp):.1f} pp\n{p_str}  {sig_marker}",
                 ha="center", va="bottom", fontsize=8, color=C_TEXT)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel("ROC-AUC (5-fold CV pooled OOF; 95% bootstrap CI)", fontsize=10)
    ax.set_ylim(0.55, 1.0)
    ax.axhline(0.9, color="grey", linestyle=":", linewidth=0.5, alpha=0.5)
    ax.text(len(targets) - 0.5, 0.903, "AUC 0.90", fontsize=7, color="grey",
             ha="right", va="bottom")
    ax.grid(axis="y", linestyle="-", linewidth=0.4, alpha=0.25)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(loc="lower left", fontsize=9, frameon=False)

    fig.tight_layout()
    for suffix in ("pdf", "png"):
        out = OUT_DIR / f"fig3_feature_ablation.{suffix}"
        fig.savefig(out, dpi=300, bbox_inches="tight")
        print(f"Wrote {out}")


if __name__ == "__main__":
    main()
