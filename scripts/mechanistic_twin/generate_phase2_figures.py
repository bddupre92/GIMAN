#!/usr/bin/env python3
"""
Phase 2 Publication Figures — Mechanistic Digital Twin
======================================================

Generates publication-quality figures (300 DPI, PDF + PNG) for the
mechanistic digital twin Phase 2 manuscript / bioRxiv preprint.

Figure inventory:
  Fig 1: Variant B ODE schematic (TikZ — generated separately in LaTeX)
  Fig 2: IS posterior — T_tox vs implied %/yr with Fearnley range
  Fig 3: Degeneracy-breaking — v4 vs v5 paired comparison (3-panel)
  Fig 4: CSF observation model — predicted vs observed CSF
  Fig 5: Counterfactual — treatment delay by scenario (2-panel)
  Fig 6: ESS stratification — cohort informativeness landscape

Style: Nature/Cell-compatible. Colorblind-safe (Okabe-Ito palette).
Font: DejaVu Sans (matplotlib default, compatible with PDF embedding).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch

# Okabe-Ito colorblind-safe palette
OI_BLUE    = "#0072B2"
OI_ORANGE  = "#E69F00"
OI_GREEN   = "#009E73"
OI_RED     = "#D55E00"
OI_PURPLE  = "#CC79A7"
OI_CYAN    = "#56B4E9"
OI_YELLOW  = "#F0E442"
OI_BLACK   = "#000000"

REPO_ROOT = Path(__file__).resolve().parents[2]
POST_DIR  = REPO_ROOT / "outputs/mechanistic_twin/data/posteriors"
OUT_DIR   = REPO_ROOT / "outputs/mechanistic_twin/phase2/figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

HR_PER_YR = 8766.0
GAMMA = 0.7

# Global matplotlib style
plt.rcParams.update({
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 8,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "pdf.fonttype": 42,   # TrueType embedding for journal compatibility
    "ps.fonttype": 42,
})


def save_fig(fig, name: str):
    for ext in ["png", "pdf"]:
        path = OUT_DIR / f"{name}.{ext}"
        fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
    print(f"  Saved: {name}.png + .pdf")


def fig2_ttox_neuron_loss():
    """Fig 2: T_tox posterior → implied neuron loss with Fearnley range."""
    v5 = pd.read_csv(POST_DIR / "phase2_coupled_is_step26v5_csf.csv")
    v5_csf = v5[v5["has_csf"] == True]  # noqa

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.5))

    # Panel A: %/yr distribution
    bins = np.logspace(np.log10(0.05), np.log10(200), 50)
    ax1.hist(v5_csf.pct_loss_per_yr_median, bins=bins, color=OI_BLUE,
             alpha=0.75, edgecolor="white", linewidth=0.5)
    ax1.axvspan(2, 5, alpha=0.15, color=OI_GREEN, zorder=0)
    ax1.text(3.3, ax1.get_ylim()[1] * 0.85, "Fearnley &\nLees 1991\n2–5%/yr",
             fontsize=8, ha="center", color=OI_GREEN, fontweight="bold")
    ax1.axvline(v5_csf.pct_loss_per_yr_median.median(), color=OI_RED,
                linewidth=2, label=f"Median: {v5_csf.pct_loss_per_yr_median.median():.1f}%/yr")
    ax1.set_xscale("log")
    ax1.set_xlabel("Implied neuron loss rate (%/yr)")
    ax1.set_ylabel("Number of patients")
    ax1.set_title("A. Cohort neuron-loss distribution (N=277)")
    ax1.legend(loc="upper right")

    # Panel B: T_tox vs ESS fraction (informativeness landscape)
    sc = ax2.scatter(v5_csf.ess_frac * 100, v5_csf.pct_loss_per_yr_median,
                     s=15, alpha=0.6, c=v5_csf.log_T_tox_sd_log10,
                     cmap="viridis_r", vmin=0, vmax=1.5)
    ax2.axhspan(2, 5, alpha=0.1, color=OI_GREEN, zorder=0)
    ax2.axvline(20, color=OI_RED, linestyle="--", alpha=0.5,
                label="HIGH-INFO threshold")
    ax2.set_xlabel("ESS fraction (%)")
    ax2.set_ylabel("Implied neuron loss (%/yr)")
    ax2.set_yscale("log")
    ax2.set_title("B. Informativeness vs progression rate")
    ax2.legend(loc="upper right", fontsize=7)
    plt.colorbar(sc, ax=ax2, label="T_tox posterior SD (log₁₀ dec)", shrink=0.8)

    fig.suptitle("Per-Patient Toxicity Flux from Joint DaT-SPECT + CSF Calibration",
                 fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout()
    save_fig(fig, "fig2_ttox_neuron_loss")
    plt.close(fig)


def fig3_degeneracy_breaking():
    """Fig 3: v4 vs v5 degeneracy-breaking (3-panel paired comparison)."""
    v4 = pd.read_csv(POST_DIR / "phase2_coupled_is_step26v4.csv")
    v5 = pd.read_csv(POST_DIR / "phase2_coupled_is_step26v5_csf.csv")
    v5_csf = v5[v5["has_csf"] == True]  # noqa

    merged = v5_csf.merge(
        v4[["PATNO", "cor_logk_logalpha", "log_k_n_sd_prior_ratio",
            "log_alpha_tox_sd_prior_ratio"]],
        on="PATNO", suffixes=("_v5", "_v4")
    )

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

    # Panel A: Correlation shift
    ax = axes[0]
    ax.scatter(merged.cor_logk_logalpha_v4, merged.cor_logk_logalpha_v5,
               s=12, alpha=0.5, color=OI_BLUE, zorder=3)
    ax.plot([-1, 0.5], [-1, 0.5], "k--", alpha=0.4, linewidth=1)
    ax.fill_between([-1, 0.5], [-1, 0.5], [0.5, 0.5], alpha=0.05, color=OI_GREEN)
    ax.set_xlabel("v4 cor(log k_n, log α_tox) — SBR only")
    ax.set_ylabel("v5 cor(log k_n, log α_tox) — SBR + CSF")
    ax.set_title(f"A. Correlation shift\n(96% above y=x)", fontsize=11)
    ax.set_xlim(-1, 0.3)
    ax.set_ylim(-1, 0.3)

    # Panel B: k_n tightening
    ax = axes[1]
    ax.scatter(merged.log_k_n_sd_prior_ratio_v4, merged.log_k_n_sd_prior_ratio_v5,
               s=12, alpha=0.5, color=OI_GREEN, zorder=3)
    ax.plot([0, 1.2], [0, 1.2], "k--", alpha=0.4, linewidth=1)
    ax.fill_between([0, 1.2], [0, 0], [0, 1.2], alpha=0.05, color=OI_GREEN)
    ax.set_xlabel("v4 k_n SD / prior SD — SBR only")
    ax.set_ylabel("v5 k_n SD / prior SD — SBR + CSF")
    ax.set_title(f"B. k_n posterior tightening\n(100% below y=x)", fontsize=11)

    # Panel C: Summary bar chart
    ax = axes[2]
    metrics = ["cor toward 0", "k_n tighter", "α_tox tighter", "T_tox stable"]
    v4_vals = [abs(merged.cor_logk_logalpha_v4.median()),
               merged.log_k_n_sd_prior_ratio_v4.median(),
               merged.log_alpha_tox_sd_prior_ratio_v4.median(),
               v4[v4.PATNO.isin(merged.PATNO)].log_T_tox_sd_log10.median()]
    v5_vals = [abs(merged.cor_logk_logalpha_v5.median()),
               merged.log_k_n_sd_prior_ratio_v5.median(),
               merged.log_alpha_tox_sd_prior_ratio_v5.median(),
               v5_csf.log_T_tox_sd_log10.median()]

    x = np.arange(len(metrics))
    w = 0.35
    bars1 = ax.bar(x - w/2, v4_vals, w, label="v4 (SBR only)", color=OI_ORANGE, alpha=0.8)
    bars2 = ax.bar(x + w/2, v5_vals, w, label="v5 (SBR + CSF)", color=OI_BLUE, alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics, rotation=15, ha="right", fontsize=8)
    ax.set_ylabel("Value (lower = better for b,c,d)")
    ax.set_title("C. Summary comparison", fontsize=11)
    ax.legend(fontsize=8)

    fig.suptitle("CSF α-Synuclein Breaks the T_tox Sloppy-Ridge Degeneracy",
                 fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout()
    save_fig(fig, "fig3_degeneracy_breaking")
    plt.close(fig)


def fig4_csf_fit():
    """Fig 4: CSF predicted vs observed."""
    v5 = pd.read_csv(POST_DIR / "phase2_coupled_is_step26v5_csf.csv")
    v5_csf = v5[v5["has_csf"] == True].dropna(subset=["csf_mean_obs", "csf_pred_at_post_mean"])  # noqa

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.5))

    # Panel A: predicted vs observed
    ax1.scatter(v5_csf.csf_pred_at_post_mean, v5_csf.csf_mean_obs,
                s=15, alpha=0.5, c=v5_csf.ess_frac, cmap="viridis_r",
                vmin=0, vmax=1.0)
    lims = [200, max(v5_csf.csf_mean_obs.max(), v5_csf.csf_pred_at_post_mean.max()) * 1.05]
    ax1.plot(lims, lims, "k--", alpha=0.4, linewidth=1)
    ax1.set_xlabel("CSF predicted (pg/mL)")
    ax1.set_ylabel("CSF observed mean (pg/mL)")
    ax1.set_title("A. CSF model fit (N=277)")
    plt.colorbar(ax1.collections[0], ax=ax1, label="ESS fraction", shrink=0.8)

    # Panel B: residual histogram
    resid_pct = (v5_csf.csf_residual_pgml / v5_csf.csf_mean_obs) * 100
    ax2.hist(resid_pct, bins=40, color=OI_PURPLE, alpha=0.75, edgecolor="white")
    ax2.axvline(0, color="k", linestyle="-", alpha=0.5)
    ax2.axvline(resid_pct.median(), color=OI_RED, linewidth=2,
                label=f"Median: {resid_pct.median():.1f}%")
    ax2.set_xlabel("CSF residual (% of observed)")
    ax2.set_ylabel("Number of patients")
    ax2.set_title("B. CSF residual distribution")
    ax2.legend()

    fig.suptitle("CSF Total α-Synuclein Observation Model Quality",
                 fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout()
    save_fig(fig, "fig4_csf_fit_quality")
    plt.close(fig)


def fig5_counterfactual():
    """Fig 5: Prasinezumab counterfactual treatment delays."""
    cf = pd.read_csv(OUT_DIR.parent / "block4_s5_counterfactual.csv")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.5))

    # Panel A: HIGH-INFO treatment delay by scenario
    hi = cf[cf.ess_frac < 0.20]
    scenarios = [("A_minimal", 0.05, OI_CYAN),
                 ("B_pasadena", 0.15, OI_ORANGE),
                 ("C_optimistic", 0.35, OI_GREEN)]

    for scenario, eta, color in scenarios:
        col = f"delay_50pct_{scenario}_median_yr"
        ax1.hist(hi[col], bins=30, alpha=0.6, color=color, edgecolor="white",
                 label=f"η={eta} (median {hi[col].median():.1f} yr)")
    ax1.set_xlabel("Delay in time to 50% SBR decline (years)")
    ax1.set_ylabel("Number of patients")
    ax1.set_title(f"A. HIGH-INFO patients (N={len(hi)})")
    ax1.legend(fontsize=8)

    # Panel B: Untreated vs treated trajectory for median patient
    t_med = hi["t_50pct_untreated_median_yr"].median()
    T_tox_median = np.log(2) / (GAMMA * t_med * HR_PER_YR)  # back-solve

    t_years = np.linspace(0, 15, 200)
    sbr_untreated = np.exp(-GAMMA * T_tox_median * t_years * HR_PER_YR)

    ax2.plot(t_years, sbr_untreated * 100, color=OI_BLACK, linewidth=2,
             label="Untreated")
    for scenario, eta, color in scenarios:
        sbr_treated = np.exp(-GAMMA * (1 - eta) * T_tox_median * t_years * HR_PER_YR)
        ax2.plot(t_years, sbr_treated * 100, color=color, linewidth=1.5,
                 linestyle="--", label=f"η={eta}")
    ax2.axhline(50, color="gray", linestyle=":", alpha=0.5)
    ax2.text(14, 52, "50% threshold", fontsize=7, color="gray", ha="right")
    ax2.set_xlabel("Time from baseline (years)")
    ax2.set_ylabel("SBR relative to baseline (%)")
    ax2.set_title("B. Median HIGH-INFO patient trajectory")
    ax2.set_ylim(0, 105)
    ax2.legend(fontsize=8, loc="lower left")

    fig.suptitle("Prasinezumab Counterfactual: Predicted Treatment Delay",
                 fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout()
    save_fig(fig, "fig5_counterfactual")
    plt.close(fig)


def fig6_ess_landscape():
    """Fig 6: ESS stratification landscape — cohort informativeness."""
    v5 = pd.read_csv(POST_DIR / "phase2_coupled_is_step26v5_csf.csv")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.5))

    # Panel A: ESS distribution with strata
    bins = np.linspace(0, 1, 50)
    colors = [OI_BLUE, OI_ORANGE, OI_RED]
    labels = ["HIGH-INFO (<20%)", "MOD-INFO (20-50%)", "LOW-INFO (≥50%)"]
    masks = [v5.ess_frac < 0.20,
             (v5.ess_frac >= 0.20) & (v5.ess_frac < 0.50),
             v5.ess_frac >= 0.50]
    for mask, color, label in zip(masks, colors, labels):
        ax1.hist(v5.loc[mask, "ess_frac"], bins=bins, alpha=0.65,
                 color=color, edgecolor="white", label=f"{label} (N={mask.sum()})")
    ax1.set_xlabel("ESS fraction")
    ax1.set_ylabel("Number of patients")
    ax1.set_title("A. Importance-sampling informativeness")
    ax1.legend(fontsize=8)

    # Panel B: k_n constraint vs cor
    sc = ax2.scatter(v5.log_k_n_sd_prior_ratio, v5.cor_logk_logalpha,
                     s=15, alpha=0.5, c=v5.ess_frac, cmap="viridis_r",
                     vmin=0, vmax=1.0)
    ax2.axhline(0, color="k", alpha=0.3)
    ax2.axhline(-0.5, color=OI_RED, linestyle="--", alpha=0.5,
                label="Sloppy ridge threshold")
    ax2.set_xlabel("k_n SD / prior SD (lower = more constrained)")
    ax2.set_ylabel("cor(log k_n, log α_tox)")
    ax2.set_title("B. Constraint landscape")
    ax2.legend(fontsize=8)
    plt.colorbar(sc, ax=ax2, label="ESS fraction", shrink=0.8)

    fig.suptitle("Cohort Informativeness Under Joint SBR + CSF Calibration",
                 fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout()
    save_fig(fig, "fig6_ess_landscape")
    plt.close(fig)


def main() -> int:
    print("=" * 60)
    print("Generating Phase 2 Publication Figures")
    print("=" * 60)
    print(f"Output directory: {OUT_DIR}")
    print()

    fig2_ttox_neuron_loss()
    fig3_degeneracy_breaking()
    fig4_csf_fit()
    fig5_counterfactual()
    fig6_ess_landscape()

    print()
    print(f"All figures saved to: {OUT_DIR}")
    print("Formats: PNG (300 DPI) + PDF (vector)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
