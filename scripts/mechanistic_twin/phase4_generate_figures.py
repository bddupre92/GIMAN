#!/usr/bin/env python3
"""
Phase 4 — Publication Figures for Paper 9

Generates 10 figures from Phase 4 results:
  Fig 1: Model schematic (three-pathway conceptual diagram)
  Fig 2: N(t)/N0 distribution
  Fig 3: Path A — N(t) vs time for OFF-UPDRS prediction (2-panel)
  Fig 4: Path A — Model comparison bar chart (AIC)
  Fig 5: Path B — ON-OFF gap vs N(t)/N0 by LEDD quartile (KEY FIGURE)
  Fig 6: Path B — Model comparison (DAIC)
  Fig 7: Path B — Gap by LEDD quartile and N(t) tertile
  Fig 8: Path C — KM curves by progression group
  Fig 9: Corrected decisive test summary (2x2 panel)
  Fig 10: Three-pathway summary (3-panel)

Output: outputs/mechanistic_twin/phase4/figures/
"""

from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
import seaborn as sns

# ── Project root ──────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

PHASE4_DIR = PROJECT_ROOT / "outputs" / "mechanistic_twin" / "phase4"
FIG_DIR = PHASE4_DIR / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# ── Style ─────────────────────────────────────────────────────────────
sns.set_style("whitegrid")
CB_PALETTE = sns.color_palette("colorblind", 10)
DPI = 300
SINGLE_COL = 3.5  # inches
DOUBLE_COL = 7.0  # inches
FONT_SIZE = 10

plt.rcParams.update({
    "font.size": FONT_SIZE,
    "axes.labelsize": FONT_SIZE,
    "axes.titlesize": FONT_SIZE,
    "xtick.labelsize": FONT_SIZE - 1,
    "ytick.labelsize": FONT_SIZE - 1,
    "legend.fontsize": FONT_SIZE - 1,
    "figure.dpi": DPI,
    "savefig.dpi": DPI,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
    "pdf.fonttype": 42,      # TrueType for publication
    "ps.fonttype": 42,
})


# ── Helpers ───────────────────────────────────────────────────────────
def save_fig(fig, name: str):
    """Save figure as PNG + PDF."""
    for ext in ("png", "pdf"):
        fig.savefig(FIG_DIR / f"{name}.{ext}", dpi=DPI)
    plt.close(fig)
    print(f"  Saved {name}.png + .pdf")


def load_json(name: str) -> dict:
    path = PHASE4_DIR / name
    with open(path) as f:
        return json.load(f)


def load_assembled_data() -> pd.DataFrame:
    """Load assembled data and recompute n_frac using pct_loss_per_yr."""
    df = pd.read_parquet(PHASE4_DIR / "phase4_assembled_data.parquet")
    # Recompute n_frac correctly (the parquet n_frac uses T_tox which clusters near 1.0)
    mask = df["has_posterior"] == 1
    df.loc[mask, "n_frac"] = (
        (1 - df.loc[mask, "pct_loss_per_yr_median"] / 100)
        ** (df.loc[mask, "months_from_baseline"] / 12)
    )
    return df


# ======================================================================
# Fig 1: Model Schematic
# ======================================================================
def fig1_model_schematic():
    print("Fig 1: Model schematic")
    fig, ax = plt.subplots(figsize=(DOUBLE_COL, 3.5))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 5)
    ax.axis("off")

    # -- Central box: N(t)/N0
    nbox = FancyBboxPatch(
        (3.8, 2.0), 2.4, 1.0,
        boxstyle="round,pad=0.15", facecolor=CB_PALETTE[0],
        edgecolor="black", linewidth=1.5, alpha=0.85,
    )
    ax.add_patch(nbox)
    ax.text(5.0, 2.5, r"$N(t)/N_0$" + "\nNeuron survival", ha="center", va="center",
            fontsize=FONT_SIZE + 1, fontweight="bold", color="white")

    # -- Path A box
    a_box = FancyBboxPatch(
        (7.2, 3.5), 2.4, 0.9,
        boxstyle="round,pad=0.12", facecolor=CB_PALETTE[1],
        edgecolor="black", linewidth=1.2, alpha=0.7,
    )
    ax.add_patch(a_box)
    ax.text(8.4, 3.95, "Path A\nOFF-UPDRS", ha="center", va="center",
            fontsize=FONT_SIZE - 1, fontweight="bold")

    # -- Path B box
    b_box = FancyBboxPatch(
        (7.2, 1.8), 2.4, 1.2,
        boxstyle="round,pad=0.12", facecolor=CB_PALETTE[2],
        edgecolor="black", linewidth=1.2, alpha=0.7,
    )
    ax.add_patch(b_box)
    ax.text(8.4, 2.4, "Path B\nON-OFF Gap\n" + r"($\times$ LEDD)",
            ha="center", va="center", fontsize=FONT_SIZE - 1, fontweight="bold")

    # -- Path C box
    c_box = FancyBboxPatch(
        (7.2, 0.4), 2.4, 0.9,
        boxstyle="round,pad=0.12", facecolor=CB_PALETTE[3],
        edgecolor="black", linewidth=1.2, alpha=0.7,
    )
    ax.add_patch(c_box)
    ax.text(8.4, 0.85, "Path C\nWearing-off", ha="center", va="center",
            fontsize=FONT_SIZE - 1, fontweight="bold")

    # -- LEDD box (input to Path B)
    ledd_box = FancyBboxPatch(
        (0.3, 1.9), 2.0, 0.8,
        boxstyle="round,pad=0.12", facecolor=CB_PALETTE[4],
        edgecolor="black", linewidth=1.2, alpha=0.7,
    )
    ax.add_patch(ledd_box)
    ax.text(1.3, 2.3, "LEDD\n(treatment)", ha="center", va="center",
            fontsize=FONT_SIZE - 1, fontweight="bold")

    # -- Phase 2 source box
    p2_box = FancyBboxPatch(
        (0.3, 3.5), 2.8, 0.9,
        boxstyle="round,pad=0.12", facecolor="lightgray",
        edgecolor="black", linewidth=1.0, alpha=0.6,
    )
    ax.add_patch(p2_box)
    ax.text(1.7, 3.95, "Phase 2 ODE\n" + r"$\alpha$-syn $\to$ N(t)",
            ha="center", va="center", fontsize=FONT_SIZE - 1, style="italic")

    # -- Arrows
    arrow_kw = dict(arrowstyle="-|>", color="black", linewidth=1.5,
                    mutation_scale=15, connectionstyle="arc3,rad=0")

    # Phase 2 -> N(t)
    ax.annotate("", xy=(3.8, 2.7), xytext=(3.1, 3.95),
                arrowprops=arrow_kw)

    # N(t) -> Path A
    ax.annotate("", xy=(7.2, 3.95), xytext=(6.2, 2.8),
                arrowprops=arrow_kw)
    # N(t) -> Path B
    ax.annotate("", xy=(7.2, 2.5), xytext=(6.2, 2.5),
                arrowprops=arrow_kw)
    # N(t) -> Path C
    ax.annotate("", xy=(7.2, 0.85), xytext=(6.2, 2.2),
                arrowprops=arrow_kw)
    # LEDD -> Path B
    ax.annotate("", xy=(7.2, 2.3), xytext=(2.3, 2.3),
                arrowprops=dict(arrowstyle="-|>", color=CB_PALETTE[4],
                                linewidth=1.5, mutation_scale=15,
                                connectionstyle="arc3,rad=-0.1"))

    # -- Result annotations
    ax.text(8.4, 4.55, r"$R^2 = 0.051$ (informative $-$)",
            ha="center", fontsize=FONT_SIZE - 2, color=CB_PALETTE[1], style="italic")
    ax.text(8.4, 1.55, r"$p_{interaction} = 0.011$ ($+$)",
            ha="center", fontsize=FONT_SIZE - 2, color=CB_PALETTE[2], fontweight="bold")
    ax.text(8.4, 0.15, r"$\rho = -0.050$ (null)",
            ha="center", fontsize=FONT_SIZE - 2, color=CB_PALETTE[3], style="italic")

    save_fig(fig, "fig1_model_schematic")


# ======================================================================
# Fig 2: N(t)/N0 Distribution
# ======================================================================
def fig2_nfrac_distribution():
    print("Fig 2: N(t)/N0 distribution")
    df = load_assembled_data()
    df_post = df[(df["has_posterior"] == 1) & df["n_frac"].notna()].copy()

    # Create progression tertiles based on pct_loss_per_yr
    pat_rate = df_post.groupby("PATNO")["pct_loss_per_yr_median"].first()
    tertile_edges = pat_rate.quantile([1/3, 2/3]).values
    def assign_tertile(r):
        if r <= tertile_edges[0]:
            return "Slow"
        elif r <= tertile_edges[1]:
            return "Medium"
        else:
            return "Fast"
    pat_tertile = pat_rate.apply(assign_tertile).rename("prog_tertile")
    df_post = df_post.merge(pat_tertile, left_on="PATNO", right_index=True, how="left")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(DOUBLE_COL, 2.8))

    # Left: overall histogram
    ax1.hist(df_post["n_frac"].dropna(), bins=60, color=CB_PALETTE[0],
             edgecolor="white", linewidth=0.3, alpha=0.85)
    ax1.set_xlabel(r"$N(t)/N_0$")
    ax1.set_ylabel("Count")
    ax1.text(0.02, 0.95, "(a)", transform=ax1.transAxes, fontweight="bold",
             va="top", fontsize=FONT_SIZE)
    ax1.axvline(df_post["n_frac"].median(), color="black", ls="--", lw=1,
                label=f"Median = {df_post['n_frac'].median():.3f}")
    ax1.legend(frameon=True, loc="upper left", fontsize=FONT_SIZE - 2)

    # Right: density by progression tertile
    for i, tert in enumerate(["Slow", "Medium", "Fast"]):
        subset = df_post[df_post["prog_tertile"] == tert]["n_frac"].dropna()
        ax2.hist(subset, bins=50, density=True, alpha=0.45,
                 color=CB_PALETTE[i], label=tert, edgecolor="white", linewidth=0.3)
    ax2.set_xlabel(r"$N(t)/N_0$")
    ax2.set_ylabel("Density")
    ax2.legend(title="Progression rate", frameon=True, fontsize=FONT_SIZE - 2)
    ax2.text(0.02, 0.95, "(b)", transform=ax2.transAxes, fontweight="bold",
             va="top", fontsize=FONT_SIZE)

    fig.tight_layout()
    save_fig(fig, "fig2_nfrac_distribution")


# ======================================================================
# Fig 3: Path A — N(t) vs Time scatter (2-panel)
# ======================================================================
def fig3_path_a_scatter():
    print("Fig 3: Path A scatter")
    res_a = load_json("phase4_path_a_results.json")
    df = load_assembled_data()
    df_off = df[(df["has_posterior"] == 1) & df["updrs3_off"].notna() & df["n_frac"].notna()].copy()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(DOUBLE_COL, 3.0))

    # Left: N(t)/N0 vs UPDRS3_OFF
    ax1.scatter(df_off["n_frac"], df_off["updrs3_off"], s=3, alpha=0.15,
                color=CB_PALETTE[0], rasterized=True)
    # OLS line
    coefs_a1 = res_a["model_a1_ols"]["coefs"]
    x_line = np.linspace(df_off["n_frac"].min(), df_off["n_frac"].max(), 100)
    y_line = coefs_a1["intercept"] + coefs_a1["n_frac"] * x_line
    ax1.plot(x_line, y_line, color=CB_PALETTE[3], linewidth=2, label=f"OLS ($R^2 = 0.051$)")
    ax1.set_xlabel(r"$N(t)/N_0$")
    ax1.set_ylabel("UPDRS-III (OFF)")
    ax1.legend(frameon=True, fontsize=FONT_SIZE - 2)
    ax1.text(0.02, 0.95, "(a)", transform=ax1.transAxes, fontweight="bold",
             va="top", fontsize=FONT_SIZE)

    # Right: Years vs UPDRS3_OFF
    ax2.scatter(df_off["years_from_baseline"], df_off["updrs3_off"], s=3, alpha=0.15,
                color=CB_PALETTE[1], rasterized=True)
    coefs_a4 = res_a["model_a4_time_only"]["coefs"]
    x_t = np.linspace(0, df_off["years_from_baseline"].max(), 100)
    y_t = coefs_a4["intercept"] + coefs_a4["years_from_baseline"] * x_t
    ax2.plot(x_t, y_t, color=CB_PALETTE[3], linewidth=2, label=f"OLS ($R^2 = 0.155$)")
    ax2.set_xlabel("Years from baseline")
    ax2.set_ylabel("UPDRS-III (OFF)")
    ax2.legend(frameon=True, fontsize=FONT_SIZE - 2)
    ax2.text(0.02, 0.95, "(b)", transform=ax2.transAxes, fontweight="bold",
             va="top", fontsize=FONT_SIZE)

    fig.tight_layout()
    save_fig(fig, "fig3_path_a_nfrac_vs_time")


# ======================================================================
# Fig 4: Path A — AIC Model Comparison
# ======================================================================
def fig4_path_a_aic():
    print("Fig 4: Path A AIC comparison")
    res_a = load_json("phase4_path_a_results.json")

    models = ["A1: OLS\n(N/N$_0$)", "A2: LME\n(N/N$_0$)",
              "A3: Hill\n(N/N$_0$)", "A4: OLS\n(time)", "A5: LME\n(time)"]
    aics = [
        res_a["model_a1_ols"]["aic"],
        res_a["model_a2_mixed"]["aic"],
        res_a["model_a3_hill"]["aic"],
        res_a["model_a4_time_only"]["aic"],
        res_a["model_a5_time_mixed"]["aic"],
    ]

    fig, ax = plt.subplots(figsize=(SINGLE_COL, 3.0))
    colors = [CB_PALETTE[0]] * 3 + [CB_PALETTE[1]] * 2
    bars = ax.bar(range(len(models)), aics, color=colors, edgecolor="black", linewidth=0.5)

    # Highlight winner
    best_idx = np.argmin(aics)
    bars[best_idx].set_edgecolor(CB_PALETTE[3])
    bars[best_idx].set_linewidth(2.5)

    ax.set_xticks(range(len(models)))
    ax.set_xticklabels(models, fontsize=FONT_SIZE - 2)
    ax.set_ylabel("AIC")

    # Star the winner
    ax.annotate("best", xy=(best_idx, aics[best_idx]),
                xytext=(best_idx, aics[best_idx] - 800),
                ha="center", fontsize=FONT_SIZE - 2, fontweight="bold",
                color=CB_PALETTE[3],
                arrowprops=dict(arrowstyle="->", color=CB_PALETTE[3]))

    fig.tight_layout()
    save_fig(fig, "fig4_path_a_aic_comparison")


# ======================================================================
# Fig 5: Path B — ON-OFF Gap vs N(t)/N0 by LEDD quartile (KEY FIGURE)
# ======================================================================
def fig5_path_b_gap_scatter():
    print("Fig 5: Path B gap scatter (KEY FIGURE)")
    # Need raw data: load assembled + raw UPDRS-III for paired ON-OFF
    # Reconstruct from the Path B script's data source
    updrs_path = PROJECT_ROOT / "data" / "00_raw" / "MDS-UPDRS Part IV" / "MDS-UPDRS_Part_III_12Apr2026.csv"
    assembled = load_assembled_data()

    updrs = pd.read_csv(updrs_path, low_memory=False)
    updrs["updrs3_total"] = pd.to_numeric(updrs["NP3TOT"], errors="coerce")
    updrs["PATNO"] = updrs["PATNO"].astype(str)

    on_df = (updrs[updrs["PDSTATE"] == "ON"][["PATNO", "EVENT_ID", "updrs3_total"]]
             .rename(columns={"updrs3_total": "updrs3_on"}).dropna(subset=["updrs3_on"]))
    off_df = (updrs[updrs["PDSTATE"] == "OFF"][["PATNO", "EVENT_ID", "updrs3_total"]]
              .rename(columns={"updrs3_total": "updrs3_off_raw"}).dropna(subset=["updrs3_off_raw"]))
    paired = on_df.merge(off_df, on=["PATNO", "EVENT_ID"], how="inner")
    paired["gap"] = paired["updrs3_off_raw"] - paired["updrs3_on"]

    # Merge with assembled for LEDD + n_frac
    assembled["PATNO"] = assembled["PATNO"].astype(str)
    asm_cols = ["PATNO", "EVENT_ID", "ledd_total", "n_frac", "pct_loss_per_yr_median",
                "months_from_baseline", "has_posterior"]
    merged = paired.merge(assembled[asm_cols], on=["PATNO", "EVENT_ID"], how="inner")
    merged = merged[(merged["has_posterior"] == 1) & merged["n_frac"].notna()
                     & merged["ledd_total"].notna() & (merged["ledd_total"] > 0)].copy()

    # LEDD quartiles
    merged["ledd_q"] = pd.qcut(merged["ledd_total"], 4, labels=["Q1 (low)", "Q2", "Q3", "Q4 (high)"])

    fig, ax = plt.subplots(figsize=(DOUBLE_COL, 4.0))

    quartile_colors = [CB_PALETTE[0], CB_PALETTE[1], CB_PALETTE[2], CB_PALETTE[3]]
    for i, q in enumerate(["Q1 (low)", "Q2", "Q3", "Q4 (high)"]):
        sub = merged[merged["ledd_q"] == q]
        ax.scatter(sub["n_frac"], sub["gap"], s=8, alpha=0.25,
                   color=quartile_colors[i], rasterized=True)
        # Regression line per quartile
        if len(sub) > 10:
            z = np.polyfit(sub["n_frac"], sub["gap"], 1)
            x_line = np.linspace(sub["n_frac"].min(), sub["n_frac"].max(), 100)
            ax.plot(x_line, np.polyval(z, x_line), color=quartile_colors[i],
                    linewidth=2.5, label=f"{q} (n={len(sub)})")

    ax.set_xlabel(r"$N(t)/N_0$")
    ax.set_ylabel("ON-OFF Gap (UPDRS-III points)")
    ax.axhline(0, color="gray", linewidth=0.5, linestyle="--")
    ax.legend(title="LEDD quartile", frameon=True, fontsize=FONT_SIZE - 1)

    fig.tight_layout()
    save_fig(fig, "fig5_path_b_gap_vs_nfrac")


# ======================================================================
# Fig 6: Path B — DAIC Model Comparison
# ======================================================================
def fig6_path_b_daic():
    print("Fig 6: Path B DAIC comparison")
    res_b = load_json("phase4_path_b_results.json")
    mr = res_b["model_results"]

    models = ["B0: Null", "B1: N/N$_0$", "B2: LEDD",
              "B3: Interaction", "B4a: Hill\n(h=2)",
              "B4b: Hill\n(h free)"]
    aics = [
        mr["B0_null"]["AIC"],
        mr["B1_nfrac"]["AIC"],
        mr["B2_ledd_matched"]["AIC"],
        mr["B3_interaction"]["AIC"],
        mr["B4a_hill_h2"]["AIC"],
        mr["B4b_hill_hfree"]["AIC"],
    ]

    # Compute DAIC relative to best (B3)
    best_aic = min(aics)
    daics = [a - best_aic for a in aics]

    fig, ax = plt.subplots(figsize=(SINGLE_COL + 1, 3.0))

    colors = []
    for i, d in enumerate(daics):
        if d == 0:
            colors.append(CB_PALETTE[2])  # winner
        elif i >= 4:
            colors.append(CB_PALETTE[3])  # Hill failures
        else:
            colors.append(CB_PALETTE[0])

    bars = ax.bar(range(len(models)), daics, color=colors, edgecolor="black", linewidth=0.5)

    # Highlight winner
    best_idx = np.argmin(daics)
    bars[best_idx].set_edgecolor(CB_PALETTE[2])
    bars[best_idx].set_linewidth(2.5)

    ax.set_xticks(range(len(models)))
    ax.set_xticklabels(models, fontsize=FONT_SIZE - 2, rotation=20, ha="right")
    ax.set_ylabel(r"$\Delta$AIC (vs. best)")
    ax.axhline(0, color="black", linewidth=0.5)

    # Annotate Hill failure
    for idx in [4, 5]:
        ax.text(idx, daics[idx] + 3, "fails", ha="center",
                fontsize=FONT_SIZE - 2, color=CB_PALETTE[3], fontweight="bold")

    fig.tight_layout()
    save_fig(fig, "fig6_path_b_daic_comparison")


# ======================================================================
# Fig 7: Path B — Gap by LEDD quartile and N(t) tertile
# ======================================================================
def fig7_path_b_grouped_bar():
    print("Fig 7: Path B grouped bar chart")
    # Reconstruct paired data again
    updrs_path = PROJECT_ROOT / "data" / "00_raw" / "MDS-UPDRS Part IV" / "MDS-UPDRS_Part_III_12Apr2026.csv"
    assembled = load_assembled_data()
    assembled["PATNO"] = assembled["PATNO"].astype(str)

    updrs = pd.read_csv(updrs_path, low_memory=False)
    updrs["updrs3_total"] = pd.to_numeric(updrs["NP3TOT"], errors="coerce")
    updrs["PATNO"] = updrs["PATNO"].astype(str)

    on_df = (updrs[updrs["PDSTATE"] == "ON"][["PATNO", "EVENT_ID", "updrs3_total"]]
             .rename(columns={"updrs3_total": "updrs3_on"}).dropna(subset=["updrs3_on"]))
    off_df = (updrs[updrs["PDSTATE"] == "OFF"][["PATNO", "EVENT_ID", "updrs3_total"]]
              .rename(columns={"updrs3_total": "updrs3_off_raw"}).dropna(subset=["updrs3_off_raw"]))
    paired = on_df.merge(off_df, on=["PATNO", "EVENT_ID"], how="inner")
    paired["gap"] = paired["updrs3_off_raw"] - paired["updrs3_on"]

    asm_cols = ["PATNO", "EVENT_ID", "ledd_total", "n_frac", "has_posterior"]
    merged = paired.merge(assembled[asm_cols], on=["PATNO", "EVENT_ID"], how="inner")
    merged = merged[(merged["has_posterior"] == 1) & merged["n_frac"].notna()
                     & merged["ledd_total"].notna() & (merged["ledd_total"] > 0)].copy()

    merged["ledd_q"] = pd.qcut(merged["ledd_total"], 4, labels=["Q1", "Q2", "Q3", "Q4"])
    merged["n_tertile"] = pd.qcut(merged["n_frac"], 3, labels=["Low N/N$_0$", "Medium", "High N/N$_0$"])

    # Compute means
    grouped = merged.groupby(["ledd_q", "n_tertile"], observed=True)["gap"].agg(["mean", "sem"]).reset_index()

    fig, ax = plt.subplots(figsize=(DOUBLE_COL, 3.0))
    tertiles = ["Low N/N$_0$", "Medium", "High N/N$_0$"]
    x = np.arange(4)
    width = 0.25

    for i, tert in enumerate(tertiles):
        sub = grouped[grouped["n_tertile"] == tert]
        offset = (i - 1) * width
        ax.bar(x + offset, sub["mean"].values, width, yerr=sub["sem"].values,
               color=CB_PALETTE[i], edgecolor="black", linewidth=0.5,
               label=tert, capsize=3)

    ax.set_xticks(x)
    ax.set_xticklabels(["Q1 (low)", "Q2", "Q3", "Q4 (high)"])
    ax.set_xlabel("LEDD quartile")
    ax.set_ylabel("Mean ON-OFF gap (UPDRS-III points)")
    ax.legend(title=r"$N(t)/N_0$ tertile", frameon=True, fontsize=FONT_SIZE - 2)

    fig.tight_layout()
    save_fig(fig, "fig7_path_b_gap_by_ledd_nfrac")


# ======================================================================
# Fig 8: Path C — KM Curves
# ======================================================================
def fig8_path_c_km():
    print("Fig 8: Path C KM curves")
    res_c = load_json("phase4_path_c_results.json")
    primary = res_c["analyses"]["primary"]

    # Load assembled data with wearing-off info
    df = load_assembled_data()
    df_post = df[(df["has_posterior"] == 1)].copy()

    # Assign progression tertiles
    pat_rate = df_post.groupby("PATNO")["pct_loss_per_yr_median"].first()
    tertile_edges = pat_rate.quantile([1/3, 2/3]).values

    def assign_tertile(r):
        if r <= tertile_edges[0]:
            return "Slow"
        elif r <= tertile_edges[1]:
            return "Medium"
        else:
            return "Fast"
    pat_tertile = pat_rate.apply(assign_tertile).rename("prog_group")
    df_post = df_post.merge(pat_tertile, left_on="PATNO", right_index=True, how="left")

    # For each patient: time to first NP4OFF >= 1 (wearing-off)
    events = []
    for pat, grp in df_post.groupby("PATNO"):
        np4_rows = grp[grp["NP4OFF"].notna()]
        if len(np4_rows) == 0:
            continue
        prog = grp["prog_group"].iloc[0]
        onset = np4_rows[np4_rows["NP4OFF"] >= 1]
        if len(onset) > 0:
            t = onset["months_from_baseline"].min()
            events.append({"PATNO": pat, "time": max(t, 0.1), "event": 1, "prog_group": prog})
        else:
            t = np4_rows["months_from_baseline"].max()
            events.append({"PATNO": pat, "time": max(t, 0.1), "event": 0, "prog_group": prog})

    events_df = pd.DataFrame(events)

    if len(events_df) == 0:
        print("  WARNING: No KM data could be assembled, skipping fig8")
        return

    # Simple KM estimation
    fig, ax = plt.subplots(figsize=(SINGLE_COL + 1, 3.5))

    for i, group in enumerate(["Slow", "Medium", "Fast"]):
        sub = events_df[events_df["prog_group"] == group].sort_values("time")
        n = len(sub)
        if n == 0:
            continue
        times = sorted(sub["time"].unique())
        surv = 1.0
        km_times = [0]
        km_surv = [1.0]
        for t in times:
            at_risk = len(sub[sub["time"] >= t])
            d = len(sub[(sub["time"] == t) & (sub["event"] == 1)])
            if at_risk > 0:
                surv *= (1 - d / at_risk)
            km_times.append(t)
            km_surv.append(surv)
        ax.step(km_times, km_surv, where="post", color=CB_PALETTE[i],
                linewidth=2, label=f"{group} (n={n})")

    ax.set_xlabel("Months from baseline")
    ax.set_ylabel("Wearing-off-free survival")
    ax.set_xlim(0, 180)
    ax.set_ylim(0, 1.05)
    ax.legend(title="N(t) progression", frameon=True, fontsize=FONT_SIZE - 1)

    # Add log-rank p-value
    lr_p = primary["C1_km"]["logrank_fast_vs_slow"]["p_value"]
    ax.text(0.95, 0.95, f"log-rank $p = {lr_p:.3f}$",
            transform=ax.transAxes, ha="right", va="top",
            fontsize=FONT_SIZE - 1, style="italic",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

    fig.tight_layout()
    save_fig(fig, "fig8_path_c_km_curves")


# ======================================================================
# Fig 9: Corrected Decisive Test Summary (2x2 panel)
# ======================================================================
def fig9_decisive_test():
    print("Fig 9: Corrected decisive test summary")
    res_t = load_json("phase4_task0_decisive_test.json")
    res_pop = load_json("phase4_population_fit.json")

    fig, axes = plt.subplots(2, 2, figsize=(DOUBLE_COL, 5.0))

    # (a) Original test: cross-patient rho ~ 0
    ax = axes[0, 0]
    rho_orig = res_t["primary_test"]["spearman_rho"]
    p_orig = res_t["primary_test"]["p_value"]
    ax.text(0.5, 0.6, f"Cross-patient\nSpearman $\\rho$ = {rho_orig:.3f}\n$p$ = {p_orig:.3f}",
            ha="center", va="center", fontsize=FONT_SIZE + 1, transform=ax.transAxes)
    ax.text(0.5, 0.25, "Verdict: FAIL\n(confounding by indication)",
            ha="center", va="center", fontsize=FONT_SIZE - 1,
            color=CB_PALETTE[3], fontweight="bold", transform=ax.transAxes)
    ax.text(0.02, 0.95, "(a) Original test", transform=ax.transAxes,
            fontweight="bold", va="top", fontsize=FONT_SIZE)
    ax.set_xticks([]); ax.set_yticks([])

    # (b) Corrected Test A: partial correlation
    ax = axes[0, 1]
    rho_part = res_t["corrected_test_A"]["spearman_rho_partial"]
    p_part = res_t["corrected_test_A"]["spearman_p_partial"]
    r2_ols = res_t["corrected_test_A"]["ols_r_squared"]
    ax.text(0.5, 0.6, f"Partial correlation\n$\\rho_{{partial}}$ = {rho_part:.3f}\n$p$ = {p_part:.1e}",
            ha="center", va="center", fontsize=FONT_SIZE + 1, transform=ax.transAxes)
    ax.text(0.5, 0.25, "Verdict: PASS\n(LEDD correlates with residuals)",
            ha="center", va="center", fontsize=FONT_SIZE - 1,
            color=CB_PALETTE[2], fontweight="bold", transform=ax.transAxes)
    ax.text(0.02, 0.95, "(b) Test A: partial corr.", transform=ax.transAxes,
            fontweight="bold", va="top", fontsize=FONT_SIZE)
    ax.set_xticks([]); ax.set_yticks([])

    # (c) Corrected Test C: model comparison
    ax = axes[1, 0]
    daic = res_t["corrected_test_C"]["delta_aic"]
    dr2 = res_t["corrected_test_C"]["delta_r_squared"]
    ax.text(0.5, 0.6, f"Model comparison\n$\\Delta$AIC = {daic:.1f}\n$\\Delta R^2$ = {dr2:.4f}",
            ha="center", va="center", fontsize=FONT_SIZE + 1, transform=ax.transAxes)
    ax.text(0.5, 0.25, "Verdict: PASS\n(N(t)+LEDD model wins)",
            ha="center", va="center", fontsize=FONT_SIZE - 1,
            color=CB_PALETTE[2], fontweight="bold", transform=ax.transAxes)
    ax.text(0.02, 0.95, r"(c) Test C: $\Delta$AIC", transform=ax.transAxes,
            fontweight="bold", va="top", fontsize=FONT_SIZE)
    ax.set_xticks([]); ax.set_yticks([])

    # (d) H1 Hill model comparison
    ax = axes[1, 1]
    daic_cf = res_pop["delta_aic"]["c_free_vs_a"]
    ax.text(0.5, 0.6,
            f"H1 Hill coupled model\n$\\Delta$AIC(coupled vs N-only)\n= +{daic_cf:.1f}",
            ha="center", va="center", fontsize=FONT_SIZE + 1, transform=ax.transAxes)
    ax.text(0.5, 0.25, "Verdict: FAIL\n(N(t)-only wins; Hill in sub-EC$_{50}$)",
            ha="center", va="center", fontsize=FONT_SIZE - 1,
            color=CB_PALETTE[3], fontweight="bold", transform=ax.transAxes)
    ax.text(0.02, 0.95, "(d) H1 Hill model", transform=ax.transAxes,
            fontweight="bold", va="top", fontsize=FONT_SIZE)
    ax.set_xticks([]); ax.set_yticks([])

    fig.tight_layout()
    save_fig(fig, "fig9_corrected_decisive_test")


# ======================================================================
# Fig 10: Three-Pathway Summary (3-panel)
# ======================================================================
def fig10_three_pathway_summary():
    print("Fig 10: Three-pathway summary")
    res_a = load_json("phase4_path_a_results.json")
    res_b = load_json("phase4_path_b_results.json")
    res_c = load_json("phase4_path_c_results.json")

    fig, axes = plt.subplots(1, 3, figsize=(DOUBLE_COL, 2.8))

    # (a) Path A: R2 comparison
    ax = axes[0]
    r2_nfrac = res_a["model_a1_ols"]["r2"]
    r2_time = res_a["model_a4_time_only"]["r2"]
    bars = ax.bar(["N(t)/N$_0$", "Time"], [r2_nfrac, r2_time],
                  color=[CB_PALETTE[0], CB_PALETTE[1]], edgecolor="black", linewidth=0.5)
    ax.set_ylabel("$R^2$")
    for bar, val in zip(bars, [r2_nfrac, r2_time]):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f"{val:.3f}", ha="center", fontsize=FONT_SIZE - 2)
    ax.set_ylim(0, 0.22)
    ax.text(0.02, 0.95, "(a) Path A: OFF-UPDRS", transform=ax.transAxes,
            fontweight="bold", va="top", fontsize=FONT_SIZE - 1)
    ax.text(0.5, 0.85, "informative negative",
            transform=ax.transAxes, ha="center", fontsize=FONT_SIZE - 2,
            style="italic", color=CB_PALETTE[3])

    # (b) Path B: interaction effect
    ax = axes[1]
    mr = res_b["model_results"]
    models_b = ["Null", "N/N$_0$", "LEDD", "Interact."]
    r2s_b = [mr["B0_null"]["R2"], mr["B1_nfrac"]["R2"],
             mr["B2_ledd_matched"]["R2"], mr["B3_interaction"]["R2"]]
    bar_colors = [CB_PALETTE[7], CB_PALETTE[0], CB_PALETTE[1], CB_PALETTE[2]]
    bars = ax.bar(models_b, r2s_b, color=bar_colors, edgecolor="black", linewidth=0.5)
    bars[3].set_edgecolor(CB_PALETTE[2])
    bars[3].set_linewidth(2.5)
    ax.set_ylabel("$R^2$")
    for bar, val in zip(bars, r2s_b):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                f"{val:.3f}", ha="center", fontsize=FONT_SIZE - 2, rotation=45)
    ax.set_ylim(0, 0.075)
    ax.text(0.02, 0.95, "(b) Path B: ON-OFF Gap", transform=ax.transAxes,
            fontweight="bold", va="top", fontsize=FONT_SIZE - 1)
    ax.text(0.5, 0.85, "$p_{interaction}$ = 0.011",
            transform=ax.transAxes, ha="center", fontsize=FONT_SIZE - 1,
            fontweight="bold", color=CB_PALETTE[2])

    # (c) Path C: C-index and rho
    ax = axes[2]
    c_idx = res_c["analyses"]["primary"]["C2_cox"]["concordance_index"]
    rho_c = res_c["analyses"]["primary"]["C4_spearman"]["events_only"]["spearman_rho"]
    p_c = res_c["analyses"]["primary"]["C4_spearman"]["events_only"]["p_value"]

    ax.text(0.5, 0.65, f"C-index = {c_idx:.3f}\n$\\rho$ = {rho_c:.3f}\n$p$ = {p_c:.3f}",
            ha="center", va="center", fontsize=FONT_SIZE + 1, transform=ax.transAxes)
    ax.text(0.5, 0.30, "Null result",
            ha="center", va="center", fontsize=FONT_SIZE,
            style="italic", color=CB_PALETTE[3], transform=ax.transAxes)
    ax.text(0.02, 0.95, "(c) Path C: Wearing-off", transform=ax.transAxes,
            fontweight="bold", va="top", fontsize=FONT_SIZE - 1)
    ax.set_xticks([]); ax.set_yticks([])
    # Add light border
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.5)
        spine.set_color("gray")

    fig.tight_layout()
    save_fig(fig, "fig10_three_pathway_summary")


# ======================================================================
# Main
# ======================================================================
def main():
    print("=" * 70)
    print("Phase 4 — Generating publication figures")
    print(f"  Output dir: {FIG_DIR}")
    print("=" * 70)

    fig1_model_schematic()
    fig2_nfrac_distribution()
    fig3_path_a_scatter()
    fig4_path_a_aic()
    fig5_path_b_gap_scatter()
    fig6_path_b_daic()
    fig7_path_b_grouped_bar()
    fig8_path_c_km()
    fig9_decisive_test()
    fig10_three_pathway_summary()

    print("\n" + "=" * 70)
    print(f"All figures saved to {FIG_DIR}")
    figs = sorted(FIG_DIR.glob("*.png"))
    print(f"  {len(figs)} PNG files generated")
    print("=" * 70)


if __name__ == "__main__":
    main()
