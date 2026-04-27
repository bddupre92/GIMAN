#!/usr/bin/env python3
"""
Phase 4 — Refined Publication Figures for Paper 9 (Priority 3)

Generates refined versions of:
  Fig 1:  Model schematic (three-pathway conceptual diagram with verdicts)
  Fig 5:  Path B — ON-OFF gap vs N(t)/N0 by LEDD quartile (KEY FIGURE)
  Fig 10: Three-pathway summary (3-panel "money figure")

Output: outputs/mechanistic_twin/phase4/figures/  (overwrites existing)
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
from scipy import stats

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
SINGLE_COL = 3.5   # inches
DOUBLE_COL = 7.0   # inches
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
    "pdf.fonttype": 42,       # TrueType for publication
    "ps.fonttype": 42,
    "font.family": "sans-serif",
    "mathtext.fontset": "dejavusans",
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
    mask = df["has_posterior"] == 1
    df.loc[mask, "n_frac"] = (
        (1 - df.loc[mask, "pct_loss_per_yr_median"] / 100)
        ** (df.loc[mask, "months_from_baseline"] / 12)
    )
    return df


def extract_paired_on_off_data() -> pd.DataFrame:
    """Extract paired ON-OFF UPDRS-III data merged with LEDD + N(t)/N0.

    Replicates the logic from phase4_path_b_on_off_gap.py.
    """
    updrs_path = (PROJECT_ROOT / "data" / "00_raw" / "MDS-UPDRS Part IV"
                  / "MDS-UPDRS_Part_III_12Apr2026.csv")
    assembled = load_assembled_data()
    assembled["PATNO"] = assembled["PATNO"].astype(str)

    updrs = pd.read_csv(updrs_path, low_memory=False)
    updrs["updrs3_total"] = pd.to_numeric(updrs["NP3TOT"], errors="coerce")
    updrs["PATNO"] = updrs["PATNO"].astype(str)

    on_df = (
        updrs[updrs["PDSTATE"] == "ON"][["PATNO", "EVENT_ID", "updrs3_total"]]
        .rename(columns={"updrs3_total": "updrs3_on"})
        .dropna(subset=["updrs3_on"])
    )
    off_df = (
        updrs[updrs["PDSTATE"] == "OFF"][["PATNO", "EVENT_ID", "updrs3_total"]]
        .rename(columns={"updrs3_total": "updrs3_off_raw"})
        .dropna(subset=["updrs3_off_raw"])
    )
    paired = on_df.merge(off_df, on=["PATNO", "EVENT_ID"], how="inner")
    paired["gap"] = paired["updrs3_off_raw"] - paired["updrs3_on"]

    asm_cols = ["PATNO", "EVENT_ID", "ledd_total", "n_frac",
                "pct_loss_per_yr_median", "months_from_baseline", "has_posterior"]
    merged = paired.merge(assembled[asm_cols], on=["PATNO", "EVENT_ID"], how="inner")
    merged = merged[
        (merged["has_posterior"] == 1)
        & merged["n_frac"].notna()
        & merged["ledd_total"].notna()
        & (merged["ledd_total"] > 0)
    ].copy()

    merged["ledd_q"] = pd.qcut(
        merged["ledd_total"], 4,
        labels=["Q1 (low)", "Q2", "Q3", "Q4 (high)"]
    )
    return merged


# ======================================================================
# Figure 1: Model Schematic (three-pathway conceptual diagram)
# ======================================================================
def fig1_model_schematic():
    """Clean three-pathway conceptual diagram with verdicts.

    Central node: N(t)/N0 from Phase 2 ODE
    Three arrows to: OFF-UPDRS (Path A), ON-OFF Gap x LEDD (Path B),
                      Wearing-off timing (Path C)
    Annotated with verdicts.
    """
    print("Fig 1: Model schematic (refined)")

    fig, ax = plt.subplots(figsize=(DOUBLE_COL, 4.2))
    ax.set_xlim(-0.1, 10.3)
    ax.set_ylim(-0.5, 5.7)
    ax.axis("off")

    # ── Colors ──
    c_central = CB_PALETTE[0]     # blue
    c_path_a  = CB_PALETTE[1]     # orange
    c_path_b  = CB_PALETTE[2]     # green
    c_path_c  = CB_PALETTE[3]     # red
    c_ledd    = CB_PALETTE[4]     # purple
    c_source  = "#D0D0D0"         # light gray
    c_pass    = CB_PALETTE[2]     # green
    c_fail    = CB_PALETTE[3]     # red

    # ── Phase 2 source ──
    p2_box = FancyBboxPatch(
        (0.2, 3.6), 2.6, 1.0,
        boxstyle="round,pad=0.15", facecolor=c_source,
        edgecolor="#888888", linewidth=1.0, alpha=0.7, zorder=2,
    )
    ax.add_patch(p2_box)
    ax.text(1.5, 4.1, "Phase 2 ODE\n" + r"$\dot{N} = -k_{death}\,\alpha(t)\,N$",
            ha="center", va="center", fontsize=FONT_SIZE - 1, style="italic",
            zorder=3)

    # ── Central node: N(t)/N0 ──
    nbox = FancyBboxPatch(
        (3.6, 2.1), 2.8, 1.3,
        boxstyle="round,pad=0.18", facecolor=c_central,
        edgecolor="black", linewidth=2.0, alpha=0.9, zorder=2,
    )
    ax.add_patch(nbox)
    ax.text(5.0, 2.75, r"$N(t)/N_0$" + "\nNeuron survival\nfraction",
            ha="center", va="center", fontsize=FONT_SIZE + 1,
            fontweight="bold", color="white", zorder=3)

    # ── LEDD input ──
    ledd_box = FancyBboxPatch(
        (0.2, 1.6), 2.2, 0.9,
        boxstyle="round,pad=0.12", facecolor=c_ledd,
        edgecolor="black", linewidth=1.2, alpha=0.75, zorder=2,
    )
    ax.add_patch(ledd_box)
    ax.text(1.3, 2.05, "LEDD\n(medication)",
            ha="center", va="center", fontsize=FONT_SIZE - 1,
            fontweight="bold", color="white", zorder=3)

    # ── Path A ──
    a_box = FancyBboxPatch(
        (7.4, 4.0), 2.4, 1.0,
        boxstyle="round,pad=0.12", facecolor=c_path_a,
        edgecolor="black", linewidth=1.2, alpha=0.75, zorder=2,
    )
    ax.add_patch(a_box)
    ax.text(8.6, 4.5, "Path A\nOFF-UPDRS",
            ha="center", va="center", fontsize=FONT_SIZE - 1,
            fontweight="bold", zorder=3)

    # ── Path B ──
    b_box = FancyBboxPatch(
        (7.4, 2.15), 2.4, 1.2,
        boxstyle="round,pad=0.12", facecolor=c_path_b,
        edgecolor="black", linewidth=1.5, alpha=0.8, zorder=2,
    )
    ax.add_patch(b_box)
    ax.text(8.6, 2.75, "Path B\nON-OFF Gap\n" + r"$\times$ LEDD",
            ha="center", va="center", fontsize=FONT_SIZE - 1,
            fontweight="bold", zorder=3)

    # ── Path C ──
    c_box = FancyBboxPatch(
        (7.4, 0.4), 2.4, 1.0,
        boxstyle="round,pad=0.12", facecolor=c_path_c,
        edgecolor="black", linewidth=1.2, alpha=0.75, zorder=2,
    )
    ax.add_patch(c_box)
    ax.text(8.6, 0.9, "Path C\nWearing-off",
            ha="center", va="center", fontsize=FONT_SIZE - 1,
            fontweight="bold", zorder=3)

    # ── Arrows ──
    arrow_kw = dict(
        arrowstyle="-|>", color="black", linewidth=1.8,
        mutation_scale=16, connectionstyle="arc3,rad=0",
    )

    # Phase 2 -> N(t)
    ax.annotate("", xy=(3.6, 3.0), xytext=(2.8, 3.9),
                arrowprops=dict(arrowstyle="-|>", color="black", linewidth=1.8,
                                mutation_scale=16, connectionstyle="arc3,rad=0.1"),
                zorder=4)

    # N(t) -> Path A
    ax.annotate("", xy=(7.4, 4.5), xytext=(6.4, 3.2),
                arrowprops=dict(arrowstyle="-|>", color="black", linewidth=1.8,
                                mutation_scale=16, connectionstyle="arc3,rad=-0.15"),
                zorder=4)
    # N(t) -> Path B
    ax.annotate("", xy=(7.4, 2.75), xytext=(6.4, 2.75),
                arrowprops=arrow_kw, zorder=4)
    # N(t) -> Path C
    ax.annotate("", xy=(7.4, 1.0), xytext=(6.4, 2.3),
                arrowprops=dict(arrowstyle="-|>", color="black", linewidth=1.8,
                                mutation_scale=16, connectionstyle="arc3,rad=0.15"),
                zorder=4)
    # LEDD -> Path B (dashed, colored)
    ax.annotate("", xy=(7.4, 2.5), xytext=(2.4, 2.05),
                arrowprops=dict(
                    arrowstyle="-|>", color=c_ledd, linewidth=1.8,
                    mutation_scale=16, connectionstyle="arc3,rad=-0.08",
                    linestyle="--",
                ), zorder=4)

    # ── Verdict annotations (right side) ──
    # Path A: informative negative
    ax.text(8.6, 5.15,
            r"$R^2_{\mathrm{cond}} = 0.800$; time wins ($\Delta$AIC = 803)",
            ha="center", fontsize=FONT_SIZE - 2, color=c_path_a,
            style="italic", zorder=5)
    verdict_a = ax.text(9.85, 4.5, "X", ha="center", va="center",
                        fontsize=FONT_SIZE + 4, fontweight="bold",
                        color=c_fail, zorder=5)

    # Path B: positive
    ax.text(8.6, 1.75,
            r"$\Delta$AIC = $-72$; $p_{\mathrm{interact}} = 0.011$",
            ha="center", fontsize=FONT_SIZE - 2, color="#1a6e1a",
            fontweight="bold", zorder=5)
    ax.text(9.85, 2.75, r"$\checkmark$", ha="center", va="center",
            fontsize=FONT_SIZE + 6, fontweight="bold",
            color=c_pass, zorder=5)

    # Path C: null
    ax.text(8.6, 0.05,
            r"$\rho = -0.050$, $p = 0.43$; log-rank $p = 0.64$",
            ha="center", fontsize=FONT_SIZE - 2, color=c_path_c,
            style="italic", zorder=5)
    ax.text(9.85, 0.9, "X", ha="center", va="center",
            fontsize=FONT_SIZE + 4, fontweight="bold",
            color=c_fail, zorder=5)

    save_fig(fig, "fig1_model_schematic")


# ======================================================================
# Figure 5: ON-OFF Gap vs N(t)/N0 by LEDD Quartile (KEY FIGURE)
# ======================================================================
def fig5_path_b_gap_scatter():
    """Scatter: N(t)/N0 on x-axis, ON-OFF gap on y-axis, colored by LEDD
    quartile with per-quartile regression lines.

    Double-column (7"), colorblind-safe, journal-ready.
    """
    print("Fig 5: Path B gap scatter (KEY FIGURE, refined)")

    merged = extract_paired_on_off_data()
    print(f"  Paired ON-OFF data: {len(merged)} rows, "
          f"{merged['PATNO'].nunique()} patients")

    # Compute LEDD quartile boundaries for legend
    ledd_bounds = merged.groupby("ledd_q", observed=True)["ledd_total"].agg(["min", "max"])

    fig, ax = plt.subplots(figsize=(DOUBLE_COL, 4.5))

    quartile_colors = [CB_PALETTE[0], CB_PALETTE[1], CB_PALETTE[2], CB_PALETTE[3]]
    quartile_labels = ["Q1 (low)", "Q2", "Q3", "Q4 (high)"]

    for i, q in enumerate(quartile_labels):
        sub = merged[merged["ledd_q"] == q]
        lo, hi = ledd_bounds.loc[q, "min"], ledd_bounds.loc[q, "max"]

        # Scatter points
        ax.scatter(
            sub["n_frac"], sub["gap"], s=14, alpha=0.22,
            color=quartile_colors[i], edgecolors="none", rasterized=True,
        )

        # Per-quartile regression line
        if len(sub) > 10:
            slope, intercept, r_val, p_val, se = stats.linregress(
                sub["n_frac"], sub["gap"]
            )
            x_line = np.linspace(sub["n_frac"].min(), sub["n_frac"].max(), 200)
            y_line = intercept + slope * x_line
            ax.plot(
                x_line, y_line, color=quartile_colors[i], linewidth=2.5,
                label=(
                    f"{q}: LEDD {lo:.0f}-{hi:.0f} mg "
                    f"(n={len(sub)}, slope={slope:.1f})"
                ),
            )

    # Zero reference line
    ax.axhline(0, color="gray", linewidth=0.6, linestyle="--", alpha=0.6)

    # Axis labels
    ax.set_xlabel(r"Neuron survival fraction $N(t)/N_0$", fontsize=FONT_SIZE)
    ax.set_ylabel("ON-OFF gap (UPDRS-III points)", fontsize=FONT_SIZE)

    # Legend
    legend = ax.legend(
        title="LEDD quartile",
        frameon=True, loc="upper left", fontsize=FONT_SIZE - 1.5,
        title_fontsize=FONT_SIZE - 1,
        borderaxespad=0.8, handlelength=1.8,
    )
    legend.get_frame().set_alpha(0.85)
    legend.get_frame().set_edgecolor("#cccccc")

    # Annotate interaction result in upper right
    res_b = load_json("phase4_path_b_results.json")
    mr = res_b["model_results"]
    r2_interact = mr["B3_interaction"]["R2"]
    beta_int = mr["B3_interaction"]["beta_interaction"]

    annotation_text = (
        f"Interaction model: $R^2 = {r2_interact:.3f}$\n"
        r"$\beta_{\mathrm{N \times LEDD}}$"
        f" = {beta_int:.2f}\n"
        r"$\Delta$AIC vs. null = $-160$"
    )
    ax.text(
        0.97, 0.97, annotation_text,
        transform=ax.transAxes, ha="right", va="top",
        fontsize=FONT_SIZE - 1.5,
        bbox=dict(
            boxstyle="round,pad=0.4", facecolor="white",
            edgecolor="#cccccc", alpha=0.9,
        ),
    )

    # Remove top/right spines for cleaner look
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    save_fig(fig, "fig5_path_b_gap_vs_nfrac")


# ======================================================================
# Figure 10: Three-Pathway Summary ("money figure")
# ======================================================================
def fig10_three_pathway_summary():
    """3-panel figure showing all three paths side by side.

    Left:   Path A -- R2 comparison (N(t) vs time), bar chart
    Center: Path B -- interaction effect visualization (key positive)
    Right:  Path C -- KM curves (null result)

    This is the figure that tells the entire story at a glance.
    """
    print("Fig 10: Three-pathway summary (refined)")

    res_a = load_json("phase4_path_a_results.json")
    res_b = load_json("phase4_path_b_results.json")
    res_c = load_json("phase4_path_c_results.json")

    fig = plt.figure(figsize=(DOUBLE_COL, 3.5))
    gs = gridspec.GridSpec(1, 3, width_ratios=[1, 1.35, 1.1], wspace=0.40)

    # ── Colors ──
    c_pass = CB_PALETTE[2]
    c_fail = CB_PALETTE[3]

    # ------------------------------------------------------------------
    # (a) Path A: R2 comparison (OLS: N(t) vs Time)
    # ------------------------------------------------------------------
    ax_a = fig.add_subplot(gs[0])

    r2_nfrac_ols = res_a["model_a1_ols"]["r2"]
    r2_time_ols = res_a["model_a4_time_only"]["r2"]
    r2_nfrac_lme = res_a["model_a2_mixed"]["conditional_r2"]
    r2_time_lme = res_a["model_a5_time_mixed"]["conditional_r2"]

    labels_a = [r"$N(t)/N_0$" + "\nOLS", "Time\nOLS",
                r"$N(t)/N_0$" + "\nLME", "Time\nLME"]
    vals_a = [r2_nfrac_ols, r2_time_ols, r2_nfrac_lme, r2_time_lme]
    colors_a = [CB_PALETTE[0], CB_PALETTE[1], CB_PALETTE[0], CB_PALETTE[1]]
    hatches_a = ["", "", "///", "///"]

    bars_a = ax_a.bar(range(4), vals_a, color=colors_a, edgecolor="black",
                      linewidth=0.6, width=0.7)
    for bar, h in zip(bars_a, hatches_a):
        bar.set_hatch(h)

    # Value labels
    for bar, val in zip(bars_a, vals_a):
        ax_a.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                  f"{val:.3f}", ha="center", fontsize=FONT_SIZE - 2.5,
                  fontweight="bold")

    # Highlight best (time LME)
    best_idx_a = np.argmax(vals_a)
    bars_a[best_idx_a].set_edgecolor(c_fail)
    bars_a[best_idx_a].set_linewidth(2.0)

    ax_a.set_xticks(range(4))
    ax_a.set_xticklabels(labels_a, fontsize=FONT_SIZE - 2.5)
    ax_a.set_ylabel("$R^2$", fontsize=FONT_SIZE)
    ax_a.set_ylim(0, max(vals_a) * 1.2)
    ax_a.spines["top"].set_visible(False)
    ax_a.spines["right"].set_visible(False)

    ax_a.text(0.50, 1.06, "(a) Path A: OFF-UPDRS",
              transform=ax_a.transAxes, fontweight="bold", va="bottom",
              ha="center", fontsize=FONT_SIZE - 1)
    # Verdict
    ax_a.text(0.5, 0.97, "informative negative",
              transform=ax_a.transAxes, ha="center", va="top",
              fontsize=FONT_SIZE - 2,
              style="italic", color=c_fail,
              bbox=dict(boxstyle="round,pad=0.2", facecolor="white",
                        edgecolor=c_fail, alpha=0.7, linewidth=0.8))

    # ------------------------------------------------------------------
    # (b) Path B: Interaction effect (grouped bar: gap by LEDD Q x N tertile)
    # ------------------------------------------------------------------
    ax_b = fig.add_subplot(gs[1])

    merged = extract_paired_on_off_data()
    merged["n_tertile"] = pd.qcut(
        merged["n_frac"], 3,
        labels=[r"Low $N/N_0$", "Medium", r"High $N/N_0$"]
    )
    grouped = (
        merged.groupby(["ledd_q", "n_tertile"], observed=True)["gap"]
        .agg(["mean", "sem"])
        .reset_index()
    )

    x_b = np.arange(4)
    width_b = 0.23
    tertile_names = [r"Low $N/N_0$", "Medium", r"High $N/N_0$"]
    tertile_colors = [CB_PALETTE[3], CB_PALETTE[7], CB_PALETTE[2]]

    for i, tert in enumerate(tertile_names):
        sub = grouped[grouped["n_tertile"] == tert]
        offset = (i - 1) * width_b
        ax_b.bar(
            x_b + offset, sub["mean"].values, width_b,
            yerr=sub["sem"].values,
            color=tertile_colors[i], edgecolor="black", linewidth=0.5,
            label=tert, capsize=2, error_kw={"linewidth": 0.8},
        )

    ax_b.set_xticks(x_b)
    ax_b.set_xticklabels(["Q1\n(low)", "Q2", "Q3", "Q4\n(high)"],
                          fontsize=FONT_SIZE - 2)
    ax_b.set_xlabel("LEDD quartile", fontsize=FONT_SIZE - 1)
    ax_b.set_ylabel("Mean ON-OFF gap (pts)", fontsize=FONT_SIZE - 1)
    ax_b.legend(
        title=r"$N(t)/N_0$ tertile", frameon=True,
        fontsize=FONT_SIZE - 3, title_fontsize=FONT_SIZE - 2.5,
        loc="upper left", borderaxespad=0.3,
        handlelength=1.2, handletextpad=0.4,
    )
    ax_b.spines["top"].set_visible(False)
    ax_b.spines["right"].set_visible(False)

    ax_b.text(0.50, 1.06, "(b) Path B: ON-OFF Gap",
              transform=ax_b.transAxes, fontweight="bold", va="bottom",
              ha="center", fontsize=FONT_SIZE - 1)
    ax_b.text(0.97, 0.97, r"$p_{\mathrm{int}} = 0.011$",
              transform=ax_b.transAxes, ha="right", va="top",
              fontsize=FONT_SIZE - 1.5,
              fontweight="bold", color="#1a6e1a",
              bbox=dict(boxstyle="round,pad=0.2", facecolor="white",
                        edgecolor=c_pass, alpha=0.7, linewidth=0.8))

    # ------------------------------------------------------------------
    # (c) Path C: KM curves (null result)
    # ------------------------------------------------------------------
    ax_c = fig.add_subplot(gs[2])

    # Reconstruct KM curves from assembled data
    df = load_assembled_data()
    df_post = df[df["has_posterior"] == 1].copy()

    pat_rate = df_post.groupby("PATNO")["pct_loss_per_yr_median"].first()
    tertile_edges = pat_rate.quantile([1 / 3, 2 / 3]).values

    def assign_tertile(r):
        if r <= tertile_edges[0]:
            return "Slow"
        elif r <= tertile_edges[1]:
            return "Medium"
        else:
            return "Fast"

    pat_tertile = pat_rate.apply(assign_tertile).rename("prog_group")
    df_post = df_post.merge(pat_tertile, left_on="PATNO", right_index=True, how="left")

    events = []
    for pat, grp in df_post.groupby("PATNO"):
        np4_rows = grp[grp["NP4OFF"].notna()]
        if len(np4_rows) == 0:
            continue
        prog = grp["prog_group"].iloc[0]
        onset = np4_rows[np4_rows["NP4OFF"] >= 1]
        if len(onset) > 0:
            t = onset["months_from_baseline"].min()
            events.append({"PATNO": pat, "time": max(t, 0.1), "event": 1,
                           "prog_group": prog})
        else:
            t = np4_rows["months_from_baseline"].max()
            events.append({"PATNO": pat, "time": max(t, 0.1), "event": 0,
                           "prog_group": prog})
    events_df = pd.DataFrame(events)

    km_colors = [CB_PALETTE[0], CB_PALETTE[1], CB_PALETTE[3]]
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
        ax_c.step(km_times, km_surv, where="post", color=km_colors[i],
                  linewidth=2, label=f"{group} (n={n})")

    ax_c.set_xlabel("Months", fontsize=FONT_SIZE - 1)
    ax_c.set_ylabel("Wearing-off-free\nsurvival", fontsize=FONT_SIZE - 1)
    ax_c.set_xlim(0, 150)
    ax_c.set_ylim(0, 1.05)
    ax_c.legend(
        title="N(t) rate", frameon=True, fontsize=FONT_SIZE - 2.5,
        title_fontsize=FONT_SIZE - 2, loc="lower left",
    )
    ax_c.spines["top"].set_visible(False)
    ax_c.spines["right"].set_visible(False)

    # Log-rank p-value
    primary_c = res_c["analyses"]["primary"]
    lr_p = primary_c["C1_km"]["logrank_fast_vs_slow"]["p_value"]
    ax_c.text(0.50, 1.06, "(c) Path C: Wearing-off",
              transform=ax_c.transAxes, fontweight="bold", va="bottom",
              ha="center", fontsize=FONT_SIZE - 1)
    ax_c.text(0.50, 0.97, f"log-rank $p = {lr_p:.2f}$",
              transform=ax_c.transAxes, ha="center", va="top",
              fontsize=FONT_SIZE - 1.5,
              style="italic", color=c_fail,
              bbox=dict(boxstyle="round,pad=0.2", facecolor="white",
                        edgecolor=c_fail, alpha=0.7, linewidth=0.8))

    fig.tight_layout()
    save_fig(fig, "fig10_three_pathway_summary")


# ======================================================================
# Main
# ======================================================================
def main():
    print("=" * 70)
    print("Phase 4 -- Refined publication figures (3 priority)")
    print(f"  Output dir: {FIG_DIR}")
    print("=" * 70)

    fig1_model_schematic()
    fig5_path_b_gap_scatter()
    fig10_three_pathway_summary()

    print("\n" + "=" * 70)
    print(f"Refined figures saved to {FIG_DIR}")
    for name in ["fig1_model_schematic", "fig5_path_b_gap_vs_nfrac",
                 "fig10_three_pathway_summary"]:
        for ext in ("png", "pdf"):
            p = FIG_DIR / f"{name}.{ext}"
            if p.exists():
                print(f"  {p.name} ({p.stat().st_size / 1024:.0f} KB)")
    print("=" * 70)


if __name__ == "__main__":
    main()
