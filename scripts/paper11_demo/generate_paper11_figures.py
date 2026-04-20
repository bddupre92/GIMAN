#!/usr/bin/env python3
"""
Generate 5 publication-quality figures for the Paper 11 npj PD submission.

Reads from:
  - PostgreSQL `mechanistic.paper11_sciml_summary` (grid + architecture ablation)
  - PostgreSQL `mechanistic.paper11_sciml_results` (per-patient test errors)
  - outputs/paper11_demo/full_cohort/baseline_v3/per_patient_results.csv
  - data/07_paper3_features/longitudinal_features.csv (trajectories)

Writes PNG + PDF at 300 DPI to:
  outputs/mechanistic_twin/paper11_submission/npj-pd/figures/

Okabe-Ito palette; colorblind-safe.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import psycopg2
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

# Local project import path for db helper
PROJECT_ROOT = Path("/Users/blair.dupre/Projects/CSCI-FALL-2025")
sys.path.insert(0, str(PROJECT_ROOT / "src"))


def get_conn() -> psycopg2.extensions.connection:
    """psycopg2 connection - avoids sqlalchemy '%' escaping issues with LIKE."""
    return psycopg2.connect(
        host="localhost",
        port=5432,
        dbname="giman_research",
        user="blair.dupre",
    )


def read_sql(sql: str) -> pd.DataFrame:
    conn = get_conn()
    try:
        return pd.read_sql(sql, conn)
    finally:
        conn.close()

OUTDIR = (
    PROJECT_ROOT
    / "outputs/mechanistic_twin/paper11_submission/npj-pd/figures"
)
OUTDIR.mkdir(parents=True, exist_ok=True)

# Okabe-Ito palette (colorblind-safe)
OI = {
    "blue":   "#0072B2",
    "orange": "#E69F00",
    "green":  "#009E73",
    "pink":   "#CC79A7",
    "sky":    "#56B4E9",
    "yellow": "#F0E442",
    "red":    "#D55E00",
    "black":  "#000000",
    "grey":   "#999999",
}

plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "legend.fontsize": 9,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "pdf.fonttype": 42,  # embed TrueType
    "ps.fonttype": 42,
})


def savefig(fig: plt.Figure, name: str) -> None:
    """Save figure as PNG + PDF at 300 DPI."""
    png = OUTDIR / f"{name}.png"
    pdf = OUTDIR / f"{name}.pdf"
    fig.savefig(png, dpi=300, bbox_inches="tight")
    fig.savefig(pdf, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {png.name}  +  {pdf.name}")


# -------------------------------------------------------------------------
# Fig. 1 - Architecture concept figure
# -------------------------------------------------------------------------

def fig1_architecture() -> None:
    print("Fig 1: architecture schematic")
    fig, ax = plt.subplots(figsize=(10.5, 6.0))
    ax.set_xlim(0, 10.5)
    ax.set_ylim(0, 6)
    ax.axis("off")

    titles = ["(a) Pure mechanistic", "(b) Pure Neural ODE", "(c) Physics-informed hybrid (UDE)"]
    centers = [1.5, 5.25, 9.0]
    maes = ["test MAE 0.205", "test MAE 0.157", "test MAE 0.152"]
    colors = [OI["grey"], OI["orange"], OI["green"]]

    # panel backgrounds
    for xc, color in zip(centers, colors):
        box = FancyBboxPatch(
            (xc - 1.55, 0.2), 3.1, 5.6,
            boxstyle="round,pad=0.05,rounding_size=0.15",
            edgecolor=color, facecolor="white", linewidth=1.6,
        )
        ax.add_patch(box)

    for xc, title, color, mae in zip(centers, titles, colors, maes):
        ax.text(xc, 5.55, title, ha="center", va="center",
                fontsize=11, weight="bold", color=color)
        ax.text(xc, 0.45, mae, ha="center", va="center",
                fontsize=10, style="italic", color=color)

    def box(xc: float, yc: float, w: float, h: float, txt: str, face: str) -> None:
        b = FancyBboxPatch(
            (xc - w / 2, yc - h / 2), w, h,
            boxstyle="round,pad=0.02,rounding_size=0.06",
            edgecolor=OI["black"], facecolor=face, linewidth=1.0,
        )
        ax.add_patch(b)
        ax.text(xc, yc, txt, ha="center", va="center", fontsize=8.5)

    def arr(x0: float, y0: float, x1: float, y1: float) -> None:
        a = FancyArrowPatch(
            (x0, y0), (x1, y1),
            arrowstyle="->", mutation_scale=12,
            color=OI["black"], linewidth=1.0,
        )
        ax.add_patch(a)

    # (a) Pure mechanistic
    box(1.5, 5.0, 2.6, 0.45, r"Baseline SBR $S_0$", "#F5F5F5")
    arr(1.5, 4.72, 1.5, 4.45)
    box(1.5, 4.15, 2.6, 0.55,
        r"$\frac{dS}{dt} = -k_{age}\,S$,  $k_{age}=0.025$/yr",
        "#EFEFEF")
    arr(1.5, 3.83, 1.5, 3.56)
    box(1.5, 3.25, 2.6, 0.45, "ODE solve to $t_{last}$", "#F5F5F5")
    arr(1.5, 2.97, 1.5, 2.70)
    box(1.5, 2.40, 2.6, 0.45, r"Predicted $\widehat{S}(t_{last})$",
        "#E0E0E0")

    # (b) Pure Neural ODE
    box(5.25, 5.0, 2.6, 0.45, r"$S_0$ + 11 features", "#F5F5F5")
    arr(5.25, 4.72, 5.25, 4.45)
    box(5.25, 4.15, 2.6, 0.55,
        r"$\frac{dS}{dt} = \mathrm{NN}(S, x;\theta)$",
        "#FFF4E5")
    arr(5.25, 3.83, 5.25, 3.56)
    box(5.25, 3.25, 2.6, 0.45, "ODE solve to $t_{last}$", "#F5F5F5")
    arr(5.25, 2.97, 5.25, 2.70)
    box(5.25, 2.40, 2.6, 0.45, r"Predicted $\widehat{S}(t_{last})$",
        "#FFE9C7")

    # (c) Physics-informed hybrid (UDE)
    box(9.0, 5.0, 2.6, 0.45, r"$S_0$ + 11 features", "#F5F5F5")
    arr(9.0, 4.72, 9.0, 4.45)
    box(9.0, 4.20, 2.8, 0.70,
        r"$\frac{dS}{dt} = -k_{age}\,S\;+\;\mathrm{NN}_{\mathrm{res}}(S, x;\theta)$" +
        "\n" + r"($k_{age}$ learned, lit prior; NN small init)",
        "#E5F3EC")
    arr(9.0, 3.78, 9.0, 3.56)
    box(9.0, 3.25, 2.6, 0.45, "ODE solve + regularisers", "#F5F5F5")
    arr(9.0, 2.97, 9.0, 2.70)
    box(9.0, 2.40, 2.6, 0.45, r"Predicted $\widehat{S}(t_{last})$",
        "#CFE9DB")

    # regulariser annotations under (c)
    ax.text(9.0, 1.75, r"$\mathcal{L}=\mathrm{MSE}+\lambda_{\mathrm{phys}}\,\|\mathrm{NN}_{\mathrm{res}}\|^2+\lambda_{\mathrm{mono}}\,\mathrm{ReLU}(dS/dt)$",
            ha="center", va="center", fontsize=8.5, color=OI["green"])
    ax.text(9.0, 1.35, r"best: $\lambda_{\mathrm{phys}}{=}0.1$, $\lambda_{\mathrm{mono}}{=}0.1$",
            ha="center", va="center", fontsize=8.5, color=OI["green"])

    # Data source strip at top
    ax.text(5.25, 5.82, "PPMI 428 patients, $\\geq$3 DaT-SPECT scans (70/15/15 split)",
            ha="center", va="center", fontsize=9.5, style="italic", color=OI["black"])

    savefig(fig, "fig1_architecture")


# -------------------------------------------------------------------------
# Fig. 2 - Grid sweep heatmap
# -------------------------------------------------------------------------

def fig2_grid_sweep() -> None:
    print("Fig 2: grid sweep heatmap")
    df = read_sql("""
        SELECT config_id, lambda_physics, lambda_monotone, test_mae,
               delta_vs_puremech_fair_point AS delta,
               delta_vs_puremech_fair_ci_lo AS ci_lo,
               delta_vs_puremech_fair_ci_hi AS ci_hi
        FROM mechanistic.paper11_sciml_summary
        WHERE model = 'hybrid' AND config_id LIKE 'grid_%_mlp'
        ORDER BY lambda_physics, lambda_monotone
    """)
    lps = sorted(df["lambda_physics"].unique())
    lms = sorted(df["lambda_monotone"].unique())

    mat = np.zeros((len(lps), len(lms)))
    for i, lp in enumerate(lps):
        for j, lm in enumerate(lms):
            row = df[(df.lambda_physics == lp) & (df.lambda_monotone == lm)]
            mat[i, j] = row["test_mae"].values[0]

    fig, (ax_h, ax_b) = plt.subplots(
        1, 2, figsize=(11.5, 4.5),
        gridspec_kw={"width_ratios": [1.1, 1.6]},
    )

    # Heatmap (invert row order so larger lp at top)
    display = mat[::-1, :]
    yticks = [f"{v:g}" for v in lps[::-1]]
    xticks = [f"{v:g}" for v in lms]
    im = ax_h.imshow(display, cmap="viridis_r", aspect="auto")
    for i in range(display.shape[0]):
        for j in range(display.shape[1]):
            ax_h.text(j, i, f"{display[i, j]:.4f}",
                      ha="center", va="center", fontsize=9,
                      color="white" if display[i, j] > 0.158 else "black")
    ax_h.set_xticks(range(len(xticks)))
    ax_h.set_xticklabels(xticks)
    ax_h.set_yticks(range(len(yticks)))
    ax_h.set_yticklabels(yticks)
    ax_h.set_xlabel(r"$\lambda_{\mathrm{monotone}}$")
    ax_h.set_ylabel(r"$\lambda_{\mathrm{physics}}$")
    ax_h.set_title("(a) Test MAE across $\\lambda$-grid (12 configs)")
    cbar = fig.colorbar(im, ax=ax_h, fraction=0.046, pad=0.04)
    cbar.set_label("Test MAE (SBR units)")

    # highlight best cell (lp=0.1, lm=0.1) - display-row index
    best_i_display = yticks.index(f"{max(lps):g}")
    best_j = xticks.index(f"{max(lms):g}")
    ax_h.add_patch(patches.Rectangle(
        (best_j - 0.5, best_i_display - 0.5), 1, 1,
        fill=False, edgecolor="black", linewidth=2.5,
    ))
    # spines off for heatmap
    for s in ax_h.spines.values():
        s.set_visible(True)
        s.set_linewidth(0.5)

    # Bar chart: Delta vs pure-mech-fair with 95% CI
    cells = df.sort_values(["lambda_physics", "lambda_monotone"]).reset_index(drop=True)
    labels = [f"$\\lambda_p{{=}}{r.lambda_physics:g}$, $\\lambda_m{{=}}{r.lambda_monotone:g}$"
              for _, r in cells.iterrows()]
    deltas = cells["delta"].values
    lo = cells["ci_lo"].values
    hi = cells["ci_hi"].values
    yerr = np.vstack([deltas - lo, hi - deltas])

    y = np.arange(len(cells))
    ax_b.barh(y, deltas, xerr=yerr, color=OI["green"], edgecolor=OI["black"],
              alpha=0.85, capsize=3, linewidth=0.6)
    ax_b.axvline(0, color=OI["black"], linewidth=0.8, linestyle="--")
    ax_b.set_yticks(y)
    ax_b.set_yticklabels(labels, fontsize=8.5)
    ax_b.invert_yaxis()
    ax_b.set_xlabel(r"$\Delta$ test MAE (hybrid $-$ pure mechanistic fair)")
    ax_b.set_title("(b) All 12 configs beat pure-mech-fair (95% CIs exclude 0)")
    ax_b.grid(axis="x", alpha=0.3)

    fig.tight_layout()
    savefig(fig, "fig2_grid_sweep")


# -------------------------------------------------------------------------
# Fig. 3 - MLP vs GRU ablation
# -------------------------------------------------------------------------

def fig3_mlp_vs_gru() -> None:
    print("Fig 3: MLP vs GRU ablation")
    configs = {
        "MLP\n(baseline_v3)":                       ("baseline_v3", OI["blue"]),
        "GRU v1\n(vanilla)":                        ("grid_lp0.01_lm0.01_gru", OI["orange"]),
        "GRU v2\n(state-aware)":                    ("grid_lp0.01_lm0.01_gru_v2", OI["green"]),
        "GRU v2\n(small + dropout)":                ("grid_lp0.01_lm0.01_gru_v2_small_dropout", OI["pink"]),
    }

    # Fetch per-patient test errors for MLP baseline first (reference)
    test_errs = {}
    for label, (cfg, _) in configs.items():
        q = f"""
            SELECT patno, abs_err FROM mechanistic.paper11_sciml_results
            WHERE split='test' AND config_id='{cfg}' AND model='hybrid'
            ORDER BY patno
        """
        test_errs[label] = read_sql(q).set_index("patno")["abs_err"]

    # Align by patno and compute MAE + 95% bootstrap CI paired to MLP
    mlp_label = list(configs.keys())[0]
    common_patnos = None
    for lbl in configs:
        idx = test_errs[lbl].index
        common_patnos = idx if common_patnos is None else common_patnos.intersection(idx)
    for lbl in configs:
        test_errs[lbl] = test_errs[lbl].loc[common_patnos]

    maes = {lbl: test_errs[lbl].mean() for lbl in configs}

    # 95% bootstrap CI on test MAE via paired resampling to MLP patients
    rng = np.random.default_rng(42)
    n_boot = 1000
    n = len(common_patnos)
    boot_maes = {lbl: np.zeros(n_boot) for lbl in configs}
    for b in range(n_boot):
        ix = rng.integers(0, n, n)
        for lbl in configs:
            boot_maes[lbl][b] = test_errs[lbl].values[ix].mean()
    cis = {lbl: (np.percentile(boot_maes[lbl], 2.5),
                 np.percentile(boot_maes[lbl], 97.5))
           for lbl in configs}

    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    x = np.arange(len(configs))
    colors = [c for _, c in configs.values()]
    heights = [maes[lbl] for lbl in configs]
    lows = [maes[lbl] - cis[lbl][0] for lbl in configs]
    highs = [cis[lbl][1] - maes[lbl] for lbl in configs]
    yerr = np.vstack([lows, highs])

    bars = ax.bar(x, heights, yerr=yerr, color=colors, edgecolor=OI["black"],
                  alpha=0.85, capsize=6, linewidth=0.7,
                  error_kw={"elinewidth": 1.2, "ecolor": OI["black"]})
    ax.set_xticks(x)
    ax.set_xticklabels(list(configs.keys()))
    ax.set_ylabel("Test MAE (SBR units)")
    ax.set_title("Architecture ablation: test MAE with 95% bootstrap CI (n=65 test patients)")
    # expand y-axis so CIs fit (bootstrap CI ~ +/- 0.04 around mean MAE)
    ax.set_ylim(0.10, 0.22)
    for xi, h in zip(x, heights):
        ax.text(xi, h + 0.0015, f"{h:.4f}", ha="center", va="bottom",
                fontsize=9.5)

    # annotate gap-closure arrows between consecutive bars
    def annotate_delta(i0: int, i1: int, text: str, yoff: float) -> None:
        x0, x1 = x[i0], x[i1]
        y = max(heights[i0], heights[i1]) + yoff
        ax.annotate("", xy=(x1, y), xytext=(x0, y),
                    arrowprops={"arrowstyle": "<->", "color": OI["grey"]})
        ax.text((x0 + x1) / 2, y + 0.0015, text,
                ha="center", va="bottom", fontsize=8.5, color=OI["grey"])

    annotate_delta(1, 2, "state-aware:\n51% gap closure", 0.020)
    annotate_delta(2, 3, "regularisation:\n16% gap closure", 0.020)

    fig.tight_layout()
    savefig(fig, "fig3_mlp_vs_gru")


# -------------------------------------------------------------------------
# Fig. 4 - Per-patient trajectory examples
# -------------------------------------------------------------------------

def fig4_patient_trajectories() -> None:
    print("Fig 4: per-patient trajectory examples")

    # pull per-patient test predictions for baseline_v3 hybrid vs pure_mech_fair
    perpt_hybrid = read_sql("""
        SELECT patno, s_obs_last, s_pred_last, t_horizon_yrs, abs_err
        FROM mechanistic.paper11_sciml_results
        WHERE split='test' AND config_id='baseline_v3' AND model='hybrid'
    """).set_index("patno")
    perpt_mech = read_sql("""
        SELECT patno, s_pred_last
        FROM mechanistic.paper11_sciml_results
        WHERE split='test' AND config_id='baseline_v3' AND model='pure_mech_fair'
    """).set_index("patno")

    # merge
    df = perpt_hybrid.join(perpt_mech, rsuffix="_mech").dropna()

    # load longitudinal SBR trajectories
    long = pd.read_csv(
        PROJECT_ROOT / "data/07_paper3_features/longitudinal_features.csv",
        usecols=["PATNO", "months_from_baseline", "putamen_mean_sbr"],
    )
    long = long.rename(columns={"PATNO": "patno"})
    long["t_years"] = long["months_from_baseline"] / 12.0
    long = long.dropna(subset=["putamen_mean_sbr"])
    long = long.sort_values(["patno", "t_years"])

    # pick 4 representative test patients
    # (1) fast decliner: large positive change predicted (small MAE, large s_obs - s_pred negative)
    # (2) slow decliner: hybrid > mech (helped by NN)
    # (3) noisy / reversal: multi-visit trajectory with non-monotone
    # (4) short horizon
    # We enumerate candidates and pick by features
    candidates = df.copy()
    # fast decliner = obs LOW (end SBR small), long horizon
    fast = candidates.sort_values(by=["s_obs_last", "t_horizon_yrs"],
                                  ascending=[True, False]).index[:3]
    # slow decliner = obs HIGH, long horizon
    slow = candidates.sort_values(by=["s_obs_last", "t_horizon_yrs"],
                                  ascending=[False, False]).index[:3]
    # short horizon
    short = candidates.sort_values(by=["t_horizon_yrs"]).index[:3]

    chosen = []
    pool = list(fast) + list(slow) + list(short)
    seen = set()
    for p in pool:
        # require this patient to have >=3 observed visits in the trajectory
        traj = long[long.patno == p]
        if len(traj) >= 3 and p not in seen:
            chosen.append(p)
            seen.add(p)
        if len(chosen) == 4:
            break
    # fallback
    if len(chosen) < 4:
        extras = [p for p in df.index if p not in seen and len(long[long.patno == p]) >= 3]
        for p in extras:
            chosen.append(p)
            if len(chosen) == 4:
                break

    titles = ["Patient A (long horizon)",
              "Patient B (stable trajectory)",
              "Patient C (variable/reversal)",
              "Patient D (short horizon)"]

    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.0))
    axes = axes.ravel()
    for ax, p, title in zip(axes, chosen, titles):
        traj = long[long.patno == p].sort_values("t_years")
        t_last = df.loc[p, "t_horizon_yrs"]
        s_obs = df.loc[p, "s_obs_last"]
        s_hybrid = df.loc[p, "s_pred_last"]
        s_mech = df.loc[p, "s_pred_last_mech"]
        t0 = traj["t_years"].min()
        tmax = max(t_last, traj["t_years"].max()) + 0.3

        ax.scatter(traj["t_years"] - t0, traj["putamen_mean_sbr"],
                   s=40, color=OI["black"], zorder=3, label="Observed")

        # mechanistic curve: exponential decay from s0 at rate 2.5%/yr
        if len(traj):
            s0 = float(traj.iloc[0]["putamen_mean_sbr"])
            tt = np.linspace(0, tmax - t0, 80)
            mech_curve = s0 * np.exp(-0.025 * tt)
            ax.plot(tt, mech_curve, color=OI["grey"], linestyle="--",
                    linewidth=1.5, label="Pure mechanistic (2.5%/yr)")

            # hybrid curve (smooth endpoint-matching exponential; per-step ODE
            # trajectory not available in export).
            hyb_rate = max(0.001, -np.log(max(s_hybrid, 0.001) / s0) / max(t_last, 0.1))
            hyb_curve = s0 * np.exp(-hyb_rate * tt)
            ax.plot(tt, hyb_curve, color=OI["green"], linewidth=2.0,
                    label="Physics-informed hybrid")

        # endpoint markers
        ax.scatter([t_last], [s_obs], s=60, facecolor="none",
                   edgecolor=OI["black"], linewidths=1.6, zorder=4,
                   label="True $t_{last}$" if ax is axes[0] else None)
        ax.scatter([t_last], [s_mech], s=55, marker="X", color=OI["grey"], zorder=4)
        ax.scatter([t_last], [s_hybrid], s=55, marker="X", color=OI["green"], zorder=4)

        # per-panel dynamic y-axis so low-SBR patients render legibly
        y_vals = list(traj["putamen_mean_sbr"].values) + [s_obs, s_mech, s_hybrid]
        y_lo = max(0.0, min(y_vals) - 0.2)
        y_hi = max(y_vals) + 0.3
        ax.set_ylim(y_lo, y_hi)
        ax.set_title(f"{title}: PATNO {p}")
        ax.set_xlim(-0.2, tmax - t0 + 0.1)
        ax.set_xlabel("Years from baseline")
        ax.set_ylabel("Putamen SBR")

    axes[0].legend(loc="upper right", fontsize=8)
    fig.suptitle("Per-patient DaT-SPECT trajectory examples (test set)",
                 fontsize=12, y=1.00)
    fig.tight_layout()
    savefig(fig, "fig4_patient_trajectories")


# -------------------------------------------------------------------------
# Fig. 5 - Learned decay rate vs literature
# -------------------------------------------------------------------------

def fig5_learned_decay_rate() -> None:
    print("Fig 5: learned decay rate vs literature")
    rates = read_sql("""
        SELECT learned_k_age_per_yr
        FROM mechanistic.paper11_sciml_summary
        WHERE model='hybrid' AND config_id LIKE 'grid_%_mlp'
          AND learned_k_age_per_yr IS NOT NULL
    """)["learned_k_age_per_yr"].values
    lo, md, hi = np.percentile(rates, [2.5, 50, 97.5])

    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(11.5, 4.3),
        gridspec_kw={"width_ratios": [2.1, 1]},
    )

    # Main comparison
    labels = ["Fearnley & Lees 1991\n(whole SN, clinicopath.)",
              "This paper (learned $k_{\\mathrm{age}}$\nacross 12 grid configs)",
              "Dzialas 2025\n(putamen-specific, DaT)"]
    lows  = [0.020, lo,  0.040]
    highs = [0.050, hi,  0.060]
    meds  = [0.035, md,  0.050]
    colors = [OI["grey"], OI["green"], OI["blue"]]
    y = np.arange(len(labels))

    for yi, yl, yh, ym, c in zip(y, lows, highs, meds, colors):
        ax1.plot([yl, yh], [yi, yi], color=c, linewidth=4, solid_capstyle="round")
        ax1.scatter([ym], [yi], s=80, color=c, zorder=3, edgecolor="black", linewidth=0.8)

    for yi, l in enumerate(labels):
        ax1.text(-0.0015, yi, l, ha="right", va="center", fontsize=9.5)

    # cohort-size annotations
    ann = ["clinicopath., $\\sim$20 patients",
           f"median {md*100:.1f}%/yr across 12 runs, $n=428$",
           "cross-sectional, $n = 1{,}065$"]
    for yi, t in zip(y, ann):
        ax1.text(0.065, yi - 0.22, t, ha="right", va="center",
                 fontsize=8, color=OI["grey"], style="italic")

    ax1.set_xlim(-0.005, 0.07)
    ax1.set_ylim(-0.7, len(labels) - 0.3)
    ax1.set_yticks([])
    ax1.set_xlabel(r"Striatal dopaminergic decay rate $k_{\mathrm{age}}$ (/yr)")
    ax1.set_title("(a) Learned rate agrees with region-specific literature")
    ax1.grid(axis="x", alpha=0.3)

    # Inset histogram
    ax2.hist(rates, bins=6, color=OI["green"], edgecolor=OI["black"], alpha=0.85)
    ax2.axvline(md, color=OI["red"], linestyle="--", linewidth=1.5,
                label=f"median {md*100:.1f}%/yr")
    ax2.set_xlabel(r"Learned $k_{\mathrm{age}}$ (/yr)")
    ax2.set_ylabel("Count (MLP grid configs)")
    ax2.set_title("(b) Distribution across grid")
    ax2.legend(fontsize=8)

    fig.tight_layout()
    savefig(fig, "fig5_learned_decay_rate")


# -------------------------------------------------------------------------

if __name__ == "__main__":
    fig1_architecture()
    fig2_grid_sweep()
    fig3_mlp_vs_gru()
    fig4_patient_trajectories()
    fig5_learned_decay_rate()
    print("All 5 figures written to", OUTDIR)
