#!/usr/bin/env python3
"""Phase 5 Task 8 — 9 publication figures for Paper 10.

Reads Task 1-7 committed JSON artifacts; produces 9 figures at 300 DPI (PNG + PDF)
following Paper 9 conventions (no titles embedded in figure, colorblind palette,
IEEE/Nature body text size).

Fig 1: Architecture diagram (PPMI -> IS calibration -> PosteriorStore -> SIR updater -> counterfactual -> validation)
Fig 2: NASEM criteria radar chart (7 axes, 0-3 scale)
Fig 3: Bidirectional MAE vs update count (THE TWIN PROOF)
Fig 4: External validation LCC HC-vs-PD SBR distributions
Fig 5: Head-to-head paired-bootstrap C-index comparison
Fig 6: Observational counterfactual predicted vs observed delta_gap scatter
Fig 7: Patient case studies (fast/medium/slow progressors from posteriors)
Fig 8: Calibration plot (counterfactual slope/intercept with 45-deg reference)
Fig 9: Dissertation arc (Papers 1-10 positioning)
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
FIG_DIR = PROJECT_ROOT / "outputs/mechanistic_twin/paper10_mech_vs_giman/figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams.update(
    {
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "legend.fontsize": 9,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)

# Colorblind-safe palette (Okabe-Ito)
COLORS = {
    "mech": "#0072B2",
    "gdt": "#D55E00",
    "hc": "#009E73",
    "pd": "#CC79A7",
    "accent": "#E69F00",
    "neutral": "#999999",
}


def _save(fig, name: str) -> None:
    for ext in ("png", "pdf"):
        fig.savefig(FIG_DIR / f"{name}.{ext}", bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {name}.png + .pdf")


def fig1_architecture() -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 6)
    ax.axis("off")

    boxes = [
        (0.3, 2.5, 2.2, 1.0, "PPMI\n(1,065 pts)", "#E8F4F8"),
        (3.0, 2.5, 2.2, 1.0, "Phase 2 IS\ncalibration\n(k_n, α_tox, T_tox)", "#FFE8CC"),
        (5.7, 4.0, 2.6, 1.0, "PosteriorStore\n(HDF5, versioned)", "#E8FFEA"),
        (5.7, 1.0, 2.6, 1.0, "Forward model\n(closed-form ODE)", "#E8FFEA"),
        (8.8, 4.0, 2.6, 1.0, "SIR updater\n(+rejuvenation)", "#FCE8F3"),
        (8.8, 1.0, 2.6, 1.0, "Counterfactual\n(LEDD ↑200mg)", "#FCE8F3"),
    ]
    for x, y, w, h, label, c in boxes:
        ax.add_patch(
            mpatches.FancyBboxPatch(
                (x, y), w, h, boxstyle="round,pad=0.05", fc=c, ec="black", lw=1
            )
        )
        ax.text(x + w / 2, y + h / 2, label, ha="center", va="center", fontsize=9.5)

    arrows = [
        ((2.5, 3.0), (3.0, 3.0)),
        ((5.2, 3.0), (5.7, 4.5)),
        ((5.2, 3.0), (5.7, 1.5)),
        ((7.0, 4.2), (8.8, 4.5)),  # forward pred to counterfactual? no, posterior to updater
        ((8.3, 4.5), (8.8, 4.5)),
        ((8.3, 1.5), (8.8, 1.5)),
        ((10.1, 3.8), (10.1, 2.2)),  # updater -> counterfactual loop
    ]
    for (x0, y0), (x1, y1) in arrows:
        ax.annotate(
            "",
            xy=(x1, y1),
            xytext=(x0, y0),
            arrowprops=dict(arrowstyle="->", color="black", lw=1.2),
        )

    ax.text(
        0.3,
        5.5,
        "A. Bidirectional-Ready Mechanistic Patient-Specific Model Architecture",
        fontsize=11,
        fontweight="bold",
    )
    ax.text(
        6.0,
        0.3,
        "Validation: external cohort (LCC) | head-to-head (GIMAN) | NASEM audit",
        fontsize=8.5,
        style="italic",
    )
    _save(fig, "fig1_architecture")


def fig2_nasem_radar() -> None:
    with open(
        PROJECT_ROOT
        / "outputs/mechanistic_twin/paper10_mech_vs_giman/nasem_audit.json"
    ) as f:
        data = json.load(f)
    crit = data["criteria"]
    labels = [
        "Virtual\nRepresentation",
        "Bidirectional\nFlow",
        "Predictive\nCapability",
        "Uncertainty\nQuantification",
        "Validation",
        "Fitness for\nPurpose",
        "Governance",
    ]
    keys = [
        "virtual_representation",
        "bidirectional_flow",
        "predictive_capability",
        "uncertainty_quantification",
        "validation",
        "fitness_for_purpose",
        "governance",
    ]
    scores = [crit[k]["score"] for k in keys]
    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    scores_closed = scores + scores[:1]
    angles_closed = angles + angles[:1]

    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))
    ax.fill(angles_closed, scores_closed, color=COLORS["mech"], alpha=0.25)
    ax.plot(angles_closed, scores_closed, color=COLORS["mech"], lw=2, marker="o", ms=7)
    ax.plot(
        angles_closed,
        [3] * len(angles_closed),
        color=COLORS["neutral"],
        lw=1,
        ls="--",
        label="Complete (3)",
    )
    ax.set_xticks(angles)
    ax.set_xticklabels(labels, fontsize=9.5)
    ax.set_yticks([0, 1, 2, 3])
    ax.set_yticklabels(["0", "1", "2", "3"], fontsize=8)
    ax.set_ylim(0, 3.2)
    ax.set_rlabel_position(90)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right", bbox_to_anchor=(1.2, -0.05))
    total = data["aggregate"]["total_score"]
    ax.text(
        0, -0.08, f"Total: {total}/21 ({data['aggregate']['compliance_pct']}%)",
        transform=ax.transAxes, fontsize=10, fontweight="bold",
    )
    _save(fig, "fig2_nasem_radar")


def fig3_bidirectional_mae() -> None:
    with open(
        PROJECT_ROOT
        / "outputs/mechanistic_twin/paper10_mech_vs_giman/bidirectional_demo.json"
    ) as f:
        data = json.load(f)
    rows = pd.DataFrame(data["per_scan_count_summary"])
    # Skip scans_used=1 (tautological anchor)
    informative = rows[rows["scans_used"] != 1].copy()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))

    ax1.plot(
        informative["scans_used"],
        informative["mae_from_mean"],
        marker="o",
        color=COLORS["mech"],
        lw=2,
        label="Posterior mean (MSE-optimal)",
    )
    ax1.plot(
        informative["scans_used"],
        informative["mae_from_median"],
        marker="s",
        color=COLORS["gdt"],
        lw=2,
        label="Posterior median (theory-MAE-optimal)",
    )
    ax1.set_xlabel("Scans used for posterior update")
    ax1.set_ylabel("MAE on held-out last DaT-SPECT scan")
    ax1.set_xticks([0, 2, 3, 4, 5])
    ax1.legend()
    ax1.grid(alpha=0.3)
    ax1.text(
        0.02, 0.97, "A", transform=ax1.transAxes, fontsize=14, fontweight="bold", va="top"
    )

    ax2.plot(
        informative["scans_used"],
        informative["coverage"],
        marker="D",
        color=COLORS["hc"],
        lw=2,
    )
    ax2.axhline(0.9, color=COLORS["neutral"], ls="--", lw=1, label="90% target")
    ax2.set_xlabel("Scans used for posterior update")
    ax2.set_ylabel("90% credible interval coverage")
    ax2.set_xticks([0, 2, 3, 4, 5])
    ax2.set_ylim(0.5, 1.0)
    ax2.legend()
    ax2.grid(alpha=0.3)
    ax2.text(
        0.02, 0.97, "B", transform=ax2.transAxes, fontsize=14, fontweight="bold", va="top"
    )
    _save(fig, "fig3_bidirectional_mae")


def fig4_external_lcc() -> None:
    path = (
        PROJECT_ROOT
        / "outputs/mechanistic_twin/paper10_mech_vs_giman/external_validation_lcc.json"
    )
    with open(path) as f:
        data = json.load(f)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    # Show group means + error bars (we have summary stats, not full distributions)
    groups = []
    means = []
    stds = []
    cols = []
    labels = []
    for key, color, lab in [
        ("ppmi_hc", COLORS["hc"], "PPMI HC"),
        ("lcc_hc", COLORS["hc"], "LCC HC"),
        ("ppmi_pd", COLORS["pd"], "PPMI PD"),
    ]:
        if key in data:
            d = data[key]
            if "mean_sbr" in d and "sd_sbr" in d:
                groups.append(key)
                means.append(d["mean_sbr"])
                stds.append(d["sd_sbr"])
                cols.append(color)
                labels.append(f"{lab}\n(n={d.get('n', '?')})")
    x = np.arange(len(groups))
    ax.bar(x, means, yerr=stds, color=cols, capsize=6, edgecolor="black", alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("DaT-SPECT putamen SBR")
    ax.grid(axis="y", alpha=0.3)
    if "hc_vs_pd_mean_pct_gap" in data:
        ax.text(
            0.5,
            0.95,
            f"HC-vs-PD mean gap: {data['hc_vs_pd_mean_pct_gap']:.1f}%",
            transform=ax.transAxes,
            ha="center",
            fontsize=10,
            bbox=dict(facecolor="white", ec="black", alpha=0.8),
        )
    _save(fig, "fig4_external_lcc")


def fig5_headtohead() -> None:
    with open(
        PROJECT_ROOT
        / "outputs/mechanistic_twin/paper10_mech_vs_giman/headtohead_wearing_off.json"
    ) as f:
        data = json.load(f)
    comp = data["cindex_comparison"]
    fig, ax = plt.subplots(figsize=(7, 4.2))
    models = ["Mechanistic\n(pct_loss_per_yr)", "Graph-DT\n(5yr total CIF)"]
    vals = [comp["ci_a"], comp["ci_b"]]
    ci_a = comp["ci_a_95"]
    ci_b = comp["ci_b_95"]
    err_lo = [vals[0] - ci_a[0], vals[1] - ci_b[0]]
    err_hi = [ci_a[1] - vals[0], ci_b[1] - vals[1]]
    ax.bar(
        models,
        vals,
        yerr=[err_lo, err_hi],
        color=[COLORS["mech"], COLORS["gdt"]],
        capsize=8,
        edgecolor="black",
        alpha=0.85,
    )
    ax.axhline(0.5, color=COLORS["neutral"], ls="--", lw=1, label="Random (0.5)")
    ax.set_ylabel("Time-to-wearing-off C-index")
    ax.set_ylim(0.4, 0.6)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    p = comp.get("p_value")
    delta = comp.get("delta_ci", comp.get("delta"))
    note = f"Δ = {delta:+.3f}" if delta is not None else ""
    if p is not None:
        note += f", paired bootstrap p = {p:.3f}"
    note += f"   (n={data['analysis_set_size']} pts, {data['events']} events)"
    ax.text(
        0.5, 0.02, note, transform=ax.transAxes, ha="center", fontsize=9,
        bbox=dict(facecolor="white", ec="black", alpha=0.8),
    )
    _save(fig, "fig5_headtohead_cindex")


def fig6_counterfactual() -> None:
    """Replay Task 6 extraction to scatter predicted vs observed delta_gap."""
    import sys
    sys.path.insert(0, str(PROJECT_ROOT / "scripts/mechanistic_twin"))
    from phase5_observational_counterfactual import (
        COEFS,
        LEDD_ESCALATION_THRESHOLD_MG,
        centering_means,
        extract_escalation_events,
        load_data,
        predict_delta_gap,
    )
    df = load_data()
    means = centering_means(df)
    events = extract_escalation_events(df, LEDD_ESCALATION_THRESHOLD_MG)
    ev = predict_delta_gap(events, means, COEFS)

    with open(
        PROJECT_ROOT
        / "outputs/mechanistic_twin/paper10_mech_vs_giman/observational_counterfactual.json"
    ) as f:
        summary = json.load(f)
    cal = summary["calibration_overall"]

    fig, ax = plt.subplots(figsize=(6.5, 6))
    ax.scatter(
        ev["pred_delta_gap"],
        ev["delta_gap_observed"],
        s=16,
        alpha=0.5,
        color=COLORS["mech"],
        edgecolor="none",
    )
    xmin, xmax = ev["pred_delta_gap"].min(), ev["pred_delta_gap"].max()
    grid = np.linspace(xmin, xmax, 100)
    ax.plot(grid, grid, ls="--", color=COLORS["neutral"], lw=1, label="Identity (slope=1)")
    slope = cal["slope"]
    intercept = cal["intercept"]
    ax.plot(
        grid,
        slope * grid + intercept,
        color=COLORS["accent"],
        lw=2,
        label=f"Fit: slope={slope:.3f} [{cal['slope_ci95'][0]:.2f}, {cal['slope_ci95'][1]:.2f}]",
    )
    ax.set_xlabel("Predicted Δgap (Phase 4 Path B model)")
    ax.set_ylabel("Observed Δgap (UPDRS-III, OFF − ON)")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)
    ax.text(
        0.03, 0.95,
        f"n={cal['n_events']} LEDD escalations ≥200 mg\nR²={cal['r2']:.2f}  MAE={cal['mae']:.2f}",
        transform=ax.transAxes, va="top", fontsize=9,
        bbox=dict(facecolor="white", ec="black", alpha=0.8),
    )
    _save(fig, "fig6_counterfactual_scatter")


def fig7_patient_cases() -> None:
    """Fast/medium/slow progressor trajectories from phase2_combined_1065.csv."""
    import sys
    sys.path.insert(0, str(PROJECT_ROOT / "src"))
    from giman_pipeline.mechanistic_twin_v2.forward_model import predict_sbr

    post = pd.read_csv(
        PROJECT_ROOT / "outputs/mechanistic_twin/data/posteriors/phase2_combined_1065.csv"
    )
    sbr_long = pd.read_parquet(
        PROJECT_ROOT / "outputs/mechanistic_twin/data/dat_spect_longitudinal.parquet"
    )
    # Pick patients with >=4 scans and clean quantiles of pct_loss
    merged = post.merge(
        sbr_long.groupby("PATNO").size().rename("n_visits_long"),
        on="PATNO",
        how="left",
    )
    eligible = merged[(merged["n_scans"] >= 4) & merged["pct_loss_per_yr_median"].notna()]
    eligible = eligible.sort_values("pct_loss_per_yr_median")
    slow = eligible.iloc[len(eligible) // 10]  # 10th percentile
    med = eligible.iloc[len(eligible) // 2]
    fast = eligible.iloc[int(len(eligible) * 0.9)]

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2), sharey=True)
    for ax, row, label, color in zip(
        axes,
        [slow, med, fast],
        ["Slow (10th %ile)", "Median (50th %ile)", "Fast (90th %ile)"],
        [COLORS["hc"], COLORS["accent"], COLORS["gdt"]],
    ):
        patno = int(row["PATNO"])
        obs = sbr_long[sbr_long["PATNO"] == patno].sort_values("t_years")
        if obs.empty:
            continue
        sbr_0 = float(obs["sbr_putamen_mean"].iloc[0])
        ax.scatter(
            obs["t_years"], obs["sbr_putamen_mean"], s=35, color=color, zorder=3, label="Observed"
        )
        t_grid = np.linspace(0, obs["t_years"].max() * 1.1, 80)
        # Draw posterior mean/quantile band using patient's pct_loss_per_yr quantiles
        median_r = row["pct_loss_per_yr_median"]
        lo_r = row["pct_loss_per_yr_q025"]
        hi_r = row["pct_loss_per_yr_q975"]
        for r, alpha, ls in [(median_r, 1.0, "-"), (lo_r, 0.4, "--"), (hi_r, 0.4, "--")]:
            ax.plot(
                t_grid,
                sbr_0 * (1 - r / 100) ** t_grid,
                color=color,
                alpha=alpha,
                lw=1.5 if ls == "-" else 1,
                ls=ls,
            )
        ax.set_title(f"{label}\nPATNO {patno}, {median_r:.2f}%/yr")
        ax.set_xlabel("Years from baseline")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("Putamen SBR")
    axes[0].legend(loc="upper right")
    _save(fig, "fig7_patient_cases")


def fig8_calibration() -> None:
    """Posterior predictive calibration: binned observed mean vs predicted mean."""
    import sys
    sys.path.insert(0, str(PROJECT_ROOT / "scripts/mechanistic_twin"))
    from phase5_observational_counterfactual import (
        COEFS,
        LEDD_ESCALATION_THRESHOLD_MG,
        centering_means,
        extract_escalation_events,
        load_data,
        predict_delta_gap,
    )
    df = load_data()
    means = centering_means(df)
    events = extract_escalation_events(df, LEDD_ESCALATION_THRESHOLD_MG)
    ev = predict_delta_gap(events, means, COEFS)
    bins = np.quantile(ev["pred_delta_gap"], np.linspace(0, 1, 11))
    bin_ids = np.clip(np.searchsorted(bins, ev["pred_delta_gap"]) - 1, 0, 9)
    ev = ev.assign(bin=bin_ids)
    g = ev.groupby("bin").agg(
        pred_mean=("pred_delta_gap", "mean"),
        obs_mean=("delta_gap_observed", "mean"),
        obs_se=("delta_gap_observed", lambda s: s.std() / np.sqrt(len(s))),
        n=("delta_gap_observed", "size"),
    ).reset_index()

    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    ax.errorbar(
        g["pred_mean"],
        g["obs_mean"],
        yerr=g["obs_se"],
        fmt="o",
        color=COLORS["mech"],
        ecolor=COLORS["mech"],
        ms=8,
        capsize=4,
        lw=1.5,
    )
    lims = [min(g["pred_mean"].min(), g["obs_mean"].min()) - 0.5,
            max(g["pred_mean"].max(), g["obs_mean"].max()) + 0.5]
    ax.plot(lims, lims, ls="--", color=COLORS["neutral"], label="Perfect calibration")
    ax.set_xlabel("Mean predicted Δgap (decile bin)")
    ax.set_ylabel("Mean observed Δgap (decile bin)")
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.legend()
    ax.grid(alpha=0.3)
    _save(fig, "fig8_calibration_bins")


def fig9_dissertation_arc() -> None:
    fig, ax = plt.subplots(figsize=(12, 4.5))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 4)
    ax.axis("off")
    papers = [
        (0.2, 2, 1.5, 1, "P1\nStage Rx", "#E8F4F8", False),
        (1.9, 2, 1.5, 1, "P2\nGIMIN", "#E8F4F8", False),
        (3.6, 2, 1.5, 1, "P3\nGraph-DT", "#E8F4F8", False),
        (5.3, 2, 1.5, 1, "P4\nConformal", "#E8F4F8", False),
        (7.0, 2, 1.5, 1, "P5\nTemporal", "#E8F4F8", False),
        (8.7, 2, 1.5, 1, "P6\nPipeline", "#E8F4F8", False),
        (0.2, 0.4, 1.5, 1, "P7\nPhase 1-2", "#FFE8CC", True),
        (1.9, 0.4, 1.5, 1, "P8a/b\nRegional", "#FFE8CC", True),
        (3.6, 0.4, 1.5, 1, "P9\nPK-PD", "#FFE8CC", True),
        (5.3, 0.4, 2.6, 1, "P10\nBidirectional\n+ NASEM audit", "#FCE8F3", True),
        (8.1, 0.4, 2.1, 1, "P11 (future)\nHybrid SciML", "#EEEEEE", False),
    ]
    for x, y, w, h, label, c, current in papers:
        ax.add_patch(
            mpatches.FancyBboxPatch(
                (x, y), w, h, boxstyle="round,pad=0.05", fc=c,
                ec="black" if current else "#777777",
                lw=2 if current else 1,
            )
        )
        ax.text(x + w / 2, y + h / 2, label, ha="center", va="center", fontsize=9)

    ax.text(0.2, 3.3, "Data-driven papers (GNN, conformal, pipeline)", fontsize=10)
    ax.text(0.2, 1.6, "Mechanistic papers (ODE, PK-PD, digital twin)", fontsize=10)
    ax.annotate(
        "", xy=(5.3, 1.4), xytext=(9, 1.9),
        arrowprops=dict(arrowstyle="->", color=COLORS["mech"], lw=1.5, ls="--"),
    )
    ax.text(10.3, 3.3, "← complementarity", fontsize=9, color=COLORS["mech"])
    _save(fig, "fig9_dissertation_arc")


def main() -> None:
    print("[Task 8] Generating 9 publication figures...")
    for fn in [
        fig1_architecture,
        fig2_nasem_radar,
        fig3_bidirectional_mae,
        fig4_external_lcc,
        fig5_headtohead,
        fig6_counterfactual,
        fig7_patient_cases,
        fig8_calibration,
        fig9_dissertation_arc,
    ]:
        try:
            fn()
        except Exception as exc:
            print(f"  FAILED {fn.__name__}: {exc}")
    print(f"[Task 8] Figures in {FIG_DIR}/")


if __name__ == "__main__":
    main()
