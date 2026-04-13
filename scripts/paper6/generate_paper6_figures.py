#!/usr/bin/env python3
"""Paper 6: Generate Publication Figures for Unified Pipeline.

Figures:
    1. Pipeline architecture overview (schematic)
    2. Per-patient 4-panel composite (one per patient):
       A: Stage trajectory over time
       B: CatBoost staging probabilities
       C: CIF curves (DeepHit + Graph-DT) with conformal bands
       D: Clinical summary dashboard
    3. Comparison summary across all patients

Usage:
    python scripts/paper6/generate_paper6_figures.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

OUTPUT_DIR = ROOT / "outputs" / "paper6" / "figures"
PIPELINE_DIR = ROOT / "outputs" / "paper6" / "pipeline_results"
SELECTION_PATH = ROOT / "outputs" / "paper6" / "patient_selection.json"

# Color scheme for NSD-ISS stages
STAGE_COLORS = {
    "0": "#2ecc71",  # Green
    "1": "#3498db",  # Blue
    "2B": "#f1c40f",  # Yellow
    "3": "#e67e22",  # Orange
    "4": "#e74c3c",  # Red
    "5": "#9b59b6",  # Purple
    "6": "#7f8c8d",  # Gray
}

STAGE_ORDER = ["0", "1", "2B", "3", "4", "5", "6"]
STAGE_Y = {s: i for i, s in enumerate(STAGE_ORDER)}

TIME_BIN_ENDS = [3, 6, 12, 18, 24, 36, 48, 60, 84, 120, 180]


def setup_style():
    """Set publication-quality matplotlib defaults."""
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 8,
            "figure.dpi": 300,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "font.family": "sans-serif",
        }
    )


def fig1_pipeline_architecture():
    """Generate pipeline architecture schematic."""
    fig, ax = plt.subplots(1, 1, figsize=(10, 4))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 4)
    ax.axis("off")

    # Pipeline boxes
    boxes = [
        (0.3, 1.5, 1.8, 1.0, "Patient\nVisit Data", "#ecf0f1"),
        (2.5, 1.5, 1.8, 1.0, "GIMIN\nImputation\n(Paper 2)", "#3498db"),
        (4.7, 2.5, 1.8, 1.0, "CatBoost\nStaging\n(Paper 1)", "#2ecc71"),
        (4.7, 0.5, 1.8, 1.0, "Graph-DT\nTransitions\n(Paper 3)", "#e67e22"),
        (7.2, 1.5, 2.2, 1.0, "Conformal\nUncertainty\n(Paper 4)", "#e74c3c"),
    ]

    for x, y, w, h, label, color in boxes:
        rect = mpatches.FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.1",
            facecolor=color,
            alpha=0.3,
            edgecolor=color,
            linewidth=2,
        )
        ax.add_patch(rect)
        ax.text(
            x + w / 2,
            y + h / 2,
            label,
            ha="center",
            va="center",
            fontsize=9,
            fontweight="bold",
        )

    # Arrows
    arrow_style = dict(arrowstyle="->,head_width=0.15", color="#2c3e50", lw=1.5)
    ax.annotate("", xy=(2.5, 2.0), xytext=(2.1, 2.0), arrowprops=arrow_style)
    ax.annotate("", xy=(4.7, 3.0), xytext=(4.3, 2.3), arrowprops=arrow_style)
    ax.annotate("", xy=(4.7, 1.0), xytext=(4.3, 1.7), arrowprops=arrow_style)
    ax.annotate("", xy=(7.2, 2.3), xytext=(6.5, 3.0), arrowprops=arrow_style)
    ax.annotate("", xy=(7.2, 1.7), xytext=(6.5, 1.0), arrowprops=arrow_style)

    ax.set_title(
        "Unified Clinical Decision Support Pipeline",
        fontsize=14,
        fontweight="bold",
        pad=15,
    )

    for fmt in ["png", "pdf"]:
        fig.savefig(OUTPUT_DIR / f"fig1_pipeline_architecture.{fmt}")
    plt.close(fig)
    print("  fig1_pipeline_architecture saved")


def fig_patient_composite(patno: int, result: dict, patient_idx: int):
    """Generate 4-panel composite figure for one patient."""
    fig = plt.figure(figsize=(14, 10))
    gs = GridSpec(2, 2, figure=fig, hspace=0.35, wspace=0.3)

    # ── Panel A: Stage trajectory ──
    ax_a = fig.add_subplot(gs[0, 0])
    stages = result["stage_trajectory"]
    times = result["visit_times_months"]

    for i, (t, s) in enumerate(zip(times, stages, strict=False)):
        color = STAGE_COLORS.get(str(s), "#95a5a6")
        y = STAGE_Y.get(str(s), 3)
        ax_a.scatter(t, y, c=color, s=40, zorder=3, edgecolors="black", linewidths=0.5)
        if i > 0:
            y_prev = STAGE_Y.get(str(stages[i - 1]), 3)
            ax_a.plot([times[i - 1], t], [y_prev, y], color="#bdc3c7", lw=1, alpha=0.7)

    ax_a.set_yticks(range(len(STAGE_ORDER)))
    ax_a.set_yticklabels([f"Stage {s}" for s in STAGE_ORDER])
    ax_a.set_xlabel("Months from Baseline")
    ax_a.set_title(f"(A) Stage Trajectory — Patient {patno}", fontweight="bold")
    ax_a.grid(axis="x", alpha=0.3)

    # Highlight current stage
    current = result["current_stage"]
    current_y = STAGE_Y.get(str(current), 3)
    ax_a.axhline(
        y=current_y,
        color=STAGE_COLORS.get(str(current), "gray"),
        linestyle="--",
        alpha=0.4,
        lw=1,
    )

    # ── Panel B: CatBoost staging probabilities ──
    ax_b = fig.add_subplot(gs[0, 1])
    staging = result["staging"]
    probs = staging["probabilities"]
    stage_labels = list(probs.keys())
    prob_values = list(probs.values())
    colors_b = [STAGE_COLORS.get(s, "#95a5a6") for s in stage_labels]

    bars = ax_b.barh(
        stage_labels,
        prob_values,
        color=colors_b,
        alpha=0.8,
        edgecolor="black",
        linewidth=0.5,
    )
    ax_b.set_xlabel("Probability")
    ax_b.set_xlim(0, 1)
    ax_b.set_title("(B) NSD-ISS Stage Prediction", fontweight="bold")

    # Annotate predicted and actual
    pred_stage = staging["predicted_stage"]
    actual_stage = staging["actual_stage"]
    ax_b.text(
        0.95,
        0.95,
        f"Predicted: Stage {pred_stage}\nActual: Stage {actual_stage}",
        transform=ax_b.transAxes,
        ha="right",
        va="top",
        fontsize=9,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
    )

    for bar, val in zip(bars, prob_values, strict=False):
        if val > 0.05:
            ax_b.text(
                bar.get_width() + 0.02,
                bar.get_y() + bar.get_height() / 2,
                f"{val:.2f}",
                va="center",
                fontsize=8,
            )

    # ── Panel C: CIF curves with conformal bands ──
    ax_c = fig.add_subplot(gs[1, 0])
    dh_cif = np.array(result["deephit_cif"])
    gdt_cif = np.array(result["graph_dt_cif"])
    dh_bands = np.array(result["deephit_cif_bands"])

    time_months = TIME_BIN_ENDS

    # Plot top 3 most likely transitions
    top_trans = result.get("top_transitions", [])[:3]
    cause_colors = ["#e74c3c", "#3498db", "#2ecc71", "#f1c40f", "#9b59b6"]

    for i, trans in enumerate(top_trans):
        k = trans["cause_idx"]
        label_dh = f"→Stage {trans['destination_stage']} (DeepHit)"
        label_gdt = f"→Stage {trans['destination_stage']} (Graph-DT)"
        color = cause_colors[i % len(cause_colors)]

        # DeepHit CIF (solid)
        ax_c.plot(time_months, dh_cif[k], color=color, lw=2, label=label_dh)

        # Graph-DT CIF (dashed)
        ax_c.plot(
            time_months,
            gdt_cif[k],
            color=color,
            lw=1.5,
            linestyle="--",
            label=label_gdt,
            alpha=0.7,
        )

        # Conformal band (fill)
        ax_c.fill_between(
            time_months, dh_bands[k, :, 0], dh_bands[k, :, 1], color=color, alpha=0.15
        )

    ax_c.set_xlabel("Time from Current Stage (months)")
    ax_c.set_ylabel("Cumulative Incidence")
    ax_c.set_title(
        "(C) Transition CIF Predictions (90% Conformal Bands)", fontweight="bold"
    )
    ax_c.legend(loc="upper left", fontsize=7, framealpha=0.8)
    ax_c.set_xlim(0, 180)
    ax_c.set_ylim(0, 1.05)
    ax_c.grid(alpha=0.3)

    # ── Panel D: Clinical summary ──
    ax_d = fig.add_subplot(gs[1, 1])
    ax_d.axis("off")

    summary_lines = [
        f"Patient {patno}",
        f"{'─' * 30}",
        f"Current Stage: {result['current_stage']}",
        f"Follow-up: {result['follow_up_months']:.0f} months ({result['n_visits']} visits)",
        "",
        f"CatBoost Predicted: Stage {staging['predicted_stage']}",
        f"   (Prob: {max(prob_values):.1%})",
        "",
    ]

    if top_trans:
        summary_lines.append("Most Likely Transitions:")
        for t in top_trans[:3]:
            summary_lines.append(
                f"  → Stage {t['destination_stage']}: "
                f"CIF@1yr={t['cif_at_12mo']:.3f}, "
                f"@3yr={t['cif_at_36mo']:.3f}"
            )

    # Missing features
    missing = result.get("missing_features", {})
    n_missing_feats = sum(1 for v in missing.values() if v["n_missing"] > 0)
    if n_missing_feats > 0:
        summary_lines.append(f"\nMissing Features: {n_missing_feats}/{len(missing)}")
        for feat, info in missing.items():
            if info["n_missing"] > 0:
                summary_lines.append(f"  {feat}: {info['pct']:.0f}% missing")

    text = "\n".join(summary_lines)
    ax_d.text(
        0.05,
        0.95,
        text,
        transform=ax_d.transAxes,
        va="top",
        fontsize=9,
        fontfamily="monospace",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="#ecf0f1", alpha=0.5),
    )
    ax_d.set_title("(D) Clinical Summary", fontweight="bold")

    fig.suptitle(
        f"Unified Pipeline — Patient {patno}", fontsize=14, fontweight="bold", y=1.01
    )

    for fmt in ["png", "pdf"]:
        fig.savefig(OUTPUT_DIR / f"fig_patient_{patno}_composite.{fmt}")
    plt.close(fig)
    print(f"  fig_patient_{patno}_composite saved")


def fig_summary_comparison(all_results: dict, selected_patnos: list[int]):
    """Generate summary comparison figure across all patients."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # ── Panel A: Stage trajectories side by side ──
    ax = axes[0]
    for i, patno in enumerate(selected_patnos):
        r = all_results.get(str(patno))
        if r is None or "error" in r:
            continue
        stages = r["stage_trajectory"]
        times = r["visit_times_months"]
        ys = [STAGE_Y.get(str(s), 3) + i * 0.08 for s in stages]
        ax.plot(
            times, ys, marker=".", markersize=3, lw=0.8, alpha=0.7, label=f"Pt {patno}"
        )

    ax.set_yticks(range(len(STAGE_ORDER)))
    ax.set_yticklabels([f"Stage {s}" for s in STAGE_ORDER])
    ax.set_xlabel("Months from Baseline")
    ax.set_title("(A) Stage Trajectories", fontweight="bold")
    ax.legend(fontsize=7, loc="upper left")
    ax.grid(alpha=0.3)

    # ── Panel B: CatBoost prediction accuracy ──
    ax = axes[1]
    pats = []
    predicted = []
    actual = []
    for patno in selected_patnos:
        r = all_results.get(str(patno))
        if r is None or "error" in r:
            continue
        pats.append(str(patno))
        predicted.append(r["staging"]["predicted_stage"])
        actual.append(r["staging"]["actual_stage"])

    x_pos = np.arange(len(pats))
    bar_width = 0.35

    pred_y = [STAGE_Y.get(str(s), 3) for s in predicted]
    act_y = [STAGE_Y.get(str(s), 3) for s in actual]

    bars1 = ax.bar(
        x_pos - bar_width / 2,
        pred_y,
        bar_width,
        label="Predicted",
        color="#3498db",
        alpha=0.7,
        edgecolor="black",
        linewidth=0.5,
    )
    bars2 = ax.bar(
        x_pos + bar_width / 2,
        act_y,
        bar_width,
        label="Actual",
        color="#e74c3c",
        alpha=0.7,
        edgecolor="black",
        linewidth=0.5,
    )

    ax.set_xticks(x_pos)
    ax.set_xticklabels([f"Pt\n{p}" for p in pats], fontsize=8)
    ax.set_yticks(range(len(STAGE_ORDER)))
    ax.set_yticklabels([f"Stage {s}" for s in STAGE_ORDER])
    ax.set_title("(B) Staging: Predicted vs Actual", fontweight="bold")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    # ── Panel C: Top transition CIF at 12 months ──
    ax = axes[2]
    for i, patno in enumerate(selected_patnos):
        r = all_results.get(str(patno))
        if r is None or "error" in r:
            continue
        top_trans = r.get("top_transitions", [])
        if not top_trans:
            continue

        top = top_trans[0]
        dh_cif = np.array(r["deephit_cif"])
        gdt_cif = np.array(r["graph_dt_cif"])
        k = top["cause_idx"]

        ax.plot(
            TIME_BIN_ENDS,
            dh_cif[k],
            lw=2,
            label=f"Pt {patno} →{top['destination_stage']}",
        )

    ax.set_xlabel("Time from Current Stage (months)")
    ax.set_ylabel("Cumulative Incidence")
    ax.set_title("(C) Top Predicted Transitions (DeepHit)", fontweight="bold")
    ax.legend(fontsize=7, loc="upper left")
    ax.set_xlim(0, 180)
    ax.set_ylim(0, 1.05)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    for fmt in ["png", "pdf"]:
        fig.savefig(OUTPUT_DIR / f"fig_summary_comparison.{fmt}")
    plt.close(fig)
    print("  fig_summary_comparison saved")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    setup_style()

    # Load patient selection
    with open(SELECTION_PATH) as f:
        selection = json.load(f)
    selected_patnos = selection["selected_patnos"]

    # Load pipeline results
    summary_path = PIPELINE_DIR / "pipeline_summary.json"
    with open(summary_path) as f:
        all_results = json.load(f)

    print("Generating Paper 6 figures...")
    print(f"  {len(selected_patnos)} patients")

    # Fig 1: Pipeline architecture
    fig1_pipeline_architecture()

    # Per-patient composite figures
    for i, patno in enumerate(selected_patnos):
        result = all_results.get(str(patno))
        if result is None or "error" in result:
            print(f"  Skipping patient {patno} (error)")
            continue
        fig_patient_composite(patno, result, i)

    # Summary comparison
    fig_summary_comparison(all_results, selected_patnos)

    # Count output files
    n_png = len(list(OUTPUT_DIR.glob("*.png")))
    n_pdf = len(list(OUTPUT_DIR.glob("*.pdf")))
    print(f"\nGenerated {n_png} PNG + {n_pdf} PDF figures in {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
