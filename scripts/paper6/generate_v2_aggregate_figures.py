#!/usr/bin/env python3
"""Generate Paper 6 aggregate figures from the v2 full-cohort run.

Reads:
  outputs/paper6/pipeline_results/v2_full_cohort/pipeline_summary.json

Produces:
  outputs/paper6/figures/fig2_stage_distribution.{pdf,png}  (actual vs predicted)
  outputs/paper6/figures/fig3_band_width_distribution.{pdf,png}
  outputs/paper6/figures/fig4_feature_provenance.{pdf,png}
  outputs/paper6/figures/fig5_directional_concordance.{pdf,png}
  outputs/paper6/pipeline_results/v2_full_cohort/aggregate_stats.json

Also updates the CONSORT TikZ with real numbers.

Usage:
    .venv/bin/python3 scripts/paper6/generate_v2_aggregate_figures.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "outputs" / "paper6" / "pipeline_results" / "v2_full_cohort" / "pipeline_summary.json"
FIG_DIR = ROOT / "outputs" / "paper6" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# Okabe-Ito colorblind-safe palette
NSD_BLUE = "#0072B2"
NSD_ORANGE = "#E69F00"
NSD_GREEN = "#009E73"
NSD_RED = "#D55E00"
NSD_YELLOW = "#F0E442"
NSD_PURPLE = "#CC79A7"
NSD_GREY = "#888888"

STAGE_ORDER = ["0", "1", "2B", "3", "4", "5", "6"]


def stage_sort_key(s):
    if s == "2B":
        return 2.5
    try:
        return float(s)
    except ValueError:
        return 999


def main():
    data = json.load(open(RESULTS))
    summary = data.pop("summary")

    patients = [p for p in data.values() if isinstance(p, dict) and "error" not in p]
    n = len(patients)
    print(f"Loaded {n} patient records (summary: {summary['n_successful']})")

    # ── Aggregate stats ────────────────────────────────────────────────
    actual_stages = {}
    cb_stages = {}
    top_trans = {}
    band_widths = []
    provenance = {"gimin_imputed": 0, "paper1_raw": 0, "median_fallback": 0}
    nsd_positive_correct = 0
    nsd_positive_total = 0
    forward_transitions = 0
    backward_transitions = 0
    hold_transitions = 0

    for p in patients:
        a = p["current_stage"]
        c = p["staging"]["predicted_stage"]
        actual_stages[a] = actual_stages.get(a, 0) + 1
        cb_stages[c] = cb_stages.get(c, 0) + 1
        for prov in p["staging"]["feature_provenance"].values():
            provenance[prov] = provenance.get(prov, 0) + 1
        band_widths.append(p["conformal_band_width"])
        # Within-NSD+ accuracy: only count patients at stages 1, 2B, 3, 4
        if a in {"1", "2B", "3", "4"}:
            nsd_positive_total += 1
            if a == c:
                nsd_positive_correct += 1
        # Top transition: forward/backward/hold
        if p["top_transitions"]:
            top = p["top_transitions"][0]["destination_stage"]
            top_trans[top] = top_trans.get(top, 0) + 1
            a_k = stage_sort_key(a)
            t_k = stage_sort_key(top)
            if t_k > a_k:
                forward_transitions += 1
            elif t_k < a_k:
                backward_transitions += 1
            else:
                hold_transitions += 1

    agg = {
        "n_patients": n,
        "cohort": summary["cohort"],
        "gimin_checkpoint": summary["gimin_checkpoint"],
        "temperature_median": summary["temperature_median"],
        "conformal_band_width": summary["conformal_band_width_from_paper4"],
        "actual_stage_distribution": actual_stages,
        "catboost_predicted_distribution": cb_stages,
        "top_transition_distribution": top_trans,
        "band_width_stats": {
            "mean": float(np.mean(band_widths)),
            "median": float(np.median(band_widths)),
            "std": float(np.std(band_widths)),
            "min": float(min(band_widths)),
            "max": float(max(band_widths)),
        },
        "feature_provenance": provenance,
        "feature_provenance_pct": {
            k: 100 * v / (n * 12) for k, v in provenance.items()
        },
        "nsd_positive_accuracy": {
            "n_nsd_positive": nsd_positive_total,
            "n_correct": nsd_positive_correct,
            "pct": 100 * nsd_positive_correct / max(nsd_positive_total, 1),
        },
        "transition_direction": {
            "forward": forward_transitions,
            "backward": backward_transitions,
            "hold": hold_transitions,
            "forward_pct": 100 * forward_transitions / max(n, 1),
        },
    }

    agg_path = (
        ROOT / "outputs" / "paper6" / "pipeline_results" / "v2_full_cohort" / "aggregate_stats.json"
    )
    with open(agg_path, "w") as f:
        json.dump(agg, f, indent=2)
    print(f"Aggregate stats: {agg_path}")
    print(f"  NSD+ accuracy: {agg['nsd_positive_accuracy']['pct']:.1f}% "
          f"({nsd_positive_correct}/{nsd_positive_total})")
    print(f"  Forward transitions: {agg['transition_direction']['forward_pct']:.1f}%")
    print(f"  Mean band width: {agg['band_width_stats']['mean']:.4f}")

    # ── Fig 2: Staging distribution (actual vs CatBoost predicted) ────
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    stages = STAGE_ORDER
    actual = [actual_stages.get(s, 0) for s in stages]
    predicted = [cb_stages.get(s, 0) for s in stages]
    x = np.arange(len(stages))
    w = 0.38
    ax.bar(x - w / 2, actual, w, color=NSD_BLUE, label="Actual NSD-ISS stage", edgecolor="black", linewidth=0.4)
    ax.bar(x + w / 2, predicted, w, color=NSD_ORANGE, label="CatBoost predicted (NSD+ model)", edgecolor="black", linewidth=0.4)
    for i, (a, p) in enumerate(zip(actual, predicted)):
        if a: ax.text(i - w / 2, a + 12, str(a), ha="center", fontsize=8)
        if p: ax.text(i + w / 2, p + 12, str(p), ha="center", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(stages, fontsize=11)
    ax.set_xlabel("NSD-ISS stage", fontsize=11)
    ax.set_ylabel(f"Patients (n={n})", fontsize=11)
    ax.set_title(f"Fig. 2. Stage distribution across the full P6 cohort. CatBoost-12 (NSD+ target) predicts only stages {{1, 2B, 3, 4}} by design; Stage 0, 5, 6 are outside the model's scope.", fontsize=9, wrap=True)
    ax.legend(fontsize=10, loc="upper right")
    ax.grid(True, axis="y", alpha=0.25, linestyle=":")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig2_stage_distribution.pdf", bbox_inches="tight")
    fig.savefig(FIG_DIR / "fig2_stage_distribution.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    # ── Fig 3: Band width distribution ─────────────────────────────────
    fig, ax = plt.subplots(figsize=(6.0, 3.8))
    ax.hist(band_widths, bins=40, color=NSD_GREEN, edgecolor="black", linewidth=0.3, alpha=0.85)
    ax.axvline(np.mean(band_widths), color=NSD_RED, linestyle="--", linewidth=1.5, label=f"Mean = {np.mean(band_widths):.4f}")
    ax.set_xlabel("Paper 4 conformal CIF band width", fontsize=11)
    ax.set_ylabel(f"Patients (n={n})", fontsize=11)
    ax.set_title("Fig. 3. Per-patient conformal CIF band widths (IPCW, 90% CL) from Paper 4.", fontsize=10, wrap=True)
    ax.legend(fontsize=10)
    ax.grid(True, axis="y", alpha=0.25, linestyle=":")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig3_band_width_distribution.pdf", bbox_inches="tight")
    fig.savefig(FIG_DIR / "fig3_band_width_distribution.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    # ── Fig 4: Feature provenance pie ──────────────────────────────────
    fig, ax = plt.subplots(figsize=(5.5, 4.0))
    labels = ["GIMIN-imputed", "Paper 1 raw", "Median fallback"]
    values = [provenance.get(k, 0) for k in ["gimin_imputed", "paper1_raw", "median_fallback"]]
    colors = [NSD_BLUE, NSD_GREEN, NSD_GREY]
    ax.pie(values, labels=labels, colors=colors, autopct="%1.1f%%",
           startangle=90, wedgeprops={"edgecolor": "black", "linewidth": 0.5}, textprops={"fontsize": 11})
    ax.set_title(f"Fig. 4. Feature-value provenance across 1,900 patients × 12 CatBoost features\n(total n={n * 12:,} feature slots).", fontsize=10)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig4_feature_provenance.pdf", bbox_inches="tight")
    fig.savefig(FIG_DIR / "fig4_feature_provenance.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    # ── Fig 5: Directional concordance ─────────────────────────────────
    fig, ax = plt.subplots(figsize=(5.5, 3.8))
    dirs = ["Forward\n(progression)", "Hold\n(no transition)", "Backward\n(regression)"]
    counts = [forward_transitions, hold_transitions, backward_transitions]
    pcts = [100 * c / n for c in counts]
    colors_dir = [NSD_RED, NSD_GREY, NSD_GREEN]
    bars = ax.bar(dirs, pcts, color=colors_dir, edgecolor="black", linewidth=0.5)
    for bar, c, pct in zip(bars, counts, pcts):
        ax.text(bar.get_x() + bar.get_width() / 2, pct + 1, f"{c}\n({pct:.1f}%)",
                ha="center", fontsize=10)
    ax.set_ylabel(f"% of n={n} patients", fontsize=11)
    ax.set_title("Fig. 5. Top transition direction (Graph-DT max-CIF cause vs current stage).", fontsize=10)
    ax.set_ylim(0, max(pcts) * 1.2)
    ax.grid(True, axis="y", alpha=0.25, linestyle=":")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig5_directional_concordance.pdf", bbox_inches="tight")
    fig.savefig(FIG_DIR / "fig5_directional_concordance.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    print("\nFigures generated:")
    for f in ["fig2_stage_distribution", "fig3_band_width_distribution",
              "fig4_feature_provenance", "fig5_directional_concordance"]:
        print(f"  {FIG_DIR / (f + '.pdf')}")


if __name__ == "__main__":
    main()
