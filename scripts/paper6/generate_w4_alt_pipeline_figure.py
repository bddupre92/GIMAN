#!/usr/bin/env python3
"""W4 alt-pipeline comparison figure — 2 panel.

Panel (a): Top-1 accuracy across 6 configurations with 95% CI error bars.
           Groups: CatBoost-12 (baseline, alt2 cost-sensitive, alt3 binary,
                   alt4 Mean) vs CatBoost-33 (alt1 GIMIN, alt4b Mean).
           Visualizes the "feature access >> architecture" finding.

Panel (b): GIMIN vs Mean imputation pairwise comparison, both schemas.
           Bar chart: 12-feat (GIMIN, Mean) side-by-side + 33-feat (GIMIN, Mean)
           side-by-side with CIs. Visualizes the "Mean ≈ GIMIN" finding.

Outputs:
  outputs/paper6/pipeline_results/fig_w4_alt_pipeline.pdf
  outputs/paper6/pipeline_results/fig_w4_alt_pipeline.png
  outputs/mechanistic_twin/paper6_submission/jamia/figures/fig_w4_alt_pipeline.{pdf,png}
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
W4_DIR = PROJECT_ROOT / "outputs" / "paper6" / "pipeline_results" / "alternatives_20260418_221709"
OUT_DIR = PROJECT_ROOT / "outputs" / "paper6" / "pipeline_results"
SUBMISSION_FIGS = (
    PROJECT_ROOT / "outputs" / "mechanistic_twin" / "paper6_submission"
    / "jamia" / "figures"
)

CONFIG_ORDER = [
    "baseline_catboost12_gimin",
    "alt1_catboost33_gimin",
    "alt2_cost_sensitive",
    "alt3_binary_collapse",
    "alt4_mean_imputation_12feat",
    "alt4b_mean_imputation_33feat",
]
LABELS = {
    "baseline_catboost12_gimin": "CatBoost-12\n+ GIMIN\n(baseline)",
    "alt1_catboost33_gimin": "CatBoost-33\n+ GIMIN\n(alt-1)",
    "alt2_cost_sensitive": "CatBoost-12\ncost-sensitive\n(alt-2)",
    "alt3_binary_collapse": "CatBoost-12\nbinary collapse\n(alt-3)",
    "alt4_mean_imputation_12feat": "CatBoost-12\n+ Mean\n(alt-4)",
    "alt4b_mean_imputation_33feat": "CatBoost-33\n+ Mean\n(alt-4b)",
}
COLORS = {
    "baseline_catboost12_gimin": "#4b6cb7",     # blue
    "alt1_catboost33_gimin": "#2b8a3e",         # green (winner)
    "alt2_cost_sensitive": "#8b5a3c",           # brown
    "alt3_binary_collapse": "#a0522d",          # tan
    "alt4_mean_imputation_12feat": "#d3a3a3",   # light pink (Mean 12-feat)
    "alt4b_mean_imputation_33feat": "#90c090",  # light green (Mean 33-feat)
}


def load_config(name: str) -> dict:
    path = W4_DIR / f"{name}_summary.json"
    with open(path) as f:
        return json.load(f)


def main():
    SUBMISSION_FIGS.mkdir(parents=True, exist_ok=True)
    summaries = {name: load_config(name) for name in CONFIG_ORDER}

    # Extract Top-1 with 95% CI
    top1 = {}
    for name, s in summaries.items():
        agg = s["aggregate"].get("top1_acc", {})
        top1[name] = {
            "mean": agg.get("mean", float("nan")),
            "lo": agg.get("ci_lo", float("nan")),
            "hi": agg.get("ci_hi", float("nan")),
        }

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    # Panel (a): all 6 configs, Top-1 with CIs
    xs = np.arange(len(CONFIG_ORDER))
    means = [top1[n]["mean"] for n in CONFIG_ORDER]
    errs_lo = [m - top1[n]["lo"] for n, m in zip(CONFIG_ORDER, means)]
    errs_hi = [top1[n]["hi"] - m for n, m in zip(CONFIG_ORDER, means)]
    bar_colors = [COLORS[n] for n in CONFIG_ORDER]
    ax1.bar(xs, means, color=bar_colors, edgecolor="black", linewidth=0.8,
            yerr=[errs_lo, errs_hi], capsize=4)
    ax1.set_xticks(xs)
    ax1.set_xticklabels([LABELS[n] for n in CONFIG_ORDER], fontsize=8, rotation=0)
    ax1.set_ylabel("Within-NSD+ Top-1 accuracy (95% CI)")
    ax1.set_title("(a) Alternative staging pipelines\n(5-fold CV, n=779 NSD+)")
    ax1.set_ylim(0.6, 1.0)
    ax1.axhline(means[0], color="#4b6cb7", linestyle="--", linewidth=0.8, alpha=0.5,
                label="Baseline Top-1")
    # Annotate +16pp for alt1 vs baseline
    ax1.annotate(
        "", xy=(1, means[1]), xytext=(0, means[0]),
        arrowprops={"arrowstyle": "->", "color": "black", "lw": 1.2},
    )
    ax1.text(0.5, (means[0] + means[1]) / 2,
             f"+{100*(means[1]-means[0]):.1f}pp", fontsize=9, ha="center",
             bbox={"boxstyle": "round,pad=0.2", "fc": "white", "ec": "none"})
    ax1.legend(loc="lower right", fontsize=8)

    # Panel (b): GIMIN vs Mean paired comparison on both schemas
    pair_labels = ["12-feat schema", "33-feat schema"]
    gimin_means = [top1["baseline_catboost12_gimin"]["mean"], top1["alt1_catboost33_gimin"]["mean"]]
    gimin_lo = [top1["baseline_catboost12_gimin"]["lo"], top1["alt1_catboost33_gimin"]["lo"]]
    gimin_hi = [top1["baseline_catboost12_gimin"]["hi"], top1["alt1_catboost33_gimin"]["hi"]]
    mean_means = [top1["alt4_mean_imputation_12feat"]["mean"], top1["alt4b_mean_imputation_33feat"]["mean"]]
    mean_lo = [top1["alt4_mean_imputation_12feat"]["lo"], top1["alt4b_mean_imputation_33feat"]["lo"]]
    mean_hi = [top1["alt4_mean_imputation_12feat"]["hi"], top1["alt4b_mean_imputation_33feat"]["hi"]]

    x = np.arange(2)
    w = 0.35
    ax2.bar(x - w/2, gimin_means, w, label="GIMIN imputation",
            color="#4b6cb7", edgecolor="black", linewidth=0.8,
            yerr=[[g - lo for g, lo in zip(gimin_means, gimin_lo)],
                  [hi - g for g, hi in zip(gimin_means, gimin_hi)]], capsize=4)
    ax2.bar(x + w/2, mean_means, w, label="Mean imputation",
            color="#d3a3a3", edgecolor="black", linewidth=0.8,
            yerr=[[m - lo for m, lo in zip(mean_means, mean_lo)],
                  [hi - m for m, hi in zip(mean_means, mean_hi)]], capsize=4)
    ax2.set_xticks(x)
    ax2.set_xticklabels(pair_labels, fontsize=10)
    ax2.set_ylabel("Within-NSD+ Top-1 accuracy (95% CI)")
    ax2.set_title("(b) GIMIN vs. Mean imputation\n(CIs overlap heavily on both schemas)")
    ax2.set_ylim(0.6, 1.0)
    ax2.legend(loc="lower left", fontsize=9)
    for i, (g, m) in enumerate(zip(gimin_means, mean_means)):
        delta = g - m
        ax2.text(i, max(g, m) + 0.035, f"Δ = {delta:+.3f}",
                 ha="center", fontsize=9,
                 bbox={"boxstyle": "round,pad=0.2", "fc": "lightyellow", "ec": "gray"})

    fig.suptitle(
        "Paper 6 W4 pre-registered alternative-pipeline A/B test",
        fontsize=11, y=1.02,
    )
    fig.tight_layout()

    # Save to both pipeline_results and submission figures
    out1 = OUT_DIR / "fig_w4_alt_pipeline.pdf"
    out2 = SUBMISSION_FIGS / "fig_w4_alt_pipeline.pdf"
    fig.savefig(out1, bbox_inches="tight")
    fig.savefig(out1.with_suffix(".png"), dpi=150, bbox_inches="tight")
    fig.savefig(out2, bbox_inches="tight")
    fig.savefig(out2.with_suffix(".png"), dpi=150, bbox_inches="tight")
    print(f"Saved: {out1}")
    print(f"Saved: {out2}")


if __name__ == "__main__":
    main()
