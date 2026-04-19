#!/usr/bin/env python3
"""W3 case-study stratified figure: per-class timing-interval coverage + width
distribution for 922 transitioning patients, with 5 featured vignettes
(Patient 3380, 3207, 3785, 3476, 3960) overlaid as highlighted dots.

Produces:
  outputs/paper4/conformal/fig_case_study_classes.pdf
  outputs/paper4/conformal/fig_case_study_classes.png
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = PROJECT_ROOT / "outputs" / "paper4" / "conformal"

FEATURED = [3380, 3207, 3785, 3476, 3960]
CLASS_ORDER = ["rapid", "regressor", "skip", "stable"]
CLASS_COLORS = {"rapid": "#d62728", "regressor": "#9467bd",
                "skip": "#ff7f0e", "stable": "#2ca02c"}
MODEL_MARKERS = {"deephit": "o", "graph_dt": "s"}
MODEL_NAMES = {"deephit": "DeepHit", "graph_dt": "Graph-DT"}


def main():
    with open(OUT_DIR / "per_patient_intervals_all.json") as f:
        records = json.load(f)
    with open(OUT_DIR / "per_class_aggregate.json") as f:
        agg = json.load(f)

    df = pd.DataFrame(records)
    print(f"Records: {len(df)}  unique patients: {df['patno'].nunique()}")

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

    # Panel (a): per-class coverage barplot (DH vs GDT)
    ax = axes[0]
    xs = np.arange(len(CLASS_ORDER))
    w = 0.35
    dh_cov = [agg[c]["deephit"].get("coverage", 0) or 0 for c in CLASS_ORDER]
    gdt_cov = [agg[c]["graph_dt"].get("coverage", 0) or 0 for c in CLASS_ORDER]
    dh_n = [agg[c]["deephit"].get("n_records", 0) for c in CLASS_ORDER]
    gdt_n = [agg[c]["graph_dt"].get("n_records", 0) for c in CLASS_ORDER]
    ax.bar(xs - w / 2, dh_cov, w, label="DeepHit", color="#4b6cb7", edgecolor="black", linewidth=0.5)
    ax.bar(xs + w / 2, gdt_cov, w, label="Graph-DT", color="#8b5a3c", edgecolor="black", linewidth=0.5)
    ax.axhline(0.90, color="red", linestyle="--", linewidth=1, alpha=0.7, label="Target (90% CL)")
    ax.set_xticks(xs)
    ax.set_xticklabels(CLASS_ORDER)
    ax.set_ylim(0.5, 1.0)
    ax.set_ylabel("Timing coverage")
    ax.set_title("Per-class conformal coverage\n(90% CL, n=922 patients)")
    for i, (d, g) in enumerate(zip(dh_n, gdt_n)):
        ax.text(i - w / 2, dh_cov[i] + 0.01, f"{d}", ha="center", fontsize=8)
        ax.text(i + w / 2, gdt_cov[i] + 0.01, f"{g}", ha="center", fontsize=8)
    ax.legend(loc="lower left", fontsize=8)

    # Panel (b): per-class median interval width
    ax = axes[1]
    dh_w = [agg[c]["deephit"].get("median_width_months", 0) or 0 for c in CLASS_ORDER]
    gdt_w = [agg[c]["graph_dt"].get("median_width_months", 0) or 0 for c in CLASS_ORDER]
    ax.bar(xs - w / 2, dh_w, w, label="DeepHit", color="#4b6cb7", edgecolor="black", linewidth=0.5)
    ax.bar(xs + w / 2, gdt_w, w, label="Graph-DT", color="#8b5a3c", edgecolor="black", linewidth=0.5)
    ax.set_xticks(xs)
    ax.set_xticklabels(CLASS_ORDER)
    ax.set_ylabel("Median interval width (months)")
    ax.set_title("Per-class interval width\n(narrower = more informative)")
    ax.legend(loc="upper left", fontsize=8)

    # Panel (c): observed time vs predicted-interval midpoint scatter,
    # colored by phenotype class, with 5 featured vignettes overlaid as stars
    ax = axes[2]
    dh_df = df[df["model"] == "deephit"].copy()
    dh_df["midpoint"] = (dh_df["timing_lo_months"] + dh_df["timing_hi_months"]) / 2.0
    for cls in CLASS_ORDER:
        sub = dh_df[dh_df["phenotype_class"] == cls]
        ax.scatter(
            sub["observed_time_months"], sub["midpoint"],
            c=CLASS_COLORS[cls], alpha=0.25, s=12, label=f"{cls} (n={len(sub)})",
        )
    # Featured vignettes: filter to records, overlay as gold stars
    featured_df = dh_df[dh_df["patno"].isin(FEATURED)]
    ax.scatter(
        featured_df["observed_time_months"], featured_df["midpoint"],
        c="gold", edgecolor="black", s=160, marker="*", zorder=10,
        label=f"Featured vignettes (n={len(featured_df)})",
    )
    # Label each featured dot with PATNO
    for _, r in featured_df.iterrows():
        ax.annotate(
            str(int(r["patno"])),
            (r["observed_time_months"], r["midpoint"]),
            textcoords="offset points", xytext=(8, 6), fontsize=7,
        )
    mx = max(dh_df["observed_time_months"].max(), dh_df["midpoint"].max())
    ax.plot([0, mx], [0, mx], "k--", alpha=0.3, linewidth=0.8, label="y = x (perfect)")
    ax.set_xlabel("Observed transition time (months)")
    ax.set_ylabel("Predicted interval midpoint (months)")
    ax.set_title("Observed vs predicted transition time\n(DeepHit, 5 featured overlaid)")
    ax.legend(loc="upper left", fontsize=7)

    fig.suptitle(
        "Paper 3+4 Case-Study Scale-Up: n=5 vignettes -> 922 stratified patients",
        fontsize=11, y=1.02,
    )
    fig.tight_layout()
    fig_path = OUT_DIR / "fig_case_study_classes.pdf"
    fig.savefig(fig_path, bbox_inches="tight")
    fig.savefig(fig_path.with_suffix(".png"), dpi=150, bbox_inches="tight")
    print(f"Saved: {fig_path}")


if __name__ == "__main__":
    main()
