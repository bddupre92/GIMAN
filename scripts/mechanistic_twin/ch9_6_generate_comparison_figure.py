#!/usr/bin/env python3
"""Comparison figure for §9.6 — like-for-like LOO across 5 comparators.

Reads the unified results:
  outputs/mechanistic_twin/ch9_6/comparators/comparator_summary.json
  outputs/mechanistic_twin/ch9_6/leaspy_loo_baseline.json
  outputs/mechanistic_twin/ch9_6/loo_forward.json (SAEM v3)

Produces two figures:
  fig_9_6_7_comparison_loo_rmse.{png,pdf}     — bar chart, LOO RMSE
  fig_9_6_8_comparison_per_patient_rmse.{png,pdf} — boxplot per-patient RMSE
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs/mechanistic_twin/ch9_6"
FIG = OUT / "figures"
FIG.mkdir(parents=True, exist_ok=True)


def main() -> None:
    # Load all result sources
    summary = json.loads((OUT / "comparators/comparator_summary.json").read_text())
    leaspy_loo = json.loads((OUT / "leaspy_loo_baseline.json").read_text())
    saem_loo = json.loads((OUT / "loo_forward.json").read_text())

    # Per-scan data for boxplots
    comparator_preds = pd.read_csv(OUT / "comparators/comparator_predictions.csv")
    leaspy_loo_per = pd.read_csv(OUT / "leaspy_loo_per_scan.csv")
    saem_loo_per = pd.read_csv(OUT / "loo_forward.csv")

    # Filter SAEM to the 618-patient matched cohort that Leaspy used
    leaspy_pats = set(leaspy_loo_per["patno"].astype(int))
    saem_matched = saem_loo_per[saem_loo_per["patno"].astype(int).isin(leaspy_pats)]

    # --- Compile the headline LOO metrics ---
    rows = [
        ("Naive per-patient\nexp (LOO)", summary.get("naive_ols_exp_per_patient_loo", {}).get("aggregate", {}).get("rmse"), "baseline"),
        ("LME inductive\n(new-patient LOO)", summary.get("statsmodels_lme_patient_loo", {}).get("aggregate", {}).get("rmse"), "baseline"),
        ("Leaspy Logistic\nper-scan LOO", leaspy_loo["rmse_sbr"], "phenomenological"),
        ("SAEM v3 5-ch\n(matched cohort)", float(np.sqrt(((saem_matched["observed"] - saem_matched["pred_mean"]) ** 2).mean())), "mechanistic"),
        ("SAEM v3 5-ch\n(full 1,051 pts)", saem_loo["coverage_95_credible_interval"] and saem_loo.get("median_relative_error") and float(np.sqrt(((saem_loo_per["observed"] - saem_loo_per["pred_mean"]) ** 2).mean())), "mechanistic"),
    ]
    df = pd.DataFrame(rows, columns=["model", "rmse", "category"])
    df = df.dropna(subset=["rmse"])

    # ─────────── Figure 7: LOO RMSE bar chart ───────────
    fig, ax = plt.subplots(figsize=(8, 4.5))
    colors = {"baseline": "#cccccc", "phenomenological": "#5B9BD5",
              "mechanistic": "#C00000"}
    bar_colors = [colors[c] for c in df["category"]]
    bars = ax.bar(range(len(df)), df["rmse"], color=bar_colors, edgecolor="black")
    for bar, val in zip(bars, df["rmse"]):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 0.005,
                f"{val:.3f}", ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax.set_xticks(range(len(df)))
    ax.set_xticklabels(df["model"], fontsize=9)
    ax.set_ylabel("LOO RMSE (SBR units)")
    ax.set_title("§9.6 Like-for-like comparison — leave-one-out prediction quality")
    ax.axhline(y=0.08, color="gray", linestyle=":", alpha=0.5)
    ax.text(len(df) - 0.5, 0.082, "DaT-SPECT test-retest ~8%",
            fontsize=8, color="gray", ha="right")
    # Legend
    from matplotlib.patches import Patch
    legend_els = [
        Patch(facecolor=colors["baseline"], edgecolor="black", label="Naive / empirical baseline"),
        Patch(facecolor=colors["phenomenological"], edgecolor="black", label="Phenomenological (Leaspy, Koval 2021)"),
        Patch(facecolor=colors["mechanistic"], edgecolor="black", label="Mechanistic ODE (this work)"),
    ]
    ax.legend(handles=legend_els, loc="upper right", fontsize=9)
    fig.tight_layout()
    fig.savefig(FIG / "fig_9_6_7_comparison_loo_rmse.png", dpi=300)
    fig.savefig(FIG / "fig_9_6_7_comparison_loo_rmse.pdf")
    plt.close(fig)
    print(f"Wrote {FIG / 'fig_9_6_7_comparison_loo_rmse.png'}")

    # ─────────── Figure 8: Per-patient RMSE boxplots ───────────
    # Compute per-patient RMSE for each LOO method on the matched 618-patient cohort
    def per_patient_rmse(df_preds, patno_col="patno"):
        rmses = []
        for pat, g in df_preds.groupby(patno_col):
            res = g["observed"] - g[g.columns[g.columns.str.contains("pred")][0]]
            rmses.append(np.sqrt((res ** 2).mean()))
        return np.array(rmses)

    leaspy_rmse = per_patient_rmse(leaspy_loo_per)
    saem_rmse = per_patient_rmse(saem_matched.rename(columns={"pred_mean": "pred"}))
    naive_preds = comparator_preds[
        comparator_preds["comparator"] == "naive_ols_exp_per_patient_loo"
    ]
    naive_rmse = per_patient_rmse(naive_preds) if len(naive_preds) else np.array([])

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    datasets = [naive_rmse, leaspy_rmse, saem_rmse]
    labels = ["Naive per-patient\nLOO (n=618)", "Leaspy per-scan\nLOO (n=618)",
              "SAEM v3 5-ch\nLOO (n=618)"]
    bp = ax.boxplot(
        datasets,
        tick_labels=labels,
        patch_artist=True,
        medianprops=dict(color="black", linewidth=2),
        boxprops=dict(alpha=0.85),
    )
    for patch, color in zip(bp["boxes"], ["#cccccc", "#5B9BD5", "#C00000"]):
        patch.set_facecolor(color)

    # Overlay median lines
    for i, d in enumerate(datasets):
        if len(d) > 0:
            ax.text(i + 1, np.median(d) + 0.01,
                    f"med={np.median(d):.3f}", ha="center", fontsize=9)

    ax.set_ylabel("Per-patient RMSE (SBR)")
    ax.set_title("Per-patient RMSE distribution — matched 618-patient cohort (≥3 scans)")
    ax.axhline(y=0.08, color="gray", linestyle=":", alpha=0.5)
    ax.text(len(datasets) + 0.3, 0.082, "DaT-SPECT test-retest", fontsize=8,
            color="gray", ha="right")
    ax.set_ylim(0, min(1.0, max([d.max() for d in datasets if len(d) > 0]) * 1.1))
    fig.tight_layout()
    fig.savefig(FIG / "fig_9_6_8_comparison_per_patient_rmse.png", dpi=300)
    fig.savefig(FIG / "fig_9_6_8_comparison_per_patient_rmse.pdf")
    plt.close(fig)
    print(f"Wrote {FIG / 'fig_9_6_8_comparison_per_patient_rmse.png'}")


if __name__ == "__main__":
    main()
