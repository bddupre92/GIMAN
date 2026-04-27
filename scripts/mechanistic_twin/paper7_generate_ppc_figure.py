#!/usr/bin/env python3
"""Per-patient posterior-predictive-check (PPC) figure for Paper 7.

Three representative PPMI Wave A patients (slow, typical, fast progressor)
selected by their IS-weighted posterior mean pct-loss/yr. For each patient,
draws 500 weighted posterior samples, forward-simulates the Variant B
slow-fast-collapse SBR trajectory on a dense grid, and plots the observed
DaT-SPECT measurements against the 90% posterior-predictive band and median.

Addresses Paper 7 Gupta-style "clinical-use individual" figure gap flagged in
PAPER_REVIEW_GUPTA_STYLE_2026-04-16.md (per-patient PPC panel is the single
largest content gap across the dissertation).
"""
from __future__ import annotations

import sys
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from giman_pipeline.mechanistic_twin_v2.forward_model import (
    HR_PER_YR,
    T_TOX_CONST,
    predict_sbr,
)

COHORT = ROOT / "outputs/mechanistic_twin/data/dat_spect_longitudinal.parquet"
POSTERIOR_STORE = (
    ROOT / "outputs/mechanistic_twin/paper10_mech_vs_giman/phase2_posteriors_full_samples.h5"
)
OUT_PNG = ROOT / "outputs/dissertation/figures/p7_per_patient_ppc.png"
OUT_PDF = ROOT / "outputs/dissertation/figures/p7_per_patient_ppc.pdf"

N_DRAWS = 500
T_GRID = np.linspace(0.0, 10.0, 201)
CI_LOW, CI_HIGH = 0.05, 0.95


def weighted_quantile(values: np.ndarray, weights: np.ndarray, q: float) -> float:
    order = np.argsort(values)
    v = values[order]
    w = weights[order]
    cdf = np.cumsum(w) / w.sum()
    return float(np.interp(q, cdf, v))


def pick_patients(cohort: pd.DataFrame) -> list[tuple[int, float, str]]:
    wave_a = cohort[cohort["wave"] == "A"].sort_values(["PATNO", "t_years"])
    counts = wave_a.groupby("PATNO").size()
    wave_a = wave_a[wave_a["PATNO"].isin(counts[counts >= 5].index)]

    stats: list[tuple[int, float]] = []
    with h5py.File(POSTERIOR_STORE, "r") as f:
        for pat in wave_a["PATNO"].unique():
            key = f"patient_{int(pat)}"
            if key not in f:
                continue
            group = f[key]["v1"]
            samples = group["samples"][:]
            weights = group["weights"][:]
            if weights.sum() <= 0:
                continue
            weights = weights / weights.sum()
            pct_loss = (1.0 - np.exp(-samples[:, 2] * HR_PER_YR)) * 100.0
            stats.append((int(pat), float((weights * pct_loss).sum())))

    ordered = sorted(stats, key=lambda x: x[1])
    n = len(ordered)
    slow = ordered[n // 10]
    typical = ordered[n // 2]
    fast = ordered[-max(1, n // 10)]
    return [
        (slow[0], slow[1], "Slow progressor"),
        (typical[0], typical[1], "Typical progressor"),
        (fast[0], fast[1], "Fast progressor"),
    ]


def patient_panel(
    ax: plt.Axes,
    patno: int,
    pct_loss_mean: float,
    label: str,
    cohort: pd.DataFrame,
) -> None:
    patient_df = cohort[cohort["PATNO"] == patno].sort_values("t_years")
    t_obs = patient_df["t_years"].to_numpy()
    sbr_obs = patient_df["sbr_putamen_mean"].to_numpy()
    sbr_0 = float(sbr_obs[0])

    with h5py.File(POSTERIOR_STORE, "r") as f:
        group = f[f"patient_{patno}"]["v1"]
        samples = group["samples"][:]
        weights = group["weights"][:]

    weights = weights / weights.sum()
    rng = np.random.default_rng(20260416)
    draw_idx = rng.choice(len(samples), size=N_DRAWS, replace=True, p=weights)
    k_n_draws = samples[draw_idx, 0]
    alpha_draws = samples[draw_idx, 1]

    pred = predict_sbr(k_n_draws, alpha_draws, T_GRID, sbr_0)
    lo = np.quantile(pred, CI_LOW, axis=0)
    hi = np.quantile(pred, CI_HIGH, axis=0)
    med = np.quantile(pred, 0.5, axis=0)

    ax.fill_between(T_GRID, lo, hi, color="#5B9BD5", alpha=0.35, label="90% PPC band")
    ax.plot(T_GRID, med, color="#1f3a66", lw=1.6, label="Posterior median")
    ax.plot(
        t_obs,
        sbr_obs,
        "o",
        color="#C00000",
        ms=7,
        mec="black",
        mew=0.5,
        label="Observed DaT-SPECT",
    )

    ax.set_xlim(0, 10)
    ax.set_ylim(0, max(sbr_obs.max() * 1.15, 2.8))
    ax.set_xlabel("Years from baseline")
    ax.set_ylabel("Putaminal SBR")
    ax.set_title(
        f"{label} (PATNO {patno})\n"
        f"posterior mean loss = {pct_loss_mean:.1f}\\%/yr",
        fontsize=10,
    )
    ax.grid(alpha=0.25)


def main() -> None:
    cohort = pd.read_parquet(COHORT)
    picks = pick_patients(cohort)
    print("Selected patients:")
    for patno, pct, label in picks:
        print(f"  {label}: PATNO={patno}, pct_loss_mean={pct:.2f}/yr")

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.5), sharey=False)
    for ax, (patno, pct, label) in zip(axes, picks):
        patient_panel(ax, patno, pct, label, cohort)

    axes[-1].legend(loc="upper right", fontsize=8, framealpha=0.9)
    fig.suptitle(
        "Per-patient posterior-predictive check — Paper 7 IS-weighted Phase 2 posterior",
        fontsize=12,
        y=1.02,
    )
    fig.tight_layout()
    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PNG, dpi=300, bbox_inches="tight")
    fig.savefig(OUT_PDF, bbox_inches="tight")
    print(f"Wrote {OUT_PNG}")
    print(f"Wrote {OUT_PDF}")


if __name__ == "__main__":
    main()
