#!/usr/bin/env python3
"""Validate L1 bridge v2 GIMIN imputations against Xing Core Lab observations.

Finding from 2026-04-19 L1 execution: of the 776 patients whose baseline DaT
is imputed by GIMIN (NaN in ppmi_full_cohort.parquet), 242 appear in the
≥2-observed-scans mechanistic-twin cohort and DO have an observed baseline
DaT in dat_spect_longitudinal.parquet (sourced from Xing Core Lab + other
supplementary data).

This script answers the VALIDATION question: when GIMIN imputes a DaT-SBR
value for a patient whose PPMI cohort parquet is missing, does it agree
with the Xing Core Lab observation for the same patient-visit?

Output metrics per DaT feature (L/R CAUDATE and L/R PUTAMEN):
  - N patients with both GIMIN imputation + Xing observation
  - Pearson correlation (GIMIN μ vs Xing obs)
  - MAE and bias (GIMIN μ − Xing obs)
  - 95% coverage: fraction of Xing obs inside [GIMIN μ − 1.96 σ, GIMIN μ + 1.96 σ]
  - Under/over-coverage diagnosis

Outputs:
  outputs/mechanistic_twin/l1_gimin_bridge/gimin_vs_xing_validation.json
  outputs/mechanistic_twin/l1_gimin_bridge/fig_gimin_vs_xing.{pdf,png}
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]

L1_BRIDGE = (
    PROJECT_ROOT / "outputs/mechanistic_twin/l1_gimin_bridge/gimin_dat_sbr_baseline_v2.parquet"
)
SCANS_LONGITUDINAL = (
    PROJECT_ROOT / "outputs/mechanistic_twin/data/dat_spect_longitudinal.parquet"
)
OUT_DIR = PROJECT_ROOT / "outputs/mechanistic_twin/l1_gimin_bridge"

# Map bridge column prefix → longitudinal scan column (bilateral mean)
FEAT_MAP = {
    "CAUDATE_MEAN_SBR": "sbr_caudate_mean",
    "PUTAMEN_MEAN_SBR": "sbr_putamen_mean",
}
# Individual-lobe columns only exist in bridge (Xing stores bilateral only in the scans parquet)
BRIDGE_ONLY_FEATURES = [
    "CAUDATE_L_SBR", "CAUDATE_R_SBR",
    "PUTAMEN_L_SBR", "PUTAMEN_R_SBR",
]


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print("Loading inputs...")
    bridge = pd.read_parquet(L1_BRIDGE)
    scans = pd.read_parquet(SCANS_LONGITUDINAL)
    print(f"  L1 bridge v2: {bridge.shape}  (2,197 staged patients)")
    print(f"  Longitudinal scans: {scans.shape}")

    # Restrict scans to t=0 (baseline visit in longitudinal cohort)
    t0 = scans[scans["t_years"] == 0.0].copy()
    print(f"  Longitudinal t=0 rows: {len(t0)}")

    # Merge: join bridge with t=0 observations
    merged = bridge.merge(t0, on="PATNO", how="inner")
    print(f"  Bridge ∩ t=0 scans: {len(merged)} patients")

    results: dict = {"overall": {}, "per_feature": {}, "imputed_vs_xing": {}}

    for bridge_feat, xing_col in FEAT_MAP.items():
        sub = merged.dropna(subset=[xing_col]).copy()
        if sub.empty:
            continue

        gimin_mean = sub[f"{bridge_feat}_imputed_mean"].values
        gimin_std = sub[f"{bridge_feat}_imputed_std"].values
        xing_obs = sub[xing_col].values
        bridge_is_observed = sub[f"{bridge_feat}_is_observed"].values

        # Split into: bridge observed (should trivially match), bridge imputed
        obs_mask = bridge_is_observed.astype(bool)
        imputed_mask = ~obs_mask

        def _summary(g_mu, g_sig, x_obs, label):
            if len(x_obs) == 0:
                return {"n": 0}
            resid = g_mu - x_obs
            bias = float(np.mean(resid))
            mae = float(np.mean(np.abs(resid)))
            rmse = float(np.sqrt(np.mean(resid**2)))
            pearson = float(np.corrcoef(g_mu, x_obs)[0, 1]) if len(x_obs) > 1 else float("nan")
            # 95% coverage: Gaussian posterior at each patient
            lo = g_mu - 1.96 * g_sig
            hi = g_mu + 1.96 * g_sig
            covered = ((x_obs >= lo) & (x_obs <= hi)).mean()
            # σ distribution
            sig_stats = {
                "median": float(np.median(g_sig)),
                "mean": float(np.mean(g_sig)),
                "p25": float(np.quantile(g_sig, 0.25)),
                "p75": float(np.quantile(g_sig, 0.75)),
            }
            # MAE / σ ratio: if <=1, σ is over-conservative; if >>1, σ is under-confident
            typical_ratio = mae / sig_stats["median"] if sig_stats["median"] > 0 else float("nan")
            return {
                "label": label,
                "n": int(len(x_obs)),
                "bias_mean_minus_obs": bias,
                "mae": mae,
                "rmse": rmse,
                "pearson_r": pearson,
                "coverage_95ci": float(covered),
                "sigma_stats": sig_stats,
                "mae_over_median_sigma": typical_ratio,
            }

        results["per_feature"][bridge_feat] = {
            "bridge_observed": _summary(
                gimin_mean[obs_mask], gimin_std[obs_mask], xing_obs[obs_mask],
                "bridge-observed (trivial match expected; sensor σ only)",
            ),
            "bridge_imputed_vs_xing": _summary(
                gimin_mean[imputed_mask], gimin_std[imputed_mask], xing_obs[imputed_mask],
                "GIMIN-imputed vs Xing observed (THE KEY VALIDATION)",
            ),
        }

    # Print headline
    print("\n" + "=" * 78)
    print("VALIDATION: GIMIN imputed (μ, σ) vs Xing Core Lab observed DaT-SBR at baseline")
    print("=" * 78)
    for bridge_feat, r in results["per_feature"].items():
        imp = r["bridge_imputed_vs_xing"]
        if imp["n"] == 0:
            continue
        print(f"\n{bridge_feat}:")
        print(f"  N patients (bridge-imputed ∩ Xing-observed): {imp['n']}")
        print(f"  Bias (GIMIN μ − Xing obs):   {imp['bias_mean_minus_obs']:+.4f}")
        print(f"  MAE:                         {imp['mae']:.4f}")
        print(f"  RMSE:                        {imp['rmse']:.4f}")
        print(f"  Pearson r:                   {imp['pearson_r']:.3f}")
        print(f"  Median GIMIN σ:              {imp['sigma_stats']['median']:.4f}")
        print(f"  95% CI coverage rate:        {imp['coverage_95ci']:.3f}")
        print(f"  MAE / median σ ratio:        {imp['mae_over_median_sigma']:.2f}  "
              f"({'over-conservative' if imp['mae_over_median_sigma'] < 1 else 'under-confident' if imp['mae_over_median_sigma'] > 1.5 else 'well-calibrated'})")

    # Scatter plot: GIMIN μ vs Xing obs for imputed-baseline patients
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    for ax, (bridge_feat, xing_col) in zip(axes, FEAT_MAP.items()):
        sub = merged.dropna(subset=[xing_col]).copy()
        if sub.empty:
            continue
        g_mu = sub[f"{bridge_feat}_imputed_mean"].values
        g_sig = sub[f"{bridge_feat}_imputed_std"].values
        x_obs = sub[xing_col].values
        obs_flag = sub[f"{bridge_feat}_is_observed"].values.astype(bool)

        # Observed subset (sanity: should lie on diagonal)
        ax.scatter(x_obs[obs_flag], g_mu[obs_flag], s=8, alpha=0.25, color="#888",
                   label=f"observed (n={obs_flag.sum()})")
        # Imputed subset (the validation cohort)
        ax.errorbar(x_obs[~obs_flag], g_mu[~obs_flag], yerr=1.96 * g_sig[~obs_flag],
                    fmt="o", ms=4, elinewidth=0.6, alpha=0.55, color="#d62728",
                    label=f"imputed (n={(~obs_flag).sum()})")
        mn = min(g_mu.min(), x_obs.min())
        mx = max(g_mu.max(), x_obs.max())
        ax.plot([mn, mx], [mn, mx], "k--", lw=0.8, alpha=0.4, label="y = x")
        ax.set_xlabel(f"Xing Core Lab observed ({xing_col})")
        ax.set_ylabel(f"GIMIN μ ± 95% CI ({bridge_feat})")
        ax.set_title(bridge_feat)
        ax.legend(loc="best", fontsize=8)

    fig.suptitle("L1 bridge validation: GIMIN imputed vs Xing Core Lab observed", y=1.02)
    fig.tight_layout()
    fig_path = OUT_DIR / "fig_gimin_vs_xing.pdf"
    fig.savefig(fig_path, bbox_inches="tight")
    fig.savefig(fig_path.with_suffix(".png"), dpi=150, bbox_inches="tight")
    print(f"\nFigure saved: {fig_path}")

    # Write JSON
    out_path = OUT_DIR / "gimin_vs_xing_validation.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"JSON saved: {out_path}")


if __name__ == "__main__":
    main()
