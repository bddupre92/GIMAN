#!/usr/bin/env python3
"""Run GIMIN inference to produce DaT-SBR imputations for the L1 bridge.

The Paper 6 v2 pipeline runs GIMIN on baseline visits of the 2,197-patient
staged PPMI cohort but only persists 4 features per patient (the CatBoost-12
overlap set: AGE, MOCA, ESS, RBD). The DaT-SBR imputations ARE computed in
memory but not saved to the per-patient JSONs. This script runs the same
GIMIN stack and saves ALL 33 features' (μ, temperature-scaled σ) to a
parquet keyed by PATNO.

We then JOIN with the P3 1,900-patient longitudinal cohort to produce the
L1 bridge parquet with DaT-SBR (observed, imputed_mean, imputed_std,
is_observed) per patient.

Scope: baseline-visit imputation only. Per-visit longitudinal imputation
requires rebuilding the graph to include longitudinal rows as separate
nodes; deferred to Phase 5 Task 5 extension.

Outputs:
  outputs/mechanistic_twin/l1_gimin_bridge/gimin_33feat_baseline.parquet
    -- 2,197 staged patients × 33 features × (mean, calibrated_std)
  outputs/mechanistic_twin/l1_gimin_bridge/gimin_dat_sbr_baseline_v2.parquet
    -- L1 bridge keyed by PATNO, joined against P3 cohort

Usage:
  .venv/bin/python scripts/mechanistic_twin/run_gimin_for_l1_bridge.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "GIMImpN_imputation"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "paper6"))

GIMIN_CHECKPOINT = (
    PROJECT_ROOT
    / "outputs"
    / "paper2_benchmark"
    / "runs"
    / "cal_retune_lambda0.1_warmup0"
    / "checkpoints"
    / "frac0.1_run0_GIMIN_StageDecoderOnly.pt"
)
OUT_DIR = PROJECT_ROOT / "outputs" / "mechanistic_twin" / "l1_gimin_bridge"
P3_FEATURES = PROJECT_ROOT / "data" / "07_paper3_features" / "longitudinal_features.csv"
GIMIN_COHORT = PROJECT_ROOT / "GIMImpN_imputation" / "outputs" / "ppmi_full_cohort.parquet"
STAGING = PROJECT_ROOT / "data" / "04_staging" / "nsd_iss_staging_results.csv"

SBR_SENSOR_SIGMA = 0.08

DAT_FEATURES = [
    "CAUDATE_L_SBR", "CAUDATE_R_SBR",
    "PUTAMEN_L_SBR", "PUTAMEN_R_SBR",
]


def select_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def run_gimin_stack():
    """Reuse P6 v2 setup_gimin_stack to run inference on the 2,197 staged cohort."""
    from unified_pipeline_demo_v2 import setup_gimin_stack  # noqa: E402

    args = argparse.Namespace(
        gimin_checkpoint=GIMIN_CHECKPOINT,
    )
    device = select_device()
    print(f"Device: {device}")
    print(f"GIMIN checkpoint: {GIMIN_CHECKPOINT.name}")
    stack = setup_gimin_stack(args, device)
    return stack


def build_baseline_parquet(stack: dict) -> pd.DataFrame:
    """Convert the in-memory GIMIN output to a tidy parquet keyed by PATNO."""
    patnos = stack["patnos"]
    feature_names = stack["feature_names"]
    imputed_mean = stack["imputed_means"]
    calibrated_std = stack["calibrated_std"]

    data = {"PATNO": [int(p) for p in patnos]}
    for fi, fname in enumerate(feature_names):
        data[f"{fname}_imputed_mean"] = imputed_mean[:, fi]
        data[f"{fname}_imputed_std"] = calibrated_std[:, fi]

    df = pd.DataFrame(data)
    return df


def build_l1_bridge_parquet(gimin_baseline: pd.DataFrame) -> pd.DataFrame:
    """Join GIMIN baseline imputations with observed DaT-SBR to build the L1 bridge.

    Schema (per PATNO, baseline visit only):
      - {DAT}_obs: observed SBR value (NaN if missing)
      - {DAT}_imputed_mean: GIMIN posterior mean (NaN only if feature not in schema)
      - {DAT}_imputed_std: GIMIN temperature-scaled posterior std
      - {DAT}_is_observed: True if obs is not NaN
      - Plus derived CAUDATE_MEAN_SBR, PUTAMEN_MEAN_SBR with error propagation
    """
    # Load observed DaT-SBR values from GIMIN cohort (baseline visit)
    gimin_cohort = pd.read_parquet(GIMIN_COHORT)
    print(f"GIMIN cohort parquet: {gimin_cohort.shape}")

    # Filter to PATNOs in our GIMIN baseline imputation
    gimin_cohort = gimin_cohort[gimin_cohort.index.isin(gimin_baseline["PATNO"])]
    # Take first row per PATNO (baseline)
    gimin_cohort = gimin_cohort[~gimin_cohort.index.duplicated(keep="first")]
    observed = gimin_cohort[DAT_FEATURES].reset_index()

    # Merge observed + imputed
    merged = gimin_baseline.merge(observed, on="PATNO", how="inner", suffixes=("", "_obs_raw"))
    print(f"Merged baseline bridge: {merged.shape}")

    # Build the L1 bridge with {obs, imputed_mean, imputed_std, is_observed}
    rows = []
    for _, r in merged.iterrows():
        record = {"PATNO": int(r["PATNO"]), "visit_date": "baseline"}
        for feat in DAT_FEATURES:
            obs_raw = r[feat]
            is_obs = pd.notna(obs_raw)
            record[f"{feat}_obs"] = float(obs_raw) if is_obs else np.nan
            record[f"{feat}_is_observed"] = is_obs
            if is_obs:
                record[f"{feat}_imputed_mean"] = float(obs_raw)
                record[f"{feat}_imputed_std"] = SBR_SENSOR_SIGMA
            else:
                record[f"{feat}_imputed_mean"] = float(r[f"{feat}_imputed_mean"])
                record[f"{feat}_imputed_std"] = float(r[f"{feat}_imputed_std"])

        # Derived bilateral means (with error propagation)
        for side_label, left, right in [
            ("CAUDATE", "CAUDATE_L_SBR", "CAUDATE_R_SBR"),
            ("PUTAMEN", "PUTAMEN_L_SBR", "PUTAMEN_R_SBR"),
        ]:
            ml, mr = record[f"{left}_imputed_mean"], record[f"{right}_imputed_mean"]
            sl, sr = record[f"{left}_imputed_std"], record[f"{right}_imputed_std"]
            both_obs = record[f"{left}_is_observed"] and record[f"{right}_is_observed"]
            mean = 0.5 * (ml + mr)
            std = 0.5 * np.sqrt(sl**2 + sr**2)
            record[f"{side_label}_MEAN_SBR_obs"] = mean if both_obs else np.nan
            record[f"{side_label}_MEAN_SBR_is_observed"] = both_obs
            record[f"{side_label}_MEAN_SBR_imputed_mean"] = mean
            record[f"{side_label}_MEAN_SBR_imputed_std"] = std

        rows.append(record)

    return pd.DataFrame(rows)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    stack = run_gimin_stack()

    print("\nBuilding GIMIN 33-feature baseline parquet...")
    gimin_baseline = build_baseline_parquet(stack)
    gimin_baseline.to_parquet(OUT_DIR / "gimin_33feat_baseline.parquet", index=False)
    print(f"  Saved: {OUT_DIR / 'gimin_33feat_baseline.parquet'}  shape={gimin_baseline.shape}")

    print("\nBuilding L1 bridge parquet (DaT-SBR obs + imputed + σ)...")
    bridge = build_l1_bridge_parquet(gimin_baseline)
    out_path = OUT_DIR / "gimin_dat_sbr_baseline_v2.parquet"
    bridge.to_parquet(out_path, index=False)
    print(f"  Saved: {out_path}  shape={bridge.shape}")

    # Summary
    print("\n" + "=" * 70)
    print("L1 BRIDGE V2 SUMMARY (baseline-visit, 2,197-patient staged cohort)")
    print("=" * 70)
    for feat in DAT_FEATURES + ["CAUDATE_MEAN_SBR", "PUTAMEN_MEAN_SBR"]:
        n_obs = bridge[f"{feat}_is_observed"].sum()
        n_total = len(bridge)
        # Distribution of imputed sigma for non-observed rows
        imp_mask = ~bridge[f"{feat}_is_observed"]
        if imp_mask.any():
            sig_med = bridge.loc[imp_mask, f"{feat}_imputed_std"].median()
            sig_iqr = (
                bridge.loc[imp_mask, f"{feat}_imputed_std"].quantile(0.75)
                - bridge.loc[imp_mask, f"{feat}_imputed_std"].quantile(0.25)
            )
            print(
                f"  {feat}: {n_obs}/{n_total} observed, "
                f"{n_total - n_obs} GIMIN-imputed (σ median={sig_med:.3f}, IQR={sig_iqr:.3f})"
            )
        else:
            print(f"  {feat}: {n_obs}/{n_total} observed, 0 imputed")

    # Also: how many patients in the P3 longitudinal cohort (1,900) are in this bridge
    p3 = pd.read_csv(P3_FEATURES, low_memory=False)
    p3_patnos = set(p3["PATNO"].unique())
    bridge_patnos = set(bridge["PATNO"])
    overlap = p3_patnos & bridge_patnos
    print(f"\n  P3 longitudinal ∩ bridge: {len(overlap)} of {len(p3_patnos)} P3 patients covered")


if __name__ == "__main__":
    main()
