#!/usr/bin/env python3
"""L1 bridge: export GIMIN DaT-SBR imputed means + temperature-scaled σ
to a Julia-readable parquet for mechanistic-twin likelihood consumption.

Spec: Docs/research_directions/2026-04-19_L1_gimin_sigma_bridge_design.md

This is a stub implementation with the I/O contract wired up. The actual
GIMIN inference for the 1,900-patient longitudinal cohort requires
running GIMIN on each patient's baseline feature vector AND producing
longitudinal CAUDATE/PUTAMEN SBR imputations at each visit; the P6
pipeline currently only does baseline-visit imputation. Extending to
longitudinal imputation is part of Phase 5 Paper 10 execution.

For now, this script:
  1. Loads GIMIN outputs from the P6 v2 pipeline (baseline-visit only).
  2. Emits the Julia-readable parquet with the spec'd schema for
     baseline visits.
  3. Documents the extension points for longitudinal inference.

Output: outputs/mechanistic_twin/l1_gimin_bridge/gimin_dat_sbr_imputed.parquet

Usage:
  .venv/bin/python scripts/mechanistic_twin/export_gimin_to_julia.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

P6_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "paper6" / "pipeline_results" / "v2_full_cohort"
GIMIN_COHORT = PROJECT_ROOT / "GIMImpN_imputation" / "outputs" / "ppmi_full_cohort.parquet"
OUT_DIR = PROJECT_ROOT / "outputs" / "mechanistic_twin" / "l1_gimin_bridge"

# Sensor noise floor for observed DaT-SPECT SBR, from Fearnley-Lees-style
# test-retest literature (e.g., Seibyl 2008, Benamer 2000). Used when an
# SBR value is directly observed (no GIMIN imputation needed).
SBR_SENSOR_SIGMA = 0.08

DAT_FEATURES = [
    "CAUDATE_L_SBR", "CAUDATE_R_SBR",
    "PUTAMEN_L_SBR", "PUTAMEN_R_SBR",
]


def load_p6_v2_patients() -> list[dict]:
    """Load all per-patient pipeline output JSONs from the P6 v2 run."""
    files = sorted(P6_OUTPUT_DIR.glob("patient_*_pipeline.json"))
    records = []
    for fp in files:
        with open(fp) as f:
            d = json.load(f)
        if d.get("error"):
            continue
        records.append(d)
    print(f"Loaded {len(records)} patient pipeline outputs from P6 v2")
    return records


def extract_dat_sbr_observed(patno: int, gimin_cohort: pd.DataFrame) -> dict:
    """Extract observed DaT-SBR values from the GIMIN cohort parquet.
    Index is PATNO; some patients have multiple visits (longitudinal)."""
    if patno not in gimin_cohort.index:
        return {f: np.nan for f in DAT_FEATURES}
    row = gimin_cohort.loc[patno]
    if isinstance(row, pd.DataFrame):  # multiple visits, take baseline (first)
        row = row.iloc[0]
    return {f: float(row[f]) if f in row.index and pd.notna(row[f]) else np.nan
            for f in DAT_FEATURES}


def build_bridge_record(patient_json: dict, observed: dict) -> dict:
    """Compose one row of the L1 bridge parquet for one patient."""
    patno = int(patient_json["patno"])
    gimin_means = patient_json.get("gimin_imputed_means", {})
    gimin_stds = patient_json.get("gimin_calibrated_stds", {})

    record = {"PATNO": patno, "visit_date": "baseline"}

    for feat in DAT_FEATURES:
        obs = observed.get(feat, np.nan)
        is_obs = not np.isnan(obs)
        record[f"{feat}_obs"] = obs if is_obs else np.nan
        record[f"{feat}_is_observed"] = is_obs
        if is_obs:
            record[f"{feat}_imputed_mean"] = obs
            record[f"{feat}_imputed_std"] = SBR_SENSOR_SIGMA
        else:
            # GIMIN may not cover all DaT features on 12-feat schema;
            # fall back to NaN and let Julia mark as censored.
            record[f"{feat}_imputed_mean"] = gimin_means.get(feat, np.nan)
            record[f"{feat}_imputed_std"] = gimin_stds.get(feat, np.nan)

    # Derived mean SBR (bilateral average) with error propagation
    for side_label, left, right in [
        ("CAUDATE", "CAUDATE_L_SBR", "CAUDATE_R_SBR"),
        ("PUTAMEN", "PUTAMEN_L_SBR", "PUTAMEN_R_SBR"),
    ]:
        ml, mr = record[f"{left}_imputed_mean"], record[f"{right}_imputed_mean"]
        sl, sr = record[f"{left}_imputed_std"], record[f"{right}_imputed_std"]
        ol_obs = record[f"{left}_is_observed"] and record[f"{right}_is_observed"]
        if np.isnan(ml) or np.isnan(mr):
            record[f"{side_label}_MEAN_SBR_obs"] = np.nan
            record[f"{side_label}_MEAN_SBR_is_observed"] = False
            record[f"{side_label}_MEAN_SBR_imputed_mean"] = np.nan
            record[f"{side_label}_MEAN_SBR_imputed_std"] = np.nan
        else:
            mean = 0.5 * (ml + mr)
            std = 0.5 * np.sqrt(sl**2 + sr**2)  # independent noise → sqrt of sum of variances, /2 for avg
            record[f"{side_label}_MEAN_SBR_obs"] = mean if ol_obs else np.nan
            record[f"{side_label}_MEAN_SBR_is_observed"] = ol_obs
            record[f"{side_label}_MEAN_SBR_imputed_mean"] = mean
            record[f"{side_label}_MEAN_SBR_imputed_std"] = std

    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=str, default=str(OUT_DIR / "gimin_dat_sbr_imputed.parquet"))
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    patients = load_p6_v2_patients()
    print(f"\nLoading GIMIN cohort parquet (for observed-value reference)...")
    gimin_cohort = pd.read_parquet(GIMIN_COHORT)
    print(f"  shape={gimin_cohort.shape}  index={gimin_cohort.index.name}")

    records = []
    for patient_json in patients:
        patno = int(patient_json["patno"])
        observed = extract_dat_sbr_observed(patno, gimin_cohort)
        rec = build_bridge_record(patient_json, observed)
        records.append(rec)

    if not records:
        print("No records built -- P6 v2 pipeline output may be missing")
        return

    df = pd.DataFrame(records)
    print(f"\nBridge parquet shape: {df.shape}")
    for feat in DAT_FEATURES + ["CAUDATE_MEAN_SBR", "PUTAMEN_MEAN_SBR"]:
        n_obs = df[f"{feat}_is_observed"].sum()
        n_nan_imputed = df[f"{feat}_imputed_mean"].isna().sum()
        print(f"  {feat}: {n_obs}/{len(df)} observed, {n_nan_imputed} still NaN after imputation")

    out_path = Path(args.output)
    df.to_parquet(out_path, index=False)
    print(f"\nSaved: {out_path}")
    print(f"\nL1 interface contract: see Docs/research_directions/2026-04-19_L1_gimin_sigma_bridge_design.md")

    # Extension points for Phase 5 execution
    print("\n" + "=" * 70)
    print("EXTENSION POINTS (deferred to Phase 5 / Paper 10 execution):")
    print("=" * 70)
    print("1. Baseline-visit only → longitudinal: P6 v2 pipeline runs GIMIN once per")
    print("   patient on baseline features. Phase 5 needs GIMIN per visit to produce")
    print("   time-varying (μ, σ) pairs. Options:")
    print("   a. Rerun GIMIN on each longitudinal feature vector (expensive; ~22s per visit × 16,699 visits)")
    print("   b. Interpolate GIMIN's baseline posterior through time using the ODE solution")
    print("      (cheaper; requires deriving a time-evolution of the imputation σ)")
    print("2. The 12-feat P6 pipeline only imputes 4 features; the full 33-feat GIMIN")
    print("   schema has 4 DaT features (CAUDATE_L/R_SBR + PUTAMEN_L/R_SBR) plus")
    print("   derived asymmetries. Phase 5 should use the full 33-feat GIMIN.")


if __name__ == "__main__":
    main()
