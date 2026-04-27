"""Paper 1 — Extended 33-feature / 3-modality ETL for the supplementary MM-GAT sensitivity variant.

Produces ``data/05_features/paper1_features_extended_33.csv`` by merging the canonical
22-feature table (``data/05_features/paper1_features_with_targets.csv``) with 11 extra
features pulled directly from raw PPMI CSVs:

  * 6 cortical thickness (entorhinal L/R, posterior-cingulate L/R, precentral L/R)
    from FS7_APARC_CTH_08Feb2026.csv — FreeSurfer 7 ASEG cortical-thickness release
  * 4 CSF biomarkers (CSF Alpha-synuclein, ABeta 1-42, pTau, tTau) from
    Current_Biospecimen_Analysis_Results_30Sep2025.csv — baseline / earliest visit per patient
  * 1 polygenic risk score (GRS_TOTAL) — SKIPPED in this pass because PPMI does not publish
    a pre-computed GRS in iu_genetic_consensus_20250515_*.csv (the file holds LRRK2/GBA/
    VPS35/SNCA/PRKN/PARK7/PINK1 carrier flags and APOE genotype strings, not Nalls-2019
    weighted PRS). Feature count is therefore 32 (22 + 10), not 33. The script, the
    supplementary S-3 text, and the submission sensitivity-variant table must state this
    explicitly. Future work: add a weighted-PRS computation using PPMI genotype data +
    Nalls-2019 summary statistics (not in scope for Paper 1 first-journal submission).

Usage::

    python scripts/paper1/assemble_extended_33_feat.py

Outputs:
  * data/05_features/paper1_features_extended_33.csv
  * data/05_features/paper1_features_extended_33_metadata.json

Author: Paper 1 / IEEE JBHI sensitivity variant, 2026-04-22.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
OUT_CSV = DATA / "05_features" / "paper1_features_extended_33.csv"
OUT_META = DATA / "05_features" / "paper1_features_extended_33_metadata.json"

FEATURES_BASE = DATA / "05_features" / "paper1_features_with_targets.csv"
CTH_FILE = DATA / "00_raw" / "FS7_APARC_CTH_08Feb2026.csv"
CSF_FILE = (
    DATA
    / "00_raw"
    / "GIMAN"
    / "ppmi_data_csv"
    / "Current_Biospecimen_Analysis_Results_30Sep2025.csv"
)

# FreeSurfer 7 APARC cortical-thickness columns we want retained
CTH_COLS = {
    "lh_entorhinal": "CTH_ENTORHINAL_L",
    "rh_entorhinal": "CTH_ENTORHINAL_R",
    "lh_posteriorcingulate": "CTH_POSTCINGULATE_L",
    "rh_posteriorcingulate": "CTH_POSTCINGULATE_R",
    "lh_precentral": "CTH_PRECENTRAL_L",
    "rh_precentral": "CTH_PRECENTRAL_R",
}

# Biospecimen TESTNAME -> output column for CSF biomarkers
CSF_TESTS = {
    "CSF Alpha-synuclein": "CSF_ASYN",
    "ABeta 1-42": "CSF_ABETA42",
    "pTau": "CSF_PTAU181",
    "tTau": "CSF_TTAU",
}

# Preferred baseline EVENT_IDs (in priority order)
BASELINE_EVENTS = ["BL", "SC", "V01", "V02", "V04"]


def load_cth() -> pd.DataFrame:
    df = pd.read_csv(CTH_FILE, low_memory=False)
    # Keep BL visit only (one row per patient at baseline)
    df = df[df["EVENT_ID"] == "BL"].copy()
    keep = ["PATNO", *CTH_COLS.keys()]
    missing = [c for c in keep if c not in df.columns]
    if missing:
        raise ValueError(f"CTH columns missing from {CTH_FILE}: {missing}")
    df = df[keep].rename(columns=CTH_COLS)
    # Numeric
    for c in CTH_COLS.values():
        df[c] = pd.to_numeric(df[c], errors="coerce")
    # One row per patient — drop any duplicates by keeping first
    df = df.drop_duplicates(subset=["PATNO"], keep="first").reset_index(drop=True)
    return df


def load_csf() -> pd.DataFrame:
    df = pd.read_csv(CSF_FILE, low_memory=False)
    df = df[df["TESTNAME"].isin(CSF_TESTS.keys())].copy()
    df["TESTVALUE"] = pd.to_numeric(df["TESTVALUE"], errors="coerce")
    df = df.dropna(subset=["TESTVALUE"])
    # Pick earliest baseline-like visit per (PATNO, TESTNAME)
    df["ev_rank"] = df["CLINICAL_EVENT"].map(
        {e: i for i, e in enumerate(BASELINE_EVENTS)}
    )
    df = df.sort_values(["PATNO", "TESTNAME", "ev_rank"]).drop_duplicates(
        subset=["PATNO", "TESTNAME"], keep="first"
    )
    wide = df.pivot(index="PATNO", columns="TESTNAME", values="TESTVALUE").reset_index()
    wide = wide.rename(columns=CSF_TESTS)
    # Guarantee all target columns exist
    for out_col in CSF_TESTS.values():
        if out_col not in wide.columns:
            wide[out_col] = np.nan
    wide = wide[["PATNO", *CSF_TESTS.values()]]
    wide["PATNO"] = wide["PATNO"].astype(int)
    return wide


def main() -> None:
    base = pd.read_csv(FEATURES_BASE)
    n_base = len(base)
    print(f"Loaded base 22-feature table: {n_base} rows x {len(base.columns)} cols")

    cth = load_cth()
    csf = load_csf()
    print(f"Cortical thickness (baseline): {len(cth)} patients")
    print(f"CSF biomarkers (first observed): {len(csf)} patients")

    base["PATNO"] = base["PATNO"].astype(int)
    cth["PATNO"] = cth["PATNO"].astype(int)

    merged = base.merge(cth, on="PATNO", how="left").merge(csf, on="PATNO", how="left")

    # Feature-availability report on the 10 extra features (GRS not available)
    extra_feats = [*CTH_COLS.values(), *CSF_TESTS.values()]
    coverage = {
        c: int(merged[c].notna().sum()) for c in extra_feats if c in merged.columns
    }
    full_coverage = int((merged[extra_feats].notna().all(axis=1)).sum())
    nan_rate = {c: 1 - v / n_base for c, v in coverage.items()}

    merged.to_csv(OUT_CSV, index=False)
    meta = {
        "n_patients": n_base,
        "n_feature_columns": len(
            [c for c in merged.columns if c not in base.columns or c in base.columns]
        ),
        "n_base_features": 22,
        "n_extra_features": len(extra_feats),
        "n_total_features": 22 + len(extra_feats),
        "grs_total_included": False,
        "grs_skip_reason": (
            "iu_genetic_consensus_20250515_*.csv contains carrier flags and APOE genotype "
            "strings but no pre-computed weighted polygenic risk score. Computing a Nalls "
            "2019 PRS from raw PPMI genotypes is out of scope for Paper 1's first journal "
            "submission."
        ),
        "cortical_thickness_coverage": {
            k: coverage[k] for k in CTH_COLS.values() if k in coverage
        },
        "csf_coverage": {k: coverage[k] for k in CSF_TESTS.values() if k in coverage},
        "full_coverage_all_10": full_coverage,
        "nan_rate_per_feature": nan_rate,
        "sources": {
            "base": str(FEATURES_BASE.relative_to(ROOT)),
            "cortical_thickness": str(CTH_FILE.relative_to(ROOT)),
            "csf_biomarkers": str(CSF_FILE.relative_to(ROOT)),
        },
        "citations": {
            "fischl2012": "Fischl B. FreeSurfer. NeuroImage 2012;62(2):774-781. doi:10.1016/j.neuroimage.2012.01.021",
            "mollenhauer2017": "Mollenhauer B, Caspell-Garcia CJ, Coffey CT, et al. Longitudinal CSF biomarkers in patients with early Parkinson disease and healthy controls. Neurology 2017;89(19):1959-1969. doi:10.1212/WNL.0000000000004609",
            "nalls2019_not_used": "Nalls MA et al. Identification of novel risk loci, causal insights, and heritable risk for Parkinson's disease: a meta-analysis of genome-wide association studies. Lancet Neurol 2019;18(12):1091-1102. doi:10.1016/S1474-4422(19)30320-5 — NOT USED in this pass (PRS not precomputed).",
        },
        "output_csv": str(OUT_CSV.relative_to(ROOT)),
    }
    with OUT_META.open("w") as fh:
        json.dump(meta, fh, indent=2)

    print(f"\nWrote extended feature table: {OUT_CSV}")
    print(f"Total rows: {len(merged)}; total columns: {len(merged.columns)}")
    print("\nPer-feature coverage (n / {}):".format(n_base))
    for feat, cnt in coverage.items():
        print(f"  {feat:<22} {cnt:>6}  ({100*(1-nan_rate[feat]):.1f}%)")
    print(f"\nPatients with all 10 extra features observed: {full_coverage}")
    print(f"Metadata: {OUT_META}")


if __name__ == "__main__":
    main()
