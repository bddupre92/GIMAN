"""Build longitudinal training dataset for Temporal GIMAN from real PPMI data.

Generalises build_training_dataset.py from baseline-only to multi-visit
extraction. Same 7 modalities, same 34 features, but extracted at up to
8 visits per patient (BL through V12 = 36 months).

Output:
  data/03_prodromal/longitudinal_training/
    longitudinal_sequences.pkl   -- dict[patno -> PatientSequence]
    longitudinal_long_form.csv   -- long-form (PATNO, EVENT_ID, features...)
    coverage_matrix.csv          -- [N_patients x N_visits] boolean
    medication_features.csv      -- per-visit medication status
    summary.json                 -- coverage statistics
"""

from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

from src.giman_pipeline.data_processing.longitudinal_assembler import (
    VISIT_MONTH_MAP,
    VISIT_ORDER,
    LongitudinalAssembler,
)

GDRIVE = (
    Path.home()
    / "Library/CloudStorage/GoogleDrive-dupre.blair92@gmail.com/My Drive/CSCI FALL 2025"
)
PPMI_07FEB = GDRIVE / "PPMI_Data_07FEB"
PPMI_CSV = GDRIVE / "data/00_raw/GIMAN/ppmi_data_csv"
RAW_00 = GDRIVE / "data/00_raw"

VALID_EVENTS = set(VISIT_MONTH_MAP.keys())


def load_endpoints() -> pd.DataFrame:
    """Load real PPMI prodromal survival endpoints."""
    path = project_root / "data" / "prodromal_cohort" / "prodromal_survival_data.csv"
    df = pd.read_csv(path)
    print(f"Loaded endpoints: {len(df)} patients, {df['phenoconverted'].sum()} events")
    return df


# ---------------------------------------------------------------------------
# Per-modality extractors (multi-visit versions of build_training_dataset.py)
# ---------------------------------------------------------------------------


def extract_clinical_features_longitudinal(
    patnos: set[int],
) -> pd.DataFrame:
    """Extract expanded clinical features at all visits."""
    # UPDRS Part I
    u1 = pd.read_csv(PPMI_CSV / "MDS-UPDRS_Part_I_18Sep2025.csv", low_memory=False)
    u1 = u1[(u1["PATNO"].isin(patnos)) & (u1["EVENT_ID"].isin(VALID_EVENTS))]
    u1 = u1.groupby(["PATNO", "EVENT_ID"]).first().reset_index()
    u1_features = u1[["PATNO", "EVENT_ID", "NP1RTOT"]].copy()
    u1_features = u1_features.rename(columns={"NP1RTOT": "UPDRS_I"})
    u1_features["NP1RTOT"] = u1_features["UPDRS_I"]

    # UPDRS Part II
    u2 = pd.read_csv(
        PPMI_CSV / "MDS_UPDRS_Part_II__Patient_Questionnaire_18Sep2025.csv",
        low_memory=False,
    )
    u2 = u2[(u2["PATNO"].isin(patnos)) & (u2["EVENT_ID"].isin(VALID_EVENTS))]
    u2 = u2.groupby(["PATNO", "EVENT_ID"]).first().reset_index()
    np2_items = [c for c in u2.columns if c.startswith("NP2")]
    u2["UPDRS_II"] = u2[np2_items].sum(axis=1)
    u2_features = u2[["PATNO", "EVENT_ID", "UPDRS_II"]]

    # UPDRS Part III
    u3 = pd.read_csv(PPMI_CSV / "MDS-UPDRS_Part_III_30Sep2025.csv", low_memory=False)
    u3 = u3[(u3["PATNO"].isin(patnos)) & (u3["EVENT_ID"].isin(VALID_EVENTS))]
    u3 = u3.groupby(["PATNO", "EVENT_ID"]).first().reset_index()

    tremor_items = [
        "NP3PTRMR", "NP3PTRML", "NP3KTRMR", "NP3KTRML",
        "NP3RTARU", "NP3RTALU", "NP3RTARL", "NP3RTALL",
        "NP3RTALJ", "NP3RTCON",
    ]
    pigd_items = ["NP3GAIT", "NP3FRZGT", "NP3PSTBL", "NP3POSTR"]

    existing_tremor = [c for c in tremor_items if c in u3.columns]
    existing_pigd = [c for c in pigd_items if c in u3.columns]

    u3["TREMOR_SCORE"] = u3[existing_tremor].sum(axis=1) if existing_tremor else np.nan
    u3["PIGD_SCORE"] = u3[existing_pigd].sum(axis=1) if existing_pigd else np.nan
    u3["SCHWAB_ENGLAND"] = 100 - u3["NP3TOT"].clip(0, 100)

    u3_features = u3[
        ["PATNO", "EVENT_ID", "NP3TOT", "TREMOR_SCORE", "PIGD_SCORE", "SCHWAB_ENGLAND"]
    ].copy()
    if "NHY" in u3.columns:
        u3_features["NHY"] = u3["NHY"].values
    else:
        u3_features["NHY"] = np.nan

    # Merge on (PATNO, EVENT_ID)
    clinical = u1_features.merge(u2_features, on=["PATNO", "EVENT_ID"], how="outer")
    clinical = clinical.merge(u3_features, on=["PATNO", "EVENT_ID"], how="outer")

    n_visits = len(clinical)
    n_patients = clinical["PATNO"].nunique()
    print(f"Clinical features: {n_visits} visit-records, {n_patients} patients")
    return clinical


def extract_cognitive_features_longitudinal(
    patnos: set[int],
) -> pd.DataFrame:
    """Extract MoCA cognitive scores at all visits."""
    moca = pd.read_csv(
        PPMI_07FEB / "Montreal_Cognitive_Assessment__MoCA__07Feb2026.csv"
    )
    moca = moca[(moca["PATNO"].isin(patnos)) & (moca["EVENT_ID"].isin(VALID_EVENTS))]
    agg = moca.groupby(["PATNO", "EVENT_ID"]).first().reset_index()
    features = agg[["PATNO", "EVENT_ID", "MCATOT"]].copy()
    print(f"Cognitive features: {len(features)} visit-records, {features['PATNO'].nunique()} patients")
    return features


def extract_olfactory_features_longitudinal(
    patnos: set[int],
) -> pd.DataFrame:
    """Extract UPSIT olfactory scores at all visits."""
    upsit = pd.read_csv(
        PPMI_07FEB
        / "University_of_Pennsylvania_Smell_Identification_Test_UPSIT_07Feb2026.csv"
    )
    # UPSIT is often only at SC/BL, but include all valid visits
    valid_plus_sc = VALID_EVENTS | {"SC"}
    upsit = upsit[
        (upsit["PATNO"].isin(patnos)) & (upsit["EVENT_ID"].isin(valid_plus_sc))
    ]
    agg = upsit.groupby(["PATNO", "EVENT_ID"]).first().reset_index()
    features = agg[["PATNO", "EVENT_ID", "TOTAL_CORRECT"]].rename(
        columns={"TOTAL_CORRECT": "UPSIT_SCORE"}
    )
    # Map SC -> BL for consistency
    features.loc[features["EVENT_ID"] == "SC", "EVENT_ID"] = "BL"
    features = features.groupby(["PATNO", "EVENT_ID"]).first().reset_index()
    print(f"Olfactory features: {len(features)} visit-records, {features['PATNO'].nunique()} patients")
    return features


def extract_sleep_features_longitudinal(
    patnos: set[int],
) -> pd.DataFrame:
    """Extract RBD and ESS sleep scores at all visits."""
    # RBD
    rbd = pd.read_csv(
        PPMI_07FEB / "REM_Sleep_Behavior_Disorder_Questionnaire_07Feb2026.csv"
    )
    rbd = rbd[(rbd["PATNO"].isin(patnos)) & (rbd["EVENT_ID"].isin(VALID_EVENTS))]
    rbd = rbd.groupby(["PATNO", "EVENT_ID"]).first().reset_index()
    rbd_items = [c for c in rbd.columns if c.startswith("DRM")]
    rbd["RBD_SCORE"] = rbd[rbd_items].sum(axis=1)
    rbd_features = rbd[["PATNO", "EVENT_ID", "RBD_SCORE"]]

    # ESS
    ess = pd.read_csv(PPMI_CSV / "Epworth_Sleepiness_Scale_18Sep2025.csv")
    ess = ess[(ess["PATNO"].isin(patnos)) & (ess["EVENT_ID"].isin(VALID_EVENTS))]
    ess = ess.groupby(["PATNO", "EVENT_ID"]).first().reset_index()
    ess_items = [c for c in ess.columns if c.startswith("ESS") and c != "ESS"]
    ess["ESS_SCORE"] = ess[ess_items].sum(axis=1)
    ess_features = ess[["PATNO", "EVENT_ID", "ESS_SCORE"]]

    features = rbd_features.merge(ess_features, on=["PATNO", "EVENT_ID"], how="outer")
    print(f"Sleep features: {len(features)} visit-records, {features['PATNO'].nunique()} patients")
    return features


def extract_autonomic_features_longitudinal(
    patnos: set[int],
) -> pd.DataFrame:
    """Extract SCOPA-AUT autonomic score at all visits."""
    scopa = pd.read_csv(PPMI_07FEB / "SCOPA-AUT_07Feb2026.csv")
    scopa = scopa[
        (scopa["PATNO"].isin(patnos)) & (scopa["EVENT_ID"].isin(VALID_EVENTS))
    ]
    agg = scopa.groupby(["PATNO", "EVENT_ID"]).first().reset_index()
    scau_items = [c for c in agg.columns if c.startswith("SCAU") and c[-1].isdigit()]
    agg["SCOPA_AUT_SCORE"] = agg[scau_items].sum(axis=1)
    features = agg[["PATNO", "EVENT_ID", "SCOPA_AUT_SCORE"]]
    print(f"Autonomic features: {len(features)} visit-records, {features['PATNO'].nunique()} patients")
    return features


def extract_genetic_features_longitudinal(
    patnos: set[int],
) -> pd.DataFrame:
    """Extract genetic risk factors (time-invariant, replicated at BL).

    Genetics don't change over time, so we only extract once and the
    assembler replicates across visits.
    """
    gen = pd.read_csv(PPMI_CSV / "iu_genetic_consensus_20250515_08Oct2025.csv")
    gen = gen[gen["PATNO"].isin(patnos)].copy()
    gen = gen.groupby("PATNO").first().reset_index()

    features = gen[["PATNO"]].copy()
    for gene in ["LRRK2", "GBA", "SNCA"]:
        if gene in gen.columns:
            features[gene] = gen[gene].apply(
                lambda x: 1
                if pd.notna(x) and str(x).strip() not in ("", "0", "N/A")
                else 0
            )
        else:
            features[gene] = 0

    if "APOE" in gen.columns:
        features["APOE_E4"] = gen["APOE"].apply(
            lambda x: 1 if pd.notna(x) and "e4" in str(x).lower() else 0
        )
    else:
        features["APOE_E4"] = 0

    features["GENETIC_RISK_SCORE"] = features[["LRRK2", "GBA", "SNCA", "APOE_E4"]].sum(
        axis=1
    )

    # Add EVENT_ID="BL" so merge works; will be replicated to all visits later
    features["EVENT_ID"] = "BL"
    print(f"Genetic features: {len(features)} patients (time-invariant)")
    return features


def extract_imaging_features_longitudinal(
    patnos: set[int],
) -> pd.DataFrame:
    """Extract structural MRI volumes and cortical thickness at all visits."""
    features_list = []

    # Structural volumes
    try:
        vol = pd.read_csv(PPMI_CSV / "FS7_ASEG_VOL_30Sep2025.csv")
        vol = vol[(vol["PATNO"].isin(patnos)) & (vol["EVENT_ID"].isin(VALID_EVENTS))]
        agg = vol.groupby(["PATNO", "EVENT_ID"]).first().reset_index()

        vol_map = {
            "Left_Caudate": "CAUDATE_L_VOL",
            "Right_Caudate": "CAUDATE_R_VOL",
            "Left_Putamen": "PUTAMEN_L_VOL",
            "Right_Putamen": "PUTAMEN_R_VOL",
            "Left_Hippocampus": "HIPPOCAMPUS_L_VOL",
            "Right_Hippocampus": "HIPPOCAMPUS_R_VOL",
        }
        vol_features = agg[["PATNO", "EVENT_ID"]].copy()
        for src, dst in vol_map.items():
            vol_features[dst] = agg[src].values if src in agg.columns else np.nan
        features_list.append(vol_features)
        print(f"Structural volumes: {len(vol_features)} visit-records, {vol_features['PATNO'].nunique()} patients")
    except Exception as e:
        print(f"Structural volumes: FAILED ({e})")

    # Cortical thickness
    try:
        cth = pd.read_csv(PPMI_CSV / "FS7_APARC_CTH_18Sep2025.csv")
        cth = cth[(cth["PATNO"].isin(patnos)) & (cth["EVENT_ID"].isin(VALID_EVENTS))]
        agg = cth.groupby(["PATNO", "EVENT_ID"]).first().reset_index()

        cth_map = {
            "lh_entorhinal": "ENTORHINAL_L_CTH",
            "rh_entorhinal": "ENTORHINAL_R_CTH",
            "lh_caudalanteriorcingulate": "CINGULATE_L_CTH",
            "rh_caudalanteriorcingulate": "CINGULATE_R_CTH",
            "lh_precentral": "PRECENTRAL_L_CTH",
            "rh_precentral": "PRECENTRAL_R_CTH",
        }
        cth_features = agg[["PATNO", "EVENT_ID"]].copy()
        for src, dst in cth_map.items():
            cth_features[dst] = agg[src].values if src in agg.columns else np.nan
        features_list.append(cth_features)
        print(f"Cortical thickness: {len(cth_features)} visit-records, {cth_features['PATNO'].nunique()} patients")
    except Exception as e:
        print(f"Cortical thickness: FAILED ({e})")

    if not features_list:
        return pd.DataFrame(columns=["PATNO", "EVENT_ID"])

    result = features_list[0]
    for df in features_list[1:]:
        result = result.merge(df, on=["PATNO", "EVENT_ID"], how="outer")
    return result


def extract_csf_biomarkers_longitudinal(
    patnos: set[int],
) -> pd.DataFrame:
    """Extract CSF biomarkers at all available visits."""
    bio = pd.read_csv(
        PPMI_CSV / "Current_Biospecimen_Analysis_Results_18Sep2025.csv",
        low_memory=False,
    )
    bio = bio[bio["PATNO"].isin(patnos)]

    test_map = {
        "ABeta": "ABETA42",
        "tTau": "TAU",
        "pTau": "PTAU181",
    }

    # Build per-test, per-visit DataFrames
    result_dfs = []
    for test_name, feature_name in test_map.items():
        subset = bio[bio["TESTNAME"] == test_name].copy()
        if "EVENT_ID" not in subset.columns:
            # CSF biospecimen may not have EVENT_ID; use CLINICAL_EVENT or default to BL
            if "CLINICAL_EVENT" in subset.columns:
                subset["EVENT_ID"] = subset["CLINICAL_EVENT"]
            else:
                subset["EVENT_ID"] = "BL"
        subset = subset[subset["EVENT_ID"].isin(VALID_EVENTS)]
        agg = subset.groupby(["PATNO", "EVENT_ID"])["TESTVALUE"].first().reset_index()
        agg.columns = ["PATNO", "EVENT_ID", feature_name]
        agg[feature_name] = pd.to_numeric(agg[feature_name], errors="coerce")
        result_dfs.append(agg)

    # Alpha-synuclein
    asyn_tests = bio[bio["TESTNAME"].str.contains("Syn|SAA|aSyn", case=False, na=False)]
    if len(asyn_tests) > 0:
        if "EVENT_ID" not in asyn_tests.columns:
            if "CLINICAL_EVENT" in asyn_tests.columns:
                asyn_tests = asyn_tests.copy()
                asyn_tests["EVENT_ID"] = asyn_tests["CLINICAL_EVENT"]
            else:
                asyn_tests = asyn_tests.copy()
                asyn_tests["EVENT_ID"] = "BL"
        asyn_tests = asyn_tests[asyn_tests["EVENT_ID"].isin(VALID_EVENTS)]
        asyn_agg = (
            asyn_tests.groupby(["PATNO", "EVENT_ID"])["TESTVALUE"]
            .first()
            .reset_index()
        )
        asyn_agg.columns = ["PATNO", "EVENT_ID", "ALPHA_SYNUCLEIN"]
        asyn_agg["ALPHA_SYNUCLEIN"] = pd.to_numeric(
            asyn_agg["ALPHA_SYNUCLEIN"], errors="coerce"
        )
        result_dfs.append(asyn_agg)

    if not result_dfs:
        return pd.DataFrame(columns=["PATNO", "EVENT_ID", "ABETA42", "TAU", "PTAU181", "ALPHA_SYNUCLEIN"])

    features = result_dfs[0]
    for df in result_dfs[1:]:
        features = features.merge(df, on=["PATNO", "EVENT_ID"], how="outer")

    n_with = features.drop(columns=["PATNO", "EVENT_ID"]).notna().any(axis=1).sum()
    print(f"CSF biomarkers: {n_with} visit-records with data, {features['PATNO'].nunique()} patients")
    return features


def extract_ppmi_medication_features(
    med_path: Path,
    conmed_path: Path | None,
    patnos: set[int],
) -> pd.DataFrame:
    """Extract PD medication status from PPMI Use_of_PD_Medication table.

    Columns: PDMEDYN (any PD med), ONLDOPA (levodopa), ONDOPAG (dopamine agonist),
    ONOTHER (other PD med).
    """
    rows: list[dict] = []

    try:
        med = pd.read_csv(med_path, low_memory=False)
        med = med[
            (med["PATNO"].isin(patnos)) & (med["EVENT_ID"].isin(VALID_EVENTS))
        ]
        med = med.groupby(["PATNO", "EVENT_ID"]).first().reset_index()

        for _, row in med.iterrows():
            entry = {
                "PATNO": int(row["PATNO"]),
                "EVENT_ID": str(row["EVENT_ID"]),
                "LEVODOPA": 0,
                "DOPAMINE_AGONIST": 0,
                "MAO_B_INHIBITOR": 0,
                "MEDICATION_ACTIVE": 0,
            }
            # PDMEDYN: 1=yes on PD meds
            pdmedyn = row.get("PDMEDYN")
            if pd.notna(pdmedyn) and str(pdmedyn).strip() == "1":
                entry["MEDICATION_ACTIVE"] = 1

            # ONLDOPA: 1=on levodopa
            onldopa = row.get("ONLDOPA")
            if pd.notna(onldopa) and str(onldopa).strip() == "1":
                entry["LEVODOPA"] = 1
                entry["MEDICATION_ACTIVE"] = 1

            # ONDOPAG: 1=on dopamine agonist
            ondopag = row.get("ONDOPAG")
            if pd.notna(ondopag) and str(ondopag).strip() == "1":
                entry["DOPAMINE_AGONIST"] = 1
                entry["MEDICATION_ACTIVE"] = 1

            # ONOTHER: 1=other PD medication (includes MAO-B inhibitors)
            onother = row.get("ONOTHER")
            if pd.notna(onother) and str(onother).strip() == "1":
                entry["MAO_B_INHIBITOR"] = 1
                entry["MEDICATION_ACTIVE"] = 1

            rows.append(entry)

        print(f"Medication history: {len(rows)} visit-records, {med['PATNO'].nunique()} patients")
    except FileNotFoundError:
        print(f"Medication file not found: {med_path}")
    except Exception as e:
        print(f"Medication extraction error: {e}")

    if not rows:
        return pd.DataFrame(
            columns=["PATNO", "EVENT_ID", "LEVODOPA", "DOPAMINE_AGONIST", "MAO_B_INHIBITOR", "MEDICATION_ACTIVE"]
        )

    result = pd.DataFrame(rows)
    return result


def replicate_static_features(
    static_df: pd.DataFrame,
    long_df: pd.DataFrame,
) -> pd.DataFrame:
    """Replicate time-invariant features (genetics) to all visits.

    Takes a DataFrame with [PATNO, EVENT_ID="BL", features...] and
    broadcasts features to all (PATNO, EVENT_ID) pairs in long_df.
    """
    static_cols = [c for c in static_df.columns if c not in ("PATNO", "EVENT_ID")]
    static_flat = static_df.drop(columns=["EVENT_ID"], errors="ignore")
    static_flat = static_flat.groupby("PATNO").first().reset_index()

    # Get all unique (PATNO, EVENT_ID) from long_df
    visits = long_df[["PATNO", "EVENT_ID"]].drop_duplicates()
    replicated = visits.merge(static_flat, on="PATNO", how="left")
    return replicated[["PATNO", "EVENT_ID"] + static_cols]


def main() -> int:
    print("=" * 70)
    print("BUILD LONGITUDINAL DATASET FOR TEMPORAL GIMAN")
    print("=" * 70 + "\n")

    endpoints = load_endpoints()
    patnos = set(endpoints["PATNO"].values)

    # ---------------------------------------------------------------
    # Extract all modalities at all visits
    # ---------------------------------------------------------------
    print(f"\nExtracting multi-visit features for {len(patnos)} patients...")
    print(f"Valid visits: {', '.join(VISIT_ORDER)}")
    print("-" * 70)

    clinical = extract_clinical_features_longitudinal(patnos)
    cognitive = extract_cognitive_features_longitudinal(patnos)
    olfactory = extract_olfactory_features_longitudinal(patnos)
    sleep = extract_sleep_features_longitudinal(patnos)
    autonomic = extract_autonomic_features_longitudinal(patnos)
    genetic = extract_genetic_features_longitudinal(patnos)
    imaging = extract_imaging_features_longitudinal(patnos)
    csf = extract_csf_biomarkers_longitudinal(patnos)

    # ---------------------------------------------------------------
    # Merge into long-form table
    # ---------------------------------------------------------------
    print(f"\n{'=' * 70}")
    print("MERGING INTO LONG-FORM TABLE")
    print(f"{'=' * 70}")

    # Start with clinical (has the most visit coverage)
    long_df = clinical.copy()
    for feat_df in [cognitive, olfactory, sleep, autonomic, imaging, csf]:
        long_df = long_df.merge(feat_df, on=["PATNO", "EVENT_ID"], how="outer")

    # Replicate genetics to all visits
    genetic_replicated = replicate_static_features(genetic, long_df)
    long_df = long_df.merge(genetic_replicated, on=["PATNO", "EVENT_ID"], how="left")

    print(f"Long-form table: {len(long_df)} visit-records, {long_df['PATNO'].nunique()} patients")

    # ---------------------------------------------------------------
    # Extract medication features
    # ---------------------------------------------------------------
    print(f"\n{'=' * 70}")
    print("EXTRACTING MEDICATION FEATURES")
    print(f"{'=' * 70}")

    med_history_path = PPMI_07FEB / "Use_of_PD_Medication-Archived_07Feb2026.csv"
    conmed_path = PPMI_07FEB / "Concomitant_Medication_Log_07Feb2026.csv"

    if not med_history_path.exists():
        # Try alternative filenames
        alt_paths = list(PPMI_07FEB.glob("*PD_Medication*"))
        if alt_paths:
            med_history_path = alt_paths[0]

    med_df = extract_ppmi_medication_features(
        med_path=med_history_path,
        conmed_path=conmed_path if conmed_path.exists() else None,
        patnos=patnos,
    )
    med_summary = {
        "n_records": len(med_df),
        "n_patients": int(med_df["PATNO"].nunique()) if len(med_df) > 0 else 0,
        "levodopa_any": int(med_df["LEVODOPA"].sum()) if len(med_df) > 0 else 0,
        "dopamine_agonist_any": int(med_df["DOPAMINE_AGONIST"].sum()) if len(med_df) > 0 else 0,
        "mao_b_any": int(med_df["MAO_B_INHIBITOR"].sum()) if len(med_df) > 0 else 0,
        "any_medication": int(med_df["MEDICATION_ACTIVE"].sum()) if len(med_df) > 0 else 0,
    }
    print(f"Medication summary: {json.dumps(med_summary, indent=2)}")

    # ---------------------------------------------------------------
    # Convert to temporal sequences
    # ---------------------------------------------------------------
    print(f"\n{'=' * 70}")
    print("CONVERTING TO TEMPORAL SEQUENCES")
    print(f"{'=' * 70}")

    assembler = LongitudinalAssembler(
        config_path=project_root / "config" / "feature_configs" / "longitudinal_34.yaml",
    )
    dataset = assembler.long_to_sequences(long_df, endpoints)

    print(f"\nDataset summary:")
    print(f"  Patients: {dataset.summary['n_patients']}")
    print(f"  Features: {dataset.summary['n_features']}")
    print(f"  Visits per patient: median={dataset.summary['n_visits_median']:.1f}, "
          f"mean={dataset.summary['n_visits_mean']:.1f}, "
          f"min={dataset.summary['n_visits_min']}, "
          f"max={dataset.summary['n_visits_max']}")
    print(f"  Coverage per visit:")
    for vid, count in dataset.summary["coverage_per_visit"].items():
        pct = 100 * count / dataset.summary["n_patients"] if dataset.summary["n_patients"] > 0 else 0
        print(f"    {vid}: {count} patients ({pct:.1f}%)")

    # ---------------------------------------------------------------
    # Missingness report
    # ---------------------------------------------------------------
    print(f"\n{'=' * 70}")
    print("FEATURE MISSINGNESS REPORT (across all visits)")
    print(f"{'=' * 70}")

    config = assembler.config
    for mod_name, mod_cfg in config["modalities"].items():
        feats = mod_cfg["features"]
        if feats[0] in long_df.columns:
            n_total = len(long_df)
            n_complete = long_df[feats].notna().all(axis=1).sum()
            pct = 100 * n_complete / n_total if n_total > 0 else 0
            print(f"  {mod_name:25s}: {n_complete:5d}/{n_total} visit-records complete ({pct:5.1f}%)")
            for f in feats:
                if f in long_df.columns:
                    missing = long_df[f].isna().sum()
                    if missing > 0:
                        print(f"    {f:25s}: {missing:5d} missing ({100 * missing / n_total:.1f}%)")
        else:
            print(f"  {mod_name:25s}: features not found in long-form table")

    # ---------------------------------------------------------------
    # Save outputs
    # ---------------------------------------------------------------
    output_dir = project_root / "data" / "03_prodromal" / "longitudinal_training"
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Pickle sequences
    seq_path = output_dir / "longitudinal_sequences.pkl"
    with open(seq_path, "wb") as f:
        pickle.dump(dataset.sequences, f)
    print(f"\nSaved sequences: {seq_path}")

    # 2. Long-form CSV
    long_path = output_dir / "longitudinal_long_form.csv"
    long_df.to_csv(long_path, index=False)
    print(f"Saved long-form: {long_path}")

    # 3. Coverage matrix
    cov_path = output_dir / "coverage_matrix.csv"
    dataset.coverage_matrix.to_csv(cov_path, index=False)
    print(f"Saved coverage: {cov_path}")

    # 4. Medication features
    med_path = output_dir / "medication_features.csv"
    med_df.to_csv(med_path, index=False)
    print(f"Saved medication: {med_path}")

    # 5. Summary JSON
    full_summary = {
        **dataset.summary,
        "feature_names": dataset.feature_names,
        "modality_map": dataset.modality_map,
        "visit_schedule": dataset.visit_schedule,
        "medication": med_summary,
        "output_dir": str(output_dir),
    }
    summary_path = output_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(full_summary, f, indent=2)
    print(f"Saved summary: {summary_path}")

    # ---------------------------------------------------------------
    # Spot-check: print 3 random patient sequences
    # ---------------------------------------------------------------
    print(f"\n{'=' * 70}")
    print("SPOT CHECK: 3 random patients")
    print(f"{'=' * 70}")

    rng = np.random.RandomState(42)
    sample_patnos = rng.choice(list(dataset.sequences.keys()), size=min(3, len(dataset.sequences)), replace=False)
    for patno in sample_patnos:
        seq = dataset.sequences[patno]
        print(f"\n  PATNO {patno}:")
        print(f"    Visits: {seq.visit_ids} ({seq.n_visits} visits)")
        print(f"    Time points: {seq.time_months.tolist()} months")
        print(f"    Feature shape: {seq.features.shape}")
        obs_pct = 100 * seq.obs_mask.mean()
        print(f"    Observation rate: {obs_pct:.1f}%")
        # Per-visit observation rates
        for i, vid in enumerate(seq.visit_ids):
            visit_obs = 100 * seq.obs_mask[i].mean()
            print(f"      {vid} (t={int(seq.time_months[i])}mo): {visit_obs:.0f}% features observed")

    print(f"\n{'=' * 70}")
    print("LONGITUDINAL DATASET BUILD COMPLETE")
    print(f"{'=' * 70}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
