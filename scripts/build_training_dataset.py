"""Build clean training dataset for True GIMAN from real PPMI data.

Assembles multimodal features for prodromal patients with real phenoconversion
endpoints. Uses MICE imputation (fit on train split only).

Feature modalities (baseline_36.yaml):
  1. Expanded Clinical: UPDRS_I, UPDRS_II, Schwab-England, PIGD, Tremor score
  2. Structural Imaging: Caudate/Putamen/Hippocampus volumes (L/R)
  3. DAT-SPECT SBR: Not available for prodromal — will be imputed or dropped
  4. CSF Biomarkers: alpha-synuclein, tau, abeta42, ptau181
  5. Clinical Biomarkers: UPSIT, RBD, SCOPA-AUT, ESS scores
  6. Cortical Thickness: Entorhinal/Cingulate/Precentral (L/R)
  7. Genetic: LRRK2, GBA, APOE, SNCA, genetic risk score

Feature modalities (full_clinical.yaml adds):
  8. Motor/Cognitive: NP3TOT, NP1RTOT, NHY, MCATOT
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

project_root = Path(__file__).resolve().parents[1]

GDRIVE = (
    Path.home()
    / "Library/CloudStorage/GoogleDrive-dupre.blair92@gmail.com/My Drive/CSCI FALL 2025"
)
PPMI_07FEB = GDRIVE / "PPMI_Data_07FEB"
PPMI_CSV = GDRIVE / "data/00_raw/GIMAN/ppmi_data_csv"
RAW_00 = GDRIVE / "data/00_raw"


def load_endpoints() -> pd.DataFrame:
    """Load real PPMI endpoints."""
    path = project_root / "data" / "prodromal_cohort" / "prodromal_survival_data.csv"
    df = pd.read_csv(path)
    print(f"Loaded endpoints: {len(df)} patients, {df['phenoconverted'].sum()} events")
    return df


def extract_clinical_features(patnos: set[int]) -> pd.DataFrame:
    """Extract expanded clinical features from UPDRS I/II/III."""
    # UPDRS Part I (Non-motor)
    u1 = pd.read_csv(PPMI_CSV / "MDS-UPDRS_Part_I_18Sep2025.csv", low_memory=False)
    u1_bl = u1[(u1["PATNO"].isin(patnos)) & (u1["EVENT_ID"] == "BL")]
    u1_agg = u1_bl.groupby("PATNO").first().reset_index()
    u1_features = u1_agg[["PATNO", "NP1RTOT"]].copy()
    u1_features = u1_features.rename(columns={"NP1RTOT": "UPDRS_I"})
    # Also keep NP1RTOT under its original name for motor_cognitive modality
    u1_features["NP1RTOT"] = u1_features["UPDRS_I"]

    # UPDRS Part II (Motor ADL)
    u2 = pd.read_csv(
        PPMI_CSV / "MDS_UPDRS_Part_II__Patient_Questionnaire_18Sep2025.csv",
        low_memory=False,
    )
    u2_bl = u2[(u2["PATNO"].isin(patnos)) & (u2["EVENT_ID"] == "BL")]
    u2_agg = u2_bl.groupby("PATNO").first().reset_index()
    # Compute total from items
    np2_items = [c for c in u2_agg.columns if c.startswith("NP2")]
    u2_agg["UPDRS_II"] = u2_agg[np2_items].sum(axis=1)
    u2_features = u2_agg[["PATNO", "UPDRS_II"]]

    # UPDRS Part III (Motor Exam)
    u3 = pd.read_csv(PPMI_CSV / "MDS-UPDRS_Part_III_30Sep2025.csv", low_memory=False)
    u3_bl = u3[(u3["PATNO"].isin(patnos)) & (u3["EVENT_ID"] == "BL")]
    u3_agg = u3_bl.groupby("PATNO").first().reset_index()

    # Extract subscores: PIGD, Tremor, Schwab-England proxy, NP3TOT, NHY
    tremor_items = [
        "NP3PTRMR",
        "NP3PTRML",
        "NP3KTRMR",
        "NP3KTRML",
        "NP3RTARU",
        "NP3RTALU",
        "NP3RTARL",
        "NP3RTALL",
        "NP3RTALJ",
        "NP3RTCON",
    ]
    pigd_items = ["NP3GAIT", "NP3FRZGT", "NP3PSTBL", "NP3POSTR"]

    existing_tremor = [c for c in tremor_items if c in u3_agg.columns]
    existing_pigd = [c for c in pigd_items if c in u3_agg.columns]

    u3_agg["TREMOR_SCORE"] = (
        u3_agg[existing_tremor].sum(axis=1) if existing_tremor else np.nan
    )
    u3_agg["PIGD_SCORE"] = (
        u3_agg[existing_pigd].sum(axis=1) if existing_pigd else np.nan
    )
    # Schwab-England is not directly in UPDRS-III; use total as proxy
    u3_agg["SCHWAB_ENGLAND"] = 100 - u3_agg["NP3TOT"].clip(0, 100)  # Inverse mapping

    u3_features = u3_agg[
        ["PATNO", "NP3TOT", "TREMOR_SCORE", "PIGD_SCORE", "SCHWAB_ENGLAND"]
    ]

    # NHY (Hoehn & Yahr) is in UPDRS-III
    if "NHY" in u3_agg.columns:
        u3_features = u3_features.copy()
        u3_features["NHY"] = u3_agg["NHY"].values
    else:
        u3_features = u3_features.copy()
        u3_features["NHY"] = np.nan

    # Merge
    clinical = u1_features.merge(u2_features, on="PATNO", how="outer")
    clinical = clinical.merge(u3_features, on="PATNO", how="outer")

    print(f"Clinical features: {len(clinical)} patients")
    return clinical


def extract_cognitive_features(patnos: set[int]) -> pd.DataFrame:
    """Extract MoCA cognitive scores."""
    moca = pd.read_csv(
        PPMI_07FEB / "Montreal_Cognitive_Assessment__MoCA__07Feb2026.csv"
    )
    bl = moca[(moca["PATNO"].isin(patnos)) & (moca["EVENT_ID"] == "BL")]
    agg = bl.groupby("PATNO").first().reset_index()
    features = agg[["PATNO", "MCATOT"]].copy()
    print(f"Cognitive features: {len(features)} patients")
    return features


def extract_olfactory_features(patnos: set[int]) -> pd.DataFrame:
    """Extract UPSIT smell identification score."""
    upsit = pd.read_csv(
        PPMI_07FEB
        / "University_of_Pennsylvania_Smell_Identification_Test_UPSIT_07Feb2026.csv"
    )
    bl = upsit[(upsit["PATNO"].isin(patnos)) & (upsit["EVENT_ID"].isin(["SC", "BL"]))]
    agg = bl.groupby("PATNO").first().reset_index()
    features = agg[["PATNO", "TOTAL_CORRECT"]].rename(
        columns={"TOTAL_CORRECT": "UPSIT_SCORE"}
    )
    print(f"Olfactory features: {len(features)} patients")
    return features


def extract_sleep_features(patnos: set[int]) -> pd.DataFrame:
    """Extract RBD and ESS sleep scores."""
    # RBD Questionnaire
    rbd = pd.read_csv(
        PPMI_07FEB / "REM_Sleep_Behavior_Disorder_Questionnaire_07Feb2026.csv"
    )
    bl = rbd[(rbd["PATNO"].isin(patnos)) & (rbd["EVENT_ID"] == "BL")]
    agg = bl.groupby("PATNO").first().reset_index()
    rbd_items = [c for c in agg.columns if c.startswith("DRM")]
    agg["RBD_SCORE"] = agg[rbd_items].sum(axis=1)
    rbd_features = agg[["PATNO", "RBD_SCORE"]]

    # ESS
    ess = pd.read_csv(PPMI_CSV / "Epworth_Sleepiness_Scale_18Sep2025.csv")
    bl = ess[(ess["PATNO"].isin(patnos)) & (ess["EVENT_ID"] == "BL")]
    agg = bl.groupby("PATNO").first().reset_index()
    ess_items = [c for c in agg.columns if c.startswith("ESS") and c != "ESS"]
    agg["ESS_SCORE"] = agg[ess_items].sum(axis=1)
    ess_features = agg[["PATNO", "ESS_SCORE"]]

    features = rbd_features.merge(ess_features, on="PATNO", how="outer")
    print(f"Sleep features: {len(features)} patients")
    return features


def extract_autonomic_features(patnos: set[int]) -> pd.DataFrame:
    """Extract SCOPA-AUT autonomic score."""
    scopa = pd.read_csv(PPMI_07FEB / "SCOPA-AUT_07Feb2026.csv")
    bl = scopa[(scopa["PATNO"].isin(patnos)) & (scopa["EVENT_ID"] == "BL")]
    agg = bl.groupby("PATNO").first().reset_index()
    scau_items = [c for c in agg.columns if c.startswith("SCAU") and c[-1].isdigit()]
    agg["SCOPA_AUT_SCORE"] = agg[scau_items].sum(axis=1)
    features = agg[["PATNO", "SCOPA_AUT_SCORE"]]
    print(f"Autonomic features: {len(features)} patients")
    return features


def extract_genetic_features(patnos: set[int]) -> pd.DataFrame:
    """Extract genetic risk factors."""
    gen = pd.read_csv(PPMI_CSV / "iu_genetic_consensus_20250515_08Oct2025.csv")
    gen = gen[gen["PATNO"].isin(patnos)].copy()
    gen = gen.groupby("PATNO").first().reset_index()

    # Binary risk alleles
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

    # APOE from APOE column (e.g., "e3/e3", "e3/e4", etc.)
    if "APOE" in gen.columns:
        features["APOE_E4"] = gen["APOE"].apply(
            lambda x: 1 if pd.notna(x) and "e4" in str(x).lower() else 0
        )
    else:
        features["APOE_E4"] = 0

    # Genetic risk score: sum of risk alleles
    features["GENETIC_RISK_SCORE"] = features[["LRRK2", "GBA", "SNCA", "APOE_E4"]].sum(
        axis=1
    )

    print(f"Genetic features: {len(features)} patients")
    return features


def extract_imaging_features(patnos: set[int]) -> pd.DataFrame:
    """Extract structural MRI volumes and cortical thickness."""
    features_list = []

    # Structural volumes (FreeSurfer ASEG)
    try:
        vol = pd.read_csv(PPMI_CSV / "FS7_ASEG_VOL_30Sep2025.csv")
        bl = vol[(vol["PATNO"].isin(patnos)) & (vol["EVENT_ID"] == "BL")]
        agg = bl.groupby("PATNO").first().reset_index()

        vol_map = {
            "Left_Caudate": "CAUDATE_L_VOL",
            "Right_Caudate": "CAUDATE_R_VOL",
            "Left_Putamen": "PUTAMEN_L_VOL",
            "Right_Putamen": "PUTAMEN_R_VOL",
            "Left_Hippocampus": "HIPPOCAMPUS_L_VOL",
            "Right_Hippocampus": "HIPPOCAMPUS_R_VOL",
        }

        vol_features = agg[["PATNO"]].copy()
        for src, dst in vol_map.items():
            vol_features[dst] = agg[src].values if src in agg.columns else np.nan
        features_list.append(vol_features)
        print(f"Structural volumes: {len(vol_features)} patients")
    except Exception as e:
        print(f"Structural volumes: FAILED ({e})")

    # Cortical thickness (FreeSurfer APARC)
    try:
        cth = pd.read_csv(PPMI_CSV / "FS7_APARC_CTH_18Sep2025.csv")
        bl = cth[(cth["PATNO"].isin(patnos)) & (cth["EVENT_ID"] == "BL")]
        agg = bl.groupby("PATNO").first().reset_index()

        cth_map = {
            "lh_entorhinal": "ENTORHINAL_L_CTH",
            "rh_entorhinal": "ENTORHINAL_R_CTH",
            "lh_caudalanteriorcingulate": "CINGULATE_L_CTH",
            "rh_caudalanteriorcingulate": "CINGULATE_R_CTH",
            "lh_precentral": "PRECENTRAL_L_CTH",
            "rh_precentral": "PRECENTRAL_R_CTH",
        }

        cth_features = agg[["PATNO"]].copy()
        for src, dst in cth_map.items():
            cth_features[dst] = agg[src].values if src in agg.columns else np.nan
        features_list.append(cth_features)
        print(f"Cortical thickness: {len(cth_features)} patients")
    except Exception as e:
        print(f"Cortical thickness: FAILED ({e})")

    if not features_list:
        return pd.DataFrame(columns=["PATNO"])

    result = features_list[0]
    for df in features_list[1:]:
        result = result.merge(df, on="PATNO", how="outer")
    return result


def extract_csf_biomarkers(patnos: set[int]) -> pd.DataFrame:
    """Extract CSF biomarkers from Biospecimen Analysis."""
    bio = pd.read_csv(
        PPMI_CSV / "Current_Biospecimen_Analysis_Results_18Sep2025.csv",
        low_memory=False,
    )
    bio = bio[bio["PATNO"].isin(patnos)]

    # Map test names to feature names
    test_map = {
        "ABeta": "ABETA42",
        "tTau": "TAU",
        "pTau": "PTAU181",
    }

    features = pd.DataFrame({"PATNO": list(patnos)})

    for test_name, feature_name in test_map.items():
        subset = bio[bio["TESTNAME"] == test_name]
        # Take first result per patient
        agg = subset.groupby("PATNO")["TESTVALUE"].first().reset_index()
        agg.columns = ["PATNO", feature_name]
        # Convert to numeric
        agg[feature_name] = pd.to_numeric(agg[feature_name], errors="coerce")
        features = features.merge(agg, on="PATNO", how="left")

    # Alpha-synuclein may have different test names
    asyn_tests = bio[bio["TESTNAME"].str.contains("Syn|SAA|aSyn", case=False, na=False)]
    if len(asyn_tests) > 0:
        asyn_agg = asyn_tests.groupby("PATNO")["TESTVALUE"].first().reset_index()
        asyn_agg.columns = ["PATNO", "ALPHA_SYNUCLEIN"]
        asyn_agg["ALPHA_SYNUCLEIN"] = pd.to_numeric(
            asyn_agg["ALPHA_SYNUCLEIN"], errors="coerce"
        )
        features = features.merge(asyn_agg, on="PATNO", how="left")
    else:
        features["ALPHA_SYNUCLEIN"] = np.nan

    print(
        f"CSF biomarkers: {features[list(test_map.values())].notna().any(axis=1).sum()} patients with data"
    )
    return features


def extract_dat_spect_features(patnos: set[int]) -> pd.DataFrame:
    """Extract DAT-SPECT SBR features.

    Note: Xing quantitative SBR is NOT available for prodromal patients.
    DaTScan_SBR_Analysis may have some data. If not, these will be imputed.
    """
    features = pd.DataFrame({"PATNO": list(patnos)})

    # Try DaTScan_SBR_Analysis first
    try:
        sbr = pd.read_csv(RAW_00 / "DaTScan_SBR_Analysis_08Feb2026.csv")
        sbr_prod = sbr[sbr["PATNO"].isin(patnos)]
        if len(sbr_prod) > 0:
            bl = sbr_prod[sbr_prod["EVENT_ID"].isin(["SC", "BL"])]
            agg = bl.groupby("PATNO").first().reset_index()
            col_map = {
                "DATSCAN_CAUDATE_R": "CAUDATE_R_SBR",
                "DATSCAN_CAUDATE_L": "CAUDATE_L_SBR",
                "DATSCAN_PUTAMEN_R": "PUTAMEN_R_SBR",
                "DATSCAN_PUTAMEN_L": "PUTAMEN_L_SBR",
            }
            for src, dst in col_map.items():
                features[dst] = features["PATNO"].map(
                    agg.set_index("PATNO").get(src, pd.Series(dtype=float))
                )
            n_with = features[list(col_map.values())].notna().any(axis=1).sum()
            print(f"DAT-SPECT SBR: {n_with} patients with data")
        else:
            for col in [
                "CAUDATE_L_SBR",
                "CAUDATE_R_SBR",
                "PUTAMEN_L_SBR",
                "PUTAMEN_R_SBR",
            ]:
                features[col] = np.nan
            print("DAT-SPECT SBR: 0 patients (not available for prodromal)")
    except Exception:
        for col in ["CAUDATE_L_SBR", "CAUDATE_R_SBR", "PUTAMEN_L_SBR", "PUTAMEN_R_SBR"]:
            features[col] = np.nan
        print("DAT-SPECT SBR: 0 patients (file not found)")

    # Compute asymmetry indices (will be NaN if SBR not available)
    features["CAUDATE_ASYMMETRY"] = (
        features["CAUDATE_R_SBR"] - features["CAUDATE_L_SBR"]
    ).abs() / (features["CAUDATE_R_SBR"] + features["CAUDATE_L_SBR"]).clip(lower=0.01)
    features["PUTAMEN_ASYMMETRY"] = (
        features["PUTAMEN_R_SBR"] - features["PUTAMEN_L_SBR"]
    ).abs() / (features["PUTAMEN_R_SBR"] + features["PUTAMEN_L_SBR"]).clip(lower=0.01)

    return features


def assemble_dataset(config_path: str) -> pd.DataFrame:
    """Assemble full training dataset according to feature config."""
    config = yaml.safe_load(open(project_root / config_path))
    modalities = config["modalities"]

    endpoints = load_endpoints()
    patnos = set(endpoints["PATNO"].values)

    print(f"\nExtracting features for {len(patnos)} patients...")
    print(f"Config: {config_path} ({config.get('total_features', '?')} features)")
    print("-" * 60)

    # Extract all modalities
    clinical = extract_clinical_features(patnos)
    cognitive = extract_cognitive_features(patnos)
    olfactory = extract_olfactory_features(patnos)
    sleep = extract_sleep_features(patnos)
    autonomic = extract_autonomic_features(patnos)
    genetic = extract_genetic_features(patnos)
    imaging = extract_imaging_features(patnos)
    csf = extract_csf_biomarkers(patnos)
    dat_spect = extract_dat_spect_features(patnos)

    # Merge all features onto endpoint data
    dataset = endpoints.copy()
    for feat_df in [
        clinical,
        cognitive,
        olfactory,
        sleep,
        autonomic,
        genetic,
        imaging,
        csf,
        dat_spect,
    ]:
        dataset = dataset.merge(feat_df, on="PATNO", how="left")

    # Build feature column list from config
    feature_cols = []
    for mod_name, mod_config in modalities.items():
        for feat in mod_config["features"]:
            if feat in dataset.columns:
                feature_cols.append(feat)
            else:
                print(f"  WARNING: Feature {feat} ({mod_name}) not found in data")
                dataset[feat] = np.nan
                feature_cols.append(feat)

    # Report missingness per modality
    print(f"\n{'=' * 60}")
    print("FEATURE MISSINGNESS REPORT")
    print(f"{'=' * 60}")
    for mod_name, mod_config in modalities.items():
        feats = mod_config["features"]
        n_patients = len(dataset)
        n_complete = dataset[feats].notna().all(axis=1).sum()
        pct = 100 * n_complete / n_patients
        print(f"  {mod_name:25s}: {n_complete:5d}/{n_patients} complete ({pct:5.1f}%)")
        for f in feats:
            missing = dataset[f].isna().sum()
            if missing > 0:
                print(
                    f"    {f:25s}: {missing:5d} missing ({100 * missing / n_patients:.1f}%)"
                )

    return dataset, feature_cols


def main() -> int:
    print("=" * 60)
    print("BUILD TRAINING DATASET FOR TRUE GIMAN")
    print("=" * 60 + "\n")

    # Build baseline 36-feature dataset
    dataset, feature_cols = assemble_dataset("config/feature_configs/baseline_36.yaml")

    # Save
    output_dir = project_root / "data" / "03_prodromal" / "final_training_dataset"
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / "prodromal_only_clean.csv"
    dataset.to_csv(output_path, index=False)

    print(f"\n{'=' * 60}")
    print("DATASET SUMMARY")
    print(f"{'=' * 60}")
    print(f"Output: {output_path}")
    print(f"Shape: {dataset.shape}")
    print(f"Patients: {len(dataset)}")
    print(
        f"Events: {dataset['phenoconverted'].sum()} ({dataset['phenoconverted'].mean():.1%})"
    )
    print(f"Features: {len(feature_cols)}")
    print(f"Overall missingness: {dataset[feature_cols].isna().mean().mean():.1%}")
    print("Endpoint source: 100% real PPMI")

    # Also build full clinical dataset
    print(f"\n\n{'=' * 60}")
    print("BUILDING FULL CLINICAL DATASET")
    print(f"{'=' * 60}\n")

    dataset_full, feature_cols_full = assemble_dataset(
        "config/feature_configs/full_clinical.yaml"
    )
    output_full = output_dir / "prodromal_full_clinical.csv"
    dataset_full.to_csv(output_full, index=False)
    print(f"\nFull clinical output: {output_full}")
    print(f"Shape: {dataset_full.shape}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
