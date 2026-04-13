"""Comprehensive PPMI Data Integration for GIMAN Project.

This script loads all 21 CSV files, extracts key biomarker features,
and creates an enriched dataset with genetic, CSF, and non-motor features
for the patient similarity graph.

Key Biomarker Categories:
- Demographics: Age, Sex
- Genetic: LRRK2, GBA, APOE genotypes
- CSF: Abeta42, pTau, tTau, aSyn levels
- Non-Motor: UPSIT, SCOPA-AUT, RBD, ESS scores
- Motor: UPDRS-III, H&Y stage (targets)
- Cognitive: MoCA (target)
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class PPMIBiomarkerIntegrator:
    """Integrates biomarker data from all PPMI CSV files."""

    def __init__(self, data_dir: str):
        """Initialize with PPMI data directory path."""
        self.data_dir = Path(data_dir)
        self.dataframes = {}

    def load_all_csv_files(self) -> dict[str, pd.DataFrame]:
        """Load all CSV files from PPMI data directory."""
        logger.info("Loading all PPMI CSV files...")

        csv_files = list(self.data_dir.glob("*.csv"))
        logger.info(f"Found {len(csv_files)} CSV files")

        for csv_file in csv_files:
            try:
                # Create clean key from filename
                key = csv_file.stem.replace("_18Sep2025", "").replace(
                    "_20250515_18Sep2025", ""
                )
                key = key.replace(
                    "Current_Biospecimen_Analysis_Results", "CSF_Biomarkers"
                )
                key = key.replace("iu_genetic_consensus", "Genetics")
                key = key.replace(
                    "University_of_Pennsylvania_Smell_Identification_Test_UPSIT",
                    "UPSIT",
                )
                key = key.replace("REM_Sleep_Behavior_Disorder_Questionnaire", "RBD")
                key = key.replace("Epworth_Sleepiness_Scale", "ESS")
                key = key.replace("MDS-UPDRS_Part_III", "UPDRS_III")
                key = key.replace("MDS-UPDRS_Part_I", "UPDRS_I")
                key = key.replace("Montreal_Cognitive_Assessment__MoCA_", "MoCA")

                df = pd.read_csv(csv_file, low_memory=False)
                self.dataframes[key] = df
                logger.info(f"Loaded {key}: {df.shape}")

            except Exception as e:
                logger.error(f"Failed to load {csv_file}: {e}")

        return self.dataframes

    def extract_genetic_features(self) -> pd.DataFrame:
        """Extract genetic risk factors from genetics file."""
        logger.info("Extracting genetic features...")

        # Try multiple possible keys for genetics data
        genetics = None
        possible_keys = ["Genetics", "iu_genetic_consensus", "Genetics_20250515"]
        for key in possible_keys:
            if key in self.dataframes:
                genetics = self.dataframes[key]
                logger.info(f"Found genetics data with key: {key}")
                break

        if genetics is None:
            logger.warning("Genetics data not found")
            return pd.DataFrame()

        # Create genetic risk features
        genetic_features = genetics[["PATNO"]].copy()

        # LRRK2 mutations (binary: 0=wildtype, 1=mutation)
        genetic_features["LRRK2"] = (genetics["LRRK2"] != "0").astype(int)

        # GBA mutations (binary: 0=wildtype, 1=mutation)
        genetic_features["GBA"] = (genetics["GBA"] != "0").astype(int)

        # APOE risk (0=E2/E2,E2/E3, 1=E3/E3, 2=E3/E4,E4/E4)
        apoe_risk_map = {
            "E2/E2": 0,
            "E2/E3": 0,  # Protective
            "E3/E3": 1,  # Neutral
            "E3/E4": 2,
            "E4/E4": 2,
            "E2/E4": 2,  # Risk
        }
        genetic_features["APOE_RISK"] = genetics["APOE"].map(apoe_risk_map)

        logger.info(f"Genetic features extracted: {genetic_features.shape}")
        logger.info(f"LRRK2 mutations: {genetic_features['LRRK2'].sum()}")
        logger.info(f"GBA mutations: {genetic_features['GBA'].sum()}")
        logger.info(
            f"APOE risk distribution: {genetic_features['APOE_RISK'].value_counts().to_dict()}"
        )

        return genetic_features

    def extract_csf_biomarkers(self) -> pd.DataFrame:
        """Extract CSF biomarkers from biospecimen results."""
        logger.info("Extracting CSF biomarkers...")

        csf = self.dataframes.get("CSF_Biomarkers")
        if csf is None:
            logger.warning("CSF biomarkers data not found")
            return pd.DataFrame()

        # Key biomarkers to extract
        target_biomarkers = {
            "pTau": "PTAU",
            "tTau": "TTAU",
            # Need to check what's available for Abeta42 and aSyn
        }

        try:
            # Convert TESTVALUE to numeric, replacing non-numeric with NaN
            csf["TESTVALUE"] = pd.to_numeric(csf["TESTVALUE"], errors="coerce")

            # Pivot CSF data to get biomarkers as columns
            csf_pivot = csf.pivot_table(
                index=["PATNO", "CLINICAL_EVENT"],
                columns="TESTNAME",
                values="TESTVALUE",
                aggfunc="mean",  # Average if multiple measurements
            ).reset_index()

            # Check available biomarkers
            available_biomarkers = [
                col for col in target_biomarkers if col in csf_pivot.columns
            ]
            logger.info(f"Available CSF biomarkers: {available_biomarkers}")

            if not available_biomarkers:
                logger.warning("No target CSF biomarkers found")
                logger.info(
                    f"Available columns in CSF data: {list(csf_pivot.columns)[:10]}..."
                )
                return pd.DataFrame()

            # Filter to baseline visit and rename columns
            baseline_csf = csf_pivot[csf_pivot["CLINICAL_EVENT"] == "BL"].copy()
            csf_features = baseline_csf[["PATNO"] + available_biomarkers].copy()

            # Rename to standard names
            for old_name, new_name in target_biomarkers.items():
                if old_name in csf_features.columns:
                    csf_features = csf_features.rename(columns={old_name: new_name})

            logger.info(f"CSF features extracted: {csf_features.shape}")
            return csf_features

        except Exception as e:
            logger.error(f"Failed to extract CSF biomarkers: {e}")
            return pd.DataFrame()

    def extract_nonmotor_features(self) -> pd.DataFrame:
        """Extract non-motor clinical features."""
        logger.info("Extracting non-motor features...")

        nonmotor_features = None

        # 1. UPSIT (Smell test)
        upsit = self.dataframes.get("UPSIT")
        if upsit is not None:
            # Filter to baseline and extract total score
            upsit_bl = upsit[upsit["EVENT_ID"] == "BL"][
                ["PATNO", "TOTAL_CORRECT"]
            ].copy()
            upsit_bl = upsit_bl.rename(columns={"TOTAL_CORRECT": "UPSIT_TOTAL"})
            nonmotor_features = upsit_bl
            logger.info(f"UPSIT data: {len(upsit_bl)} patients")

        # 2. SCOPA-AUT (Autonomic dysfunction)
        scopa = self.dataframes.get("SCOPA-AUT")
        if scopa is not None:
            # Look for total score column
            score_cols = [col for col in scopa.columns if "TOTAL" in col.upper()]
            if score_cols:
                scopa_bl = scopa[scopa["EVENT_ID"] == "BL"][
                    ["PATNO"] + score_cols
                ].copy()
                scopa_bl = scopa_bl.rename(columns={score_cols[0]: "SCOPA_AUT_TOTAL"})

                if nonmotor_features is not None:
                    nonmotor_features = pd.merge(
                        nonmotor_features, scopa_bl, on="PATNO", how="outer"
                    )
                else:
                    nonmotor_features = scopa_bl
                logger.info(f"SCOPA-AUT data: {len(scopa_bl)} patients")

        # 3. RBD (REM sleep behavior disorder)
        rbd = self.dataframes.get("RBD")
        if rbd is not None:
            score_cols = [
                col
                for col in rbd.columns
                if "TOTAL" in col.upper() or "SCORE" in col.upper()
            ]
            if score_cols:
                rbd_bl = rbd[rbd["EVENT_ID"] == "BL"][["PATNO"] + score_cols[:1]].copy()
                rbd_bl = rbd_bl.rename(columns={score_cols[0]: "RBD_TOTAL"})

                if nonmotor_features is not None:
                    nonmotor_features = pd.merge(
                        nonmotor_features, rbd_bl, on="PATNO", how="outer"
                    )
                else:
                    nonmotor_features = rbd_bl
                logger.info(f"RBD data: {len(rbd_bl)} patients")

        # 4. ESS (Epworth Sleepiness Scale)
        ess = self.dataframes.get("ESS")
        if ess is not None:
            score_cols = [
                col
                for col in ess.columns
                if "TOTAL" in col.upper() or "SCORE" in col.upper()
            ]
            if score_cols:
                ess_bl = ess[ess["EVENT_ID"] == "BL"][["PATNO"] + score_cols[:1]].copy()
                ess_bl = ess_bl.rename(columns={score_cols[0]: "ESS_TOTAL"})

                if nonmotor_features is not None:
                    nonmotor_features = pd.merge(
                        nonmotor_features, ess_bl, on="PATNO", how="outer"
                    )
                else:
                    nonmotor_features = ess_bl
                logger.info(f"ESS data: {len(ess_bl)} patients")

        if nonmotor_features is not None:
            logger.info(f"Combined non-motor features: {nonmotor_features.shape}")
        else:
            logger.warning("No non-motor features extracted")
            nonmotor_features = pd.DataFrame()

        return nonmotor_features

    def extract_demographics_and_targets(self) -> pd.DataFrame:
        """Extract demographics and target variables."""
        logger.info("Extracting demographics and target variables...")

        # Demographics
        demographics = self.dataframes.get("Demographics")
        if demographics is None:
            logger.error("Demographics data not found!")
            return pd.DataFrame()

        demo_features = demographics[["PATNO", "SEX"]].copy()

        # Calculate age if birthdate available
        if "BIRTHDT" in demographics.columns and "INFODT" in demographics.columns:
            demo_features["AGE_COMPUTED"] = (
                pd.to_datetime(demographics["INFODT"])
                - pd.to_datetime(demographics["BIRTHDT"])
            ).dt.days / 365.25
        else:
            logger.warning("Cannot compute age - missing birth/info dates")

        # Cohort definition
        participant_status = self.dataframes.get("Participant_Status")
        if participant_status is not None:
            cohort_info = participant_status[["PATNO", "COHORT_DEFINITION"]].copy()
            demo_features = pd.merge(demo_features, cohort_info, on="PATNO", how="left")

        # Motor targets (UPDRS-III, H&Y)
        updrs3 = self.dataframes.get("UPDRS_III")
        if updrs3 is not None:
            updrs3_bl = updrs3[updrs3["EVENT_ID"] == "BL"]
            if "NP3TOT" in updrs3_bl.columns:
                motor_scores = updrs3_bl[["PATNO", "NP3TOT"]].copy()
                demo_features = pd.merge(
                    demo_features, motor_scores, on="PATNO", how="left"
                )

        # H&Y stage (if available)
        if updrs3 is not None and "NHY" in updrs3.columns:
            hy_scores = updrs3[updrs3["EVENT_ID"] == "BL"][["PATNO", "NHY"]].copy()
            demo_features = pd.merge(demo_features, hy_scores, on="PATNO", how="left")

        # Cognitive target (MoCA)
        moca = self.dataframes.get("MoCA")
        if moca is not None:
            moca_bl = moca[moca["EVENT_ID"] == "BL"]
            total_cols = [col for col in moca_bl.columns if "TOTAL" in col.upper()]
            if total_cols:
                cognitive_scores = moca_bl[["PATNO"] + total_cols[:1]].copy()
                cognitive_scores = cognitive_scores.rename(
                    columns={total_cols[0]: "MCATOT"}
                )
                demo_features = pd.merge(
                    demo_features, cognitive_scores, on="PATNO", how="left"
                )

        logger.info(f"Demographics and targets extracted: {demo_features.shape}")
        return demo_features

    def integrate_imaging_data(self, base_df: pd.DataFrame) -> pd.DataFrame:
        """Add imaging availability information."""
        logger.info("Adding imaging information...")

        # Load existing imaging manifest if available
        try:
            existing_data = pd.read_csv("data/01_processed/giman_dataset_final.csv")
            imaging_info = existing_data[
                ["PATNO", "nifti_conversions", "nifti_paths", "imaging_modalities"]
            ].copy()

            # Merge with base dataframe
            enriched_df = pd.merge(base_df, imaging_info, on="PATNO", how="left")
            logger.info(
                f"Added imaging info for {enriched_df['nifti_conversions'].notna().sum()} patients"
            )

            return enriched_df

        except FileNotFoundError:
            logger.warning(
                "No existing imaging data found - will need to process DICOM files"
            )
            base_df["nifti_conversions"] = np.nan
            base_df["nifti_paths"] = np.nan
            base_df["imaging_modalities"] = np.nan
            return base_df

    def create_enriched_dataset(
        self, output_path: str = "data/01_processed/giman_dataset_enriched.csv"
    ) -> pd.DataFrame:
        """Create enriched dataset with all biomarker features."""
        logger.info("=== Creating Enriched GIMAN Dataset ===")

        # Load all CSV files
        self.load_all_csv_files()

        # Extract feature categories
        demographics = self.extract_demographics_and_targets()
        genetic_features = self.extract_genetic_features()
        csf_features = self.extract_csf_biomarkers()
        nonmotor_features = self.extract_nonmotor_features()

        # Start with demographics as base
        enriched_df = demographics.copy()

        # Merge genetic features
        if not genetic_features.empty:
            enriched_df = pd.merge(
                enriched_df, genetic_features, on="PATNO", how="left"
            )
            logger.info("Merged genetic features")

        # Merge CSF features
        if not csf_features.empty:
            enriched_df = pd.merge(enriched_df, csf_features, on="PATNO", how="left")
            logger.info("Merged CSF features")

        # Merge non-motor features
        if not nonmotor_features.empty:
            enriched_df = pd.merge(
                enriched_df, nonmotor_features, on="PATNO", how="left"
            )
            logger.info("Merged non-motor features")

        # Add imaging information
        enriched_df = self.integrate_imaging_data(enriched_df)

        # Save enriched dataset
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        enriched_df.to_csv(output_path, index=False)

        # Report final dataset
        logger.info("=== ENRICHED DATASET COMPLETE ===")
        logger.info(f"Output saved to: {output_path}")
        logger.info(f"Total patients: {len(enriched_df)}")
        logger.info(f"Total features: {len(enriched_df.columns)}")
        logger.info(
            f"Multimodal patients: {enriched_df['nifti_conversions'].notna().sum()}"
        )

        logger.info("Feature categories:")
        biomarker_categories = {
            "Demographics": ["AGE_COMPUTED", "SEX", "COHORT_DEFINITION"],
            "Genetic": ["LRRK2", "GBA", "APOE_RISK"],
            "CSF": ["PTAU", "TTAU", "ABETA_42", "ASYN"],
            "Non-Motor": ["UPSIT_TOTAL", "SCOPA_AUT_TOTAL", "RBD_TOTAL", "ESS_TOTAL"],
            "Targets": ["NP3TOT", "NHY", "MCATOT"],
            "Imaging": ["nifti_conversions", "nifti_paths", "imaging_modalities"],
        }

        for category, features in biomarker_categories.items():
            available = [f for f in features if f in enriched_df.columns]
            logger.info(f"  {category}: {len(available)}/{len(features)} - {available}")

        return enriched_df


def main():
    """Main execution function."""
    data_dir = "data/00_raw/GIMAN/ppmi_data_csv"

    integrator = PPMIBiomarkerIntegrator(data_dir)
    enriched_dataset = integrator.create_enriched_dataset()

    print("\n🎯 BIOMARKER INTEGRATION COMPLETE!")
    print(f"   Dataset shape: {enriched_dataset.shape}")
    print(
        f"   Multimodal patients: {enriched_dataset['nifti_conversions'].notna().sum()}"
    )
    print("   Output: data/01_processed/giman_dataset_enriched.csv")


if __name__ == "__main__":
    main()
