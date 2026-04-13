"""
RESEARCH PLAN PHASE 3 - TASK 3.1: MULTIMODAL FEATURE EXTRACTION
================================================================

Extracts and integrates multimodal features for Phase 1 cohort (2,046 patients):
1. DAT-SPECT striatal binding ratios (SBRs) - caudate, putamen
2. Grey matter volumetric features
3. CSF biomarkers (alpha-synuclein, tau, amyloid-beta)
4. Clinical trajectories (already in Phase 1)

Integrates with Phase 1 prognostic dataset to create comprehensive multimodal feature matrix.

Author: Research Plan Phase 3 Implementation
Date: October 2025
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Tuple, Optional
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MultimodalFeatureExtractor:
    """Extract and integrate multimodal features from PPMI data"""

    def __init__(self, ppmi_csv_dir: Path):
        """
        Initialize feature extractor.

        Args:
            ppmi_csv_dir: Path to PPMI CSV data directory
        """
        self.ppmi_csv_dir = Path(ppmi_csv_dir)
        self.features_dict = {}

        logger.info(f"MultimodalFeatureExtractor initialized with: {ppmi_csv_dir}")

    def extract_datscan_sbr_features(self) -> pd.DataFrame:
        """
        Extract DAT-SPECT striatal binding ratios.

        Returns:
            DataFrame with PATNO, EVENT_ID, and SBR features
        """
        logger.info("\n" + "="*80)
        logger.info("EXTRACTING DAT-SPECT SBR FEATURES")
        logger.info("="*80)

        # Load SBR data
        sbr_file = self.ppmi_csv_dir / "Xing_Core_Lab_-_Quant_SBR_18Sep2025.csv"

        if not sbr_file.exists():
            logger.warning(f"SBR file not found: {sbr_file}")
            return pd.DataFrame()

        df = pd.read_csv(sbr_file)
        logger.info(f"Loaded {len(df)} SBR records for {df['PATNO'].nunique()} patients")

        # Key SBR features (reference: cerebellum white matter)
        sbr_features = [
            # Bilateral striatum
            'STRIATUM_REF_CWM',      # Total striatum
            'CAUDATE_REF_CWM',       # Total caudate
            'PUTAMEN_REF_CWM',       # Total putamen

            # Left hemisphere
            'STRIATUM_L_REF_CWM',
            'CAUDATE_L_REF_CWM',
            'PUTAMEN_L_REF_CWM',

            # Right hemisphere
            'STRIATUM_R_REF_CWM',
            'CAUDATE_R_REF_CWM',
            'PUTAMEN_R_REF_CWM',

            # Regional subdivisions (left)
            'PRECAUDATE_L_REF_CWM',
            'POSCAUDATE_L_REF_CWM',
            'PRECOMMISSURAL_PUTAMEN_L_REF_CWM',
            'POSCOMMISSURAL_PUTAMEN_L_REF_CWM',

            # Regional subdivisions (right)
            'PRECAUDATE_R_REF_CWM',
            'POSCAUDATE_R_REF_CWM',
            'PRECOMMISSURAL_PUTAMEN_R_REF_CWM',
            'POSCOMMISSURAL_PUTAMEN_R_REF_CWM',
        ]

        # Check which features are available
        available_features = [f for f in sbr_features if f in df.columns]
        logger.info(f"Using {len(available_features)}/{len(sbr_features)} SBR features")

        # Select relevant columns
        sbr_df = df[['PATNO', 'EVENT_ID'] + available_features].copy()

        # Calculate asymmetry indices (L-R differences)
        if 'CAUDATE_L_REF_CWM' in df.columns and 'CAUDATE_R_REF_CWM' in df.columns:
            sbr_df['CAUDATE_ASYMMETRY'] = sbr_df['CAUDATE_L_REF_CWM'] - sbr_df['CAUDATE_R_REF_CWM']
            logger.info("Calculated caudate asymmetry index")

        if 'PUTAMEN_L_REF_CWM' in df.columns and 'PUTAMEN_R_REF_CWM' in df.columns:
            sbr_df['PUTAMEN_ASYMMETRY'] = sbr_df['PUTAMEN_L_REF_CWM'] - sbr_df['PUTAMEN_R_REF_CWM']
            logger.info("Calculated putamen asymmetry index")

        # Calculate caudate/putamen ratios (diagnostic marker)
        if 'CAUDATE_REF_CWM' in df.columns and 'PUTAMEN_REF_CWM' in df.columns:
            sbr_df['CAUDATE_PUTAMEN_RATIO'] = sbr_df['CAUDATE_REF_CWM'] / (sbr_df['PUTAMEN_REF_CWM'] + 1e-6)
            logger.info("Calculated caudate/putamen ratio")

        logger.info(f"\nFinal SBR features: {sbr_df.shape[1] - 2} features")
        logger.info(f"Coverage: {len(sbr_df)} records, {sbr_df['PATNO'].nunique()} patients")
        logger.info(f"EVENT_IDs: {sbr_df['EVENT_ID'].unique()}")

        self.features_dict['datscan_sbr'] = sbr_df
        return sbr_df

    def extract_grey_matter_volumes(self) -> pd.DataFrame:
        """
        Extract grey matter volume features.

        Returns:
            DataFrame with PATNO, EVENT_ID, and volume features
        """
        logger.info("\n" + "="*80)
        logger.info("EXTRACTING GREY MATTER VOLUME FEATURES")
        logger.info("="*80)

        gm_file = self.ppmi_csv_dir / "Grey_Matter_Volume_30Sep2025.csv"

        if not gm_file.exists():
            logger.warning(f"Grey matter file not found: {gm_file}")
            return pd.DataFrame()

        df = pd.read_csv(gm_file)
        logger.info(f"Loaded {len(df)} GM volume records for {df['PATNO'].nunique()} patients")

        gm_df = df[['PATNO', 'EVENT_ID', 'GM_VOLUME']].copy()

        logger.info(f"\nFinal GM features: {gm_df.shape[1] - 2} features")
        logger.info(f"Coverage: {len(gm_df)} records, {gm_df['PATNO'].nunique()} patients")

        self.features_dict['grey_matter'] = gm_df
        return gm_df

    def extract_csf_biomarkers(self) -> pd.DataFrame:
        """
        Extract CSF biomarker features.

        Returns:
            DataFrame with PATNO, EVENT_ID, and biomarker features
        """
        logger.info("\n" + "="*80)
        logger.info("EXTRACTING CSF BIOMARKER FEATURES")
        logger.info("="*80)

        # Try multiple CSF files
        csf_files = [
            "Current_Biospecimen_Analysis_Results_30Sep2025.csv",
            "PPMI_Project_9000_CSF_NEU_NPX_30Sep2025.csv",
            "PPMI_Project_222_CSF_NEU_NPX_30Sep2025.csv"
        ]

        csf_dfs = []

        for csv_file in csf_files:
            file_path = self.ppmi_csv_dir / csv_file
            if file_path.exists():
                try:
                    df = pd.read_csv(file_path)
                    logger.info(f"Loaded {csv_file}: {df.shape}")

                    # Check for key biomarker columns
                    biomarker_keywords = ['alpha', 'syn', 'tau', 'amyloid', 'abeta', 'ab42', 'ptau', 'ttau']
                    biomarker_cols = []
                    for col in df.columns:
                        if any(keyword in col.lower() for keyword in biomarker_keywords):
                            biomarker_cols.append(col)

                    if biomarker_cols and 'PATNO' in df.columns:
                        logger.info(f"  Found {len(biomarker_cols)} biomarker columns")

                        # Determine EVENT_ID column
                        event_col = None
                        for possible in ['EVENT_ID', 'EVENTID', 'EVENT', 'VISCODE']:
                            if possible in df.columns:
                                event_col = possible
                                break

                        if event_col:
                            csf_df = df[['PATNO', event_col] + biomarker_cols].copy()
                            csf_df = csf_df.rename(columns={event_col: 'EVENT_ID'})
                        else:
                            csf_df = df[['PATNO'] + biomarker_cols].copy()
                            csf_df['EVENT_ID'] = 'BL'  # Assume baseline if no event info

                        csf_dfs.append(csf_df)
                        logger.info(f"  Extracted {len(csf_df)} records, {csf_df['PATNO'].nunique()} patients")
                except Exception as e:
                    logger.warning(f"  Error loading {csv_file}: {e}")

        if csf_dfs:
            # Merge all CSF dataframes
            csf_combined = csf_dfs[0]
            for csf_df in csf_dfs[1:]:
                csf_combined = csf_combined.merge(
                    csf_df,
                    on=['PATNO', 'EVENT_ID'],
                    how='outer',
                    suffixes=('', '_dup')
                )

            # Remove duplicate columns
            csf_combined = csf_combined.loc[:, ~csf_combined.columns.str.endswith('_dup')]

            logger.info(f"\nFinal CSF features: {csf_combined.shape[1] - 2} features")
            logger.info(f"Coverage: {len(csf_combined)} records, {csf_combined['PATNO'].nunique()} patients")

            self.features_dict['csf_biomarkers'] = csf_combined
            return csf_combined
        else:
            logger.warning("No CSF biomarker data extracted")
            return pd.DataFrame()

    def integrate_with_phase1_cohort(self, phase1_df: pd.DataFrame,
                                     baseline_only: bool = True) -> pd.DataFrame:
        """
        Integrate multimodal features with Phase 1 prognostic cohort.

        Args:
            phase1_df: Phase 1 prognostic dataset (2,046 patients)
            baseline_only: If True, only use baseline features (default)

        Returns:
            Integrated multimodal dataset
        """
        logger.info("\n" + "="*80)
        logger.info("INTEGRATING MULTIMODAL FEATURES WITH PHASE 1 COHORT")
        logger.info("="*80)

        logger.info(f"Phase 1 cohort: {len(phase1_df)} patients")

        # Start with Phase 1 data
        integrated_df = phase1_df.copy()

        # Determine which EVENT_ID to use
        if baseline_only:
            event_id = 'BL'
            logger.info(f"Using baseline ({event_id}) features only")
        else:
            # Use whatever EVENT_ID is in Phase 1 (if it has that column)
            event_id = None

        # Merge each modality
        modalities_merged = 0

        # 1. DAT-SPECT SBR features
        if 'datscan_sbr' in self.features_dict:
            sbr_df = self.features_dict['datscan_sbr']

            if baseline_only:
                # Filter to baseline or screening
                sbr_bl = sbr_df[sbr_df['EVENT_ID'].isin(['BL', 'SC'])].copy()
                # If patient has multiple, take first one
                sbr_bl = sbr_bl.sort_values('EVENT_ID').groupby('PATNO').first().reset_index()
                sbr_bl = sbr_bl.drop(columns=['EVENT_ID'])
            else:
                sbr_bl = sbr_df

            # Merge
            before_cols = integrated_df.shape[1]
            integrated_df = integrated_df.merge(sbr_bl, on='PATNO', how='left', suffixes=('', '_sbr'))
            after_cols = integrated_df.shape[1]
            new_cols = after_cols - before_cols

            logger.info(f"  [+] DAT-SPECT SBR: Added {new_cols} features")
            logger.info(f"      Coverage: {integrated_df[sbr_bl.columns[1]].notna().sum()}/{len(integrated_df)} patients")
            modalities_merged += 1

        # 2. Grey matter volumes
        if 'grey_matter' in self.features_dict:
            gm_df = self.features_dict['grey_matter']

            if baseline_only:
                gm_bl = gm_df[gm_df['EVENT_ID'].isin(['BL', 'SC', 'V04'])].copy()
                gm_bl = gm_bl.groupby('PATNO').first().reset_index()
                gm_bl = gm_bl.drop(columns=['EVENT_ID'])
            else:
                gm_bl = gm_df

            before_cols = integrated_df.shape[1]
            integrated_df = integrated_df.merge(gm_bl, on='PATNO', how='left', suffixes=('', '_gm'))
            after_cols = integrated_df.shape[1]
            new_cols = after_cols - before_cols

            logger.info(f"  [+] Grey Matter Volume: Added {new_cols} features")
            logger.info(f"      Coverage: {integrated_df['GM_VOLUME'].notna().sum()}/{len(integrated_df)} patients")
            modalities_merged += 1

        # 3. CSF biomarkers
        if 'csf_biomarkers' in self.features_dict:
            csf_df = self.features_dict['csf_biomarkers']

            if baseline_only and 'EVENT_ID' in csf_df.columns:
                csf_bl = csf_df[csf_df['EVENT_ID'].isin(['BL', 'SC', 'V04'])].copy()
                csf_bl = csf_bl.groupby('PATNO').first().reset_index()
                csf_bl = csf_bl.drop(columns=['EVENT_ID'])
            else:
                csf_bl = csf_df.drop(columns=['EVENT_ID'], errors='ignore')

            before_cols = integrated_df.shape[1]
            integrated_df = integrated_df.merge(csf_bl, on='PATNO', how='left', suffixes=('', '_csf'))
            after_cols = integrated_df.shape[1]
            new_cols = after_cols - before_cols

            logger.info(f"  [+] CSF Biomarkers: Added {new_cols} features")
            modalities_merged += 1

        logger.info(f"\n{'='*80}")
        logger.info(f"INTEGRATION COMPLETE")
        logger.info(f"{'='*80}")
        logger.info(f"Modalities integrated: {modalities_merged}")
        logger.info(f"Total features: {integrated_df.shape[1]} columns")
        logger.info(f"Total patients: {len(integrated_df)} patients")

        return integrated_df

    def create_feature_summary(self, integrated_df: pd.DataFrame) -> Dict:
        """
        Create summary statistics for integrated multimodal features.

        Args:
            integrated_df: Integrated multimodal dataset

        Returns:
            Dictionary with feature summary statistics
        """
        logger.info("\n" + "="*80)
        logger.info("MULTIMODAL FEATURE SUMMARY")
        logger.info("="*80)

        summary = {
            'total_patients': len(integrated_df),
            'total_features': integrated_df.shape[1],
            'feature_groups': {},
            'missingness': {}
        }

        # Identify feature groups by suffix/prefix
        sbr_features = [c for c in integrated_df.columns if 'SBR' in c.upper() or 'STRIATUM' in c.upper() or 'CAUDATE' in c.upper() or 'PUTAMEN' in c.upper()]
        gm_features = [c for c in integrated_df.columns if 'GM_VOLUME' in c.upper()]
        csf_features = [c for c in integrated_df.columns if any(x in c.upper() for x in ['ALPHA', 'TAU', 'AMYLOID', 'ABETA'])]
        clinical_features = [c for c in integrated_df.columns if c in ['UPDRS_III_BL', 'MOCA_BL', 'AGE_APPROX', 'SEX', 'HANDED']]

        summary['feature_groups'] = {
            'clinical': len(clinical_features),
            'datscan_sbr': len(sbr_features),
            'grey_matter': len(gm_features),
            'csf_biomarkers': len(csf_features)
        }

        # Calculate missingness
        for group_name, features in [
            ('clinical', clinical_features),
            ('datscan_sbr', sbr_features),
            ('grey_matter', gm_features),
            ('csf_biomarkers', csf_features)
        ]:
            if features:
                missing_pct = integrated_df[features].isna().mean().mean() * 100
                coverage_pct = 100 - missing_pct
                summary['missingness'][group_name] = {
                    'missing_pct': missing_pct,
                    'coverage_pct': coverage_pct,
                    'n_features': len(features)
                }

        # Print summary
        for group, count in summary['feature_groups'].items():
            logger.info(f"{group:20s}: {count:3d} features")

        logger.info(f"\n{'Coverage Statistics:':40s}")
        for group, stats in summary['missingness'].items():
            logger.info(f"  {group:20s}: {stats['coverage_pct']:5.1f}% coverage ({stats['n_features']} features)")

        return summary


def run_multimodal_feature_extraction(
    ppmi_csv_dir: Path,
    phase1_dataset_path: Path,
    output_dir: Path
) -> Tuple[pd.DataFrame, Dict]:
    """
    Main pipeline: Extract multimodal features and integrate with Phase 1 cohort.

    Args:
        ppmi_csv_dir: Path to PPMI CSV directory
        phase1_dataset_path: Path to Phase 1 prognostic dataset
        output_dir: Output directory for results

    Returns:
        Tuple of (integrated_df, summary_stats)
    """
    logger.info("\n" + "="*80)
    logger.info("RESEARCH PLAN PHASE 3, TASK 3.1: MULTIMODAL FEATURE EXTRACTION")
    logger.info("="*80 + "\n")

    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize extractor
    extractor = MultimodalFeatureExtractor(ppmi_csv_dir)

    # Extract each modality
    sbr_df = extractor.extract_datscan_sbr_features()
    gm_df = extractor.extract_grey_matter_volumes()
    csf_df = extractor.extract_csf_biomarkers()

    # Load Phase 1 dataset
    logger.info(f"\nLoading Phase 1 dataset: {phase1_dataset_path}")
    phase1_df = pd.read_csv(phase1_dataset_path)
    logger.info(f"Phase 1 cohort: {len(phase1_df)} patients")

    # Integrate all modalities
    integrated_df = extractor.integrate_with_phase1_cohort(phase1_df, baseline_only=True)

    # Create feature summary
    summary = extractor.create_feature_summary(integrated_df)

    # Save outputs
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    output_file = output_dir / f"multimodal_dataset_{timestamp}.csv"
    integrated_df.to_csv(output_file, index=False)
    logger.info(f"\nSaved multimodal dataset: {output_file}")

    summary_file = output_dir / f"feature_summary_{timestamp}.txt"
    with open(summary_file, 'w') as f:
        f.write("MULTIMODAL FEATURE EXTRACTION SUMMARY\n")
        f.write("="*80 + "\n\n")
        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"Total Patients: {summary['total_patients']}\n")
        f.write(f"Total Features: {summary['total_features']}\n\n")
        f.write("Feature Groups:\n")
        for group, count in summary['feature_groups'].items():
            f.write(f"  {group:20s}: {count:3d} features\n")
        f.write("\nCoverage Statistics:\n")
        for group, stats in summary['missingness'].items():
            f.write(f"  {group:20s}: {stats['coverage_pct']:5.1f}% ({stats['n_features']} features)\n")

    logger.info(f"Saved feature summary: {summary_file}")

    logger.info("\n" + "="*80)
    logger.info("TASK 3.1 COMPLETE")
    logger.info("="*80 + "\n")

    return integrated_df, summary


if __name__ == "__main__":
    # Set up paths
    base_dir = Path(__file__).parent.parent.parent.parent  # Go up to CSCI FALL 2025
    ppmi_csv_dir = base_dir / "data" / "00_raw" / "GIMAN" / "ppmi_data_csv"
    phase1_dir = base_dir / "archive" / "development" / "phase1"
    output_dir = Path(__file__).parent / "multimodal_output"

    # Find most recent Phase 1 prognostic dataset
    phase1_files = list(phase1_dir.glob("prognostic_dataset_complete_*.csv"))
    if not phase1_files:
        raise FileNotFoundError(f"No Phase 1 dataset found in {phase1_dir}")

    phase1_dataset = max(phase1_files, key=lambda p: p.stat().st_mtime)

    logger.info(f"Phase 1 dataset: {phase1_dataset.name}")
    logger.info(f"PPMI CSV directory: {ppmi_csv_dir}")
    logger.info(f"Output directory: {output_dir}\n")

    # Run extraction
    integrated_df, summary = run_multimodal_feature_extraction(
        ppmi_csv_dir=ppmi_csv_dir,
        phase1_dataset_path=phase1_dataset,
        output_dir=output_dir
    )

    print(f"\nMultimodal dataset shape: {integrated_df.shape}")
    print(f"Output saved to: {output_dir}")
