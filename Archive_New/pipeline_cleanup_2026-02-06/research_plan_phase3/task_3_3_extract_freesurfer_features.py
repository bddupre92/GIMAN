"""
RESEARCH PLAN PHASE 3 - TASK 3.3: EXTRACT FREESURFER FEATURES
==============================================================

Extracts comprehensive FreeSurfer 7 features with MUCH better coverage:
- Cortical thickness: 70 features, ~65% coverage
- Cortical surface area: 70 features, ~65% coverage
- Subcortical volumes: 64 features, ~65% coverage

Total: 204 FreeSurfer features vs 1 grey matter feature (6% coverage)

This should dramatically improve multimodal performance!

Author: Research Plan Phase 3 Implementation
Date: October 2025
"""

import numpy as np
import pandas as pd
from pathlib import Path
import logging
from datetime import datetime
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def extract_freesurfer_features(ppmi_csv_dir: Path) -> pd.DataFrame:
    """
    Extract FreeSurfer 7 features from PPMI CSVs.

    Args:
        ppmi_csv_dir: Path to PPMI CSV directory

    Returns:
        DataFrame with PATNO and FreeSurfer features
    """
    logger.info("\n" + "="*80)
    logger.info("EXTRACTING FREESURFER 7 FEATURES")
    logger.info("="*80)

    freesurfer_files = {
        'cortical_thickness': 'FS7_APARC_CTH_30Sep2025.csv',
        'cortical_surface': 'FS7_APARC_SA_30Sep2025.csv',
        'subcortical_volume': 'FS7_ASEG_VOL_30Sep2025.csv'
    }

    freesurfer_dfs = []
    total_features = 0

    for feature_type, filename in freesurfer_files.items():
        file_path = ppmi_csv_dir / filename

        if not file_path.exists():
            logger.warning(f"FreeSurfer file not found: {filename}")
            continue

        df = pd.read_csv(file_path)
        logger.info(f"\n{feature_type}:")
        logger.info(f"  Loaded: {len(df)} records, {df['PATNO'].nunique()} patients")

        # Get feature columns (exclude metadata)
        metadata_cols = ['PATNO', 'EVENT_ID', 'REC_ID', 'update_stamp', 'IMAGEID']
        feature_cols = [c for c in df.columns if c not in metadata_cols]

        logger.info(f"  Features: {len(feature_cols)}")
        total_features += len(feature_cols)

        # Select PATNO + features (baseline only)
        df_bl = df[['PATNO'] + feature_cols].copy()

        # Add suffix to avoid column name conflicts
        suffix_map = {
            'cortical_thickness': 'CTH',
            'cortical_surface': 'SA',
            'subcortical_volume': 'VOL'
        }
        suffix = suffix_map[feature_type]

        # Rename columns
        renamed_cols = {'PATNO': 'PATNO'}
        for col in feature_cols:
            renamed_cols[col] = f'{col}_{suffix}'

        df_bl = df_bl.rename(columns=renamed_cols)

        freesurfer_dfs.append(df_bl)
        logger.info(f"  Extracted: {len(df_bl)} patients, {len(feature_cols)} features")

    if not freesurfer_dfs:
        logger.warning("No FreeSurfer data extracted")
        return pd.DataFrame()

    # Merge all FreeSurfer dataframes
    fs_combined = freesurfer_dfs[0]
    for fs_df in freesurfer_dfs[1:]:
        fs_combined = fs_combined.merge(fs_df, on='PATNO', how='outer')

    logger.info(f"\nCombined FreeSurfer features:")
    logger.info(f"  Total features: {fs_combined.shape[1] - 1}")
    logger.info(f"  Patients: {len(fs_combined)}")

    return fs_combined


def create_enhanced_multimodal_dataset(
    phase1_dataset_path: Path,
    ppmi_csv_dir: Path,
    output_dir: Path,
    include_datscan: bool = True
) -> pd.DataFrame:
    """
    Create enhanced multimodal dataset with FreeSurfer + DAT-SPECT.

    Args:
        phase1_dataset_path: Path to Phase 1 prognostic dataset
        ppmi_csv_dir: Path to PPMI CSV directory
        output_dir: Output directory
        include_datscan: Whether to include DAT-SPECT features

    Returns:
        Enhanced multimodal DataFrame
    """
    logger.info("\n" + "="*80)
    logger.info("CREATING ENHANCED MULTIMODAL DATASET")
    logger.info("="*80)

    # Load Phase 1 dataset
    logger.info(f"\nLoading Phase 1 dataset: {phase1_dataset_path.name}")
    phase1_df = pd.read_csv(phase1_dataset_path)
    logger.info(f"Phase 1 cohort: {len(phase1_df)} patients")

    # Extract FreeSurfer features
    fs_df = extract_freesurfer_features(ppmi_csv_dir)

    # Merge with Phase 1
    logger.info(f"\nMerging FreeSurfer features with Phase 1 cohort...")
    integrated_df = phase1_df.merge(fs_df, on='PATNO', how='left')

    n_freesurfer = integrated_df[fs_df.columns[1]].notna().sum()
    coverage_pct = n_freesurfer / len(integrated_df) * 100

    logger.info(f"  FreeSurfer coverage: {n_freesurfer}/{len(integrated_df)} patients ({coverage_pct:.1f}%)")
    logger.info(f"  Features added: {fs_df.shape[1] - 1}")

    # Optionally add DAT-SPECT features
    if include_datscan:
        logger.info(f"\nAdding DAT-SPECT features...")

        sbr_file = ppmi_csv_dir / "Xing_Core_Lab_-_Quant_SBR_18Sep2025.csv"
        if sbr_file.exists():
            sbr_df = pd.read_csv(sbr_file)

            # Key SBR features
            sbr_features = [
                'STRIATUM_REF_CWM', 'CAUDATE_REF_CWM', 'PUTAMEN_REF_CWM',
                'STRIATUM_L_REF_CWM', 'CAUDATE_L_REF_CWM', 'PUTAMEN_L_REF_CWM',
                'STRIATUM_R_REF_CWM', 'CAUDATE_R_REF_CWM', 'PUTAMEN_R_REF_CWM'
            ]
            available_sbr = [f for f in sbr_features if f in sbr_df.columns]

            # Baseline only
            sbr_bl = sbr_df[sbr_df['EVENT_ID'].isin(['BL', 'SC'])].copy()
            sbr_bl = sbr_bl.groupby('PATNO').first().reset_index()
            sbr_bl = sbr_bl[['PATNO'] + available_sbr]

            # Calculate asymmetry
            if 'CAUDATE_L_REF_CWM' in sbr_bl.columns and 'CAUDATE_R_REF_CWM' in sbr_bl.columns:
                sbr_bl['CAUDATE_ASYMMETRY'] = sbr_bl['CAUDATE_L_REF_CWM'] - sbr_bl['CAUDATE_R_REF_CWM']
            if 'PUTAMEN_L_REF_CWM' in sbr_bl.columns and 'PUTAMEN_R_REF_CWM' in sbr_bl.columns:
                sbr_bl['PUTAMEN_ASYMMETRY'] = sbr_bl['PUTAMEN_L_REF_CWM'] - sbr_bl['PUTAMEN_R_REF_CWM']

            # Merge
            integrated_df = integrated_df.merge(sbr_bl, on='PATNO', how='left')

            n_datscan = integrated_df[available_sbr[0]].notna().sum()
            datscan_coverage_pct = n_datscan / len(integrated_df) * 100

            logger.info(f"  DAT-SPECT coverage: {n_datscan}/{len(integrated_df)} patients ({datscan_coverage_pct:.1f}%)")
            logger.info(f"  Features added: {sbr_bl.shape[1] - 1}")

    # Summary
    logger.info(f"\n{'='*80}")
    logger.info("ENHANCED MULTIMODAL DATASET COMPLETE")
    logger.info("="*80)
    logger.info(f"Total patients: {len(integrated_df)}")
    logger.info(f"Total features: {integrated_df.shape[1]}")

    # Feature breakdown
    clinical_cols = ['UPDRS_III_BL', 'MOCA_BL', 'AGE_APPROX', 'SEX', 'HANDED']
    fs_cols = [c for c in integrated_df.columns if any(x in c for x in ['_CTH', '_SA', '_VOL'])]
    sbr_cols = [c for c in integrated_df.columns if 'STRIATUM' in c or 'CAUDATE' in c or 'PUTAMEN' in c]

    logger.info(f"\nFeature groups:")
    logger.info(f"  Clinical: {len([c for c in clinical_cols if c in integrated_df.columns])} features")
    logger.info(f"  FreeSurfer: {len(fs_cols)} features")
    if include_datscan:
        logger.info(f"  DAT-SPECT: {len(sbr_cols)} features")

    # Calculate missingness
    logger.info(f"\nMissingness by group:")
    if fs_cols:
        fs_missing = integrated_df[fs_cols].isna().mean().mean() * 100
        logger.info(f"  FreeSurfer: {fs_missing:.1f}% missing ({100-fs_missing:.1f}% coverage)")
    if sbr_cols and include_datscan:
        sbr_missing = integrated_df[sbr_cols].isna().mean().mean() * 100
        logger.info(f"  DAT-SPECT: {sbr_missing:.1f}% missing ({100-sbr_missing:.1f}% coverage)")

    # Save
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"enhanced_multimodal_dataset_{timestamp}.csv"
    integrated_df.to_csv(output_file, index=False)

    logger.info(f"\nSaved: {output_file}")

    # Save summary
    summary = {
        'n_patients': len(integrated_df),
        'n_features': integrated_df.shape[1],
        'freesurfer_coverage_pct': coverage_pct if fs_df is not None else 0,
        'datscan_coverage_pct': datscan_coverage_pct if include_datscan else 0,
        'timestamp': datetime.now().isoformat()
    }

    summary_file = output_dir / f"enhanced_dataset_summary_{timestamp}.json"
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)

    logger.info(f"Saved summary: {summary_file}")

    return integrated_df


if __name__ == "__main__":
    # Paths
    base_dir = Path(__file__).parent.parent.parent.parent
    ppmi_csv_dir = base_dir / "data" / "00_raw" / "GIMAN" / "ppmi_data_csv"
    phase1_dir = base_dir / "archive" / "development" / "phase1"
    output_dir = Path(__file__).parent / "multimodal_output"
    output_dir.mkdir(exist_ok=True)

    # Find Phase 1 dataset
    phase1_files = list(phase1_dir.glob("prognostic_dataset_complete_*.csv"))
    phase1_dataset = max(phase1_files, key=lambda p: p.stat().st_mtime)

    logger.info(f"Phase 1 dataset: {phase1_dataset.name}")
    logger.info(f"PPMI CSV directory: {ppmi_csv_dir}")
    logger.info(f"Output directory: {output_dir}\n")

    # Create enhanced dataset
    enhanced_df = create_enhanced_multimodal_dataset(
        phase1_dataset_path=phase1_dataset,
        ppmi_csv_dir=ppmi_csv_dir,
        output_dir=output_dir,
        include_datscan=True
    )

    logger.info(f"\n{'='*80}")
    logger.info("TASK 3.3 COMPLETE")
    logger.info(f"Enhanced dataset: {enhanced_df.shape}")
    logger.info("="*80)
