"""
Phase 8.3: Align SAA Labels with Phase 8.2 Features

Simple, efficient script to merge SAA labels with Phase 8.2 multimodal features.

Author: GIMAN Research Team
Date: October 13, 2025
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path

print("=" * 80)
print("PHASE 8.3: SAA FEATURE ALIGNMENT")
print("=" * 80)

# ==============================================================================
# 1. LOAD DATA
# ==============================================================================
print("\n1. LOADING DATA")
print("-" * 80)

# Load SAA labels
saa_labels = pd.read_csv('data/04_saa/saa_raw_labels.csv')
print(f"✓ SAA Labels: {len(saa_labels)} patients")
print(f"  Columns: {list(saa_labels.columns)}")
print(f"  SAA+: {saa_labels['SAA_POSITIVE'].sum()} ({100*saa_labels['SAA_POSITIVE'].mean():.1f}%)")

# Load Phase 8.2 features
phase82 = pd.read_csv('data/03_prodromal/final_training_dataset/unified_longitudinal_early_pd.csv')
print(f"\n✓ Phase 8.2 Features: {len(phase82)} observations, {phase82['PATNO'].nunique()} patients")
print(f"  Columns: {len(phase82.columns)}")
print(f"  Landmark months: {sorted(phase82['landmark_month'].unique())}")

# ==============================================================================
# 2. FILTER TO BASELINE ONLY
# ==============================================================================
print("\n2. FILTERING TO BASELINE (MONTH 0)")
print("-" * 80)

baseline_phase82 = phase82[phase82['landmark_month'] == 0].copy()
print(f"✓ Baseline observations: {len(baseline_phase82)}")
print(f"✓ Baseline patients: {baseline_phase82['PATNO'].nunique()}")

# ==============================================================================
# 3. MERGE SAA LABELS WITH FEATURES
# ==============================================================================
print("\n3. MERGING SAA LABELS WITH FEATURES")
print("-" * 80)

# Inner join on PATNO (keep only patients with both SAA labels and features)
merged = pd.merge(
    saa_labels,
    baseline_phase82,
    on='PATNO',
    how='inner',
    suffixes=('_saa', '_phase82')
)

print(f"✓ Merged dataset: {len(merged)} patients")
print(f"✓ Total features: {len(merged.columns)}")

# Check SAA distribution
saa_dist = merged['SAA_POSITIVE'].value_counts().sort_index()
print(f"\nSAA Distribution:")
print(f"  SAA Negative: {saa_dist.get(0, 0)} ({100*saa_dist.get(0, 0)/len(merged):.1f}%)")
print(f"  SAA Positive: {saa_dist.get(1, 0)} ({100*saa_dist.get(1, 0)/len(merged):.1f}%)")

# ==============================================================================
# 4. ANALYZE MISSING DATA
# ==============================================================================
print("\n4. ANALYZING MISSING DATA")
print("-" * 80)

# Get feature columns (exclude identifiers and SAA labels)
id_cols = ['PATNO', 'EVENT_ID']
saa_cols = ['SAA_POSITIVE', 'ALPHA_SYN_VALUE']
metadata_cols = ['time_to_event', 'phenoconverted', 'landmark_month', 'original_time', 'original_event', 'cohort']

feature_cols = [col for col in merged.columns if col not in id_cols + saa_cols + metadata_cols]
print(f"Total feature columns: {len(feature_cols)}")

# Calculate missingness
missing_summary = {}
for col in feature_cols:
    n_missing = merged[col].isnull().sum()
    pct_missing = 100 * n_missing / len(merged)
    if pct_missing > 0:
        missing_summary[col] = {
            'count': int(n_missing),
            'percent': float(pct_missing)
        }

print(f"Features with missing data: {len(missing_summary)}")
print(f"Average missingness: {np.mean([s['percent'] for s in missing_summary.values()]):.1f}%")

# Show top 10 most missing
if missing_summary:
    print("\nTop 10 features with most missing data:")
    sorted_missing = sorted(missing_summary.items(), key=lambda x: x[1]['percent'], reverse=True)[:10]
    for col, stats in sorted_missing:
        print(f"  {col}: {stats['percent']:.1f}% ({stats['count']} missing)")

# ==============================================================================
# 5. HANDLE MISSING DATA
# ==============================================================================
print("\n5. HANDLING MISSING DATA")
print("-" * 80)

# Identify columns that are all missing indicators
missing_indicator_cols = [col for col in merged.columns if col.endswith('_missing')]
print(f"Missing indicator columns: {len(missing_indicator_cols)}")

# For features with missing data, keep as is (models can handle NaN)
# Note: PyG GCN can handle missing features via masked aggregation
print("Strategy: Keep NaN values (PyG models can handle missing data)")

# ==============================================================================
# 6. FEATURE GROUPING
# ==============================================================================
print("\n6. FEATURE GROUPS")
print("-" * 80)

feature_groups = {
    'clinical': [col for col in feature_cols if any(x in col.upper() for x in ['UPDRS', 'SCHWAB', 'PIGD', 'TREMOR'])],
    'genetics': [col for col in feature_cols if any(x in col.upper() for x in ['LRRK2', 'GBA', 'APOE', 'SNCA', 'GENETIC'])],
    'mri_volume': [col for col in feature_cols if 'VOL' in col.upper()],
    'mri_thickness': [col for col in feature_cols if 'CTH' in col.upper()],
    'dat_spect': [col for col in feature_cols if 'SBR' in col.upper() or 'ASYMMETRY' in col.upper()],
    'csf_biomarkers': [col for col in feature_cols if any(x in col.upper() for x in ['ALPHA_SYNUCLEIN', 'TAU', 'ABETA', 'PTAU'])],
    'clinical_biomarkers': [col for col in feature_cols if any(x in col.upper() for x in ['UPSIT', 'RBD', 'SCOPA', 'ESS'])]
}

for group_name, group_cols in feature_groups.items():
    if group_cols:
        print(f"  {group_name}: {len(group_cols)} features")

# ==============================================================================
# 7. SAVE OUTPUTS
# ==============================================================================
print("\n7. SAVING OUTPUTS")
print("-" * 80)

# Save merged dataset
output_path = Path('data/04_saa/saa_training_data.csv')
merged.to_csv(output_path, index=False)
print(f"✓ Saved training data: {output_path}")
print(f"  Shape: {merged.shape}")

# Save feature summary
summary = {
    'creation_date': '2025-10-13',
    'total_patients': int(len(merged)),
    'total_observations': int(len(merged)),
    'total_features': int(len(feature_cols)),
    'saa_positive_count': int(merged['SAA_POSITIVE'].sum()),
    'saa_negative_count': int((merged['SAA_POSITIVE'] == 0).sum()),
    'saa_positive_rate': float(merged['SAA_POSITIVE'].mean()),
    'feature_groups': {k: len(v) for k, v in feature_groups.items()},
    'missing_data_summary': missing_summary,
    'feature_columns': feature_cols,
    'id_columns': id_cols,
    'saa_columns': saa_cols,
    'metadata_columns': metadata_cols
}

summary_path = Path('data/04_saa/feature_summary.json')
with open(summary_path, 'w') as f:
    json.dump(summary, f, indent=2)
print(f"✓ Saved feature summary: {summary_path}")

# Save detailed report
report_path = Path('data/04_saa/alignment_report.txt')
with open(report_path, 'w') as f:
    f.write("=" * 80 + "\n")
    f.write("PHASE 8.3: SAA FEATURE ALIGNMENT REPORT\n")
    f.write("=" * 80 + "\n\n")
    
    f.write("SUMMARY\n")
    f.write("-" * 80 + "\n")
    f.write(f"Total patients: {len(merged):,}\n")
    f.write(f"Total features: {len(feature_cols):,}\n")
    f.write(f"SAA Positive: {merged['SAA_POSITIVE'].sum():,} ({100*merged['SAA_POSITIVE'].mean():.1f}%)\n")
    f.write(f"SAA Negative: {(merged['SAA_POSITIVE']==0).sum():,} ({100*(1-merged['SAA_POSITIVE'].mean()):.1f}%)\n\n")
    
    f.write("FEATURE GROUPS\n")
    f.write("-" * 80 + "\n")
    for group_name, group_cols in feature_groups.items():
        if group_cols:
            f.write(f"\n{group_name.upper()} ({len(group_cols)} features):\n")
            for col in sorted(group_cols):
                f.write(f"  - {col}\n")
    
    f.write("\n" + "=" * 80 + "\n")
    f.write("READY FOR GIMAN-SAA TRAINING\n")
    f.write("=" * 80 + "\n")

print(f"✓ Saved alignment report: {report_path}")

# ==============================================================================
# 8. FINAL SUMMARY
# ==============================================================================
print("\n" + "=" * 80)
print("ALIGNMENT COMPLETE")
print("=" * 80)
print(f"\n✅ Successfully merged {len(merged)} patients")
print(f"✅ {len(feature_cols)} features ready for training")
print(f"✅ Class balance: {100*merged['SAA_POSITIVE'].mean():.1f}% SAA+")
print(f"\nOutput files:")
print(f"  • {output_path}")
print(f"  • {summary_path}")
print(f"  • {report_path}")
print(f"\n🎯 Ready for GIMAN-SAA model training!")
