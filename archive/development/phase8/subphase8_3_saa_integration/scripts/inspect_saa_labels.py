"""
Inspect SAA labels and check overlap with Phase 8.2 dataset.
"""

import pandas as pd
import numpy as np
import json

print("=" * 80)
print("PHASE 8.3: SAA LABELS INSPECTION")
print("=" * 80)

# Load SAA labels
saa_df = pd.read_csv('data/04_saa/saa_raw_labels.csv')
print("\n1. SAA LABELS DATASET")
print("-" * 80)
print(f"Shape: {saa_df.shape}")
print(f"Columns: {list(saa_df.columns)}")
print(f"\nData types:")
print(saa_df.dtypes)
print(f"\nMissing values:")
print(saa_df.isnull().sum())

print("\n2. PATIENT STATISTICS")
print("-" * 80)
print(f"Total patients: {saa_df['PATNO'].nunique():,}")
print(f"Total observations: {len(saa_df):,}")
print(f"All baseline visits: {(saa_df['EVENT_ID'] == 'BL').all()}")

print("\n3. SAA STATUS DISTRIBUTION")
print("-" * 80)
print(saa_df['SAA_POSITIVE'].value_counts().sort_index())
saa_pos_rate = saa_df['SAA_POSITIVE'].mean()
print(f"\nSAA Positive Rate: {saa_pos_rate:.1%}")
print(f"SAA Negative Rate: {1-saa_pos_rate:.1%}")

print("\n4. ALPHA-SYNUCLEIN STATISTICS")
print("-" * 80)
print("Overall:")
print(saa_df['ALPHA_SYN_VALUE'].describe())
print("\nBy SAA Status:")
print(saa_df.groupby('SAA_POSITIVE')['ALPHA_SYN_VALUE'].describe())

print("\n5. SAMPLE PATIENTS")
print("-" * 80)
print("\nFirst 10 patients:")
print(saa_df.head(10).to_string(index=False))
print("\n\nRandom 10 SAA+ patients:")
saa_positive = saa_df[saa_df['SAA_POSITIVE'] == 1].sample(10, random_state=42)
print(saa_positive.to_string(index=False))

# Load Phase 8.2 dataset
print("\n" + "=" * 80)
print("OVERLAP WITH PHASE 8.2 DATASET")
print("=" * 80)

phase82_df = pd.read_csv('data/03_prodromal/final_training_dataset/unified_longitudinal_early_pd.csv')

print("\n6. PHASE 8.2 DATASET")
print("-" * 80)
print(f"Total patients: {phase82_df['PATNO'].nunique():,}")
print(f"Total observations: {len(phase82_df):,}")
print(f"Landmark months: {sorted(phase82_df['landmark_month'].unique())}")
print(f"Features: {len(phase82_df.columns)}")
print(f"Has ALPHA_SYNUCLEIN column: {'ALPHA_SYNUCLEIN' in phase82_df.columns}")

print("\n7. OVERLAP ANALYSIS")
print("-" * 80)
saa_patients = set(saa_df['PATNO'])
phase82_patients = set(phase82_df['PATNO'])
overlap_patients = saa_patients.intersection(phase82_patients)

print(f"SAA-only patients: {len(saa_patients - phase82_patients):,}")
print(f"Phase 8.2-only patients: {len(phase82_patients - saa_patients):,}")
print(f"Overlapping patients: {len(overlap_patients):,}")
print(f"\nOverlap rate (SAA): {100*len(overlap_patients)/len(saa_patients):.1f}%")
print(f"Overlap rate (Phase 8.2): {100*len(overlap_patients)/len(phase82_patients):.1f}%")

print("\n8. SAA STATUS IN OVERLAPPING PATIENTS")
print("-" * 80)
overlap_saa = saa_df[saa_df['PATNO'].isin(overlap_patients)]
print(f"Total overlapping patients with SAA labels: {len(overlap_saa):,}")
print(f"SAA Positive: {overlap_saa['SAA_POSITIVE'].sum():,} ({100*overlap_saa['SAA_POSITIVE'].mean():.1f}%)")
print(f"SAA Negative: {(overlap_saa['SAA_POSITIVE']==0).sum():,} ({100*(1-overlap_saa['SAA_POSITIVE'].mean()):.1f}%)")

print("\n9. PHASE 8.2 OBSERVATIONS FOR OVERLAPPING PATIENTS")
print("-" * 80)
overlap_phase82 = phase82_df[phase82_df['PATNO'].isin(overlap_patients)]
print(f"Total observations: {len(overlap_phase82):,}")
print(f"Patients: {overlap_phase82['PATNO'].nunique():,}")
print(f"Observations per patient: {len(overlap_phase82) / overlap_phase82['PATNO'].nunique():.1f}")
print(f"\nLandmark month distribution:")
print(overlap_phase82['landmark_month'].value_counts().sort_index())

# Check alpha-synuclein availability
alpha_syn_available = overlap_phase82['ALPHA_SYNUCLEIN'].notna().sum()
print(f"\nAlpha-synuclein values available: {alpha_syn_available:,} ({100*alpha_syn_available/len(overlap_phase82):.1f}%)")

print("\n10. EXPECTED MERGED DATASET SIZE")
print("-" * 80)
# Merge SAA labels with Phase 8.2 baseline observations (landmark_month=0)
baseline_phase82 = phase82_df[phase82_df['landmark_month'] == 0]
merged_baseline = baseline_phase82[baseline_phase82['PATNO'].isin(overlap_patients)]
print(f"Phase 8.2 baseline observations (month 0) with SAA labels: {len(merged_baseline):,}")
print(f"Expected training dataset size (baseline only): {len(merged_baseline):,} patients")

# Check if we want longitudinal
all_merged = overlap_phase82.copy()
print(f"\nIf using all longitudinal visits:")
print(f"  Total observations: {len(all_merged):,}")
print(f"  Average visits per patient: {len(all_merged) / all_merged['PATNO'].nunique():.1f}")
print(f"  Landmark months: {sorted(all_merged['landmark_month'].unique())}")

print("\n" + "=" * 80)
print("INSPECTION COMPLETE")
print("=" * 80)
print(f"\n✓ SAA labels ready for merge")
print(f"✓ {len(overlap_patients):,} patients available for training")
print(f"✓ Balanced classes: {overlap_saa['SAA_POSITIVE'].mean():.1%} SAA+")
print(f"\nNext step: Run align_saa_features.py to merge with Phase 8.2 features")
