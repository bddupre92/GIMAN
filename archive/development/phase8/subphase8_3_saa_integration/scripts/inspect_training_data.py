"""Quick inspection of SAA training dataset."""
import pandas as pd
import numpy as np

print("=" * 80)
print("SAA TRAINING DATASET INSPECTION")
print("=" * 80)

# Load data
df = pd.read_csv('data/04_saa/saa_training_data.csv')

print(f"\n1. DATASET SHAPE")
print(f"   Rows (observations): {len(df):,}")
print(f"   Columns (features): {len(df.columns):,}")
print(f"   Unique patients: {df['PATNO'].nunique():,}")

print(f"\n2. COLUMN STRUCTURE")
print(f"   First 10 columns:")
for i, col in enumerate(df.columns[:10], 1):
    print(f"      {i:2d}. {col}")
print(f"   ...")
print(f"   Last 5 columns:")
for i, col in enumerate(df.columns[-5:], len(df.columns)-4):
    print(f"      {i:2d}. {col}")

print(f"\n3. SAA STATUS DISTRIBUTION")
saa_dist = df['SAA_POSITIVE'].value_counts().sort_index()
print(f"   SAA Negative (0): {saa_dist.get(0, 0):,} ({100*saa_dist.get(0, 0)/len(df):.1f}%)")
print(f"   SAA Positive (1): {saa_dist.get(1, 0):,} ({100*saa_dist.get(1, 0)/len(df):.1f}%)")

print(f"\n4. MISSING VALUES")
missing = df.isnull().sum()
missing_sorted = missing[missing > 0].sort_values(ascending=False)
if len(missing_sorted) > 0:
    print(f"   Top 10 features with missing values:")
    for col, count in missing_sorted.head(10).items():
        pct = 100 * count / len(df)
        print(f"      {col}: {count:,} ({pct:.1f}%)")
else:
    print(f"   ✓ NO MISSING VALUES!")

print(f"\n5. DATA TYPES")
dtypes_count = df.dtypes.value_counts()
for dtype, count in dtypes_count.items():
    print(f"   {dtype}: {count} columns")

print(f"\n6. SAMPLE DATA (First 3 Patients)")
key_cols = ['PATNO', 'SAA_POSITIVE', 'LRRK2', 'GBA', 'UPDRS_I', 'PUTAMEN_L_SBR', 'ALPHA_SYNUCLEIN']
available_key_cols = [col for col in key_cols if col in df.columns]
print(df[available_key_cols].head(3).to_string(index=False))

print(f"\n7. NUMERIC FEATURE STATISTICS")
numeric_cols = df.select_dtypes(include=[np.number]).columns
print(f"   Total numeric features: {len(numeric_cols)}")
print(f"   Sample statistics for ALPHA_SYNUCLEIN:")
if 'ALPHA_SYNUCLEIN' in df.columns:
    print(df['ALPHA_SYNUCLEIN'].describe())
else:
    print("   ALPHA_SYNUCLEIN column not found")

print("\n" + "=" * 80)
print("DATASET READY FOR TRAINING!")
print("=" * 80)
