"""
Diagnostic analysis of GIMAN-SAA training issues.
"""

import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

print("=" * 80)
print("GIMAN-SAA TRAINING DIAGNOSTIC ANALYSIS")
print("=" * 80)

# Load results
with open('outputs/phase8_3_saa/results/test_results.json', 'r') as f:
    results = json.load(f)

print("\n1. ISSUE IDENTIFIED:")
print("   - Model predicts ALL samples as SAA+ (positive class)")
print("   - Confusion Matrix:")
print("     [[  0  74]  <- All SAA- predicted as SAA+")
print("      [  0  16]]  <- All SAA+ predicted as SAA+")
print("\n   - This indicates:")
print("     ✗ Model is not learning to discriminate between classes")
print("     ✗ Loss function may be dominated by positive class weight")
print("     ✗ Learning rate may be too high/low")
print("     ✗ Feature standardization may be needed")

print("\n2. ROOT CAUSES:")
print("   a) Class Imbalance: 82% SAA- vs 18% SAA+")
print("      - Weighted BCE may be over-compensating")
print("      - Current pos_weight calculated dynamically (n_neg/n_pos ≈ 4.5)")
print("\n   b) Feature Scaling Issues:")
print("      - Features may not be standardized")
print("      - Different modalities have different ranges")
print("\n   c) Learning Dynamics:")
print("      - Validation AUC peaked at 0.64 (epoch 31)")
print("      - But test AUC dropped to 0.57")
print("      - Suggests overfitting or poor generalization")

print("\n3. PROPOSED FIXES:")
print("   □ Fix 1: Add StandardScaler for feature normalization")
print("   □ Fix 2: Reduce pos_weight (try 2.0 instead of dynamic)")
print("   □ Fix 3: Add focal loss option for better class balance")
print("   □ Fix 4: Increase model capacity (hidden_dim 128→256)")
print("   □ Fix 5: Add L2 regularization to prevent overfitting")
print("   □ Fix 6: Use label smoothing to prevent overconfidence")
print("   □ Fix 7: Try different thresholds (not just 0.5)")

print("\n4. QUICK WIN - CHECK FEATURE DISTRIBUTIONS:")
df = pd.read_csv('data/04_saa/saa_training_data.csv')

# Check feature statistics
exclude_cols = ['PATNO', 'EVENT_ID', 'SAA_POSITIVE', 'ALPHA_SYN_VALUE',
                'time_to_event', 'phenoconverted', 'landmark_month',
                'original_time', 'original_event', 'cohort']
feature_cols = [col for col in df.columns if col not in exclude_cols]

print(f"\n   Feature statistics (first 10 features):")
for col in feature_cols[:10]:
    mean = df[col].mean()
    std = df[col].std()
    min_val = df[col].min()
    max_val = df[col].max()
    print(f"   {col:25s}: mean={mean:8.2f}, std={std:8.2f}, range=[{min_val:8.2f}, {max_val:8.2f}]")

print("\n   ⚠️  Features have very different scales!")
print("   → StandardScaler is REQUIRED")

print("\n5. RECOMMENDED ACTION PLAN:")
print("   1. Add StandardScaler to preprocessing")
print("   2. Reduce pos_weight to 2.0 (fixed)")
print("   3. Increase hidden_dim to 256")
print("   4. Add early stopping with min_delta=0.001")
print("   5. Re-train and evaluate")

print("\n" + "=" * 80)
print("Would you like me to implement these fixes?")
print("=" * 80)
