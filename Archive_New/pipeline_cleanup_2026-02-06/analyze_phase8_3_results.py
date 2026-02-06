"""
Analyze Phase 8.3 training results and diagnose issues.
"""

import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

print("=" * 80)
print("PHASE 8.3 TRAINING RESULTS ANALYSIS")
print("=" * 80)

# Load results
results_path = "e:/My Drive/CSCI FALL 2025/outputs/phase8_3_saa/results/test_results.json"
with open(results_path, 'r') as f:
    results = json.load(f)

print("\nTest Set Performance:")
print(f"  AUC: {results['auc']:.4f}")
print(f"  Accuracy: {results['accuracy']:.4f}")
print(f"  Balanced Accuracy: {results['balanced_accuracy']:.4f}")

print("\nConfusion Matrix:")
cm = np.array(results['confusion_matrix'])
print(f"  True Negative (SAA-): {cm[0][0]}")
print(f"  False Positive (SAA- predicted as SAA+): {cm[0][1]}")
print(f"  False Negative (SAA+ predicted as SAA-): {cm[1][0]}")
print(f"  True Positive (SAA+): {cm[1][1]}")

print("\nClassification Report:")
report = results['classification_report']
for label in ['SAA-', 'SAA+']:
    if label in report:
        print(f"  {label}:")
        print(f"    Precision: {report[label]['precision']:.4f}")
        print(f"    Recall: {report[label]['recall']:.4f}")
        print(f"    F1-score: {report[label]['f1-score']:.4f}")

print("\n" + "=" * 80)
print("DIAGNOSIS")
print("=" * 80)

print("\nISSUE: Model predicting all samples as SAA+ (majority class)")
print("\nProbable Causes:")
print("  1. Class imbalance (82% SAA-, 18% SAA+)")
print("  2. Graph structure may be too connected")
print("  3. Need stronger regularization or different loss weighting")
print("  4. Small dataset (608 observations)")

print("\nRECOMMENDED FIXES:")
print("  ✓ Increase pos_weight in BCE loss (try 4.0 or 5.0)")
print("  ✓ Use focal loss instead of BCE")
print("  ✓ Add more aggressive data augmentation")
print("  ✓ Try different graph construction (e.g., k=5 instead of k=10)")
print("  ✓ Use class balancing (oversample SAA+ or undersample SAA-)")
print("  ✓ Add label smoothing")

# Load training data to check distribution
print("\n" + "=" * 80)
print("DATA DISTRIBUTION CHECK")
print("=" * 80)

df = pd.read_csv("data/04_saa/saa_training_data.csv")
print(f"\nTotal samples: {len(df)}")
print(f"SAA+ samples: {(df['SAA_POSITIVE']==1).sum()} ({100*(df['SAA_POSITIVE']==1).mean():.1f}%)")
print(f"SAA- samples: {(df['SAA_POSITIVE']==0).sum()} ({100*(df['SAA_POSITIVE']==0).mean():.1f}%)")
print(f"Imbalance ratio: {(df['SAA_POSITIVE']==0).sum() / (df['SAA_POSITIVE']==1).sum():.1f}:1")

# Check feature statistics
exclude_cols = ['PATNO', 'EVENT_ID', 'SAA_POSITIVE', 'ALPHA_SYN_VALUE',
                'time_to_event', 'phenoconverted', 'landmark_month',
                'original_time', 'original_event', 'cohort']
feature_cols = [col for col in df.columns if col not in exclude_cols]

print(f"\nFeature Statistics:")
print(f"  Number of features: {len(feature_cols)}")
print(f"  Feature value ranges:")

# Check for any problematic features
for col in feature_cols[:5]:  # Show first 5
    vals = df[col].dropna()
    print(f"    {col}: [{vals.min():.3f}, {vals.max():.3f}], mean={vals.mean():.3f}")

print("\n" + "=" * 80)
print("NEXT STEPS")
print("=" * 80)
print("\n1. Modify training script with fixes:")
print("   - Increase pos_weight to 5.0")
print("   - Add focal loss option")
print("   - Try k=5 for graph construction")
print("\n2. Consider data augmentation:")
print("   - SMOTE for minority class")
print("   - Feature perturbation")
print("\n3. Model architecture adjustments:")
print("   - Try simpler model (fewer layers)")
print("   - Add stronger dropout (0.5)")
print("   - Use attention regularization")

print("\n" + "=" * 80)
