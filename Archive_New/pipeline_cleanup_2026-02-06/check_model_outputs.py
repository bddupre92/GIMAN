"""
Quick diagnostic to check model probability outputs.
"""
import json
import numpy as np

# Load test results
results_path = "outputs/phase8_3_saa/results/test_results.json"
with open(results_path, 'r') as f:
    results = json.load(f)

probs = np.array(results['probabilities'])
labels = np.array(results['labels'])
preds = np.array(results['predictions'])

print("=" * 80)
print("MODEL OUTPUT DIAGNOSTIC")
print("=" * 80)

print(f"\nProbability Statistics:")
print(f"  Min:    {probs.min():.6f}")
print(f"  Max:    {probs.max():.6f}")
print(f"  Mean:   {probs.mean():.6f}")
print(f"  Median: {np.median(probs):.6f}")
print(f"  Std:    {probs.std():.6f}")

print(f"\nProbability Distribution:")
print(f"  < 0.1:  {(probs < 0.1).sum()} samples")
print(f"  < 0.3:  {(probs < 0.3).sum()} samples")
print(f"  < 0.5:  {(probs < 0.5).sum()} samples")
print(f"  > 0.5:  {(probs > 0.5).sum()} samples")
print(f"  > 0.7:  {(probs > 0.7).sum()} samples")
print(f"  > 0.9:  {(probs > 0.9).sum()} samples")

print(f"\nBy True Label:")
print(f"  SAA- (n={(labels==0).sum()}): mean prob = {probs[labels==0].mean():.4f} ± {probs[labels==0].std():.4f}")
print(f"  SAA+ (n={(labels==1).sum()}): mean prob = {probs[labels==1].mean():.4f} ± {probs[labels==1].std():.4f}")

print(f"\nPredictions at threshold=0.5:")
print(f"  Predicted SAA-: {(preds==0).sum()}")
print(f"  Predicted SAA+: {(preds==1).sum()}")

# Check if all predictions are the same
if len(np.unique(preds)) == 1:
    print(f"\n⚠️  WARNING: All predictions are the same class!")
    print(f"  This means ALL probabilities are {'> 0.5' if preds[0] == 1 else '< 0.5'}")

# Show some example probabilities
print(f"\nFirst 20 probabilities:")
for i in range(min(20, len(probs))):
    print(f"  [{i}] prob={probs[i]:.4f}, label={labels[i]}, pred={preds[i]}")
