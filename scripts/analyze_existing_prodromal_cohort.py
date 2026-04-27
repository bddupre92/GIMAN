"""
Quick Analysis of Existing Prodromal Cohort

Analyzes the pre-extracted prodromal cohort (n=381) to confirm it's
suitable for Phase 8.1 GIMAN training.

Outputs:
- Cohort summary statistics
- Event distribution
- Feature availability assessment
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

# Configuration
DATA_DIR = Path("data/prodromal_cohort")
OUTPUT_DIR = Path("Docs")

def main():
    """Analyze existing prodromal cohort."""
    
    print("\n" + "=" * 70)
    print("ANALYZING EXISTING PRODROMAL COHORT")
    print("=" * 70)
    
    # Load prodromal survival data
    survival_file = DATA_DIR / "prodromal_survival_data.csv"
    df = pd.read_csv(survival_file)
    
    print(f"\n✓ Loaded prodromal cohort: {len(df)} patients")
    print(f"  Columns: {list(df.columns)}")
    
    # Event statistics
    n_converted = df['phenoconverted'].sum()
    conversion_rate = n_converted / len(df) * 100
    
    print(f"\n" + "=" * 70)
    print("PHENOCONVERSION EVENTS")
    print("=" * 70)
    print(f"  Total patients: {len(df)}")
    print(f"  Phenoconverted: {n_converted} ({conversion_rate:.1f}%)")
    print(f"  Censored: {len(df) - n_converted} ({100 - conversion_rate:.1f}%)")
    print(f"  Mean follow-up: {df['time_to_event'].mean():.1f} ± {df['time_to_event'].std():.1f} months")
    print(f"  Median follow-up: {df['time_to_event'].median():.1f} months")
    
    # Demographics
    print(f"\n" + "=" * 70)
    print("DEMOGRAPHICS")
    print("=" * 70)
    print(f"  Age: {df['age_approx'].mean():.1f} ± {df['age_approx'].std():.1f} years")
    print(f"  Sex: {(df['sex'] == 1).sum()} male, {(df['sex'] == 0).sum()} female")
    
    # Clinical features
    print(f"\n" + "=" * 70)
    print("BASELINE CLINICAL FEATURES")
    print("=" * 70)
    print(f"  UPDRS: {df['baseline_updrs'].mean():.1f} ± {df['baseline_updrs'].std():.1f}")
    print(f"    Available: {df['baseline_updrs'].notna().sum()}/{len(df)} ({df['baseline_updrs'].notna().mean()*100:.1f}%)")
    print(f"  MoCA: {df['baseline_moca'].mean():.1f} ± {df['baseline_moca'].std():.1f}")
    print(f"    Available: {df['baseline_moca'].notna().sum()}/{len(df)} ({df['baseline_moca'].notna().mean()*100:.1f}%)")
    
    # Comparison: Converters vs Non-converters
    converters = df[df['phenoconverted'] == 1]
    non_converters = df[df['phenoconverted'] == 0]
    
    print(f"\n" + "=" * 70)
    print("CONVERTERS VS NON-CONVERTERS COMPARISON")
    print("=" * 70)
    print(f"\nAge:")
    print(f"  Converters: {converters['age_approx'].mean():.1f} ± {converters['age_approx'].std():.1f} years")
    print(f"  Non-converters: {non_converters['age_approx'].mean():.1f} ± {non_converters['age_approx'].std():.1f} years")
    
    print(f"\nBaseline UPDRS:")
    print(f"  Converters: {converters['baseline_updrs'].mean():.1f} ± {converters['baseline_updrs'].std():.1f}")
    print(f"  Non-converters: {non_converters['baseline_updrs'].mean():.1f} ± {non_converters['baseline_updrs'].std():.1f}")
    
    print(f"\nBaseline MoCA:")
    print(f"  Converters: {converters['baseline_moca'].mean():.1f} ± {converters['baseline_moca'].std():.1f}")
    print(f"  Non-converters: {non_converters['baseline_moca'].mean():.1f} ± {non_converters['baseline_moca'].std():.1f}")
    
    # Load report JSON for additional statistics
    report_file = DATA_DIR / "prodromal_cohort_report.json"
    if report_file.exists():
        with open(report_file, 'r') as f:
            report = json.load(f)
        
        print(f"\n" + "=" * 70)
        print("STATISTICAL TESTS (from cohort report)")
        print("=" * 70)
        
        baseline_comp = report['cohort_statistics']['baseline_comparisons']
        
        print(f"\nAge difference:")
        print(f"  p-value: {baseline_comp['age']['p_value']:.4f}")
        print(f"  Significant: {baseline_comp['age']['significant']}")
        
        print(f"\nBaseline UPDRS difference:")
        print(f"  p-value: {baseline_comp['baseline_updrs']['p_value']:.2e}")
        print(f"  Significant: {baseline_comp['baseline_updrs']['significant']}")
        
        print(f"\nKaplan-Meier survival:")
        km = report['cohort_statistics']['kaplan_meier']
        print(f"  12-month survival: {km['survival_at_12mo']*100:.1f}%")
        print(f"  24-month survival: {km['survival_at_24mo']*100:.1f}%")
    
    # Suitability assessment
    print(f"\n" + "=" * 70)
    print("PHASE 8.1 SUITABILITY ASSESSMENT")
    print("=" * 70)
    
    # Check against target criteria
    criteria = {
        'Sample size ≥150': len(df) >= 150,
        'Real phenoconversion events ≥10': n_converted >= 10,
        'Conversion rate >2%': conversion_rate > 2,
        'Follow-up ≥12 months': df['time_to_event'].median() >= 12,
        'Clinical features available': df['baseline_updrs'].notna().mean() > 0.5,
    }
    
    print("\nCriteria checklist:")
    for criterion, met in criteria.items():
        status = "✓" if met else "✗"
        print(f"  {status} {criterion}: {'PASS' if met else 'FAIL'}")
    
    all_met = all(criteria.values())
    
    print(f"\n{'=' * 70}")
    if all_met:
        print("✓ PRODROMAL COHORT SUITABLE FOR PHASE 8.1 TRAINING")
        print("  All criteria met. Proceed to Task 11: Prepare Training Data")
    else:
        print("⚠️  PRODROMAL COHORT MAY NEED ENRICHMENT")
        print("  Some criteria not met. Consider data augmentation or hybrid approach")
    print("=" * 70)
    
    # Comparison to manifest PD cohort
    print(f"\n" + "=" * 70)
    print("COMPARISON TO MANIFEST PD COHORT (Week 4)")
    print("=" * 70)
    
    print("\nCohort characteristics:")
    print(f"  Sample size:")
    print(f"    Manifest PD: 127 patients")
    print(f"    Prodromal: {len(df)} patients (+{len(df)-127} = {(len(df)-127)/127*100:.0f}% larger)")
    
    print(f"\n  Real events:")
    print(f"    Manifest PD: 3 progression events")
    print(f"    Prodromal: {n_converted} phenoconversion events (+{n_converted-3} = {(n_converted-3)/3*100:.0f}% more)")
    
    print(f"\n  Event rate:")
    print(f"    Manifest PD: 29.9% (hybrid enriched)")
    print(f"    Prodromal: {conversion_rate:.1f}% (real events only)")
    
    print(f"\n  Follow-up:")
    print(f"    Manifest PD: 2.3 ± 0.9 years")
    print(f"    Prodromal: {df['time_to_event'].mean()/12:.1f} ± {df['time_to_event'].std()/12:.1f} years")
    
    print(f"\n  Age:")
    print(f"    Manifest PD: 61 ± 9 years (post-diagnosis)")
    print(f"    Prodromal: {df['age_approx'].mean():.0f} ± {df['age_approx'].std():.0f} years (pre-diagnosis)")
    
    print(f"\nPredicted performance comparison:")
    print(f"  Manifest PD test C-index: 0.38 [0.19, 0.69]")
    print(f"  Prodromal target C-index: ≥0.55 (improved generalization expected)")
    print(f"  Rationale:")
    print(f"    - More real events (15 vs 3) → Better signal")
    print(f"    - No simulation dependency → True performance")
    print(f"    - Longer follow-up (24 vs 27 months) → More mature endpoints")
    
    print(f"\n{'=' * 70}")
    print("NEXT STEPS")
    print("=" * 70)
    print(f"\n1. Create training data preparation script")
    print(f"   - Merge prodromal cohort with enhanced PPMI features")
    print(f"   - Extract same 32 features as manifest PD")
    print(f"   - Construct similarity graph (k=10)")
    print(f"   - Stratified split 70/15/15 (train ~267, val ~57, test ~57)")
    print(f"   - Expected split: train ~10-11 events, val ~2, test ~2-3")
    
    print(f"\n2. Train GIMAN-Prognostic model")
    print(f"   - Identical architecture (32 features, 3 GAT layers, 4 heads)")
    print(f"   - CoxPH loss for time-to-event")
    print(f"   - Early stopping on validation C-index")
    print(f"   - Bootstrap CI evaluation (1000 samples)")
    
    print(f"\n3. Generate visualizations & reports")
    print(f"   - 5 publication figures (same format as Week 4)")
    print(f"   - Comparative analysis manifest PD vs prodromal")
    print(f"   - Phase 8.1 completion report")
    print(f"\n{'=' * 70}\n")


if __name__ == "__main__":
    main()
