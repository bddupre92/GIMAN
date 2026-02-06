"""Check data availability for Phase 8.5 Multi-Task GIMAN."""

import pandas as pd
import json
from pathlib import Path

print("=" * 80)
print("PHASE 8.5 DATA AVAILABILITY CHECK")
print("=" * 80)

# Task 1: Progression (from Phase 8.2)
print("\n=== TASK 1: PROGRESSION PREDICTION ===")
try:
    surv_df = pd.read_csv('data/02_processed/progression_survival_data.csv')
    print(f"✓ Data found: {surv_df.shape}")
    print(f"  Columns: {surv_df.columns.tolist()}")
    print(f"  Events observed: {surv_df['event_observed'].sum()}/{len(surv_df)}")
    print(f"  Endpoint types: {surv_df['endpoint_type'].unique()}")
except Exception as e:
    print(f"✗ Error: {e}")

# Task 2: Conversion (from Phase 8.2 / Phase 5)
print("\n=== TASK 2: PHENOCONVERSION PREDICTION ===")
try:
    conv_df = pd.read_csv('data/02_processed/conversion_labels.csv')
    print(f"✓ Data found: {conv_df.shape}")
    print(f"  Columns: {conv_df.columns.tolist()}")
    print(f"  Converted: {conv_df['converted'].sum()}/{len(conv_df)}")
    print(f"  Conversion types: {conv_df['conversion_type'].value_counts().to_dict()}")
except Exception as e:
    print(f"✗ Error: {e}")

# Also check hybrid data
try:
    conv_hybrid = pd.read_csv('data/02_processed/progression_survival_data_hybrid.csv')
    print(f"✓ Hybrid progression data: {conv_hybrid.shape}")
    if 'phenoconverted' in conv_hybrid.columns:
        print(f"  Phenoconverted: {conv_hybrid['phenoconverted'].sum()}/{len(conv_hybrid)}")
except Exception as e:
    print(f"  No hybrid data: {e}")

# Task 3: SAA (from Phase 8.3)
print("\n=== TASK 3: SAA PREDICTION ===")
try:
    saa_df = pd.read_csv('data/04_saa/saa_training_data.csv')
    print(f"✓ Data found: {saa_df.shape}")
    print(f"  Columns (first 15): {saa_df.columns.tolist()[:15]}")
    print(f"  SAA Positive: {saa_df['SAA_POSITIVE'].sum()}/{len(saa_df)}")
    print(f"  Cohorts: {saa_df['cohort'].value_counts().to_dict()}")
except Exception as e:
    print(f"✗ Error: {e}")

# Task 4: Diagnostic (need to check cohort labels)
print("\n=== TASK 4: DIAGNOSTIC CLASSIFICATION ===")
try:
    # Check SAA data for cohort distribution
    print(f"  From SAA data:")
    print(f"  Cohorts: {saa_df['cohort'].value_counts().to_dict()}")
    
    # Check if we have a master cohort file
    try:
        master_df = pd.read_csv('data/02_processed/enhanced_real_ppmi_cohort.csv')
        print(f"\n  ✓ Master cohort file: {master_df.shape}")
        if 'COHORT' in master_df.columns:
            print(f"  Cohorts: {master_df['COHORT'].value_counts().to_dict()}")
        elif 'cohort' in master_df.columns:
            print(f"  Cohorts: {master_df['cohort'].value_counts().to_dict()}")
    except:
        print("  No master cohort file found")
        
except Exception as e:
    print(f"✗ Error: {e}")

# Check Phase 8.2 embeddings (used in Phase 8.4)
print("\n=== EMBEDDINGS (Phase 8.2 → Phase 8.4) ===")
try:
    emb_df = pd.read_csv('data/05_embeddings/giman_gat_embeddings.csv')
    print(f"✓ GAT embeddings: {emb_df.shape}")
    print(f"  Embedding columns: {[c for c in emb_df.columns if c.startswith('embedding_')][:5]}...")
    print(f"  Metadata columns: {[c for c in emb_df.columns if not c.startswith('embedding_')]}")
except Exception as e:
    print(f"✗ Error: {e}")

# Check trained models
print("\n=== TRAINED MODELS ===")

# Phase 8.2 model
phase82_model = Path('outputs/phase8_2_final_training/giman_survival_final.pth')
if phase82_model.exists():
    print(f"✓ Phase 8.2 Progression model: {phase82_model}")
    with open('outputs/phase8_2_final_training/training_results.json') as f:
        results = json.load(f)
        print(f"  Best test C-index: {results.get('best_test_c_index', 'N/A')}")
else:
    print("✗ Phase 8.2 model not found")

# Phase 8.3 model
phase83_model = Path('outputs/phase8_3_saa/models/best_giman_saa_model.pt')
if phase83_model.exists():
    print(f"✓ Phase 8.3 SAA model: {phase83_model}")
else:
    print("✗ Phase 8.3 model not found")

# Phase 8.4 VAE model
phase84_model = Path('archive/development/phase8/subphase8_4_vae_heterogeneity/checkpoints/best_vae_latent12.pth')
if phase84_model.exists():
    print(f"✓ Phase 8.4 VAE model: {phase84_model}")
else:
    print("✗ Phase 8.4 model not found")

print("\n" + "=" * 80)
print("SUMMARY FOR PHASE 8.5 MULTI-TASK GIMAN")
print("=" * 80)

summary = {
    "Task 1 (Progression)": "✓ Data available" if Path('data/02_processed/progression_survival_data.csv').exists() else "✗ Missing",
    "Task 2 (Conversion)": "✓ Data available" if Path('data/02_processed/conversion_labels.csv').exists() else "✗ Missing", 
    "Task 3 (SAA)": "✓ Data available" if Path('data/04_saa/saa_training_data.csv').exists() else "✗ Missing",
    "Task 4 (Diagnostic)": "? Need to extract from cohort labels",
    "Phase 8.2 Model": "✓ Available" if phase82_model.exists() else "✗ Missing",
    "Phase 8.3 Model": "✓ Available" if phase83_model.exists() else "✗ Missing",
}

for key, value in summary.items():
    print(f"  {key}: {value}")

print("\n" + "=" * 80)
