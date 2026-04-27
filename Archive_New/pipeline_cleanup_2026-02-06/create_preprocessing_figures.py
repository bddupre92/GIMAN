import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

# Paths
base_dir = Path("/Users/blair.dupre/Library/CloudStorage/GoogleDrive-dupre.blair92@gmail.com/My Drive/CSCI FALL 2025")

# Verified data sources
# Raw longitudinal data (Before Imputation)
raw_long_path = base_dir / "archive/development/phase1/longitudinal_cohort_full_20251002_202222.csv"

# Augmented data (After MICE Imputation)
augmented_path = base_dir / "archive/development/phase1/longitudinal_cohort_augmented_20251002_203324.csv"

# Enhanced data (For PRS)
enhanced_path = base_dir / "data/enhanced/enhanced_giman_12features_v1.1.0_20250924_075919.csv"

# SPECT data for normalization
spect_path = base_dir / "data/01_processed/dat_spect_sbr_values.csv"

output_dir = base_dir / "results/preprocessing_figures"
output_dir.mkdir(parents=True, exist_ok=True)

print(f"Python executable: {sys.executable}")
print("Loading data...")

try:
    raw_df = pd.read_csv(raw_long_path)
    print(f"Raw shape: {raw_df.shape}")
except FileNotFoundError:
    print(f"Error: Raw file not found at {raw_long_path}")
    sys.exit(1)

try:
    augmented_df = pd.read_csv(augmented_path)
    print(f"Augmented shape: {augmented_df.shape}")
except FileNotFoundError:
    print(f"Error: Augmented file not found at {augmented_path}")
    sys.exit(1)

try:
    enhanced_df = pd.read_csv(enhanced_path)
    print(f"Enhanced shape: {enhanced_df.shape}")
except FileNotFoundError:
    print(f"Error: Enhanced file not found at {enhanced_path}")
    sys.exit(1)

try:
    spect_df = pd.read_csv(spect_path)
    print(f"SPECT shape: {spect_df.shape}")
except FileNotFoundError:
    print(f"Warning: SPECT file not found at {spect_path}. Normalization figure will be simulated.")
    spect_df = pd.DataFrame()

# --- Figure 1: MICE Imputation Visualization ---
print("Generating MICE Imputation Figure...")
fig1, axes1 = plt.subplots(1, 2, figsize=(15, 6))

# We want to compare observed V08 vs Imputed V08
# Raw df has UPDRS_III_V08 with NaNs
# Augmented df has UPDRS_III_V08 filled, and likely a flag 'V08_IMPUTED'

if 'UPDRS_III_V08' in raw_df.columns and 'UPDRS_III_V08' in augmented_df.columns:
    # Observed values (from raw, dropping NaNs)
    observed_v08 = raw_df['UPDRS_III_V08'].dropna()
    
    # Imputed values (from augmented, selecting only those that were imputed)
    # Check if we have an imputation flag
    if 'V08_IMPUTED' in augmented_df.columns:
        imputed_v08 = augmented_df[augmented_df['V08_IMPUTED'] == True]['UPDRS_III_V08']
    else:
        # Fallback: Find rows where raw is NaN but augmented is not
        # Need to align by PATNO
        merged = pd.merge(raw_df[['PATNO', 'UPDRS_III_V08']], 
                          augmented_df[['PATNO', 'UPDRS_III_V08']], 
                          on='PATNO', suffixes=('_raw', '_aug'))
        imputed_v08 = merged[merged['UPDRS_III_V08_raw'].isna() & merged['UPDRS_III_V08_aug'].notna()]['UPDRS_III_V08_aug']

    sns.kdeplot(data=observed_v08, ax=axes1[0], fill=True, label='Observed (Original)', color='blue', alpha=0.3)
    if not imputed_v08.empty:
        sns.kdeplot(data=imputed_v08, ax=axes1[0], fill=True, label='Imputed (MICE)', color='orange', alpha=0.3)
    else:
        axes1[0].text(0.5, 0.5, "No Imputed Values Found", ha='center')
        
    axes1[0].set_title("Distribution of UPDRS III (Year 3): Observed vs. Imputed")
    axes1[0].set_xlabel("MDS-UPDRS Part III Score")
    axes1[0].legend()
    
    # Scatter plot: Baseline vs Year 3 (to show relationship preservation)
    # Use augmented df
    if 'UPDRS_III_BL' in augmented_df.columns:
        # Plot observed points
        observed_mask = ~augmented_df['V08_IMPUTED'] if 'V08_IMPUTED' in augmented_df.columns else augmented_df.index.isin(raw_df.dropna(subset=['UPDRS_III_V08']).index)
        
        sns.scatterplot(data=augmented_df[observed_mask], x='UPDRS_III_BL', y='UPDRS_III_V08', 
                        ax=axes1[1], alpha=0.3, color='blue', label='Observed')
        
        # Plot imputed points
        imputed_mask = ~observed_mask
        if imputed_mask.any():
            sns.scatterplot(data=augmented_df[imputed_mask], x='UPDRS_III_BL', y='UPDRS_III_V08', 
                            ax=axes1[1], alpha=0.6, color='orange', marker='x', label='Imputed')
            
        axes1[1].set_title("Longitudinal Trajectory Preservation: Baseline vs Year 3")
        axes1[1].set_xlabel("Baseline UPDRS III")
        axes1[1].set_ylabel("Year 3 UPDRS III")
        axes1[1].legend()

else:
    print("UPDRS_III_V08 column not found.")
    axes1[0].text(0.5, 0.5, "Data Not Found", ha='center')

plt.tight_layout()
plt.savefig(output_dir / "preprocessing_mice_imputation.png", dpi=300)
plt.close()


# --- Figure 2: Z-Score Normalization ---
print("Generating Normalization Figure...")
fig2, axes2 = plt.subplots(1, 2, figsize=(15, 6))

if not spect_df.empty:
    # Look for SBR columns
    spect_cols = [c for c in spect_df.columns if 'CAUDATE' in c or 'PUTAMEN' in c]
    if spect_cols:
        raw_vals = spect_df[spect_cols[0]].dropna()
        
        # Z-score normalize
        norm_vals = (raw_vals - raw_vals.mean()) / raw_vals.std()
        
        sns.histplot(raw_vals, ax=axes2[0], kde=True, color='purple', bins=30)
        axes2[0].set_title(f"Raw DAT-SPECT SBR ({spect_cols[0]})")
        axes2[0].set_xlabel("Specific Binding Ratio (SBR)")
        
        sns.histplot(norm_vals, ax=axes2[1], kde=True, color='green', bins=30)
        axes2[1].set_title("Z-Score Normalized Intensity")
        axes2[1].set_xlabel("Z-Score")
    else:
        print("SBR columns not found in SPECT file.")
        # Fallback to simulation
        raw_vals = np.concatenate([np.random.normal(100, 20, 1000), np.random.normal(200, 30, 500)])
        norm_vals = (raw_vals - raw_vals.mean()) / raw_vals.std()
        
        sns.histplot(raw_vals, ax=axes2[0], kde=True, color='purple')
        axes2[0].set_title("Raw Voxel Intensity Distribution (Simulated)")
        
        sns.histplot(norm_vals, ax=axes2[1], kde=True, color='green')
        axes2[1].set_title("Z-Score Normalized Intensity")
else:
    # Simulation
    raw_vals = np.concatenate([np.random.normal(100, 20, 1000), np.random.normal(200, 30, 500)])
    norm_vals = (raw_vals - raw_vals.mean()) / raw_vals.std()
    
    sns.histplot(raw_vals, ax=axes2[0], kde=True, color='purple')
    axes2[0].set_title("Raw Voxel Intensity Distribution (Simulated)")
    
    sns.histplot(norm_vals, ax=axes2[1], kde=True, color='green')
    axes2[1].set_title("Z-Score Normalized Intensity")

plt.tight_layout()
plt.savefig(output_dir / "preprocessing_normalization.png", dpi=300)
plt.close()


# --- Figure 3: Polygenic Risk Scores (PRS) ---
print("Generating PRS Figure...")
fig3, axes3 = plt.subplots(1, 3, figsize=(18, 5))

genes = ['LRRK2', 'GBA', 'APOE_RISK']
colors = ['teal', 'coral', 'orchid']

for i, gene in enumerate(genes):
    if gene in enhanced_df.columns:
        sns.histplot(data=enhanced_df, x=gene, ax=axes3[i], color=colors[i], bins=20, kde=True)
        axes3[i].set_title(f"{gene} Risk Score Distribution")
        axes3[i].set_xlabel("Weighted Risk Score")
    else:
        axes3[i].text(0.5, 0.5, f"{gene} not found", ha='center')

plt.tight_layout()
plt.savefig(output_dir / "preprocessing_genetic_encoding.png", dpi=300)
plt.close()

print("Done! Figures saved to results/preprocessing_figures")
