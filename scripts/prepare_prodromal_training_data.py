"""
Phase 8.1 Task 11: Prepare Prodromal Training Data

Prepares PyTorch Geometric Data objects for prodromal cohort (n=381)
with the same 32 features as manifest PD cohort for fair comparison.

Outputs:
- train_data.pt, val_data.pt, test_data.pt (PyG Data objects)
- split_info.json (patient IDs and event counts per split)
- feature_names.json (feature list for reproducibility)

Author: GIMAN Research Team
Date: October 12, 2025
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.impute import KNNImputer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import kneighbors_graph
from sklearn.preprocessing import StandardScaler
from torch_geometric.data import Data

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

# ============================================================================
# CONFIGURATION
# ============================================================================

PRODROMAL_DATA_DIR = Path("data/prodromal_cohort")
MANIFEST_DATA_DIR = Path("data/02_processed")
OUTPUT_DIR = Path("data/03_prodromal/training_ready")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Feature configuration (32 features matching manifest PD cohort)
DEMOGRAPHIC_FEATURES = ['AGE_COMPUTED', 'SEX']
CLINICAL_FEATURES = ['NP3TOT', 'NHY', 'MOCA_TOTAL']
GENETIC_FEATURES = ['LRRK2', 'GBA', 'APOE_RISK', 'SNCA_STATUS', 'GENETIC_RISK_SCORE']
IMAGING_FEATURES = [
    'CAUDATE_R', 'CAUDATE_L', 'PUTAMEN_R', 'PUTAMEN_L',
    'STRIATUM_R', 'STRIATUM_L'
]
BIOMARKER_FEATURES = [
    'PTAU', 'TTAU', 'ABETA_1_42', 'ABETA_1_40',
    'PTAU_ABETA_RATIO', 'TTAU_ABETA_RATIO',
    'UPSIT_TOTAL', 'ALPHA_SYN', 'NFL', 'TREM2'
]

ALL_FEATURES = (
    DEMOGRAPHIC_FEATURES + CLINICAL_FEATURES + GENETIC_FEATURES +
    IMAGING_FEATURES + BIOMARKER_FEATURES
)

# Graph construction
K_NEIGHBORS = 10
TRAIN_SIZE = 0.70
VAL_SIZE = 0.15
TEST_SIZE = 0.15
RANDOM_SEED = 42

print(f"\n{'=' * 70}")
print("PHASE 8.1 TASK 11: PREPARE PRODROMAL TRAINING DATA")
print(f"{'=' * 70}")
print(f"Output directory: {OUTPUT_DIR}")
print(f"Target features: {len(ALL_FEATURES)} (same as manifest PD)")
print(f"Split: {TRAIN_SIZE:.0%} train / {VAL_SIZE:.0%} val / {TEST_SIZE:.0%} test")
print(f"Graph: k={K_NEIGHBORS} nearest neighbors")


# ============================================================================
# DATA LOADING
# ============================================================================

def load_prodromal_cohort():
    """Load existing prodromal cohort with phenoconversion endpoints."""
    print(f"\n{'=' * 70}")
    print("STEP 1: LOAD PRODROMAL COHORT")
    print(f"{'=' * 70}")
    
    survival_file = PRODROMAL_DATA_DIR / "prodromal_survival_data.csv"
    df = pd.read_csv(survival_file)
    
    print(f"  ✓ Loaded {len(df)} patients from {survival_file.name}")
    print(f"  Phenoconversion events: {df['phenoconverted'].sum()}")
    print(f"  Median follow-up: {df['time_to_event'].median():.1f} months")
    
    # Rename columns to match expected format
    df = df.rename(columns={
        'phenoconverted': 'event',
        'time_to_event': 'time',
        'baseline_updrs': 'NP3TOT',
        'baseline_moca': 'MOCA_TOTAL',
        'sex': 'SEX',
        'age_approx': 'AGE_COMPUTED'
    })
    
    # Convert time from months to years (for consistency with manifest PD)
    df['time'] = df['time'] / 12.0
    
    return df


def load_enhanced_ppmi_features():
    """Load enhanced PPMI cohort for additional feature extraction."""
    print(f"\n{'=' * 70}")
    print("STEP 2: LOAD ENHANCED PPMI FEATURES")
    print(f"{'=' * 70}")
    
    enhanced_file = MANIFEST_DATA_DIR / "enhanced_real_ppmi_cohort.csv"
    
    if not enhanced_file.exists():
        print(f"  ⚠️  Enhanced PPMI file not found: {enhanced_file}")
        print(f"  Will proceed with limited features from prodromal cohort")
        return None
    
    df = pd.read_csv(enhanced_file)
    print(f"  ✓ Loaded {len(df)} patients from {enhanced_file.name}")
    print(f"  Available features: {len(df.columns)}")
    
    return df


def merge_features(prodromal_df, enhanced_df):
    """
    Merge prodromal cohort with enhanced features.
    
    Args:
        prodromal_df: Prodromal cohort with endpoints
        enhanced_df: Enhanced PPMI cohort with full features
    
    Returns:
        Merged DataFrame with all 32 features
    """
    print(f"\n{'=' * 70}")
    print("STEP 3: MERGE FEATURES")
    print(f"{'=' * 70}")
    
    if enhanced_df is None:
        print(f"  Using prodromal cohort features only")
        merged_df = prodromal_df.copy()
    else:
        # Try to merge on PATNO
        print(f"  Attempting merge on PATNO...")
        merged_df = prodromal_df.merge(
            enhanced_df,
            on='PATNO',
            how='left',
            suffixes=('', '_enhanced')
        )
        
        # Count how many patients have enhanced features
        n_matched = merged_df['PATNO'].notna().sum()
        print(f"  Matched {n_matched}/{len(prodromal_df)} patients with enhanced features")
    
    # Check which features are available
    print(f"\n  Feature availability:")
    available_features = []
    missing_features = []
    
    for feature in ALL_FEATURES:
        if feature in merged_df.columns:
            n_available = merged_df[feature].notna().sum()
            pct_available = n_available / len(merged_df) * 100
            available_features.append(feature)
            print(f"    ✓ {feature}: {n_available}/{len(merged_df)} ({pct_available:.1f}%)")
        else:
            missing_features.append(feature)
            print(f"    ✗ {feature}: NOT AVAILABLE")
    
    print(f"\n  Summary:")
    print(f"    Available: {len(available_features)}/{len(ALL_FEATURES)} features")
    print(f"    Missing: {len(missing_features)} features")
    
    if missing_features:
        print(f"    Missing features: {', '.join(missing_features[:10])}")
        if len(missing_features) > 10:
            print(f"    ... and {len(missing_features) - 10} more")
    
    return merged_df, available_features


# ============================================================================
# FEATURE ENGINEERING
# ============================================================================

def extract_feature_matrix(df, feature_list):
    """Extract feature matrix with available features."""
    print(f"\n{'=' * 70}")
    print("STEP 4: EXTRACT FEATURE MATRIX")
    print(f"{'=' * 70}")
    
    # Extract only available features
    X = df[feature_list].values
    
    print(f"  Feature matrix shape: {X.shape}")
    print(f"  Total values: {X.size}")
    print(f"  Missing values: {np.isnan(X).sum()} ({np.isnan(X).sum()/X.size*100:.1f}%)")
    
    return X


def impute_and_normalize(X_train, X_val, X_test):
    """
    Apply imputation and normalization to train/val/test sets.
    
    Fit on training set, apply to all sets.
    
    Args:
        X_train, X_val, X_test: Feature matrices
    
    Returns:
        Imputed and normalized feature matrices
    """
    print(f"\n{'=' * 70}")
    print("STEP 5: IMPUTE AND NORMALIZE")
    print(f"{'=' * 70}")
    
    # Imputation (KNN with k=5)
    print(f"\n  Imputing missing values (KNNImputer, k=5)...")
    imputer = KNNImputer(n_neighbors=5, weights='distance')
    
    X_train_imputed = imputer.fit_transform(X_train)
    X_val_imputed = imputer.transform(X_val)
    X_test_imputed = imputer.transform(X_test)
    
    print(f"    ✓ Train: {X_train.shape} → {X_train_imputed.shape}")
    print(f"    ✓ Val: {X_val.shape} → {X_val_imputed.shape}")
    print(f"    ✓ Test: {X_test.shape} → {X_test_imputed.shape}")
    
    # Normalization (StandardScaler)
    print(f"\n  Normalizing features (StandardScaler)...")
    scaler = StandardScaler()
    
    X_train_normalized = scaler.fit_transform(X_train_imputed)
    X_val_normalized = scaler.transform(X_val_imputed)
    X_test_normalized = scaler.transform(X_test_imputed)
    
    print(f"    ✓ Train mean: {X_train_normalized.mean():.2e}, std: {X_train_normalized.std():.2f}")
    print(f"    ✓ Val mean: {X_val_normalized.mean():.2e}, std: {X_val_normalized.std():.2f}")
    print(f"    ✓ Test mean: {X_test_normalized.mean():.2e}, std: {X_test_normalized.std():.2f}")
    
    return X_train_normalized, X_val_normalized, X_test_normalized


# ============================================================================
# GRAPH CONSTRUCTION
# ============================================================================

def construct_patient_graph(X, k=10):
    """
    Construct patient similarity graph using k-nearest neighbors.
    
    Args:
        X: Normalized feature matrix (n_patients × n_features)
        k: Number of nearest neighbors
    
    Returns:
        edge_index: PyTorch tensor (2 × n_edges) for edge list
    """
    print(f"\n{'=' * 70}")
    print("STEP 6: CONSTRUCT SIMILARITY GRAPH")
    print(f"{'=' * 70}")
    
    print(f"  Computing k-NN graph (k={k})...")
    
    # Compute k-NN graph (cosine similarity via sklearn)
    adj_matrix = kneighbors_graph(
        X, 
        n_neighbors=k,
        mode='connectivity',
        metric='cosine',
        include_self=False
    )
    
    # Make symmetric (bidirectional edges)
    adj_matrix = adj_matrix + adj_matrix.T
    adj_matrix[adj_matrix > 0] = 1
    
    # Convert to edge_index format (COO)
    edge_index = torch.tensor(
        np.array(adj_matrix.nonzero()),
        dtype=torch.long
    )
    
    n_nodes = X.shape[0]
    n_edges = edge_index.shape[1]
    avg_degree = n_edges / n_nodes
    
    print(f"  ✓ Graph constructed:")
    print(f"    Nodes: {n_nodes}")
    print(f"    Edges: {n_edges}")
    print(f"    Avg degree: {avg_degree:.1f}")
    
    return edge_index


# ============================================================================
# DATA SPLITTING
# ============================================================================

def stratified_split_with_events(df, train_size=0.70, val_size=0.15, random_state=42):
    """
    Stratified split ensuring events are distributed across splits.
    
    Args:
        df: DataFrame with 'event' column
        train_size, val_size: Split proportions
        random_state: Random seed
    
    Returns:
        train_indices, val_indices, test_indices
    """
    print(f"\n{'=' * 70}")
    print("STEP 7: STRATIFIED TRAIN/VAL/TEST SPLIT")
    print(f"{'=' * 70}")
    
    n_total = len(df)
    n_events = df['event'].sum()
    
    print(f"\n  Total patients: {n_total}")
    print(f"  Total events: {n_events} ({n_events/n_total*100:.1f}%)")
    
    # First split: train vs (val + test)
    train_idx, temp_idx = train_test_split(
        np.arange(n_total),
        train_size=train_size,
        stratify=df['event'].values,
        random_state=random_state
    )
    
    # Second split: val vs test
    val_proportion = val_size / (val_size + (1 - train_size - val_size))
    val_idx, test_idx = train_test_split(
        temp_idx,
        train_size=val_proportion,
        stratify=df.iloc[temp_idx]['event'].values,
        random_state=random_state
    )
    
    # Report split statistics
    train_events = df.iloc[train_idx]['event'].sum()
    val_events = df.iloc[val_idx]['event'].sum()
    test_events = df.iloc[test_idx]['event'].sum()
    
    print(f"\n  Split statistics:")
    print(f"    Train: {len(train_idx)} patients, {train_events} events ({train_events/len(train_idx)*100:.1f}%)")
    print(f"    Val: {len(val_idx)} patients, {val_events} events ({val_events/len(val_idx)*100:.1f}%)")
    print(f"    Test: {len(test_idx)} patients, {test_events} events ({test_events/len(test_idx)*100:.1f}%)")
    
    # Verify split proportions
    assert len(train_idx) + len(val_idx) + len(test_idx) == n_total
    assert train_events + val_events + test_events == n_events
    
    print(f"\n  ✓ Split verified:")
    print(f"    Train: {len(train_idx)/n_total*100:.1f}% patients, {train_events/n_events*100:.1f}% events")
    print(f"    Val: {len(val_idx)/n_total*100:.1f}% patients, {val_events/n_events*100:.1f}% events")
    print(f"    Test: {len(test_idx)/n_total*100:.1f}% patients, {test_events/n_events*100:.1f}% events")
    
    return train_idx, val_idx, test_idx


# ============================================================================
# PYTORCH GEOMETRIC DATA CREATION
# ============================================================================

def create_pyg_data(X, edge_index, y_event, y_time):
    """
    Create PyTorch Geometric Data object.
    
    Args:
        X: Feature matrix (n × d)
        edge_index: Edge list (2 × e)
        y_event: Event indicator (n,)
        y_time: Time to event (n,)
    
    Returns:
        PyG Data object
    """
    data = Data(
        x=torch.tensor(X, dtype=torch.float32),
        edge_index=edge_index,
        event=torch.tensor(y_event, dtype=torch.float32),
        time=torch.tensor(y_time, dtype=torch.float32)
    )
    
    return data


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Prepare prodromal training data."""
    
    # Load data
    prodromal_df = load_prodromal_cohort()
    enhanced_df = load_enhanced_ppmi_features()
    
    # Merge features
    merged_df, available_features = merge_features(prodromal_df, enhanced_df)
    
    # Check if we have enough features
    if len(available_features) < 10:
        print(f"\n⚠️  WARNING: Only {len(available_features)} features available")
        print(f"  Proceeding with available features: {available_features}")
    
    # Extract feature matrix
    X = extract_feature_matrix(merged_df, available_features)
    y_event = merged_df['event'].values
    y_time = merged_df['time'].values
    patnos = merged_df['PATNO'].values
    
    # Split data
    train_idx, val_idx, test_idx = stratified_split_with_events(
        merged_df, 
        train_size=TRAIN_SIZE,
        val_size=VAL_SIZE,
        random_state=RANDOM_SEED
    )
    
    # Extract split data
    X_train, X_val, X_test = X[train_idx], X[val_idx], X[test_idx]
    y_event_train, y_event_val, y_event_test = y_event[train_idx], y_event[val_idx], y_event[test_idx]
    y_time_train, y_time_val, y_time_test = y_time[train_idx], y_time[val_idx], y_time[test_idx]
    
    # Impute and normalize
    X_train_norm, X_val_norm, X_test_norm = impute_and_normalize(X_train, X_val, X_test)
    
    # Construct graphs
    edge_index_train = construct_patient_graph(X_train_norm, k=K_NEIGHBORS)
    edge_index_val = construct_patient_graph(X_val_norm, k=K_NEIGHBORS)
    edge_index_test = construct_patient_graph(X_test_norm, k=K_NEIGHBORS)
    
    # Create PyG Data objects
    print(f"\n{'=' * 70}")
    print("STEP 8: CREATE PYTORCH GEOMETRIC DATA OBJECTS")
    print(f"{'=' * 70}")
    
    train_data = create_pyg_data(X_train_norm, edge_index_train, y_event_train, y_time_train)
    val_data = create_pyg_data(X_val_norm, edge_index_val, y_event_val, y_time_val)
    test_data = create_pyg_data(X_test_norm, edge_index_test, y_event_test, y_time_test)
    
    print(f"\n  ✓ Created PyG Data objects:")
    print(f"    Train: {train_data}")
    print(f"    Val: {val_data}")
    print(f"    Test: {test_data}")
    
    # Save data
    print(f"\n{'=' * 70}")
    print("STEP 9: SAVE DATA")
    print(f"{'=' * 70}")
    
    torch.save(train_data, OUTPUT_DIR / "train_data.pt")
    torch.save(val_data, OUTPUT_DIR / "val_data.pt")
    torch.save(test_data, OUTPUT_DIR / "test_data.pt")
    
    print(f"  ✓ Saved train_data.pt")
    print(f"  ✓ Saved val_data.pt")
    print(f"  ✓ Saved test_data.pt")
    
    # Save split info
    split_info = {
        'train_patnos': patnos[train_idx].tolist(),
        'val_patnos': patnos[val_idx].tolist(),
        'test_patnos': patnos[test_idx].tolist(),
        'train_events': int(y_event_train.sum()),
        'val_events': int(y_event_val.sum()),
        'test_events': int(y_event_test.sum()),
        'n_train': len(train_idx),
        'n_val': len(val_idx),
        'n_test': len(test_idx),
        'random_seed': RANDOM_SEED
    }
    
    with open(OUTPUT_DIR / "split_info.json", 'w') as f:
        json.dump(split_info, f, indent=2)
    
    print(f"  ✓ Saved split_info.json")
    
    # Save feature names
    feature_info = {
        'feature_names': available_features,
        'n_features': len(available_features),
        'feature_groups': {
            'demographic': [f for f in DEMOGRAPHIC_FEATURES if f in available_features],
            'clinical': [f for f in CLINICAL_FEATURES if f in available_features],
            'genetic': [f for f in GENETIC_FEATURES if f in available_features],
            'imaging': [f for f in IMAGING_FEATURES if f in available_features],
            'biomarker': [f for f in BIOMARKER_FEATURES if f in available_features]
        }
    }
    
    with open(OUTPUT_DIR / "feature_names.json", 'w') as f:
        json.dump(feature_info, f, indent=2)
    
    print(f"  ✓ Saved feature_names.json")
    
    # Final summary
    print(f"\n{'=' * 70}")
    print("PRODROMAL TRAINING DATA PREPARATION COMPLETE!")
    print(f"{'=' * 70}")
    print(f"\n✓ Output directory: {OUTPUT_DIR}")
    print(f"\n✓ Files created:")
    print(f"  - train_data.pt ({len(train_idx)} patients, {int(y_event_train.sum())} events)")
    print(f"  - val_data.pt ({len(val_idx)} patients, {int(y_event_val.sum())} events)")
    print(f"  - test_data.pt ({len(test_idx)} patients, {int(y_event_test.sum())} events)")
    print(f"  - split_info.json (patient IDs and event counts)")
    print(f"  - feature_names.json ({len(available_features)} features)")
    
    print(f"\n✓ Ready for Phase 8.1 Task 12: Train GIMAN-Prognostic")
    print(f"\nNext command:")
    print(f"  python scripts/train_giman_prognostic_prodromal.py")
    print(f"\n{'=' * 70}\n")


if __name__ == "__main__":
    main()
