#!/usr/bin/env python3
"""
Fix Enhanced Graph Structure
===========================

This script fixes the mismatch between the enhanced dataset (297 patients) 
and production graph structure (557 nodes) by creating a proper graph
structure that matches the available enhanced data.

The issue: Enhanced dataset has 297 unique patients with longitudinal data,
but production graph expects 557 nodes.

Solution: Create a new graph structure with 297 nodes using cosine similarity
between patient feature vectors, matching the production model approach.
"""

import pandas as pd
import torch
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
from torch_geometric.data import Data
from pathlib import Path
import json

def aggregate_longitudinal_data(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate longitudinal data to one record per patient."""
    print("📊 Aggregating longitudinal data to unique patients...")
    
    # Use baseline visit where available, otherwise use first visit
    if 'EVENT_ID' in df.columns:
        # Prefer baseline visits
        baseline_df = df[df['EVENT_ID'] == 'BL'].copy()
        print(f"   Found {len(baseline_df)} baseline visits")
        
        # For patients without baseline, use first available visit
        patients_with_baseline = set(baseline_df['PATNO'].unique())
        all_patients = set(df['PATNO'].unique())
        missing_baseline = all_patients - patients_with_baseline
        
        if missing_baseline:
            print(f"   Adding {len(missing_baseline)} patients without baseline visits")
            missing_df = df[df['PATNO'].isin(missing_baseline)].groupby('PATNO').first().reset_index()
            aggregated_df = pd.concat([baseline_df, missing_df], ignore_index=True)
        else:
            aggregated_df = baseline_df
    else:
        # Simple aggregation: first visit per patient
        aggregated_df = df.groupby('PATNO').first().reset_index()
    
    print(f"✅ Aggregated to {len(aggregated_df)} unique patients")
    return aggregated_df

def create_similarity_graph(feature_matrix: np.ndarray, k: int = 6) -> torch.Tensor:
    """Create graph edges using cosine similarity (top-k connections)."""
    print(f"🔗 Creating similarity graph with k={k} connections per node...")
    
    # Compute cosine similarity matrix
    similarity_matrix = cosine_similarity(feature_matrix)
    
    # Create edge list from top-k connections
    edge_list = []
    for i in range(len(similarity_matrix)):
        # Get top-k most similar nodes (excluding self)
        similarities = similarity_matrix[i]
        similarities[i] = -1  # Exclude self-connection
        top_k_indices = np.argsort(similarities)[-k:]
        
        # Add edges
        for j in top_k_indices:
            edge_list.append([i, j])
            edge_list.append([j, i])  # Make undirected
    
    # Remove duplicates and convert to tensor
    edge_array = np.array(edge_list).T
    edge_index = torch.LongTensor(edge_array)
    
    # Remove duplicate edges
    edge_index = torch.unique(edge_index, dim=1)
    
    print(f"✅ Created graph with {edge_index.shape[1]} edges")
    return edge_index

def create_labels(df: pd.DataFrame) -> torch.Tensor:
    """Create binary labels from cohort definition."""
    print("🏷️  Creating binary labels...")
    
    if 'COHORT_DEFINITION' in df.columns:
        # Convert cohort to binary labels (PD=1, HC=0)
        labels = (df['COHORT_DEFINITION'] == "Parkinson's Disease").astype(int)
    elif 'labels' in df.columns:
        labels = df['labels']
    elif 'y' in df.columns:
        labels = df['y']
    else:
        raise ValueError("No label column found")
    
    label_tensor = torch.LongTensor(labels.values)
    
    print(f"✅ Created labels: {torch.bincount(label_tensor)} (HC=0, PD=1)")
    return label_tensor

def main():
    """Fix the enhanced graph structure."""
    print("🔧 FIXING ENHANCED GRAPH STRUCTURE")
    print("=" * 50)
    
    # Load enhanced dataset
    enhanced_path = Path("data/enhanced/enhanced_dataset_latest.csv")
    if not enhanced_path.exists():
        raise FileNotFoundError(f"Enhanced dataset not found: {enhanced_path}")
    
    df = pd.read_csv(enhanced_path)
    print(f"📥 Loaded enhanced dataset: {df.shape}")
    print(f"   Unique patients: {df['PATNO'].nunique()}")
    print(f"   Total visits: {len(df)}")
    
    # Aggregate longitudinal data
    aggregated_df = aggregate_longitudinal_data(df)
    
    # Extract 12 features
    feature_columns = [
        'LRRK2', 'GBA', 'APOE_RISK', 'PTAU', 'TTAU', 'UPSIT_TOTAL', 'ALPHA_SYN',  # Current 7
        'AGE_COMPUTED', 'NHY', 'SEX', 'NP3TOT', 'HAS_DATSCAN'  # Enhanced +5
    ]
    
    print(f"🔢 Extracting {len(feature_columns)} features...")
    feature_matrix = aggregated_df[feature_columns].values
    
    # Check for missing values
    missing_count = np.isnan(feature_matrix).sum()
    if missing_count > 0:
        print(f"⚠️  Found {missing_count} missing values, filling with median...")
        from sklearn.impute import SimpleImputer
        imputer = SimpleImputer(strategy='median')
        feature_matrix = imputer.fit_transform(feature_matrix)
    
    # Standardize features
    print("📏 Standardizing features...")
    scaler = StandardScaler()
    feature_matrix_scaled = scaler.fit_transform(feature_matrix)
    
    # Create graph structure
    edge_index = create_similarity_graph(feature_matrix_scaled, k=6)
    
    # Create labels
    labels = create_labels(aggregated_df)
    
    # Create PyTorch Geometric data
    print("🔄 Creating PyTorch Geometric data...")
    graph_data = Data(
        x=torch.FloatTensor(feature_matrix_scaled),
        edge_index=edge_index,
        y=labels,
        num_nodes=len(aggregated_df)
    )
    
    # Add metadata
    graph_data.feature_names = feature_columns
    graph_data.patient_ids = aggregated_df['PATNO'].values
    graph_data.scaler = scaler
    graph_data.version = "v1.1.0_fixed"
    
    print(f"✅ Fixed graph data created:")
    print(f"   📊 Nodes: {graph_data.num_nodes}")
    print(f"   📊 Edges: {graph_data.num_edges}")
    print(f"   📊 Features: {graph_data.x.shape[1]}")
    print(f"   📊 Labels: {torch.bincount(graph_data.y)}")
    
    # Save fixed graph data
    output_dir = Path("data/enhanced")
    output_dir.mkdir(exist_ok=True)
    
    timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    
    # Save graph data
    graph_path = output_dir / f"enhanced_graph_data_fixed_{timestamp}.pth"
    torch.save(graph_data, graph_path)
    
    # Update latest symlink
    latest_path = output_dir / "enhanced_graph_data_latest.pth"
    if latest_path.exists():
        latest_path.unlink()
    latest_path.symlink_to(graph_path.name)
    
    # Save aggregated dataset
    dataset_path = output_dir / f"enhanced_dataset_aggregated_{timestamp}.csv"
    aggregated_df.to_csv(dataset_path, index=False)
    
    # Update latest dataset symlink
    latest_dataset_path = output_dir / "enhanced_dataset_latest.csv"
    if latest_dataset_path.exists():
        latest_dataset_path.unlink()
    latest_dataset_path.symlink_to(dataset_path.name)
    
    # Save scaler
    import pickle
    scaler_path = output_dir / f"enhanced_scaler_fixed_{timestamp}.pkl"
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
    
    # Update latest scaler symlink
    latest_scaler_path = output_dir / "enhanced_scaler_latest.pkl"
    if latest_scaler_path.exists():
        latest_scaler_path.unlink()
    latest_scaler_path.symlink_to(scaler_path.name)
    
    # Save metadata
    metadata = {
        'version': 'v1.1.0_fixed',
        'timestamp': timestamp,
        'nodes': int(graph_data.num_nodes),
        'edges': int(graph_data.num_edges),
        'features': len(feature_columns),
        'feature_names': feature_columns,
        'label_distribution': {
            'healthy_control': int(torch.bincount(graph_data.y)[0]),
            'parkinsons_disease': int(torch.bincount(graph_data.y)[1])
        },
        'aggregation_method': 'baseline_preferred',
        'graph_construction': 'cosine_similarity_k6',
        'scaling_method': 'standard_scaler'
    }
    
    metadata_path = output_dir / f"enhanced_metadata_fixed_{timestamp}.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    # Update latest metadata symlink
    latest_metadata_path = output_dir / "enhanced_metadata_latest.json"
    if latest_metadata_path.exists():
        latest_metadata_path.unlink()
    latest_metadata_path.symlink_to(metadata_path.name)
    
    print(f"\n🎉 FIXED ENHANCED GRAPH SAVED!")
    print(f"📁 Graph data: {graph_path}")
    print(f"📁 Dataset: {dataset_path}")
    print(f"📁 Scaler: {scaler_path}")
    print(f"📁 Metadata: {metadata_path}")
    print(f"\n✅ Ready for training with {graph_data.num_nodes} nodes and {graph_data.x.shape[1]} features!")

if __name__ == "__main__":
    main()