"""Prepare PyTorch Geometric Training Data from Final Unified Dataset.

Creates PyG Data objects with:
- Node features: 49 multimodal features (genetic, clinical, imaging, biomarkers)
- Graph structure: kNN graph based on feature similarity
- Labels: time_to_event and phenoconverted for survival analysis
- Split: 85% train, 15% test (no separate val, will use 5-fold CV on training)
"""

from __future__ import annotations

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
project_root = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(project_root))


def load_unified_dataset() -> pd.DataFrame:
    """Load final unified training dataset."""
    data_path = project_root / "data" / "03_prodromal" / "final_training_dataset" / "unified_longitudinal_early_pd.csv"
    
    if not data_path.exists():
        raise FileNotFoundError("Run merge_final_training_dataset.py first")
    
    df = pd.read_csv(data_path)
    print(f"✓ Loaded unified dataset: {df.shape}")
    print(f"  Observations: {len(df)}")
    print(f"  Events: {df['phenoconverted'].sum()} ({df['phenoconverted'].mean()*100:.1f}%)")
    print(f"  Features: {df.shape[1] - 7}")  # Exclude metadata
    
    return df


def prepare_features(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """Prepare features, times, and events for PyG."""
    # Normalize endpoint schema
    if 'time_to_event' not in df.columns and 'event_time' in df.columns:
        df = df.rename(columns={'event_time': 'time_to_event'})
    if 'phenoconverted' not in df.columns and 'event_observed' in df.columns:
        df = df.rename(columns={'event_observed': 'phenoconverted'})

    required = {'time_to_event', 'phenoconverted'}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required survival columns: {sorted(missing)}")

    # Identify feature columns
    exclude_cols = {'PATNO', 'time_to_event', 'phenoconverted', 'landmark_month',
                   'original_time', 'original_event', 'cohort'}
    feature_cols = [c for c in df.columns if c not in exclude_cols]
    
    print(f"\n✓ Feature columns: {len(feature_cols)}")
    
    # Extract features
    X = df[feature_cols].values
    times = df['time_to_event'].values
    events = df['phenoconverted'].values
    
    # Check for missing values
    n_missing = np.isnan(X).sum()
    if n_missing > 0:
        print(f"⚠️  Found {n_missing} missing values, applying KNN imputation...")
        imputer = KNNImputer(n_neighbors=5)
        X = imputer.fit_transform(X)
        print(f"✓ Imputation complete")
    
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    print(f"✓ Features prepared: {X_scaled.shape}")
    print(f"  Mean: {X_scaled.mean():.6f}")
    print(f"  Std: {X_scaled.std():.6f}")
    
    return X_scaled, times, events, feature_cols


def construct_knn_graph(X: np.ndarray, k: int = 10) -> torch.Tensor:
    """Construct kNN graph based on feature similarity."""
    print(f"\n✓ Constructing kNN graph (k={k})...")
    
    # Compute kNN graph
    adj = kneighbors_graph(X, n_neighbors=k, mode='connectivity', include_self=False)
    
    # Convert to edge index
    adj_coo = adj.tocoo()
    edge_index = torch.tensor(np.vstack([adj_coo.row, adj_coo.col]), dtype=torch.long)
    
    print(f"✓ Graph constructed: {edge_index.shape[1]} edges")
    
    return edge_index


def create_pyg_data(
    X: np.ndarray,
    times: np.ndarray,
    events: np.ndarray,
    edge_index: torch.Tensor,
    split: str
) -> Data:
    """Create PyTorch Geometric Data object."""
    data = Data(
        x=torch.tensor(X, dtype=torch.float32),
        edge_index=edge_index,
        time=torch.tensor(times, dtype=torch.float32),
        event=torch.tensor(events, dtype=torch.long),
        split=split
    )
    
    return data


def split_data(
    X: np.ndarray,
    times: np.ndarray,
    events: np.ndarray,
    test_size: float = 0.15,
    random_state: int = 42
) -> tuple:
    """Split data into train and test sets with stratification."""
    print(f"\n✓ Splitting data (train={1-test_size:.0%}, test={test_size:.0%})...")
    
    # Stratify by event status
    indices = np.arange(len(X))
    train_idx, test_idx = train_test_split(
        indices,
        test_size=test_size,
        stratify=events,
        random_state=random_state
    )
    
    X_train, X_test = X[train_idx], X[test_idx]
    times_train, times_test = times[train_idx], times[test_idx]
    events_train, events_test = events[train_idx], events[test_idx]
    
    print(f"✓ Train set: {len(X_train)} observations")
    print(f"  Events: {events_train.sum()} ({events_train.mean()*100:.1f}%)")
    print(f"✓ Test set: {len(X_test)} observations")
    print(f"  Events: {events_test.sum()} ({events_test.mean()*100:.1f}%)")
    
    return (X_train, times_train, events_train, train_idx), \
           (X_test, times_test, events_test, test_idx)


def save_pyg_data(
    train_data: Data,
    test_data: Data,
    feature_names: list[str],
    output_dir: Path
) -> None:
    """Save PyG data objects and metadata."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save data objects
    torch.save(train_data, output_dir / "train_data.pt")
    torch.save(test_data, output_dir / "test_data.pt")
    
    print(f"\n✓ Saved PyG data objects to: {output_dir}")
    print(f"  train_data.pt: {train_data.num_nodes} nodes, {train_data.num_edges} edges")
    print(f"  test_data.pt: {test_data.num_nodes} nodes, {test_data.num_edges} edges")
    
    # Save metadata
    import json
    metadata = {
        'n_features': len(feature_names),
        'feature_names': feature_names,
        'train_size': int(train_data.num_nodes),
        'test_size': int(test_data.num_nodes),
        'train_events': int(train_data.event.sum()),
        'test_events': int(test_data.event.sum()),
        'train_event_rate': float(train_data.event.float().mean()),
        'test_event_rate': float(test_data.event.float().mean())
    }
    
    metadata_path = output_dir / "pyg_data_metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"✓ Saved metadata: {metadata_path}")


def main() -> None:
    """Prepare PyG training data from unified dataset."""
    print("="*60)
    print("PHASE 8.2: PyG TRAINING DATA PREPARATION")
    print("="*60 + "\n")
    
    # Load dataset
    df = load_unified_dataset()
    
    # Prepare features
    X, times, events, feature_names = prepare_features(df)
    
    # Split data
    train_data, test_data = split_data(X, times, events, test_size=0.15)
    X_train, times_train, events_train, train_idx = train_data
    X_test, times_test, events_test, test_idx = test_data
    
    # Construct graphs
    edge_index_train = construct_knn_graph(X_train, k=10)
    edge_index_test = construct_knn_graph(X_test, k=10)
    
    # Create PyG Data objects
    print("\n✓ Creating PyG Data objects...")
    train_pyg = create_pyg_data(X_train, times_train, events_train, edge_index_train, "train")
    test_pyg = create_pyg_data(X_test, times_test, events_test, edge_index_test, "test")
    
    # Save
    output_dir = project_root / "data" / "03_prodromal" / "final_pyg_data"
    save_pyg_data(train_pyg, test_pyg, feature_names, output_dir)
    
    print("\n" + "="*60)
    print("PyG DATA PREPARATION COMPLETE")
    print("="*60)
    print(f"\nReady for training with:")
    print(f"  - {len(feature_names)} features")
    print(f"  - {len(X_train)} training observations ({events_train.sum()} events)")
    print(f"  - {len(X_test)} test observations ({events_test.sum()} events)")
    print(f"  - Use 5-fold CV on training set for model selection")


if __name__ == "__main__":
    main()
