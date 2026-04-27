"""Prepare Multi-Task Training Data for Phase 8.5 GIMAN.

This script merges all 4 task datasets with missing label handling:
1. Task 1: Progression (time-to-disability milestones) - 127 patients
2. Task 2: Conversion (phenoconversion) - 127 patients  
3. Task 3: SAA (alpha-synuclein prediction) - 608 observations
4. Task 4: Diagnostic (early_pd vs prodromal) - 608 observations

Strategy: Option A (Union with Masking)
- Use all 608 observations from SAA dataset
- Merge progression/conversion labels where PATNO matches
- Create label masks per task to handle missing labels
- Generate PyTorch Geometric Data objects with graph structure

Output:
- multitask_train_data.pt: Training set PyG Data objects
- multitask_val_data.pt: Validation set PyG Data objects
- multitask_test_data.pt: Test set PyG Data objects
- multitask_data_summary.json: Dataset statistics and metadata

Author: GIMAN Phase 8.5 Development
Date: October 14, 2025
"""

from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from torch_geometric.data import Data

warnings.filterwarnings("ignore")

# Add project root to path
project_root = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(project_root))


class MultiTaskDataPreparator:
    """Prepare and merge data for multi-task GIMAN training."""
    
    def __init__(
        self,
        data_dir: Path,
        output_dir: Path,
        k_neighbors: int = 10,
        test_size: float = 0.15,
        val_size: float = 0.15,
        random_state: int = 42
    ):
        """
        Initialize data preparator.
        
        Args:
            data_dir: Root data directory
            output_dir: Output directory for processed data
            k_neighbors: Number of neighbors for graph construction
            test_size: Fraction of data for test set
            val_size: Fraction of data for validation set
            random_state: Random seed for reproducibility
        """
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.k_neighbors = k_neighbors
        self.test_size = test_size
        self.val_size = val_size
        self.random_state = random_state
        
        # Data storage
        self.saa_data: Optional[pd.DataFrame] = None
        self.progression_data: Optional[pd.DataFrame] = None
        self.conversion_data: Optional[pd.DataFrame] = None
        
        # Processed data
        self.merged_data: Optional[pd.DataFrame] = None
        self.scaler: Optional[StandardScaler] = None
        self.feature_columns: Optional[List[str]] = None
        
        print("=" * 80)
        print("PHASE 8.5: MULTI-TASK DATA PREPARATION")
        print("=" * 80)
    
    def load_datasets(self) -> None:
        """Load all 4 task datasets."""
        print("\n[1/6] Loading datasets...")
        
        # Task 3: SAA data (primary dataset with 608 observations)
        saa_path = self.data_dir / "04_saa" / "saa_training_data.csv"
        self.saa_data = pd.read_csv(saa_path)
        print(f"  ✓ Loaded SAA data: {self.saa_data.shape}")
        print(f"    - Unique patients: {self.saa_data['PATNO'].nunique()}")
        print(f"    - Total observations: {len(self.saa_data)}")
        
        # Task 1: Progression data
        prog_path = self.data_dir / "02_processed" / "progression_survival_data.csv"
        self.progression_data = pd.read_csv(prog_path)
        print(f"  ✓ Loaded Progression data: {self.progression_data.shape}")
        print(f"    - Unique patients: {self.progression_data['PATNO'].nunique()}")
        
        # Task 2: Conversion data
        conv_path = self.data_dir / "02_processed" / "conversion_labels.csv"
        self.conversion_data = pd.read_csv(conv_path)
        print(f"  ✓ Loaded Conversion data: {self.conversion_data.shape}")
        print(f"    - Unique patients: {self.conversion_data['PATNO'].nunique()}")
        print(f"    - Converters: {self.conversion_data['converted'].sum()}")
    
    def merge_datasets(self) -> None:
        """Merge all datasets with missing label handling."""
        print("\n[2/6] Merging datasets with label masking...")
        
        # Start with SAA data (608 observations)
        merged = self.saa_data.copy()
        print(f"  → Base dataset (SAA): {len(merged)} observations")
        
        # Merge progression data (left join to keep all SAA observations)
        merged = merged.merge(
            self.progression_data[['PATNO', 'event_time', 'event_observed', 'endpoint_type']],
            on='PATNO',
            how='left',
            suffixes=('', '_prog')
        )
        print(f"  → After progression merge: {len(merged)} observations")
        print(f"    - With progression labels: {merged['event_time'].notna().sum()}")
        
        # Merge conversion data (left join)
        merged = merged.merge(
            self.conversion_data[['PATNO', 'converted', 'conversion_type']],
            on='PATNO',
            how='left',
            suffixes=('', '_conv')
        )
        print(f"  → After conversion merge: {len(merged)} observations")
        print(f"    - With conversion labels: {merged['converted'].notna().sum()}")
        
        # Create label masks (True = label available, False = missing)
        merged['progression_mask'] = merged['event_time'].notna().astype(int)
        merged['conversion_mask'] = merged['converted'].notna().astype(int)
        merged['saa_mask'] = merged['SAA_POSITIVE'].notna().astype(int)
        merged['diagnostic_mask'] = merged['cohort'].notna().astype(int)
        
        # Map diagnostic labels (early_pd=0, prodromal=1)
        merged['diagnostic_label'] = merged['cohort'].map({
            'early_pd': 0,
            'prodromal': 1
        })
        
        # Fill missing survival labels with placeholder values
        merged['event_time'] = merged['event_time'].fillna(-1.0)
        merged['event_observed'] = merged['event_observed'].fillna(0).astype(int)
        merged['converted'] = merged['converted'].fillna(0).astype(int)
        
        self.merged_data = merged
        
        # Print label availability summary
        print("\n  Label Availability Summary:")
        print(f"    - Progression labels: {merged['progression_mask'].sum()} / {len(merged)} ({100*merged['progression_mask'].mean():.1f}%)")
        print(f"    - Conversion labels:  {merged['conversion_mask'].sum()} / {len(merged)} ({100*merged['conversion_mask'].mean():.1f}%)")
        print(f"    - SAA labels:         {merged['saa_mask'].sum()} / {len(merged)} ({100*merged['saa_mask'].mean():.1f}%)")
        print(f"    - Diagnostic labels:  {merged['diagnostic_mask'].sum()} / {len(merged)} ({100*merged['diagnostic_mask'].mean():.1f}%)")
    
    def prepare_features(self) -> None:
        """Prepare and standardize features."""
        print("\n[3/6] Preparing features...")
        
        # Define feature columns (exclude labels, masks, metadata)
        exclude_cols = [
            'PATNO', 'EVENT_ID', 'SAA_POSITIVE', 'ALPHA_SYN_VALUE',
            'time_to_event', 'phenoconverted', 'landmark_month',
            'original_time', 'original_event', 'cohort',
            'event_time', 'event_observed', 'endpoint_type',
            'converted', 'conversion_type',
            'progression_mask', 'conversion_mask', 'saa_mask', 'diagnostic_mask',
            'diagnostic_label'
        ]
        
        # Get feature columns
        self.feature_columns = [
            col for col in self.merged_data.columns 
            if col not in exclude_cols
        ]
        
        print(f"  → Selected {len(self.feature_columns)} features")
        print(f"    Example features: {self.feature_columns[:5]}")
        
        # Standardize features
        X = self.merged_data[self.feature_columns].values
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)
        
        # Replace in dataframe
        for i, col in enumerate(self.feature_columns):
            self.merged_data[col] = X_scaled[:, i]
        
        print(f"  ✓ Features standardized (mean≈0, std≈1)")
    
    def construct_graphs(self, features: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Construct k-NN graph from patient features.
        
        Args:
            features: Feature matrix (n_samples, n_features)
            
        Returns:
            edge_index: Edge indices (2, num_edges)
            edge_attr: Edge attributes (num_edges, 1) - distances
        """
        # Build k-NN graph
        knn = NearestNeighbors(n_neighbors=self.k_neighbors + 1, metric='euclidean')
        knn.fit(features)
        distances, indices = knn.kneighbors(features)
        
        # Convert to edge list (exclude self-loops)
        edge_list = []
        edge_distances = []
        
        for i in range(len(features)):
            for j in range(1, self.k_neighbors + 1):  # Skip first (self)
                neighbor_idx = indices[i, j]
                distance = distances[i, j]
                
                # Add edge i -> neighbor
                edge_list.append([i, neighbor_idx])
                edge_distances.append(distance)
        
        edge_index = np.array(edge_list, dtype=np.int64).T
        edge_attr = np.array(edge_distances, dtype=np.float32).reshape(-1, 1)
        
        return edge_index, edge_attr
    
    def create_pyg_data_objects(self) -> Tuple[List[Data], List[Data], List[Data]]:
        """
        Create PyTorch Geometric Data objects with train/val/test splits.
        
        Returns:
            train_data_list: List of training Data objects
            val_data_list: List of validation Data objects
            test_data_list: List of test Data objects
        """
        print("\n[4/6] Creating PyTorch Geometric Data objects...")
        
        # Get features
        X = self.merged_data[self.feature_columns].values.astype(np.float32)
        
        # Construct graph
        print(f"  → Constructing k-NN graph (k={self.k_neighbors})...")
        edge_index, edge_attr = self.construct_graphs(X)
        print(f"    - Nodes: {len(X)}")
        print(f"    - Edges: {edge_index.shape[1]}")
        
        # Convert to tensors
        x = torch.tensor(X, dtype=torch.float)
        edge_index = torch.tensor(edge_index, dtype=torch.long)
        edge_attr = torch.tensor(edge_attr, dtype=torch.float)
        
        # Task 1: Progression labels
        progression_time = torch.tensor(
            self.merged_data['event_time'].values, dtype=torch.float
        )
        progression_event = torch.tensor(
            self.merged_data['event_observed'].values, dtype=torch.long
        )
        progression_mask = torch.tensor(
            self.merged_data['progression_mask'].values, dtype=torch.bool
        )
        
        # Task 2: Conversion labels
        conversion_label = torch.tensor(
            self.merged_data['converted'].values, dtype=torch.long
        )
        conversion_mask = torch.tensor(
            self.merged_data['conversion_mask'].values, dtype=torch.bool
        )
        
        # Task 3: SAA labels
        saa_label = torch.tensor(
            self.merged_data['SAA_POSITIVE'].values, dtype=torch.long
        )
        saa_mask = torch.tensor(
            self.merged_data['saa_mask'].values, dtype=torch.bool
        )
        
        # Task 4: Diagnostic labels
        diagnostic_label = torch.tensor(
            self.merged_data['diagnostic_label'].values, dtype=torch.long
        )
        diagnostic_mask = torch.tensor(
            self.merged_data['diagnostic_mask'].values, dtype=torch.bool
        )
        
        # Patient IDs
        patient_ids = torch.tensor(
            self.merged_data['PATNO'].values, dtype=torch.long
        )
        
        # Create single Data object with all tasks
        data = Data(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
            # Task 1: Progression
            progression_time=progression_time,
            progression_event=progression_event,
            progression_mask=progression_mask,
            # Task 2: Conversion
            conversion_label=conversion_label,
            conversion_mask=conversion_mask,
            # Task 3: SAA
            saa_label=saa_label,
            saa_mask=saa_mask,
            # Task 4: Diagnostic
            diagnostic_label=diagnostic_label,
            diagnostic_mask=diagnostic_mask,
            # Metadata
            patient_ids=patient_ids,
            num_nodes=len(X)
        )
        
        print(f"  ✓ Created PyG Data object with {data.num_nodes} nodes")
        
        # Split into train/val/test based on patients
        print(f"\n[5/6] Splitting data (train/val/test)...")
        unique_patients = self.merged_data['PATNO'].unique()
        
        # First split: train+val vs test
        train_val_patients, test_patients = train_test_split(
            unique_patients,
            test_size=self.test_size,
            random_state=self.random_state,
            shuffle=True
        )
        
        # Second split: train vs val
        train_patients, val_patients = train_test_split(
            train_val_patients,
            test_size=self.val_size / (1 - self.test_size),
            random_state=self.random_state,
            shuffle=True
        )
        
        # Create masks for each split
        train_mask = self.merged_data['PATNO'].isin(train_patients).values
        val_mask = self.merged_data['PATNO'].isin(val_patients).values
        test_mask = self.merged_data['PATNO'].isin(test_patients).values
        
        print(f"  → Train: {train_mask.sum()} observations ({len(train_patients)} patients)")
        print(f"  → Val:   {val_mask.sum()} observations ({len(val_patients)} patients)")
        print(f"  → Test:  {test_mask.sum()} observations ({len(test_patients)} patients)")
        
        # Add split masks to data object
        data.train_mask = torch.tensor(train_mask, dtype=torch.bool)
        data.val_mask = torch.tensor(val_mask, dtype=torch.bool)
        data.test_mask = torch.tensor(test_mask, dtype=torch.bool)
        
        # Return as lists (single Data object per split for now)
        return [data], [data], [data]
    
    def save_processed_data(
        self,
        train_data: List[Data],
        val_data: List[Data],
        test_data: List[Data]
    ) -> None:
        """Save processed PyG data and metadata."""
        print("\n[6/6] Saving processed data...")
        
        # Save PyG data objects
        torch.save(train_data, self.output_dir / "multitask_train_data.pt")
        torch.save(val_data, self.output_dir / "multitask_val_data.pt")
        torch.save(test_data, self.output_dir / "multitask_test_data.pt")
        print(f"  ✓ Saved PyG Data objects to {self.output_dir}")
        
        # Save metadata
        metadata = {
            "data_preparation": {
                "date": "2025-10-14",
                "strategy": "Option A (Union with Masking)",
                "k_neighbors": self.k_neighbors,
                "test_size": self.test_size,
                "val_size": self.val_size,
                "random_state": self.random_state
            },
            "dataset_stats": {
                "total_observations": int(len(self.merged_data)),
                "unique_patients": int(self.merged_data['PATNO'].nunique()),
                "num_features": len(self.feature_columns),
                "train_size": int(train_data[0].train_mask.sum()),
                "val_size": int(val_data[0].val_mask.sum()),
                "test_size": int(test_data[0].test_mask.sum())
            },
            "task_labels": {
                "task1_progression": {
                    "total_with_labels": int(self.merged_data['progression_mask'].sum()),
                    "events_observed": int(self.merged_data[self.merged_data['progression_mask'] == 1]['event_observed'].sum()),
                    "censored": int(self.merged_data[self.merged_data['progression_mask'] == 1]['event_observed'].eq(0).sum())
                },
                "task2_conversion": {
                    "total_with_labels": int(self.merged_data['conversion_mask'].sum()),
                    "converters": int(self.merged_data[self.merged_data['conversion_mask'] == 1]['converted'].sum()),
                    "non_converters": int(self.merged_data[self.merged_data['conversion_mask'] == 1]['converted'].eq(0).sum())
                },
                "task3_saa": {
                    "total_with_labels": int(self.merged_data['saa_mask'].sum()),
                    "saa_positive": int(self.merged_data[self.merged_data['saa_mask'] == 1]['SAA_POSITIVE'].sum()),
                    "saa_negative": int(self.merged_data[self.merged_data['saa_mask'] == 1]['SAA_POSITIVE'].eq(0).sum()),
                    "class_weights": [0.822, 0.178]  # For WeightedBCELoss
                },
                "task4_diagnostic": {
                    "total_with_labels": int(self.merged_data['diagnostic_mask'].sum()),
                    "early_pd": int((self.merged_data['diagnostic_label'] == 0).sum()),
                    "prodromal": int((self.merged_data['diagnostic_label'] == 1).sum()),
                    "class_weights": [
                        float((self.merged_data['diagnostic_label'] == 1).sum() / len(self.merged_data)),
                        float((self.merged_data['diagnostic_label'] == 0).sum() / len(self.merged_data))
                    ]
                }
            },
            "feature_columns": self.feature_columns,
            "graph_structure": {
                "num_nodes": int(train_data[0].num_nodes),
                "num_edges": int(train_data[0].edge_index.shape[1]),
                "avg_degree": float(train_data[0].edge_index.shape[1] / train_data[0].num_nodes)
            }
        }
        
        with open(self.output_dir / "multitask_data_summary.json", 'w') as f:
            json.dump(metadata, f, indent=2)
        print(f"  ✓ Saved metadata to multitask_data_summary.json")
        
        # Save scaler
        import joblib
        joblib.dump(self.scaler, self.output_dir / "feature_scaler.pkl")
        print(f"  ✓ Saved feature scaler")
    
    def run(self) -> None:
        """Execute full data preparation pipeline."""
        try:
            self.load_datasets()
            self.merge_datasets()
            self.prepare_features()
            train_data, val_data, test_data = self.create_pyg_data_objects()
            self.save_processed_data(train_data, val_data, test_data)
            
            print("\n" + "=" * 80)
            print("✓ DATA PREPARATION COMPLETE")
            print("=" * 80)
            print(f"\nOutput files saved to: {self.output_dir}")
            print("  - multitask_train_data.pt")
            print("  - multitask_val_data.pt")
            print("  - multitask_test_data.pt")
            print("  - multitask_data_summary.json")
            print("  - feature_scaler.pkl")
            print("\nNext step: Phase 8.5 Architecture Design (giman_multitask.py)")
            
        except Exception as e:
            print(f"\n✗ ERROR during data preparation: {e}")
            raise


def main():
    """Main execution function."""
    # Set paths
    project_root = Path(__file__).resolve().parents[5]  # Fixed: Go back 5 levels from scripts/
    data_dir = project_root / "data"
    output_dir = project_root / "archive" / "development" / "phase8" / "subphase8_5_multitask_architecture" / "data"
    
    # Create preparator
    preparator = MultiTaskDataPreparator(
        data_dir=data_dir,
        output_dir=output_dir,
        k_neighbors=10,
        test_size=0.15,
        val_size=0.15,
        random_state=42
    )
    
    # Run pipeline
    preparator.run()


if __name__ == "__main__":
    main()
