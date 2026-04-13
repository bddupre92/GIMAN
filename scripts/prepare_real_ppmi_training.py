"""
Real PPMI Cohort Training Preparation.

This module prepares the 127-patient real PPMI cohort for GIMAN dual model training.
It handles data splitting, patient similarity graph construction, feature normalization,
and PyTorch DataLoader creation.

Features:
    - Stratified train/val/test splitting (70/15/15)
    - K-nearest neighbors patient similarity graph
    - Z-score feature normalization
    - PyTorch Geometric Data objects
    - DataLoader with graph batching support

Author: GIMAN Development Team
Date: October 8, 2025
Version: 8.1.0 - Week 2 Real PPMI Preparation
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import KNNImputer
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import cosine_similarity
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader as GeometricDataLoader

# Import configuration
import sys
sys.path.append(str(Path(__file__).parent.parent))
from src.utils.config_loader import load_config


class RealPPMIDataPreparator:
    """
    Prepare real PPMI cohort for GIMAN dual model training.
    
    This class handles all data preparation steps including splitting,
    normalization, graph construction, and DataLoader creation.
    
    Args:
        config_path: Path to configuration YAML file
        data_dir: Base directory for data files
        output_dir: Directory for saving prepared data
        random_seed: Random seed for reproducibility
        
    Example:
        >>> prep = RealPPMIDataPreparator("configs/real_ppmi_dual_model.yaml")
        >>> prep.prepare_data()
        >>> train_loader = prep.get_train_loader()
    """
    
    def __init__(
        self,
        config_path: str = "configs/real_ppmi_dual_model.yaml",
        data_dir: str = "data/02_processed",
        output_dir: str = "data/02_processed/training_ready",
        random_seed: int = 42,
    ):
        self.config_path = Path(config_path)
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.random_seed = random_seed
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load configuration
        self.config = load_config(self.config_path)
        
        # Set random seeds
        np.random.seed(random_seed)
        torch.manual_seed(random_seed)
        
        # Initialize data structures
        self.df = None
        self.train_df = None
        self.val_df = None
        self.test_df = None
        self.scaler = None
        self.imputer = None
        self.feature_columns = None
        
        print(f"[INIT] Real PPMI Data Preparator initialized")
        print(f"   Config: {self.config_path}")
        print(f"   Data dir: {self.data_dir}")
        print(f"   Output dir: {self.output_dir}")
        print(f"   Random seed: {random_seed}")
    
    def load_cohort(self) -> pd.DataFrame:
        """
        Load real PPMI cohort from CSV.
        
        Returns:
            DataFrame with cohort data
        """
        cohort_path = self.data_dir / "enhanced_real_ppmi_cohort.csv"
        
        print(f"\n[LOAD] Loading real PPMI cohort...")
        print(f"   Path: {cohort_path}")
        
        if not cohort_path.exists():
            raise FileNotFoundError(f"Cohort file not found: {cohort_path}")
        
        self.df = pd.read_csv(cohort_path)
        
        print(f"   ✓ Loaded: {len(self.df)} patients × {len(self.df.columns)} columns")
        
        return self.df
    
    def identify_feature_columns(self) -> List[str]:
        """
        Identify feature columns for model input.
        
        Excludes: PATNO, SOURCE, completeness_score, high_quality, 
                 COHORT_DEFINITION (categorical), and any other metadata.
        
        Returns:
            List of feature column names
        """
        print(f"\n[FEATURES] Identifying feature columns...")
        
        # Columns to exclude
        exclude_columns = [
            'PATNO', 'SOURCE', 'completeness_score', 'high_quality',
            'COHORT_DEFINITION',  # Will be used for stratification, not features
        ]
        
        # Get all numeric columns
        numeric_columns = self.df.select_dtypes(include=[np.number]).columns.tolist()
        
        # Remove excluded columns
        self.feature_columns = [col for col in numeric_columns if col not in exclude_columns]
        
        print(f"   ✓ Identified {len(self.feature_columns)} feature columns")
        print(f"   Feature groups:")
        
        # Categorize features for clarity
        demographics = [c for c in self.feature_columns if c in ['SEX', 'AGE_COMPUTED']]
        clinical = [c for c in self.feature_columns if c in ['NP3TOT', 'NHY', 'MoCA']]
        genetics = [c for c in self.feature_columns if any(g in c for g in ['LRRK2', 'GBA', 'APOE', 'SNCA', 'GENETIC'])]
        imaging = [c for c in self.feature_columns if any(i in c for i in ['CAUDATE', 'PUTAMEN', 'STRIATUM', 'ABNORMAL'])]
        biomarkers = [c for c in self.feature_columns if any(b in c for b in ['TAU', 'UPSIT', 'ALPHA_SYN'])]
        
        print(f"      Demographics: {len(demographics)}")
        print(f"      Clinical: {len(clinical)}")
        print(f"      Genetics: {len(genetics)}")
        print(f"      Imaging: {len(imaging)}")
        print(f"      Biomarkers: {len(biomarkers)}")
        
        return self.feature_columns
    
    def split_data(
        self,
        train_ratio: float = 0.70,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Split data into train/val/test sets with stratification.
        
        Stratifies by COHORT_DEFINITION to maintain PD/HC balance.
        
        Args:
            train_ratio: Training set ratio (default: 0.70)
            val_ratio: Validation set ratio (default: 0.15)
            test_ratio: Test set ratio (default: 0.15)
            
        Returns:
            Tuple of (train_df, val_df, test_df)
        """
        print(f"\n[SPLIT] Splitting data ({train_ratio:.0%}/{val_ratio:.0%}/{test_ratio:.0%})...")
        
        # Verify ratios sum to 1.0
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 0.01, "Ratios must sum to 1.0"
        
        # Get stratification column
        if 'COHORT_DEFINITION' in self.df.columns:
            stratify_col = self.df['COHORT_DEFINITION']
            print(f"   Stratifying by: COHORT_DEFINITION")
            print(f"   Class distribution:")
            for cohort, count in stratify_col.value_counts().items():
                print(f"      {cohort}: {count} ({count/len(self.df):.1%})")
        else:
            stratify_col = None
            print(f"   ⚠️  No stratification column found, using random split")
        
        # First split: train vs (val + test)
        train_df, temp_df = train_test_split(
            self.df,
            train_size=train_ratio,
            random_state=self.random_seed,
            stratify=stratify_col,
        )
        
        # Second split: val vs test
        val_ratio_adjusted = val_ratio / (val_ratio + test_ratio)
        val_df, test_df = train_test_split(
            temp_df,
            train_size=val_ratio_adjusted,
            random_state=self.random_seed,
            stratify=temp_df['COHORT_DEFINITION'] if 'COHORT_DEFINITION' in temp_df.columns else None,
        )
        
        self.train_df = train_df
        self.val_df = val_df
        self.test_df = test_df
        
        print(f"   ✓ Split complete:")
        print(f"      Train: {len(train_df)} patients ({len(train_df)/len(self.df):.1%})")
        print(f"      Val:   {len(val_df)} patients ({len(val_df)/len(self.df):.1%})")
        print(f"      Test:  {len(test_df)} patients ({len(test_df)/len(self.df):.1%})")
        
        return train_df, val_df, test_df
    
    def impute_missing_values(self) -> KNNImputer:
        """
        Impute missing values using K-Nearest Neighbors.
        
        Fits imputer on training set only, then transforms all sets.
        
        Returns:
            Fitted KNNImputer
        """
        print(f"\n[IMPUTE] Handling missing values with KNN imputation (k=5)...")
        
        # Check for missing values
        train_missing = self.train_df[self.feature_columns].isnull().sum().sum()
        val_missing = self.val_df[self.feature_columns].isnull().sum().sum()
        test_missing = self.test_df[self.feature_columns].isnull().sum().sum()
        
        print(f"   Missing values before imputation:")
        print(f"      Train: {train_missing}")
        print(f"      Val:   {val_missing}")
        print(f"      Test:  {test_missing}")
        
        if train_missing + val_missing + test_missing == 0:
            print(f"   ✓ No missing values found, skipping imputation")
            return None
        
        # Fit imputer on training data only
        self.imputer = KNNImputer(n_neighbors=5, weights='distance')
        self.imputer.fit(self.train_df[self.feature_columns])
        
        # Transform all splits
        self.train_df[self.feature_columns] = self.imputer.transform(
            self.train_df[self.feature_columns]
        )
        self.val_df[self.feature_columns] = self.imputer.transform(
            self.val_df[self.feature_columns]
        )
        self.test_df[self.feature_columns] = self.imputer.transform(
            self.test_df[self.feature_columns]
        )
        
        # Verify no missing values remain
        train_missing_after = self.train_df[self.feature_columns].isnull().sum().sum()
        val_missing_after = self.val_df[self.feature_columns].isnull().sum().sum()
        test_missing_after = self.test_df[self.feature_columns].isnull().sum().sum()
        
        print(f"   ✓ Imputation complete")
        print(f"      Train: {train_missing_after} missing (was {train_missing})")
        print(f"      Val:   {val_missing_after} missing (was {val_missing})")
        print(f"      Test:  {test_missing_after} missing (was {test_missing})")
        
        return self.imputer
    
    def normalize_features(self) -> StandardScaler:
        """
        Normalize features using z-score standardization.
        
        Fits scaler on training set only, then transforms all sets.
        
        Returns:
            Fitted StandardScaler
        """
        print(f"\n[NORMALIZE] Z-score standardization...")
        
        # Fit scaler on training data only
        self.scaler = StandardScaler()
        self.scaler.fit(self.train_df[self.feature_columns])
        
        # Transform all splits
        self.train_df[self.feature_columns] = self.scaler.transform(
            self.train_df[self.feature_columns]
        )
        self.val_df[self.feature_columns] = self.scaler.transform(
            self.val_df[self.feature_columns]
        )
        self.test_df[self.feature_columns] = self.scaler.transform(
            self.test_df[self.feature_columns]
        )
        
        print(f"   ✓ Features normalized")
        print(f"      Mean (train): {self.train_df[self.feature_columns].mean().mean():.6f}")
        print(f"      Std (train): {self.train_df[self.feature_columns].std().mean():.6f}")
        
        return self.scaler
    
    def build_patient_similarity_graph(
        self,
        df: pd.DataFrame,
        k_neighbors: int = 10,
        similarity_metric: str = 'cosine',
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Build patient similarity graph using k-nearest neighbors.
        
        Args:
            df: DataFrame with patient features
            k_neighbors: Number of neighbors per patient
            similarity_metric: 'cosine' or 'euclidean'
            
        Returns:
            Tuple of (edge_index, edge_weights)
                edge_index: [2, num_edges] array of edges
                edge_weights: [num_edges] array of similarity scores
        """
        # Extract features
        X = df[self.feature_columns].values
        
        # Build k-NN graph
        if similarity_metric == 'cosine':
            # Use cosine similarity
            knn = NearestNeighbors(n_neighbors=k_neighbors + 1, metric='cosine')
        else:
            # Use euclidean distance
            knn = NearestNeighbors(n_neighbors=k_neighbors + 1, metric='euclidean')
        
        knn.fit(X)
        distances, indices = knn.kneighbors(X)
        
        # Build edge list (exclude self-loops)
        edge_list = []
        edge_weights = []
        
        for i in range(len(df)):
            for j_idx in range(1, k_neighbors + 1):  # Skip first neighbor (self)
                j = indices[i, j_idx]
                dist = distances[i, j_idx]
                
                # Convert distance to similarity
                if similarity_metric == 'cosine':
                    similarity = 1.0 - dist  # Cosine distance to similarity
                else:
                    similarity = 1.0 / (1.0 + dist)  # Euclidean distance to similarity
                
                edge_list.append([i, j])
                edge_weights.append(similarity)
        
        # Convert to numpy arrays
        edge_index = np.array(edge_list, dtype=np.int64).T  # [2, num_edges]
        edge_weights = np.array(edge_weights, dtype=np.float32)
        
        return edge_index, edge_weights
    
    def create_pytorch_geometric_data(
        self,
        df: pd.DataFrame,
        k_neighbors: int = 10,
        similarity_metric: str = 'cosine',
    ) -> Data:
        """
        Create PyTorch Geometric Data object for graph neural network.
        
        Args:
            df: DataFrame with patient features
            k_neighbors: Number of neighbors for graph construction
            similarity_metric: Similarity metric for graph
            
        Returns:
            PyTorch Geometric Data object
        """
        # Extract features
        x = torch.tensor(
            df[self.feature_columns].values,
            dtype=torch.float32
        )
        
        # Build patient similarity graph
        edge_index, edge_weights = self.build_patient_similarity_graph(
            df, k_neighbors, similarity_metric
        )
        
        # Convert to PyTorch tensors
        edge_index = torch.tensor(edge_index, dtype=torch.long)
        edge_attr = torch.tensor(edge_weights, dtype=torch.float32).unsqueeze(1)
        
        # Create Data object
        data = Data(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
        )
        
        # Add patient IDs for tracking
        data.patient_ids = df['PATNO'].values
        
        # Add cohort definition labels if available
        if 'COHORT_DEFINITION' in df.columns:
            cohort_labels = (df['COHORT_DEFINITION'] == 'Parkinson\'s Disease').astype(int).values
            data.y = torch.tensor(cohort_labels, dtype=torch.long)
        
        return data
    
    def prepare_data(self):
        """
        Execute complete data preparation pipeline.
        
        Steps:
            1. Load cohort
            2. Identify features
            3. Split data
            4. Normalize features
            5. Build graphs for each split
            6. Save prepared data
        """
        print("\n" + "=" * 70)
        print("REAL PPMI COHORT TRAINING PREPARATION")
        print("=" * 70)
        
        # Load cohort
        self.load_cohort()
        
        # Identify features
        self.identify_feature_columns()
        
        # Split data
        self.split_data(
            train_ratio=self.config.data.train_split,
            val_ratio=self.config.data.val_split,
            test_ratio=self.config.data.test_split,
        )
        
        # Impute missing values
        self.impute_missing_values()
        
        # Normalize features
        self.normalize_features()
        
        # Build graphs
        print(f"\n[GRAPH] Building patient similarity graphs...")
        k_neighbors = self.config.data.graph_construction.k_neighbors
        similarity_metric = self.config.data.graph_construction.similarity_metric
        
        print(f"   Method: k-NN (k={k_neighbors})")
        print(f"   Similarity: {similarity_metric}")
        
        self.train_data = self.create_pytorch_geometric_data(
            self.train_df, k_neighbors, similarity_metric
        )
        print(f"   ✓ Train graph: {self.train_data.num_nodes} nodes, {self.train_data.num_edges} edges")
        
        self.val_data = self.create_pytorch_geometric_data(
            self.val_df, k_neighbors, similarity_metric
        )
        print(f"   ✓ Val graph: {self.val_data.num_nodes} nodes, {self.val_data.num_edges} edges")
        
        self.test_data = self.create_pytorch_geometric_data(
            self.test_df, k_neighbors, similarity_metric
        )
        print(f"   ✓ Test graph: {self.test_data.num_nodes} nodes, {self.test_data.num_edges} edges")
        
        # Save prepared data
        self.save_prepared_data()
        
        print("\n" + "=" * 70)
        print("✅ REAL PPMI COHORT PREPARATION COMPLETE")
        print("=" * 70)
        print(f"\n📁 Prepared data saved to: {self.output_dir}")
    
    def save_prepared_data(self):
        """Save prepared data to disk."""
        print(f"\n[SAVE] Saving prepared data...")
        
        # Save PyTorch Geometric Data objects
        torch.save(self.train_data, self.output_dir / "train_data.pt")
        torch.save(self.val_data, self.output_dir / "val_data.pt")
        torch.save(self.test_data, self.output_dir / "test_data.pt")
        print(f"   ✓ Saved: train_data.pt, val_data.pt, test_data.pt")
        
        # Save scaler and imputer
        import joblib
        joblib.dump(self.scaler, self.output_dir / "feature_scaler.pkl")
        print(f"   ✓ Saved: feature_scaler.pkl")
        
        if self.imputer is not None:
            joblib.dump(self.imputer, self.output_dir / "feature_imputer.pkl")
            print(f"   ✓ Saved: feature_imputer.pkl")
        
        # Save feature columns
        with open(self.output_dir / "feature_columns.json", 'w') as f:
            json.dump(self.feature_columns, f, indent=2)
        print(f"   ✓ Saved: feature_columns.json")
        
        # Save split indices
        split_info = {
            'train_patnos': self.train_df['PATNO'].tolist(),
            'val_patnos': self.val_df['PATNO'].tolist(),
            'test_patnos': self.test_df['PATNO'].tolist(),
            'train_size': len(self.train_df),
            'val_size': len(self.val_df),
            'test_size': len(self.test_df),
            'random_seed': self.random_seed,
        }
        with open(self.output_dir / "split_info.json", 'w') as f:
            json.dump(split_info, f, indent=2)
        print(f"   ✓ Saved: split_info.json")
        
        # Save preparation metadata
        metadata = {
            'preparation_date': pd.Timestamp.now().isoformat(),
            'cohort_size': len(self.df),
            'num_features': len(self.feature_columns),
            'train_size': len(self.train_df),
            'val_size': len(self.val_df),
            'test_size': len(self.test_df),
            'k_neighbors': self.config.data.graph_construction.k_neighbors,
            'similarity_metric': self.config.data.graph_construction.similarity_metric,
            'normalization': 'z-score',
            'random_seed': self.random_seed,
        }
        with open(self.output_dir / "preparation_metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)
        print(f"   ✓ Saved: preparation_metadata.json")
    
    def get_data_objects(self) -> Tuple[Data, Data, Data]:
        """
        Get prepared PyTorch Geometric Data objects.
        
        Returns:
            Tuple of (train_data, val_data, test_data)
        """
        return self.train_data, self.val_data, self.test_data


def main():
    """Execute real PPMI cohort training preparation."""
    
    # Initialize preparator
    prep = RealPPMIDataPreparator(
        config_path="configs/real_ppmi_dual_model.yaml",
        data_dir="data/02_processed",
        output_dir="data/02_processed/training_ready",
        random_seed=42,
    )
    
    # Prepare data
    prep.prepare_data()
    
    # Print summary
    train_data, val_data, test_data = prep.get_data_objects()
    
    print("\n📊 Preparation Summary:")
    print(f"   Train: {train_data.num_nodes} patients, {train_data.num_edges} edges")
    print(f"   Val:   {val_data.num_nodes} patients, {val_data.num_edges} edges")
    print(f"   Test:  {test_data.num_nodes} patients, {test_data.num_edges} edges")
    print(f"   Features: {train_data.num_features}")
    print(f"\n🚀 Ready for GIMAN dual model training!")


if __name__ == '__main__':
    main()
