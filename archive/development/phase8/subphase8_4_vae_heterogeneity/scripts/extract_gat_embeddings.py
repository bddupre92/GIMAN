"""
Extract 128-dim GAT embeddings from trained Phase 8.2 GIMAN-Progression model.

This script loads the trained Phase 8.2 survival model and extracts patient-level
embeddings from the final GAT layer (before the risk prediction head). These
embeddings capture the learned multimodal disease representations that achieved
C-index 0.9980 on phenoconversion prediction.

Author: GIMAN Research Team
Date: October 14, 2025
"""

import sys
from pathlib import Path
import json

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GATConv
from sklearn.neighbors import NearestNeighbors
from typing import Dict, Tuple

# Add project root to path
script_path = Path(__file__).resolve()
project_root = script_path.parents[3]
sys.path.insert(0, str(project_root))


class GIMANSurvivalGAT(nn.Module):
    """
    GIMAN-GAT architecture from Phase 8.2.
    
    This is a copy of the model from train_final_giman_survival.py to ensure
    we can load the trained weights correctly.
    """
    
    def __init__(
        self,
        in_features: int,
        hidden_dim: int = 128,
        num_heads: int = 4,
        num_layers: int = 3,
        dropout: float = 0.3
    ):
        super().__init__()
        
        self.in_features = in_features
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        
        # GAT layers
        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        
        # Input layer
        self.convs.append(GATConv(in_features, hidden_dim, heads=num_heads, dropout=dropout))
        self.batch_norms.append(nn.BatchNorm1d(hidden_dim * num_heads))
        
        # Hidden layers
        for _ in range(num_layers - 2):
            self.convs.append(
                GATConv(hidden_dim * num_heads, hidden_dim, heads=num_heads, dropout=dropout)
            )
            self.batch_norms.append(nn.BatchNorm1d(hidden_dim * num_heads))
        
        # Output layer (single head)
        self.convs.append(GATConv(hidden_dim * num_heads, hidden_dim, heads=1, dropout=dropout))
        self.batch_norms.append(nn.BatchNorm1d(hidden_dim))
        
        # Survival risk predictor
        self.risk_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)  # Risk score (log hazard)
        )
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, data: Data, return_embeddings: bool = False) -> torch.Tensor:
        """
        Forward pass with option to return embeddings.
        
        Args:
            data: PyG Data object
            return_embeddings: If True, return 128-dim embeddings instead of risk scores
            
        Returns:
            Either risk scores (default) or 128-dim embeddings
        """
        x, edge_index = data.x, data.edge_index
        
        # GAT layers with residual connections
        for i, (conv, bn) in enumerate(zip(self.convs[:-1], self.batch_norms[:-1])):
            x_new = conv(x, edge_index)
            x_new = bn(x_new)
            x_new = F.elu(x_new)
            x_new = self.dropout(x_new)
            
            # Residual connection (if dimensions match)
            if i > 0 and x.size(-1) == x_new.size(-1):
                x = x + x_new
            else:
                x = x_new
        
        # Final GAT layer - THIS IS OUR 128-DIM EMBEDDING
        embeddings = self.convs[-1](x, edge_index)
        embeddings = self.batch_norms[-1](embeddings)
        embeddings = F.elu(embeddings)
        
        if return_embeddings:
            return embeddings
        
        # Risk prediction
        risk_scores = self.risk_head(embeddings)
        return risk_scores.squeeze(-1)


class EmbeddingExtractor:
    """Extract embeddings from trained GIMAN-Progression model."""
    
    def __init__(self, device: str = "cpu"):
        self.device = torch.device(device)
        self.model = None
        self.data = None
        
        # Paths
        self.project_root = Path(__file__).resolve().parents[5]  # Go up to project root
        self.model_path = self.project_root / "outputs" / "phase8_2_final_training" / "giman_survival_final.pth"
        self.data_path = self.project_root / "data" / "03_prodromal" / "final_training_dataset" / "unified_longitudinal_early_pd.csv"
        self.output_dir = self.project_root / "data" / "05_embeddings"
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def load_model(self) -> None:
        """Load trained Phase 8.2 model."""
        print("=" * 80)
        print("LOADING PHASE 8.2 GIMAN-PROGRESSION MODEL")
        print("=" * 80)
        print(f"Model path: {self.model_path}")
        
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found: {self.model_path}")
        
        # Load checkpoint
        checkpoint = torch.load(self.model_path, map_location=self.device)
        
        # Get model config
        model_config = checkpoint['model_config']
        print(f"\nModel Configuration:")
        print(f"  Input features: {model_config['in_features']}")
        print(f"  Hidden dim: {model_config['hidden_dim']}")
        print(f"  Num heads: {model_config['num_heads']}")
        print(f"  Num layers: {model_config['num_layers']}")
        print(f"  Dropout: {model_config['dropout']}")
        print(f"\nModel Performance:")
        print(f"  Test C-index: {checkpoint['best_test_c_index']:.4f}")
        
        # Initialize model
        self.model = GIMANSurvivalGAT(**model_config).to(self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        print(f"\n✅ Model loaded successfully")
        print(f"   Embedding dimension: {model_config['hidden_dim']}")
    
    def load_data(self) -> Tuple[pd.DataFrame, torch.Tensor, list]:
        """
        Load Phase 8.2 data and prepare for embedding extraction.
        
        Returns:
            DataFrame, feature matrix, patient IDs
        """
        print("\n" + "=" * 80)
        print("LOADING PHASE 8.2 DATA")
        print("=" * 80)
        print(f"Data path: {self.data_path}")
        
        if not self.data_path.exists():
            raise FileNotFoundError(f"Data not found: {self.data_path}")
        
        # Load data
        df = pd.read_csv(self.data_path)
        print(f"\nLoaded {len(df)} observations from {df['PATNO'].nunique()} patients")
        
        # Extract features (same as Phase 8.2)
        exclude_cols = [
            'PATNO', 'EVENT_ID', 'PHENOCONVERSION', 'TIME_TO_EVENT',
            'time_to_event', 'phenoconverted', 'landmark_month', 
            'original_time', 'original_event', 'cohort'
        ]
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        
        X = df[feature_cols].values.astype(np.float32)
        patient_ids = df['PATNO'].values
        
        print(f"Feature matrix: {X.shape}")
        print(f"Features: {len(feature_cols)}")
        
        return df, torch.tensor(X, dtype=torch.float32), patient_ids, feature_cols
    
    def build_graph(self, X: torch.Tensor, k: int = 10) -> torch.Tensor:
        """
        Build k-NN graph (same as Phase 8.2).
        
        Args:
            X: Feature matrix
            k: Number of nearest neighbors
            
        Returns:
            edge_index tensor
        """
        print(f"\nBuilding k-NN graph (k={k})...")
        
        # Find k nearest neighbors
        nbrs = NearestNeighbors(n_neighbors=k+1, metric='cosine', n_jobs=-1)
        nbrs.fit(X.numpy())
        distances, indices = nbrs.kneighbors(X.numpy())
        
        # Build edge list (exclude self-loops)
        edge_list = []
        for i in range(len(X)):
            for j in indices[i][1:]:  # Skip first neighbor (self)
                edge_list.append([i, j])
        
        edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
        
        print(f"Graph: {len(X)} nodes, {edge_index.shape[1]} edges")
        print(f"Average degree: {edge_index.shape[1] / len(X):.1f}")
        
        return edge_index
    
    def extract_embeddings(self) -> Dict:
        """
        Extract 128-dim embeddings for all patients.
        
        Returns:
            Dictionary with embeddings and metadata
        """
        print("\n" + "=" * 80)
        print("EXTRACTING 128-DIM GAT EMBEDDINGS")
        print("=" * 80)
        
        # Load data
        df, X, patient_ids, feature_cols = self.load_data()
        
        # Build graph
        edge_index = self.build_graph(X)
        
        # Create PyG Data object
        data = Data(x=X, edge_index=edge_index).to(self.device)
        
        # Extract embeddings
        print("\nExtracting embeddings...")
        with torch.no_grad():
            embeddings = self.model(data, return_embeddings=True)
        
        embeddings_np = embeddings.cpu().numpy()
        
        print(f"\nEmbeddings shape: {embeddings_np.shape}")
        print(f"Embedding dimension: {embeddings_np.shape[1]}")
        print(f"Number of samples: {embeddings_np.shape[0]}")
        
        # Create embeddings DataFrame
        embedding_cols = [f"EMB_{i+1}" for i in range(embeddings_np.shape[1])]
        embeddings_df = pd.DataFrame(embeddings_np, columns=embedding_cols)
        
        # Add patient IDs and metadata
        embeddings_df.insert(0, 'PATNO', patient_ids)
        if 'landmark_month' in df.columns:
            embeddings_df.insert(1, 'LANDMARK_MONTH', df['landmark_month'].values)
        
        # Add clinical metadata
        metadata_cols = ['phenoconverted', 'time_to_event', 'cohort']
        for col in metadata_cols:
            if col in df.columns:
                embeddings_df[col] = df[col].values
        
        # Add genetic features if available
        genetic_cols = [col for col in df.columns if col in ['LRRK2', 'GBA', 'SNCA', 'APOE']]
        for col in genetic_cols:
            embeddings_df[col] = df[col].values
        
        # Add UPDRS scores if available
        updrs_cols = [col for col in df.columns if 'UPDRS' in col or 'MoCA' in col]
        for col in updrs_cols:
            if col in df.columns:
                embeddings_df[col] = df[col].values
        
        return {
            'embeddings_df': embeddings_df,
            'embedding_dim': embeddings_np.shape[1],
            'n_samples': embeddings_np.shape[0],
            'n_patients': df['PATNO'].nunique(),
            'feature_names': feature_cols
        }
    
    def save_embeddings(self, results: Dict) -> None:
        """Save embeddings and metadata."""
        print("\n" + "=" * 80)
        print("SAVING EMBEDDINGS")
        print("=" * 80)
        
        embeddings_df = results['embeddings_df']
        
        # Save embeddings
        embeddings_path = self.output_dir / "giman_gat_embeddings.csv"
        embeddings_df.to_csv(embeddings_path, index=False)
        print(f"\n✅ Saved embeddings: {embeddings_path}")
        print(f"   Shape: {embeddings_df.shape}")
        print(f"   Columns: {list(embeddings_df.columns[:5])} ... {list(embeddings_df.columns[-3:])}")
        
        # Save metadata
        metadata = {
            'extraction_date': '2025-10-14',
            'source_model': 'Phase 8.2 GIMAN-Progression (C-index 0.9980)',
            'model_path': str(self.model_path),
            'embedding_dimension': results['embedding_dim'],
            'n_samples': results['n_samples'],
            'n_patients': results['n_patients'],
            'n_features': len(results['feature_names']),
            'feature_names': results['feature_names']
        }
        
        metadata_path = self.output_dir / "giman_gat_embeddings_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        print(f"\n✅ Saved metadata: {metadata_path}")
        
        # Print summary statistics
        print("\n" + "=" * 80)
        print("EMBEDDING SUMMARY STATISTICS")
        print("=" * 80)
        
        embedding_cols = [col for col in embeddings_df.columns if col.startswith('EMB_')]
        embedding_stats = embeddings_df[embedding_cols].describe()
        
        print(f"\nEmbedding statistics:")
        print(f"  Mean: {embedding_stats.loc['mean'].mean():.4f}")
        print(f"  Std: {embedding_stats.loc['std'].mean():.4f}")
        print(f"  Min: {embedding_stats.loc['min'].min():.4f}")
        print(f"  Max: {embedding_stats.loc['max'].max():.4f}")
        
        # Check for phenoconversion rates
        if 'phenoconverted' in embeddings_df.columns:
            conv_rate = embeddings_df['phenoconverted'].mean()
            print(f"\nPhenoconversion rate: {conv_rate:.1%}")
            print(f"  Converters: {embeddings_df['phenoconverted'].sum()}")
            print(f"  Non-converters: {(~embeddings_df['phenoconverted'].astype(bool)).sum()}")


def main():
    """Main extraction pipeline."""
    print("=" * 80)
    print("PHASE 8.4: GAT EMBEDDING EXTRACTION")
    print("=" * 80)
    print("\nExtracting 128-dim embeddings from Phase 8.2 GIMAN-Progression model")
    print("for use in VAE Heterogeneity Analysis\n")
    
    # Check for GPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}\n")
    
    # Initialize extractor
    extractor = EmbeddingExtractor(device=device)
    
    # Load model
    extractor.load_model()
    
    # Extract embeddings
    results = extractor.extract_embeddings()
    
    # Save results
    extractor.save_embeddings(results)
    
    print("\n" + "=" * 80)
    print("✅ EMBEDDING EXTRACTION COMPLETE")
    print("=" * 80)
    print("\nNext steps:")
    print("  1. Verify embedding quality (check for NaNs, outliers)")
    print("  2. Adapt Phase 4 VaDER VAE for 128-dim static embeddings")
    print("  3. Train VAE to learn continuous disease heterogeneity")
    print("\nOutputs:")
    print(f"  - Embeddings: data/05_embeddings/giman_gat_embeddings.csv")
    print(f"  - Metadata: data/05_embeddings/giman_gat_embeddings_metadata.json")


if __name__ == "__main__":
    main()
