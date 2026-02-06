"""
Extract latent codes from best trained VAE model.

Loads the best VAE model (12-dim latent space) and extracts latent codes
for all patients. These codes represent continuous disease heterogeneity
signatures that will be correlated with biological features.

Author: GIMAN Research Team
Date: October 14, 2025
"""

import sys
from pathlib import Path
import json

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

# Add models directory to path
models_path = Path(__file__).resolve().parent.parent / "models"
sys.path.insert(0, str(models_path))

from heterogeneity_vae import build_heterogeneity_vae


class LatentCodeExtractor:
    """Extract and save latent codes from trained VAE."""
    
    def __init__(self, latent_dim: int = 12, device: str = "cpu"):
        self.latent_dim = latent_dim
        self.device = torch.device(device)
        
        # Paths
        self.project_root = Path(__file__).resolve().parents[5]
        self.embeddings_path = self.project_root / "data" / "05_embeddings" / "giman_gat_embeddings.csv"
        self.checkpoint_path = self.project_root / "archive" / "development" / "phase8" / "subphase8_4_vae_heterogeneity" / "checkpoints" / f"best_vae_latent{latent_dim}.pth"
        self.output_dir = self.project_root / "data" / "05_embeddings"
        
        self.model = None
        self.scaler = StandardScaler()
    
    def load_model(self) -> None:
        """Load trained VAE model and scaler."""
        print("=" * 80)
        print(f"LOADING BEST VAE MODEL (latent_dim={self.latent_dim})")
        print("=" * 80)
        print(f"Checkpoint: {self.checkpoint_path}\n")
        
        # Load checkpoint
        checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
        
        # Build model
        self.model = build_heterogeneity_vae(
            input_dim=128,
            latent_dim=self.latent_dim,
            dropout=0.3,
            device=str(self.device)
        )
        
        # Load weights
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        # Load scaler
        self.scaler.mean_ = checkpoint['scaler_mean']
        self.scaler.scale_ = checkpoint['scaler_scale']
        
        print(f"✅ Model loaded successfully")
        print(f"   Epoch: {checkpoint['epoch']}")
        print(f"   Best val loss: {checkpoint['best_val_loss']:.4f}")
        print(f"   Val recon loss: {checkpoint['val_metrics']['recon_loss']:.4f}")
        print(f"   Val KL loss: {checkpoint['val_metrics']['kl_loss']:.4f}")
    
    def extract_latent_codes(self) -> pd.DataFrame:
        """Extract latent codes for all patients."""
        print("\n" + "=" * 80)
        print("EXTRACTING LATENT CODES")
        print("=" * 80)
        
        # Load embeddings
        df = pd.read_csv(self.embeddings_path)
        print(f"\nLoaded: {len(df)} observations from {df['PATNO'].nunique()} patients")
        
        # Extract embedding columns
        embedding_cols = [col for col in df.columns if col.startswith('EMB_')]
        X = df[embedding_cols].values
        
        # Standardize (using training scaler)
        print("Standardizing embeddings...")
        X_scaled = self.scaler.transform(X)
        
        # Convert to tensor
        X_tensor = torch.tensor(X_scaled, dtype=torch.float32).to(self.device)
        
        # Extract latent codes
        print("Encoding to latent space...")
        with torch.no_grad():
            z, mu, logvar = self.model.encode(X_tensor, return_dist=True)
        
        print(f"✅ Extracted latent codes:")
        print(f"   Shape: {z.shape}")
        print(f"   Latent dimension: {self.latent_dim}")
        
        # Create latent codes DataFrame
        latent_cols = [f"LATENT_{i+1}" for i in range(self.latent_dim)]
        latent_df = pd.DataFrame(z, columns=latent_cols)
        
        # Add mu and logvar for analysis
        mu_cols = [f"MU_{i+1}" for i in range(self.latent_dim)]
        logvar_cols = [f"LOGVAR_{i+1}" for i in range(self.latent_dim)]
        mu_df = pd.DataFrame(mu, columns=mu_cols)
        logvar_df = pd.DataFrame(logvar, columns=logvar_cols)
        
        # Add patient metadata
        metadata_cols = ['PATNO', 'LANDMARK_MONTH', 'phenoconverted', 'time_to_event', 'cohort']
        metadata_cols = [col for col in metadata_cols if col in df.columns]
        
        result_df = pd.concat([
            df[metadata_cols].reset_index(drop=True),
            latent_df,
            mu_df,
            logvar_df
        ], axis=1)
        
        # Add genetic features if available
        genetic_cols = [col for col in df.columns if col in ['LRRK2', 'GBA', 'SNCA', 'APOE', 'GENETIC_RISK_SCORE']]
        for col in genetic_cols:
            result_df[col] = df[col].values
        
        # Add UPDRS scores if available
        updrs_cols = [col for col in df.columns if 'UPDRS' in col or 'MoCA' in col or 'SCHWAB' in col]
        for col in updrs_cols:
            if col in df.columns:
                result_df[col] = df[col].values
        
        return result_df
    
    def analyze_latent_structure(self, latent_df: pd.DataFrame) -> None:
        """Analyze latent code structure."""
        print("\n" + "=" * 80)
        print("LATENT SPACE ANALYSIS")
        print("=" * 80)
        
        latent_cols = [col for col in latent_df.columns if col.startswith('LATENT_')]
        latent_codes = latent_df[latent_cols].values
        
        # Summary statistics
        print(f"\nLatent code statistics:")
        print(f"  Mean: {latent_codes.mean():.4f}")
        print(f"  Std: {latent_codes.std():.4f}")
        print(f"  Min: {latent_codes.min():.4f}")
        print(f"  Max: {latent_codes.max():.4f}")
        
        # Per-dimension statistics
        print(f"\nPer-dimension statistics:")
        for i, col in enumerate(latent_cols):
            values = latent_df[col].values
            print(f"  {col}: mean={values.mean():.4f}, std={values.std():.4f}, "
                  f"range=[{values.min():.4f}, {values.max():.4f}]")
        
        # Correlation between latent dimensions
        print(f"\nLatent dimension correlations:")
        corr = np.corrcoef(latent_codes.T)
        max_corr = np.max(np.abs(corr[np.triu_indices_from(corr, k=1)]))
        print(f"  Max abs correlation: {max_corr:.4f}")
        if max_corr > 0.5:
            print(f"  ⚠️  High correlation detected (>{0.5:.1f})")
        else:
            print(f"  ✅ Dimensions are well-separated")
        
        # Visualize latent space structure
        self.visualize_latent_space(latent_df)
    
    def visualize_latent_space(self, latent_df: pd.DataFrame) -> None:
        """Generate latent space visualizations."""
        print("\n" + "=" * 80)
        print("GENERATING LATENT SPACE VISUALIZATIONS")
        print("=" * 80)
        
        latent_cols = [col for col in latent_df.columns if col.startswith('LATENT_')]
        latent_codes = latent_df[latent_cols].values
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        fig.suptitle(f'Latent Space Structure (dim={self.latent_dim})', 
                     fontsize=16, fontweight='bold')
        
        # 1. Distribution of latent dimensions
        ax = axes[0, 0]
        ax.boxplot([latent_df[col].values for col in latent_cols], 
                   labels=[f'L{i+1}' for i in range(len(latent_cols))],
                   patch_artist=True)
        ax.set_xlabel('Latent Dimension')
        ax.set_ylabel('Value')
        ax.set_title('Distribution per Dimension')
        ax.grid(True, alpha=0.3)
        
        # 2. Correlation heatmap
        ax = axes[0, 1]
        corr = np.corrcoef(latent_codes.T)
        im = ax.imshow(corr, cmap='coolwarm', vmin=-1, vmax=1, aspect='auto')
        ax.set_xticks(range(len(latent_cols)))
        ax.set_yticks(range(len(latent_cols)))
        ax.set_xticklabels([f'L{i+1}' for i in range(len(latent_cols))], rotation=45)
        ax.set_yticklabels([f'L{i+1}' for i in range(len(latent_cols))])
        ax.set_title('Inter-Dimension Correlation')
        plt.colorbar(im, ax=ax)
        
        # 3. First two latent dimensions by phenoconversion
        if 'phenoconverted' in latent_df.columns:
            ax = axes[1, 0]
            converters = latent_df['phenoconverted'].values.astype(bool)
            ax.scatter(latent_codes[~converters, 0], latent_codes[~converters, 1],
                      c='blue', alpha=0.4, s=20, label='Non-converter')
            ax.scatter(latent_codes[converters, 0], latent_codes[converters, 1],
                      c='red', alpha=0.4, s=20, label='Converter')
            ax.set_xlabel('LATENT_1')
            ax.set_ylabel('LATENT_2')
            ax.set_title('Latent Space (L1 vs L2) by Phenoconversion')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # 4. Variance explained by each dimension
        ax = axes[1, 1]
        variances = np.var(latent_codes, axis=0)
        total_var = variances.sum()
        explained_var = (variances / total_var) * 100
        ax.bar(range(len(explained_var)), explained_var, alpha=0.7, edgecolor='black')
        ax.set_xlabel('Latent Dimension')
        ax.set_ylabel('Variance Explained (%)')
        ax.set_title('Variance Captured per Dimension')
        ax.set_xticks(range(len(latent_cols)))
        ax.set_xticklabels([f'L{i+1}' for i in range(len(latent_cols))])
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        save_path = self.output_dir.parent / "archive" / "development" / "phase8" / "subphase8_4_vae_heterogeneity" / "visualizations" / f"latent_space_structure_dim{self.latent_dim}.png"
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\n✅ Saved: {save_path}")
        plt.close()
    
    def save_latent_codes(self, latent_df: pd.DataFrame) -> None:
        """Save latent codes to CSV."""
        print("\n" + "=" * 80)
        print("SAVING LATENT CODES")
        print("=" * 80)
        
        # Save main latent codes
        save_path = self.output_dir / f"vae_latent_codes_dim{self.latent_dim}.csv"
        latent_df.to_csv(save_path, index=False)
        
        print(f"\n✅ Saved latent codes: {save_path}")
        print(f"   Shape: {latent_df.shape}")
        print(f"   Columns: {list(latent_df.columns[:5])} ... {list(latent_df.columns[-3:])}")
        
        # Save metadata
        metadata = {
            'extraction_date': '2025-10-14',
            'source_model': f'Phase 8.4 Heterogeneity VAE (latent_dim={self.latent_dim})',
            'checkpoint': str(self.checkpoint_path),
            'latent_dim': self.latent_dim,
            'n_observations': len(latent_df),
            'n_patients': latent_df['PATNO'].nunique() if 'PATNO' in latent_df.columns else None,
            'latent_columns': [col for col in latent_df.columns if col.startswith('LATENT_')]
        }
        
        metadata_path = self.output_dir / f"vae_latent_codes_dim{self.latent_dim}_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"✅ Saved metadata: {metadata_path}")


def main():
    """Main extraction pipeline."""
    print("=" * 80)
    print("PHASE 8.4: LATENT CODE EXTRACTION")
    print("=" * 80)
    print("\nExtracting latent codes from best VAE model (12-dim)\n")
    
    # Check for GPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}\n")
    
    # Initialize extractor
    extractor = LatentCodeExtractor(latent_dim=12, device=device)
    
    # Load model
    extractor.load_model()
    
    # Extract latent codes
    latent_df = extractor.extract_latent_codes()
    
    # Analyze structure
    extractor.analyze_latent_structure(latent_df)
    
    # Save results
    extractor.save_latent_codes(latent_df)
    
    print("\n" + "=" * 80)
    print("✅ LATENT CODE EXTRACTION COMPLETE")
    print("=" * 80)
    print("\nNext steps:")
    print("  1. Perform biological correlation analysis")
    print("  2. Compare to Phase 4 discrete subtypes")
    print("  3. Generate comprehensive visualizations")


if __name__ == "__main__":
    main()
