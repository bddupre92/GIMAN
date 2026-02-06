"""
Verify quality of extracted GAT embeddings from Phase 8.2 model.

This script performs comprehensive quality checks on the 128-dim embeddings:
- Check for NaNs, infinities, and extreme outliers
- Visualize embedding distributions
- Compute correlation structure
- Verify separation between phenoconverters and non-converters
- Generate dimensionality reduction visualizations (PCA, t-SNE, UMAP)

Author: GIMAN Research Team
Date: October 14, 2025
"""

import sys
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from scipy import stats
from typing import Dict, Tuple

# Suppress warnings
warnings.filterwarnings('ignore')

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10


class EmbeddingVerifier:
    """Verify quality of GAT embeddings."""
    
    def __init__(self):
        self.project_root = Path(__file__).resolve().parents[5]
        self.embeddings_path = self.project_root / "data" / "05_embeddings" / "giman_gat_embeddings.csv"
        self.output_dir = self.project_root / "archive" / "development" / "phase8" / "subphase8_4_vae_heterogeneity" / "visualizations"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.df = None
        self.embeddings = None
        self.embedding_cols = None
    
    def load_embeddings(self) -> None:
        """Load extracted embeddings."""
        print("=" * 80)
        print("LOADING GAT EMBEDDINGS")
        print("=" * 80)
        print(f"Path: {self.embeddings_path}\n")
        
        self.df = pd.read_csv(self.embeddings_path)
        
        # Extract embedding columns
        self.embedding_cols = [col for col in self.df.columns if col.startswith('EMB_')]
        self.embeddings = self.df[self.embedding_cols].values
        
        print(f"✅ Loaded embeddings:")
        print(f"   Total observations: {len(self.df)}")
        print(f"   Unique patients: {self.df['PATNO'].nunique()}")
        print(f"   Embedding dimension: {len(self.embedding_cols)}")
        print(f"   Phenoconversion rate: {self.df['phenoconverted'].mean():.1%}")
    
    def check_data_quality(self) -> Dict:
        """Check for NaNs, infinities, and outliers."""
        print("\n" + "=" * 80)
        print("DATA QUALITY CHECKS")
        print("=" * 80)
        
        results = {}
        
        # Check for NaNs
        nan_count = np.isnan(self.embeddings).sum()
        nan_pct = (nan_count / self.embeddings.size) * 100
        results['nan_count'] = nan_count
        results['nan_pct'] = nan_pct
        print(f"\n1. NaN Check:")
        print(f"   NaNs found: {nan_count} ({nan_pct:.4f}%)")
        if nan_count > 0:
            print(f"   ⚠️  WARNING: NaNs detected!")
        else:
            print(f"   ✅ No NaNs")
        
        # Check for infinities
        inf_count = np.isinf(self.embeddings).sum()
        inf_pct = (inf_count / self.embeddings.size) * 100
        results['inf_count'] = inf_count
        results['inf_pct'] = inf_pct
        print(f"\n2. Infinity Check:")
        print(f"   Infinities found: {inf_count} ({inf_pct:.4f}%)")
        if inf_count > 0:
            print(f"   ⚠️  WARNING: Infinities detected!")
        else:
            print(f"   ✅ No infinities")
        
        # Check for extreme outliers (> 5 std from mean)
        z_scores = np.abs(stats.zscore(self.embeddings, axis=0, nan_policy='omit'))
        outliers = (z_scores > 5).sum(axis=1)
        extreme_outliers = (outliers > len(self.embedding_cols) * 0.1).sum()  # >10% dims are outliers
        results['extreme_outliers'] = extreme_outliers
        results['extreme_outlier_pct'] = (extreme_outliers / len(self.embeddings)) * 100
        
        print(f"\n3. Extreme Outlier Check (|z| > 5):")
        print(f"   Observations with >10% outlier dimensions: {extreme_outliers} ({results['extreme_outlier_pct']:.2f}%)")
        if extreme_outliers > len(self.embeddings) * 0.05:
            print(f"   ⚠️  WARNING: High outlier rate (>{len(self.embeddings) * 0.05:.0f} observations)")
        else:
            print(f"   ✅ Acceptable outlier rate")
        
        # Summary statistics
        print(f"\n4. Embedding Statistics:")
        print(f"   Mean: {self.embeddings.mean():.4f}")
        print(f"   Std: {self.embeddings.std():.4f}")
        print(f"   Min: {self.embeddings.min():.4f}")
        print(f"   Max: {self.embeddings.max():.4f}")
        print(f"   Range: {self.embeddings.max() - self.embeddings.min():.4f}")
        
        results['mean'] = self.embeddings.mean()
        results['std'] = self.embeddings.std()
        results['min'] = self.embeddings.min()
        results['max'] = self.embeddings.max()
        
        return results
    
    def visualize_distributions(self) -> None:
        """Visualize embedding value distributions."""
        print("\n" + "=" * 80)
        print("GENERATING DISTRIBUTION VISUALIZATIONS")
        print("=" * 80)
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('GAT Embedding Distributions', fontsize=16, fontweight='bold')
        
        # 1. Overall distribution histogram
        ax = axes[0, 0]
        ax.hist(self.embeddings.flatten(), bins=100, alpha=0.7, edgecolor='black')
        ax.set_xlabel('Embedding Value')
        ax.set_ylabel('Frequency')
        ax.set_title('Overall Embedding Value Distribution')
        ax.axvline(self.embeddings.mean(), color='red', linestyle='--', 
                   label=f'Mean: {self.embeddings.mean():.2f}')
        ax.legend()
        
        # 2. Per-dimension mean and std
        ax = axes[0, 1]
        dim_means = self.embeddings.mean(axis=0)
        dim_stds = self.embeddings.std(axis=0)
        x = np.arange(len(dim_means))
        ax.errorbar(x[::4], dim_means[::4], yerr=dim_stds[::4], fmt='o', 
                    alpha=0.6, capsize=3)
        ax.set_xlabel('Embedding Dimension (every 4th shown)')
        ax.set_ylabel('Value')
        ax.set_title('Per-Dimension Statistics (Mean ± Std)')
        ax.grid(True, alpha=0.3)
        
        # 3. Correlation heatmap (sample of dimensions)
        ax = axes[1, 0]
        sample_dims = np.linspace(0, len(self.embedding_cols)-1, 20, dtype=int)
        corr = np.corrcoef(self.embeddings[:, sample_dims].T)
        im = ax.imshow(corr, cmap='coolwarm', vmin=-1, vmax=1, aspect='auto')
        ax.set_xlabel('Embedding Dimension (sampled)')
        ax.set_ylabel('Embedding Dimension (sampled)')
        ax.set_title('Inter-Dimension Correlation (20 dims sampled)')
        plt.colorbar(im, ax=ax)
        
        # 4. Box plot of embedding ranges per dimension (sample)
        ax = axes[1, 1]
        sample_data = self.embeddings[:, sample_dims]
        bp = ax.boxplot(sample_data, showfliers=False, patch_artist=True)
        for patch in bp['boxes']:
            patch.set_facecolor('lightblue')
        ax.set_xlabel('Embedding Dimension (sampled)')
        ax.set_ylabel('Value')
        ax.set_title('Value Range Distribution (20 dims sampled)')
        ax.set_xticklabels([f'{i}' for i in sample_dims], rotation=45)
        
        plt.tight_layout()
        
        # Save figure
        save_path = self.output_dir / "embedding_distributions.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\n✅ Saved: {save_path}")
        plt.close()
    
    def check_phenoconversion_separation(self) -> None:
        """Check if embeddings separate phenoconverters vs non-converters."""
        print("\n" + "=" * 80)
        print("PHENOCONVERSION SEPARATION ANALYSIS")
        print("=" * 80)
        
        # Get labels
        converters = self.df['phenoconverted'].values.astype(bool)
        
        # Run PCA for visualization
        print("\nRunning PCA...")
        pca = PCA(n_components=3)
        pca_coords = pca.fit_transform(self.embeddings)
        
        print(f"Explained variance (first 3 PCs): {pca.explained_variance_ratio_.sum():.1%}")
        print(f"  PC1: {pca.explained_variance_ratio_[0]:.1%}")
        print(f"  PC2: {pca.explained_variance_ratio_[1]:.1%}")
        print(f"  PC3: {pca.explained_variance_ratio_[2]:.1%}")
        
        # Visualize
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        fig.suptitle('Phenoconversion Separation in Embedding Space', 
                     fontsize=16, fontweight='bold')
        
        # PC1 vs PC2
        ax = axes[0]
        scatter = ax.scatter(pca_coords[~converters, 0], pca_coords[~converters, 1], 
                            c='blue', alpha=0.4, s=20, label='Non-converter')
        ax.scatter(pca_coords[converters, 0], pca_coords[converters, 1], 
                  c='red', alpha=0.4, s=20, label='Converter')
        ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})')
        ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})')
        ax.set_title('PCA: PC1 vs PC2')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # PC1 vs PC3
        ax = axes[1]
        ax.scatter(pca_coords[~converters, 0], pca_coords[~converters, 2], 
                  c='blue', alpha=0.4, s=20, label='Non-converter')
        ax.scatter(pca_coords[converters, 0], pca_coords[converters, 2], 
                  c='red', alpha=0.4, s=20, label='Converter')
        ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})')
        ax.set_ylabel(f'PC3 ({pca.explained_variance_ratio_[2]:.1%})')
        ax.set_title('PCA: PC1 vs PC3')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        save_path = self.output_dir / "phenoconversion_separation_pca.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\n✅ Saved: {save_path}")
        plt.close()
    
    def run_tsne_visualization(self) -> None:
        """Run t-SNE visualization."""
        print("\n" + "=" * 80)
        print("GENERATING t-SNE VISUALIZATION")
        print("=" * 80)
        
        # Standardize for t-SNE
        print("Standardizing embeddings...")
        scaler = StandardScaler()
        embeddings_scaled = scaler.fit_transform(self.embeddings)
        
        # Run t-SNE (this may take a few minutes)
        print("Running t-SNE (perplexity=30)...")
        tsne = TSNE(n_components=2, perplexity=30, random_state=42, n_iter=1000, verbose=1)
        tsne_coords = tsne.fit_transform(embeddings_scaled)
        
        # Get labels
        converters = self.df['phenoconverted'].values.astype(bool)
        cohorts = self.df['cohort'].values if 'cohort' in self.df.columns else None
        
        # Create visualization
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        fig.suptitle('t-SNE Projection of GAT Embeddings', fontsize=16, fontweight='bold')
        
        # By phenoconversion
        ax = axes[0]
        ax.scatter(tsne_coords[~converters, 0], tsne_coords[~converters, 1],
                  c='blue', alpha=0.4, s=20, label='Non-converter')
        ax.scatter(tsne_coords[converters, 0], tsne_coords[converters, 1],
                  c='red', alpha=0.4, s=20, label='Converter')
        ax.set_xlabel('t-SNE 1')
        ax.set_ylabel('t-SNE 2')
        ax.set_title('Colored by Phenoconversion')
        ax.legend()
        
        # By cohort
        ax = axes[1]
        if cohorts is not None:
            unique_cohorts = np.unique(cohorts)
            colors = plt.cm.Set3(np.linspace(0, 1, len(unique_cohorts)))
            for cohort, color in zip(unique_cohorts, colors):
                mask = cohorts == cohort
                ax.scatter(tsne_coords[mask, 0], tsne_coords[mask, 1],
                          c=[color], alpha=0.5, s=20, label=cohort)
            ax.set_xlabel('t-SNE 1')
            ax.set_ylabel('t-SNE 2')
            ax.set_title('Colored by Cohort')
            ax.legend()
        else:
            ax.text(0.5, 0.5, 'Cohort data not available', 
                   ha='center', va='center', transform=ax.transAxes)
        
        plt.tight_layout()
        
        save_path = self.output_dir / "tsne_projection.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\n✅ Saved: {save_path}")
        plt.close()
    
    def generate_summary_report(self, quality_results: Dict) -> None:
        """Generate summary report."""
        print("\n" + "=" * 80)
        print("EMBEDDING QUALITY SUMMARY")
        print("=" * 80)
        
        report = []
        report.append("=" * 80)
        report.append("GAT EMBEDDING QUALITY VERIFICATION REPORT")
        report.append("=" * 80)
        report.append(f"\nDate: October 14, 2025")
        report.append(f"Source: Phase 8.2 GIMAN-Progression Model (C-index 0.9980)")
        report.append(f"Embedding Dimension: {len(self.embedding_cols)}")
        report.append(f"Total Observations: {len(self.df)}")
        report.append(f"Unique Patients: {self.df['PATNO'].nunique()}")
        
        report.append("\n" + "=" * 80)
        report.append("DATA QUALITY METRICS")
        report.append("=" * 80)
        report.append(f"\nNaN Count: {quality_results['nan_count']} ({quality_results['nan_pct']:.4f}%)")
        report.append(f"Infinity Count: {quality_results['inf_count']} ({quality_results['inf_pct']:.4f}%)")
        report.append(f"Extreme Outliers: {quality_results['extreme_outliers']} ({quality_results['extreme_outlier_pct']:.2f}%)")
        
        report.append("\n" + "=" * 80)
        report.append("EMBEDDING STATISTICS")
        report.append("=" * 80)
        report.append(f"\nMean: {quality_results['mean']:.4f}")
        report.append(f"Std: {quality_results['std']:.4f}")
        report.append(f"Min: {quality_results['min']:.4f}")
        report.append(f"Max: {quality_results['max']:.4f}")
        report.append(f"Range: {quality_results['max'] - quality_results['min']:.4f}")
        
        report.append("\n" + "=" * 80)
        report.append("QUALITY ASSESSMENT")
        report.append("=" * 80)
        
        issues = []
        if quality_results['nan_count'] > 0:
            issues.append("- NaN values detected")
        if quality_results['inf_count'] > 0:
            issues.append("- Infinity values detected")
        if quality_results['extreme_outlier_pct'] > 5.0:
            issues.append("- High outlier rate (>5%)")
        
        if not issues:
            report.append("\n✅ EMBEDDINGS PASS ALL QUALITY CHECKS")
            report.append("\nThe embeddings are clean and ready for VAE training.")
        else:
            report.append("\n⚠️  QUALITY ISSUES DETECTED:")
            for issue in issues:
                report.append(issue)
            report.append("\nRecommend investigating and potentially filtering outliers before VAE training.")
        
        report.append("\n" + "=" * 80)
        report.append("GENERATED VISUALIZATIONS")
        report.append("=" * 80)
        report.append("\n1. embedding_distributions.png")
        report.append("   - Overall value distribution")
        report.append("   - Per-dimension statistics")
        report.append("   - Inter-dimension correlation")
        report.append("   - Value range distributions")
        report.append("\n2. phenoconversion_separation_pca.png")
        report.append("   - PCA projection (PC1 vs PC2, PC1 vs PC3)")
        report.append("   - Colored by phenoconversion status")
        report.append("\n3. tsne_projection.png")
        report.append("   - t-SNE 2D projection")
        report.append("   - Colored by phenoconversion and cohort")
        
        report.append("\n" + "=" * 80)
        report.append("NEXT STEPS")
        report.append("=" * 80)
        report.append("\n1. Review visualizations for any unexpected patterns")
        report.append("2. If quality checks pass, proceed with VAE architecture creation")
        report.append("3. Use these embeddings as input to VAE training")
        report.append("4. Target VAE reconstruction loss < 0.1")
        
        # Print report
        report_text = "\n".join(report)
        print(report_text)
        
        # Save report
        report_path = self.output_dir.parent / "EMBEDDING_QUALITY_REPORT.md"
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report_text)
        
        print(f"\n✅ Saved report: {report_path}")


def main():
    """Main verification pipeline."""
    print("=" * 80)
    print("PHASE 8.4: GAT EMBEDDING QUALITY VERIFICATION")
    print("=" * 80)
    print("\nVerifying 128-dim embeddings extracted from Phase 8.2 model\n")
    
    # Initialize verifier
    verifier = EmbeddingVerifier()
    
    # Load embeddings
    verifier.load_embeddings()
    
    # Run quality checks
    quality_results = verifier.check_data_quality()
    
    # Generate visualizations
    verifier.visualize_distributions()
    verifier.check_phenoconversion_separation()
    verifier.run_tsne_visualization()
    
    # Generate summary report
    verifier.generate_summary_report(quality_results)
    
    print("\n" + "=" * 80)
    print("✅ VERIFICATION COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
