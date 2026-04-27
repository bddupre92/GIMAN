"""
Biological Correlation Analysis of VAE Latent Space.

This script correlates the 12-dimensional latent codes with biological features
to understand what disease patterns each latent axis captures.

Analysis includes:
- Genetic risk factors (GBA, LRRK2, SNCA, APOE)
- Motor phenotype (UPDRS-I, UPDRS-II, PIGD, Tremor)
- Cognitive phenotype (MoCA, Schwab-England)
- Disease progression (time to event, phenoconversion)
- Imaging biomarkers (caudate/putamen volumes, SBRs, asymmetry)
- CSF biomarkers (alpha-synuclein, tau, abeta)
- Non-motor symptoms (RBD, UPSIT, SCOPA-AUT, ESS)

Statistical testing with FDR correction for multiple comparisons.

Author: GIMAN Research Team
Date: October 14, 2025
"""

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import pearsonr, spearmanr
from statsmodels.stats.multitest import multipletests


class BiologicalCorrelationAnalyzer:
    """Analyze correlations between latent codes and biological features."""
    
    def __init__(self):
        self.project_root = Path(__file__).resolve().parents[5]
        self.latent_codes_path = self.project_root / "data" / "05_embeddings" / "vae_latent_codes_dim12.csv"
        self.original_data_path = self.project_root / "data" / "03_prodromal" / "final_training_dataset" / "unified_longitudinal_early_pd.csv"
        self.output_dir = self.project_root / "archive" / "development" / "phase8" / "subphase8_4_vae_heterogeneity" / "results"
        self.viz_dir = self.project_root / "archive" / "development" / "phase8" / "subphase8_4_vae_heterogeneity" / "visualizations"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.viz_dir.mkdir(parents=True, exist_ok=True)
        
        self.df = None
        self.latent_cols = None
        self.correlation_results = {}
    
    def load_data(self) -> None:
        """Load latent codes and merge with original features."""
        print("=" * 80)
        print("LOADING DATA")
        print("=" * 80)
        
        # Load latent codes
        latent_df = pd.read_csv(self.latent_codes_path)
        print(f"\nLoaded latent codes: {latent_df.shape}")
        
        # Load original features
        original_df = pd.read_csv(self.original_data_path)
        print(f"Loaded original data: {original_df.shape}")
        
        # Merge to get all features
        # Note: latent_df already has most features, but let's ensure we have everything
        self.df = latent_df.copy()
        
        # Identify latent columns
        self.latent_cols = [col for col in self.df.columns if col.startswith('LATENT_')]
        print(f"\nLatent dimensions: {len(self.latent_cols)}")
        print(f"Total features: {len(self.df.columns)}")
        print(f"Observations: {len(self.df)}")
    
    def define_feature_groups(self) -> Dict[str, List[str]]:
        """Define biological feature groups for analysis."""
        print("\n" + "=" * 80)
        print("DEFINING BIOLOGICAL FEATURE GROUPS")
        print("=" * 80)
        
        feature_groups = {}
        
        # 1. Genetic risk factors
        genetic_features = []
        for col in ['LRRK2', 'GBA', 'SNCA', 'APOE_E4', 'GENETIC_RISK_SCORE']:
            if col in self.df.columns:
                genetic_features.append(col)
        feature_groups['Genetic Risk'] = genetic_features
        
        # 2. Motor phenotype
        motor_features = []
        for col in ['UPDRS_I', 'UPDRS_II', 'PIGD_SCORE', 'TREMOR_SCORE', 'SCHWAB_ENGLAND']:
            if col in self.df.columns:
                motor_features.append(col)
        feature_groups['Motor Phenotype'] = motor_features
        
        # 3. Structural MRI (volumes)
        mri_volume_features = []
        for col in ['CAUDATE_L_VOL', 'CAUDATE_R_VOL', 'PUTAMEN_L_VOL', 'PUTAMEN_R_VOL',
                    'HIPPOCAMPUS_L_VOL', 'HIPPOCAMPUS_R_VOL']:
            if col in self.df.columns:
                mri_volume_features.append(col)
        feature_groups['MRI Volumes'] = mri_volume_features
        
        # 4. Cortical thickness
        mri_thickness_features = []
        for col in ['ENTORHINAL_L_CTH', 'ENTORHINAL_R_CTH', 'CINGULATE_L_CTH', 
                    'CINGULATE_R_CTH', 'PRECENTRAL_L_CTH', 'PRECENTRAL_R_CTH']:
            if col in self.df.columns:
                mri_thickness_features.append(col)
        feature_groups['Cortical Thickness'] = mri_thickness_features
        
        # 5. DAT-SPECT (striatal binding ratios)
        datspect_features = []
        for col in ['CAUDATE_L_SBR', 'CAUDATE_R_SBR', 'PUTAMEN_L_SBR', 'PUTAMEN_R_SBR',
                    'CAUDATE_ASYMMETRY', 'PUTAMEN_ASYMMETRY']:
            if col in self.df.columns:
                datspect_features.append(col)
        feature_groups['DAT-SPECT'] = datspect_features
        
        # 6. CSF biomarkers
        csf_features = []
        for col in ['ALPHA_SYNUCLEIN', 'TOTAL_TAU', 'ABETA42', 'PTAU181']:
            if col in self.df.columns:
                csf_features.append(col)
        feature_groups['CSF Biomarkers'] = csf_features
        
        # 7. Non-motor symptoms
        nonmotor_features = []
        for col in ['RBD_SCORE', 'UPSIT_SCORE', 'SCOPA_AUT_SCORE', 'ESS_SCORE']:
            if col in self.df.columns:
                nonmotor_features.append(col)
        feature_groups['Non-Motor Symptoms'] = nonmotor_features
        
        # 8. Disease progression
        progression_features = []
        for col in ['time_to_event', 'phenoconverted', 'landmark_month']:
            if col in self.df.columns:
                progression_features.append(col)
        feature_groups['Disease Progression'] = progression_features
        
        # Print summary
        print("\nFeature groups defined:")
        for group_name, features in feature_groups.items():
            print(f"  {group_name}: {len(features)} features")
            if len(features) > 0:
                print(f"    {', '.join(features[:3])}{' ...' if len(features) > 3 else ''}")
        
        return feature_groups
    
    def compute_correlations(self, feature_groups: Dict[str, List[str]]) -> pd.DataFrame:
        """Compute correlations between latent codes and biological features."""
        print("\n" + "=" * 80)
        print("COMPUTING CORRELATIONS")
        print("=" * 80)
        
        results = []
        
        total_tests = 0
        for group_features in feature_groups.values():
            total_tests += len(group_features) * len(self.latent_cols)
        
        print(f"\nTotal correlation tests: {total_tests}")
        print("Computing Pearson correlations...\n")
        
        for group_name, features in feature_groups.items():
            if len(features) == 0:
                continue
            
            print(f"  {group_name}...")
            
            for latent_col in self.latent_cols:
                latent_values = self.df[latent_col].values
                
                for feature in features:
                    if feature not in self.df.columns:
                        continue
                    
                    feature_values = self.df[feature].values
                    
                    # Remove NaNs
                    mask = ~(np.isnan(latent_values) | np.isnan(feature_values))
                    if mask.sum() < 10:  # Need at least 10 observations
                        continue
                    
                    # Compute Pearson correlation
                    r, p_value = pearsonr(latent_values[mask], feature_values[mask])
                    
                    results.append({
                        'latent_axis': latent_col,
                        'feature_group': group_name,
                        'feature': feature,
                        'n_obs': mask.sum(),
                        'r': r,
                        'p_value': p_value,
                        'abs_r': abs(r)
                    })
        
        # Create DataFrame
        results_df = pd.DataFrame(results)
        
        # FDR correction
        print("\nApplying FDR correction for multiple testing...")
        if len(results_df) > 0:
            _, p_adjusted, _, _ = multipletests(
                results_df['p_value'].values,
                method='fdr_bh',
                alpha=0.05
            )
            results_df['p_adjusted'] = p_adjusted
            results_df['significant'] = p_adjusted < 0.05
        
        # Sort by absolute correlation
        results_df = results_df.sort_values('abs_r', ascending=False)
        
        print(f"\n✅ Computed {len(results_df)} correlations")
        print(f"   Significant (FDR < 0.05): {results_df['significant'].sum()}")
        print(f"   Strong (|r| > 0.4): {(results_df['abs_r'] > 0.4).sum()}")
        print(f"   Moderate (|r| > 0.3): {(results_df['abs_r'] > 0.3).sum()}")
        
        return results_df
    
    def summarize_top_correlations(self, results_df: pd.DataFrame) -> None:
        """Print summary of top correlations."""
        print("\n" + "=" * 80)
        print("TOP CORRELATIONS (|r| > 0.3, FDR < 0.05)")
        print("=" * 80)
        
        top_corrs = results_df[
            (results_df['abs_r'] > 0.3) & 
            (results_df['significant'])
        ].head(20)
        
        if len(top_corrs) == 0:
            print("\nNo strong significant correlations found (|r| > 0.3, FDR < 0.05)")
            print("Showing top 20 by correlation strength:")
            top_corrs = results_df.head(20)
        
        print(f"\n{'Latent':<10} {'Feature Group':<25} {'Feature':<25} {'r':<8} {'p_adj':<10} {'n':<6}")
        print("-" * 90)
        
        for _, row in top_corrs.iterrows():
            print(f"{row['latent_axis']:<10} {row['feature_group']:<25} "
                  f"{row['feature']:<25} {row['r']:>7.3f} {row['p_adjusted']:>9.4f} {row['n_obs']:>5}")
    
    def analyze_latent_interpretations(self, results_df: pd.DataFrame) -> Dict:
        """Interpret what each latent axis represents."""
        print("\n" + "=" * 80)
        print("LATENT AXIS BIOLOGICAL INTERPRETATIONS")
        print("=" * 80)
        
        interpretations = {}
        
        for latent_col in self.latent_cols:
            latent_results = results_df[
                (results_df['latent_axis'] == latent_col) &
                (results_df['significant']) &
                (results_df['abs_r'] > 0.25)
            ].sort_values('abs_r', ascending=False).head(5)
            
            if len(latent_results) > 0:
                print(f"\n{latent_col}:")
                top_features = []
                for _, row in latent_results.iterrows():
                    direction = "↑" if row['r'] > 0 else "↓"
                    print(f"  {direction} {row['feature']} (r={row['r']:.3f}, {row['feature_group']})")
                    top_features.append({
                        'feature': row['feature'],
                        'group': row['feature_group'],
                        'r': row['r'],
                        'p_adjusted': row['p_adjusted']
                    })
                
                interpretations[latent_col] = {
                    'top_correlations': top_features,
                    'n_significant': len(latent_results)
                }
            else:
                print(f"\n{latent_col}: No strong significant correlations")
                interpretations[latent_col] = {
                    'top_correlations': [],
                    'n_significant': 0
                }
        
        return interpretations
    
    def visualize_correlation_heatmap(self, results_df: pd.DataFrame) -> None:
        """Generate correlation heatmap."""
        print("\n" + "=" * 80)
        print("GENERATING CORRELATION HEATMAP")
        print("=" * 80)
        
        # Select significant correlations with |r| > 0.2
        sig_results = results_df[
            (results_df['significant']) & 
            (results_df['abs_r'] > 0.2)
        ].copy()
        
        if len(sig_results) == 0:
            print("\n⚠️  No significant correlations with |r| > 0.2 found")
            print("Using top 50 correlations by |r|...")
            sig_results = results_df.head(50)
        
        # Pivot to create correlation matrix
        corr_matrix = sig_results.pivot_table(
            index='feature',
            columns='latent_axis',
            values='r',
            fill_value=0
        )
        
        # Sort features by feature group
        feature_to_group = dict(zip(sig_results['feature'], sig_results['feature_group']))
        corr_matrix['group'] = corr_matrix.index.map(feature_to_group)
        corr_matrix = corr_matrix.sort_values('group')
        
        # Remove group column before plotting
        groups = corr_matrix['group'].values
        corr_matrix = corr_matrix.drop('group', axis=1)
        
        # Plot
        fig, ax = plt.subplots(figsize=(14, max(10, len(corr_matrix) * 0.3)))
        
        sns.heatmap(
            corr_matrix,
            cmap='RdBu_r',
            center=0,
            vmin=-0.6,
            vmax=0.6,
            annot=True,
            fmt='.2f',
            linewidths=0.5,
            cbar_kws={'label': 'Pearson r'},
            ax=ax
        )
        
        ax.set_title('Latent Axis - Biological Feature Correlations\n(Significant, |r| > 0.2)', 
                     fontsize=14, fontweight='bold', pad=20)
        ax.set_xlabel('Latent Dimension', fontsize=12)
        ax.set_ylabel('Biological Feature', fontsize=12)
        
        plt.tight_layout()
        
        save_path = self.viz_dir / "biological_correlation_heatmap.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\n✅ Saved: {save_path}")
        plt.close()
    
    def visualize_top_correlations_by_group(self, results_df: pd.DataFrame, feature_groups: Dict) -> None:
        """Visualize top correlations grouped by feature type."""
        print("\n" + "=" * 80)
        print("GENERATING GROUP-WISE CORRELATION PLOTS")
        print("=" * 80)
        
        fig, axes = plt.subplots(3, 3, figsize=(20, 16))
        axes = axes.flatten()
        
        for idx, (group_name, _) in enumerate(feature_groups.items()):
            if idx >= 9:  # Only 9 subplots
                break
            
            ax = axes[idx]
            
            # Get top correlations for this group
            group_results = results_df[
                (results_df['feature_group'] == group_name) &
                (results_df['significant'])
            ].sort_values('abs_r', ascending=False).head(10)
            
            if len(group_results) > 0:
                # Create bar plot
                colors = ['red' if r < 0 else 'blue' for r in group_results['r'].values]
                y_pos = np.arange(len(group_results))
                
                ax.barh(y_pos, group_results['r'].values, color=colors, alpha=0.7)
                ax.set_yticks(y_pos)
                ax.set_yticklabels([
                    f"{row['feature'][:20]} ({row['latent_axis']})" 
                    for _, row in group_results.iterrows()
                ], fontsize=8)
                ax.set_xlabel('Pearson r', fontsize=9)
                ax.set_title(f'{group_name}\n(Top {len(group_results)} significant)', 
                            fontsize=10, fontweight='bold')
                ax.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
                ax.grid(True, alpha=0.3, axis='x')
            else:
                ax.text(0.5, 0.5, f'{group_name}\nNo significant\ncorrelations',
                       ha='center', va='center', transform=ax.transAxes, fontsize=10)
                ax.set_xticks([])
                ax.set_yticks([])
        
        # Remove empty subplots
        for idx in range(len(feature_groups), 9):
            fig.delaxes(axes[idx])
        
        fig.suptitle('Top Correlations by Feature Group (FDR < 0.05)', 
                     fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        save_path = self.viz_dir / "correlations_by_feature_group.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\n✅ Saved: {save_path}")
        plt.close()
    
    def save_results(self, results_df: pd.DataFrame, interpretations: Dict) -> None:
        """Save correlation results and interpretations."""
        print("\n" + "=" * 80)
        print("SAVING RESULTS")
        print("=" * 80)
        
        # Save full correlation table
        results_path = self.output_dir / "biological_correlations_full.csv"
        results_df.to_csv(results_path, index=False)
        print(f"\n✅ Saved full results: {results_path}")
        
        # Save significant correlations only
        sig_results_path = self.output_dir / "biological_correlations_significant.csv"
        sig_results = results_df[results_df['significant']].copy()
        sig_results.to_csv(sig_results_path, index=False)
        print(f"✅ Saved significant results: {sig_results_path}")
        print(f"   ({len(sig_results)} significant correlations)")
        
        # Save interpretations as text report
        report_path = self.output_dir / "latent_axis_interpretations.txt"
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("LATENT AXIS BIOLOGICAL INTERPRETATIONS\n")
            f.write("=" * 80 + "\n\n")
            f.write("Phase 8.4 Heterogeneity VAE (12-dim latent space)\n")
            f.write("Correlations with biological features (FDR-corrected)\n\n")
            
            for latent_axis, data in interpretations.items():
                f.write(f"\n{latent_axis}:\n")
                f.write(f"  Significant correlations: {data['n_significant']}\n")
                
                if len(data['top_correlations']) > 0:
                    f.write("  Top features:\n")
                    for feat in data['top_correlations']:
                        direction = "↑" if feat['r'] > 0 else "↓"
                        f.write(f"    {direction} {feat['feature']} "
                               f"(r={feat['r']:.3f}, p_adj={feat['p_adjusted']:.4f}) "
                               f"[{feat['group']}]\n")
                else:
                    f.write("  No strong significant correlations found\n")
        
        print(f"✅ Saved interpretations: {report_path}")


def main():
    """Main analysis pipeline."""
    print("=" * 80)
    print("PHASE 8.4: BIOLOGICAL CORRELATION ANALYSIS")
    print("=" * 80)
    print("\nAnalyzing correlations between latent codes and biological features\n")
    
    # Initialize analyzer
    analyzer = BiologicalCorrelationAnalyzer()
    
    # Load data
    analyzer.load_data()
    
    # Define feature groups
    feature_groups = analyzer.define_feature_groups()
    
    # Compute correlations
    results_df = analyzer.compute_correlations(feature_groups)
    
    # Summarize top correlations
    analyzer.summarize_top_correlations(results_df)
    
    # Interpret latent axes
    interpretations = analyzer.analyze_latent_interpretations(results_df)
    
    # Generate visualizations
    analyzer.visualize_correlation_heatmap(results_df)
    analyzer.visualize_top_correlations_by_group(results_df, feature_groups)
    
    # Save results
    analyzer.save_results(results_df, interpretations)
    
    print("\n" + "=" * 80)
    print("✅ BIOLOGICAL CORRELATION ANALYSIS COMPLETE")
    print("=" * 80)
    print("\nKey outputs:")
    print("  - biological_correlations_full.csv")
    print("  - biological_correlations_significant.csv")
    print("  - latent_axis_interpretations.txt")
    print("  - biological_correlation_heatmap.png")
    print("  - correlations_by_feature_group.png")
    print("\nNext steps:")
    print("  1. Review latent axis interpretations")
    print("  2. Compare to Phase 4 discrete subtypes")
    print("  3. Generate comprehensive visualizations")


if __name__ == "__main__":
    main()
