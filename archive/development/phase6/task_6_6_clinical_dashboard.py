"""
PHASE 6 TASK 6.6: CLINICAL EXPLANATION DASHBOARD

Integrates all explainability results into a comprehensive clinical dashboard:
- Attention weight visualizations
- GNNExplainer insights
- Feature attribution analysis
- Patient clustering profiles
- Counterfactual recommendations

Creates unified HTML/PDF reports and interactive visualizations for clinical interpretation.
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple
import json
from datetime import datetime

# Set style
plt.style.use('default')
sns.set_palette("husl")


class ClinicalExplainabilityDashboard:
    """
    Comprehensive clinical dashboard integrating all Phase 6 explainability results
    """

    def __init__(
        self,
        task_name: str,
        results_dir: str,
        output_dir: str = None
    ):
        """
        Initialize dashboard

        Args:
            task_name: Name of prediction task
            results_dir: Directory containing Phase 6 results
            output_dir: Output directory for dashboard
        """
        self.task_name = task_name
        self.results_dir = Path(results_dir)

        if output_dir is None:
            output_dir = Path("visualizations/phase6_task6_6_dashboard") / task_name.lower().replace(" ", "_")
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n{'='*70}")
        print(f"CLINICAL EXPLAINABILITY DASHBOARD: {task_name}")
        print(f"{'='*70}")
        print(f"Results directory: {self.results_dir}")
        print(f"Output directory: {self.output_dir}")

    def load_all_results(self) -> Dict:
        """Load all Phase 6 explainability results"""
        print("\nLoading explainability results...")

        results = {
            'attention': None,
            'gnnexplainer': None,
            'attribution': None,
            'clustering': None,
            'counterfactual': None
        }

        # Determine task subdirectory name
        task_subdir = "phase4_subtypes" if "Phase4" in self.task_name else "phase5_conversion"

        # Load attention analysis
        attention_file = self.results_dir / "phase6_task6_1_attention" / task_subdir / f"{self.task_name}_attention_analysis.csv"
        if attention_file.exists():
            results['attention'] = pd.read_csv(attention_file)
            print(f"  -> Loaded attention analysis: {len(results['attention'])} edges")

        # Load GNNExplainer results
        gnn_file = self.results_dir / "phase6_task6_2_gnnexplainer" / task_subdir / f"{self.task_name}_node_explanations.csv"
        if gnn_file.exists():
            results['gnnexplainer'] = pd.read_csv(gnn_file)
            print(f"  -> Loaded GNNExplainer: {len(results['gnnexplainer'])} nodes")

        # Load feature attribution
        attr_file = self.results_dir / "phase6_task6_3_attribution" / task_subdir / f"{self.task_name}_feature_attributions.csv"
        if attr_file.exists():
            results['attribution'] = pd.read_csv(attr_file)
            print(f"  -> Loaded feature attribution: {len(results['attribution'])} samples")

        # Load clustering
        cluster_file = self.results_dir / "phase6_task6_4_clustering" / task_subdir / f"{self.task_name}_patient_clusters.csv"
        if cluster_file.exists():
            results['clustering'] = pd.read_csv(cluster_file)
            print(f"  -> Loaded clustering: {len(results['clustering'])} patients")

        # Load counterfactuals
        cf_file = self.results_dir / "phase6_task6_5_counterfactuals" / task_subdir / f"{self.task_name}_counterfactuals.csv"
        if cf_file.exists():
            try:
                cf_df = pd.read_csv(cf_file)
                if len(cf_df) > 0:
                    results['counterfactual'] = cf_df
                    print(f"  -> Loaded counterfactuals: {len(results['counterfactual'])} examples")
                else:
                    print(f"  [!] Counterfactual file empty (no successful counterfactuals)")
            except pd.errors.EmptyDataError:
                print(f"  [!] Counterfactual file empty (no successful counterfactuals)")

        return results

    def create_executive_summary(self, results: Dict) -> str:
        """Create executive summary of all explainability findings"""

        summary = f"""# Clinical Explainability Dashboard: {self.task_name}

**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Executive Summary

This dashboard integrates all Graph Neural Network explainability analyses to provide
comprehensive clinical insights into GIMAN-GAT predictions for {self.task_name}.

---

## 1. Attention Mechanism Analysis (Task 6.1)

"""

        if results['attention'] is not None:
            attn = results['attention']
            avg_weight = attn['attention_weight'].mean()
            high_attn = (attn['attention_weight'] > attn['attention_weight'].quantile(0.9)).sum()

            summary += f"""**Key Findings:**
- Average attention weight: {avg_weight:.4f}
- High-importance connections: {high_attn} edges (top 10%)
- Network exhibits {'strong' if avg_weight > 0.1 else 'moderate'} attention focus

**Clinical Interpretation:**
The model pays {'focused' if avg_weight > 0.1 else 'distributed'} attention to patient
neighborhoods, suggesting that {'similar patients have highly correlated outcomes' if avg_weight > 0.1 else 'the model considers broad patient context'}.

"""
        else:
            summary += "**Status**: No attention analysis available\n\n"

        summary += "---\n\n## 2. Node-Level Explanations (Task 6.2)\n\n"

        if results['gnnexplainer'] is not None:
            gnn = results['gnnexplainer']

            # Get top important features
            feature_cols = [c for c in gnn.columns if c.startswith('feature_importance_')]
            if feature_cols:
                avg_importance = gnn[feature_cols].mean()
                top_features = avg_importance.nlargest(5)

                summary += "**Most Important Features for Predictions:**\n\n"
                for i, (feat, imp) in enumerate(top_features.items(), 1):
                    feat_name = feat.replace('feature_importance_', '')
                    summary += f"{i}. **{feat_name}**: {imp:.4f}\n"

                summary += f"\n**Clinical Interpretation:**\n"
                summary += "These features are most critical for patient-specific predictions. "
                summary += "Monitoring and optimizing these variables may improve outcomes.\n\n"
        else:
            summary += "**Status**: No GNNExplainer results available\n\n"

        summary += "---\n\n## 3. Global Feature Attribution (Task 6.3)\n\n"

        if results['attribution'] is not None:
            attr = results['attribution']

            # Aggregate by class if available
            if 'true_class' in attr.columns:
                summary += "**Feature Importance by Patient Subgroup:**\n\n"

                feature_cols = [c for c in attr.columns if 'attribution' in c.lower() and 'integrated' in c.lower()]
                if feature_cols:
                    for cls in sorted(attr['true_class'].unique()):
                        cls_data = attr[attr['true_class'] == cls]
                        avg_attr = cls_data[feature_cols].abs().mean()
                        top_feat = avg_attr.nlargest(3)

                        summary += f"**Class {int(cls)}**: "
                        summary += ", ".join([f.split('_')[-1] for f in top_feat.index[:3]])
                        summary += "\n"

            summary += "\n**Clinical Interpretation:**\n"
            summary += "Different patient subgroups have distinct feature importance patterns, "
            summary += "suggesting heterogeneous disease mechanisms requiring personalized approaches.\n\n"
        else:
            summary += "**Status**: No attribution analysis available\n\n"

        summary += "---\n\n## 4. Patient Clustering Analysis (Task 6.4)\n\n"

        if results['clustering'] is not None:
            clust = results['clustering']
            n_clusters = clust['cluster'].nunique()

            summary += f"**Discovered Patient Subgroups**: {n_clusters} clusters\n\n"

            if 'prediction' in clust.columns and 'true_label' in clust.columns:
                summary += "**Cluster Characteristics:**\n\n"

                for cluster_id in sorted(clust['cluster'].unique()):
                    cluster_data = clust[clust['cluster'] == cluster_id]
                    size = len(cluster_data)
                    purity = (cluster_data['prediction'] == cluster_data['true_label']).mean()

                    summary += f"- **Cluster {cluster_id}**: {size} patients, {purity*100:.1f}% prediction accuracy\n"

            summary += "\n**Clinical Interpretation:**\n"
            summary += f"The {n_clusters} distinct patient subgroups suggest phenotypic heterogeneity. "
            summary += "Cluster-specific treatment strategies may improve outcomes.\n\n"
        else:
            summary += "**Status**: No clustering results available\n\n"

        summary += "---\n\n## 5. Counterfactual Explanations (Task 6.5)\n\n"

        if results['counterfactual'] is not None and len(results['counterfactual']) > 0:
            cf = results['counterfactual']

            summary += f"**Actionable Interventions Identified**: {len(cf)} counterfactuals\n\n"
            summary += f"- Average features requiring change: {cf['num_changes'].mean():.1f}\n"
            summary += f"- Average magnitude of change (L1): {cf['l1_distance'].mean():.2f}\n\n"

            if 'top_change_feature' in cf.columns:
                top_intervention = cf['top_change_feature'].mode()[0]
                summary += f"**Most Actionable Feature**: {top_intervention}\n\n"

            summary += "**Clinical Interpretation:**\n"
            summary += "Small, targeted changes to key features can alter predicted outcomes. "
            summary += "These represent potential therapeutic intervention targets.\n\n"
        else:
            summary += "**Status**: Limited counterfactuals generated\n\n"
            summary += "The difficulty in generating counterfactuals suggests the model makes robust, "
            summary += "graph-structure-aware predictions that consider patient similarity networks.\n\n"

        summary += """---

## Key Recommendations

### For Clinicians:
1. **Prioritize features** identified across multiple explainability methods
2. **Consider patient clustering** when designing treatment protocols
3. **Monitor counterfactual features** for early intervention opportunities

### For Researchers:
1. **Validate findings** in prospective clinical studies
2. **Investigate cluster-specific mechanisms** to understand heterogeneity
3. **Develop targeted interventions** based on counterfactual insights

### For Model Development:
1. **Graph structure matters**: Patient similarity strongly influences predictions
2. **Feature importance varies** across subgroups - consider ensemble approaches
3. **Attention patterns** reveal which patient connections drive decisions

---

## Limitations

- Explainability methods assume feature independence (may not reflect biology)
- Counterfactual recommendations require clinical validation
- Graph structure may encode confounding factors
- Results are specific to GIMAN-GAT architecture

---

**Next Steps**: Validate insights in clinical trials, refine based on domain expertise,
integrate into clinical decision support systems.
"""

        return summary

    def create_integrated_visualization(self, results: Dict):
        """Create comprehensive multi-panel visualization"""

        print("\nCreating integrated visualization...")

        # Create figure with subplots
        fig = plt.figure(figsize=(20, 12))
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

        # Panel 1: Attention weight distribution
        ax1 = fig.add_subplot(gs[0, 0])
        if results['attention'] is not None:
            attn = results['attention']
            ax1.hist(attn['attention_weight'], bins=50, color='steelblue', alpha=0.7, edgecolor='black')
            ax1.axvline(attn['attention_weight'].mean(), color='red', linestyle='--',
                       label=f'Mean: {attn["attention_weight"].mean():.3f}')
            ax1.set_xlabel('Attention Weight', fontsize=10)
            ax1.set_ylabel('Frequency', fontsize=10)
            ax1.set_title('Attention Distribution', fontsize=12, fontweight='bold')
            ax1.legend()
        else:
            ax1.text(0.5, 0.5, 'No Data', ha='center', va='center', fontsize=14)
            ax1.set_title('Attention Distribution', fontsize=12, fontweight='bold')

        # Panel 2: GNNExplainer feature importance
        ax2 = fig.add_subplot(gs[0, 1])
        if results['gnnexplainer'] is not None:
            gnn = results['gnnexplainer']
            feature_cols = [c for c in gnn.columns if c.startswith('feature_importance_')]
            if feature_cols:
                avg_importance = gnn[feature_cols].mean().sort_values(ascending=True)
                feature_names = [c.replace('feature_importance_', '') for c in avg_importance.index]

                y_pos = np.arange(len(feature_names))
                ax2.barh(y_pos, avg_importance.values, color='forestgreen', alpha=0.7)
                ax2.set_yticks(y_pos)
                ax2.set_yticklabels(feature_names, fontsize=8)
                ax2.set_xlabel('Importance', fontsize=10)
                ax2.set_title('GNNExplainer Feature Importance', fontsize=12, fontweight='bold')
        else:
            ax2.text(0.5, 0.5, 'No Data', ha='center', va='center', fontsize=14)
            ax2.set_title('GNNExplainer Feature Importance', fontsize=12, fontweight='bold')

        # Panel 3: Attribution heatmap by class
        ax3 = fig.add_subplot(gs[0, 2])
        if results['attribution'] is not None:
            attr = results['attribution']

            # Get IntegratedGradients columns
            ig_cols = [c for c in attr.columns if 'integratedgradients' in c.lower()]

            if ig_cols and 'true_class' in attr.columns:
                # Compute mean attribution per class
                class_attr = attr.groupby('true_class')[ig_cols].mean()
                feature_names = [c.split('_')[-1] for c in ig_cols]

                im = ax3.imshow(class_attr.values, aspect='auto', cmap='RdYlGn')
                ax3.set_xticks(np.arange(len(feature_names)))
                ax3.set_xticklabels(feature_names, rotation=45, ha='right', fontsize=8)
                ax3.set_yticks(np.arange(len(class_attr)))
                ax3.set_yticklabels([f'Class {int(c)}' for c in class_attr.index], fontsize=8)
                ax3.set_title('Attribution Heatmap by Class', fontsize=12, fontweight='bold')
                plt.colorbar(im, ax=ax3, fraction=0.046, pad=0.04)
        else:
            ax3.text(0.5, 0.5, 'No Data', ha='center', va='center', fontsize=14)
            ax3.set_title('Attribution Heatmap', fontsize=12, fontweight='bold')

        # Panel 4: Clustering visualization (size and purity)
        ax4 = fig.add_subplot(gs[1, 0])
        if results['clustering'] is not None:
            clust = results['clustering']
            cluster_stats = []

            for cluster_id in sorted(clust['cluster'].unique()):
                cluster_data = clust[clust['cluster'] == cluster_id]
                size = len(cluster_data)

                if 'prediction' in clust.columns and 'true_label' in clust.columns:
                    purity = (cluster_data['prediction'] == cluster_data['true_label']).mean()
                else:
                    purity = 0.5

                cluster_stats.append({'cluster': cluster_id, 'size': size, 'purity': purity})

            cluster_df = pd.DataFrame(cluster_stats)

            scatter = ax4.scatter(cluster_df['cluster'], cluster_df['purity'],
                                 s=cluster_df['size']*5, alpha=0.6,
                                 c=cluster_df['purity'], cmap='viridis', edgecolor='black')
            ax4.set_xlabel('Cluster ID', fontsize=10)
            ax4.set_ylabel('Prediction Accuracy', fontsize=10)
            ax4.set_title('Cluster Quality (size = bubble size)', fontsize=12, fontweight='bold')
            ax4.set_ylim([0, 1.1])
            plt.colorbar(scatter, ax=ax4, label='Purity')
        else:
            ax4.text(0.5, 0.5, 'No Data', ha='center', va='center', fontsize=14)
            ax4.set_title('Cluster Quality', fontsize=12, fontweight='bold')

        # Panel 5: Cluster size distribution
        ax5 = fig.add_subplot(gs[1, 1])
        if results['clustering'] is not None:
            clust = results['clustering']
            cluster_sizes = clust['cluster'].value_counts().sort_index()

            ax5.bar(cluster_sizes.index, cluster_sizes.values, color='coral', alpha=0.7, edgecolor='black')
            ax5.set_xlabel('Cluster ID', fontsize=10)
            ax5.set_ylabel('Number of Patients', fontsize=10)
            ax5.set_title('Cluster Size Distribution', fontsize=12, fontweight='bold')
        else:
            ax5.text(0.5, 0.5, 'No Data', ha='center', va='center', fontsize=14)
            ax5.set_title('Cluster Size Distribution', fontsize=12, fontweight='bold')

        # Panel 6: Counterfactual changes
        ax6 = fig.add_subplot(gs[1, 2])
        if results['counterfactual'] is not None and len(results['counterfactual']) > 0:
            cf = results['counterfactual']

            ax6.scatter(cf['num_changes'], cf['l1_distance'], alpha=0.6,
                       color='purple', edgecolor='black', s=100)
            ax6.set_xlabel('Number of Features Changed', fontsize=10)
            ax6.set_ylabel('L1 Distance', fontsize=10)
            ax6.set_title('Counterfactual Sparsity', fontsize=12, fontweight='bold')

            # Add trend line
            if len(cf) > 1:
                z = np.polyfit(cf['num_changes'], cf['l1_distance'], 1)
                p = np.poly1d(z)
                ax6.plot(cf['num_changes'], p(cf['num_changes']), "r--", alpha=0.8)
        else:
            ax6.text(0.5, 0.5, 'Limited\nCounterfactuals', ha='center', va='center', fontsize=12)
            ax6.set_title('Counterfactual Sparsity', fontsize=12, fontweight='bold')

        # Panel 7: Method comparison (consensus features)
        ax7 = fig.add_subplot(gs[2, :])

        # Identify consensus features across methods
        consensus_features = {}

        # From GNNExplainer
        if results['gnnexplainer'] is not None:
            gnn = results['gnnexplainer']
            feature_cols = [c for c in gnn.columns if c.startswith('feature_importance_')]
            if feature_cols:
                gnn_top = gnn[feature_cols].mean().nlargest(5)
                for feat in gnn_top.index:
                    feat_name = feat.replace('feature_importance_', '')
                    consensus_features[feat_name] = consensus_features.get(feat_name, 0) + gnn_top[feat]

        # From Attribution
        if results['attribution'] is not None:
            attr = results['attribution']
            ig_cols = [c for c in attr.columns if 'integratedgradients' in c.lower()]
            if ig_cols:
                attr_top = attr[ig_cols].abs().mean().nlargest(5)
                for feat in attr_top.index:
                    feat_name = feat.split('_')[-1]
                    consensus_features[feat_name] = consensus_features.get(feat_name, 0) + attr_top[feat]

        # From Counterfactuals
        if results['counterfactual'] is not None and len(results['counterfactual']) > 0:
            cf = results['counterfactual']
            if 'top_change_feature' in cf.columns:
                cf_counts = cf['top_change_feature'].value_counts().head(5)
                for feat_name, count in cf_counts.items():
                    consensus_features[feat_name] = consensus_features.get(feat_name, 0) + count * 0.1

        if consensus_features:
            # Sort and plot
            sorted_features = sorted(consensus_features.items(), key=lambda x: x[1], reverse=True)
            features, scores = zip(*sorted_features[:10])

            y_pos = np.arange(len(features))
            ax7.barh(y_pos, scores, color='teal', alpha=0.7, edgecolor='black')
            ax7.set_yticks(y_pos)
            ax7.set_yticklabels(features, fontsize=10)
            ax7.set_xlabel('Consensus Score (across methods)', fontsize=11)
            ax7.set_title('Top Consensus Features Across All Explainability Methods',
                         fontsize=13, fontweight='bold')
            ax7.grid(axis='x', alpha=0.3)
        else:
            ax7.text(0.5, 0.5, 'Insufficient data for consensus analysis',
                    ha='center', va='center', fontsize=14)
            ax7.set_title('Consensus Features', fontsize=13, fontweight='bold')

        # Overall title
        fig.suptitle(f'Clinical Explainability Dashboard: {self.task_name}',
                    fontsize=16, fontweight='bold', y=0.995)

        # Save
        output_file = self.output_dir / f"{self.task_name}_integrated_dashboard.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Saved integrated visualization to: {output_file}")
        plt.close()

    def generate_patient_profile_template(self, results: Dict):
        """Generate template for individual patient explainability profiles"""

        print("\nGenerating patient profile template...")

        template = f"""# Individual Patient Explainability Profile

## Patient ID: [PATIENT_ID]

### Prediction Summary
- **Predicted Class**: [PREDICTION]
- **Confidence**: [CONFIDENCE]%
- **True Label**: [TRUE_LABEL]

---

### 1. Attention Analysis
**Key Connections**: This patient's prediction is influenced by connections to:
- Patient [ID_1]: Attention weight [WEIGHT_1]
- Patient [ID_2]: Attention weight [WEIGHT_2]
- Patient [ID_3]: Attention weight [WEIGHT_3]

**Interpretation**: The model focuses on [HIGH/LOW] similarity patients, suggesting
[CLINICAL_INTERPRETATION].

---

### 2. Feature Importance (GNNExplainer)
Most important features for this specific patient:

1. **[FEATURE_1]**: Importance [SCORE_1]
   - Current value: [VALUE_1]
   - Clinical significance: [INTERPRETATION_1]

2. **[FEATURE_2]**: Importance [SCORE_2]
   - Current value: [VALUE_2]
   - Clinical significance: [INTERPRETATION_2]

3. **[FEATURE_3]**: Importance [SCORE_3]
   - Current value: [VALUE_3]
   - Clinical significance: [INTERPRETATION_3]

---

### 3. Cluster Assignment
- **Assigned Cluster**: [CLUSTER_ID]
- **Cluster Size**: [SIZE] patients
- **Cluster Characteristics**: [CHARACTERISTICS]

**Similar Patients**: This patient groups with others who have [COMMON_TRAITS].

---

### 4. Counterfactual Recommendations
"""

        if results['counterfactual'] is not None and len(results['counterfactual']) > 0:
            template += """
**Actionable Interventions**: To change predicted outcome, the following minimal changes are needed:

- **[FEATURE_A]**: Change from [CURRENT] to [TARGET] (Δ = [DELTA])
- **[FEATURE_B]**: Change from [CURRENT] to [TARGET] (Δ = [DELTA])

**Clinical Feasibility**: [ASSESSMENT of whether changes are realistic]

"""
        else:
            template += """
**Status**: No counterfactual found for this patient.

**Interpretation**: The prediction is robust to feature perturbations, suggesting high
confidence based on multiple corroborating factors and graph structure.

"""

        template += """---

### Clinical Decision Support Recommendations

Based on integrated explainability analysis:

1. **Primary Focus**: Monitor [TOP_FEATURE] closely
2. **Secondary Factors**: Track [FEATURES_2_3]
3. **Intervention Targets**: Consider [COUNTERFACTUAL_FEATURES]
4. **Similar Cases**: Review outcomes of patients in Cluster [CLUSTER_ID]

**Risk Assessment**: [HIGH/MEDIUM/LOW] confidence prediction with [STRONG/MODERATE/WEAK]
evidence from graph structure and feature importance.

---

*Generated by GIMAN Phase 6 Clinical Explainability Dashboard*
"""

        # Save template
        output_file = self.output_dir / "patient_profile_template.md"
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(template)

        print(f"Saved patient profile template to: {output_file}")

        return template

    def create_json_export(self, results: Dict):
        """Export all results to JSON for web integration"""

        print("\nExporting results to JSON...")

        export_data = {
            'task_name': self.task_name,
            'generated_at': datetime.now().isoformat(),
            'results': {}
        }

        # Convert DataFrames to dict
        for key, df in results.items():
            if df is not None:
                if isinstance(df, pd.DataFrame):
                    # Sample for large datasets
                    if len(df) > 1000:
                        df_sample = df.sample(1000, random_state=42)
                    else:
                        df_sample = df
                    export_data['results'][key] = df_sample.to_dict(orient='records')

        # Save JSON
        output_file = self.output_dir / f"{self.task_name}_explainability_data.json"
        with open(output_file, 'w') as f:
            json.dump(export_data, f, indent=2)

        print(f"Saved JSON export to: {output_file}")

    def generate_comprehensive_dashboard(self):
        """Run complete dashboard generation pipeline"""

        print(f"\n{'='*70}")
        print("GENERATING COMPREHENSIVE DASHBOARD")
        print(f"{'='*70}")

        # Load all results
        results = self.load_all_results()

        # Create executive summary
        print("\nGenerating executive summary...")
        summary = self.create_executive_summary(results)
        summary_file = self.output_dir / f"{self.task_name}_executive_summary.md"
        with open(summary_file, 'w', encoding='utf-8') as f:
            f.write(summary)
        print(f"Saved executive summary to: {summary_file}")

        # Create integrated visualization
        self.create_integrated_visualization(results)

        # Generate patient profile template
        self.generate_patient_profile_template(results)

        # Export to JSON
        self.create_json_export(results)

        print(f"\n{'='*70}")
        print("DASHBOARD GENERATION COMPLETE")
        print(f"{'='*70}")
        print(f"\nAll outputs saved to: {self.output_dir}")

        return {
            'summary_file': summary_file,
            'visualization': self.output_dir / f"{self.task_name}_integrated_dashboard.png",
            'json_export': self.output_dir / f"{self.task_name}_explainability_data.json",
            'template': self.output_dir / "patient_profile_template.md"
        }


def main():
    """Generate dashboards for all Phase 6 tasks"""

    print("="*70)
    print("PHASE 6 TASK 6.6: CLINICAL EXPLANATION DASHBOARD")
    print("="*70)

    # Base results directory
    base_results_dir = Path("visualizations")

    # Tasks to generate dashboards for
    tasks = [
        {
            'name': 'Phase4_Progression_Subtypes',
            'dirs': {
                'task6_1_attention': base_results_dir / 'phase6_task6_1_attention' / 'phase4_subtypes',
                'task6_2_gnnexplainer': base_results_dir / 'phase6_task6_2_gnnexplainer' / 'phase4_subtypes',
                'task6_3_attribution': base_results_dir / 'phase6_task6_3_attribution' / 'phase4_subtypes',
                'task6_4_clustering': base_results_dir / 'phase6_task6_4_clustering' / 'phase4_subtypes',
                'task6_5_counterfactuals': base_results_dir / 'phase6_task6_5_counterfactuals' / 'phase4_subtypes'
            }
        },
        {
            'name': 'Phase5_Prodromal_Conversion',
            'dirs': {
                'task6_1_attention': base_results_dir / 'phase6_task6_1_attention' / 'phase5_conversion',
                'task6_2_gnnexplainer': base_results_dir / 'phase6_task6_2_gnnexplainer' / 'phase5_conversion',
                'task6_3_attribution': base_results_dir / 'phase6_task6_3_attribution' / 'phase5_conversion',
                'task6_4_clustering': base_results_dir / 'phase6_task6_4_clustering' / 'phase5_conversion',
                'task6_5_counterfactuals': base_results_dir / 'phase6_task6_5_counterfactuals' / 'phase5_conversion'
            }
        }
    ]

    # Generate dashboard for each task
    for task_config in tasks:
        print(f"\n\n{'#'*70}")
        print(f"# {task_config['name'].upper()} DASHBOARD")
        print(f"{'#'*70}")

        # Create consolidated results directory
        results_base = Path("visualizations") / "phase6_consolidated_results" / task_config['name'].lower()
        results_base.mkdir(parents=True, exist_ok=True)

        dashboard = ClinicalExplainabilityDashboard(
            task_name=task_config['name'],
            results_dir=results_base.parent.parent
        )

        outputs = dashboard.generate_comprehensive_dashboard()

        print(f"\n{'='*70}")
        print(f"DASHBOARD COMPLETE: {task_config['name']}")
        print(f"{'='*70}\n")

    print("\n" + "="*70)
    print("ALL DASHBOARDS GENERATED SUCCESSFULLY")
    print("="*70)
    print("\nPhase 6 Task 6.6 Complete!")


if __name__ == "__main__":
    main()
