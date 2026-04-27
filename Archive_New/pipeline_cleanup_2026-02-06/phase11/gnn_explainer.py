import sys
from pathlib import Path
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from torch_geometric.explain import Explainer, GNNExplainer
from torch_geometric.explain import unfaithfulness, fidelity

# Add project root
project_root = Path(__file__).resolve().parents[3]
sys.path.append(str(project_root))

sys.path.append(str(project_root / "archive/development/phase8/subphase8_2_dynamic_endpoints"))
from train_final_giman_survival import GIMANSurvivalGAT
from archive.development.phase9.neuro_fuzzy import NeuroFuzzyGIMAN

# Setup output
gnn_explain_output = project_root / "visualizations/gnn_explainability"
gnn_explain_output.mkdir(parents=True, exist_ok=True)

sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 300

def explain_phase8_with_gnn_explainer(dataset, device):
    """Use GNNExplainer on Phase 8 Survival Model"""
    print("="*80)
    print("GNNExplainer: Phase 8 Survival Model")
    print("="*80)
    
    # Load model
    in_features = dataset.x.shape[1]
    model = GIMANSurvivalGAT(in_features=in_features, hidden_dim=128).to(device)
    checkpoint = project_root / "outputs/phase8_2_final_training/giman_survival_final.pth"
    
    if not checkpoint.exists():
        print("Phase 8 checkpoint not found.")
        return
    
    model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=False)['model_state_dict'])
    model.eval()
    
    dataset = dataset.to(device)
    
    # Get risk predictions
    with torch.no_grad():
        risk_scores = model(dataset)
    
    # Select high-risk and low-risk patients for explanation
    risk_np = risk_scores.cpu().numpy().flatten()
    high_risk_idx = np.argsort(risk_np)[-10:]  # Top 10 highest risk
    low_risk_idx = np.argsort(risk_np)[:10]    # Top 10 lowest risk
    patients_to_explain = np.concatenate([high_risk_idx[:3], low_risk_idx[:3]])
    
    print(f"\nExplaining {len(patients_to_explain)} patients...")
    print(f"  High risk: indices {high_risk_idx[:3]}")
    print(f"  Low risk: indices {low_risk_idx[:3]}")
    
    # Create wrapper model for Explainer API
    class SurvivalWrapper(torch.nn.Module):
        def __init__(self, model, data):
            super().__init__()
            self.model = model
            self.data = data
        
        def forward(self, x, edge_index, **kwargs):
            # Create temporary data object
            from torch_geometric.data import Data
            temp_data = Data(x=x, edge_index=edge_index,
                           time=self.data.time, event=self.data.event)
            return self.model(temp_data)
    
    wrapped_model = SurvivalWrapper(model, dataset).to(device)
    
    # Initialize GNNExplainer
    explainer = Explainer(
        model=wrapped_model,
        algorithm=GNNExplainer(epochs=100),
        explanation_type='model',
        node_mask_type='attributes',
        edge_mask_type='object',
        model_config=dict(
            mode='regression',
            task_level='node',
            return_type='raw',
        ),
    )
    
    # Explain each patient
    explanations = []
    feature_importances = []
    
    for idx in patients_to_explain:
        print(f"\n  Explaining patient {idx} (risk: {risk_np[idx]:.4f})...")
        
        explanation = explainer(
            dataset.x,
            dataset.edge_index,
            index=idx,
        )
        
        # Get feature importance (node_mask)
        if hasattr(explanation, 'node_mask') and explanation.node_mask is not None:
            feature_imp = explanation.node_mask[idx].cpu().numpy()
            feature_importances.append(feature_imp)
        
        explanations.append({
            'patient_idx': idx,
            'risk_score': risk_np[idx],
            'explanation': explanation
        })
    
    # Aggregate feature importances
    if feature_importances:
        avg_feature_importance = np.mean(feature_importances, axis=0)
        
        # Plot top features
        top_k = 15
        top_features = np.argsort(avg_feature_importance)[::-1][:top_k]
        
        plt.figure(figsize=(10, 6))
        plt.barh(range(top_k), avg_feature_importance[top_features])
        plt.yticks(range(top_k), [f'Feature {i}' for i in top_features])
        plt.xlabel('Average Feature Importance')
        plt.title('Phase 8: GNNExplainer - Top Features for Survival Risk', fontweight='bold')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.savefig(gnn_explain_output / "phase8_gnn_feature_importance.png")
        plt.close()
        print(f"\n✅ Saved feature importance plot")
        
        # Save detailed results
        results_df = pd.DataFrame({
            'feature_idx': top_features,
            'importance': avg_feature_importance[top_features]
        })
        results_df.to_csv(gnn_explain_output / "phase8_gnn_top_features.csv", index=False)
    
    return explanations

def explain_phase9_with_gnn_explainer(dataset, device):
    """Use GNNExplainer on Phase 9 Neuro-Fuzzy Model"""
    print("\n" + "="*80)
    print("GNNExplainer: Phase 9 Neuro-Fuzzy Model")
    print("="*80)
    
    # Load model
    in_features = dataset.x.shape[1]
    gat_encoder = GIMANSurvivalGAT(in_features=in_features, hidden_dim=128)
    model = NeuroFuzzyGIMAN(gat_encoder, num_classes=2, num_rules=32).to(device)
    checkpoint = project_root / "outputs/phase9_neuro_fuzzy/neuro_fuzzy_best.pth"
    
    if not checkpoint.exists():
        print("Phase 9 checkpoint not found.")
        return
    
    model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=False))
    model.eval()
    
    dataset = dataset.to(device)
    
    # Get predictions
    with torch.no_grad():
        logits, weights = model(dataset)
        probs = F.softmax(logits, dim=1)[:, 1].cpu().numpy()
    
    # Select SAA+ and SAA- patients
    targets = dataset.event.cpu().numpy()
    saa_pos_idx = np.where(targets == 1)[0]
    saa_neg_idx = np.where(targets == 0)[0]
    
    # Select patients with high confidence
    pos_confident = saa_pos_idx[np.argsort(probs[saa_pos_idx])[-3:]]
    neg_confident = saa_neg_idx[np.argsort(probs[saa_neg_idx])[:3]]
    patients_to_explain = np.concatenate([pos_confident, neg_confident])
    
    print(f"\nExplaining {len(patients_to_explain)} patients...")
    print(f"  SAA+ (confident): indices {pos_confident}")
    print(f"  SAA- (confident): indices {neg_confident}")
    
    # Wrapper model that returns class 1 logits (SAA+)
    class NF_Wrapper(torch.nn.Module):
        def __init__(self, model, data):
            super().__init__()
            self.model = model
            self.data = data
        
        def forward(self, x, edge_index, **kwargs):
            # Create temp data object
            from torch_geometric.data import Data
            temp_data = Data(x=x, edge_index=edge_index,
                           time=self.data.time, event=self.data.event)
            logits, _ = self.model(temp_data)
            return logits
    
    wrapped_model = NF_Wrapper(model, dataset).to(device)
    
    # Initialize explainer
    explainer = Explainer(
        model=wrapped_model,
        algorithm=GNNExplainer(epochs=100),
        explanation_type='model',
        node_mask_type='attributes',
        edge_mask_type='object',
        model_config=dict(
            mode='multiclass_classification',
            task_level='node',
            return_type='raw',
        ),
    )
    
    # Explain each patient
    feature_importances = []
    
    for idx in patients_to_explain:
        print(f"\n  Explaining patient {idx} (SAA prob: {probs[idx]:.4f})...")
        
        explanation = explainer(
            dataset.x,
            dataset.edge_index,
            target=1,  # Explain class 1 (SAA+)
            index=idx,
        )
        
        if hasattr(explanation, 'node_mask') and explanation.node_mask is not None:
            feature_imp = explanation.node_mask[idx].cpu().numpy()
            feature_importances.append(feature_imp)
    
    # Aggregate and visualize
    if feature_importances:
        avg_feature_importance = np.mean(feature_importances, axis=0)
        
        top_k = 15
        top_features = np.argsort(avg_feature_importance)[::-1][:top_k]
        
        plt.figure(figsize=(10, 6))
        plt.barh(range(top_k), avg_feature_importance[top_features])
        plt.yticks(range(top_k), [f'Feature {i}' for i in top_features])
        plt.xlabel('Average Feature Importance')
        plt.title('Phase 9: GNNExplainer - Top Features for SAA Prediction', fontweight='bold')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.savefig(gnn_explain_output / "phase9_gnn_feature_importance.png")
        plt.close()
        print(f"\n✅ Saved feature importance plot")
        
        # Compare with fuzzy rule importance
        rule_importance = (model.output_layer.weight[1] - model.output_layer.weight[0]).detach().cpu().numpy()
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # GNN feature importance
        axes[0].barh(range(top_k), avg_feature_importance[top_features], color='steelblue')
        axes[0].set_yticks(range(top_k))
        axes[0].set_yticklabels([f'F{i}' for i in top_features])
        axes[0].set_xlabel('GNNExplainer Importance')
        axes[0].set_title('Feature-Level Explanation')
        axes[0].invert_yaxis()
        
        # Fuzzy rule importance
        top_rules = np.argsort(np.abs(rule_importance))[::-1][:15]
        colors = ['red' if rule_importance[i] > 0 else 'blue' for i in top_rules]
        axes[1].barh(range(15), rule_importance[top_rules], color=colors)
        axes[1].set_yticks(range(15))
        axes[1].set_yticklabels([f'R{i}' for i in top_rules])
        axes[1].set_xlabel('Rule Importance (Red=SAA+)')
        axes[1].set_title('Rule-Level Explanation')
        axes[1].axvline(0, color='black', linewidth=0.8)
        axes[1].invert_yaxis()
        
        plt.suptitle('Phase 9: Multi-Level Explainability (GNN + Fuzzy Rules)', fontweight='bold')
        plt.tight_layout()
        plt.savefig(gnn_explain_output / "phase9_multilevel_explanation.png")
        plt.close()
        print(f"✅ Saved multi-level explanation comparison")
        
        results_df = pd.DataFrame({
            'feature_idx': top_features,
            'gnn_importance': avg_feature_importance[top_features]
        })
        results_df.to_csv(gnn_explain_output / "phase9_gnn_top_features.csv", index=False)

def main():
    print("="*80)
    print("GNN EXPLAINABILITY: GNNExplainer for GIMAN Models")
    print("="*80)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}\n")
    
    # Load data
    data_path = project_root / "data/03_prodromal/final_pyg_data/train_data.pt"
    if not data_path.exists():
        print(f"Data not found at {data_path}")
        return
    
    dataset = torch.load(data_path, weights_only=False)
    
    # Run explainers
    explain_phase8_with_gnn_explainer(dataset, device)
    explain_phase9_with_gnn_explainer(dataset, device)
    
    print("\n" + "="*80)
    print("GNN EXPLAINABILITY COMPLETE")
    print("="*80)
    print(f"\nAll visualizations saved to: {gnn_explain_output}")
    print("\nKey Insights:")
    print("  - GNNExplainer identifies which node features contribute most to predictions")
    print("  - For Phase 9, we can compare feature-level (GNN) with rule-level (Fuzzy) explanations")
    print("  - This provides multi-level interpretability: features → rules → predictions")

if __name__ == "__main__":
    main()
