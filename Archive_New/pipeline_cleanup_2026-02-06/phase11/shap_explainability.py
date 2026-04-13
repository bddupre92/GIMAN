import sys
import os
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import shap

# Add project root to sys.path
project_root = Path(__file__).resolve().parents[3]
sys.path.append(str(project_root))

# Import Models
sys.path.append(str(project_root / "archive/development/phase8/subphase8_2_dynamic_endpoints"))
from train_final_giman_survival import GIMANSurvivalGAT
from archive.development.phase9.neuro_fuzzy import NeuroFuzzyGIMAN

# Setup Output Directories
shap_output = project_root / "visualizations/shap_figures"
shap_output.mkdir(parents=True, exist_ok=True)

# Set Style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 300

def explain_phase8_survival(dataset, device):
    """Generate SHAP explanations for Phase 8 Survival Model"""
    print("🔍 Generating SHAP Explanations for Phase 8 (Survival)...")
    
    # Load Model
    in_features = dataset.x.shape[1]
    model = GIMANSurvivalGAT(in_features=in_features, hidden_dim=128).to(device)
    checkpoint = project_root / "outputs/phase8_2_final_training/giman_survival_final.pth"
    
    if not checkpoint.exists():
        print("Phase 8 checkpoint not found. Skipping.")
        return
        
    model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=False)['model_state_dict'])
    model.eval()
    
    # Prepare data for SHAP
    # SHAP expects a function that takes samples and returns predictions
    # For graph models, we need to create a wrapper
    
    dataset = dataset.to(device)
    
    # Create a wrapper function that SHAP can call
    def model_predict(node_features):
        """Wrapper for SHAP - predict risk from node features only"""
        # Convert to tensor
        if not isinstance(node_features, torch.Tensor):
            node_features = torch.FloatTensor(node_features).to(device)
        
        # Create a temporary data object with same structure but new features
        temp_data = dataset.clone()
        temp_data.x = node_features
        
        with torch.no_grad():
            risk_scores = model(temp_data)
        
        return risk_scores.cpu().numpy()
    
    # Sample 100 patients for background
    n_samples = min(100, dataset.num_nodes)
    background_indices = np.random.choice(dataset.num_nodes, n_samples, replace=False)
    background = dataset.x[background_indices].cpu().numpy()
    
    # Select 20 test samples to explain
    test_indices = np.random.choice(dataset.num_nodes, 20, replace=False)
    test_samples = dataset.x[test_indices].cpu().numpy()
    
    print(f"Creating SHAP Explainer with {n_samples} background samples...")
    
    # Use DeepExplainer (GradientExplainer for simpler alternative)
    # For graph models, we use KernelExplainer as it's model-agnostic
    explainer = shap.KernelExplainer(model_predict, background)
    
    print("Computing SHAP values...")
    shap_values = explainer.shap_values(test_samples, nsamples=100)
    
    # Save SHAP values
    np.save(shap_output / "phase8_shap_values.npy", shap_values)
    np.save(shap_output / "phase8_test_samples.npy", test_samples)
    
    # Generate visualizations
    print("Generating SHAP visualizations...")
    
    # 1. Summary Plot
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, test_samples, show=False, max_display=15)
    plt.title("Phase 8: SHAP Feature Importance (Survival Risk)", fontweight='bold')
    plt.tight_layout()
    plt.savefig(shap_output / "Phase8_SHAP_Summary.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Waterfall plot for top 3 highest risk patients
    high_risk_idx = np.argsort(model_predict(test_samples).flatten())[-3:]
    
    for i, idx in enumerate(high_risk_idx):
        plt.figure(figsize=(10, 6))
        shap.waterfall_plot(shap.Explanation(
            values=shap_values[idx], 
            base_values=explainer.expected_value,
            data=test_samples[idx]
        ), show=False)
        plt.title(f"Phase 8: SHAP Waterfall - High Risk Patient {i+1}", fontweight='bold')
        plt.tight_layout()
        plt.savefig(shap_output / f"Phase8_Waterfall_HighRisk_{i+1}.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    print(f"✅ Phase 8 SHAP analysis complete. Saved to {shap_output}")

def explain_phase9_neurofuzzy(dataset, device):
    """Generate SHAP explanations for Phase 9 Neuro-Fuzzy Model"""
    print("\n🔍 Generating SHAP Explanations for Phase 9 (Neuro-Fuzzy)...")
    
    # Load Model
    in_features = dataset.x.shape[1]
    gat_encoder = GIMANSurvivalGAT(in_features=in_features, hidden_dim=128)
    model = NeuroFuzzyGIMAN(gat_encoder, num_classes=2, num_rules=32).to(device)
    checkpoint = project_root / "outputs/phase9_neuro_fuzzy/neuro_fuzzy_best.pth"
    
    if not checkpoint.exists():
        print("Phase 9 checkpoint not found. Skipping.")
        return
        
    model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=False))
    model.eval()
    
    dataset = dataset.to(device)
    
    # Wrapper for classification probability
    def model_predict_proba(node_features):
        """Wrapper for SHAP - predict SAA+ probability"""
        if not isinstance(node_features, torch.Tensor):
            node_features = torch.FloatTensor(node_features).to(device)
        
        temp_data = dataset.clone()
        temp_data.x = node_features
        
        with torch.no_grad():
            logits, _ = model(temp_data)
            probs = F.softmax(logits, dim=1)
        
        return probs[:, 1].cpu().numpy()  # Probability of class 1 (SAA+)
    
    # Sample background and test
    n_samples = min(100, dataset.num_nodes)
    background_indices = np.random.choice(dataset.num_nodes, n_samples, replace=False)
    background = dataset.x[background_indices].cpu().numpy()
    
    # Select SAA+ and SAA- patients
    targets = dataset.event.cpu().numpy()
    pos_idx = np.where(targets == 1)[0]
    neg_idx = np.where(targets == 0)[0]
    
    test_indices = np.concatenate([
        np.random.choice(pos_idx, min(10, len(pos_idx)), replace=False),
        np.random.choice(neg_idx, min(10, len(neg_idx)), replace=False)
    ])
    test_samples = dataset.x[test_indices].cpu().numpy()
    test_labels = targets[test_indices]
    
    print(f"Creating SHAP Explainer...")
    explainer = shap.KernelExplainer(model_predict_proba, background)
    
    print("Computing SHAP values...")
    shap_values = explainer.shap_values(test_samples, nsamples=100)
    
    # Save
    np.save(shap_output / "phase9_shap_values.npy", shap_values)
    np.save(shap_output / "phase9_test_samples.npy", test_samples)
    np.save(shap_output / "phase9_test_labels.npy", test_labels)
    
    # Visualizations
    print("Generating SHAP visualizations...")
    
    # 1. Summary Plot
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, test_samples, show=False, max_display=15)
    plt.title("Phase 9: SHAP Feature Importance (SAA Prediction)", fontweight='bold')
    plt.tight_layout()
    plt.savefig(shap_output / "Phase9_SHAP_Summary.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Waterfall plots for SAA+ and SAA- examples
    pos_test_idx = np.where(test_labels == 1)[0]
    neg_test_idx = np.where(test_labels == 0)[0]
    
    # Top 2 SAA+ patients
    if len(pos_test_idx) > 0:
        for i in range(min(2, len(pos_test_idx))):
            idx = pos_test_idx[i]
            plt.figure(figsize=(10, 6))
            shap.waterfall_plot(shap.Explanation(
                values=shap_values[idx], 
                base_values=explainer.expected_value,
                data=test_samples[idx]
            ), show=False)
            plt.title(f"Phase 9: SHAP Waterfall - SAA+ Patient {i+1}", fontweight='bold')
            plt.tight_layout()
            plt.savefig(shap_output / f"Phase9_Waterfall_SAAPos_{i+1}.png", dpi=300, bbox_inches='tight')
            plt.close()
    
    # Top 2 SAA- patients
    if len(neg_test_idx) > 0:
        for i in range(min(2, len(neg_test_idx))):
            idx = neg_test_idx[i]
            plt.figure(figsize=(10, 6))
            shap.waterfall_plot(shap.Explanation(
                values=shap_values[idx], 
                base_values=explainer.expected_value,
                data=test_samples[idx]
            ), show=False)
            plt.title(f"Phase 9: SHAP Waterfall - SAA- Patient {i+1}", fontweight='bold')
            plt.tight_layout()
            plt.savefig(shap_output / f"Phase9_Waterfall_SAANeg_{i+1}.png", dpi=300, bbox_inches='tight')
            plt.close()
    
    print(f"✅ Phase 9 SHAP analysis complete. Saved to {shap_output}")

def main():
    print("="*80)
    print("PHASE 11: SHAP EXPLAINABILITY ANALYSIS")
    print("="*80)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load Data
    data_path = project_root / "data/03_prodromal/final_pyg_data/train_data.pt"
    if not data_path.exists():
        print(f"Data not found at {data_path}")
        return
    
    dataset = torch.load(data_path, weights_only=False)
    
    # Run SHAP analyses
    explain_phase8_survival(dataset, device)
    explain_phase9_neurofuzzy(dataset, device)
    
    print("\n" + "="*80)
    print("SHAP ANALYSIS COMPLETE")
    print("="*80)
    print(f"\nAll SHAP figures saved to: {shap_output}")

if __name__ == "__main__":
    main()
