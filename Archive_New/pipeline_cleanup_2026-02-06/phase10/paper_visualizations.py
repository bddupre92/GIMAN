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
from lifelines import KaplanMeierFitter
from sklearn.metrics import roc_auc_score

# Add project root to sys.path
project_root = Path(__file__).resolve().parents[3]
sys.path.append(str(project_root))

# Import Models
sys.path.append(str(project_root / "archive/development/phase8/subphase8_2_dynamic_endpoints"))
from train_final_giman_survival import GIMANSurvivalGAT

from archive.development.phase9.neuro_fuzzy import NeuroFuzzyGIMAN

# Setup Output Directory
output_dir = project_root / "visualizations/paper_figures"
output_dir.mkdir(parents=True, exist_ok=True)

# Set Style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['font.family'] = 'sans-serif'

def load_data():
    data_path = project_root / "data/03_prodromal/final_pyg_data/train_data.pt"
    dataset = torch.load(data_path, weights_only=False)
    return dataset

def plot_kaplan_meier_stratification(dataset, model, device):
    """
    Figure 1: Risk Stratification (Kaplan-Meier)
    """
    print("Generating Figure 1: Kaplan-Meier Stratification...")
    
    model.eval()
    with torch.no_grad():
        dataset = dataset.to(device)
        risk_scores = model(dataset).cpu().numpy()
        times = dataset.time.cpu().numpy()
        events = dataset.event.cpu().numpy()
        
    # Stratify into Low, Medium, High Risk
    risk_percentiles = np.percentile(risk_scores, [33, 66])
    groups = np.zeros_like(risk_scores)
    groups[risk_scores > risk_percentiles[0]] = 1
    groups[risk_scores > risk_percentiles[1]] = 2
    
    kmf = KaplanMeierFitter()
    
    plt.figure(figsize=(10, 6))
    
    labels = ['Low Risk', 'Medium Risk', 'High Risk']
    colors = ['#2ecc71', '#f1c40f', '#e74c3c']
    
    print(f"Risk Scores: Min={risk_scores.min():.4f}, Max={risk_scores.max():.4f}")
    print(f"Percentiles: {risk_percentiles}")
    
    for i in range(3):
        mask = (groups == i)
        print(f"Group {i} ({labels[i]}): {mask.sum()} patients")
        
        if mask.sum() > 0:
            try:
                kmf.fit(times[mask], events[mask], label=f"{labels[i]} (n={mask.sum()})")
                kmf.plot(ci_show=True, color=colors[i], linewidth=2)
            except Exception as e:
                print(f"Error fitting Group {i}: {e}")
                print(f"Times type: {times[mask].dtype}, Events type: {events[mask].dtype}")
        else:
            print(f"Skipping Group {i} (Empty)")
        
    plt.title("GIMAN-GAT Risk Stratification (Prodromal to PD Conversion)", fontsize=14, fontweight='bold')
    plt.xlabel("Time (Months)", fontsize=12)
    plt.ylabel("Survival Probability (Non-Converter)", fontsize=12)
    plt.legend(fontsize=10)
    plt.tight_layout()
    
    save_path = output_dir / "Figure1_KM_Stratification.png"
    plt.savefig(save_path)
    print(f"Saved {save_path}")
    plt.close()

def plot_fuzzy_rule_heatmap(dataset, model, device):
    """
    Figure 3: Fuzzy Rule Heatmap
    """
    print("Generating Figure 3: Fuzzy Rule Heatmap...")
    
    model.eval()
    with torch.no_grad():
        dataset = dataset.to(device)
        # We need the weights (firing strengths)
        # Forward pass returns: logits, weights
        logits, weights = model(dataset)
        
        # Get SAA status (using event as proxy for now as per training)
        targets = dataset.event.cpu().numpy()
        
        # Select a subset of patients (e.g., 20 Positive, 20 Negative)
        pos_indices = np.where(targets == 1)[0][:20]
        neg_indices = np.where(targets == 0)[0][:20]
        indices = np.concatenate([pos_indices, neg_indices])
        
        # Extract weights for these patients
        # weights shape: (num_nodes, num_rules)
        subset_weights = weights[indices].cpu().numpy()
        
        # Sort rules by importance (variance or magnitude)
        # Or use the top rules from training
        # Let's use variance across patients to find discriminative rules
        rule_variance = np.var(subset_weights, axis=0)
        top_rule_indices = np.argsort(rule_variance)[::-1][:10]
        
        heatmap_data = subset_weights[:, top_rule_indices]
        
    plt.figure(figsize=(12, 8))
    
    # Create annotation for y-axis
    y_labels = ['SAA+'] * 20 + ['SAA-'] * 20
    
    sns.heatmap(heatmap_data, cmap="viridis", xticklabels=[f"Rule {i}" for i in top_rule_indices], yticklabels=y_labels)
    
    plt.title("Neuro-Fuzzy Rule Activation Patterns (Top 10 Discriminative Rules)", fontsize=14, fontweight='bold')
    plt.xlabel("Fuzzy Rules", fontsize=12)
    plt.ylabel("Patients", fontsize=12)
    plt.tight_layout()
    
    save_path = output_dir / "Figure3_Fuzzy_Heatmap.png"
    plt.savefig(save_path)
    print(f"Saved {save_path}")
    plt.close()

def plot_rule_importance(model):
    """
    Figure 4: Rule Importance Bar Plot
    """
    print("Generating Figure 4: Rule Importance...")
    
    # Calculate importance based on consequent weights
    # Importance = |weight_class_1 - weight_class_0|
    # output_layer weight shape: (num_classes, num_rules)
    importance = model.output_layer.weight[1] - model.output_layer.weight[0]
    importance = importance.detach().cpu().numpy()
    
    # Sort
    indices = np.argsort(np.abs(importance))[::-1][:10]
    top_importance = importance[indices]
    
    plt.figure(figsize=(10, 6))
    
    colors = ['#e74c3c' if x > 0 else '#3498db' for x in top_importance]
    
    sns.barplot(x=[f"Rule {i}" for i in indices], y=top_importance, palette=colors)
    
    plt.title("Top 10 Fuzzy Rules by Predictive Importance", fontsize=14, fontweight='bold')
    plt.xlabel("Fuzzy Rules", fontsize=12)
    plt.ylabel("Importance Score (Positive = Predicts SAA+)", fontsize=12)
    plt.axhline(0, color='black', linewidth=0.8)
    plt.tight_layout()
    
    save_path = output_dir / "Figure4_Rule_Importance.png"
    plt.savefig(save_path)
    print(f"Saved {save_path}")
    plt.close()

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataset = load_data()
    
    # 1. Load Phase 8 Model (Survival)
    print("Loading Phase 8 Model...")
    in_features = dataset.x.shape[1]
    gat_model = GIMANSurvivalGAT(in_features=in_features, hidden_dim=128).to(device)
    checkpoint_p8 = project_root / "outputs/phase8_2_final_training/giman_survival_final.pth"
    if checkpoint_p8.exists():
        cp = torch.load(checkpoint_p8, map_location=device, weights_only=False)
        gat_model.load_state_dict(cp['model_state_dict'])
        plot_kaplan_meier_stratification(dataset, gat_model, device)
    else:
        print("Phase 8 checkpoint not found. Skipping Figure 1.")

    # 2. Load Phase 9 Model (Neuro-Fuzzy)
    print("Loading Phase 9 Model...")
    # Re-initialize GAT for Phase 9 wrapper
    gat_encoder = GIMANSurvivalGAT(in_features=in_features, hidden_dim=128)
    nf_model = NeuroFuzzyGIMAN(gat_encoder, num_classes=2, num_rules=32).to(device)
    checkpoint_p9 = project_root / "outputs/phase9_neuro_fuzzy/neuro_fuzzy_best.pth"
    if checkpoint_p9.exists():
        nf_model.load_state_dict(torch.load(checkpoint_p9, map_location=device, weights_only=False))
        plot_fuzzy_rule_heatmap(dataset, nf_model, device)
        plot_rule_importance(nf_model)
    else:
        print("Phase 9 checkpoint not found. Skipping Figures 3 & 4.")
        
    print("\nVisualization Generation Complete!")

if __name__ == "__main__":
    main()
