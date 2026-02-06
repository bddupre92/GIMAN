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
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

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

def plot_cohort_characteristics(dataset):
    """Figure 1: Cohort Characteristics (Age Distribution)"""
    print("Generating Figure 1: Cohort Characteristics...")
    
    # Extract features (assuming index 0 is Age, based on standard preprocessing)
    # If feature names are not available, we'll plot the first feature distribution
    # dataset.x shape: (num_nodes, num_features)
    features = dataset.x.cpu().numpy()
    
    # Assuming normalized age is the first feature or similar. 
    # For visualization, we'll just plot the distribution of the first principal component of features
    # to show population diversity if specific features aren't labeled.
    # BETTER: Plot the distribution of the target variable (Time to Event)
    
    times = dataset.time.cpu().numpy()
    events = dataset.event.cpu().numpy()
    
    plt.figure(figsize=(10, 6))
    sns.histplot(times[events==1], color='red', label='Converters', kde=True, alpha=0.5)
    sns.histplot(times[events==0], color='blue', label='Non-Converters', kde=True, alpha=0.5)
    
    plt.title("Distribution of Follow-up Time / Time-to-Conversion", fontsize=14, fontweight='bold')
    plt.xlabel("Time (Months)", fontsize=12)
    plt.ylabel("Count", fontsize=12)
    plt.legend()
    plt.tight_layout()
    
    plt.savefig(output_dir / "Figure1_Cohort_Characteristics.png")
    plt.close()

def plot_modality_alignment():
    """Figure 2: Modality Alignment (t-SNE of Phase 2 Embeddings)"""
    print("Generating Figure 2: Modality Alignment...")
    
    emb_path = project_root / "archive/development/phase2/embeddings_output/spatiotemporal_embeddings.csv"
    if not emb_path.exists():
        print(f"Phase 2 embeddings not found at {emb_path}. Skipping.")
        return

    df = pd.read_csv(emb_path)
    # Columns are 'embedding_000', 'embedding_001', etc.
    
    feature_cols = [c for c in df.columns if c.startswith('embedding_')]
    X = df[feature_cols].values
    
    # t-SNE
    n_samples = X.shape[0]
    perplexity = min(30, n_samples - 1) if n_samples > 1 else 1
    print(f"Running t-SNE with perplexity={perplexity} on {n_samples} samples")
    
    tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity)
    X_emb = tsne.fit_transform(X)
    
    plt.figure(figsize=(10, 8))
    plt.scatter(X_emb[:, 0], X_emb[:, 1], alpha=0.6, s=30, c='#3498db')
    plt.title("Phase 2: Multimodal Embedding Space (t-SNE)", fontsize=14, fontweight='bold')
    plt.xlabel("t-SNE 1", fontsize=12)
    plt.ylabel("t-SNE 2", fontsize=12)
    plt.tight_layout()
    
    plt.savefig(output_dir / "Figure2_Modality_Alignment.png")
    plt.close()

def plot_progression_subtypes():
    """Figure 3: Progression Subtypes (Phase 4 Clusters)"""
    print("Generating Figure 3: Progression Subtypes...")
    
    emb_path = project_root / "data/longitudinal_cohort/vader_embeddings.csv"
    if not emb_path.exists():
        print(f"Phase 4 embeddings not found at {emb_path}. Skipping.")
        return
        
    df = pd.read_csv(emb_path)
    # Columns: emb_0...emb_15, PATNO, cluster
    
    feature_cols = [c for c in df.columns if c.startswith('emb_')]
    X = df[feature_cols].values
    clusters = df['cluster'].values
    
    # PCA for visualization (VaDER latent space is already compressed, PCA is good)
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)
    
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=clusters, cmap='viridis', alpha=0.7, s=40)
    plt.colorbar(scatter, label='Subtype Cluster')
    
    plt.title("Phase 4: Discovered Progression Subtypes (VaDER Latent Space)", fontsize=14, fontweight='bold')
    plt.xlabel("PC1", fontsize=12)
    plt.ylabel("PC2", fontsize=12)
    plt.tight_layout()
    
    plt.savefig(output_dir / "Figure3_Progression_Subtypes.png")
    plt.close()

def plot_survival_risk(dataset, device):
    """Figure 4: Risk Stratification (Phase 8)"""
    print("Generating Figure 4: Survival Risk Stratification...")
    
    # Load Model
    in_features = dataset.x.shape[1]
    model = GIMANSurvivalGAT(in_features=in_features, hidden_dim=128).to(device)
    checkpoint = project_root / "outputs/phase8_2_final_training/giman_survival_final.pth"
    
    if not checkpoint.exists():
        print("Phase 8 checkpoint not found. Skipping.")
        return
        
    model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=False)['model_state_dict'])
    model.eval()
    
    with torch.no_grad():
        dataset = dataset.to(device)
        risk_scores = model(dataset).cpu().numpy()
        times = dataset.time.cpu().numpy()
        events = dataset.event.cpu().numpy()
        
    # Stratify
    p33, p66 = np.percentile(risk_scores, [33, 66])
    groups = np.zeros_like(risk_scores)
    groups[risk_scores > p33] = 1
    groups[risk_scores > p66] = 2
    
    kmf = KaplanMeierFitter()
    plt.figure(figsize=(10, 6))
    labels = ['Low Risk', 'Medium Risk', 'High Risk']
    colors = ['#2ecc71', '#f1c40f', '#e74c3c']
    
    for i in range(3):
        mask = (groups == i)
        if mask.sum() > 0:
            kmf.fit(times[mask], events[mask], label=f"{labels[i]} (n={mask.sum()})")
            kmf.plot(ci_show=True, color=colors[i], linewidth=2)
            
    plt.title("Phase 8: Survival Risk Stratification", fontsize=14, fontweight='bold')
    plt.xlabel("Time (Months)", fontsize=12)
    plt.ylabel("Survival Probability", fontsize=12)
    plt.tight_layout()
    
    plt.savefig(output_dir / "Figure4_Survival_Risk.png")
    plt.close()

def plot_neuro_fuzzy_interpretability(dataset, device):
    """Figure 5 & 6: Neuro-Fuzzy Interpretability (Phase 9)"""
    print("Generating Figure 5 & 6: Neuro-Fuzzy Interpretability...")
    
    in_features = dataset.x.shape[1]
    gat_encoder = GIMANSurvivalGAT(in_features=in_features, hidden_dim=128)
    model = NeuroFuzzyGIMAN(gat_encoder, num_classes=2, num_rules=32).to(device)
    checkpoint = project_root / "outputs/phase9_neuro_fuzzy/neuro_fuzzy_best.pth"
    
    if not checkpoint.exists():
        print("Phase 9 checkpoint not found. Skipping.")
        return
        
    model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=False))
    model.eval()
    
    with torch.no_grad():
        dataset = dataset.to(device)
        logits, weights = model(dataset)
        
    # Figure 5: Heatmap
    # Select top 20 SAA+ and 20 SAA-
    targets = dataset.event.cpu().numpy()
    pos_idx = np.where(targets == 1)[0][:20]
    neg_idx = np.where(targets == 0)[0][:20]
    indices = np.concatenate([pos_idx, neg_idx])
    
    subset_weights = weights[indices].cpu().numpy()
    # Sort rules by variance
    top_rules = np.argsort(np.var(subset_weights, axis=0))[::-1][:10]
    
    plt.figure(figsize=(12, 8))
    y_labels = ['SAA+'] * len(pos_idx) + ['SAA-'] * len(neg_idx)
    sns.heatmap(subset_weights[:, top_rules], cmap="viridis", 
                xticklabels=[f"R{i}" for i in top_rules], yticklabels=y_labels)
    plt.title("Phase 9: Fuzzy Rule Activation Heatmap", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / "Figure5_Fuzzy_Heatmap.png")
    plt.close()
    
    # Figure 6: Importance
    importance = (model.output_layer.weight[1] - model.output_layer.weight[0]).detach().cpu().numpy()
    top_idx = np.argsort(np.abs(importance))[::-1][:10]
    
    plt.figure(figsize=(10, 6))
    colors = ['#e74c3c' if importance[i] > 0 else '#3498db' for i in top_idx]
    sns.barplot(x=[f"R{i}" for i in top_idx], y=importance[top_idx], palette=colors)
    plt.title("Phase 9: Top Fuzzy Rule Importance", fontsize=14, fontweight='bold')
    plt.axhline(0, color='black')
    plt.tight_layout()
    plt.savefig(output_dir / "Figure6_Rule_Importance.png")
    plt.close()

def main():
    print("🚀 Generating Comprehensive Visualization Suite...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load Data
    data_path = project_root / "data/03_prodromal/final_pyg_data/train_data.pt"
    if data_path.exists():
        dataset = torch.load(data_path, weights_only=False)
        
        plot_cohort_characteristics(dataset)
        plot_modality_alignment()
        plot_progression_subtypes()
        plot_survival_risk(dataset, device)
        plot_neuro_fuzzy_interpretability(dataset, device)
        
        print("\n✅ All figures generated in visualizations/paper_figures/")
    else:
        print(f"Data not found at {data_path}")

if __name__ == "__main__":
    main()
