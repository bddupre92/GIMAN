"""
Figure 9: Patient Embedding Space (t-SNE/UMAP)
Panel A: Hard clustering (crisp boundaries)
Panel B: Fuzzy membership (gradient coloring)
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.manifold import TSNE

project_root = Path(__file__).resolve().parents[3]
output_dir = project_root / "visualizations/publication"
output_dir.mkdir(parents=True, exist_ok=True)

# Set style
plt.rcParams['figure.dpi'] = 300
plt.rcParams['font.size'] = 10
plt.rcParams['font.family'] = 'sans-serif'

def load_embedding_data():
    """Load VaDER embeddings and FCM results"""
    # VaDER embeddings
    vader_emb_path = project_root / "data/longitudinal_cohort/vader_embeddings.csv"
    vader_df = pd.read_csv(vader_emb_path)
    
    # Extract embedding columns (emb_0 through emb_15)
    emb_cols = [f'emb_{i}' for i in range(16)]
    embeddings = vader_df[emb_cols].values
    
    # VaDER hard clusters
    traj_path = project_root / "data/longitudinal_cohort/patient_trajectories_clustered.csv"
    traj_df = pd.read_csv(traj_path)
    hard_clusters = traj_df['cluster'].values
    
    # FCM fuzzy memberships
    fcm_path = project_root / "visualizations/fcm_clustering/fuzzy_cmeans_results.csv"
    fcm_df = pd.read_csv(fcm_path)
    fuzzy_memberships = fcm_df[['fcm_membership_c0', 'fcm_membership_c1', 'fcm_membership_c2']].values
    
    print(f"Loaded {len(embeddings)} patient embeddings")
    print(f"Embedding dimension: {embeddings.shape[1]}")
    print(f"Hard clusters: {np.unique(hard_clusters)}")
    print(f"Fuzzy membership shape: {fuzzy_memberships.shape}")
    
    return embeddings, hard_clusters, fuzzy_memberships

def create_tsne_projection(embeddings):
    """Create t-SNE 2D projection"""
    print("Computing t-SNE projection...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    coords_2d = tsne.fit_transform(embeddings)
    print(f"t-SNE projection complete: {coords_2d.shape}")
    return coords_2d

def rgb_from_fuzzy_membership(fuzzy_memberships):
    """Convert fuzzy memberships to RGB colors with vivid saturation"""
    # Use exact same colors as Panel A for consistency
    # Convert hex to RGB
    def hex_to_rgb(hex_color):
        hex_color = hex_color.lstrip('#')
        return np.array([int(hex_color[i:i+2], 16) for i in (0, 2, 4)]) / 255.0
    
    colors = np.array([
        hex_to_rgb('#e63946'),  # Red - Cluster 0 (Rapid)
        hex_to_rgb('#457b9d'),  # Blue - Cluster 1 (Moderate)
        hex_to_rgb('#2a9d8f')   # Green - Cluster 2 (Slow)
    ])
    
    # Weighted sum based on memberships
    rgb_colors = fuzzy_memberships @ colors
    
    # Enhance saturation: scale up the differences
    # Find dominant color for each patient
    dominant_idx = fuzzy_memberships.argmax(axis=1)
    dominant_strength = fuzzy_memberships.max(axis=1)
    
    # Amplify dominant color while preserving gradients
    for i in range(len(rgb_colors)):
        # Scale towards dominant color based on membership strength
        base_color = colors[dominant_idx[i]]
        strength = (dominant_strength[i] - 0.333) / (1.0 - 0.333)  # Normalize from 0.333-1.0 to 0-1
        strength = np.clip(strength ** 0.7, 0, 1)  # Power curve for better visibility
        rgb_colors[i] = rgb_colors[i] * (1 - strength * 0.3) + base_color * strength * 0.3
    
    return np.clip(rgb_colors, 0, 1)

def create_embedding_figure(coords_2d, hard_clusters, fuzzy_memberships):
    """Create two-panel figure"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Overall title
    fig.suptitle('Patient Embedding Space: Hard vs Fuzzy Clustering', 
                 fontsize=14, fontweight='bold', y=0.98)
    
    # EXACT colors for both panels - matching legend
    cluster_colors = ['#e63946', '#457b9d', '#2a9d8f']  # Red, Blue, Green
    
    # === PANEL A: Hard Clustering ===
    ax1.set_title('Panel A: Hard K-Means Clustering\n(Crisp Boundaries)', 
                  fontsize=11, fontweight='bold')
    
    for cluster_id in np.unique(hard_clusters):
        mask = hard_clusters == cluster_id
        ax1.scatter(coords_2d[mask, 0], coords_2d[mask, 1], 
                   c=cluster_colors[cluster_id], 
                   label=f'Cluster {cluster_id+1}',
                   s=50, alpha=0.7, edgecolors='black', linewidth=0.5)
    
    ax1.set_xlabel('t-SNE Dimension 1', fontsize=10)
    ax1.set_ylabel('t-SNE Dimension 2', fontsize=10)
    ax1.legend(loc='upper right', frameon=True, fontsize=9)
    ax1.grid(True, alpha=0.3, linestyle='--')
    
    # Add annotation
    ax1.text(0.02, 0.02, 'Sharp boundaries\nbetween subtypes', 
            transform=ax1.transAxes, fontsize=9, style='italic',
            verticalalignment='bottom',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.8, edgecolor='red'))
    
    # === PANEL B: Fuzzy Clustering ===
    ax2.set_title('Panel B: Fuzzy C-Means Clustering\n(Soft Gradients)', 
                  fontsize=11, fontweight='bold')
    
    # Convert fuzzy memberships to RGB
    rgb_colors = rgb_from_fuzzy_membership(fuzzy_memberships)
    
    ax2.scatter(coords_2d[:, 0], coords_2d[:, 1], 
               c=rgb_colors, 
               s=50, alpha=0.8, edgecolors='black', linewidth=0.5)
    
    ax2.set_xlabel('t-SNE Dimension 1', fontsize=10)
    ax2.set_ylabel('t-SNE Dimension 2', fontsize=10)
    ax2.grid(True, alpha=0.3, linestyle='--')
    
    # Add annotation
    ax2.text(0.02, 0.02, 'Smooth transitions\nreflect biological reality', 
            transform=ax2.transAxes, fontsize=9, style='italic',
            verticalalignment='bottom',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.8, edgecolor='green'))
    
    # Add color legend for fuzzy (show pure colors)
    legend_elements = [
        mpatches.Patch(facecolor=cluster_colors[0], label='Rapid Progressor', edgecolor='black'),
        mpatches.Patch(facecolor=cluster_colors[1], label='Moderate Progressor', edgecolor='black'),
        mpatches.Patch(facecolor=cluster_colors[2], label='Slow Progressor', edgecolor='black'),
        mpatches.Patch(facecolor='white', label='Mixed (gradient)', edgecolor='black', linestyle='--', linewidth=2)
    ]
    ax2.legend(handles=legend_elements, loc='upper right', frameon=True, fontsize=9)
    
    plt.tight_layout()
    output_path = output_dir / "Figure9_Embedding_Space.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"✅ Saved Figure 9: {output_path}")

def main():
    print("="*80)
    print("FIGURE 9: PATIENT EMBEDDING SPACE")
    print("="*80)
    
    # Load data
    embeddings, hard_clusters, fuzzy_memberships = load_embedding_data()
    
    # Create t-SNE projection
    coords_2d = create_tsne_projection(embeddings)
    
    # Create figure
    create_embedding_figure(coords_2d, hard_clusters, fuzzy_memberships)
    
    print("\n✅ Figure 9 complete!")

if __name__ == "__main__":
    main()
