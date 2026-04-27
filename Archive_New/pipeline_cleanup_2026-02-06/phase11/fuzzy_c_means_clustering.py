import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import seaborn as sns
import skfuzzy as fuzz
from sklearn.decomposition import PCA
import plotly.graph_objects as go
import plotly.express as px

# Project root
project_root = Path(__file__).resolve().parents[3]

# Setup output
fcm_output = project_root / "visualizations/fcm_clustering"
fcm_output.mkdir(parents=True, exist_ok=True)
interactive_output = project_root / "visualizations/interactive"

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 300

def perform_fuzzy_cmeans_clustering():
    """
    Implement Fuzzy C-Means clustering on Phase 4 VaDER embeddings
    and compare with hard K-Means clusters
    """
    print("="*80)
    print("FUZZY C-MEANS CLUSTERING ANALYSIS")
    print("="*80)
    
    # Load VaDER embeddings
    emb_path = project_root / "data/longitudinal_cohort/vader_embeddings.csv"
    if not emb_path.exists():
        print(f"VaDER embeddings not found at {emb_path}")
        return
    
    df = pd.read_csv(emb_path)
    feature_cols = [c for c in df.columns if c.startswith('emb_')]
    X = df[feature_cols].values
    patient_ids = df['PATNO'].values
    hard_clusters = df['cluster'].values if 'cluster' in df.columns else None
    
    print(f"\nLoaded {len(X)} patients with {X.shape[1]} embedding dimensions")
    
    # Determine optimal number of clusters (3-5 as per paper)
    n_clusters_range = [3, 4, 5]
    fpc_scores = []
    
    print("\nEvaluating optimal number of clusters...")
    for n_clusters in n_clusters_range:
        cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(
            X.T, n_clusters, 2, error=0.005, maxiter=1000, init=None
        )
        fpc_scores.append(fpc)
        print(f"  n={n_clusters}: FPC (Fuzzy Partition Coefficient) = {fpc:.4f}")
    
    # Choose optimal (max FPC)
    optimal_n = n_clusters_range[np.argmax(fpc_scores)]
    print(f"\nOptimal clusters: {optimal_n} (max FPC = {max(fpc_scores):.4f})")
    
    # Run FCM with optimal n
    print(f"\nRunning Fuzzy C-Means with {optimal_n} clusters...")
    cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(
        X.T, optimal_n, 2, error=0.005, maxiter=1000, init=None
    )
    
    # u is the fuzzy membership matrix (n_clusters x n_samples)
    # cntr is the cluster centers
    
    # Get hard cluster assignments (argmax of fuzzy memberships)
    fcm_labels = np.argmax(u, axis=0)
    
    # Get membership degrees
    max_memberships = np.max(u, axis=0)
    
    print(f"FCM Complete. Final objective function: {float(jm[-1]):.4f}")
    print(f"Fuzzy Partition Coefficient: {fpc:.4f}")
    
    # Analyze cluster characteristics
    print("\nFCM Cluster Sizes:")
    for i in range(optimal_n):
        n_patients = (fcm_labels == i).sum()
        avg_membership = max_memberships[fcm_labels == i].mean()
        print(f"  Cluster {i}: {n_patients} patients (avg membership: {avg_membership:.3f})")
    
    # Compare with VaDER hard clusters if available
    if hard_clusters is not None:
        from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
        ari = adjusted_rand_score(hard_clusters, fcm_labels)
        nmi = normalized_mutual_info_score(hard_clusters, fcm_labels)
        print(f"\nComparison with VaDER clusters:")
        print(f"  Adjusted Rand Index: {ari:.4f}")
        print(f"  Normalized Mutual Information: {nmi:.4f}")
    
    # Save results
    results_df = df.copy()
    results_df['fcm_cluster'] = fcm_labels
    for i in range(optimal_n):
        results_df[f'fcm_membership_c{i}'] = u[i, :]
    results_df['fcm_max_membership'] = max_memberships
    
    results_path = fcm_output / "fuzzy_cmeans_results.csv"
    results_df.to_csv(results_path, index=False)
    print(f"\nSaved results to {results_path}")
    
    # Visualizations
    create_fcm_visualizations(X, u, fcm_labels, cntr, hard_clusters, optimal_n)
    
    return results_df, u, cntr

def create_fcm_visualizations(X, u, fcm_labels, cntr, hard_clusters, n_clusters):
    """Create visualizations for FCM results"""
    print("\n📊 Creating FCM Visualizations...")
    
    # 1. PCA projection with fuzzy membership as opacity
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    
    # Hard clustering
    ax = axes[0]
    scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], c=fcm_labels, 
                        cmap='viridis', alpha=0.6, s=50, edgecolors='k')
    plt.colorbar(scatter, ax=ax, label='FCM Cluster')
    ax.set_title('Fuzzy C-Means: Hard Assignments', fontweight='bold')
    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)')
    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)')
    
    # Soft clustering (opacity by membership confidence)
    ax = axes[1]
    max_memberships = np.max(u, axis=0)
    scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], c=fcm_labels, 
                        cmap='viridis', alpha=max_memberships, s=50, edgecolors='k')
    plt.colorbar(scatter, ax=ax, label='FCM Cluster')
    ax.set_title('Fuzzy C-Means: Soft Assignments (opacity = confidence)', fontweight='bold')
    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)')
    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)')
    
    plt.tight_layout()
    plt.savefig(fcm_output / "fcm_pca_projection.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("  ✅ Saved: fcm_pca_projection.png")
    
    # 2. Interactive 3D ternary plot (for 3 clusters)
    if n_clusters == 3:
        create_ternary_plot(u)
    
    # 3. Membership distribution
    fig, axes = plt.subplots(1, n_clusters, figsize=(5*n_clusters, 4))
    if n_clusters == 1:
        axes = [axes]
    
    for i in range(n_clusters):
        ax = axes[i]
        memberships = u[i, :]
        ax.hist(memberships, bins=30, color=f'C{i}', alpha=0.7, edgecolor='black')
        ax.axvline(memberships.mean(), color='red', linestyle='--', 
                   label=f'Mean: {memberships.mean():.3f}')
        ax.set_xlabel('Membership Degree')
        ax.set_ylabel('Count')
        ax.set_title(f'Cluster {i} Membership Distribution')
        ax.legend()
    
    plt.tight_layout()
    plt.savefig(fcm_output / "fcm_membership_distributions.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("  ✅ Saved: fcm_membership_distributions.png")

def create_ternary_plot(u):
    """Create ternary plot for 3-cluster FCM"""
    print("  Creating ternary plot (3 clusters)...")
    
    # For a ternary plot, we need memberships to sum to 1 (they should already)
    fig = go.Figure()
    
    # Ternary scatter
    fig.add_trace(go.Scatterternary(
        a=u[0, :],
        b=u[1, :],
        c=u[2, :],
        mode='markers',
        marker=dict(
            size=5,
            color=np.argmax(u, axis=0),
            colorscale='Viridis',
            showscale=True,
            opacity=0.7
        ),
        text=[f'Patient {i}' for i in range(u.shape[1])],
        hovertemplate='<b>%{text}</b><br>C0: %{a:.3f}<br>C1: %{b:.3f}<br>C2: %{c:.3f}<extra></extra>'
    ))
    
    fig.update_layout(
        title='Fuzzy C-Means: Ternary Membership Plot',
        ternary=dict(
            aaxis=dict(title='Cluster 0'),
            baxis=dict(title='Cluster 1'),
            caxis=dict(title='Cluster 2')
        ),
        width=800,
        height=700
    )
    
    output_path = interactive_output / "fcm_ternary.html"
    fig.write_html(str(output_path))
    print(f"  ✅ Saved: {output_path}")

def main():
    results_df, u, cntr = perform_fuzzy_cmeans_clustering()
    
    print("\n" + "="*80)
    print("FUZZY C-MEANS CLUSTERING COMPLETE")
    print("="*80)
    print(f"\nOutputs saved to:")
    print(f"  - Static figures: {fcm_output}")
    print(f"  - Interactive plots: {interactive_output}")

if __name__ == "__main__":
    main()
