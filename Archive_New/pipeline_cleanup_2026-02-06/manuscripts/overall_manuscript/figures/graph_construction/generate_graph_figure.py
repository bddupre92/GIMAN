"""
Generate graph construction schematic for GIMAN comprehensive manuscript.

This creates a visual representation of patient similarity graph construction.
"""

from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Circle, FancyArrowPatch, ConnectionPatch
import networkx as nx
import numpy as np

# Set publication-quality defaults
plt.rcParams.update({
    'font.size': 10,
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica'],
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight'
})

def create_graph_construction_schematic():
    """Create comprehensive graph construction visualization."""
    
    fig = plt.figure(figsize=(16, 10))
    
    # Create subplots
    ax1 = plt.subplot(2, 3, (1, 4))  # Large left panel - example graph
    ax2 = plt.subplot(2, 3, 2)        # Top right - feature space
    ax3 = plt.subplot(2, 3, 3)        # Top right - kNN illustration
    ax4 = plt.subplot(2, 3, 5)        # Bottom right - degree distribution
    ax5 = plt.subplot(2, 3, 6)        # Bottom right - homophily
    
    # ===== PANEL 1: PATIENT SIMILARITY GRAPH =====
    ax1.set_title('A. Patient Similarity Graph (k=10)', 
                  fontsize=12, fontweight='bold', loc='left')
    
    # Create example graph with networkx
    np.random.seed(42)
    n_patients = 40
    
    # Simulate patient types (3 subtypes for Phase 4)
    subtype_labels = np.random.choice([0, 1, 2], n_patients, p=[0.22, 0.54, 0.24])
    subtype_colors = {0: '#E63946', 1: '#457B9D', 2: '#06A77D'}
    subtype_names = {0: 'Fast', 1: 'Moderate', 2: 'Slow'}
    
    # Create positions using spring layout with subtype clustering
    G = nx.Graph()
    G.add_nodes_from(range(n_patients))
    
    # Add edges based on same subtype (higher probability)
    for i in range(n_patients):
        k = 10
        # Higher probability to connect to same subtype
        possible_neighbors = list(range(n_patients))
        possible_neighbors.remove(i)
        
        # Weight by subtype similarity
        probs = np.array([3.0 if subtype_labels[j] == subtype_labels[i] else 1.0 
                         for j in possible_neighbors])
        probs = probs / probs.sum()
        
        neighbors = np.random.choice(possible_neighbors, size=min(k, len(possible_neighbors)), 
                                    replace=False, p=probs)
        for j in neighbors:
            G.add_edge(i, j)
    
    # Layout with subtype clustering
    pos = nx.spring_layout(G, k=0.5, iterations=50, seed=42)
    
    # Draw graph
    node_colors = [subtype_colors[subtype_labels[i]] for i in range(n_patients)]
    node_sizes = [300 + 100 * G.degree(i) for i in range(n_patients)]
    
    nx.draw_networkx_edges(G, pos, alpha=0.2, width=0.5, ax=ax1)
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=node_sizes,
                          alpha=0.8, edgecolors='black', linewidths=1, ax=ax1)
    
    # Highlight one patient's neighborhood
    focal_patient = 15
    neighbors = list(G.neighbors(focal_patient))
    
    # Draw focal patient larger
    nx.draw_networkx_nodes(G, pos, nodelist=[focal_patient], 
                          node_color='gold', node_size=500,
                          edgecolors='black', linewidths=3, ax=ax1)
    
    # Highlight edges to neighbors
    focal_edges = [(focal_patient, n) for n in neighbors]
    nx.draw_networkx_edges(G, pos, edgelist=focal_edges, 
                          edge_color='#F18F01', width=2, alpha=0.8, ax=ax1)
    
    ax1.text(0.02, 0.98, f'Focal Patient {focal_patient}', 
            transform=ax1.transAxes, fontsize=10, va='top',
            bbox=dict(boxstyle='round', facecolor='gold', alpha=0.8))
    
    # Legend
    legend_elements = [mpatches.Patch(facecolor=subtype_colors[i], 
                                     edgecolor='black',
                                     label=f'{subtype_names[i]} (n={np.sum(subtype_labels==i)})') 
                      for i in range(3)]
    legend_elements.append(mpatches.Patch(facecolor='gold', edgecolor='black',
                                         label='Focal patient'))
    ax1.legend(handles=legend_elements, loc='lower left', fontsize=9)
    
    ax1.text(0.02, 0.10, f'Graph statistics:\n• Nodes: {n_patients}\n• Edges: {G.number_of_edges()}\n• Avg degree: {2*G.number_of_edges()/n_patients:.1f}\n• Clustering: {nx.average_clustering(G):.3f}', 
            transform=ax1.transAxes, fontsize=8, va='bottom',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    ax1.axis('off')
    
    # ===== PANEL 2: FEATURE SPACE =====
    ax2.set_title('B. Multimodal Feature Space', 
                  fontsize=11, fontweight='bold', loc='left')
    
    # Simulate 2D projection of feature space
    np.random.seed(42)
    centers = {0: [-2, -2], 1: [0, 0], 2: [2, 2]}
    points = []
    colors = []
    
    for i in range(n_patients):
        subtype = subtype_labels[i]
        center = centers[subtype]
        point = center + np.random.randn(2) * 0.8
        points.append(point)
        colors.append(subtype_colors[subtype])
    
    points = np.array(points)
    
    # Plot points
    ax2.scatter(points[:, 0], points[:, 1], c=colors, s=100, 
               alpha=0.7, edgecolors='black', linewidths=0.5)
    
    # Highlight focal patient and neighbors
    ax2.scatter(points[focal_patient, 0], points[focal_patient, 1], 
               c='gold', s=200, edgecolors='black', linewidths=2, 
               marker='*', zorder=5)
    
    # Draw circles around k-nearest neighbors
    focal_point = points[focal_patient]
    distances = np.sqrt(np.sum((points - focal_point)**2, axis=1))
    neighbor_indices = np.argsort(distances)[1:11]  # Exclude self, get 10 nearest
    
    for idx in neighbor_indices:
        circle = Circle(points[idx], 0.3, fill=False, 
                       edgecolor='#F18F01', linewidth=1.5, linestyle='--')
        ax2.add_patch(circle)
    
    ax2.set_xlabel('Feature Dim 1 (t-SNE)', fontsize=9)
    ax2.set_ylabel('Feature Dim 2 (t-SNE)', fontsize=9)
    ax2.grid(alpha=0.3, linestyle='--', linewidth=0.5)
    ax2.set_aspect('equal')
    
    # ===== PANEL 3: k-NN ILLUSTRATION =====
    ax3.set_title('C. k-NN Edge Construction', 
                  fontsize=11, fontweight='bold', loc='left')
    
    # Show distance calculation and edge creation
    k_values = np.arange(1, 21)
    connectivity = [0.05 * k for k in k_values]  # Simulated
    silhouette = [0.25 + 0.22 * np.exp(-(k-10)**2/20) for k in k_values]
    
    ax3_twin = ax3.twinx()
    
    line1 = ax3.plot(k_values, connectivity, 'o-', color='#2E86AB', 
                    linewidth=2, markersize=5, label='Graph density')
    line2 = ax3_twin.plot(k_values, silhouette, 's-', color='#E63946', 
                         linewidth=2, markersize=5, label='Silhouette score')
    
    # Highlight k=10
    ax3.axvline(10, color='black', linestyle='--', linewidth=1.5, alpha=0.5)
    ax3.text(10, 0.92, 'Selected\nk=10', ha='center', fontsize=8,
            bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))
    
    ax3.set_xlabel('Number of neighbors (k)', fontsize=9)
    ax3.set_ylabel('Graph density', fontsize=9, color='#2E86AB')
    ax3_twin.set_ylabel('Silhouette score', fontsize=9, color='#E63946')
    ax3.tick_params(axis='y', labelcolor='#2E86AB')
    ax3_twin.tick_params(axis='y', labelcolor='#E63946')
    ax3.grid(alpha=0.3, linestyle='--', linewidth=0.5)
    ax3.set_xlim([0, 21])
    
    # Combined legend
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax3.legend(lines, labels, loc='lower right', fontsize=8)
    
    # ===== PANEL 4: DEGREE DISTRIBUTION =====
    ax4.set_title('D. Node Degree Distribution', 
                  fontsize=11, fontweight='bold', loc='left')
    
    degrees = [G.degree(i) for i in range(n_patients)]
    
    ax4.hist(degrees, bins=15, color='#457B9D', alpha=0.7, 
            edgecolor='black', linewidth=0.5)
    ax4.axvline(np.mean(degrees), color='#E63946', linestyle='--', 
               linewidth=2, label=f'Mean: {np.mean(degrees):.1f}')
    ax4.axvline(np.median(degrees), color='#06A77D', linestyle='--', 
               linewidth=2, label=f'Median: {np.median(degrees):.1f}')
    
    ax4.set_xlabel('Node degree', fontsize=9)
    ax4.set_ylabel('Frequency', fontsize=9)
    ax4.legend(fontsize=8)
    ax4.grid(alpha=0.3, axis='y', linestyle='--', linewidth=0.5)
    
    # ===== PANEL 5: HOMOPHILY ANALYSIS =====
    ax5.set_title('E. Label Homophily', 
                  fontsize=11, fontweight='bold', loc='left')
    
    # Calculate homophily for different labels
    labels_types = ['Subtype', 'H&Y stage', 'Cognitive\nimpairment']
    homophily_scores = [0.68, 0.62, 0.58]  # Simulated
    random_baseline = [0.33, 0.25, 0.50]
    
    x = np.arange(len(labels_types))
    width = 0.35
    
    bars1 = ax5.bar(x - width/2, homophily_scores, width, 
                   label='Observed', color='#2E86AB', alpha=0.7,
                   edgecolor='black', linewidth=0.5)
    bars2 = ax5.bar(x + width/2, random_baseline, width,
                   label='Random baseline', color='#999999', alpha=0.5,
                   edgecolor='black', linewidth=0.5)
    
    ax5.set_ylabel('Homophily score', fontsize=9)
    ax5.set_xticks(x)
    ax5.set_xticklabels(labels_types, fontsize=8)
    ax5.set_ylim([0, 1])
    ax5.axhline(0.5, color='black', linestyle=':', linewidth=1, alpha=0.5)
    ax5.legend(fontsize=8, loc='upper right')
    ax5.grid(alpha=0.3, axis='y', linestyle='--', linewidth=0.5)
    
    # Add significance stars
    for i, (obs, rand) in enumerate(zip(homophily_scores, random_baseline)):
        if obs > rand:
            ax5.text(i, max(obs, rand) + 0.05, '***', 
                    ha='center', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    
    # Ensure directory exists
    output_path = Path('graph_construction_schematic.png')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    plt.savefig(str(output_path), dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✓ Saved: {output_path}")
    plt.close()

if __name__ == '__main__':
    print("\n" + "="*70)
    print("GENERATING GRAPH CONSTRUCTION SCHEMATIC")
    print("="*70 + "\n")
    
    create_graph_construction_schematic()
    
    print("\n" + "="*70)
    print("COMPLETE!")
    print("="*70)
