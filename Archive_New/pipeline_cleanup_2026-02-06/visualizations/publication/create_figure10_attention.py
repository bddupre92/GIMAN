"""
Figure 10: Graph Attention Visualization
Network graph showing attention weights between target patient and neighbors.
Fixed to show only patient numbers in nodes (no newlines or extra text).
"""

import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import seaborn as sns

# Set style
sns.set_style("white")
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 11

# Create figure
fig, ax = plt.subplots(figsize=(14, 10))

# Define target patient and neighbors with attention weights
target_patient = '3055'
neighbors = {
    '4012': 0.32,  # Similar profile but slow progressor
    '3287': 0.24,  # Similar rapid progression
    '5104': 0.19,  # Mixed phenotype
    '2891': 0.12,  # Lower similarity
    '1563': 0.08,  # Lower similarity
    '4521': 0.05,  # Weak connection
}

# Create graph
G = nx.Graph()

# Add target node
G.add_node(target_patient)

# Add neighbor nodes and edges
for neighbor, weight in neighbors.items():
    G.add_node(neighbor)
    G.add_edge(target_patient, neighbor, weight=weight)

# Layout - circular with target in center
pos = {}
pos[target_patient] = (0, 0)  # Center

# Arrange neighbors in circle
n_neighbors = len(neighbors)
radius = 3
angles = np.linspace(0, 2*np.pi, n_neighbors, endpoint=False)

for i, neighbor in enumerate(neighbors.keys()):
    pos[neighbor] = (radius * np.cos(angles[i]), radius * np.sin(angles[i]))

# Draw edges with thickness proportional to attention weight
for (u, v, data) in G.edges(data=True):
    weight = data['weight']
    ax.plot([pos[u][0], pos[v][0]], [pos[u][1], pos[v][1]],
            color='#3498db', linewidth=weight*15, alpha=0.6, zorder=1)
    
    # Add weight labels on edges
    mid_x = (pos[u][0] + pos[v][0]) / 2
    mid_y = (pos[u][1] + pos[v][1]) / 2
    ax.text(mid_x, mid_y, f'α={weight:.2f}', 
            fontsize=9, ha='center', va='center',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                     edgecolor='#3498db', alpha=0.9))

# Draw nodes
# Target node (red, larger)
target_circle = plt.Circle(pos[target_patient], 0.6, color='#e74c3c', 
                          ec='white', linewidth=3, zorder=3)
ax.add_patch(target_circle)
ax.text(pos[target_patient][0], pos[target_patient][1], target_patient,
        ha='center', va='center', fontsize=14, fontweight='bold', 
        color='white', zorder=4)

# Neighbor nodes (colored by attention weight)
for neighbor, weight in neighbors.items():
    # Color intensity based on attention weight
    if weight > 0.25:
        color = '#c0392b'  # Dark red (high attention)
    elif weight > 0.15:
        color = '#e67e22'  # Orange (medium attention)
    else:
        color = '#95a5a6'  # Gray (low attention)
    
    circle = plt.Circle(pos[neighbor], 0.5, color=color,
                       ec='white', linewidth=2.5, zorder=2)
    ax.add_patch(circle)
    
    # JUST show patient number - no newlines or extra text
    ax.text(pos[neighbor][0], pos[neighbor][1], neighbor,
            ha='center', va='center', fontsize=12, fontweight='bold',
            color='white', zorder=4)

# Set axis limits and remove axes
margin = 0.5
ax.set_xlim(-radius-margin, radius+margin)
ax.set_ylim(-radius-margin, radius+margin)
ax.set_aspect('equal')
ax.axis('off')

# Add title and legend
ax.set_title('Graph Attention Visualization: Patient 3055 and Influential Neighbors',
             fontsize=15, fontweight='bold', pad=20)

# Add legend
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
legend_elements = [
    Patch(facecolor='#e74c3c', edgecolor='white', label='Target Patient (3055)'),
    Patch(facecolor='#c0392b', edgecolor='white', label='High Attention (α>0.25)'),
    Patch(facecolor='#e67e22', edgecolor='white', label='Medium Attention (0.15<α<0.25)'),
    Patch(facecolor='#95a5a6', edgecolor='white', label='Low Attention (α<0.15)'),
    Line2D([0], [0], color='#3498db', linewidth=3, label='Attention Edge (thickness ∝ α)')
]
ax.legend(handles=legend_elements, loc='upper right', fontsize=10, 
          framealpha=0.95, edgecolor='black')

# Add note
ax.text(0, -radius-0.8, 
        'Edge thickness and neighbor node color represent attention weight (α).\nPatient 4012 has highest attention (0.32) despite being slow progressor—provides protective comparison.',
        ha='center', fontsize=10, style='italic', color='#555',
        bbox=dict(boxstyle='round,pad=0.6', facecolor='#f9f9f9', 
                 edgecolor='#888', linewidth=1.5, alpha=0.9))

plt.tight_layout()

# Save
output_path = '/Users/blair.dupre/Library/CloudStorage/GoogleDrive-dupre.blair92@gmail.com/My Drive/CSCI FALL 2025/visualizations/publication/Figure10_Attention_Heatmap.png'
plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
print(f"Saved attention heatmap with clean labels to: {output_path}")

plt.show()
