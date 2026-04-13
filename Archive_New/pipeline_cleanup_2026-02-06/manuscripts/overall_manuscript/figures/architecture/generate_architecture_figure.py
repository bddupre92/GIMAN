"""
Generate GIMAN architecture diagram for comprehensive manuscript.

This creates a visual representation of the complete GIMAN system architecture.
"""

from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle, Rectangle
import numpy as np

# Set publication-quality defaults
plt.rcParams.update({
    'font.size': 9,
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica'],
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight'
})

def create_giman_architecture():
    """Create comprehensive GIMAN architecture diagram."""
    
    fig, ax = plt.subplots(figsize=(18, 12))
    ax.set_xlim(0, 20)
    ax.set_ylim(0, 14)
    ax.axis('off')
    
    # Color scheme
    colors = {
        'input': '#E8F4F8',        # Light blue
        'gat': '#FFF4E6',          # Light orange
        'embedding': '#F3E5F5',    # Light purple
        'task': '#E8F5E9',         # Light green
        'xai': '#FFE6E6'           # Light red
    }
    
    box_style = "round,pad=0.15"
    
    # ===== TITLE =====
    ax.text(10, 13.5, 'GIMAN: Graph-Informed Multimodal Attention Network', 
            fontsize=18, fontweight='bold', ha='center', va='top')
    ax.text(10, 13, 'Complete System Architecture', 
            fontsize=14, ha='center', va='top', style='italic')
    
    # ===== INPUT LAYER =====
    y_input = 11.5
    
    # Patient graph
    box1 = FancyBboxPatch((1, y_input-0.8), 3.5, 1.5, 
                          boxstyle=box_style, 
                          facecolor=colors['input'], 
                          edgecolor='black', linewidth=2)
    ax.add_patch(box1)
    ax.text(2.75, y_input+0.3, 'Input: Patient Graph', 
            fontsize=11, fontweight='bold', ha='center', va='center')
    ax.text(2.75, y_input-0.05, 'G = (V, E, X)', 
            fontsize=10, ha='center', va='center', style='italic')
    ax.text(2.75, y_input-0.35, f'n = 536 (Phase 4)\nn = 194 (Phase 5)', 
            fontsize=8, ha='center', va='center')
    
    # Draw mini graph visualization
    mini_nodes_x = np.array([2.2, 2.5, 2.8, 3.1, 3.3, 2.4, 2.9]) + 0.1
    mini_nodes_y = np.array([y_input-0.5, y_input-0.3, y_input-0.5, 
                            y_input-0.3, y_input-0.5, y_input-0.65, y_input-0.65]) - 0.05
    
    # Draw edges
    edges = [(0,1), (1,2), (2,3), (3,4), (0,5), (2,6), (4,6), (1,5)]
    for i, j in edges:
        ax.plot([mini_nodes_x[i], mini_nodes_x[j]], 
               [mini_nodes_y[i], mini_nodes_y[j]], 
               'gray', linewidth=0.5, alpha=0.5)
    
    # Draw nodes
    for x, y in zip(mini_nodes_x, mini_nodes_y):
        circle = Circle((x, y), 0.06, facecolor='#2E86AB', 
                       edgecolor='black', linewidth=0.5)
        ax.add_patch(circle)
    
    # Feature matrix
    box2 = FancyBboxPatch((5, y_input-0.8), 3.5, 1.5, 
                          boxstyle=box_style, 
                          facecolor=colors['input'], 
                          edgecolor='black', linewidth=2)
    ax.add_patch(box2)
    ax.text(6.75, y_input+0.3, 'Node Features (X)', 
            fontsize=11, fontweight='bold', ha='center', va='center')
    ax.text(6.75, y_input-0.05, 'X ∈ ℝⁿˣᵈ', 
            fontsize=10, ha='center', va='center', style='italic')
    ax.text(6.75, y_input-0.35, 'd = 87 features\n(42 clinical, 28 imaging,\n5 genetic, 12 derived)', 
            fontsize=8, ha='center', va='center')
    
    # Arrow down
    arrow1 = FancyArrowPatch((5.5, y_input-0.9), (5.5, y_input-1.5),
                            arrowstyle='->', mutation_scale=25, 
                            linewidth=3, color='black')
    ax.add_patch(arrow1)
    
    # ===== GAT LAYERS =====
    y_gat_start = y_input - 2.5
    gat_heights = [1.6, 1.4, 1.2]
    gat_y_positions = [y_gat_start, y_gat_start-2, y_gat_start-4]
    gat_dims = ['87 → 128 (4×32)', '128 → 64 (4×16)', '64 → 32 (avg)']
    
    for layer_idx, (y_pos, height, dims) in enumerate(zip(gat_y_positions, gat_heights, gat_dims)):
        # GAT layer box
        box = FancyBboxPatch((1, y_pos-height/2), 7, height, 
                            boxstyle=box_style, 
                            facecolor=colors['gat'], 
                            edgecolor='black', linewidth=2)
        ax.add_patch(box)
        
        ax.text(4.5, y_pos+height/2-0.2, f'GAT Layer {layer_idx+1}', 
                fontsize=11, fontweight='bold', ha='center', va='center')
        ax.text(4.5, y_pos+0.05, dims, 
                fontsize=10, ha='center', va='center', style='italic', 
                color='#E63946', fontweight='bold')
        
        # Attention mechanism equation
        if layer_idx == 0:
            ax.text(4.5, y_pos-0.3, 'hᵢ⁽ˡ⁺¹⁾ = σ(Σⱼ αᵢⱼ⁽ˡ⁾ W⁽ˡ⁾ hⱼ⁽ˡ⁾)', 
                    fontsize=9, ha='center', va='center', 
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Multi-head attention visualization
        head_y = y_pos - 0.15
        for head in range(4):
            head_x = 2 + head * 1.5
            head_box = Rectangle((head_x, head_y-0.15), 1.2, 0.3, 
                                facecolor='#A23B72', alpha=0.6,
                                edgecolor='black', linewidth=0.5)
            ax.add_patch(head_box)
            ax.text(head_x+0.6, head_y, f'Head {head+1}', 
                    fontsize=7, ha='center', va='center', color='white', fontweight='bold')
        
        # Dropout indicator
        ax.text(7.5, y_pos, f'Dropout\n{0.3 if layer_idx < 2 else 0.2}', 
                fontsize=7, ha='center', va='center',
                bbox=dict(boxstyle='round', facecolor='#FFE6E6', alpha=0.7))
        
        # Arrow to next layer
        if layer_idx < 2:
            arrow = FancyArrowPatch((4.5, y_pos-height/2-0.1), 
                                   (4.5, gat_y_positions[layer_idx+1]+gat_heights[layer_idx+1]/2+0.1),
                                   arrowstyle='->', mutation_scale=20, 
                                   linewidth=2.5, color='black')
            ax.add_patch(arrow)
    
    # ===== LEARNED EMBEDDINGS =====
    y_embed = y_gat_start - 6.2
    
    box_embed = FancyBboxPatch((1.5, y_embed-0.6), 6, 1.2, 
                               boxstyle=box_style, 
                               facecolor=colors['embedding'], 
                               edgecolor='black', linewidth=2.5)
    ax.add_patch(box_embed)
    ax.text(4.5, y_embed+0.25, 'Learned Patient Embeddings', 
            fontsize=12, fontweight='bold', ha='center', va='center')
    ax.text(4.5, y_embed-0.1, 'H ∈ ℝⁿˣ³²', 
            fontsize=11, ha='center', va='center', style='italic', fontweight='bold')
    ax.text(4.5, y_embed-0.4, '32-dimensional latent representation', 
            fontsize=9, ha='center', va='center')
    
    # Split arrow to tasks and XAI
    arrow_split = FancyArrowPatch((4.5, y_embed-0.7), (4.5, y_embed-1.2),
                                 arrowstyle='->', mutation_scale=20, 
                                 linewidth=2.5, color='black')
    ax.add_patch(arrow_split)
    
    # Horizontal split
    y_split = y_embed - 1.3
    ax.plot([2, 4.5, 7], [y_split, y_split, y_split], 
            'k-', linewidth=2.5)
    
    # Arrows to tasks (left) and XAI (right)
    arrow_left = FancyArrowPatch((2, y_split), (2, y_split-0.6),
                                arrowstyle='->', mutation_scale=20, 
                                linewidth=2, color='black')
    arrow_right = FancyArrowPatch((7, y_split), (7, y_split-0.6),
                                 arrowstyle='->', mutation_scale=20, 
                                 linewidth=2, color='black')
    ax.add_patch(arrow_left)
    ax.add_patch(arrow_right)
    
    # ===== TASK-SPECIFIC HEADS =====
    y_tasks = y_split - 2.5
    
    # Phase 4 task
    box_task1 = FancyBboxPatch((0.3, y_tasks-0.5), 2.8, 1.5, 
                               boxstyle=box_style, 
                               facecolor=colors['task'], 
                               edgecolor='#2E86AB', linewidth=2)
    ax.add_patch(box_task1)
    ax.text(1.7, y_tasks+0.6, 'Phase 4 Task', 
            fontsize=10, fontweight='bold', ha='center', va='center')
    ax.text(1.7, y_tasks+0.25, 'Progression Subtypes', 
            fontsize=9, ha='center', va='center', style='italic')
    ax.text(1.7, y_tasks-0.05, 'MLP: 32→16→3', 
            fontsize=8, ha='center', va='center')
    ax.text(1.7, y_tasks-0.3, 'Softmax\nClassification', 
            fontsize=8, ha='center', va='center',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
    
    # Phase 5 task
    box_task2 = FancyBboxPatch((3.5, y_tasks-0.5), 2.8, 1.5, 
                               boxstyle=box_style, 
                               facecolor=colors['task'], 
                               edgecolor='#A23B72', linewidth=2)
    ax.add_patch(box_task2)
    ax.text(4.9, y_tasks+0.6, 'Phase 5 Task', 
            fontsize=10, fontweight='bold', ha='center', va='center')
    ax.text(4.9, y_tasks+0.25, 'Prodromal Conversion', 
            fontsize=9, ha='center', va='center', style='italic')
    ax.text(4.9, y_tasks-0.05, 'Cox Head: 32→1', 
            fontsize=8, ha='center', va='center')
    ax.text(4.9, y_tasks-0.3, 'Log-hazard\nSurvival', 
            fontsize=8, ha='center', va='center',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
    
    # ===== XAI METHODS =====
    y_xai = y_split - 1.5
    
    # XAI overview box
    box_xai = FancyBboxPatch((9, y_xai-3.2), 10, 4.5, 
                            boxstyle=box_style, 
                            facecolor=colors['xai'], 
                            edgecolor='#E63946', linewidth=3)
    ax.add_patch(box_xai)
    ax.text(14, y_xai+0.9, 'Phase 6: Explainability Framework', 
            fontsize=12, fontweight='bold', ha='center', va='center')
    ax.text(14, y_xai+0.5, 'Six Complementary XAI Methods', 
            fontsize=10, ha='center', va='center', style='italic')
    
    # Six XAI method boxes
    xai_methods = [
        ('1. Attention\nVisualization', 'αᵢⱼ weights\nPatient similarity'),
        ('2. GNNExplainer', 'Minimal subgraphs\nFeature masks'),
        ('3. IntegratedGradients', 'Gradient-based\nattribution'),
        ('4. GradientSHAP', 'Shapley values\nFeature importance'),
        ('5. Clustering', 'Embedding space\nSubgroup discovery'),
        ('6. Counterfactuals', 'What-if analysis\nMinimal changes')
    ]
    
    xai_positions = [
        (9.5, y_xai-0.3), (12.5, y_xai-0.3), (15.5, y_xai-0.3),
        (9.5, y_xai-1.8), (12.5, y_xai-1.8), (15.5, y_xai-1.8)
    ]
    
    for (title, desc), (x, y) in zip(xai_methods, xai_positions):
        box = FancyBboxPatch((x, y-0.5), 2.5, 1, 
                            boxstyle=box_style, 
                            facecolor='white', 
                            edgecolor='#E63946', linewidth=1.5)
        ax.add_patch(box)
        ax.text(x+1.25, y+0.2, title, 
                fontsize=8, fontweight='bold', ha='center', va='center')
        ax.text(x+1.25, y-0.2, desc, 
                fontsize=7, ha='center', va='center')
    
    # Cross-method validation
    box_validate = FancyBboxPatch((10, y_xai-3), 8, 0.8, 
                                 boxstyle=box_style, 
                                 facecolor='#FFF9C4', 
                                 edgecolor='black', linewidth=2)
    ax.add_patch(box_validate)
    ax.text(14, y_xai-2.6, '88-95% Cross-Method Consensus on Feature Importance', 
            fontsize=10, fontweight='bold', ha='center', va='center')
    
    # ===== OUTPUT LAYER =====
    y_output = 0.8
    
    # Outputs box
    box_output = FancyBboxPatch((0.5, y_output-0.5), 18, 1, 
                               boxstyle=box_style, 
                               facecolor='#E8F5E9', 
                               edgecolor='black', linewidth=2.5)
    ax.add_patch(box_output)
    ax.text(9.5, y_output+0.15, 'GIMAN Outputs', 
            fontsize=12, fontweight='bold', ha='center', va='center')
    ax.text(4, y_output-0.2, '• 3 Progression Subtypes\n• 38% Trial Size Reduction', 
            fontsize=8, ha='center', va='center')
    ax.text(9.5, y_output-0.2, '• Survival Risk: C-index 0.79\n• Prognostic Stratification', 
            fontsize=8, ha='center', va='center')
    ax.text(15, y_output-0.2, '• Patient-Specific Explanations\n• Interpretable Predictions', 
            fontsize=8, ha='center', va='center')
    
    # Legend
    legend_y = 0.2
    legend_elements = [
        ('Input Layer', colors['input']),
        ('GAT Layers', colors['gat']),
        ('Embeddings', colors['embedding']),
        ('Task Heads', colors['task']),
        ('XAI Methods', colors['xai'])
    ]
    
    for i, (label, color) in enumerate(legend_elements):
        x_pos = 3 + i * 3
        box = Rectangle((x_pos, legend_y-0.15), 0.4, 0.3, 
                       facecolor=color, edgecolor='black', linewidth=0.5)
        ax.add_patch(box)
        ax.text(x_pos+0.6, legend_y, label, 
                fontsize=7, ha='left', va='center')
    
    plt.tight_layout()
    
    # Ensure directory exists
    output_path = Path('giman_architecture.png')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    plt.savefig(str(output_path), dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✓ Saved: {output_path}")
    plt.close()

if __name__ == '__main__':
    print("\n" + "="*70)
    print("GENERATING GIMAN ARCHITECTURE DIAGRAM")
    print("="*70 + "\n")
    
    create_giman_architecture()
    
    print("\n" + "="*70)
    print("COMPLETE!")
    print("="*70)
