"""
Figure 1: High-Level GIMAN Framework
Shows the end-to-end pipeline with multimodal inputs, encoders, graph construction, and outputs
"""
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle
import numpy as np
from pathlib import Path

project_root = Path(__file__).resolve().parents[3]
output_dir = project_root / "visualizations/publication"
output_dir.mkdir(parents=True, exist_ok=True)

# Set style
plt.rcParams['figure.dpi'] = 300
plt.rcParams['font.size'] = 10
plt.rcParams['font.family'] = 'sans-serif'

def create_giman_architecture():
    fig, ax = plt.subplots(1, 1, figsize=(16, 6))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 6)
    ax.axis('off')
    
    # Colors
    color_imaging = '#3498db'  # Blue
    color_genetics = '#27ae60'  # Green
    color_clinical = '#e67e22'  # Orange
    color_graph = '#9b59b6'    # Purple
    color_output = '#e74c3c'   # Red
    
    # Title
    ax.text(8, 5.6, 'GIMAN Architecture: End-to-End Pipeline', 
            ha='center', va='center', fontsize=14, fontweight='bold')
    
    # Define vertical positions for perfect symmetry (centered at y=3)
    y_top = 4.2      # Top modality
    y_center = 3.0   # Center modality (middle of canvas)
    y_bottom = 1.8   # Bottom modality
    
    # Define horizontal positions for each stage
    x_inputs = 1.5
    x_encoders = 4.0
    x_concat = 6.5
    x_graph = 9.0
    x_gnn = 11.5
    x_outputs = 14.5
    
    # Box dimensions
    box_width = 1.4
    box_height = 0.9
    
    # === STAGE 1: INPUTS (Left) ===
    # Imaging input (top)
    imaging_box = FancyBboxPatch((x_inputs-box_width/2, y_top-box_height/2), box_width, box_height,
                                  boxstyle="round,pad=0.05", 
                                  edgecolor=color_imaging, facecolor=color_imaging, alpha=0.3, linewidth=2)
    ax.add_patch(imaging_box)
    ax.text(x_inputs, y_top, '3D Brain\nScans', ha='center', va='center', fontweight='bold', color=color_imaging)
    
    # Genetics input (center)
    genetics_box = FancyBboxPatch((x_inputs-box_width/2, y_center-box_height/2), box_width, box_height,
                                   boxstyle="round,pad=0.05",
                                   edgecolor=color_genetics, facecolor=color_genetics, alpha=0.3, linewidth=2)
    ax.add_patch(genetics_box)
    ax.text(x_inputs, y_center, 'DNA\nSequence', ha='center', va='center', fontweight='bold', color=color_genetics)
    
    # Clinical input (bottom)
    clinical_box = FancyBboxPatch((x_inputs-box_width/2, y_bottom-box_height/2), box_width, box_height,
                                   boxstyle="round,pad=0.05",
                                   edgecolor=color_clinical, facecolor=color_clinical, alpha=0.3, linewidth=2)
    ax.add_patch(clinical_box)
    ax.text(x_inputs, y_bottom, 'Clinical\nTimeseries', ha='center', va='center', fontweight='bold', color=color_clinical)
    
    # === STAGE 2: ENCODERS ===
    # 3D-CNN-GRU (top)
    cnn_box = FancyBboxPatch((x_encoders-box_width/2, y_top-box_height/2), box_width, box_height,
                              boxstyle="round,pad=0.05", edgecolor=color_imaging, 
                              facecolor=color_imaging, alpha=0.5, linewidth=2)
    ax.add_patch(cnn_box)
    ax.text(x_encoders, y_top, '3D-CNN-GRU\nEncoder', ha='center', va='center', fontweight='bold', fontsize=9)
    
    # Genomic Transformer (center)
    transformer_box = FancyBboxPatch((x_encoders-box_width/2, y_center-box_height/2), box_width, box_height,
                                      boxstyle="round,pad=0.05", edgecolor=color_genetics,
                                      facecolor=color_genetics, alpha=0.5, linewidth=2)
    ax.add_patch(transformer_box)
    ax.text(x_encoders, y_center, 'Genomic\nTransformer', ha='center', va='center', fontweight='bold', fontsize=9)
    
    # Clinical GRU (bottom)
    gru_box = FancyBboxPatch((x_encoders-box_width/2, y_bottom-box_height/2), box_width, box_height,
                              boxstyle="round,pad=0.05", edgecolor=color_clinical,
                              facecolor=color_clinical, alpha=0.5, linewidth=2)
    ax.add_patch(gru_box)
    ax.text(x_encoders, y_bottom, 'Clinical\nGRU', ha='center', va='center', fontweight='bold', fontsize=9)
    
    # Horizontal arrows: Inputs to Encoders (perfectly straight)
    for y_pos, color in [(y_top, color_imaging), (y_center, color_genetics), (y_bottom, color_clinical)]:
        arrow = FancyArrowPatch((x_inputs+box_width/2, y_pos), (x_encoders-box_width/2, y_pos),
                                arrowstyle='->', mutation_scale=20, linewidth=2, color=color)
        ax.add_patch(arrow)
    
    # === STAGE 3: CONCATENATION ===
    # Concatenation circle (centered vertically)
    concat_circle = Circle((x_concat, y_center), 0.35, edgecolor='gray', facecolor='lightgray', linewidth=2)
    ax.add_patch(concat_circle)
    ax.text(x_concat, y_center, '⊕', ha='center', va='center', fontsize=16, fontweight='bold')
    ax.text(x_concat, y_center-0.6, 'Concat', ha='center', va='top', fontsize=9)
    
    # Arrows from encoders to concat (converging)
    for y_start, color in [(y_top, color_imaging), (y_center, color_genetics), (y_bottom, color_clinical)]:
        arrow = FancyArrowPatch((x_encoders+box_width/2, y_start), (x_concat-0.35, y_center),
                                arrowstyle='->', mutation_scale=15, linewidth=1.5, color=color)
        ax.add_patch(arrow)
    
    # === STAGE 4: GRAPH CONSTRUCTION ===
    graph_box = FancyBboxPatch((x_graph-0.9, y_center-1.3), 1.8, 2.6,
                                boxstyle="round,pad=0.1", edgecolor=color_graph,
                                facecolor=color_graph, alpha=0.3, linewidth=2)
    ax.add_patch(graph_box)
    ax.text(x_graph, y_center+1.1, 'Patient Similarity', ha='center', va='center', fontweight='bold', fontsize=10)
    ax.text(x_graph, y_center+0.7, 'Graph (k-NN)', ha='center', va='center', fontsize=9, style='italic')
    
    # Draw mini graph visualization (centered)
    node_positions = [(x_graph, y_center), (x_graph+0.35, y_center+0.35), (x_graph+0.35, y_center-0.35),
                      (x_graph-0.35, y_center+0.35), (x_graph-0.35, y_center-0.35)]
    for pos in node_positions:
        node = Circle(pos, 0.08, color=color_graph, edgecolor='white', linewidth=1.5, zorder=10)
        ax.add_patch(node)
    # Edges
    edge_pairs = [(0,1), (0,2), (0,3), (0,4), (1,2), (3,4)]
    for i, j in edge_pairs:
        ax.plot([node_positions[i][0], node_positions[j][0]], 
                [node_positions[i][1], node_positions[j][1]], 
                'k-', alpha=0.3, linewidth=1.5, zorder=5)
    
    # Arrow from concat to graph (straight)
    arrow = FancyArrowPatch((x_concat+0.35, y_center), (x_graph-0.9, y_center),
                            arrowstyle='->', mutation_scale=20, linewidth=2, color='gray')
    ax.add_patch(arrow)
    
    # === STAGE 5: GNN LAYER ===
    gnn_box = FancyBboxPatch((x_gnn-0.9, y_center-1.3), 1.8, 2.6,
                              boxstyle="round,pad=0.1", edgecolor=color_graph,
                              facecolor=color_graph, alpha=0.6, linewidth=2)
    ax.add_patch(gnn_box)
    ax.text(x_gnn, y_center+0.9, 'Graph Attention', ha='center', va='center', fontweight='bold', fontsize=10)
    ax.text(x_gnn, y_center+0.5, 'Network (GAT)', ha='center', va='center', fontweight='bold', fontsize=10)
    ax.text(x_gnn, y_center, 'Multi-Head\nAttention', ha='center', va='center', fontsize=9, style='italic')
    
    # Arrow from graph to GNN (straight)
    arrow = FancyArrowPatch((x_graph+0.9, y_center), (x_gnn-0.9, y_center),
                            arrowstyle='->', mutation_scale=20, linewidth=2, color=color_graph)
    ax.add_patch(arrow)
    
    # === STAGE 6: OUTPUTS (Right) ===
    # Prognosis output (top)
    prognosis_box = FancyBboxPatch((x_outputs-box_width/2, y_top-box_height/2), box_width, box_height,
                                    boxstyle="round,pad=0.05", edgecolor=color_output,
                                    facecolor=color_output, alpha=0.5, linewidth=2)
    ax.add_patch(prognosis_box)
    ax.text(x_outputs, y_top, 'Prognosis\n(DeepSurv)', ha='center', va='center', fontweight='bold', fontsize=9)
    
    # Classification output (bottom)
    classification_box = FancyBboxPatch((x_outputs-box_width/2, y_bottom-box_height/2), box_width, box_height,
                                         boxstyle="round,pad=0.05", edgecolor=color_output,
                                         facecolor=color_output, alpha=0.5, linewidth=2)
    ax.add_patch(classification_box)
    ax.text(x_outputs, y_bottom, 'Classification\n(SAA Predict)', ha='center', va='center', fontweight='bold', fontsize=9)
    
    # Arrows from GNN to outputs (diverging symmetrically)
    arrow1 = FancyArrowPatch((x_gnn+0.9, y_center), (x_outputs-box_width/2, y_top),
                             arrowstyle='->', mutation_scale=20, linewidth=2, color=color_output)
    ax.add_patch(arrow1)
    
    arrow2 = FancyArrowPatch((x_gnn+0.9, y_center), (x_outputs-box_width/2, y_bottom),
                             arrowstyle='->', mutation_scale=20, linewidth=2, color=color_output)
    ax.add_patch(arrow2)
    
    # Legend
    legend_elements = [
        mpatches.Patch(facecolor=color_imaging, edgecolor=color_imaging, label='Imaging', alpha=0.5),
        mpatches.Patch(facecolor=color_genetics, edgecolor=color_genetics, label='Genetics', alpha=0.5),
        mpatches.Patch(facecolor=color_clinical, edgecolor=color_clinical, label='Clinical', alpha=0.5),
        mpatches.Patch(facecolor=color_graph, edgecolor=color_graph, label='Graph', alpha=0.5),
        mpatches.Patch(facecolor=color_output, edgecolor=color_output, label='Output', alpha=0.5)
    ]
    ax.legend(handles=legend_elements, loc='lower center', ncol=5, frameon=False, fontsize=9)
    
    plt.tight_layout()
    output_path = output_dir / "Figure1_GIMAN_Architecture.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"✅ Saved Figure 1: {output_path}")

if __name__ == "__main__":
    print("Generating Figure 1: GIMAN Architecture...")
    create_giman_architecture()
