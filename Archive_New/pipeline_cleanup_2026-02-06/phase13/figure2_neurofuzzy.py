"""
Figure 2: Neuro-Fuzzy Enhancement
Shows the novel hybrid architecture: GAT → FCM → Fusion → Predictor
"""
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle, Rectangle
import numpy as np
from pathlib import Path

project_root = Path(__file__).resolve().parents[3]
output_dir = project_root / "visualizations/publication"
output_dir.mkdir(parents=True, exist_ok=True)

# Set style
plt.rcParams['figure.dpi'] = 300
plt.rcParams['font.size'] = 10
plt.rcParams['font.family'] = 'sans-serif'

def create_neuro_fuzzy_enhancement():
    fig, ax = plt.subplots(1, 1, figsize=(16, 5))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 5)
    ax.axis('off')
    
    # Colors
    color_gat = '#9b59b6'      # Purple (from Figure 1)
    color_fuzzy = '#f39c12'    # Orange/Gold for fuzzy
    color_fusion = '#16a085'   # Teal for fusion
    color_output = '#e74c3c'   # Red (from Figure 1)
    
    # Title
    ax.text(8, 4.6, 'Neuro-Fuzzy Enhancement Architecture', 
            ha='center', va='center', fontsize=14, fontweight='bold')
    
    # Define positions (centered vertically at y=2.5)
    y_center = 2.5
    x_gat = 2.5
    x_fcm = 5.5
    x_membership = 8.5
    x_fusion = 11.5
    x_output = 14.5
    
    # Box dimensions
    box_width = 1.6
    box_height = 1.4
    
    # === STAGE 1: GAT Embedding ===
    gat_box = FancyBboxPatch((x_gat-box_width/2, y_center-box_height/2), box_width, box_height,
                              boxstyle="round,pad=0.1", edgecolor=color_gat,
                              facecolor=color_gat, alpha=0.6, linewidth=2)
    ax.add_patch(gat_box)
    ax.text(x_gat, y_center+0.35, 'GAT', ha='center', va='center', fontweight='bold', fontsize=11)
    ax.text(x_gat, y_center, 'Embedding', ha='center', va='center', fontweight='bold', fontsize=11)
    ax.text(x_gat, y_center-0.4, r'$\vec{h}_i \in \mathbb{R}^{128}$', ha='center', va='center', fontsize=9, style='italic')
    
    # === STAGE 2: Fuzzy C-Means ===
    fcm_box = FancyBboxPatch((x_fcm-box_width/2, y_center-box_height/2), box_width, box_height,
                              boxstyle="round,pad=0.1", edgecolor=color_fuzzy,
                              facecolor=color_fuzzy, alpha=0.4, linewidth=2)
    ax.add_patch(fcm_box)
    ax.text(x_fcm, y_center+0.45, 'Fuzzy C-Means', ha='center', va='center', fontweight='bold', fontsize=11)
    ax.text(x_fcm, y_center+0.15, 'Clustering', ha='center', va='center', fontweight='bold', fontsize=11)
    
    # Draw 3 cluster centers
    cluster_y = y_center - 0.25
    cluster_spacing = 0.45
    cluster_colors = ['#e74c3c', '#3498db', '#2ecc71']  # Red, Blue, Green
    
    for i, color in enumerate(cluster_colors):
        cluster_x = x_fcm - cluster_spacing + i * cluster_spacing
        circle = Circle((cluster_x, cluster_y), 0.12, facecolor=color, edgecolor='white', linewidth=2, alpha=0.8)
        ax.add_patch(circle)
        ax.text(cluster_x, cluster_y, f'C{i+1}', ha='center', va='center', fontsize=8, fontweight='bold', color='white')
    
    # Arrow: GAT to FCM
    arrow = FancyArrowPatch((x_gat+box_width/2, y_center), (x_fcm-box_width/2, y_center),
                            arrowstyle='->', mutation_scale=20, linewidth=2.5, color='gray')
    ax.add_patch(arrow)
    
    # === STAGE 3: Membership Vector ===
    membership_box = FancyBboxPatch((x_membership-box_width/2, y_center-box_height/2), box_width, box_height,
                                     boxstyle="round,pad=0.05", edgecolor=color_fuzzy,
                                     facecolor='white', linewidth=2)
    ax.add_patch(membership_box)
    ax.text(x_membership, y_center+0.5, 'Fuzzy', ha='center', va='center', fontweight='bold', fontsize=11)
    ax.text(x_membership, y_center+0.2, 'Membership', ha='center', va='center', fontweight='bold', fontsize=11)
    
    # Draw membership bars
    memberships = [0.7, 0.2, 0.1]
    bar_width = 0.25
    bar_spacing = 0.35
    bar_start_y = y_center + 0.05
    
    for i, (membership, color) in enumerate(zip(memberships, cluster_colors)):
        bar_x = x_membership - 0.35 + i * bar_spacing
        bar_height = membership * 0.7
        bar = Rectangle((bar_x, bar_start_y - bar_height), bar_width, bar_height,
                        facecolor=color, edgecolor='black', linewidth=1, alpha=0.8)
        ax.add_patch(bar)
        ax.text(bar_x + bar_width/2, bar_start_y - bar_height - 0.12, 
                f'{membership:.1f}', ha='center', va='top', fontsize=8, fontweight='bold')
    
    ax.text(x_membership, y_center-0.52, r'$\vec{u}_i = [0.7, 0.2, 0.1]$', 
            ha='center', va='center', fontsize=9, style='italic')
    
    # Arrow: FCM to Membership
    arrow = FancyArrowPatch((x_fcm+box_width/2, y_center), (x_membership-box_width/2, y_center),
                            arrowstyle='->', mutation_scale=20, linewidth=2.5, color=color_fuzzy)
    ax.add_patch(arrow)
    
    # === STAGE 4: Feature Fusion ===
    fusion_box = FancyBboxPatch((x_fusion-box_width/2, y_center-box_height/2), box_width, box_height,
                                 boxstyle="round,pad=0.1", edgecolor=color_fusion,
                                 facecolor=color_fusion, alpha=0.5, linewidth=2)
    ax.add_patch(fusion_box)
    ax.text(x_fusion, y_center+0.45, 'Feature', ha='center', va='center', fontweight='bold', fontsize=11)
    ax.text(x_fusion, y_center+0.15, 'Fusion', ha='center', va='center', fontweight='bold', fontsize=11)
    
    # Concatenation symbol
    ax.text(x_fusion, y_center-0.2, '⊕', ha='center', va='center', fontsize=22, fontweight='bold')
    ax.text(x_fusion, y_center-0.55, r'$[\vec{h}_i \| \vec{u}_i]$', ha='center', va='center', fontsize=9, style='italic')
    
    # Arrow: GAT to Fusion (bypass connection - routed via top)
    # Path: up from GAT, across, down into Fusion
    bypass_y = y_center + box_height/2 + 0.8
    
    # Segment 1: Up from GAT
    ax.plot([x_gat, x_gat], [y_center+box_height/2, bypass_y], 
            color=color_gat, linestyle='dashed', linewidth=2)
    
    # Segment 2: Across to above Fusion
    ax.plot([x_gat, x_fusion], [bypass_y, bypass_y], 
            color=color_gat, linestyle='dashed', linewidth=2)
    
    # Segment 3: Down into Fusion (with arrow)
    arrow_bypass = FancyArrowPatch((x_fusion, bypass_y), (x_fusion, y_center+box_height/2),
                                   arrowstyle='->', mutation_scale=15, linewidth=2, 
                                   color=color_gat, linestyle='dashed')
    ax.add_patch(arrow_bypass)
    
    # Label on top path
    ax.text((x_gat+x_fusion)/2, bypass_y+0.15, 'Bypass (Original Embedding)', 
            ha='center', va='bottom', fontsize=8, style='italic', color=color_gat)

    
    # Arrow: Membership to Fusion
    arrow = FancyArrowPatch((x_membership+box_width/2, y_center), (x_fusion-box_width/2, y_center),
                            arrowstyle='->', mutation_scale=20, linewidth=2.5, color=color_fuzzy)
    ax.add_patch(arrow)
    
    # === STAGE 5: Predictor ===
    output_box = FancyBboxPatch((x_output-box_width/2, y_center-box_height/2), box_width, box_height,
                                 boxstyle="round,pad=0.05", edgecolor=color_output,
                                 facecolor=color_output, alpha=0.5, linewidth=2)
    ax.add_patch(output_box)
    ax.text(x_output, y_center+0.3, 'Predictor', ha='center', va='center', fontweight='bold', fontsize=11)
    ax.text(x_output, y_center-0.05, 'Head', ha='center', va='center', fontweight='bold', fontsize=11)
    ax.text(x_output, y_center-0.38, '(SAA/Survival)', ha='center', va='center', fontsize=9, style='italic')
    
    # Arrow: Fusion to Output
    arrow = FancyArrowPatch((x_fusion+box_width/2, y_center), (x_output-box_width/2, y_center),
                            arrowstyle='->', mutation_scale=20, linewidth=2.5, color='gray')
    ax.add_patch(arrow)
    
    # Add simple legend
    legend_elements = [
        mpatches.Patch(facecolor=color_gat, edgecolor=color_gat, label='Neural (GAT)', alpha=0.6),
        mpatches.Patch(facecolor=color_fuzzy, edgecolor=color_fuzzy, label='Fuzzy (FCM)', alpha=0.4),
        mpatches.Patch(facecolor=color_fusion, edgecolor=color_fusion, label='Fusion', alpha=0.5),
        mpatches.Patch(facecolor=color_output, edgecolor=color_output, label='Output', alpha=0.5)
    ]
    ax.legend(handles=legend_elements, loc='lower center', ncol=4, frameon=False, fontsize=9)
    
    plt.tight_layout()
    output_path = output_dir / "Figure2_Neuro_Fuzzy_Enhancement.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"✅ Saved Figure 2: {output_path}")

if __name__ == "__main__":
    print("Generating Figure 2: Neuro-Fuzzy Enhancement...")
    create_neuro_fuzzy_enhancement()
