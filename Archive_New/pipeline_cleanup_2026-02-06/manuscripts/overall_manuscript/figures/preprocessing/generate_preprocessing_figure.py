"""
Generate preprocessing flowchart figure for GIMAN comprehensive manuscript.

This creates a visual flowchart showing PPMI data flow through QC and feature engineering.
"""

from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
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

def create_preprocessing_flowchart():
    """Create comprehensive preprocessing flowchart."""
    
    fig, ax = plt.subplots(figsize=(12, 14))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 20)
    ax.axis('off')
    
    # Color scheme
    colors = {
        'source': '#E8F4F8',      # Light blue - data sources
        'process': '#FFF4E6',     # Light orange - processing steps
        'qc': '#FFE6E6',          # Light red - quality control
        'output': '#E8F5E9',      # Light green - outputs
        'modality': '#F3E5F5'     # Light purple - modalities
    }
    
    box_style = "round,pad=0.1"
    
    # Title
    ax.text(5, 19, 'GIMAN Data Preprocessing Pipeline', 
            fontsize=16, fontweight='bold', ha='center', va='top')
    ax.text(5, 18.3, 'Parkinson\'s Progression Markers Initiative (PPMI)', 
            fontsize=12, ha='center', va='top', style='italic')
    
    # ===== DATA SOURCES =====
    y_start = 17
    
    # Main PPMI box
    box1 = FancyBboxPatch((1, y_start-0.8), 8, 1.2, 
                          boxstyle=box_style, 
                          facecolor=colors['source'], 
                          edgecolor='black', linewidth=2)
    ax.add_patch(box1)
    ax.text(5, y_start-0.2, 'PPMI Database', 
            fontsize=12, fontweight='bold', ha='center', va='center')
    ax.text(5, y_start-0.5, '2,046 participants • 3 cohorts • 2010-2025', 
            fontsize=9, ha='center', va='center')
    
    # Arrow down
    arrow1 = FancyArrowPatch((5, y_start-0.9), (5, y_start-1.5),
                            arrowstyle='->', mutation_scale=20, 
                            linewidth=2, color='black')
    ax.add_patch(arrow1)
    
    # ===== COHORT SELECTION =====
    y_cohort = y_start - 2.5
    
    box2 = FancyBboxPatch((1.5, y_cohort-0.6), 7, 1.5, 
                          boxstyle=box_style, 
                          facecolor=colors['process'], 
                          edgecolor='black', linewidth=1.5)
    ax.add_patch(box2)
    ax.text(5, y_cohort+0.5, 'Cohort Selection Criteria', 
            fontsize=11, fontweight='bold', ha='center', va='center')
    ax.text(5, y_cohort+0.1, '• PD diagnosis (MDS criteria)', 
            fontsize=9, ha='center', va='center')
    ax.text(5, y_cohort-0.2, '• ≥3 visits spanning ≥2 years', 
            fontsize=9, ha='center', va='center')
    ax.text(5, y_cohort-0.5, '• Complete clinical assessments', 
            fontsize=9, ha='center', va='center')
    
    # Split arrows to two cohorts
    arrow2 = FancyArrowPatch((5, y_cohort-0.7), (5, y_cohort-1.3),
                            arrowstyle='->', mutation_scale=15, 
                            linewidth=1.5, color='black')
    ax.add_patch(arrow2)
    
    # Arrow split
    ax.plot([3, 5, 7], [y_cohort-1.3, y_cohort-1.3, y_cohort-1.3], 
            'k-', linewidth=1.5)
    arrow3a = FancyArrowPatch((3, y_cohort-1.3), (3, y_cohort-1.8),
                             arrowstyle='->', mutation_scale=15, 
                             linewidth=1.5, color='black')
    arrow3b = FancyArrowPatch((7, y_cohort-1.3), (7, y_cohort-1.8),
                             arrowstyle='->', mutation_scale=15, 
                             linewidth=1.5, color='black')
    ax.add_patch(arrow3a)
    ax.add_patch(arrow3b)
    
    # Two cohorts
    y_cohorts = y_cohort - 2.6
    
    # Phase 4 cohort
    box3a = FancyBboxPatch((0.5, y_cohorts-0.5), 2.3, 1, 
                           boxstyle=box_style, 
                           facecolor=colors['output'], 
                           edgecolor='#2E86AB', linewidth=2)
    ax.add_patch(box3a)
    ax.text(1.65, y_cohorts+0.2, 'Phase 4: PD Patients', 
            fontsize=10, fontweight='bold', ha='center', va='center')
    ax.text(1.65, y_cohorts-0.15, 'n = 536', 
            fontsize=10, ha='center', va='center', fontweight='bold')
    ax.text(1.65, y_cohorts-0.35, '5-year follow-up', 
            fontsize=8, ha='center', va='center')
    
    # Phase 5 cohort
    box3b = FancyBboxPatch((5.5, y_cohorts-0.5), 2.3, 1, 
                           boxstyle=box_style, 
                           facecolor=colors['output'], 
                           edgecolor='#A23B72', linewidth=2)
    ax.add_patch(box3b)
    ax.text(6.65, y_cohorts+0.2, 'Phase 5: Prodromal', 
            fontsize=10, fontweight='bold', ha='center', va='center')
    ax.text(6.65, y_cohorts-0.15, 'n = 194', 
            fontsize=10, ha='center', va='center', fontweight='bold')
    ax.text(6.65, y_cohorts-0.35, '68 converters', 
            fontsize=8, ha='center', va='center')
    
    # Arrows down from both cohorts
    arrow4a = FancyArrowPatch((1.65, y_cohorts-0.6), (1.65, y_cohorts-1.2),
                             arrowstyle='->', mutation_scale=15, 
                             linewidth=1.5, color='black')
    arrow4b = FancyArrowPatch((6.65, y_cohorts-0.6), (6.65, y_cohorts-1.2),
                             arrowstyle='->', mutation_scale=15, 
                             linewidth=1.5, color='black')
    ax.add_patch(arrow4a)
    ax.add_patch(arrow4b)
    
    # Merge arrows
    y_merge = y_cohorts - 1.4
    ax.plot([1.65, 4.15], [y_merge, y_merge], 'k-', linewidth=1.5)
    ax.plot([6.65, 4.15], [y_merge, y_merge], 'k-', linewidth=1.5)
    arrow5 = FancyArrowPatch((4.15, y_merge), (4.15, y_merge-0.6),
                            arrowstyle='->', mutation_scale=15, 
                            linewidth=1.5, color='black')
    ax.add_patch(arrow5)
    
    # ===== MULTIMODAL DATA EXTRACTION =====
    y_modal = y_cohorts - 3.2
    
    # Main extraction box
    box4 = FancyBboxPatch((1, y_modal-0.5), 6.3, 0.8, 
                          boxstyle=box_style, 
                          facecolor=colors['process'], 
                          edgecolor='black', linewidth=1.5)
    ax.add_patch(box4)
    ax.text(4.15, y_modal, 'Multimodal Data Extraction', 
            fontsize=11, fontweight='bold', ha='center', va='center')
    
    # Four modality boxes
    y_modalities = y_modal - 1.8
    modality_width = 1.4
    modality_height = 1.2
    
    modalities = [
        ('Clinical', '42 features', 'UPDRS, MoCA\nRBD, UPSIT', 1),
        ('Imaging', '28 features', 'DaTscan SBR\nMRI cortical', 2.8),
        ('Genetic', '5 features', 'LRRK2, GBA\nSNCA, APOE', 4.6),
        ('Derived', '12 features', 'Slopes\nRatios', 6.4)
    ]
    
    for title, count, details, x_pos in modalities:
        box = FancyBboxPatch((x_pos, y_modalities-modality_height/2), 
                            modality_width, modality_height, 
                            boxstyle=box_style, 
                            facecolor=colors['modality'], 
                            edgecolor='black', linewidth=1)
        ax.add_patch(box)
        ax.text(x_pos+modality_width/2, y_modalities+0.35, title, 
                fontsize=9, fontweight='bold', ha='center', va='center')
        ax.text(x_pos+modality_width/2, y_modalities+0.05, count, 
                fontsize=8, ha='center', va='center', color='#E63946', fontweight='bold')
        ax.text(x_pos+modality_width/2, y_modalities-0.3, details, 
                fontsize=7, ha='center', va='center')
    
    # Arrow from extraction to modalities
    arrow6 = FancyArrowPatch((4.15, y_modal-0.6), (4.15, y_modalities+0.65),
                            arrowstyle='->', mutation_scale=15, 
                            linewidth=1.5, color='black')
    ax.add_patch(arrow6)
    
    # Arrows from modalities converging
    y_converge = y_modalities - 0.75
    for x_pos in [1.7, 3.5, 5.3, 7.1]:
        ax.plot([x_pos, x_pos], [y_modalities-0.6, y_converge], 
                'k-', linewidth=1)
    ax.plot([1.7, 7.1], [y_converge, y_converge], 'k-', linewidth=1.5)
    arrow7 = FancyArrowPatch((4.4, y_converge), (4.4, y_converge-0.6),
                            arrowstyle='->', mutation_scale=15, 
                            linewidth=1.5, color='black')
    ax.add_patch(arrow7)
    
    # ===== QUALITY CONTROL =====
    y_qc = y_converge - 1.5
    
    box5 = FancyBboxPatch((1.5, y_qc-0.6), 5.8, 1.4, 
                          boxstyle=box_style, 
                          facecolor=colors['qc'], 
                          edgecolor='#E63946', linewidth=2)
    ax.add_patch(box5)
    ax.text(4.4, y_qc+0.5, 'Quality Control Pipeline', 
            fontsize=11, fontweight='bold', ha='center', va='center')
    ax.text(4.4, y_qc+0.15, '1. Missing data threshold: <30% baseline, <50% longitudinal', 
            fontsize=8, ha='center', va='center')
    ax.text(4.4, y_qc-0.1, '2. Outlier detection: ±3 SD from cohort mean', 
            fontsize=8, ha='center', va='center')
    ax.text(4.4, y_qc-0.35, '3. Imaging quality: Motion artifacts, segmentation failures', 
            fontsize=8, ha='center', va='center')
    ax.text(4.4, y_qc-0.6, '4. Trajectory smoothness: Non-monotonic jump detection', 
            fontsize=8, ha='center', va='center')
    
    arrow8 = FancyArrowPatch((4.4, y_qc-0.7), (4.4, y_qc-1.3),
                            arrowstyle='->', mutation_scale=15, 
                            linewidth=1.5, color='black')
    ax.add_patch(arrow8)
    
    # ===== MISSING DATA IMPUTATION =====
    y_impute = y_qc - 2.2
    
    box6 = FancyBboxPatch((1.5, y_impute-0.5), 5.8, 1.1, 
                          boxstyle=box_style, 
                          facecolor=colors['process'], 
                          edgecolor='black', linewidth=1.5)
    ax.add_patch(box6)
    ax.text(4.4, y_impute+0.3, 'Missing Data Imputation (median 8% per feature)', 
            fontsize=10, fontweight='bold', ha='center', va='center')
    ax.text(4.4, y_impute, '• Clinical: KNN (k=5) • Imaging: MICE', 
            fontsize=8, ha='center', va='center')
    ax.text(4.4, y_impute-0.25, '• Genetic: None (binary) • Trajectories: Linear interpolation', 
            fontsize=8, ha='center', va='center')
    
    arrow9 = FancyArrowPatch((4.4, y_impute-0.6), (4.4, y_impute-1.2),
                            arrowstyle='->', mutation_scale=15, 
                            linewidth=1.5, color='black')
    ax.add_patch(arrow9)
    
    # ===== NORMALIZATION =====
    y_norm = y_impute - 1.9
    
    box7 = FancyBboxPatch((1.5, y_norm-0.4), 5.8, 0.8, 
                          boxstyle=box_style, 
                          facecolor=colors['process'], 
                          edgecolor='black', linewidth=1.5)
    ax.add_patch(box7)
    ax.text(4.4, y_norm+0.1, 'Feature Normalization & Scaling', 
            fontsize=10, fontweight='bold', ha='center', va='center')
    ax.text(4.4, y_norm-0.2, 'Z-score (mean=0, SD=1) + Min-Max [0,1]', 
            fontsize=8, ha='center', va='center')
    
    arrow10 = FancyArrowPatch((4.4, y_norm-0.5), (4.4, y_norm-1.1),
                             arrowstyle='->', mutation_scale=15, 
                             linewidth=1.5, color='black')
    ax.add_patch(arrow10)
    
    # ===== FINAL DATASETS =====
    y_final = y_norm - 2.1
    
    # Phase 4 final
    box8a = FancyBboxPatch((0.5, y_final-0.5), 3, 1, 
                           boxstyle=box_style, 
                           facecolor=colors['output'], 
                           edgecolor='#2E86AB', linewidth=2.5)
    ax.add_patch(box8a)
    ax.text(2, y_final+0.2, 'Phase 4 Dataset', 
            fontsize=10, fontweight='bold', ha='center', va='center')
    ax.text(2, y_final-0.1, '536 patients × 87 features', 
            fontsize=9, ha='center', va='center', fontweight='bold')
    ax.text(2, y_final-0.35, 'Median 6 visits, 5-year f/u', 
            fontsize=8, ha='center', va='center')
    
    # Phase 5 final
    box8b = FancyBboxPatch((5.5, y_final-0.5), 3, 1, 
                           boxstyle=box_style, 
                           facecolor=colors['output'], 
                           edgecolor='#A23B72', linewidth=2.5)
    ax.add_patch(box8b)
    ax.text(7, y_final+0.2, 'Phase 5 Dataset', 
            fontsize=10, fontweight='bold', ha='center', va='center')
    ax.text(7, y_final-0.1, '194 subjects × 78 features', 
            fontsize=9, ha='center', va='center', fontweight='bold')
    ax.text(7, y_final-0.35, '68 converters, 3.2-year median', 
            fontsize=8, ha='center', va='center')
    
    # Arrow merge and down
    arrow11a = FancyArrowPatch((2, y_final-0.6), (2, y_final-1),
                              arrowstyle='->', mutation_scale=15, 
                              linewidth=1.5, color='black')
    arrow11b = FancyArrowPatch((7, y_final-0.6), (7, y_final-1),
                              arrowstyle='->', mutation_scale=15, 
                              linewidth=1.5, color='black')
    ax.add_patch(arrow11a)
    ax.add_patch(arrow11b)
    
    y_last = y_final - 1.2
    ax.plot([2, 4.5], [y_last, y_last], 'k-', linewidth=1.5)
    ax.plot([7, 4.5], [y_last, y_last], 'k-', linewidth=1.5)
    arrow12 = FancyArrowPatch((4.5, y_last), (4.5, y_last-0.6),
                             arrowstyle='->', mutation_scale=20, 
                             linewidth=2, color='black')
    ax.add_patch(arrow12)
    
    # ===== NEXT STEPS =====
    box9 = FancyBboxPatch((1.5, 0.3), 6, 0.8, 
                          boxstyle=box_style, 
                          facecolor='#FFF9C4', 
                          edgecolor='black', linewidth=2)
    ax.add_patch(box9)
    ax.text(4.5, 0.8, 'Ready for Graph Construction & GIMAN Modeling', 
            fontsize=11, fontweight='bold', ha='center', va='center')
    ax.text(4.5, 0.5, 'Phase 3: Patient Similarity Graphs → Phase 4-6: GNN Analysis', 
            fontsize=9, ha='center', va='center', style='italic')
    
    plt.tight_layout()
    
    # Ensure directory exists
    output_path = Path('preprocessing_flowchart.png')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    plt.savefig(str(output_path), dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✓ Saved: {output_path}")
    plt.close()

if __name__ == '__main__':
    print("\n" + "="*70)
    print("GENERATING PREPROCESSING FLOWCHART")
    print("="*70 + "\n")
    
    create_preprocessing_flowchart()
    
    print("\n" + "="*70)
    print("COMPLETE!")
    print("="*70)
