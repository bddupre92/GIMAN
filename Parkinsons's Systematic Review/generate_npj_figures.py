"""
Generate all 4 figures for NPJ Parkinson's Disease manuscript
Publication-quality figures with proper formatting
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, Rectangle
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec

# Set publication-quality defaults
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.linewidth'] = 1.0
plt.rcParams['xtick.major.width'] = 1.0
plt.rcParams['ytick.major.width'] = 1.0

# Create output directory
import os
os.makedirs('/home/sandbox/figures', exist_ok=True)

print("Generating NPJ Parkinson's Disease manuscript figures...")
print("="*80)

# ============================================================================
# FIGURE 1: PRISMA 2020 FLOW DIAGRAM
# ============================================================================

def create_prisma_diagram():
    """Create PRISMA 2020 flow diagram"""
    fig, ax = plt.subplots(figsize=(8, 11))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 14)
    ax.axis('off')
    
    # Define box styling
    box_width = 3.5
    box_height = 0.8
    
    # Title
    ax.text(5, 13.5, 'PRISMA 2020 Flow Diagram', 
            ha='center', va='center', fontsize=14, fontweight='bold')
    
    # IDENTIFICATION
    ax.text(1, 12.8, 'Identification', fontsize=11, fontweight='bold')
    
    # Database searches box
    box1 = FancyBboxPatch((1, 11.5), box_width, box_height, 
                          boxstyle="round,pad=0.05", 
                          edgecolor='black', facecolor='lightblue', linewidth=1.5)
    ax.add_patch(box1)
    ax.text(2.75, 11.9, 'Records identified from databases', 
            ha='center', va='center', fontsize=9, fontweight='bold')
    ax.text(2.75, 11.65, '(n = 354)', ha='center', va='center', fontsize=9)
    
    # Database breakdown (right side)
    db_text = 'SciSpace: n=142\nPubMed: n=98\nGoogle Scholar: n=87\nArXiv: n=27'
    ax.text(6.5, 11.9, db_text, ha='left', va='center', fontsize=8,
            bbox=dict(boxstyle='round', facecolor='white', edgecolor='gray'))
    
    # Arrow down
    ax.arrow(2.75, 11.5, 0, -0.3, head_width=0.15, head_length=0.08, 
             fc='black', ec='black')
    
    # Duplicates removed box
    box2 = FancyBboxPatch((1, 10.5), box_width, box_height,
                          boxstyle="round,pad=0.05",
                          edgecolor='black', facecolor='#FFE6E6', linewidth=1.5)
    ax.add_patch(box2)
    ax.text(2.75, 10.9, 'Records removed before screening', 
            ha='center', va='center', fontsize=9, fontweight='bold')
    ax.text(2.75, 10.65, 'Duplicate records (n = 67)', 
            ha='center', va='center', fontsize=8)
    
    # Arrow down
    ax.arrow(2.75, 10.5, 0, -0.3, head_width=0.15, head_length=0.08,
             fc='black', ec='black')
    
    # SCREENING
    ax.text(1, 9.8, 'Screening', fontsize=11, fontweight='bold')
    
    # Records screened box
    box3 = FancyBboxPatch((1, 8.5), box_width, box_height,
                          boxstyle="round,pad=0.05",
                          edgecolor='black', facecolor='lightblue', linewidth=1.5)
    ax.add_patch(box3)
    ax.text(2.75, 8.9, 'Records screened', 
            ha='center', va='center', fontsize=9, fontweight='bold')
    ax.text(2.75, 8.65, '(n = 287)', ha='center', va='center', fontsize=9)
    
    # Excluded at screening (right side)
    box4 = FancyBboxPatch((5.5, 8.5), box_width, box_height,
                          boxstyle="round,pad=0.05",
                          edgecolor='black', facecolor='#FFE6E6', linewidth=1.5)
    ax.add_patch(box4)
    ax.text(7.25, 8.9, 'Records excluded', 
            ha='center', va='center', fontsize=9, fontweight='bold')
    ax.text(7.25, 8.65, '(n = 267)', ha='center', va='center', fontsize=9)
    
    # Exclusion reasons (small text)
    exclusion_text = ('No comparator: n=156 (57.4%)\n'
                     'Diagnostic only: n=68 (25.0%)\n'
                     'No dynamic model: n=31 (11.4%)\n'
                     'Other reasons: n=17 (6.3%)')
    ax.text(7.25, 7.8, exclusion_text, ha='center', va='top', fontsize=7,
            bbox=dict(boxstyle='round', facecolor='white', edgecolor='gray'))
    
    # Arrow from screening to excluded
    ax.arrow(4.5, 8.9, 0.9, 0, head_width=0.08, head_length=0.1,
             fc='black', ec='black')
    
    # Arrow down from screened
    ax.arrow(2.75, 8.5, 0, -0.3, head_width=0.15, head_length=0.08,
             fc='black', ec='black')
    
    # Full-text assessed box
    box5 = FancyBboxPatch((1, 7.2), box_width, box_height,
                          boxstyle="round,pad=0.05",
                          edgecolor='black', facecolor='lightblue', linewidth=1.5)
    ax.add_patch(box5)
    ax.text(2.75, 7.6, 'Reports assessed for eligibility', 
            ha='center', va='center', fontsize=9, fontweight='bold')
    ax.text(2.75, 7.35, '(n = 20)', ha='center', va='center', fontsize=9)
    
    # Excluded at full-text (right side)
    box6 = FancyBboxPatch((5.5, 7.2), box_width, box_height,
                          boxstyle="round,pad=0.05",
                          edgecolor='black', facecolor='#FFE6E6', linewidth=1.5)
    ax.add_patch(box6)
    ax.text(7.25, 7.6, 'Reports excluded', 
            ha='center', va='center', fontsize=9, fontweight='bold')
    ax.text(7.25, 7.35, '(n = 5)', ha='center', va='center', fontsize=9)
    
    # Arrow from full-text to excluded
    ax.arrow(4.5, 7.6, 0.9, 0, head_width=0.08, head_length=0.1,
             fc='black', ec='black')
    
    # Arrow down
    ax.arrow(2.75, 7.2, 0, -0.3, head_width=0.15, head_length=0.08,
             fc='black', ec='black')
    
    # INCLUDED
    ax.text(1, 6.5, 'Included', fontsize=11, fontweight='bold')
    
    # Studies included box
    box7 = FancyBboxPatch((1, 5.2), box_width, box_height*1.2,
                          boxstyle="round,pad=0.05",
                          edgecolor='black', facecolor='lightgreen', linewidth=2)
    ax.add_patch(box7)
    ax.text(2.75, 5.9, 'Studies included in review', 
            ha='center', va='center', fontsize=9, fontweight='bold')
    ax.text(2.75, 5.6, '(n = 15)', ha='center', va='center', fontsize=10, fontweight='bold')
    ax.text(2.75, 5.35, 'Inclusion rate: 5.2%', 
            ha='center', va='center', fontsize=8, style='italic')
    
    # Comparative studies box (below)
    box8 = FancyBboxPatch((1, 3.8), box_width, box_height,
                          boxstyle="round,pad=0.05",
                          edgecolor='darkgreen', facecolor='#E6FFE6', 
                          linewidth=1.5, linestyle='--')
    ax.add_patch(box8)
    ax.text(2.75, 4.3, 'With head-to-head comparisons', 
            ha='center', va='center', fontsize=8)
    ax.text(2.75, 4.05, '(n = 6, 40%)', ha='center', va='center', fontsize=8, fontweight='bold')
    
    # Hypothesis testing box (below)
    box9 = FancyBboxPatch((1, 2.8), box_width, box_height,
                          boxstyle="round,pad=0.05",
                          edgecolor='darkgreen', facecolor='#E6FFE6',
                          linewidth=1.5, linestyle='--')
    ax.add_patch(box9)
    ax.text(2.75, 3.3, 'Testing dynamic vs. static hypothesis', 
            ha='center', va='center', fontsize=8)
    ax.text(2.75, 3.05, '(n = 2, 13%)', ha='center', va='center', fontsize=8, fontweight='bold')
    
    # Inter-rater agreement note
    ax.text(5, 1.5, 'Inter-rater agreement: κ=0.82 (95% CI: 0.71-0.93)', 
            ha='center', va='center', fontsize=8, style='italic',
            bbox=dict(boxstyle='round', facecolor='lightyellow', edgecolor='gray'))
    
    plt.tight_layout()
    plt.savefig('/home/sandbox/figures/Figure1_PRISMA_Diagram.png', dpi=300, bbox_inches='tight')
    plt.savefig('/home/sandbox/figures/Figure1_PRISMA_Diagram.pdf', bbox_inches='tight')
    print("✓ Figure 1: PRISMA diagram saved")
    plt.close()

# ============================================================================
# FIGURE 2: PROBAST RISK OF BIAS TRAFFIC LIGHT PLOT
# ============================================================================

def create_probast_plot():
    """Create PROBAST traffic light plot"""
    # Data for 15 studies
    studies = [
        'Lian 2024',
        'Dadu 2022', 
        'Ren 2021',
        'Hayete 2017',
        'Gao 2018',
        'Amprimo 2024',
        'Li 2022',
        'Venuto 2023',
        'van Wegen 2016',
        'Paper 13 (2025)',
        'Paper 14 (2025)',
        'Paper 15 (2026)',
        'Warmerdam 2019',
        'Tan 2022',
        'Paper 17 (2025)'
    ]
    
    # PROBAST domains: Participants, Predictors, Outcome, Analysis, Overall
    # 0 = Low (green), 1 = Moderate (yellow), 2 = High (red)
    risk_data = np.array([
        [0, 0, 0, 1, 1],  # Lian 2024
        [0, 0, 0, 1, 1],  # Dadu 2022
        [0, 0, 0, 0, 0],  # Ren 2021
        [1, 1, 0, 1, 1],  # Hayete 2017
        [0, 0, 0, 0, 0],  # Gao 2018
        [1, 1, 1, 2, 2],  # Amprimo 2024
        [1, 0, 0, 1, 1],  # Li 2022
        [0, 0, 0, 0, 0],  # Venuto 2023
        [0, 0, 0, 0, 0],  # van Wegen 2016
        [0, 0, 0, 1, 1],  # Paper 13
        [0, 0, 0, 1, 1],  # Paper 14
        [0, 0, 0, 1, 1],  # Paper 15
        [1, 1, 0, 1, 1],  # Warmerdam 2019
        [0, 0, 0, 1, 1],  # Tan 2022
        [1, 1, 1, 2, 2],  # Paper 17
    ])
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Color map
    colors = ['#90EE90', '#FFD700', '#FF6B6B']  # Green, Yellow, Red
    color_map = {0: colors[0], 1: colors[1], 2: colors[2]}
    
    # Plot each cell
    for i in range(len(studies)):
        for j in range(5):
            color = color_map[risk_data[i, j]]
            rect = Rectangle((j, i), 1, 1, facecolor=color, 
                           edgecolor='black', linewidth=1)
            ax.add_patch(rect)
    
    # Set limits and labels
    ax.set_xlim(0, 5)
    ax.set_ylim(0, len(studies))
    
    # X-axis labels
    domains = ['Participants', 'Predictors', 'Outcome', 'Analysis', 'Overall\nRisk']
    ax.set_xticks([0.5, 1.5, 2.5, 3.5, 4.5])
    ax.set_xticklabels(domains, fontsize=10, fontweight='bold')
    ax.xaxis.tick_top()
    
    # Y-axis labels
    ax.set_yticks([i + 0.5 for i in range(len(studies))])
    ax.set_yticklabels(studies, fontsize=9)
    ax.invert_yaxis()
    
    # Remove spines
    for spine in ax.spines.values():
        spine.set_visible(False)
    
    # Add legend
    legend_elements = [
        mpatches.Patch(facecolor=colors[0], edgecolor='black', label='Low risk'),
        mpatches.Patch(facecolor=colors[1], edgecolor='black', label='Moderate risk'),
        mpatches.Patch(facecolor=colors[2], edgecolor='black', label='High risk')
    ]
    ax.legend(handles=legend_elements, loc='upper right', 
             bbox_to_anchor=(1.15, 1.0), frameon=True)
    
    # Add summary statistics
    low_count = np.sum(risk_data[:, 4] == 0)
    mod_count = np.sum(risk_data[:, 4] == 1)
    high_count = np.sum(risk_data[:, 4] == 2)
    
    summary_text = (f'Overall Risk of Bias:\n'
                   f'Low: {low_count} (27%)\n'
                   f'Moderate: {mod_count} (60%)\n'
                   f'High: {high_count} (13%)')
    ax.text(0.5, -1.5, summary_text, fontsize=9, 
           bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.5))
    
    plt.title('PROBAST Risk of Bias Assessment (n=15 studies)', 
             fontsize=12, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig('/home/sandbox/figures/Figure2_PROBAST_Plot.png', dpi=300, bbox_inches='tight')
    plt.savefig('/home/sandbox/figures/Figure2_PROBAST_Plot.pdf', bbox_inches='tight')
    print("✓ Figure 2: PROBAST plot saved")
    plt.close()

# ============================================================================
# FIGURE 3: HARVEST PLOT
# ============================================================================

def create_harvest_plot():
    """Create harvest plot of comparative effectiveness"""
    # Data for 6 comparative studies
    studies = ['Ren 2021\n(Progression)', 
               'Chaithanya 2025\n(Progression)',
               'van Wegen 2016\n(Falls)',
               'Gao 2018\n(Falls)',
               'Tan 2022\n(Progression)',
               'Pishva 2022\n(Cognitive)']
    
    effect_sizes = [9.3, 28.9, 19.3, -2.3, 4.3, 4.4]  # Relative improvement (%)
    validation_tiers = [2, 0, 2, 2, 2, 0]  # 0=Tier 0, 1=Tier 1, 2=Tier 2
    
    # Colors by validation tier
    colors = ['#87CEEB', '#4682B4', '#191970']  # Light, medium, dark blue
    tier_colors = [colors[tier] for tier in validation_tiers]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Create bars
    x_pos = np.arange(len(studies))
    bars = ax.bar(x_pos, effect_sizes, color=tier_colors, 
                  edgecolor='black', linewidth=1.5, alpha=0.8)
    
    # Add horizontal reference lines
    ax.axhline(y=0, color='black', linestyle='-', linewidth=1.5)
    ax.axhline(y=5, color='gray', linestyle='--', linewidth=1, alpha=0.5, label='5% threshold')
    ax.axhline(y=10, color='gray', linestyle='--', linewidth=1, alpha=0.5, label='10% threshold')
    
    # Labels and formatting
    ax.set_xlabel('Study (Prediction Target)', fontsize=11, fontweight='bold')
    ax.set_ylabel('Relative Improvement (%)', fontsize=11, fontweight='bold')
    ax.set_title('Comparative Effectiveness: Dynamic vs. Static Models (n=6 studies)', 
                fontsize=12, fontweight='bold')
    
    ax.set_xticks(x_pos)
    ax.set_xticklabels(studies, fontsize=9, rotation=0, ha='center')
    ax.set_ylim(-5, 32)
    ax.grid(axis='y', alpha=0.3, linestyle=':', linewidth=0.5)
    
    # Add value labels on bars
    for i, (bar, val) in enumerate(zip(bars, effect_sizes)):
        height = bar.get_height()
        y_pos = height + 1 if height > 0 else height - 1.5
        ax.text(bar.get_x() + bar.get_width()/2., y_pos,
               f'{val:+.1f}%', ha='center', va='bottom' if height > 0 else 'top',
               fontsize=9, fontweight='bold')
    
    # Legend for validation tiers
    legend_elements = [
        mpatches.Patch(facecolor=colors[2], edgecolor='black', label='Tier 2 (External validation)'),
        mpatches.Patch(facecolor=colors[1], edgecolor='black', label='Tier 1 (Temporal validation)'),
        mpatches.Patch(facecolor=colors[0], edgecolor='black', label='Tier 0 (Internal validation)')
    ]
    ax.legend(handles=legend_elements, loc='upper left', frameon=True, fontsize=9)
    
    # Add summary text
    summary = '5/6 studies (83%) favor dynamic models\nEffect range: -2.3% to +28.9%'
    ax.text(0.98, 0.02, summary, transform=ax.transAxes,
           fontsize=9, verticalalignment='bottom', horizontalalignment='right',
           bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig('/home/sandbox/figures/Figure3_Harvest_Plot.png', dpi=300, bbox_inches='tight')
    plt.savefig('/home/sandbox/figures/Figure3_Harvest_Plot.pdf', bbox_inches='tight')
    print("✓ Figure 3: Harvest plot saved")
    plt.close()

# ============================================================================
# FIGURE 4: EVIDENCE GAPS (4 PANELS)
# ============================================================================

def create_evidence_gaps():
    """Create 4-panel evidence gaps visualization"""
    fig = plt.figure(figsize=(12, 10))
    gs = GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.3)
    
    # PANEL A: Benchmarking Gap
    ax1 = fig.add_subplot(gs[0, 0])
    categories = ['No direct\ncomparison', 'Quantitative\ncomparison', 'Tests\nhypothesis']
    values = [9, 6, 2]
    colors_panel_a = ['#FF6B6B', '#FFD700', '#90EE90']
    
    bars = ax1.bar(categories, values, color=colors_panel_a, 
                   edgecolor='black', linewidth=1.5)
    ax1.set_ylabel('Number of Studies', fontsize=10, fontweight='bold')
    ax1.set_title('A. Benchmarking Gap', fontsize=11, fontweight='bold', loc='left')
    ax1.set_ylim(0, 10)
    ax1.grid(axis='y', alpha=0.3, linestyle=':', linewidth=0.5)
    
    # Add percentages
    for bar, val in zip(bars, values):
        pct = (val/15)*100
        ax1.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.2,
                f'{val}\n({pct:.0f}%)', ha='center', va='bottom', 
                fontsize=9, fontweight='bold')
    
    # PANEL B: Digital Twin Implementation Gap
    ax2 = fig.add_subplot(gs[0, 1])
    labels = ['True mechanistic\ndigital twins', 'Data-driven\ntemporal models']
    sizes = [0, 15]
    colors_panel_b = ['#FF6B6B', '#4682B4']
    explode = (0.1, 0)
    
    wedges, texts, autotexts = ax2.pie(sizes, labels=labels, autopct='%1.0f%%',
                                        colors=colors_panel_b, explode=explode,
                                        startangle=90, textprops={'fontsize': 9})
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontweight('bold')
        autotext.set_fontsize(11)
    
    ax2.set_title('B. Digital Twin Implementation Gap', fontsize=11, fontweight='bold', loc='left')
    
    # PANEL C: Reporting Quality Gap
    ax3 = fig.add_subplot(gs[1, 0])
    metrics = ['95% CI for\nintervention', '95% CI for\ncomparator', 'Calibration\nmetrics']
    reported = [0, 2, 4]
    not_reported = [15, 13, 11]
    
    x = np.arange(len(metrics))
    width = 0.35
    
    bars1 = ax3.barh(x, reported, width, label='Reported', 
                     color='#90EE90', edgecolor='black', linewidth=1)
    bars2 = ax3.barh(x, not_reported, width, left=reported, label='Not reported',
                     color='#FF6B6B', edgecolor='black', linewidth=1)
    
    ax3.set_yticks(x)
    ax3.set_yticklabels(metrics, fontsize=9)
    ax3.set_xlabel('Number of Studies', fontsize=10, fontweight='bold')
    ax3.set_title('C. Reporting Quality Gap', fontsize=11, fontweight='bold', loc='left')
    ax3.set_xlim(0, 16)
    ax3.legend(loc='lower right', frameon=True, fontsize=9)
    ax3.grid(axis='x', alpha=0.3, linestyle=':', linewidth=0.5)
    
    # Add values
    for i, (r, nr) in enumerate(zip(reported, not_reported)):
        if r > 0:
            ax3.text(r/2, i, str(r), ha='center', va='center', 
                    fontsize=9, fontweight='bold', color='black')
        ax3.text(r + nr/2, i, str(nr), ha='center', va='center',
                fontsize=9, fontweight='bold', color='white')
    
    # PANEL D: Meta-Analysis Barriers (Venn-style)
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.set_xlim(0, 10)
    ax4.set_ylim(0, 10)
    ax4.axis('off')
    
    # Draw overlapping circles
    from matplotlib.patches import Circle
    circle1 = Circle((3, 6), 2, color='#FF6B6B', alpha=0.3, ec='black', linewidth=2)
    circle2 = Circle((7, 6), 2, color='#4682B4', alpha=0.3, ec='black', linewidth=2)
    circle3 = Circle((3, 3), 2, color='#FFD700', alpha=0.3, ec='black', linewidth=2)
    circle4 = Circle((7, 3), 2, color='#90EE90', alpha=0.3, ec='black', linewidth=2)
    
    ax4.add_patch(circle1)
    ax4.add_patch(circle2)
    ax4.add_patch(circle3)
    ax4.add_patch(circle4)
    
    # Labels
    ax4.text(3, 8.2, 'Heterogeneous\nmetrics', ha='center', va='center', 
            fontsize=8, fontweight='bold')
    ax4.text(7, 8.2, 'Heterogeneous\ntargets', ha='center', va='center',
            fontsize=8, fontweight='bold')
    ax4.text(3, 1, 'Heterogeneous\nfollow-up', ha='center', va='center',
            fontsize=8, fontweight='bold')
    ax4.text(7, 1, 'Missing\nvariance', ha='center', va='center',
            fontsize=8, fontweight='bold')
    
    # Center text
    ax4.text(5, 4.5, 'All 15 studies\nexhibit multiple\nbarriers', 
            ha='center', va='center', fontsize=10, fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='white', edgecolor='black', linewidth=2))
    
    ax4.set_title('D. Meta-Analysis Feasibility Barriers', 
                 fontsize=11, fontweight='bold', loc='left')
    
    # Main title
    fig.suptitle('Critical Evidence Gaps in Dynamic Modeling for Parkinson Disease Prognosis',
                fontsize=13, fontweight='bold', y=0.98)
    
    plt.savefig('/home/sandbox/figures/Figure4_Evidence_Gaps.png', dpi=300, bbox_inches='tight')
    plt.savefig('/home/sandbox/figures/Figure4_Evidence_Gaps.pdf', bbox_inches='tight')
    print("✓ Figure 4: Evidence gaps visualization saved")
    plt.close()

# ============================================================================
# GENERATE ALL FIGURES
# ============================================================================

if __name__ == "__main__":
    print("\nGenerating Figure 1: PRISMA Flow Diagram...")
    create_prisma_diagram()
    
    print("\nGenerating Figure 2: PROBAST Traffic Light Plot...")
    create_probast_plot()
    
    print("\nGenerating Figure 3: Harvest Plot...")
    create_harvest_plot()
    
    print("\nGenerating Figure 4: Evidence Gaps (4 panels)...")
    create_evidence_gaps()
    
    print("\n" + "="*80)
    print("✓✓✓ ALL FIGURES GENERATED SUCCESSFULLY ✓✓✓")
    print("="*80)
    print("\nOutput files:")
    print("  - /home/sandbox/figures/Figure1_PRISMA_Diagram.png (and .pdf)")
    print("  - /home/sandbox/figures/Figure2_PROBAST_Plot.png (and .pdf)")
    print("  - /home/sandbox/figures/Figure3_Harvest_Plot.png (and .pdf)")
    print("  - /home/sandbox/figures/Figure4_Evidence_Gaps.png (and .pdf)")
    print("\nAll figures are publication-quality (300 DPI) and ready for NPJ submission.")
    print("="*80)
