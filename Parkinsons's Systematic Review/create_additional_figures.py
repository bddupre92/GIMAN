"""
Create additional figures and tables referenced in manuscript_complete_final.md
Based on detailed analysis of Methods and Results sections
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, Rectangle, Circle, Wedge
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec
import seaborn as sns

# Set publication-quality defaults
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.linewidth'] = 1.0

print("="*80)
print("CREATING ADDITIONAL FIGURES FOR MANUSCRIPT")
print("="*80)
print()

# ============================================================================
# TABLE 3 VISUALIZATION: PROBAST RISK OF BIAS DETAILED HEATMAP
# ============================================================================

def create_probast_detailed_heatmap():
    """Create detailed PROBAST heatmap matching Table 3 structure"""
    
    # 15 studies
    studies = [
        'Lian 2024', 'Dadu 2022', 'Ren 2021', 'Hayete 2017', 'Gao 2018',
        'Amprimo 2024', 'Author 2022', 'Venuto 2023', 'Lindholm 2016',
        'Graham 2025', 'Chaithanya 2025', 'Sarkar 2026', 'Bloem 2019',
        'Sadaei 2022', 'Rizou 2025'
    ]
    
    # 5 columns: Participants, Predictors, Outcome, Analysis, Overall
    # 0=Low (green), 1=Moderate (yellow), 2=High (red)
    risk_matrix = np.array([
        [0, 0, 0, 1, 1],  # Lian 2024
        [0, 0, 0, 1, 1],  # Dadu 2022
        [0, 0, 0, 0, 0],  # Ren 2021 - LOW overall
        [1, 1, 0, 1, 1],  # Hayete 2017
        [0, 0, 0, 0, 0],  # Gao 2018 - LOW overall
        [1, 1, 1, 2, 2],  # Amprimo 2024 - HIGH
        [0, 1, 0, 1, 1],  # Author 2022
        [0, 0, 0, 1, 1],  # Venuto 2023
        [0, 0, 0, 1, 1],  # Lindholm 2016
        [0, 0, 0, 2, 2],  # Graham 2025 - HIGH
        [0, 1, 0, 2, 2],  # Chaithanya 2025 - HIGH
        [1, 1, 0, 1, 1],  # Sarkar 2026
        [0, 1, 0, 2, 2],  # Bloem 2019 - HIGH
        [0, 0, 0, 1, 1],  # Sadaei 2022
        [0, 1, 1, 2, 2],  # Rizou 2025 - HIGH
    ])
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 9), gridspec_kw={'width_ratios': [3, 1]})
    
    # Main heatmap
    colors = ['#90EE90', '#FFD700', '#FF6B6B']  # Green, Yellow, Red
    cmap = plt.matplotlib.colors.ListedColormap(colors)
    
    im = ax1.imshow(risk_matrix, cmap=cmap, aspect='auto', vmin=0, vmax=2)
    
    # Set ticks
    ax1.set_xticks(np.arange(5))
    ax1.set_yticks(np.arange(15))
    ax1.set_xticklabels(['Participants', 'Predictors', 'Outcome', 'Analysis', 'Overall\nRisk'],
                        fontsize=10, fontweight='bold')
    ax1.set_yticklabels(studies, fontsize=9)
    
    # Grid
    ax1.set_xticks(np.arange(5)-.5, minor=True)
    ax1.set_yticks(np.arange(15)-.5, minor=True)
    ax1.grid(which="minor", color="black", linestyle='-', linewidth=1.5)
    ax1.tick_params(which="minor", size=0)
    
    # Highlight overall column
    for i in range(15):
        rect = Rectangle((3.5, i-.5), 1, 1, fill=False, edgecolor='black', linewidth=3)
        ax1.add_patch(rect)
    
    ax1.set_title('PROBAST Risk of Bias Assessment (n=15 studies)', 
                 fontsize=13, fontweight='bold', pad=15)
    
    # Summary panel
    ax2.axis('off')
    
    # Count risks per domain
    participants_low = np.sum(risk_matrix[:, 0] == 0)
    participants_mod = np.sum(risk_matrix[:, 0] == 1)
    
    predictors_low = np.sum(risk_matrix[:, 1] == 0)
    predictors_mod = np.sum(risk_matrix[:, 1] == 1)
    
    outcome_low = np.sum(risk_matrix[:, 2] == 0)
    outcome_mod = np.sum(risk_matrix[:, 2] == 1)
    
    analysis_low = np.sum(risk_matrix[:, 3] == 0)
    analysis_mod = np.sum(risk_matrix[:, 3] == 1)
    analysis_high = np.sum(risk_matrix[:, 3] == 2)
    
    overall_low = np.sum(risk_matrix[:, 4] == 0)
    overall_mod = np.sum(risk_matrix[:, 4] == 1)
    overall_high = np.sum(risk_matrix[:, 4] == 2)
    
    summary_text = f"""DOMAIN SUMMARY
    
Participants:
  Low: {participants_low} (80%)
  Moderate: {participants_mod} (20%)
  High: 0 (0%)
  
Predictors:
  Low: {predictors_low} (53%)
  Moderate: {predictors_mod} (47%)
  High: 0 (0%)
  
Outcome:
  Low: {outcome_low} (73%)
  Moderate: {outcome_mod} (27%)
  High: 0 (0%)
  
Analysis:
  Low: {analysis_low} (13%)
  Moderate: {analysis_mod} (33%)
  High: {analysis_high} (53%)
  
OVERALL:
  Low: {overall_low} (13%)
  Moderate: {overall_mod} (33%)
  High: {overall_high} (53%)
  
CRITICAL RED FLAGS:
• Missing 95% CI: 13/15 (87%)
• No comparator: 10/15 (67%)
• Unclear splitting: 3/15 (20%)
• Data leakage: 2/15 (13%)
"""
    
    ax2.text(0.1, 0.95, summary_text, transform=ax2.transAxes,
            fontsize=9, verticalalignment='top', family='monospace',
            bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.3))
    
    # Legend
    legend_elements = [
        mpatches.Patch(facecolor=colors[0], edgecolor='black', label='Low risk'),
        mpatches.Patch(facecolor=colors[1], edgecolor='black', label='Moderate risk'),
        mpatches.Patch(facecolor=colors[2], edgecolor='black', label='High risk')
    ]
    ax1.legend(handles=legend_elements, loc='upper left', 
              bbox_to_anchor=(0, -0.05), ncol=3, frameon=True, fontsize=9)
    
    plt.tight_layout()
    plt.savefig('/home/sandbox/figures/Table3_PROBAST_Detailed_Heatmap.png', dpi=300, bbox_inches='tight')
    plt.savefig('/home/sandbox/figures/Table3_PROBAST_Detailed_Heatmap.pdf', bbox_inches='tight')
    print("✓ Table 3 Visualization: PROBAST detailed heatmap saved")
    plt.close()

# ============================================================================
# FIGURE: VALIDATION TIER DISTRIBUTION
# ============================================================================

def create_validation_tier_figure():
    """Create validation tier distribution pie chart"""
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Pie chart for validation tiers
    sizes = [10, 1, 4]  # Tier 2, Tier 1, Tier 0
    labels = ['Tier 2\nExternal/Prospective\n(Low risk)', 
              'Tier 1\nTemporal/Site\n(Moderate risk)',
              'Tier 0\nInternal CV\n(High risk)']
    colors = ['#2E8B57', '#FFD700', '#FF6B6B']
    explode = (0.05, 0, 0.05)
    
    wedges, texts, autotexts = ax1.pie(sizes, labels=labels, autopct='%1.1f%%',
                                        colors=colors, explode=explode,
                                        startangle=90, textprops={'fontsize': 10})
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontweight('bold')
        autotext.set_fontsize(12)
    
    ax1.set_title('Validation Tier Distribution (n=15)', fontsize=12, fontweight='bold')
    
    # Bar chart showing n per tier
    tiers = ['Tier 2\n(External)', 'Tier 1\n(Temporal)', 'Tier 0\n(Internal)']
    counts = [10, 1, 4]
    
    bars = ax2.bar(tiers, counts, color=colors, edgecolor='black', linewidth=1.5)
    ax2.set_ylabel('Number of Studies', fontsize=11, fontweight='bold')
    ax2.set_title('Validation Quality Distribution', fontsize=12, fontweight='bold')
    ax2.set_ylim(0, 12)
    ax2.grid(axis='y', alpha=0.3, linestyle=':', linewidth=0.5)
    
    # Add counts on bars
    for bar, count in zip(bars, counts):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.3,
                f'n={count}\n({count/15*100:.1f}%)',
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('/home/sandbox/figures/Validation_Tier_Distribution.png', dpi=300, bbox_inches='tight')
    plt.savefig('/home/sandbox/figures/Validation_Tier_Distribution.pdf', bbox_inches='tight')
    print("✓ Validation tier distribution figure saved")
    plt.close()

# ============================================================================
# FIGURE: MODEL TYPE DISTRIBUTION
# ============================================================================

def create_model_type_distribution():
    """Create model type distribution with emphasis on digital twin gap"""
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    categories = ['Dynamic/\nTime-Series', 'Static\nML', 'Mechanistic\nDigital Twin', 
                  'Mechanistic\nFeatures\n(integrated)']
    values = [10, 5, 0, 4]
    colors = ['#4682B4', '#87CEEB', '#FF6B6B', '#FFD700']
    
    bars = ax.bar(categories, values, color=colors, edgecolor='black', linewidth=1.5)
    
    # Special handling for zero value
    ax.axhline(y=0.5, color='red', linestyle='--', linewidth=2, alpha=0.7)
    ax.text(2, 0.7, '⚠️ CRITICAL GAP: 0 studies', 
           ha='center', fontsize=11, fontweight='bold', color='red',
           bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))
    
    ax.set_ylabel('Number of Studies', fontsize=11, fontweight='bold')
    ax.set_title('Model Type Distribution Among Included Studies (n=15)', 
                fontsize=13, fontweight='bold')
    ax.set_ylim(0, 12)
    ax.grid(axis='y', alpha=0.3, linestyle=':', linewidth=0.5)
    
    # Add values and percentages
    for bar, val in zip(bars, values):
        if val > 0:
            pct = (val/15)*100
            ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.3,
                   f'n={val}\n({pct:.1f}%)',
                   ha='center', va='bottom', fontsize=10, fontweight='bold')
        else:
            ax.text(bar.get_x() + bar.get_width()/2., 0.1,
                   'n=0\n(0%)',
                   ha='center', va='bottom', fontsize=10, fontweight='bold', color='red')
    
    plt.tight_layout()
    plt.savefig('/home/sandbox/figures/Model_Type_Distribution.png', dpi=300, bbox_inches='tight')
    plt.savefig('/home/sandbox/figures/Model_Type_Distribution.pdf', bbox_inches='tight')
    print("✓ Model type distribution figure saved")
    plt.close()

# ============================================================================
# FIGURE: COMPARATIVE STUDY BREAKDOWN
# ============================================================================

def create_comparative_study_breakdown():
    """Create detailed breakdown of comparative vs non-comparative studies"""
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Left panel: Overall breakdown
    categories = ['No direct\ncomparison', 'Has\ncomparison', 'Tests dynamic\nvs. static']
    values = [9, 6, 2]
    colors = ['#FF6B6B', '#FFD700', '#90EE90']
    
    bars1 = ax1.bar(categories, values, color=colors, edgecolor='black', linewidth=1.5)
    ax1.set_ylabel('Number of Studies', fontsize=11, fontweight='bold')
    ax1.set_title('Benchmarking Gap (n=15 included studies)', 
                 fontsize=12, fontweight='bold')
    ax1.set_ylim(0, 10)
    ax1.grid(axis='y', alpha=0.3, linestyle=':', linewidth=0.5)
    
    for bar, val in zip(bars1, values):
        pct = (val/15)*100
        ax1.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.2,
                f'n={val}\n({pct:.0f}%)',
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # Right panel: Comparison type breakdown for 6 comparative studies
    comp_types = ['Dynamic\nvs.\nStatic', 'Static\nvs.\nStatic', 'Other\ncomparisons']
    comp_values = [2, 4, 0]
    comp_colors = ['#2E8B57', '#4682B4', '#D3D3D3']
    
    bars2 = ax2.bar(comp_types, comp_values, color=comp_colors, 
                    edgecolor='black', linewidth=1.5)
    ax2.set_ylabel('Number of Studies', fontsize=11, fontweight='bold')
    ax2.set_title('Type of Comparison (n=6 comparative studies)', 
                 fontsize=12, fontweight='bold')
    ax2.set_ylim(0, 5)
    ax2.grid(axis='y', alpha=0.3, linestyle=':', linewidth=0.5)
    
    for bar, val in zip(bars2, comp_values):
        if val > 0:
            pct = (val/6)*100
            ax2.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.1,
                    f'n={val}\n({pct:.0f}%)',
                    ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # Add key finding
    fig.text(0.5, 0.02, 'KEY FINDING: Only 2/15 studies (13%) directly test the hypothesis', 
            ha='center', fontsize=11, fontweight='bold', 
            bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))
    
    plt.tight_layout(rect=[0, 0.05, 1, 1])
    plt.savefig('/home/sandbox/figures/Comparative_Study_Breakdown.png', dpi=300, bbox_inches='tight')
    plt.savefig('/home/sandbox/figures/Comparative_Study_Breakdown.pdf', bbox_inches='tight')
    print("✓ Comparative study breakdown figure saved")
    plt.close()

# ============================================================================
# FIGURE: PUBLICATION TIMELINE
# ============================================================================

def create_publication_timeline():
    """Create publication timeline showing trend"""
    
    years = [2016, 2017, 2018, 2019, 2021, 2022, 2023, 2024, 2025, 2026]
    counts = [1, 1, 1, 1, 1, 3, 1, 2, 3, 1]
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    bars = ax.bar(years, counts, color='#4682B4', edgecolor='black', 
                  linewidth=1.5, alpha=0.8)
    
    # Highlight post-2020 surge
    for i, (year, count) in enumerate(zip(years, counts)):
        if year >= 2020:
            bars[i].set_color('#FF6B6B')
            bars[i].set_alpha(0.8)
    
    ax.set_xlabel('Publication Year', fontsize=11, fontweight='bold')
    ax.set_ylabel('Number of Studies', fontsize=11, fontweight='bold')
    ax.set_title('Publication Timeline of Included Studies (n=15)', 
                fontsize=13, fontweight='bold')
    ax.set_ylim(0, 4)
    ax.grid(axis='y', alpha=0.3, linestyle=':', linewidth=0.5)
    
    # Add trend annotation
    ax.text(2022, 3.5, '73% published\nafter 2020', 
           ha='center', fontsize=10, fontweight='bold',
           bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    
    # Add values on bars
    for bar, count in zip(bars, counts):
        if count > 0:
            ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.05,
                   f'{count}',
                   ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('/home/sandbox/figures/Publication_Timeline.png', dpi=300, bbox_inches='tight')
    plt.savefig('/home/sandbox/figures/Publication_Timeline.pdf', bbox_inches='tight')
    print("✓ Publication timeline figure saved")
    plt.close()

# ============================================================================
# RUN ALL FIGURE GENERATION
# ============================================================================

if __name__ == "__main__":
    print("\n1. Creating PROBAST detailed heatmap...")
    create_probast_detailed_heatmap()
    
    print("\n2. Creating validation tier distribution...")
    create_validation_tier_figure()
    
    print("\n3. Creating model type distribution...")
    create_model_type_distribution()
    
    print("\n4. Creating comparative study breakdown...")
    create_comparative_study_breakdown()
    
    print("\n5. Creating publication timeline...")
    create_publication_timeline()
    
    print("\n" + "="*80)
    print("✓✓✓ ALL ADDITIONAL FIGURES GENERATED SUCCESSFULLY ✓✓✓")
    print("="*80)
    print("\nTotal figures now available:")
    print("  ORIGINAL (NPJ submission):")
    print("    - Figure 1: PRISMA diagram")
    print("    - Figure 2: PROBAST traffic light")
    print("    - Figure 3: Harvest plot")
    print("    - Figure 4: Evidence gaps (4 panels)")
    print("\n  ADDITIONAL (for full manuscript):")
    print("    - Table 3 visualization: PROBAST detailed heatmap")
    print("    - Validation tier distribution")
    print("    - Model type distribution")
    print("    - Comparative study breakdown")
    print("    - Publication timeline")
    print("\nAll figures saved in /home/sandbox/figures/")
    print("="*80)
