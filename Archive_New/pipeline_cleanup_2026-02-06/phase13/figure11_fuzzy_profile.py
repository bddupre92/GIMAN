"""
Figure 11: Individual Patient Fuzzy Profile (Radar Chart)
Shows a single patient's fuzzy membership across 3 subtypes
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Polygon
from math import pi

project_root = Path(__file__).resolve().parents[3]
output_dir = project_root / "visualizations/publication"
output_dir.mkdir(parents=True, exist_ok=True)

# Set style
plt.rcParams['figure.dpi'] = 300
plt.rcParams['font.size'] = 10
plt.rcParams['font.family'] = 'sans-serif'

def create_radar_chart():
    """Create radar chart for a sample patient"""
    # Load fuzzy membership data
    fcm_df = pd.read_csv(project_root / "visualizations/fcm_clustering/fuzzy_cmeans_results.csv")
    
    # Find an interesting patient with mixed membership (not too dominant in one cluster)
    memberships = fcm_df[['fcm_membership_c0', 'fcm_membership_c1', 'fcm_membership_c2']].values
    
    # Find patient with most balanced membership (closest to showing mixture)
    max_memberships = memberships.max(axis=1)
    # Want someone with max around 0.45-0.55 (not too extreme)
    target_range = np.abs(max_memberships - 0.45)
    interesting_idx = target_range.argmin()
    
    patient_memberships = memberships[interesting_idx]
    patient_id = fcm_df.iloc[interesting_idx]['PATNO']
    
    print(f"Selected Patient {patient_id}")
    print(f"Memberships: {patient_memberships}")
    
    # Create figure with 3 panels showing different patients
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5), subplot_kw=dict(projection='polar'))
    
    fig.suptitle('Individual Patient Fuzzy Profiles (Radar Charts)', 
                 fontsize=14, fontweight='bold', y=0.98)
    
    # Categories
    categories = ['Rapid\nProgressor', 'Moderate\nProgressor', 'Slow\nProgressor']
    N = len(categories)
    
    # Angles for radar chart
    angles = [n / float(N) * 2 * pi for n in range(N)]
    angles += angles[:1]  # Close the plot
    
    # Colors matching our scheme
    colors = ['#e63946', '#457b9d', '#2a9d8f']
    
    # === PANEL 1: Mixed Patient ===
    patient1_memberships = list(patient_memberships) + [patient_memberships[0]]
    
    ax1.plot(angles, patient1_memberships, 'o-', linewidth=2, color='purple', label='Membership')
    ax1.fill(angles, patient1_memberships, alpha=0.25, color='purple')
    ax1.set_xticks(angles[:-1])
    ax1.set_xticklabels(categories, size=9)
    ax1.set_ylim(0, 1)
    ax1.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax1.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], size=8)
    ax1.grid(True, linestyle='--', alpha=0.7)
    ax1.set_title(f'Panel A: Mixed Profile\\n(Patient {int(patient_id)})', 
                  fontsize=11, fontweight='bold', pad=20)
    
    # Add membership values as text
    for angle, membership, category in zip(angles[:-1], patient_memberships, categories):
        ax1.text(angle, membership + 0.1, f'{membership:.0%}', 
                ha='center', va='center', fontsize=9, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
    
    # === PANEL 2: Rapid Progressor Dominant ===
    # Find a patient dominant in cluster 0
    rapid_idx = memberships[:, 0].argmax()
    patient2_memberships = list(memberships[rapid_idx]) + [memberships[rapid_idx][0]]
    patient2_id = fcm_df.iloc[rapid_idx]['PATNO']
    
    ax2.plot(angles, patient2_memberships, 'o-', linewidth=2, color=colors[0], label='Membership')
    ax2.fill(angles, patient2_memberships, alpha=0.25, color=colors[0])
    ax2.set_xticks(angles[:-1])
    ax2.set_xticklabels(categories, size=9)
    ax2.set_ylim(0, 1)
    ax2.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax2.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], size=8)
    ax2.grid(True, linestyle='--', alpha=0.7)
    ax2.set_title(f'Panel B: Rapid Dominant\\n(Patient {int(patient2_id)})', 
                  fontsize=11, fontweight='bold', pad=20)
    
    for angle, membership in zip(angles[:-1], memberships[rapid_idx]):
        ax2.text(angle, membership + 0.1, f'{membership:.0%}', 
                ha='center', va='center', fontsize=9, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
    
    # === PANEL 3: Slow Progressor Dominant ===
    # Find a patient dominant in cluster 2
    slow_idx = memberships[:, 2].argmax()
    patient3_memberships = list(memberships[slow_idx]) + [memberships[slow_idx][0]]
    patient3_id = fcm_df.iloc[slow_idx]['PATNO']
    
    ax3.plot(angles, patient3_memberships, 'o-', linewidth=2, color=colors[2], label='Membership')
    ax3.fill(angles, patient3_memberships, alpha=0.25, color=colors[2])
    ax3.set_xticks(angles[:-1])
    ax3.set_xticklabels(categories, size=9)
    ax3.set_ylim(0, 1)
    ax3.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax3.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], size=8)
    ax3.grid(True, linestyle='--', alpha=0.7)
    ax3.set_title(f'Panel C: Slow Dominant\\n(Patient {int(patient3_id)})', 
                  fontsize=11, fontweight='bold', pad=20)
    
    for angle, membership in zip(angles[:-1], memberships[slow_idx]):
        ax3.text(angle, membership + 0.1, f'{membership:.0%}', 
                ha='center', va='center', fontsize=9, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    output_path = output_dir / "Figure11_Patient_Fuzzy_Profile.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"✅ Saved Figure 11: {output_path}")

def main():
    print("="*80)
    print("FIGURE 11: INDIVIDUAL PATIENT FUZZY PROFILES")
    print("="*80)
    create_radar_chart()
    print("\n✅ Figure 11 complete!")

if __name__ == "__main__":
    main()
