"""
Figure 11: Heterogeneous Progression Rates - Two Patient Comparison
Shows progression rates for two patients with contrasting patterns.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Set style
sns.set_style("whitegrid")
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 11

# Define clinical dimensions
dimensions = [
    'Motor Symptoms',
    'Cognitive Decline',
    'Imaging Severity',
    'Genetic Risk',
    'Autonomic\nDysfunction',
    'Gait Impairment',
    'Tremor Severity',
    'Rigidity Score'
]

# Patient 2155 - Mixed progression (rapid cognition/motor, slow tremor)
patient_2155_rates = [
    0.20,  # Motor: rapid
    0.30,  # Cognition: rapid (declining fast!)
    0.25,  # Imaging: rapid
    0.10,  # Genetic: slow (stable)
    0.15,  # Autonomic: moderate
    0.18,  # Gait: moderate
    0.08,  # Tremor: slow (stable!)
    0.22,  # Rigidity: rapid
]

# Patient 4832 - Predominantly slow progression (tremor-dominant)
patient_4832_rates = [
    0.12,  # Motor: slow
    0.10,  # Cognition: slow (preserved!)
    0.14,  # Imaging: slow
    0.08,  # Genetic: slow
    0.09,  # Autonomic: slow
    0.11,  # Gait: slow
    0.18,  # Tremor: moderate (tremor-dominant)
    0.10,  # Rigidity: slow
]

# Assign colors based on progression rate thresholds
def get_color_and_label(rate):
    if rate < 0.15:
        return '#2ecc71', 'Slow'
    elif rate < 0.25:
        return '#f39c12', 'Moderate'
    else:
        return '#e74c3c', 'Rapid'

# Create figure with 2 panels
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7), sharey=True)

y_pos = np.arange(len(dimensions))

# Panel 1: Patient 2155
colors_2155 = [get_color_and_label(r)[0] for r in patient_2155_rates]
bars1 = ax1.barh(y_pos, patient_2155_rates, color=colors_2155, alpha=0.85, 
                 edgecolor='white', linewidth=2.5)

for i, val in enumerate(patient_2155_rates):
    label = get_color_and_label(val)[1]
    ax1.text(val + 0.01, i, f'{val:.2f}', va='center', fontsize=10, fontweight='bold')

ax1.set_yticks(y_pos)
ax1.set_yticklabels(dimensions, fontsize=12)
ax1.set_xlabel('Progression Rate', fontsize=12, fontweight='bold')
ax1.set_title('Patient 2155: Mixed Progression\n(Rapid cognition/motor, Slow tremor)', 
              fontsize=13, fontweight='bold', pad=15, color='#8e44ad')
ax1.set_xlim(0, 0.35)
ax1.grid(axis='x', alpha=0.35, linewidth=0.8)

# Annotation for Patient 2155
ax1.annotate('Rapid cognition\n(0.30)', 
             xy=(0.30, 1), xytext=(0.24, 3),
             arrowprops=dict(arrowstyle='->', color='#e74c3c', lw=2),
             fontsize=10, color='#e74c3c', fontweight='bold',
             bbox=dict(boxstyle='round,pad=0.4', facecolor='#ffe6e6', edgecolor='#e74c3c', linewidth=2))

ax1.annotate('Slow tremor\n(0.08)', 
             xy=(0.08, 6), xytext=(0.14, 5),
             arrowprops=dict(arrowstyle='->', color='#2ecc71', lw=2),
             fontsize=10, color='#2ecc71', fontweight='bold',
             bbox=dict(boxstyle='round,pad=0.4', facecolor='#e6ffe6', edgecolor='#2ecc71', linewidth=2))

# Panel 2: Patient 4832
colors_4832 = [get_color_and_label(r)[0] for r in patient_4832_rates]
bars2 = ax2.barh(y_pos, patient_4832_rates, color=colors_4832, alpha=0.85,
                 edgecolor='white', linewidth=2.5)

for i, val in enumerate(patient_4832_rates):
    label = get_color_and_label(val)[1]
    ax2.text(val + 0.01, i, f'{val:.2f}', va='center', fontsize=10, fontweight='bold')

ax2.set_xlabel('Progression Rate', fontsize=12, fontweight='bold')
ax2.set_title('Patient 4832: Slow Progression\n(Preserved cognition, Tremor-dominant)', 
              fontsize=13, fontweight='bold', pad=15, color='#16a085')
ax2.set_xlim(0, 0.35)
ax2.grid(axis='x', alpha=0.35, linewidth=0.8)

# Annotation for Patient 4832
ax2.annotate('Preserved cognition\n(0.10)', 
             xy=(0.10, 1), xytext=(0.16, 3),
             arrowprops=dict(arrowstyle='->', color='#2ecc71', lw=2),
             fontsize=10, color='#2ecc71', fontweight='bold',
             bbox=dict(boxstyle='round,pad=0.4', facecolor='#e6ffe6', edgecolor='#2ecc71', linewidth=2))

ax2.annotate('Moderate tremor\n(0.18)', 
             xy=(0.18, 6), xytext=(0.22, 5),
             arrowprops=dict(arrowstyle='->', color='#f39c12', lw=2),
             fontsize=10, color='#f39c12', fontweight='bold',
             bbox=dict(boxstyle='round,pad=0.4', facecolor='#fff4e6', edgecolor='#f39c12', linewidth=2))

# Add shared legend
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor='#2ecc71', edgecolor='white', label='Slow (<0.15)'),
    Patch(facecolor='#f39c12', edgecolor='white', label='Moderate (0.15-0.25)'),
    Patch(facecolor='#e74c3c', edgecolor='white', label='Rapid (>0.25)')
]
ax2.legend(handles=legend_elements, loc='lower right', fontsize=11, framealpha=0.95, edgecolor='black')

# Overall title
fig.suptitle('Heterogeneous Progression Rates: Patient Comparison',
             fontsize=15, fontweight='bold', y=0.98)

plt.tight_layout()

# Save
output_path = '/Users/blair.dupre/Library/CloudStorage/GoogleDrive-dupre.blair92@gmail.com/My Drive/CSCI FALL 2025/visualizations/publication/Figure11_Patient_Fuzzy_Profile.png'
plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
print(f"Saved 2-patient comparison to: {output_path}")

plt.show()
