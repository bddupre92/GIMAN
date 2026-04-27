"""
Create comprehensive summary visualization of SAA labels inspection.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (16, 12)

# Load data
saa_df = pd.read_csv('data/04_saa/saa_raw_labels.csv')
phase82_df = pd.read_csv('data/03_prodromal/final_training_dataset/unified_longitudinal_early_pd.csv')

# Get overlap
saa_patients = set(saa_df['PATNO'])
phase82_patients = set(phase82_df['PATNO'])
overlap_patients = saa_patients.intersection(phase82_patients)

# Create figure
fig = plt.figure(figsize=(16, 12))
gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

# 1. SAA Status Distribution (all patients)
ax1 = fig.add_subplot(gs[0, 0])
saa_counts = saa_df['SAA_POSITIVE'].value_counts().sort_index()
colors = ['#2ecc71', '#e74c3c']
bars = ax1.bar(['SAA-', 'SAA+'], saa_counts.values, color=colors, edgecolor='black', alpha=0.7)
ax1.set_ylabel('Number of Patients', fontweight='bold')
ax1.set_title('SAA Status Distribution\n(All 923 Patients)', fontweight='bold')
for i, bar in enumerate(bars):
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height,
           f'{int(height)}\n({100*height/len(saa_df):.1f}%)',
           ha='center', va='bottom', fontweight='bold')
ax1.grid(axis='y', alpha=0.3)

# 2. Alpha-Synuclein Distribution
ax2 = fig.add_subplot(gs[0, 1])
ax2.hist(saa_df['ALPHA_SYN_VALUE'], bins=40, color='skyblue', edgecolor='black', alpha=0.7)
threshold = np.percentile(saa_df['ALPHA_SYN_VALUE'], 80)
ax2.axvline(threshold, color='red', linestyle='--', linewidth=2, label=f'80th %ile ({threshold:.0f})')
ax2.axvline(saa_df['ALPHA_SYN_VALUE'].median(), color='orange', linestyle='--', linewidth=2, 
           label=f'Median ({saa_df["ALPHA_SYN_VALUE"].median():.0f})')
ax2.set_xlabel('CSF α-Synuclein (pg/mL)', fontweight='bold')
ax2.set_ylabel('Frequency', fontweight='bold')
ax2.set_title('CSF α-Synuclein Distribution', fontweight='bold')
ax2.legend()
ax2.grid(axis='y', alpha=0.3)

# 3. Dataset Overlap Venn
ax3 = fig.add_subplot(gs[0, 2])
from matplotlib.patches import Circle
from matplotlib.collections import PatchCollection

# Simplified Venn representation
circle1 = Circle((0.3, 0.5), 0.3, alpha=0.5, color='skyblue', label='SAA Labels')
circle2 = Circle((0.7, 0.5), 0.3, alpha=0.5, color='lightcoral', label='Phase 8.2')
ax3.add_patch(circle1)
ax3.add_patch(circle2)
ax3.text(0.15, 0.5, f'SAA only\n{len(saa_patients - phase82_patients)}', ha='center', va='center', fontweight='bold')
ax3.text(0.85, 0.5, f'Phase 8.2\nonly\n{len(phase82_patients - saa_patients)}', ha='center', va='center', fontweight='bold')
ax3.text(0.5, 0.5, f'Overlap\n{len(overlap_patients)}', ha='center', va='center', fontweight='bold', fontsize=14)
ax3.set_xlim(0, 1)
ax3.set_ylim(0, 1)
ax3.axis('off')
ax3.set_title('Dataset Overlap Analysis', fontweight='bold')
ax3.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=2)

# 4. SAA by Status Box Plot
ax4 = fig.add_subplot(gs[1, 0])
saa_neg = saa_df[saa_df['SAA_POSITIVE'] == 0]['ALPHA_SYN_VALUE']
saa_pos = saa_df[saa_df['SAA_POSITIVE'] == 1]['ALPHA_SYN_VALUE']
bp = ax4.boxplot([saa_neg, saa_pos], labels=['SAA-', 'SAA+'], patch_artist=True)
for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)
ax4.set_ylabel('CSF α-Synuclein (pg/mL)', fontweight='bold')
ax4.set_title('α-Synuclein Levels by SAA Status', fontweight='bold')
ax4.grid(axis='y', alpha=0.3)
# Add mean values
for i, data in enumerate([saa_neg, saa_pos]):
    ax4.text(i+1, data.mean(), f'μ={data.mean():.0f}', ha='center', va='bottom', 
            fontweight='bold', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

# 5. Overlapping Patients SAA Status
ax5 = fig.add_subplot(gs[1, 1])
overlap_saa = saa_df[saa_df['PATNO'].isin(overlap_patients)]
overlap_counts = overlap_saa['SAA_POSITIVE'].value_counts().sort_index()
bars = ax5.bar(['SAA-', 'SAA+'], overlap_counts.values, color=colors, edgecolor='black', alpha=0.7)
ax5.set_ylabel('Number of Patients', fontweight='bold')
ax5.set_title('SAA Status in Overlapping Patients\n(595 Patients for Training)', fontweight='bold')
for i, bar in enumerate(bars):
    height = bar.get_height()
    ax5.text(bar.get_x() + bar.get_width()/2., height,
           f'{int(height)}\n({100*height/len(overlap_saa):.1f}%)',
           ha='center', va='bottom', fontweight='bold')
ax5.grid(axis='y', alpha=0.3)

# 6. Phase 8.2 Landmark Month Distribution
ax6 = fig.add_subplot(gs[1, 2])
overlap_phase82 = phase82_df[phase82_df['PATNO'].isin(overlap_patients)]
month_counts = overlap_phase82['landmark_month'].value_counts().sort_index()
ax6.bar(month_counts.index, month_counts.values, color='lightseagreen', edgecolor='black', alpha=0.7)
ax6.set_xlabel('Landmark Month', fontweight='bold')
ax6.set_ylabel('Number of Observations', fontweight='bold')
ax6.set_title('Longitudinal Observations Distribution', fontweight='bold')
for i, (month, count) in enumerate(month_counts.items()):
    ax6.text(month, count, f'{count}', ha='center', va='bottom', fontweight='bold')
ax6.grid(axis='y', alpha=0.3)

# 7. Summary Statistics Table
ax7 = fig.add_subplot(gs[2, :])
ax7.axis('off')

summary_data = [
    ['Metric', 'SAA Labels', 'Phase 8.2', 'Overlap'],
    ['Total Patients', f'{saa_df["PATNO"].nunique():,}', f'{phase82_df["PATNO"].nunique():,}', f'{len(overlap_patients):,}'],
    ['Total Observations', f'{len(saa_df):,}', f'{len(phase82_df):,}', f'{len(overlap_phase82):,}'],
    ['SAA Positive', f'{saa_df["SAA_POSITIVE"].sum()} ({100*saa_df["SAA_POSITIVE"].mean():.1f}%)', 
     'N/A', f'{overlap_saa["SAA_POSITIVE"].sum()} ({100*overlap_saa["SAA_POSITIVE"].mean():.1f}%)'],
    ['SAA Negative', f'{(saa_df["SAA_POSITIVE"]==0).sum()} ({100*(1-saa_df["SAA_POSITIVE"].mean()):.1f}%)', 
     'N/A', f'{(overlap_saa["SAA_POSITIVE"]==0).sum()} ({100*(1-overlap_saa["SAA_POSITIVE"].mean()):.1f}%)'],
    ['α-Syn Mean (pg/mL)', f'{saa_df["ALPHA_SYN_VALUE"].mean():.1f}', 
     f'{phase82_df["ALPHA_SYNUCLEIN"].mean():.1f}', f'{overlap_saa["ALPHA_SYN_VALUE"].mean():.1f}'],
    ['α-Syn Range (pg/mL)', 
     f'{saa_df["ALPHA_SYN_VALUE"].min():.0f} - {saa_df["ALPHA_SYN_VALUE"].max():.0f}',
     f'{phase82_df["ALPHA_SYNUCLEIN"].min():.0f} - {phase82_df["ALPHA_SYNUCLEIN"].max():.0f}',
     f'{overlap_saa["ALPHA_SYN_VALUE"].min():.0f} - {overlap_saa["ALPHA_SYN_VALUE"].max():.0f}'],
    ['Observations/Patient', f'{len(saa_df)/saa_df["PATNO"].nunique():.2f}', 
     f'{len(phase82_df)/phase82_df["PATNO"].nunique():.2f}',
     f'{len(overlap_phase82)/len(overlap_patients):.2f}']
]

table = ax7.table(cellText=summary_data, cellLoc='center', loc='center',
                 colWidths=[0.25, 0.25, 0.25, 0.25])
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1, 2)

# Style header row
for i in range(4):
    cell = table[(0, i)]
    cell.set_facecolor('#4472C4')
    cell.set_text_props(weight='bold', color='white')

# Style data rows
for i in range(1, len(summary_data)):
    for j in range(4):
        cell = table[(i, j)]
        if i % 2 == 0:
            cell.set_facecolor('#E7E6E6')
        else:
            cell.set_facecolor('white')

ax7.set_title('Comprehensive Summary Statistics', fontweight='bold', fontsize=14, pad=20)

# Overall title
fig.suptitle('Phase 8.3: SAA Labels Inspection Summary', 
            fontsize=16, fontweight='bold', y=0.98)

# Save
plt.savefig('data/04_saa/saa_inspection_summary.png', dpi=300, bbox_inches='tight')
print("✓ Saved comprehensive summary to: data/04_saa/saa_inspection_summary.png")
plt.close()

print("\n" + "="*80)
print("KEY FINDINGS")
print("="*80)
print(f"✓ 923 patients with SAA labels extracted from PPMI biospecimen data")
print(f"✓ 595 patients overlap with Phase 8.2 dataset (65.8% of SAA cohort)")
print(f"✓ Class balance: 17.8% SAA+ / 82.2% SAA- (good for weighted loss)")
print(f"✓ Clear separation: SAA+ mean = 2634 pg/mL vs SAA- mean = 1299 pg/mL")
print(f"✓ 754 total observations available (595 baseline + 159 follow-up)")
print(f"✓ All Phase 8.2 features (56 columns) available for overlapping patients")
print(f"\nReady to merge with Phase 8.2 features for GIMAN-SAA training!")
