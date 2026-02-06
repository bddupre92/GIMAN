"""
Figure 13: Tipping Point Analysis
Compares crisp (step function) vs fuzzy (sigmoid) classification boundaries
"""
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

project_root = Path(__file__).resolve().parents[3]
output_dir = project_root / "visualizations/publication"
output_dir.mkdir(parents=True, exist_ok=True)

plt.rcParams['figure.dpi'] = 300
plt.rcParams['font.size'] = 10
plt.rcParams['font.family'] = 'sans-serif'

def sigmoid(x, center=0, steepness=1):
    """Smooth sigmoid function"""
    return 1 / (1 + np.exp(-steepness * (x - center)))

def create_tipping_point_plot():
    """Create comparison of crisp vs fuzzy boundaries"""
    
    # Biomarker change (e.g., alpha-synuclein level)
    biomarker = np.linspace(-3, 3, 200)
    
    # Crisp classification (step function at 0)
    crisp_prob = np.where(biomarker >= 0, 1.0, 0.0)
    
    # Fuzzy classification (smooth sigmoid)
    fuzzy_prob = sigmoid(biomarker, center=0, steepness=2)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle('Tipping Point Analysis: Crisp vs Fuzzy Classification Boundaries', 
                 fontsize=14, fontweight='bold', y=0.98)
    
    # === Panel A: Comparison ===
    ax1.plot(biomarker, crisp_prob, '-', linewidth=3, color='#457b9d', 
            label='GIMAN (Crisp - Step Function)', alpha=0.8)
    ax1.plot(biomarker, fuzzy_prob, '-', linewidth=3, color='#e63946', 
            label='Neuro-Fuzzy (Smooth Sigmoid)', alpha=0.8)
    
    # Add uncertainty region
    ax1.fill_between(biomarker, 0.3, 0.7, alpha=0.2, color='gray',
                     label='Uncertainty Region')
    
    ax1.set_xlabel('Biomarker Change (e.g., α-synuclein level)', 
                   fontsize=11, fontweight='bold')
    ax1.set_ylabel('P(Fast Progressor)', fontsize=11, fontweight='bold')
    ax1.set_title('Panel A: Classification Boundary Shape\\n(Response to biomarker changes)', 
                  fontsize=11, fontweight='bold', pad=10)
    ax1.set_xlim(-3, 3)
    ax1.set_ylim(-0.05, 1.1)
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.legend(loc='upper left', fontsize=9, frameon=True, shadow=True)
    
    # Add threshold line
    ax1.axvline(x=0, color='black', linestyle=':', linewidth=2, alpha=0.5)
    ax1.text(0.1, 0.5, 'Decision threshold', rotation=90, 
            fontsize=9, va='center', style='italic')
    
    # Annotations
    ax1.annotate('Abrupt transition\\n(all-or-nothing)',
                xy=(0, 0.5), xytext=(-2, 0.8),
                arrowprops=dict(arrowstyle='->', color='#457b9d', lw=2),
                fontsize=9, fontweight='bold', color='#457b9d',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='white', 
                         edgecolor='#457b9d', linewidth=2, alpha=0.9))
    
    ax1.annotate('Gradual transition\\n(captures uncertainty)',
                xy=(0.7, 0.7), xytext=(1.5, 0.85),
                arrowprops=dict(arrowstyle='->', color='#e63946', lw=2),
                fontsize=9, fontweight='bold', color='#e63946',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='white', 
                         edgecolor='#e63946', linewidth=2, alpha=0.9))
    
    # === Panel B: Minimal Change Required ===
    # Show how much change needed to flip classification
    
    # For crisp: any tiny change crosses threshold (unstable)
    # For fuzzy: gradual change (stable)
    
    start_point = -0.1  # Just below threshold
    changes = np.linspace(0, 0.5, 100)
    
    crisp_response = np.where(start_point + changes >= 0, 1.0, 0.0)
    fuzzy_response = sigmoid(start_point + changes, center=0, steepness=2)
    
    ax2.plot(changes, crisp_response, '-', linewidth=3, color='#457b9d', 
            label='GIMAN (Crisp)', alpha=0.8)
    ax2.plot(changes, fuzzy_response, '-', linewidth=3, color='#e63946', 
            label='Neuro-Fuzzy', alpha=0.8)
    
    ax2.set_xlabel('Biomarker Increase from Baseline', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Change in P(Fast Progressor)', fontsize=11, fontweight='bold')
    ax2.set_title('Panel B: Minimal Perturbation Sensitivity\\n(Starting from borderline case)', 
                  fontsize=11, fontweight='bold', pad=10)
    ax2.set_xlim(0, 0.5)
    ax2.set_ylim(-0.05, 1.1)
    ax2.grid(True, alpha=0.3, linestyle='--')
    ax2.legend(loc='upper left', fontsize=9, frameon=True, shadow=True)
    
    # Annotate key behaviors
    ax2.text(0.15, 0.5, 'Sudden flip at 0.1\\n(unstable)', 
            fontsize=9, fontweight='bold', color='#457b9d',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='white', alpha=0.9,
                     edgecolor='#457b9d', linewidth=1.5))
    
    ax2.text(0.3, 0.65, 'Smooth response\\n(stable to noise)', 
            fontsize=9, fontweight='bold', color='#e63946',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='white', alpha=0.9,
                     edgecolor='#e63946', linewidth=1.5))
    
    plt.tight_layout()
    output_path = output_dir / "Figure13_Tipping_Point.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white', pad_inches=0.2)
    plt.close()
    print(f"✅ Saved Figure 13: {output_path}")
    
    print("\nTipping Point Analysis:")
    print("="*70)
    print("Comparison of classification boundary behaviors:")
    print("\nCrisp (GIMAN):")
    print("  - Step function at threshold")
    print("  - Abrupt 0→1 transition")
    print("  - Unstable to small perturbations")
    print("  - Clinically problematic: patient 'flips' with tiny measurement change")
    print("\nFuzzy (Neuro-Fuzzy GIMAN):")
    print("  - Smooth sigmoid transition")
    print("  - Gradual probability change")
    print("  - Stable to measurement noise")
    print("  - Clinically realistic: captures uncertainty around threshold")

if __name__ == "__main__":
    print("="*80)
    print("FIGURE 13: TIPPING POINT ANALYSIS")
    print("="*80)
    create_tipping_point_plot()
    print("\n✅ Figure 13 complete!")
    print("\n" + "="*80)
    print("🎉 ALL 13 PUBLICATION FIGURES COMPLETE! 🎉")
    print("="*80)
