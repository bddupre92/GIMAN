"""
Verify that all Phase 5 figures exist and are ready for compilation.

Phase 5 already has all figures as individual PNGs - no assembly needed!
This script just verifies they're all present and creates a summary.
"""

from pathlib import Path

def main():
    """Verify all Phase 5 figures exist."""
    print("\n" + "="*70)
    print("PHASE 5 PRODROMAL MANUSCRIPT - FIGURE VERIFICATION")
    print("="*70 + "\n")
    
    figures_dir = Path('figures')
    
    expected_figures = [
        'prodromal_cohort_characterization.png',
        'cox_model_analysis.png',
        'deepsurv_analysis.png',
        'risk_stratification_dashboard.png',
        'time_varying_biomarkers_analysis.png',
        'biomarker_thresholds_analysis.png'
    ]
    
    print("Checking for required figures...\n")
    
    all_present = True
    for fig in expected_figures:
        fig_path = figures_dir / fig
        if fig_path.exists():
            size_mb = fig_path.stat().st_size / (1024 * 1024)
            print(f"  ✓ {fig} ({size_mb:.2f} MB)")
        else:
            print(f"  ✗ MISSING: {fig}")
            all_present = False
    
    print("\n" + "="*70)
    if all_present:
        print("SUCCESS! All 6 Phase 5 figures are present and ready!")
        print("="*70)
        print("\nPhase 5 manuscript is ready to compile!")
        print("  • All figures already exist as publication-quality PNGs")
        print("  • No assembly required")
        print("  • Run compile.bat (Windows) or compile.sh (Mac/Linux)")
        print("  • Or upload phase5_manuscript_overleaf.zip to Overleaf")
    else:
        print("WARNING: Some figures are missing!")
        print("="*70)
        print("\nPlease check the original results directory:")
        print("  data/prodromal_cohort/")
    print()

if __name__ == '__main__':
    main()
