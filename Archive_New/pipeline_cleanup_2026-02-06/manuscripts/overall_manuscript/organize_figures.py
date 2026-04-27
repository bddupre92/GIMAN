"""
Organize figures for GIMAN comprehensive manuscript.

This script copies figures from individual manuscript directories 
(phase4, phase5, phase6) into the overall_manuscript/figures/ directory
with proper organization.
"""

import shutil
from pathlib import Path

def setup_figure_directories():
    """Create figure subdirectories."""
    base = Path('figures')
    base.mkdir(exist_ok=True)
    
    subdirs = [
        'preprocessing',
        'graph_construction',
        'architecture',
        'phase4_longitudinal',
        'phase5_prodromal',
        'phase6_explainability'
    ]
    
    for subdir in subdirs:
        (base / subdir).mkdir(exist_ok=True)
    
    print("✓ Created figure directory structure")

def copy_phase4_figures():
    """Copy Phase 4 figures."""
    source = Path('../phase4_longitudinal/figures')
    dest = Path('figures/phase4_longitudinal')
    
    if not source.exists():
        print("⚠ Phase 4 source directory not found")
        return
    
    figures = list(source.glob('*.png'))
    for fig in figures:
        shutil.copy2(fig, dest / fig.name)
        print(f"  ✓ Copied {fig.name}")
    
    print(f"✓ Copied {len(figures)} Phase 4 figures")

def copy_phase5_figures():
    """Copy Phase 5 figures."""
    source = Path('../phase5_prodromal/figures')
    dest = Path('figures/phase5_prodromal')
    
    if not source.exists():
        print("⚠ Phase 5 source directory not found")
        return
    
    figures = list(source.glob('*.png'))
    for fig in figures:
        shutil.copy2(fig, dest / fig.name)
        print(f"  ✓ Copied {fig.name}")
    
    print(f"✓ Copied {len(figures)} Phase 5 figures")

def copy_phase6_figures():
    """Copy Phase 6 assembled figures."""
    source = Path('../phase6_explainability/figures')
    dest = Path('figures/phase6_explainability')
    
    if not source.exists():
        print("⚠ Phase 6 source directory not found")
        return
    
    # Copy assembled combined figures
    combined_figures = [
        'combined_attention_analysis.png',
        'combined_gnnexplainer_analysis.png',
        'combined_attribution_analysis.png',
        'combined_clustering_analysis.png',
        'combined_counterfactual_analysis.png'
    ]
    
    copied = 0
    for fig_name in combined_figures:
        fig_path = source / fig_name
        if fig_path.exists():
            shutil.copy2(fig_path, dest / fig_name)
            print(f"  ✓ Copied {fig_name}")
            copied += 1
        else:
            print(f"  ⚠ Missing: {fig_name}")
    
    print(f"✓ Copied {copied}/5 Phase 6 assembled figures")

def create_figure_index():
    """Create index of all figures."""
    base = Path('figures')
    
    index_content = "# Figure Organization for GIMAN Comprehensive Manuscript\n\n"
    index_content += "This directory contains all figures organized by phase.\n\n"
    
    for subdir in sorted(base.iterdir()):
        if subdir.is_dir():
            figures = list(subdir.glob('*.png')) + list(subdir.glob('*.jpg'))
            index_content += f"## {subdir.name}/\n"
            index_content += f"**Count:** {len(figures)} figures\n\n"
            
            for fig in sorted(figures):
                size_mb = fig.stat().st_size / (1024 * 1024)
                index_content += f"- `{fig.name}` ({size_mb:.2f} MB)\n"
            
            index_content += "\n"
    
    with open('figures/FIGURE_INDEX.md', 'w') as f:
        f.write(index_content)
    
    print("✓ Created figures/FIGURE_INDEX.md")

def create_placeholder_instructions():
    """Create instructions for manual figures."""
    instructions = """# Manual Figure Creation Instructions

The following figures need to be created manually using design tools:

## 1. Preprocessing Flowchart
**File:** `figures/preprocessing/preprocessing_flowchart.png`
**Tool:** PowerPoint, BioRender, or Inkscape
**Content:**
- PPMI data sources (clinical, imaging, genetic)
- Quality control steps
- Missing data imputation
- Feature engineering
- Final datasets (536 PD, 194 prodromal)

**Suggested Layout:** Vertical flowchart with boxes and arrows

---

## 2. Graph Construction Schematic
**File:** `figures/graph_construction/graph_construction_schematic.png`
**Tool:** NetworkX + Matplotlib, or manual illustration
**Content:**
- Patient nodes (color by clinical state)
- k-NN edges
- Example patient neighborhood
- Graph statistics callout

**Suggested Layout:** Network diagram with legend

---

## 3. GIMAN Architecture Diagram
**File:** `figures/architecture/giman_architecture.png`
**Tool:** draw.io, PowerPoint, or BioRender
**Content:**
- Input: Multimodal patient graph
- GAT layers (3 layers, multi-head attention)
- Learned embeddings
- Task-specific heads (Phase 4, Phase 5 outputs)
- Explainability modules (6 methods)

**Suggested Layout:** Horizontal flow with vertical branches

---

## Templates and Examples

See individual manuscript figures for style reference:
- Phase 4: `../phase4_longitudinal/figures/`
- Phase 5: `../phase5_prodromal/figures/`
- Phase 6: `../phase6_explainability/figures/`

## Color Palette (for consistency)

- Primary: #2E86AB (blue)
- Secondary: #A23B72 (purple)
- Accent: #F18F01 (orange)
- Success: #06A77D (green)
- Warning: #E63946 (red)

Use these colors consistently across all manually created figures.
"""
    
    with open('MANUAL_FIGURE_INSTRUCTIONS.md', 'w') as f:
        f.write(instructions)
    
    print("✓ Created MANUAL_FIGURE_INSTRUCTIONS.md")

def main():
    """Main execution."""
    print("\n" + "="*70)
    print("GIMAN COMPREHENSIVE MANUSCRIPT - FIGURE ORGANIZATION")
    print("="*70 + "\n")
    
    print("Step 1: Setting up directory structure...")
    setup_figure_directories()
    print()
    
    print("Step 2: Copying Phase 4 figures...")
    copy_phase4_figures()
    print()
    
    print("Step 3: Copying Phase 5 figures...")
    copy_phase5_figures()
    print()
    
    print("Step 4: Copying Phase 6 figures...")
    copy_phase6_figures()
    print()
    
    print("Step 5: Creating figure index...")
    create_figure_index()
    print()
    
    print("Step 6: Creating manual figure instructions...")
    create_placeholder_instructions()
    print()
    
    print("="*70)
    print("FIGURE ORGANIZATION COMPLETE!")
    print("="*70)
    print("\nNext steps:")
    print("  1. Review figures/FIGURE_INDEX.md")
    print("  2. Create manual figures (see MANUAL_FIGURE_INSTRUCTIONS.md)")
    print("  3. Update figures.tex with proper figure references")
    print()

if __name__ == '__main__':
    main()
