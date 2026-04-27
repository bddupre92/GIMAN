# Phase 6 Explainability Manuscript - Overleaf Setup Guide

**Target Journal**: Nature Machine Intelligence
**Submission Deadline**: October 31, 2025
**Status**: Complete draft ready for compilation

---

## 📁 Manuscript Structure

```
phase6_explainability/
├── main.tex                  # Main manuscript document
├── abstract.tex              # Abstract (150 words)
├── introduction.tex          # Introduction section
├── methods.tex               # Methods section (comprehensive)
├── results.tex               # Results section (6 subsections)
├── discussion.tex            # Discussion section
├── figures.tex               # Figure captions and layout
├── references.bib            # BibTeX references (80+ entries)
├── supplementary.tex         # Supplementary materials
├── figures/                  # Figure directory
│   ├── framework_overview.png
│   ├── combined_attention_analysis.png
│   ├── combined_gnnexplainer_analysis.png
│   ├── combined_attribution_analysis.png
│   ├── combined_clustering_analysis.png
│   ├── combined_counterfactual_dashboard.png
│   └── supplementary/        # Supplementary figures
├── compile.sh                # Linux/Mac compilation script
└── compile.bat               # Windows compilation script
```

---

## 🚀 Quick Start: Upload to Overleaf

### Option 1: Direct Upload (Recommended)

1. **Create New Overleaf Project**:
   - Go to [Overleaf](https://www.overleaf.com)
   - Click "New Project" → "Upload Project"
   - Create a ZIP file of this directory:
     ```bash
     # Navigate to manuscripts directory
     cd "e:\My Drive\CSCI FALL 2025\manuscripts"

     # Create ZIP (Windows)
     powershell Compress-Archive -Path phase6_explainability -DestinationPath phase6_manuscript.zip

     # Or (Mac/Linux)
     zip -r phase6_manuscript.zip phase6_explainability/
     ```
   - Upload `phase6_manuscript.zip` to Overleaf

2. **Set Compiler**:
   - In Overleaf, click Menu → Settings
   - Set "Compiler" to **pdfLaTeX**
   - Set "Main document" to **main.tex**

3. **Compile**:
   - Click "Recompile" button
   - First compile may take 2-3 minutes (generating references)

### Option 2: Git Integration

1. **Initialize Git Repository** (if not already):
   ```bash
   cd phase6_explainability
   git init
   git add .
   git commit -m "Initial manuscript draft"
   ```

2. **Link to Overleaf**:
   - Create new Overleaf project
   - Click Menu → Git → Copy git URL
   - Add Overleaf as remote:
     ```bash
     git remote add overleaf <overleaf-git-url>
     git push overleaf main
     ```

---

## 🔧 Local Compilation

### Prerequisites

- **LaTeX Distribution**:
  - Windows: [MiKTeX](https://miktex.org/download)
  - Mac: [MacTeX](https://www.tug.org/mactex/)
  - Linux: `sudo apt-get install texlive-full`

- **BibTeX** for references (included with above distributions)

### Compilation Steps

#### Windows
```bat
# Double-click compile.bat or run in terminal:
compile.bat
```

#### Mac/Linux
```bash
# Make executable
chmod +x compile.sh

# Run
./compile.sh
```

#### Manual Compilation
```bash
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex  # Second pass resolves references
```

**Output**: `main.pdf` (complete manuscript with references)

---

## 📊 Figures Preparation

### Current Status

Figures are referenced in `figures.tex` but need to be **assembled** from existing results.

### Figure Assembly Tasks

**Figure 1: Framework Overview**
- **Source**: Create composite from:
  - Architecture diagram (to be created from GIMAN model diagram)
  - Six method icons/diagrams
- **Tool**: PowerPoint, Inkscape, or Adobe Illustrator
- **Output**: `figures/framework_overview.png` (300 DPI minimum)

**Figure 2: Attention Analysis**
- **Source**: Combine existing files:
  - `phase6_task6_1_attention/phase4_subtypes/Phase4_Progression_Subtypes_attention_heatmap.png`
  - `phase6_task6_1_attention/phase5_conversion/Phase5_Prodromal_Conversion_attention_heatmap.png`
  - `phase6_task6_1_attention/phase4_subtypes/Phase4_Progression_Subtypes_patient_neighborhoods.png`
- **Tool**: Python (matplotlib subplots) or manual assembly
- **Output**: `figures/combined_attention_analysis.png`

**Figure 3: GNNExplainer**
- **Source**: Combine:
  - `phase6_task6_2_gnnexplainer/phase4_subtypes/Phase4_Progression_Subtypes_subgraph_node_*.png` (3 subgraphs)
  - `phase6_task6_2_gnnexplainer/phase4_subtypes/Phase4_Progression_Subtypes_feature_importance.png`
- **Output**: `figures/combined_gnnexplainer_analysis.png`

**Figure 4: Attribution**
- **Source**: Combine:
  - `phase6_task6_3_attribution/phase4_subtypes/*_distributions.png`
  - `phase6_task6_3_attribution/phase4_subtypes/*_importance_comparison.png`
  - `phase6_task6_3_attribution/phase5_conversion/*_distributions.png`
- **Output**: `figures/combined_attribution_analysis.png`

**Figure 5: Clustering**
- **Source**: Combine:
  - `phase6_task6_4_clustering/phase4_subtypes/Phase4_Progression_Subtypes_dendrogram.png`
  - `phase6_task6_4_clustering/phase4_subtypes/Phase4_Progression_Subtypes_Hierarchical_visualization.png`
  - `phase6_task6_4_clustering/phase5_conversion/Phase5_Prodromal_Conversion_KMeans_visualization.png`
- **Output**: `figures/combined_clustering_analysis.png`

**Figure 6: Counterfactuals & Dashboard**
- **Source**: Combine:
  - `phase6_task6_5_counterfactuals/phase4_subtypes/Phase4_Progression_Subtypes_cf_changes.png`
  - `phase6_task6_6_dashboard/phase4_progression_subtypes/Phase4_Progression_Subtypes_integrated_dashboard.png`
- **Output**: `figures/combined_counterfactual_dashboard.png`

### Python Script for Figure Assembly

Create `assemble_figures.py`:

```python
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from PIL import Image
import os

# Example for Figure 2: Attention Analysis
def create_figure2():
    fig = plt.figure(figsize=(16, 12))
    gs = GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.3)

    # Panel A: Phase 4 attention heatmap
    ax1 = fig.add_subplot(gs[0, 0])
    img1 = Image.open('phase6_task6_1_attention/phase4_subtypes/Phase4_Progression_Subtypes_attention_heatmap.png')
    ax1.imshow(img1)
    ax1.axis('off')
    ax1.set_title('A', fontsize=18, fontweight='bold', loc='left')

    # Panel B: Phase 5 attention heatmap
    ax2 = fig.add_subplot(gs[0, 1])
    img2 = Image.open('phase6_task6_1_attention/phase5_conversion/Phase5_Prodromal_Conversion_attention_heatmap.png')
    ax2.imshow(img2)
    ax2.axis('off')
    ax2.set_title('B', fontsize=18, fontweight='bold', loc='left')

    # Panel C: Patient neighborhoods
    ax3 = fig.add_subplot(gs[1, 0])
    img3 = Image.open('phase6_task6_1_attention/phase4_subtypes/Phase4_Progression_Subtypes_patient_neighborhoods.png')
    ax3.imshow(img3)
    ax3.axis('off')
    ax3.set_title('C', fontsize=18, fontweight='bold', loc='left')

    # Panel D: Clinical similarity (placeholder - create from CSV data)
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.set_title('D', fontsize=18, fontweight='bold', loc='left')
    # Add bar chart code here

    plt.savefig('figures/combined_attention_analysis.png', dpi=300, bbox_inches='tight')
    print("Saved Figure 2: combined_attention_analysis.png")

if __name__ == "__main__":
    os.makedirs('figures', exist_ok=True)
    create_figure2()
    # Repeat for Figures 3-6
```

---

## ✅ Pre-Submission Checklist

### Content Review
- [ ] Abstract: Exactly 150 words (currently 177 - needs trimming)
- [ ] Introduction: References all 6 methods
- [ ] Methods: All equations formatted correctly
- [ ] Results: All 6 subsections complete
- [ ] Discussion: Addresses limitations
- [ ] References: All citations exist in `references.bib`

### Figures
- [ ] Figure 1: Framework overview assembled
- [ ] Figure 2: Attention analysis 4-panel
- [ ] Figure 3: GNNExplainer 4-panel
- [ ] Figure 4: Attribution 4-panel
- [ ] Figure 5: Clustering 4-panel
- [ ] Figure 6: Counterfactual/dashboard 4-panel
- [ ] All figures: 300 DPI minimum
- [ ] All figures: Legible text (font size ≥8pt)
- [ ] Supplementary Figures: Referenced correctly

### Formatting
- [ ] Line numbers added (for review)
- [ ] Author affiliations filled in
- [ ] Competing interests statement complete
- [ ] Author contributions detailed
- [ ] Data availability URLs verified
- [ ] Code repository created (GitHub)

### Nature Machine Intelligence Specific
- [ ] Abstract ≤150 words (**NEEDS TRIMMING: currently 177**)
- [ ] Main text: 3000-5000 words (check current count)
- [ ] References: Nature style (numbered, not author-year)
- [ ] Figures: Maximum 6 main figures (currently 6 ✓)
- [ ] Supplementary: Unlimited (currently comprehensive ✓)

---

## 📝 Word Count

Run in Overleaf or locally:
```bash
# Count words in main sections
texcount main.tex -inc -incbib
```

**Target**: 4000-4500 words (excluding abstract, references, captions)

---

## 🎯 Next Steps (Prioritized for Oct 31 Deadline)

### Week 1 (Oct 6-12): Figure Assembly
1. Create `assemble_figures.py` script
2. Generate all 6 combined figures
3. Create framework overview diagram (Figure 1)
4. Upload figures to Overleaf

### Week 2 (Oct 13-19): Content Refinement
1. Trim abstract to 150 words (remove 27 words)
2. Verify all references compile correctly
3. Add author names and affiliations
4. Create GitHub repository for code
5. Internal review by co-authors

### Week 3 (Oct 20-26): Formatting & Submission Prep
1. Format for Nature Machine Intelligence style
2. Generate supplementary PDF
3. Prepare cover letter
4. Complete submission forms
5. Final proofreading

### Week 4 (Oct 27-31): Submit!
1. Final compile and PDF export
2. Upload to journal submission portal
3. **Submit by October 31, 2025** 🎯

---

## 🆘 Troubleshooting

### Compilation Errors

**Error**: `! LaTeX Error: File 'naturemag.bst' not found`
- **Fix**: Use `plain` style temporarily:
  ```latex
  \bibliographystyle{plain}  % Instead of naturemag
  ```
- **Better Fix**: Download `naturemag.bst` from [Nature website](https://www.nature.com/natmachintell/for-authors/preparing-your-submission)

**Error**: `! Package inputenc Error: Unicode char \u8:XX not set up for use with LaTeX`
- **Fix**: Ensure UTF-8 encoding in Overleaf (Settings → Encoding → UTF-8)

**Error**: Figures not displaying
- **Fix**: Ensure figure paths are correct:
  ```latex
  \includegraphics{figures/combined_attention_analysis.png}  % No leading slash
  ```

### Overleaf Performance

If Overleaf is slow with many figures:
- Compile locally and upload PDF
- Use lower resolution figures during editing (150 DPI), final compile with 300 DPI

---

## 📞 Support

- **LaTeX Questions**: [TeX StackExchange](https://tex.stackexchange.com/)
- **Overleaf Support**: [Overleaf Help](https://www.overleaf.com/learn)
- **Journal Formatting**: [Nature Machine Intelligence Author Guidelines](https://www.nature.com/natmachintell/for-authors)

---

**Last Updated**: October 6, 2025
**Manuscript Status**: Complete draft, pending figure assembly and final review
**Days Until Submission**: 25 days
