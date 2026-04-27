# GIMAN Manuscripts Directory

**Created**: October 5, 2025  
**Purpose**: Organized manuscript preparation with LaTeX templates and source files

---

## 📁 Directory Structure

```
manuscripts/
├── README.md (this file)
├── phase4_longitudinal/          # npj Parkinson's Disease (Jan 2026)
│   ├── FILES_INDEX.md            # Complete file inventory
│   ├── data/                     # 20 result files (CSV, JSON)
│   ├── figures/                  # 6 PNG visualizations
│   ├── supplementary/            # Supplementary materials
│   └── [LaTeX files - coming next]
│
├── phase5_prodromal/             # Lancet Neurology (Feb 2026)
│   ├── FILES_INDEX.md            # Complete file inventory
│   ├── data/                     # 16 result files (CSV, JSON, PTH)
│   ├── figures/                  # 6 PNG visualizations
│   ├── supplementary/            # Supplementary materials
│   └── [LaTeX files - coming next]
│
└── phase6_explainability/        # Nature Machine Intelligence (Oct 2025)
    ├── FILES_INDEX.md            # Complete file inventory
    ├── figures/                  # 6 task directories with visualizations
    │   ├── phase6_task6_1_attention/
    │   ├── phase6_task6_2_gnnexplainer/
    │   ├── phase6_task6_3_attribution/
    │   ├── phase6_task6_4_clustering/
    │   ├── phase6_task6_5_counterfactuals/
    │   └── phase6_task6_6_dashboard/
    ├── data/                     # (will contain aggregated metrics)
    ├── supplementary/            # Supplementary materials
    └── [LaTeX files - coming next]
```

---

## 📊 File Inventory Summary

### Phase 4: Longitudinal Progression Subtypes
- **20 data files** copied from `data/longitudinal_cohort/`
  - 8 CSV files (trajectories, observations, embeddings)
  - 6 JSON files (quality reports, metrics)
  - 6 PNG files (publication-quality figures)
- **Target**: npj Parkinson's Disease
- **Submission**: January 2026
- **Status**: ✅ All files copied and indexed

### Phase 5: Prodromal Transition Prediction
- **16 data files** copied from `data/prodromal_cohort/`
  - 4 CSV files (survival data, biomarkers, stratification)
  - 5 JSON files (model results, thresholds)
  - 6 PNG files (publication-quality figures)
  - 1 PTH file (trained DeepSurv model)
- **Target**: Lancet Neurology
- **Submission**: February 2026
- **Status**: ✅ All files copied and indexed

### Phase 6: GNN Explainability Framework
- **6 task directories** copied from `visualizations/phase6_task6_*/`
  - Task 6.1: Attention weight analysis
  - Task 6.2: GNNExplainer subgraphs
  - Task 6.3: Feature attribution
  - Task 6.4: Patient clustering
  - Task 6.5: Counterfactual explanations
  - Task 6.6: Clinical dashboard
- **Target**: Nature Machine Intelligence
- **Submission**: October 31, 2025 (🚨 URGENT - 26 days!)
- **Status**: ✅ All files copied and indexed

---

## 🎯 Next Steps

### Immediate (Today - October 5, 2025)
- [x] Create manuscript directory structure
- [x] Copy all result files
- [x] Create file inventory indices
- [ ] Create LaTeX templates for all three manuscripts
- [ ] Create compilation scripts

### Week 1 (October 6-12)
- [ ] **Phase 6 PRIORITY**: Begin drafting Results section
- [ ] Select figures for Phase 6 manuscript
- [ ] Draft Phase 6 Methods section

### Week 2 (October 13-19)
- [ ] Complete Phase 6 Abstract, Introduction, Discussion
- [ ] Create Phase 6 figure panels

### Week 3-4 (October 20-31)
- [ ] Phase 6 internal review
- [ ] Format for Nature Machine Intelligence
- [ ] 🎯 **SUBMIT Phase 6 by October 31**

### November-December
- [ ] Phase 4 manuscript preparation
- [ ] Extract metrics into LaTeX tables
- [ ] Submit Phase 4 by January 2026

### January-February
- [ ] Phase 5 manuscript preparation
- [ ] Generate survival curves
- [ ] Submit Phase 5 by February 2026

---

## 📝 LaTeX Templates (Coming Next)

Each manuscript directory will contain:

### Main Files
- `main.tex` - Master document
- `abstract.tex` - Abstract section
- `introduction.tex` - Introduction section
- `methods.tex` - Methods section
- `results.tex` - Results section
- `discussion.tex` - Discussion section
- `references.bib` - Bibliography

### Support Files
- `preamble.tex` - LaTeX packages and settings
- `macros.tex` - Custom commands
- `journal_style.sty` - Journal-specific formatting (if needed)

### Compilation
- `compile.sh` / `compile.bat` - Build script
- `Makefile` - Alternative build system

---

## 🔧 Compilation Instructions (Coming)

### Requirements
- LaTeX distribution (TeX Live, MiKTeX, or MacTeX)
- BibTeX or Biber for references
- Recommended: Overleaf for collaborative editing

### Basic Compilation
```bash
# For each manuscript
cd phase4_longitudinal/
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

### Using Makefile
```bash
cd phase4_longitudinal/
make
```

---

## 📚 Journal-Specific Requirements

### npj Parkinson's Disease (Phase 4)
- Format: Nature style
- Abstract: 250 words max
- Main text: No strict limit (typical 3000-5000 words)
- Figures: 6-8 figures recommended
- References: Nature style (numbered)

### Lancet Neurology (Phase 5)
- Format: Lancet style
- Abstract: 300 words (structured)
- Main text: 3000-5000 words
- Figures: 6 figures maximum
- References: Vancouver style (numbered)

### Nature Machine Intelligence (Phase 6)
- Format: Nature style
- Abstract: 150 words max
- Main text: 3000-5000 words
- Figures: 4 main figures (multi-panel allowed)
- References: Nature style (numbered)

---

## 🔗 Quick Links

### Documentation
- [Phase 4 Files Index](phase4_longitudinal/FILES_INDEX.md)
- [Phase 5 Files Index](phase5_prodromal/FILES_INDEX.md)
- [Phase 6 Files Index](phase6_explainability/FILES_INDEX.md)

### Source Data Locations
- Phase 4 original: `../data/longitudinal_cohort/`
- Phase 5 original: `../data/prodromal_cohort/`
- Phase 6 original: `../visualizations/phase6_task6_*/`

### Planning Documents
- [Publication Readiness Report](../GIMAN_PUBLICATION_READINESS_REPORT.md)
- [Results Quick Reference](../RESULTS_FILES_QUICK_REFERENCE.md)
- [Manuscript Action Plan](../MANUSCRIPT_PREPARATION_ACTION_PLAN.md)

---

## 🎓 Best Practices

### File Organization
1. **Keep originals safe**: All files are copies; originals remain in source directories
2. **Self-contained**: Each manuscript directory is independent
3. **Version control**: Consider using Git for LaTeX files
4. **Backups**: Regular backups of entire manuscripts/ directory

### LaTeX Writing
1. **Modular approach**: Separate files for each section
2. **Comments**: Use `%` for notes and TODOs
3. **Version management**: Use `\newcommand` for values that may change
4. **Figure quality**: Ensure 300 DPI minimum, use vector formats when possible

### Collaboration
1. **Overleaf**: Consider for real-time collaboration
2. **Track changes**: Use `\usepackage{changes}` for version tracking
3. **Comments**: Use `\usepackage{todonotes}` for inline comments
4. **Co-author review**: Share compiled PDFs for feedback

---

## 📊 Progress Tracking

### Phase 4 - Longitudinal Subtypes
- [x] Files copied (20 files)
- [x] Files indexed
- [ ] LaTeX template created
- [ ] Manuscript drafted
- [ ] Internal review complete
- [ ] Submitted to journal

### Phase 5 - Prodromal Prediction
- [x] Files copied (16 files)
- [x] Files indexed
- [ ] LaTeX template created
- [ ] Manuscript drafted
- [ ] Internal review complete
- [ ] Submitted to journal

### Phase 6 - Explainability
- [x] Files copied (6 directories)
- [x] Files indexed
- [ ] LaTeX template created
- [ ] Manuscript drafted
- [ ] Internal review complete
- [ ] Submitted to journal (🚨 DUE: Oct 31, 2025)

---

**Last Updated**: October 5, 2025  
**Status**: Files organized, ready for LaTeX template creation
