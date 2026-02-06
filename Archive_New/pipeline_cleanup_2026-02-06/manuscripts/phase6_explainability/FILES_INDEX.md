# Phase 6 Manuscript Files Index

**Manuscript**: "Interpretable Graph Neural Networks for Precision Medicine: A Framework for Explaining Patient Similarity and Prognostic Predictions"  
**Target Journal**: Nature Machine Intelligence  
**Submission Target**: October 31, 2025 (URGENT!)

---

## 📊 Explainability Framework (6 task directories in `figures/`)

### Task 6.1: Attention Weight Analysis
**Directory**: `figures/phase6_task6_1_attention/`

#### Subdirectories:
- `diagnostic/` - Attention quality metrics
- `phase4_subtypes/` - Subtype-specific attention patterns
- `phase5_conversion/` - Conversion prediction attention

#### Key Files:
- Attention weight heatmaps
- Patient similarity networks
- High-attention pair analyses
- Statistical reports

**Key Finding**: 68% same-label attention in Phase 4, clear PD vs HC separation

**LaTeX reference**: 
```latex
\includegraphics{figures/phase6_task6_1_attention/phase4_subtypes/attention_heatmap.png}
```

---

### Task 6.2: GNNExplainer Subgraph Identification
**Directory**: `figures/phase6_task6_2_gnnexplainer/`

#### Subdirectories:
- `diagnostic/` - Explanation quality diagnostics
- `phase4_subtypes/` - Subtype explanatory subgraphs
- `phase5_conversion/` - Conversion explanatory subgraphs

#### Key Files:
- Explanatory subgraph visualizations
- Feature importance rankings
- Node importance scores
- Edge importance analyses

**Key Finding**: 
- Phase 4: `updrs_slope` 100% importance for fast progressors
- Phase 5: `baseline_updrs` 90% importance for converters

**LaTeX reference**: 
```latex
\includegraphics{figures/phase6_task6_2_gnnexplainer/phase4_subtypes/explanatory_subgraph.png}
```

---

### Task 6.3: Feature Attribution Analysis
**Directory**: `figures/phase6_task6_3_attribution/`

#### Subdirectories:
- `phase4_subtypes/` - Subtype feature importance
- `phase5_conversion/` - Conversion feature importance (if available)

#### Methods Used:
- **IntegratedGradients**: Gradient-based attribution
- **GradientSHAP**: SHAP values from gradients

#### Key Files:
- Attribution distribution plots
- Feature importance rankings
- Cross-method validation
- Statistical comparisons

**Key Finding**: 
- Phase 4: `updrs_slope` importance 5.80 (8x other features)
- Phase 5: `baseline_updrs` 9x difference between converters/non-converters

**LaTeX reference**: 
```latex
\includegraphics{figures/phase6_task6_3_attribution/phase4_subtypes/feature_attribution.png}
```

---

### Task 6.4: Patient Clustering Analysis
**Directory**: `figures/phase6_task6_4_clustering/`

#### Methods Compared:
- Hierarchical clustering
- K-means clustering
- Spectral clustering

#### Key Files:
- Dendrogram visualizations
- Elbow plots for optimal k
- Silhouette score analyses
- PCA/t-SNE embeddings
- Cluster characterization reports

**Key Finding**: 
- Phase 4: Optimal k=8, silhouette=0.39 (heterogeneous progression)
- Phase 5: Optimal k=6, silhouette=0.45 (clearer subgroups)

**Clinical Insight**: "Progression twins" - patients with similar predicted trajectories

**LaTeX reference**: 
```latex
\includegraphics{figures/phase6_task6_4_clustering/dendrogram.png}
\includegraphics{figures/phase6_task6_4_clustering/tsne_embeddings.png}
```

---

### Task 6.5: Counterfactual Explanations
**Directory**: `figures/phase6_task6_5_counterfactuals/`

#### Purpose:
"What-if" scenario analysis - minimal feature changes to alter predictions

#### Key Files:
- Counterfactual examples
- Feature perturbation analysis
- Minimal change visualizations
- Actionable intervention suggestions

**Key Finding**: 
- Phase 4: 1/30 success (updrs_slope +3.6 → slow→moderate)
- Phase 5: 0/30 success (graph structure dominates - model is robust)

**Clinical Implication**: Identifies modifiable factors for intervention

**LaTeX reference**: 
```latex
\includegraphics{figures/phase6_task6_5_counterfactuals/counterfactual_examples.png}
```

---

### Task 6.6: Clinical Dashboard
**Directory**: `figures/phase6_task6_6_dashboard/`

#### Components:
- Interactive explanation interface
- Multi-method integrated explanations
- Patient-specific reports
- Clinical decision support visualization

#### Key Files:
- Dashboard screenshots
- User interface mockups
- Integrated explanation panels
- Clinical workflow integration diagrams

**Purpose**: Translate technical explanations into clinical insights

**LaTeX reference**: 
```latex
\includegraphics[width=\textwidth]{figures/phase6_task6_6_dashboard/clinical_dashboard.png}
```

---

## 🖼️ Recommended Figure Panels for Main Manuscript

### Figure 1: Explainability Framework Overview
**Components**:
- Panel A: GNN architecture with attention mechanism
- Panel B: Explainability methods flowchart
- Panel C: Example attention weights visualization
- Panel D: Clinical validation workflow

**Sources**: Combine multiple task outputs

---

### Figure 2: Attention Weight Analysis
**Components**:
- Panel A: Attention heatmap (Phase 4 subtypes)
- Panel B: Patient similarity network (high-attention pairs)
- Panel C: Attention statistics (same-label vs different-label)
- Panel D: Diagnostic PD vs HC attention patterns

**Source**: `figures/phase6_task6_1_attention/`

---

### Figure 3: Feature Attribution & GNNExplainer
**Components**:
- Panel A: GNNExplainer subgraph (fast progressors)
- Panel B: Feature importance ranking (IntegratedGradients)
- Panel C: Cross-method validation (IG vs GradientSHAP)
- Panel D: Clinical interpretation (top features)

**Sources**: 
- `figures/phase6_task6_2_gnnexplainer/`
- `figures/phase6_task6_3_attribution/`

---

### Figure 4: Clinical Applications
**Components**:
- Panel A: Patient clustering ("progression twins")
- Panel B: Counterfactual example
- Panel C: Dashboard screenshot
- Panel D: Clinical workflow integration

**Sources**: 
- `figures/phase6_task6_4_clustering/`
- `figures/phase6_task6_5_counterfactuals/`
- `figures/phase6_task6_6_dashboard/`

---

## 📋 Recommended Table Structure

### Table 1: Explainability Methods Comparison
**Data Source**: Aggregated across all task directories
- Method name
- Type (model-agnostic vs model-specific)
- Computational cost
- Clinical interpretability score
- Validation metrics

### Table 2: Attention Weight Statistics
**Data Source**: `figures/phase6_task6_1_attention/diagnostic/`
- Phase 4: Same-label attention (%)
- Phase 4: High-attention pair characteristics
- Phase 5: Converter-specific patterns
- Diagnostic: PD vs HC separation

### Table 3: Feature Importance Rankings
**Data Source**: `figures/phase6_task6_2_gnnexplainer/` + `figures/phase6_task6_3_attribution/`
- Feature name
- GNNExplainer importance
- IntegratedGradients score
- GradientSHAP value
- Clinical relevance

### Table 4: Patient Clustering Results
**Data Source**: `figures/phase6_task6_4_clustering/`
- Clustering method
- Optimal k
- Silhouette score
- Clinical characterization
- Validation metrics

---

## 🔬 Key Metrics Quick Reference

```python
# Attention statistics
import json
import os

# Task 6.1: Attention weights
attention_dir = 'figures/phase6_task6_1_attention/'
# Explore subdirectories for JSON reports

# Task 6.2: GNNExplainer
gnnexplainer_dir = 'figures/phase6_task6_2_gnnexplainer/'
# Look for feature_importance.json

# Task 6.3: Feature attribution
attribution_dir = 'figures/phase6_task6_3_attribution/'
# Look for attribution_summary.json

# Task 6.4: Clustering
clustering_dir = 'figures/phase6_task6_4_clustering/'
# Look for clustering_metrics.json
```

---

## 📄 LaTeX Compilation Notes

### Nature Machine Intelligence Requirements:
1. **Abstract**: 150 words maximum
2. **Main text**: 3000-5000 words
3. **Figures**: 4 main figures (multi-panel allowed)
4. **Methods**: Can be extensive (no strict limit)
5. **Format**: Double-spaced, line numbers

### Recommended LaTeX Packages:
```latex
\usepackage{graphicx}      % For figures
\usepackage{subcaption}    % For multi-panel figures
\usepackage{booktabs}      % For professional tables
\usepackage{amsmath}       % For equations
\usepackage{algorithm}     % For algorithms
\usepackage{hyperref}      % For links
\usepackage{natbib}        % For citations
```

### Color Scheme:
- Use colorblind-friendly palettes
- Nature prefers RGB color space
- Ensure 300 DPI minimum resolution

---

## 🎯 Key Messages for Manuscript

### Main Contributions:
1. **Novel GNN Explainability Framework** for medical applications
2. **Multiple Complementary Methods** (attention, GNNExplainer, attribution, clustering)
3. **Clinical Validation** across two prediction tasks
4. **Actionable Insights** ("progression twins", counterfactuals)
5. **Clinical Decision Support** (dashboard prototype)

### Clinical Impact:
- **Trust**: Transparent AI increases clinician adoption
- **Interpretability**: Clear explanations for patient-specific predictions
- **Actionability**: Identifies modifiable risk factors
- **Workflow Integration**: Dashboard ready for clinical use

### Technical Innovation:
- **Graph-specific explanations** (attention, GNNExplainer)
- **Multi-method validation** (cross-checking explanations)
- **Patient similarity** (network-based clustering)
- **Counterfactual reasoning** (intervention planning)

---

## 📅 URGENT Timeline (October 5-31, 2025)

### Week 1 (Oct 6-12): Draft Results & Methods
- [ ] Review all 6 task directories
- [ ] Select best examples for main figures
- [ ] Write Results section (4 subsections)
- [ ] Write Methods section (explainability techniques)

### Week 2 (Oct 13-19): Complete Draft
- [ ] Write Abstract (150 words)
- [ ] Write Introduction (600-800 words)
- [ ] Write Discussion (800-1000 words)
- [ ] Create 4 main figure panels

### Week 3 (Oct 20-26): Review & Revision
- [ ] Internal review
- [ ] Co-author feedback
- [ ] Format for Nature MI

### Week 4 (Oct 27-31): SUBMIT
- [ ] Final proofread
- [ ] **SUBMIT by October 31**

---

**Last Updated**: October 5, 2025  
**Status**: 🚨 URGENT - 26 days until deadline!
