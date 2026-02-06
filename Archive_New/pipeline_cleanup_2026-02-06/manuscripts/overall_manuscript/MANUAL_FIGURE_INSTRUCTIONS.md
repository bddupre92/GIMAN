# Manual Figure Creation Instructions

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
