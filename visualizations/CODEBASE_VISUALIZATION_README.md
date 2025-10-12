# 🧠 GIMAN Codebase Dependency Graph

This directory contains comprehensive visualizations and analysis of the GIMAN (Graph-Informed Multimodal Attention Network) codebase structure.

## 📊 Visualizations

### 1. Static Dependency Graph
![GIMAN Dependency Graph](https://github.com/bddupre92/GIMAN/blob/codegen-artifacts-store/visualizations/codebase_dependency_graph.png?raw=true)

**File**: `codebase_dependency_graph.png`

A high-resolution static visualization showing:
- **Internal modules** (colored by category)
- **External dependencies** (libraries)
- **Connection strength** (node size indicates connectivity)
- **Module relationships** (arrows show import dependencies)

### 2. Interactive Dependency Graph
**File**: `codebase_dependency_graph_interactive.html`

An interactive D3.js-powered visualization with features:
- 🔍 **Zoom & Pan**: Explore the graph in detail
- 🖱️ **Drag nodes**: Rearrange the layout dynamically
- 👁️ **Toggle external dependencies**: Focus on internal structure
- 💡 **Hover tooltips**: See module details
- 🎨 **Color-coded categories**: Easy visual identification

**[View Interactive Graph](https://github.com/bddupre92/GIMAN/blob/codegen-artifacts-store/visualizations/codebase_dependency_graph_interactive.html?raw=true)** (download and open in browser)

## 📈 Codebase Statistics

### Overview
- **Total Internal Modules**: 36
- **External Dependencies**: 37
- **Total Connections**: 409

### Module Categories

| Category | Color | Description |
|----------|-------|-------------|
| 🟢 Data Processing | Light Green | Data loading, preprocessing, cleaning, imaging |
| 🌸 Training | Light Pink | Model training, evaluation, optimization |
| 🔮 Modeling | Plum | Core model architectures, patient similarity |
| 🌟 Interpretability | Khaki | Explainability, GNN explainer |
| 🎨 Other Internal | Moccasin | CLI, utilities, pipeline orchestration |
| 🔵 External | Sky Blue | Third-party libraries and dependencies |

### Most Connected Modules

These are the core modules that have the most dependencies (both importing from and being imported by other modules):

1. **`giman_pipeline.training.create_final_binary_model`** - 47 connections
2. **`giman_pipeline.training.train_giman_complete`** - 46 connections
3. **`giman_pipeline.training.optimize_binary_classifier`** - 45 connections
4. **`giman_pipeline.training.train_giman`** - 45 connections
5. **`giman_pipeline.cli`** - 24 connections
6. **`giman_pipeline.data_processing.explainability_Gemini`** - 22 connections
7. **`giman_pipeline`** - 20 connections
8. **`giman_pipeline.training.evaluator`** - 20 connections

### Key External Dependencies

The codebase relies on these major external libraries:

**Machine Learning & Deep Learning**
- `torch` / `pytorch` - Deep learning framework
- `optuna` - Hyperparameter optimization
- `mlflow` - Experiment tracking

**Data Processing & Analysis**
- `pandas` - Data manipulation
- `numpy` - Numerical computing
- `networkx` - Graph operations

**Medical Imaging**
- `nibabel` - NIfTI file handling
- `SimpleITK` - Medical image processing

**Visualization & Reporting**
- `matplotlib` - Plotting and visualization
- `seaborn` - Statistical visualization

## 🏗️ Architecture Overview

### Core Pipeline Components

```
┌─────────────────────────────────────────────────────┐
│                   CLI Interface                      │
│            (giman_pipeline.cli)                      │
└────────────────────┬────────────────────────────────┘
                     │
          ┌──────────┴──────────┐
          ▼                     ▼
┌──────────────────┐  ┌──────────────────┐
│  Data Processing │  │     Training     │
│                  │  │                  │
│ • Loaders        │  │ • Models         │
│ • Preprocessors  │  │ • Trainer        │
│ • Cleaners       │  │ • Evaluator      │
│ • Imaging        │  │ • Optimization   │
└────────┬─────────┘  └────────┬─────────┘
         │                     │
         └──────────┬──────────┘
                    ▼
          ┌─────────────────┐
          │  Interpretability│
          │                  │
          │ • GNN Explainer  │
          │ • Explainability │
          └──────────────────┘
```

## 📁 Repository Structure

```
GIMAN/
├── src/
│   └── giman_pipeline/
│       ├── data_processing/     # Data handling & preprocessing
│       ├── training/             # Model training & evaluation
│       ├── modeling/             # Core model architectures
│       ├── interpretability/     # Explainability tools
│       ├── evaluation/           # Performance evaluation
│       └── cli.py               # Command-line interface
├── scripts/                     # Utility scripts
├── tests/                       # Unit tests
├── configs/                     # Configuration files
└── visualizations/              # Output visualizations
```

## 🔗 Module Relationships

### Data Processing → Training
The data processing modules feed preprocessed data into training pipelines:
- `loaders.py` → `data_loaders.py`
- `preprocessors.py` → `trainer.py`
- `imaging_loaders.py` → `models.py`

### Training → Interpretability
Trained models are analyzed for explainability:
- `models.py` → `gnn_explainer.py`
- `evaluator.py` → `explainability_Gemini.py`

### Circular Dependencies
The graph analysis reveals minimal circular dependencies, indicating a well-structured codebase with clear separation of concerns.

## 🛠️ How to Use These Visualizations

### For Development
1. **Before adding new modules**: Check where they should fit in the architecture
2. **Refactoring**: Identify highly connected modules that might need decomposition
3. **Dependency management**: Track external library usage

### For Documentation
1. **Onboarding**: Help new developers understand the codebase structure
2. **Architecture reviews**: Visual aid for discussing design decisions
3. **Technical debt**: Identify areas with high coupling

### For Analysis
1. **Code coverage**: Prioritize testing for highly connected modules
2. **Impact analysis**: Understand downstream effects of changes
3. **Performance optimization**: Focus on modules with many dependencies

## 📊 Analysis Report

A detailed JSON report is available at `codebase_analysis_report.json` containing:
- Complete list of all modules
- Full dependency mapping
- Connectivity metrics
- Module statistics

## 🔄 Regenerating Visualizations

To regenerate these visualizations with updated code:

```bash
# Install dependencies
pip install networkx matplotlib

# Run the analysis script (included in this repository)
python scripts/generate_dependency_graph.py

# Interactive version
python scripts/generate_interactive_graph.py
```

## 📝 Notes

- **Node Size**: Larger nodes have more connections (imports + being imported)
- **Edge Direction**: Arrows point from importing module to imported module
- **Colors**: Consistent across both static and interactive visualizations
- **Updates**: Visualizations should be regenerated after significant architectural changes

## 🤝 Contributing

When adding new modules, consider:
1. Minimizing dependencies to keep the graph clean
2. Following existing architectural patterns (visible in the graph)
3. Placing modules in appropriate categories
4. Updating these visualizations after major changes

---

**Generated**: October 2025  
**Tool**: Custom Python analysis using NetworkX, Matplotlib, and D3.js  
**Codebase**: GIMAN v1.0+