# Notebooks Directory

## Purpose

This directory contains Jupyter notebooks for exploratory data analysis, prototyping, and research experiments. **Notebooks are for exploration only** and should NOT contain code that is critical for production pipelines.

## Guidelines

### What Belongs Here
- Exploratory data analysis (EDA)
- Data visualization and plotting
- Prototype model experiments 
- Research investigations
- Documentation of findings
- Educational materials (like HW assignments)

### What Does NOT Belong Here
- Production pipeline code
- Critical data processing functions
- Model training pipelines
- Utility functions used by multiple notebooks

### Best Practices

1. **Use descriptive names**: `01_demographics_eda.ipynb`, `02_updrs_analysis.ipynb`
2. **Include date prefixes**: Helps with chronological organization
3. **Clear documentation**: Each notebook should have a markdown cell explaining its purpose
4. **Extract reusable code**: Move useful functions to the `src/` package
5. **Keep notebooks focused**: One analysis theme per notebook
6. **Clean outputs**: Clear outputs before committing to git

### Notebook Naming Convention

```
[number]_[descriptive_name]_[author_initials].ipynb
```

Examples:
- `01_ppmi_data_overview_bd.ipynb`
- `02_cortical_thickness_analysis_bd.ipynb`
- `03_genetic_risk_exploration_bd.ipynb`

## Current Notebooks

- `HW1_S1.ipynb` - Perceptron implementation homework (migrated from HW/)

## Moving Code to Production

When notebook code proves valuable:

1. **Refactor** the code into proper functions with docstrings
2. **Add to appropriate module** in `src/giman_pipeline/`
3. **Write tests** in the `tests/` directory
4. **Update the notebook** to import and use the new functions

## Data Access

Notebooks can access data using relative paths:

```python
# Raw data (via symlink)
df = pd.read_csv("../data/00_raw/Demographics_18Sep2025.csv")

# Or use the pipeline functions
from giman_pipeline.data_processing import load_ppmi_data
data = load_ppmi_data("../GIMAN/ppmi_data_csv/")
```
