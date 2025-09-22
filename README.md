# GIMAN Preprocessing Pipeline

A standardized, modular pipeline for preprocessing multimodal data from the Parkinson's Progression Markers Initiative (PPMI) to prepare it for the Graph-Informed Multimodal Attention Network (GIMAN) model.

## Project Overview

This project implements a robust data preprocessing pipeline that cleans, merges, and curates multimodal PPMI data into analysis-ready master dataframes. The pipeline handles various data sources including demographics, clinical assessments, imaging features, and genetic information.

## Repository Structure

```
├── src/
│   └── giman_pipeline/          # Main package
│       ├── data_processing/     # PPMI data loading & cleaning
│       ├── models/              # GIMAN model components  
│       ├── training/            # Training pipeline
│       └── evaluation/          # Model evaluation
├── config/                      # YAML configuration files
├── data/                        # Data directories (00_raw, 01_interim, 02_processed)
├── notebooks/                   # Exploratory analysis
├── tests/                       # Unit tests
├── docs/                        # Documentation
├── GIMAN/                       # Raw PPMI data (preserved)
│   └── ppmi_data_csv/          # Raw CSV files
└── HW/                         # Homework assignments (preserved)
```

## Key Data Sources

The pipeline processes several critical PPMI datasets:

- **Demographics**: Baseline patient information
- **Participant Status**: Cohort definitions (PD vs Healthy Control)
- **Clinical Assessments**: MDS-UPDRS Parts I & III scores
- **Structural MRI**: FS7_APARC cortical thickness features
- **DAT-SPECT**: Xing Core Lab Striatal Binding Ratios
- **Genetics**: Consensus genetic markers (LRRK2, GBA, APOE)

All merging operations use `PATNO` (patient ID) and `EVENT_ID` (visit ID) as key columns for longitudinal analysis.

## Installation

### Prerequisites
- Python 3.10+
- Poetry (recommended) or pip

### Setup
```bash
# Clone the repository
git clone <your-repo-url>
cd CSCI-FALL-2025

# Install dependencies with Poetry
poetry install

# Or with pip
pip install -e .
```

## Usage

### Basic Data Preprocessing
```python
from giman_pipeline.data_processing import load_ppmi_data, preprocess_master_df

# Load and merge PPMI data
raw_data = load_ppmi_data("GIMAN/ppmi_data_csv/")
master_df = preprocess_master_df(raw_data)
```

### Running the Pipeline
```bash
# Run the complete preprocessing pipeline
giman-preprocess --config config/preprocessing.yaml

# Run with custom configuration  
giman-preprocess --config-path /path/to/config --config-name custom_config.yaml
```

## Configuration

The pipeline uses YAML configuration files for reproducible experiments:

- `config/data_sources.yaml`: PPMI file mappings and paths
- `config/preprocessing.yaml`: Cleaning and merging parameters  
- `config/model.yaml`: GIMAN model configuration

## Development

### Code Standards
- Follow PEP 8 guidelines (enforced by Ruff)
- Use Google-style docstrings
- Maintain type hints for all functions
- Target Python 3.10+ features

### Testing
```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src/giman_pipeline --cov-report=html
```

### Code Quality
```bash
# Format code
ruff format .

# Lint code  
ruff check .

# Type checking
mypy src/
```

## Data Pipeline Workflow

1. **Load**: Import individual CSV files into pandas DataFrames
2. **Clean**: Preprocess each DataFrame individually  
3. **Merge**: Combine DataFrames using `PATNO` + `EVENT_ID`
4. **Engineer**: Create derived features and handle missing values
5. **Scale**: Normalize numerical features for model input

## Contributing

1. Follow the established coding standards in `.github/instructions/`
2. Write tests for new functionality
3. Update documentation for API changes
4. Use conventional commit messages

## Project Structure Rationale

This project follows the **src layout** pattern to:
- Avoid common Python import issues
- Enable clean packaging and distribution
- Separate volatile exploratory code (notebooks) from stable source code
- Support both development and production deployments

## License

[Add your license information here]

## Acknowledgments

- Parkinson's Progression Markers Initiative (PPMI) for providing the data
- [Add other acknowledgments as appropriate]