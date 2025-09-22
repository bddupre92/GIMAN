# GIMAN Development Environment Setup

This guide covers setting up the development environment for the Graph-Informed Multimodal Attention Network (GIMAN) preprocessing pipeline for PPMI data.

## Table of Contents
- [Prerequisites](#prerequisites)
- [Environment Setup Options](#environment-setup-options)
- [Option 1: Virtual Environment (venv) - Recommended](#option-1-virtual-environment-venv---recommended)
- [Option 2: Poetry Environment](#option-2-poetry-environment)
- [Verify Installation](#verify-installation)
- [Development Workflow](#development-workflow)
- [CLI Usage](#cli-usage)
- [Testing](#testing)
- [Code Quality Tools](#code-quality-tools)
- [Troubleshooting](#troubleshooting)

## Prerequisites

- **Python 3.10+** (tested with Python 3.12.3)
- **Git** for version control
- **macOS/Linux** (Windows users should use WSL)

Check your Python version:
```bash
python --version
# or
python3 --version
```

## Environment Setup Options

The project supports two development environment approaches. We recommend **Option 1 (venv)** for broader compatibility and simplicity.

---

## Option 1: Virtual Environment (venv) - Recommended

### 1. Clone and Navigate to Project
```bash
git clone https://github.com/bddupre92/CSCI-FALL-2025.git
cd CSCI-FALL-2025
```

### 2. Create Virtual Environment
```bash
# Create virtual environment
python -m venv .venv

# Activate virtual environment (macOS/Linux)
source .venv/bin/activate

# On Windows (if not using WSL)
# .venv\Scripts\activate
```

### 3. Install Dependencies
```bash
# Upgrade pip first
pip install --upgrade pip

# Install the package in editable mode
pip install -e .

# Install development dependencies
pip install pytest pytest-cov mypy ruff
```

### 4. Verify Installation
```bash
# Check if CLI is working
giman-preprocess --help
giman-preprocess --version

# Check if packages are installed
pip list | grep giman
```

---

## Option 2: Poetry Environment

### 1. Install Poetry
```bash
# Install Poetry (recommended method)
curl -sSL https://install.python-poetry.org | python3 -

# Or via pip (alternative)
pip install poetry
```

### 2. Configure Poetry (Optional)
```bash
# Configure Poetry to create virtual environments in project directory
poetry config virtualenvs.in-project true
```

### 3. Clone and Setup Project
```bash
git clone https://github.com/bddupre92/CSCI-FALL-2025.git
cd CSCI-FALL-2025

# Install dependencies and create virtual environment
poetry install

# Activate Poetry shell
poetry shell
```

### 4. Verify Installation
```bash
# Check Poetry environment
poetry env info

# Test CLI (within Poetry shell)
giman-preprocess --help
giman-preprocess --version
```

---

## Verify Installation

Regardless of which option you chose, verify your setup:

```bash
# 1. Check Python environment
python --version
which python

# 2. Test CLI functionality
giman-preprocess --help
giman-preprocess --version

# 3. Run basic tests
pytest tests/test_simple.py -v

# 4. Check development tools
ruff --version
pytest --version
mypy --version
```

Expected output:
- Python 3.12.3 (or your Python version)
- CLI help and version information
- Tests passing
- Tool versions displayed

---

## Development Workflow

### Daily Development Setup

**For venv users:**
```bash
cd /path/to/CSCI-FALL-2025
source .venv/bin/activate
```

**For Poetry users:**
```bash
cd /path/to/CSCI-FALL-2025
poetry shell
```

### Deactivate Environment
```bash
# For both venv and Poetry
deactivate
```

---

## CLI Usage

The GIMAN preprocessing pipeline provides a command-line interface for processing PPMI data:

### Basic Usage
```bash
# Show help
giman-preprocess --help

# Show version
giman-preprocess --version

# Basic preprocessing (when you have data)
giman-preprocess --data-dir /path/to/ppmi_data_csv/

# With custom output directory
giman-preprocess --data-dir /path/to/ppmi_data_csv/ --output /path/to/processed_data/

# With configuration file
giman-preprocess --data-dir /path/to/ppmi_data_csv/ --config config/preprocessing.yaml
```

### Expected PPMI Data Structure
When you're ready to process data, organize your PPMI CSV files like this:
```
ppmi_data_csv/
├── Demographics_18Sep2025.csv
├── Participant_Status_18Sep2025.csv
├── MDS-UPDRS_Part_I_18Sep2025.csv
├── MDS-UPDRS_Part_III_18Sep2025.csv
├── FS7_APARC_CTH_18Sep2025.csv
├── Xing_Core_Lab_-_Quant_SBR_18Sep2025.csv
└── iu_genetic_consensus_20250515_18Sep2025.csv
```

---

## Testing

### Run All Tests
```bash
# Run all tests with coverage
pytest --cov=src/ --cov-report=html

# Run specific test file
pytest tests/test_simple.py -v

# Run tests with detailed output
pytest -v --tb=short
```

### Test Discovery
```bash
# See what tests are available
pytest --collect-only
```

---

## Code Quality Tools

### Linting with Ruff
```bash
# Check code style
ruff check src/

# Fix automatically fixable issues
ruff check src/ --fix

# Format code
ruff format src/
```

### Type Checking with MyPy
```bash
# Check types
mypy src/
```

### Pre-commit Quality Check
```bash
# Run all quality checks before committing
ruff check src/ --fix
ruff format src/
mypy src/
pytest
```

---

## Project Structure

```
CSCI-FALL-2025/
├── src/giman_pipeline/          # Main package
│   ├── cli.py                   # Command-line interface
│   ├── data_processing/         # Data processing modules
│   ├── models/                  # Model implementations
│   ├── training/                # Training utilities
│   └── evaluation/              # Evaluation metrics
├── tests/                       # Test suite
├── docs/                        # Documentation
├── config/                      # Configuration files
├── pyproject.toml              # Project configuration
└── .github/workflows/          # CI/CD pipeline
```

---

## Troubleshooting

### Common Issues

#### CLI Command Not Found
```bash
# If giman-preprocess command not found:
# For venv users:
pip install -e .

# For Poetry users:
poetry install
```

#### Python Version Issues
```bash
# Check which Python is being used
which python
python --version

# Make sure you're in the correct environment
# venv: source .venv/bin/activate
# Poetry: poetry shell
```

#### Import Errors
```bash
# Reinstall package in development mode
pip install -e .
# or
poetry install
```

#### Test Failures
```bash
# Run tests with more verbose output
pytest -vvv --tb=long

# Check if environment is activated
echo $VIRTUAL_ENV  # Should show path to .venv or Poetry env
```

### Getting Help

1. **Check if environment is activated**: Look for `(.venv)` or `(CSCI-FALL-2025-py3.12)` in your terminal prompt
2. **Verify package installation**: `pip list | grep giman`
3. **Check Python path**: `which python` should point to your virtual environment
4. **Review logs**: Most commands provide helpful error messages

---

## Next Steps

Once your environment is set up:

1. **Read the PPMI data processing instructions** (see `.github/instructions/ppmi_GIMAN.instructions.md`)
2. **Prepare your PPMI CSV data files** in the expected structure
3. **Run the preprocessing pipeline** when ready:
   ```bash
   giman-preprocess --data-dir /path/to/your/ppmi_data_csv/
   ```

---

## Environment Variables (Optional)

For advanced users, you can set environment variables:

```bash
# Add to your ~/.bashrc or ~/.zshrc
export GIMAN_DATA_DIR="/path/to/your/ppmi_data"
export GIMAN_OUTPUT_DIR="/path/to/output"
export GIMAN_CONFIG="/path/to/config.yaml"
```

---

**Happy coding! 🚀**

For questions or issues, refer to the project's GitHub Issues or the comprehensive instructions in `.github/instructions/`.