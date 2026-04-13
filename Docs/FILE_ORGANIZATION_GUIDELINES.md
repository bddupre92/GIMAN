# GIMAN Project File Organization Guidelines

This document establishes the proper file organization structure for the GIMAN project going forward.

## Directory Structure

```
GIMAN/
├── src/                          # Production source code
│   └── giman_pipeline/
│       ├── data_processing/
│       ├── training/
│       ├── imaging/
│       └── quality/
├── tests/                        # All test files
│   ├── __init__.py
│   ├── test_*.py                 # Unit and integration tests
│   └── conftest.py              # pytest configuration
├── scripts/                      # Utility and demonstration scripts
│   ├── standalone_*.py          # Standalone demonstrations
│   ├── demo_*.py                # Workflow demonstrations
│   ├── create_*.py              # Data creation utilities
│   ├── debug_*.py               # Debugging utilities
│   └── phase*.py                # Phase execution scripts
├── notebooks/                    # Jupyter notebooks for research/exploration
├── data/                        # Data directories
├── config/                      # Configuration files
├── Docs/                        # Documentation
└── README.md                    # Project documentation
```

## File Naming Conventions

### Test Files (`tests/`)
- **Pattern**: `test_<module_name>.py`
- **Examples**: 
  - `test_giman_phase1.py` - Phase 1 GIMAN tests
  - `test_data_processing.py` - Data processing tests
  - `test_imaging_processing.py` - Imaging pipeline tests
- **Requirements**: 
  - Must be importable by pytest
  - Use relative imports: `from src.giman_pipeline import ...`
  - Project root path: `Path(__file__).parent.parent`

### Script Files (`scripts/`)
- **Patterns**: 
  - `standalone_*.py` - Independent demonstrations
  - `demo_*.py` - Workflow demonstrations  
  - `create_*.py` - Data creation utilities
  - `debug_*.py` - Debugging and analysis
  - `phase*.py` - Phase execution scripts
- **Examples**:
  - `standalone_imputation_demo.py` - Biomarker imputation demo
  - `demo_complete_workflow.py` - Full PPMI processing demo
  - `create_patient_registry.py` - Patient registry creation
  - `debug_event_id.py` - EVENT_ID debugging
  - `phase2_scale_imaging_conversion.py` - Phase 2 execution
- **Requirements**:
  - Executable from command line
  - Use absolute imports: `from giman_pipeline import ...`
  - Project root path: `Path(__file__).parent.parent`

### Source Files (`src/`)
- **Pattern**: Production code organized by functionality
- **Structure**: Package-based with proper `__init__.py` files
- **Requirements**: Importable by both tests and scripts

## Import Path Guidelines

### For Test Files (tests/)
```python
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.giman_pipeline.training import GIMANDataLoader
```

### For Script Files (scripts/)
```python
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from giman_pipeline.data_processing import BiommarkerImputationPipeline
```

## File Placement Rules

### ✅ Files that belong in `tests/`:
- Unit tests for modules
- Integration tests for workflows
- Test data fixtures
- Test configuration files
- Any file that validates functionality

### ✅ Files that belong in `scripts/`:
- Demonstration workflows
- Data processing utilities
- Debugging and analysis tools
- Phase execution scripts
- Standalone examples
- CLI tools and utilities

### ❌ Files that should NOT be in project root:
- Test files (`test_*.py`)
- Demo scripts (`demo_*.py`)
- Utility scripts (`create_*.py`, `debug_*.py`)
- Standalone examples (`standalone_*.py`)
- Phase execution scripts (`phase*.py`)

## Enforcement

Going forward, ALL new files must follow these conventions:

1. **Test files** → `tests/` directory with proper naming
2. **Scripts/utilities** → `scripts/` directory with descriptive names
3. **Production code** → `src/` directory with package structure
4. **Documentation** → `Docs/` directory or root-level `.md` files

## Migration Completed

The following files have been moved to their proper locations:

### Moved to `tests/`:
- `test_giman_phase1.py`
- `test_giman_real_data.py`
- `test_giman_simplified.py`
- `test_phase2_pipeline.py`
- `test_production_imputation.py`

### Moved to `scripts/`:
- `standalone_imputation_demo.py`
- `demo_complete_workflow.py`
- `create_patient_registry.py`
- `create_ppmi_dcm_manifest.py`
- `debug_event_id.py`
- `phase2_scale_imaging_conversion.py`

All import paths have been updated to work from their new locations.

## Usage Examples

### Running Tests
```bash
# From project root
python -m pytest tests/test_giman_phase1.py
poetry run pytest tests/
```

### Running Scripts
```bash
# From project root  
python scripts/standalone_imputation_demo.py
python scripts/demo_complete_workflow.py
```

This organization ensures clean separation of concerns and makes the project structure clear and maintainable.