from __future__ import annotations

import sys
from pathlib import Path

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root / "src"))

from giman_pipeline.explainability.appendix_figures import generate_appendix_package

if __name__ == "__main__":
    generate_appendix_package(
        output_root=root / "visualizations" / "appendix",
        index_md=root / "Docs" / "audit" / "APPENDIX_EXPLAINABILITY_INDEX.md",
        provenance_json=root / "visualizations" / "appendix" / "provenance.json",
    )
