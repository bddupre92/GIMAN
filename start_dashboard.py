#!/usr/bin/env python3
"""Start the NSD-ISS Clinical Digital Twin Dashboard.

This script must be run from the project root.
It adds all necessary paths and starts uvicorn.
"""

import os
import sys
from pathlib import Path

# Resolve project root
PROJECT_ROOT = Path(__file__).resolve().parent
DASHBOARD_DIR = PROJECT_ROOT / "clinical_dashboard"

# Add all necessary directories to Python path
sys.path.insert(0, str(DASHBOARD_DIR))
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "GIMImpN_imputation"))

# Change working directory so static files and templates resolve
os.chdir(DASHBOARD_DIR)

import uvicorn

if __name__ == "__main__":
    print(f"\n{'=' * 60}")
    print("  NSD-ISS Clinical Digital Twin Dashboard")
    print(f"  http://127.0.0.1:8000")
    print(f"{'=' * 60}\n")

    uvicorn.run(
        "app.main:app",
        host="127.0.0.1",
        port=8000,
        log_level="info",
    )
