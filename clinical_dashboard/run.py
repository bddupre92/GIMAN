#!/usr/bin/env python3
"""NSD-ISS Clinical Digital Twin Dashboard — Entry Point.

Usage:
    python run.py              # Start on default port 8000
    python run.py --port 8080  # Start on custom port
"""

import argparse
import sys
from pathlib import Path

# Ensure project src directories are importable
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "GIMImpN_imputation"))

import uvicorn


def main():
    parser = argparse.ArgumentParser(description="NSD-ISS Clinical Digital Twin Dashboard")
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload for development")
    args = parser.parse_args()

    print(f"\n{'=' * 60}")
    print("  NSD-ISS Clinical Digital Twin Dashboard")
    print(f"  http://{args.host}:{args.port}")
    print(f"{'=' * 60}\n")

    uvicorn.run(
        "app.main:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level="info",
    )


if __name__ == "__main__":
    main()
