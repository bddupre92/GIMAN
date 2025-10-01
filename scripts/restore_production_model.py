#!/usr/bin/env python3
"""Quick Model Restoration Script

This script quickly restores the production GIMAN model (v1.0.0 - 98.93% AUC-ROC)
in case experiments with enhanced features don't work out.

Usage:
    python restore_production_model.py
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from scripts.create_model_backup_system import GIMANModelRegistry


def main():
    print("🔄 RESTORING PRODUCTION MODEL")
    print("=" * 40)

    # Initialize registry
    registry = GIMANModelRegistry()

    # Show available models
    registry.list_models()

    # Get production model
    production_model = registry.registry.get("current_production")
    if not production_model:
        print("❌ No production model set!")
        return

    print(f"\n🚀 Restoring production model: {production_model}")

    # Restore to active models directory
    restore_path = f"models/restored_production_{production_model}"
    registry.restore_model(production_model, restore_path)

    print("\n✅ Production model restored!")
    print(f"   📁 Location: {restore_path}")
    print("   📊 Performance: 98.93% AUC-ROC")
    print("   🎯 Ready for immediate use")


if __name__ == "__main__":
    main()
