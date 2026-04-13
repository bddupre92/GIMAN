#!/usr/bin/env python3
"""Quick test to verify Phase 5 import paths are working."""

import os
import sys

print("🔧 Testing Phase 5 Import Path Resolution...")
print("=" * 50)

try:
    # Test Phase 5 task-specific import
    print("📝 Testing task-specific GIMAN import...")
    sys.path.insert(
        0,
        "/Users/blair.dupre/Library/CloudStorage/GoogleDrive-dupre.blair92@gmail.com/My Drive/CSCI FALL 2025/archive/development/phase5",
    )

    # Test the import path resolution
    phase3_path = os.path.join(
        "/Users/blair.dupre/Library/CloudStorage/GoogleDrive-dupre.blair92@gmail.com/My Drive/CSCI FALL 2025/archive/development",
        "phase3",
    )
    print(f"Phase 3 path: {phase3_path}")
    print(f"Phase 3 exists: {os.path.exists(phase3_path)}")

    phase3_file = os.path.join(phase3_path, "phase3_1_real_data_integration.py")
    print(f"Phase 3 file: {phase3_file}")
    print(f"Phase 3 file exists: {os.path.exists(phase3_file)}")

    sys.path.insert(0, phase3_path)
    from phase3_1_real_data_integration import RealDataPhase3Integration

    print("✅ Phase 3 import successful!")

    # Test Phase 4 import
    phase4_path = os.path.join(
        "/Users/blair.dupre/Library/CloudStorage/GoogleDrive-dupre.blair92@gmail.com/My Drive/CSCI FALL 2025/archive/development",
        "phase4",
    )
    print(f"Phase 4 path: {phase4_path}")
    print(f"Phase 4 exists: {os.path.exists(phase4_path)}")

    phase4_file = os.path.join(phase4_path, "phase4_ultra_regularized_system.py")
    print(f"Phase 4 file: {phase4_file}")
    print(f"Phase 4 file exists: {os.path.exists(phase4_file)}")

    sys.path.insert(0, phase4_path)
    from phase4_ultra_regularized_system import LOOCVEvaluator

    print("✅ Phase 4 import successful!")

    # Test Phase 5 imports
    from phase5_task_specific_giman import (
        TaskSpecificGIMANSystem,
        TaskSpecificLOOCVEvaluator,
    )

    print("✅ Phase 5 task-specific import successful!")

    from phase5_dynamic_loss_system import DynamicLossGIMANSystem, DynamicLossWeighter

    print("✅ Phase 5 dynamic loss import successful!")

    print("\n🎉 ALL IMPORTS SUCCESSFUL!")
    print("Phase 5 is ready for real-data validation!")

except ImportError as e:
    print(f"❌ Import error: {e}")
    import traceback

    traceback.print_exc()

except Exception as e:
    print(f"❌ General error: {e}")
    import traceback

    traceback.print_exc()
