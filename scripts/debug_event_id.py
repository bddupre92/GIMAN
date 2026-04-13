#!/usr/bin/env python3
"""Debug EVENT_ID Data Type Issues in PPMI Data

This script systematically examines EVENT_ID columns across all PPMI datasets
to understand the data type mismatch and create a standardization strategy.

Priority: CRITICAL - This blocks longitudinal data integration across 7,550 patients
"""

import sys
from pathlib import Path

import pandas as pd

# Add src to path for imports
project_root = Path(__file__).parent.parent
src_path = project_root / "src"
sys.path.append(str(src_path))

from giman_pipeline.data_processing.loaders import load_ppmi_data


def analyze_event_id_patterns():
    """Analyze EVENT_ID patterns across all PPMI datasets."""
    print("🔍 EVENT_ID Data Type Analysis")
    print("=" * 60)

    # Set data directory
    data_root = project_root / "data" / "00_raw" / "GIMAN" / "ppmi_data_csv"

    # Load all CSV datasets
    try:
        print("📚 Loading all PPMI datasets...")
        data = load_ppmi_data(str(data_root))
        print(f"✅ Successfully loaded {len(data)} datasets")
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return

    # Analyze EVENT_ID in each dataset
    event_id_analysis = {}

    for dataset_name, df in data.items():
        print(f"\n📊 Analyzing {dataset_name}:")
        print(f"   Shape: {df.shape}")

        if "EVENT_ID" in df.columns:
            event_col = df["EVENT_ID"]

            analysis = {
                "dtype": str(event_col.dtype),
                "null_count": event_col.isna().sum(),
                "null_percentage": (event_col.isna().sum() / len(df)) * 100,
                "unique_values": sorted([str(v) for v in event_col.dropna().unique()]),
                "total_records": len(df),
                "non_null_records": event_col.notna().sum(),
            }

            event_id_analysis[dataset_name] = analysis

            print(f"   EVENT_ID dtype: {analysis['dtype']}")
            print(
                f"   Null values: {analysis['null_count']} ({analysis['null_percentage']:.1f}%)"
            )
            print(f"   Unique values: {analysis['unique_values']}")

        else:
            print("   ⚠️ No EVENT_ID column found")
            event_id_analysis[dataset_name] = {"status": "missing_column"}

    # Summary and standardization strategy
    print("\n" + "=" * 60)
    print("🎯 EVENT_ID STANDARDIZATION STRATEGY")
    print("=" * 60)

    # Group datasets by EVENT_ID patterns
    object_datasets = []
    float_datasets = []
    missing_datasets = []

    for name, analysis in event_id_analysis.items():
        if "status" in analysis and analysis["status"] == "missing_column":
            missing_datasets.append(name)
        elif analysis["dtype"] == "object":
            object_datasets.append((name, analysis))
        elif "float" in analysis["dtype"]:
            float_datasets.append((name, analysis))

    print(f"\n📋 Object type EVENT_ID datasets: {len(object_datasets)}")
    for name, analysis in object_datasets:
        print(f"   - {name}: {analysis['unique_values']}")

    print(f"\n📋 Float type EVENT_ID datasets: {len(float_datasets)}")
    for name, analysis in float_datasets:
        print(f"   - {name}: {analysis['null_percentage']:.1f}% null")

    print(f"\n📋 Missing EVENT_ID datasets: {len(missing_datasets)}")
    for name in missing_datasets:
        print(f"   - {name}")

    # Create standardization mapping
    print("\n🔧 PROPOSED STANDARDIZATION:")
    print("   1. Convert all EVENT_ID columns to object type")
    print("   2. Standardize visit codes:")
    print("      - 'SC' → 'SCREENING' (for demographics screening)")
    print("      - 'TRANS' → 'TRANSITION' (for demographics transition)")
    print("      - Keep 'BL', 'V01', 'V04', etc. as-is (standard longitudinal)")
    print("      - NaN → 'UNKNOWN' (for imaging data without visit info)")
    print("   3. Update merge logic to handle mixed visit types")

    return event_id_analysis


def test_problematic_merge():
    """Test the specific merge that's failing."""
    print("\n🧪 TESTING PROBLEMATIC MERGE")
    print("=" * 40)

    data_root = project_root / "data" / "00_raw" / "GIMAN" / "ppmi_data_csv"

    try:
        # Load the datasets that are causing issues
        data = load_ppmi_data(str(data_root))

        # Get demographics (object type with SC/TRANS)
        demo_df = data.get("demographics")

        # Get a clinical dataset (object type with BL/V01/V04)
        updrs_df = data.get("mds_updrs_i") or data.get("mds_updrs_iii")

        if demo_df is not None and updrs_df is not None:
            print(f"Demographics EVENT_ID: {demo_df['EVENT_ID'].dtype}")
            print(f"UPDRS EVENT_ID: {updrs_df['EVENT_ID'].dtype}")

            # Try the merge that fails
            try:
                merged = pd.merge(
                    demo_df.head(10),
                    updrs_df.head(10),
                    on=["PATNO", "EVENT_ID"],
                    how="outer",
                )
                print(f"✅ Merge successful: {merged.shape}")

            except Exception as merge_error:
                print(f"❌ Merge failed: {merge_error}")
                print("This confirms the EVENT_ID type mismatch issue!")

        else:
            print("⚠️ Could not load demographics or UPDRS data for testing")

    except Exception as e:
        print(f"❌ Error in merge test: {e}")


if __name__ == "__main__":
    # Run the analysis
    event_id_analysis = analyze_event_id_patterns()

    # Test the problematic merge
    test_problematic_merge()

    print("\n🎯 Next step: Implement EVENT_ID standardization in cleaners.py")
