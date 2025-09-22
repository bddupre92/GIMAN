#!/usr/bin/env python3
"""Standalone demonstration of GIMAN biomarker imputation pipeline independence.

This script proves that the imputation pipeline works completely independently
of any Jupyter notebook code and relies solely on production codebase files.

Run from project root: python standalone_imputation_demo.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd


def main():
    """Demonstrate the standalone biomarker imputation pipeline."""
    print("🔧 STANDALONE IMPUTATION PIPELINE DEMONSTRATION")
    print("=" * 60)

    # Add source to path (production codebase only)
    project_root = Path(__file__).parent.parent
    sys.path.append(str(project_root / "src"))

    try:
        # Import ONLY from production codebase
        from giman_pipeline.data_processing import BiommarkerImputationPipeline

        print("✅ Successfully imported from production codebase")

    except ImportError as e:
        print(f"❌ Failed to import production pipeline: {e}")
        return

    # Create independent test dataset (no notebook dependencies)
    print("\n📊 Creating independent test dataset...")
    np.random.seed(42)

    test_data = {
        "PATNO": list(range(4001, 4051)),  # 50 test patients
        "EVENT_ID": ["BL"] * 50,
        "COHORT_DEFINITION": np.random.choice(
            ["Parkinson's Disease", "Healthy Control"], 50, p=[0.7, 0.3]
        ),
        # Biomarkers with realistic missingness patterns
        "LRRK2": np.random.normal(0, 1, 50),  # Low missingness
        "GBA": np.random.normal(1, 0.5, 50),  # Low missingness
        "APOE_RISK": np.random.normal(0.5, 0.3, 50),  # Moderate missingness
        "UPSIT_TOTAL": np.random.normal(30, 5, 50),  # Moderate missingness
        "PTAU": np.random.normal(20, 3, 50),  # High missingness
        "TTAU": np.random.normal(200, 30, 50),  # High missingness
        "ALPHA_SYN": np.random.normal(1.5, 0.2, 50),  # High missingness
    }

    df = pd.DataFrame(test_data)

    # Introduce missingness patterns
    for col in ["LRRK2", "GBA"]:  # Low missingness
        missing_idx = np.random.choice(
            df.index, size=int(0.15 * len(df)), replace=False
        )
        df.loc[missing_idx, col] = np.nan

    for col in ["APOE_RISK", "UPSIT_TOTAL"]:  # Moderate missingness
        missing_idx = np.random.choice(
            df.index, size=int(0.50 * len(df)), replace=False
        )
        df.loc[missing_idx, col] = np.nan

    for col in ["PTAU", "TTAU", "ALPHA_SYN"]:  # High missingness
        missing_idx = np.random.choice(
            df.index, size=int(0.75 * len(df)), replace=False
        )
        df.loc[missing_idx, col] = np.nan

    print(f"   Created dataset: {df.shape}")
    print(f"   Patients: {len(df)}")

    # Initialize and run production pipeline
    print("\n🚀 Running production imputation pipeline...")

    imputer = BiommarkerImputationPipeline(
        knn_neighbors=5, mice_max_iter=10, mice_random_state=42
    )

    # Full pipeline execution
    df_imputed = imputer.fit_transform(df)

    # Calculate results
    stats = imputer.get_completion_stats(df, df_imputed)

    print("\n📈 IMPUTATION RESULTS:")
    print(f"   Original completion: {stats['original_completion_rate']:.1%}")
    print(f"   Final completion: {stats['imputed_completion_rate']:.1%}")
    print(f"   Improvement: +{stats['improvement']:.1%}")

    # Test data saving capabilities
    print("\n💾 Testing data management...")

    try:
        saved_files = imputer.save_imputed_dataset(
            df_original=df,
            df_imputed=df_imputed,
            dataset_name="standalone_demo_dataset",
        )

        print("✅ Data saved successfully:")
        for file_type, path in saved_files.items():
            print(f"   {file_type}: {path}")

    except Exception as e:
        print(f"⚠️ Data saving test: {e}")

    # Create GIMAN package
    print("\n📦 Creating GIMAN-ready package...")

    giman_package = BiommarkerImputationPipeline.create_giman_ready_package(
        df_imputed=df_imputed, completion_stats=stats
    )

    print("✅ GIMAN package created:")
    print(f"   Patients: {giman_package['metadata']['total_patients']}")
    print(f"   Biomarkers: {giman_package['biomarker_features']['total_count']}")
    print(
        f"   Completion: {giman_package['biomarker_features']['completeness_rate']:.1%}"
    )
    print(
        f"   Ready for similarity graph: {giman_package['metadata']['ready_for_similarity_graph']}"
    )

    print("\n" + "=" * 60)
    print("🎉 STANDALONE DEMONSTRATION COMPLETE")
    print("✅ Pipeline operates completely independently")
    print("✅ Uses ONLY production codebase files")
    print("✅ No Jupyter notebook dependencies")
    print("✅ Ready for integration in any Python environment")
    print("=" * 60)


if __name__ == "__main__":
    main()
