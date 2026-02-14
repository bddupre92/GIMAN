"""Merge Final Training Dataset: Longitudinal Prodromal (36 features).

Creates the final clean training dataset using ONLY prodromal patients
with real PPMI phenoconversion endpoints. No early PD merge.

Historical note: Previous versions merged early PD patients (time_to_event=0,
phenoconverted=1) into training data, inflating C-index to 0.998 via trivial
discrimination. This version removes that contamination entirely.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import IterativeImputer

# Add project root to path
project_root = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(project_root))


def assert_no_early_pd_contamination(df: pd.DataFrame) -> None:
    """Verify dataset contains no early PD contamination.

    Early PD patients have time_to_event=0 AND phenoconverted=1,
    which represents already-diagnosed patients and inflates metrics.
    """
    contaminated = (df["time_to_event"] == 0) & (df["phenoconverted"] == 1)
    n_contaminated = contaminated.sum()
    if n_contaminated > 0:
        raise ValueError(
            f"CONTAMINATION DETECTED: {n_contaminated} rows have "
            f"time_to_event=0 AND phenoconverted=1 (early PD signature). "
            f"These must be removed before training."
        )


def load_longitudinal_prodromal() -> pd.DataFrame:
    """Load longitudinal prodromal cohort with expanded features."""
    longitudinal_path = (
        project_root
        / "data"
        / "03_prodromal"
        / "enhanced_longitudinal"
        / "prodromal_longitudinal_expanded.csv"
    )
    feature_candidates = [
        project_root
        / "data"
        / "03_prodromal"
        / "enhanced"
        / "prodromal_multimodal_features.csv",
        project_root
        / "data"
        / "03_prodromal"
        / "enhanced_36_features"
        / "prodromal_36_features_imputed.csv",
    ]

    # Load longitudinal data
    longitudinal = pd.read_csv(longitudinal_path)
    print(f"✓ Loaded longitudinal prodromal: {longitudinal.shape}")
    print(f"  Observations: {len(longitudinal)}")
    print(
        f"  Events: {longitudinal['phenoconverted'].sum()} ({longitudinal['phenoconverted'].mean() * 100:.1f}%)"
    )

    features_source = None
    features_df = None
    for candidate in feature_candidates:
        if candidate.exists():
            features_source = candidate
            features_df = pd.read_csv(candidate)
            break
    if features_df is None:
        raise FileNotFoundError(
            "No feature matrix found. Expected one of: "
            + ", ".join(str(p) for p in feature_candidates)
        )
    print(f"✓ Loaded feature matrix: {features_df.shape}")
    print(f"  Source: {features_source}")

    # Merge to get 36 features for longitudinal observations
    # Keep landmark-specific time_to_event and phenoconverted from longitudinal
    survival_cols = [
        "PATNO",
        "time_to_event",
        "phenoconverted",
        "landmark_month",
        "original_time",
        "original_event",
    ]
    survival_data = longitudinal[survival_cols].copy()

    # Drop survival cols from imputed, merge on PATNO
    feature_cols = [c for c in features_df.columns if c != "PATNO"]
    features = features_df[["PATNO"] + feature_cols].copy()
    features = features.groupby("PATNO", as_index=False).first()

    merged = survival_data.merge(features, on="PATNO", how="left")
    print(f"✓ Merged longitudinal + features: {merged.shape}")

    missing_frac = merged[feature_cols].isna().mean().mean() if feature_cols else 1.0
    print(f"  Mean feature missingness after merge: {missing_frac * 100:.1f}%")
    if missing_frac >= 0.999:
        raise ValueError(
            "Merged feature matrix is effectively empty (>=99.9% missing). "
            "Likely PATNO mismatch between longitudinal cohort and feature source."
        )

    return merged


## load_early_pd_cohort REMOVED — early PD merge causes data contamination.
## See docstring for historical context.


def apply_advanced_imputation(df: pd.DataFrame) -> pd.DataFrame:
    """Apply MICE imputation to remaining missing values."""
    print("\n" + "=" * 60)
    print("APPLYING MICE IMPUTATION")
    print("=" * 60 + "\n")

    # Identify feature columns (exclude metadata)
    exclude_cols = {
        "PATNO",
        "time_to_event",
        "phenoconverted",
        "landmark_month",
        "original_time",
        "original_event",
        "cohort",
    }
    feature_cols = [c for c in df.columns if c not in exclude_cols]

    print(f"Features to impute: {len(feature_cols)}")

    # Remove columns with no observed values before MICE to avoid shape mismatch.
    all_null_cols = [c for c in feature_cols if df[c].notna().sum() == 0]
    if all_null_cols:
        print(
            f"⚠ Dropping {len(all_null_cols)} all-null feature columns before imputation: "
            f"{all_null_cols[:10]}{'...' if len(all_null_cols) > 10 else ''}"
        )
        df = df.drop(columns=all_null_cols)
        feature_cols = [c for c in feature_cols if c not in all_null_cols]

    if not feature_cols:
        raise ValueError(
            "No usable feature columns remain after dropping all-null columns. "
            "Cannot continue to model training dataset build."
        )

    # Check missing
    missing_before = df[feature_cols].isnull().sum().sum()
    total_vals = df[feature_cols].size
    print(
        f"Missing values: {missing_before} / {total_vals} ({missing_before / total_vals * 100:.1f}%)"
    )

    if missing_before == 0:
        print("✓ No missing values, skipping imputation")
        return df

    # Apply MICE
    imputer = IterativeImputer(
        estimator=RandomForestRegressor(n_estimators=10, random_state=42, max_depth=5),
        max_iter=10,
        random_state=42,
        verbose=1,
    )

    print("\nRunning MICE imputation...")
    imputed_features = imputer.fit_transform(df[feature_cols])

    # Replace features
    df_imputed = df.copy()
    df_imputed[feature_cols] = imputed_features

    # Validate
    missing_after = df_imputed[feature_cols].isnull().sum().sum()
    print("\n✓ Imputation complete")
    print(f"  Missing before: {missing_before}")
    print(f"  Missing after: {missing_after}")
    print(f"  Imputed: {missing_before - missing_after} values")

    return df_imputed


## merge_and_validate REMOVED — no longer merging early PD cohort.


def main() -> None:
    """Create final clean prodromal-only training dataset."""
    print("=" * 60)
    print("TRUE GIMAN: CLEAN PRODROMAL-ONLY DATASET PREPARATION")
    print("=" * 60 + "\n")

    # Load prodromal cohort only
    prodromal = load_longitudinal_prodromal()
    prodromal["cohort"] = "prodromal"

    # CRITICAL: Assert no early PD contamination
    assert_no_early_pd_contamination(prodromal)
    print("PASSED: No early PD contamination detected")

    # Apply imputation to fill any remaining missing values
    dataset = apply_advanced_imputation(prodromal)

    # Final contamination check after imputation
    assert_no_early_pd_contamination(dataset)

    # Validate event rate is in expected prodromal range
    event_rate = dataset["phenoconverted"].mean()
    if event_rate > 0.15:
        print(
            f"WARNING: Event rate {event_rate:.1%} is unusually high for "
            f"prodromal phenoconversion (expected 3-10%). "
            f"Verify endpoint extraction."
        )

    # Save
    output_dir = project_root / "data" / "03_prodromal" / "final_training_dataset"
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / "prodromal_only_clean.csv"
    dataset.to_csv(output_path, index=False)

    print(f"\n{'=' * 60}")
    print("CLEAN DATASET SUMMARY")
    print(f"{'=' * 60}")
    print(f"  Output: {output_path}")
    print(f"  Shape: {dataset.shape}")
    print(f"  Observations: {len(dataset)}")
    print(f"  Patients: {dataset['PATNO'].nunique()}")
    print(f"  Events: {dataset['phenoconverted'].sum()}")
    print(f"  Event rate: {event_rate:.1%}")
    n_features = dataset.shape[1] - 7  # Exclude metadata columns
    print(f"  Features: {n_features}")

    # Save metadata
    metadata = {
        "n_observations": int(len(dataset)),
        "n_patients": int(dataset["PATNO"].nunique()),
        "n_events": int(dataset["phenoconverted"].sum()),
        "event_rate": float(event_rate),
        "n_features": n_features,
        "cohort": "prodromal_only",
        "early_pd_contamination": False,
        "simulated_endpoints": False,
    }

    metadata_path = output_dir / "dataset_metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"  Metadata: {metadata_path}")
    print(f"\n{'=' * 60}")
    print("CLEAN DATASET PREPARATION COMPLETE")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
