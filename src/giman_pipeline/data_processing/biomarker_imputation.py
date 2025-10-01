"""Biomarker imputation module for GIMAN preprocessing pipeline.

This module implements production-ready imputation strategies for biomarker data
in the PPMI dataset, specifically designed for the GIMAN model preprocessing.

Author: GIMAN Development Team
Date: 2024
"""

import logging
import warnings
from datetime import datetime
from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import IterativeImputer, KNNImputer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)


class BiommarkerImputationPipeline:
    """Production-ready biomarker imputation pipeline for GIMAN preprocessing.

    This class implements a multi-strategy approach for imputing missing biomarker
    values based on missingness patterns:
    - KNN imputation for low-to-moderate missingness (<40%)
    - MICE with RandomForest for moderate-to-high missingness (40-70%)
    - Cohort-based median imputation for very high missingness (>70%)

    Attributes:
        biomarker_columns (List[str]): List of biomarker column names to impute
        knn_imputer (KNNImputer): KNN imputer for low missingness features
        mice_imputer (IterativeImputer): MICE imputer for moderate missingness features
        cohort_medians (Dict[str, float]): Cached cohort median values
        imputation_metadata (Dict): Metadata about imputation process
        is_fitted (bool): Whether the pipeline has been fitted
    """

    def __init__(
        self,
        biomarker_columns: list[str] | None = None,
        knn_neighbors: int = 5,
        mice_max_iter: int = 10,
        mice_random_state: int = 42,
    ):
        """Initialize the biomarker imputation pipeline.

        Args:
            biomarker_columns: List of biomarker columns to impute. If None,
                             will be inferred from data.
            knn_neighbors: Number of neighbors for KNN imputation
            mice_max_iter: Maximum iterations for MICE imputation
            mice_random_state: Random state for reproducibility
        """
        # Default biomarker columns based on PPMI GIMAN analysis
        self.biomarker_columns = biomarker_columns or [
            "LRRK2",
            "GBA",
            "APOE_RISK",
            "UPSIT_TOTAL",
            "PTAU",
            "TTAU",
            "ALPHA_SYN",
        ]

        # Initialize imputers
        self.knn_imputer = KNNImputer(n_neighbors=knn_neighbors)
        self.mice_imputer = IterativeImputer(
            estimator=RandomForestRegressor(
                n_estimators=10, random_state=mice_random_state
            ),
            max_iter=mice_max_iter,
            random_state=mice_random_state,
        )

        # State tracking
        self.cohort_medians = {}
        self.imputation_metadata = {}
        self.is_fitted = False

        logger.info(
            f"Initialized BiommarkerImputationPipeline with {len(self.biomarker_columns)} biomarkers"
        )

    def analyze_missingness(self, df: pd.DataFrame) -> dict[str, float]:
        """Analyze missingness patterns in biomarker data.

        Args:
            df: Input DataFrame containing biomarker data

        Returns:
            Dictionary mapping biomarker names to their missingness percentages
        """
        available_biomarkers = [
            col for col in self.biomarker_columns if col in df.columns
        ]

        missingness = {}
        for biomarker in available_biomarkers:
            missing_pct = (df[biomarker].isna().sum() / len(df)) * 100
            missingness[biomarker] = missing_pct

        logger.info(
            f"Missingness analysis complete for {len(available_biomarkers)} biomarkers"
        )
        return missingness

    def categorize_by_missingness(
        self, missingness: dict[str, float]
    ) -> tuple[list[str], list[str], list[str]]:
        """Categorize biomarkers by missingness level for different imputation strategies.

        Args:
            missingness: Dictionary of biomarker missingness percentages

        Returns:
            Tuple of (low_missing, moderate_missing, high_missing) biomarker lists
        """
        low_missing = []  # <40% missing - KNN imputation
        moderate_missing = []  # 40-70% missing - MICE imputation
        high_missing = []  # >70% missing - Cohort median imputation

        for biomarker, pct in missingness.items():
            if pct < 20:
                low_missing.append(biomarker)
            elif 20 <= pct < 40:
                # Handle intermediate missingness with KNN (20-40% missing)
                low_missing.append(biomarker)
            elif 40 <= pct <= 55:
                moderate_missing.append(biomarker)
            elif 55 < pct <= 70:
                # Handle high-intermediate missingness with MICE (55-70% missing)
                moderate_missing.append(biomarker)
            elif pct > 70:
                high_missing.append(biomarker)

        logger.info(
            f"Categorized biomarkers: {len(low_missing)} low, "
            f"{len(moderate_missing)} moderate, {len(high_missing)} high missingness"
        )

        return low_missing, moderate_missing, high_missing

    def _calculate_cohort_medians(self, df: pd.DataFrame) -> dict[str, float]:
        """Calculate cohort-specific median values for high-missingness imputation.

        Args:
            df: Input DataFrame with COHORT_DEFINITION column

        Returns:
            Dictionary of biomarker medians by cohort
        """
        cohort_medians = {}

        if "COHORT_DEFINITION" not in df.columns:
            logger.warning("COHORT_DEFINITION not found, using overall medians")
            for biomarker in self.biomarker_columns:
                if biomarker in df.columns:
                    cohort_medians[biomarker] = df[biomarker].median()
            return cohort_medians

        # Calculate medians by cohort
        for cohort in df["COHORT_DEFINITION"].unique():
            if pd.isna(cohort):
                continue

            cohort_data = df[df["COHORT_DEFINITION"] == cohort]

            for biomarker in self.biomarker_columns:
                if biomarker in df.columns:
                    median_val = cohort_data[biomarker].median()
                    key = f"{biomarker}_{cohort}"
                    cohort_medians[key] = median_val

        logger.info(
            f"Calculated cohort medians for {len(cohort_medians)} biomarker-cohort pairs"
        )
        return cohort_medians

    def fit(self, df: pd.DataFrame) -> "BiommarkerImputationPipeline":
        """Fit the imputation pipeline on training data.

        Args:
            df: Training DataFrame containing biomarker data

        Returns:
            Self for method chaining
        """
        logger.info("Fitting biomarker imputation pipeline...")

        # Analyze missingness patterns
        missingness = self.analyze_missingness(df)
        low_missing, moderate_missing, high_missing = self.categorize_by_missingness(
            missingness
        )

        # Store metadata
        self.imputation_metadata = {
            "missingness_analysis": missingness,
            "low_missing_biomarkers": low_missing,
            "moderate_missing_biomarkers": moderate_missing,
            "high_missing_biomarkers": high_missing,
        }

        # Fit KNN imputer for low missingness biomarkers
        if low_missing:
            knn_data = df[low_missing].copy()
            self.knn_imputer.fit(knn_data)
            logger.info(
                f"Fitted KNN imputer for {len(low_missing)} low-missingness biomarkers"
            )

        # Fit MICE imputer for moderate missingness biomarkers
        if moderate_missing:
            mice_data = df[moderate_missing].copy()
            self.mice_imputer.fit(mice_data)
            logger.info(
                f"Fitted MICE imputer for {len(moderate_missing)} moderate-missingness biomarkers"
            )

        # Calculate cohort medians for high missingness biomarkers
        if high_missing:
            self.cohort_medians = self._calculate_cohort_medians(df)
            logger.info(
                f"Calculated cohort medians for {len(high_missing)} high-missingness biomarkers"
            )

        self.is_fitted = True
        logger.info("Biomarker imputation pipeline fitting complete")
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply imputation to biomarker data.

        Args:
            df: DataFrame to impute

        Returns:
            DataFrame with imputed biomarker values
        """
        if not self.is_fitted:
            raise ValueError("Pipeline must be fitted before transformation")

        logger.info("Applying biomarker imputation...")

        # Create copy to avoid modifying original data
        df_imputed = df.copy()

        # Get categorized biomarkers
        low_missing = self.imputation_metadata["low_missing_biomarkers"]
        moderate_missing = self.imputation_metadata["moderate_missing_biomarkers"]
        high_missing = self.imputation_metadata["high_missing_biomarkers"]

        # Apply KNN imputation for low missingness
        if low_missing:
            available_low = [col for col in low_missing if col in df_imputed.columns]
            if available_low:
                knn_imputed = self.knn_imputer.transform(df_imputed[available_low])
                df_imputed[available_low] = knn_imputed
                logger.info(
                    f"Applied KNN imputation to {len(available_low)} biomarkers"
                )

        # Apply MICE imputation for moderate missingness
        if moderate_missing:
            available_moderate = [
                col for col in moderate_missing if col in df_imputed.columns
            ]
            if available_moderate:
                mice_imputed = self.mice_imputer.transform(
                    df_imputed[available_moderate]
                )
                df_imputed[available_moderate] = mice_imputed
                logger.info(
                    f"Applied MICE imputation to {len(available_moderate)} biomarkers"
                )

        # Apply cohort median imputation for high missingness
        if high_missing and self.cohort_medians:
            for biomarker in high_missing:
                if biomarker not in df_imputed.columns:
                    continue

                missing_mask = df_imputed[biomarker].isna()

                if "COHORT_DEFINITION" in df_imputed.columns:
                    # Cohort-specific imputation
                    for cohort in df_imputed["COHORT_DEFINITION"].unique():
                        if pd.isna(cohort):
                            continue

                        cohort_mask = (
                            df_imputed["COHORT_DEFINITION"] == cohort
                        ) & missing_mask
                        key = f"{biomarker}_{cohort}"

                        if key in self.cohort_medians and cohort_mask.sum() > 0:
                            df_imputed.loc[cohort_mask, biomarker] = (
                                self.cohort_medians[key]
                            )
                else:
                    # Overall median if no cohort info
                    if biomarker in self.cohort_medians:
                        df_imputed.loc[missing_mask, biomarker] = self.cohort_medians[
                            biomarker
                        ]

            logger.info(
                f"Applied cohort median imputation to {len(high_missing)} biomarkers"
            )

        logger.info("Biomarker imputation complete")
        return df_imputed

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fit the pipeline and transform data in one step.

        Args:
            df: DataFrame to fit and transform

        Returns:
            DataFrame with imputed biomarker values
        """
        return self.fit(df).transform(df)

    def get_completion_stats(
        self, df_original: pd.DataFrame, df_imputed: pd.DataFrame
    ) -> dict:
        """Calculate biomarker profile completion statistics.

        Args:
            df_original: Original DataFrame before imputation
            df_imputed: DataFrame after imputation

        Returns:
            Dictionary containing completion statistics
        """
        available_biomarkers = [
            col for col in self.biomarker_columns if col in df_original.columns
        ]

        # Calculate original completion
        original_complete_profiles = (
            df_original[available_biomarkers].notna().all(axis=1).sum()
        )
        original_completion_rate = original_complete_profiles / len(df_original)

        # Calculate imputed completion
        imputed_complete_profiles = (
            df_imputed[available_biomarkers].notna().all(axis=1).sum()
        )
        imputed_completion_rate = imputed_complete_profiles / len(df_imputed)

        stats = {
            "total_patients": len(df_original),
            "biomarkers_analyzed": len(available_biomarkers),
            "original_complete_profiles": original_complete_profiles,
            "original_completion_rate": original_completion_rate,
            "imputed_complete_profiles": imputed_complete_profiles,
            "imputed_completion_rate": imputed_completion_rate,
            "improvement": imputed_completion_rate - original_completion_rate,
        }

        logger.info(
            f"Completion improved from {original_completion_rate:.1%} to {imputed_completion_rate:.1%}"
        )
        return stats

    def get_imputation_summary(self) -> dict:
        """Get summary of imputation strategies applied.

        Returns:
            Dictionary containing imputation summary
        """
        if not self.is_fitted:
            return {"error": "Pipeline not fitted"}

        return {
            "biomarkers_processed": len(self.biomarker_columns),
            "imputation_strategies": {
                "knn_imputation": self.imputation_metadata["low_missing_biomarkers"],
                "mice_imputation": self.imputation_metadata[
                    "moderate_missing_biomarkers"
                ],
                "cohort_median_imputation": self.imputation_metadata[
                    "high_missing_biomarkers"
                ],
            },
            "missingness_analysis": self.imputation_metadata["missingness_analysis"],
        }

    def save_imputed_dataset(
        self,
        df_original: pd.DataFrame,
        df_imputed: pd.DataFrame,
        output_dir: str | Path = None,
        dataset_name: str = "giman_imputed_dataset",
        include_metadata: bool = True,
    ) -> dict[str, Path]:
        """Save imputed dataset to the 02_processed directory with proper versioning.

        This function saves the imputed dataset without overwriting base data,
        following the data organization principle of keeping raw data intact.

        Args:
            df_original: Original DataFrame before imputation
            df_imputed: DataFrame after imputation
            output_dir: Output directory path (defaults to data/02_processed)
            dataset_name: Base name for the dataset files
            include_metadata: Whether to save metadata alongside dataset

        Returns:
            Dictionary containing paths to saved files
        """
        if not self.is_fitted:
            raise ValueError("Pipeline must be fitted before saving dataset")

        # Set default output directory to 02_processed
        if output_dir is None:
            # Try to find project root and construct path
            current_path = Path.cwd()
            project_root = current_path

            # Look for common project indicators
            while project_root.parent != project_root:
                if any(
                    (project_root / indicator).exists()
                    for indicator in ["pyproject.toml", ".git", "src"]
                ):
                    break
                project_root = project_root.parent

            output_dir = project_root / "data" / "02_processed"
        else:
            output_dir = Path(output_dir)

        # Create output directory if it doesn't exist
        output_dir.mkdir(parents=True, exist_ok=True)

        # Generate timestamp for versioning
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Create filenames
        dataset_filename = f"{dataset_name}_{len(df_imputed)}_patients_{timestamp}.csv"
        metadata_filename = f"{dataset_name}_metadata_{timestamp}.json"

        # File paths
        dataset_path = output_dir / dataset_filename
        metadata_path = output_dir / metadata_filename

        # Save imputed dataset
        logger.info(f"Saving imputed dataset to {dataset_path}")
        df_imputed.to_csv(dataset_path, index=False)

        saved_files = {"dataset": dataset_path}

        # Save metadata if requested
        if include_metadata:
            import json

            # Calculate completion statistics
            completion_stats = self.get_completion_stats(df_original, df_imputed)

            # Create comprehensive metadata
            metadata = {
                "dataset_info": {
                    "name": dataset_name,
                    "timestamp": timestamp,
                    "total_patients": len(df_imputed),
                    "total_features": len(df_imputed.columns),
                    "file_path": str(dataset_path),
                    "file_size_mb": round(
                        dataset_path.stat().st_size / (1024 * 1024), 2
                    ),
                },
                "imputation_pipeline": {
                    "pipeline_version": "1.0",
                    "biomarkers_processed": self.biomarker_columns,
                    "strategies_applied": self.get_imputation_summary()[
                        "imputation_strategies"
                    ],
                },
                "completion_statistics": completion_stats,
                "data_quality": {
                    "original_missing_values": int(
                        df_original[self.biomarker_columns].isna().sum().sum()
                    ),
                    "remaining_missing_values": int(
                        df_imputed[self.biomarker_columns].isna().sum().sum()
                    ),
                    "imputation_success_rate": float(
                        (
                            completion_stats["imputed_completion_rate"]
                            - completion_stats["original_completion_rate"]
                        )
                        / (1 - completion_stats["original_completion_rate"])
                    )
                    if completion_stats["original_completion_rate"] < 1
                    else 1.0,
                },
                "preservation_note": "Original base data preserved in 00_raw and 01_interim directories",
            }

            logger.info(f"Saving metadata to {metadata_path}")
            with open(metadata_path, "w") as f:
                json.dump(metadata, f, indent=2, default=str)

            saved_files["metadata"] = metadata_path

        logger.info(
            f"Successfully saved imputed dataset with {completion_stats['imputed_completion_rate']:.1%} biomarker completeness"
        )
        return saved_files

    def load_imputed_dataset(self, dataset_path: str | Path) -> pd.DataFrame:
        """Load a previously saved imputed dataset.

        Args:
            dataset_path: Path to the imputed dataset CSV file

        Returns:
            DataFrame containing the imputed dataset
        """
        dataset_path = Path(dataset_path)

        if not dataset_path.exists():
            raise FileNotFoundError(f"Imputed dataset not found: {dataset_path}")

        logger.info(f"Loading imputed dataset from {dataset_path}")
        df = pd.read_csv(dataset_path)

        # Validate that expected biomarkers are present
        available_biomarkers = [
            col for col in self.biomarker_columns if col in df.columns
        ]

        if len(available_biomarkers) == 0:
            logger.warning("No expected biomarkers found in loaded dataset")
        else:
            logger.info(
                f"Loaded dataset with {len(available_biomarkers)} biomarkers: {available_biomarkers}"
            )

        return df

    @classmethod
    def create_giman_ready_package(
        cls,
        df_imputed: pd.DataFrame,
        completion_stats: dict,
        output_dir: str | Path = None,
    ) -> dict:
        """Create a GIMAN-ready data package with all necessary components.

        Args:
            df_imputed: Imputed dataset
            completion_stats: Completion statistics from imputation
            output_dir: Directory to save package components

        Returns:
            Dictionary containing the complete GIMAN package
        """
        # Set default output directory
        if output_dir is None:
            current_path = Path.cwd()
            project_root = current_path

            while project_root.parent != project_root:
                if any(
                    (project_root / indicator).exists()
                    for indicator in ["pyproject.toml", ".git", "src"]
                ):
                    break
                project_root = project_root.parent

            output_dir = project_root / "data" / "02_processed"
        else:
            output_dir = Path(output_dir)

        output_dir.mkdir(parents=True, exist_ok=True)

        # Create GIMAN package
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        giman_package = {
            "dataset": df_imputed,
            "metadata": {
                "creation_timestamp": timestamp,
                "total_patients": len(df_imputed),
                "completion_stats": completion_stats,
                "ready_for_similarity_graph": True,
                "data_location": str(output_dir),
                "preservation_status": "Base data preserved in 00_raw and 01_interim",
            },
            "biomarker_features": {
                "available": [
                    col for col in cls().biomarker_columns if col in df_imputed.columns
                ],
                "total_count": len(
                    [
                        col
                        for col in cls().biomarker_columns
                        if col in df_imputed.columns
                    ]
                ),
                "completeness_rate": completion_stats.get("imputed_completion_rate", 0),
            },
            "next_steps": [
                "Reconstruct patient similarity graph with enhanced biomarker features",
                "Implement multimodal attention mechanisms",
                "Train GIMAN model with improved feature representation",
            ],
        }

        logger.info(f"Created GIMAN-ready package with {len(df_imputed)} patients")
        logger.info(
            f"Biomarker completeness: {completion_stats.get('imputed_completion_rate', 0):.1%}"
        )

        return giman_package
