#!/usr/bin/env python3
"""Enhanced Dataset Creation for GIMAN v1.1.0
==========================================

Creates enhanced 12-feature dataset by adding genetics and CSF biomarkers
to the current 7-feature baseline while preserving original data integrity.

Features:
- Current 7: Age, Education_Years, MoCA_Score, UPDRS_I_Total, UPDRS_III_Total, Caudate_SBR, Putamen_SBR
- Enhanced +5: LRRK2, GBA, APOE_RISK, ALPHA_SYN, NHY

Author: GIMAN Enhancement Team
Date: September 23, 2025
Version: 1.0.0
"""

import os
import sys
import warnings
from datetime import datetime
from pathlib import Path

import pandas as pd

warnings.filterwarnings("ignore")

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class EnhancedDatasetCreator:
    """Creates enhanced GIMAN dataset with genetics and CSF biomarkers."""

    def __init__(self, data_dir: str = "data", output_dir: str = "data/enhanced"):
        """Initialize enhanced dataset creator.

        Args:
            data_dir: Directory containing original processed data
            output_dir: Directory for enhanced dataset output
        """
        self.data_dir = Path(data_dir)
        self.processed_dir = Path(data_dir) / "01_processed"
        self.raw_dir = Path(data_dir) / "00_raw" / "GIMAN" / "ppmi_data_csv"
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Feature definitions
        self.current_features = [
            "Age",
            "Education_Years",
            "MoCA_Score",
            "UPDRS_I_Total",
            "UPDRS_III_Total",
            "Caudate_SBR",
            "Putamen_SBR",
        ]

        self.enhancement_features = ["LRRK2", "GBA", "APOE_RISK", "ALPHA_SYN", "NHY"]

        self.all_features = self.current_features + self.enhancement_features

        print("🚀 Enhanced Dataset Creator Initialized")
        print(f"📂 Data Directory: {self.data_dir}")
        print(f"📂 Processed Directory: {self.processed_dir}")
        print(f"📂 Raw Directory: {self.raw_dir}")
        print(f"📂 Output Directory: {self.output_dir}")
        print(f"📊 Current Features: {len(self.current_features)}")
        print(f"📊 Enhancement Features: {len(self.enhancement_features)}")
        print(f"📊 Total Features: {len(self.all_features)}")

    def load_original_data(self) -> pd.DataFrame:
        """Load the original processed dataset.

        Returns:
            Original processed dataframe
        """
        print("\n📥 Loading original processed dataset...")

        # Look for processed dataset files in 01_processed directory
        processed_files = [
            "giman_biomarker_imputed_557_patients_v1.csv",
            "giman_imputed_dataset_557_patients.csv",
            "giman_dataset_final.csv",
            "giman_dataset_enhanced.csv",
            "expanded_multimodal_cohort.csv",
        ]

        original_data = None
        for filename in processed_files:
            filepath = self.processed_dir / filename
            if filepath.exists():
                print(f"✅ Found: {filename}")
                original_data = pd.read_csv(filepath)
                print(f"📊 Shape: {original_data.shape}")
                print(f"📊 Columns: {list(original_data.columns)}")
                break

        if original_data is None:
            raise FileNotFoundError(
                f"No processed dataset found in {self.processed_dir}. "
                f"Looking for: {processed_files}"
            )

        return original_data

    def load_raw_source_files(self) -> dict[str, pd.DataFrame]:
        """Load raw source CSV files for enhancement features.

        Returns:
            Dictionary of dataframes from source CSV files
        """
        print("\n📥 Loading raw source files for enhancement...")

        source_files = {
            "demographics": "Demographics_18Sep2025.csv",
            "participant_status": "Participant_Status_18Sep2025.csv",
            "genetics": "iu_genetic_consensus_20250515_18Sep2025.csv",
            "updrs_i": "MDS-UPDRS_Part_I_18Sep2025.csv",
            "datscan": "Xing_Core_Lab_-_Quant_SBR_18Sep2025.csv",
            "csf": "Current_Biospecimen_Analysis_Results_18Sep2025.csv",
            "neuro_exam": "Neurological_Exam_18Sep2025.csv",
        }

        loaded_data = {}
        for key, filename in source_files.items():
            filepath = self.raw_dir / filename
            if filepath.exists():
                print(f"✅ Loading {key}: {filename}")
                df = pd.read_csv(filepath)
                loaded_data[key] = df
                print(f"   Shape: {df.shape}")

                # Show available columns for enhancement features
                if key == "genetics":
                    genetics_cols = [
                        col
                        for col in df.columns
                        if any(gene in col.upper() for gene in ["LRRK2", "GBA", "APOE"])
                    ]
                    print(f"   Genetics columns: {genetics_cols}")
                elif key == "csf":
                    csf_cols = [
                        col
                        for col in df.columns
                        if any(
                            term in col.upper()
                            for term in ["ALPHA", "SYN", "PTAU", "TTAU"]
                        )
                    ]
                    print(f"   CSF columns: {csf_cols}")
                elif key == "neuro_exam":
                    nhy_cols = [
                        col
                        for col in df.columns
                        if any(term in col.upper() for term in ["NHY", "HOEHN", "YAHR"])
                    ]
                    print(f"   NHY columns: {nhy_cols}")

            else:
                print(f"⚠️  Missing {key}: {filename}")

        return loaded_data

    def extract_enhancement_features(
        self, source_data: dict[str, pd.DataFrame]
    ) -> pd.DataFrame:
        """Extract enhancement features from source data.

        Args:
            source_data: Dictionary of source dataframes

        Returns:
            Dataframe with enhancement features
        """
        print("\n🔧 Extracting enhancement features...")

        enhancement_df = pd.DataFrame()

        # Extract genetics features (LRRK2, GBA, APOE_RISK)
        if "genetics" in source_data:
            genetics_df = source_data["genetics"].copy()
            print(f"📊 Genetics data shape: {genetics_df.shape}")

            # Look for genetics columns
            genetic_features = {}
            for feature in ["LRRK2", "GBA", "APOE_RISK"]:
                # Try different column name patterns
                possible_cols = [
                    col for col in genetics_df.columns if feature.lower() in col.lower()
                ]

                if possible_cols:
                    col_name = possible_cols[0]  # Take first match
                    genetic_features[feature] = col_name
                    print(f"   ✅ {feature} -> {col_name}")
                else:
                    print(f"   ⚠️  {feature} not found")

            # Create base enhancement dataframe with PATNO and EVENT_ID
            if "PATNO" in genetics_df.columns:
                enhancement_df = genetics_df[["PATNO", "EVENT_ID"]].copy()

                # Add genetic features
                for feature, col_name in genetic_features.items():
                    enhancement_df[feature] = genetics_df[col_name]
                    coverage = (
                        enhancement_df[feature].notna().sum() / len(enhancement_df)
                    ) * 100
                    print(f"   📈 {feature} coverage: {coverage:.1f}%")

        # Extract ALPHA_SYN from CSF data (if available)
        if "csf" in source_data:
            csf_df = source_data["csf"].copy()
            print(f"📊 CSF data shape: {csf_df.shape}")

            # Look for alpha-synuclein related columns
            alpha_cols = [
                col
                for col in csf_df.columns
                if any(
                    term in col.lower() for term in ["alpha", "syn", "asyn", "a-syn"]
                )
            ]

            if alpha_cols:
                print(f"   Alpha-synuclein candidates: {alpha_cols}")
                # Use first alpha-synuclein column found
                alpha_col = alpha_cols[0]

                # Merge with enhancement_df
                if not enhancement_df.empty:
                    csf_subset = csf_df[["PATNO", "EVENT_ID", alpha_col]].rename(
                        columns={alpha_col: "ALPHA_SYN"}
                    )
                    enhancement_df = enhancement_df.merge(
                        csf_subset, on=["PATNO", "EVENT_ID"], how="left"
                    )
                else:
                    enhancement_df = csf_df[["PATNO", "EVENT_ID", alpha_col]].rename(
                        columns={alpha_col: "ALPHA_SYN"}
                    )

                coverage = (
                    enhancement_df["ALPHA_SYN"].notna().sum() / len(enhancement_df)
                ) * 100
                print(f"   📈 ALPHA_SYN coverage: {coverage:.1f}%")

        # Extract NHY (Hoehn & Yahr) from UPDRS or participant status
        nhy_found = False
        for key in ["updrs_i", "participant_status"]:
            if key in source_data and not nhy_found:
                df = source_data[key]
                nhy_cols = [
                    col
                    for col in df.columns
                    if any(term in col.upper() for term in ["NHY", "HOEHN", "YAHR"])
                ]

                if nhy_cols:
                    nhy_col = nhy_cols[0]
                    print(f"   ✅ NHY found in {key}: {nhy_col}")

                    # Merge NHY data
                    if not enhancement_df.empty:
                        nhy_subset = df[["PATNO", "EVENT_ID", nhy_col]].rename(
                            columns={nhy_col: "NHY"}
                        )
                        enhancement_df = enhancement_df.merge(
                            nhy_subset, on=["PATNO", "EVENT_ID"], how="left"
                        )
                    else:
                        enhancement_df = df[["PATNO", "EVENT_ID", nhy_col]].rename(
                            columns={nhy_col: "NHY"}
                        )

                    coverage = (
                        enhancement_df["NHY"].notna().sum() / len(enhancement_df)
                    ) * 100
                    print(f"   📈 NHY coverage: {coverage:.1f}%")
                    nhy_found = True

        if not nhy_found:
            print("   ⚠️  NHY not found in source data")

        print(f"\n📊 Enhancement features extracted: {enhancement_df.shape}")
        return enhancement_df

    def create_enhanced_dataset(self) -> tuple[pd.DataFrame, dict]:
        """Create the enhanced dataset combining original + enhancement features.

        Returns:
            Enhanced dataframe and metadata dictionary
        """
        print("\n🔄 Creating enhanced dataset...")

        # Load original processed data
        original_df = self.load_original_data()

        # Load raw source files for enhancement
        source_data = self.load_raw_source_files()

        # Extract enhancement features
        enhancement_df = self.extract_enhancement_features(source_data)

        if enhancement_df.empty:
            raise ValueError("No enhancement features could be extracted!")

        # Merge original data with enhancement features
        print("\n🔗 Merging original data with enhancement features...")
        enhanced_df = original_df.merge(
            enhancement_df, on=["PATNO", "EVENT_ID"], how="left"
        )

        print(f"📊 Enhanced dataset shape: {enhanced_df.shape}")

        # Analyze feature coverage
        coverage_stats = {}
        for feature in self.all_features:
            if feature in enhanced_df.columns:
                coverage = (enhanced_df[feature].notna().sum() / len(enhanced_df)) * 100
                coverage_stats[feature] = coverage
                print(f"   📈 {feature}: {coverage:.1f}% coverage")
            else:
                print(f"   ❌ {feature}: Missing from dataset")
                coverage_stats[feature] = 0.0

        # Create metadata
        metadata = {
            "creation_date": datetime.now().isoformat(),
            "original_shape": original_df.shape,
            "enhanced_shape": enhanced_df.shape,
            "current_features": self.current_features,
            "enhancement_features": self.enhancement_features,
            "all_features": self.all_features,
            "feature_coverage": coverage_stats,
            "total_patients": len(enhanced_df["PATNO"].unique()),
            "total_visits": len(enhanced_df),
        }

        return enhanced_df, metadata

    def save_enhanced_dataset(self, enhanced_df: pd.DataFrame, metadata: dict) -> str:
        """Save enhanced dataset and metadata.

        Args:
            enhanced_df: Enhanced dataframe
            metadata: Metadata dictionary

        Returns:
            Path to saved dataset
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Save enhanced dataset
        dataset_filename = f"enhanced_dataset_12features_{timestamp}.csv"
        dataset_path = self.output_dir / dataset_filename
        enhanced_df.to_csv(dataset_path, index=False)

        # Save metadata
        metadata_filename = f"enhanced_dataset_metadata_{timestamp}.json"
        metadata_path = self.output_dir / metadata_filename

        import json

        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

        # Create symlinks for latest versions
        latest_dataset_path = self.output_dir / "enhanced_dataset_latest.csv"
        latest_metadata_path = self.output_dir / "enhanced_metadata_latest.json"

        # Remove existing symlinks if they exist
        if latest_dataset_path.exists():
            latest_dataset_path.unlink()
        if latest_metadata_path.exists():
            latest_metadata_path.unlink()

        # Create new symlinks
        latest_dataset_path.symlink_to(dataset_filename)
        latest_metadata_path.symlink_to(metadata_filename)

        print("\n💾 Enhanced dataset saved:")
        print(f"   📄 Dataset: {dataset_path}")
        print(f"   📄 Metadata: {metadata_path}")
        print(f"   🔗 Latest: {latest_dataset_path}")

        return str(dataset_path)

    def create_feature_comparison_report(
        self, enhanced_df: pd.DataFrame, metadata: dict
    ):
        """Create comparison report between original and enhanced features."""
        report_path = self.output_dir / "feature_enhancement_report.md"

        with open(report_path, "w") as f:
            f.write("# 📊 Feature Enhancement Report\n\n")
            f.write(
                f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
            )

            f.write("## 📈 Dataset Comparison\n\n")
            f.write("| Metric | Original | Enhanced |\n")
            f.write("|--------|----------|----------|\n")
            f.write(
                f"| Rows | {metadata['original_shape'][0]:,} | {metadata['enhanced_shape'][0]:,} |\n"
            )
            f.write(
                f"| Columns | {metadata['original_shape'][1]} | {metadata['enhanced_shape'][1]} |\n"
            )
            f.write(
                f"| Features | {len(metadata['current_features'])} | {len(metadata['all_features'])} |\n\n"
            )

            f.write("## 🧬 Feature Coverage Analysis\n\n")
            f.write("| Feature | Type | Coverage | Status |\n")
            f.write("|---------|------|----------|--------|\n")

            for feature in metadata["all_features"]:
                feature_type = (
                    "Current" if feature in metadata["current_features"] else "Enhanced"
                )
                coverage = metadata["feature_coverage"][feature]
                status = "✅" if coverage > 80 else "⚠️" if coverage > 50 else "❌"
                f.write(
                    f"| {feature} | {feature_type} | {coverage:.1f}% | {status} |\n"
                )

            f.write("\n## 🎯 Enhancement Strategy\n\n")
            f.write("### High Coverage Features (>80%)\n")
            high_coverage = [
                f
                for f in metadata["enhancement_features"]
                if metadata["feature_coverage"][f] > 80
            ]
            for feature in high_coverage:
                f.write(
                    f"- **{feature}**: {metadata['feature_coverage'][feature]:.1f}% coverage\n"
                )

            f.write("\n### Medium Coverage Features (50-80%)\n")
            med_coverage = [
                f
                for f in metadata["enhancement_features"]
                if 50 < metadata["feature_coverage"][f] <= 80
            ]
            for feature in med_coverage:
                f.write(
                    f"- **{feature}**: {metadata['feature_coverage'][feature]:.1f}% coverage\n"
                )

            f.write("\n### Low Coverage Features (<50%)\n")
            low_coverage = [
                f
                for f in metadata["enhancement_features"]
                if metadata["feature_coverage"][f] <= 50
            ]
            for feature in low_coverage:
                f.write(
                    f"- **{feature}**: {metadata['feature_coverage'][feature]:.1f}% coverage\n"
                )

        print(f"📋 Feature enhancement report saved: {report_path}")


def main():
    """Main execution function."""
    print("🚀 GIMAN Enhanced Dataset Creation")
    print("=" * 50)

    # Initialize creator
    creator = EnhancedDatasetCreator()

    try:
        # Create enhanced dataset
        enhanced_df, metadata = creator.create_enhanced_dataset()

        # Save dataset
        dataset_path = creator.save_enhanced_dataset(enhanced_df, metadata)

        # Create comparison report
        creator.create_feature_comparison_report(enhanced_df, metadata)

        print("\n✅ Enhanced dataset creation complete!")
        print(f"📊 Total features: {len(metadata['all_features'])}")
        print(f"📈 Dataset shape: {metadata['enhanced_shape']}")
        print(f"💾 Saved to: {dataset_path}")

        # Display summary
        print("\n📋 Feature Summary:")
        print(f"Current (7): {', '.join(metadata['current_features'])}")
        print(f"Enhanced (+5): {', '.join(metadata['enhancement_features'])}")

        return dataset_path

    except Exception as e:
        print(f"❌ Error creating enhanced dataset: {e}")
        raise


if __name__ == "__main__":
    main()
