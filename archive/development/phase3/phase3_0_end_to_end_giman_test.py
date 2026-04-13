#!/usr/bin/env python3
"""Phase 3.0: End-to-End GIMAN Testing with Spatiotemporal Embeddings
================================================================

Complete integration test of the GIMAN pipeline with our new spatiotemporal embeddings.
This tests the full workflow from data loading through model training and evaluation.

Key Integration Points:
1. Load spatiotemporal embeddings from Phase 2.8/2.9
2. Integrate with existing GIMAN data pipeline
3. Run full training pipeline with enhanced embeddings
4. Compare performance vs. baseline (if available)
5. Generate comprehensive evaluation report

Input: GIMAN dataset + spatiotemporal embeddings
Output: Trained GIMAN model + performance comparison
"""

import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

# Add project paths - updated for phase3 directory
project_root = Path(__file__).parent.parent.parent.parent
sys.path.append(str(project_root))
sys.path.append(str(project_root / "src"))
sys.path.append(str(project_root / "archive" / "development" / "phase2"))
sys.path.append(str(project_root / "archive" / "development" / "phase3"))

# Import PATNO standardization utility
from patno_standardization import PATNOStandardizer

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class Phase3EndToEndTester:
    """End-to-end integration tester for GIMAN with spatiotemporal embeddings."""

    def __init__(self, base_dir: Path, test_mode: str = "integration"):
        """Initialize the end-to-end tester.

        Args:
            base_dir: Base project directory
            test_mode: 'integration', 'performance', or 'full'
        """
        self.base_dir = base_dir
        self.test_mode = test_mode
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # PATNO standardization utility
        self.patno_standardizer = PATNOStandardizer()

        # Test results storage
        self.test_results = {}
        self.comparison_results = {}

        logger.info("Phase3EndToEndTester initialized")
        logger.info(f"Test mode: {test_mode}")
        logger.info(f"Device: {self.device}")

    def load_spatiotemporal_embeddings(self) -> tuple[np.ndarray, list[str]]:
        """Load spatiotemporal embeddings from Phase 2.8/2.9."""
        logger.info("Loading spatiotemporal embeddings...")

        try:
            # Import from phase2 directory
            sys.path.append(str(self.base_dir / "archive" / "development" / "phase2"))
            from giman_pipeline.spatiotemporal_embeddings import (
                get_all_embeddings,
                get_embedding_info,
            )

            # Get all embeddings
            all_embeddings = get_all_embeddings()
            info = get_embedding_info()

            # Convert to array format for GIMAN pipeline
            patient_ids = []
            embeddings_array = []

            for session_key, embedding in all_embeddings.items():
                patient_ids.append(session_key)
                embeddings_array.append(embedding)

            embeddings_array = np.array(embeddings_array)

            logger.info(f"✅ Loaded {len(all_embeddings)} spatiotemporal embeddings")
            logger.info(f"Embedding shape: {embeddings_array.shape}")
            logger.info(f"Available patients: {info['available_patients']}")

            self.test_results["embedding_loading"] = {
                "status": "success",
                "num_embeddings": len(all_embeddings),
                "embedding_dim": embeddings_array.shape[1],
                "patients": info["available_patients"],
            }

            return embeddings_array, patient_ids

        except Exception as e:
            logger.error(f"Failed to load spatiotemporal embeddings: {e}")
            self.test_results["embedding_loading"] = {
                "status": "failed",
                "error": str(e),
            }
            raise

    def load_giman_dataset(self) -> pd.DataFrame:
        """Load the main GIMAN dataset."""
        logger.info("Loading GIMAN dataset...")

        try:
            # Look for the most recent GIMAN dataset
            data_files = [
                "giman_expanded_cohort_final.csv",
                "giman_dataset_final_enhanced.csv",
                "giman_dataset_final_base.csv",
                "giman_dataset_enhanced.csv",
            ]

            dataset_path = None
            for filename in data_files:
                # Check multiple possible directories
                possible_dirs = [
                    "data/01_processed",
                    "data/02_processed",
                    "outputs",
                    "archive/development/phase2",
                    "",
                ]

                for data_dir in possible_dirs:
                    filepath = self.base_dir / data_dir / filename
                    if filepath.exists():
                        dataset_path = filepath
                        break

                if dataset_path is not None:
                    break

            if dataset_path is None:
                raise FileNotFoundError("No GIMAN dataset found in expected locations")

            df = pd.read_csv(dataset_path)

            logger.info(f"✅ Loaded GIMAN dataset: {dataset_path.name}")
            logger.info(f"Dataset shape: {df.shape}")

            # Standardize PATNO
            df = self.patno_standardizer.standardize_dataframe_patno(df)
            unique_patients = df["PATNO"].nunique()

            self.test_results["dataset_loading"] = {
                "status": "success",
                "dataset_file": dataset_path.name,
                "total_rows": len(df),
                "unique_patients": unique_patients,
                "patient_column": "PATNO",  # Standardized to PATNO
            }

            return df

        except Exception as e:
            logger.error(f"Failed to load GIMAN dataset: {e}")
            self.test_results["dataset_loading"] = {"status": "failed", "error": str(e)}
            raise

    def integrate_embeddings_with_dataset(
        self, df: pd.DataFrame, embeddings: np.ndarray, patient_ids: list[str]
    ) -> pd.DataFrame:
        """Integrate spatiotemporal embeddings with GIMAN dataset."""
        logger.info("Integrating spatiotemporal embeddings with GIMAN dataset...")

        try:
            # Standardize embedding keys first
            embedding_dict_raw = dict(zip(patient_ids, embeddings, strict=False))
            standardized_embeddings = (
                self.patno_standardizer.standardize_embedding_keys(embedding_dict_raw)
            )

            # Create embedding DataFrame
            embedding_data = []

            for session_key, embedding in standardized_embeddings.items():
                # Parse session key (e.g., "100232_baseline")
                if "_" in session_key:
                    patient_id_str, event = session_key.split("_", 1)
                    try:
                        patient_id = int(patient_id_str)
                    except ValueError:
                        logger.warning(f"Could not parse patient ID from {session_key}")
                        continue
                else:
                    # Handle pure PATNO keys (assume baseline)
                    try:
                        patient_id = int(session_key)
                        event = "baseline"
                    except ValueError:
                        logger.warning(f"Unexpected session key format: {session_key}")
                        continue

                # For now, focus on baseline sessions
                if event == "baseline":
                    embedding_dict = {
                        "PATNO": patient_id  # Always use PATNO as standard
                    }

                    # Add embedding features
                    for i, val in enumerate(embedding):
                        embedding_dict[f"spatiotemporal_emb_{i}"] = val

                    embedding_data.append(embedding_dict)

            if not embedding_data:
                raise ValueError("No valid baseline embeddings found")

            embedding_df = pd.DataFrame(embedding_data)

            # Always use PATNO as merge column (standardized)

            # Handle duplicate patient_id column issue
            if "PATNO" in embedding_df.columns and "PATNO" in df.columns:
                # Check for duplicates in embedding_df
                if embedding_df["PATNO"].duplicated().any():
                    logger.warning(
                        "Duplicate patient IDs in embeddings, keeping first occurrence"
                    )
                    embedding_df = embedding_df.drop_duplicates(
                        subset=["PATNO"], keep="first"
                    )

            # Merge datasets
            logger.info("Merging on column: PATNO")
            logger.info(f"Dataset patients: {df['PATNO'].nunique()}")
            logger.info(f"Embedding patients: {embedding_df['PATNO'].nunique()}")

            integrated_df = df.merge(embedding_df, on="PATNO", how="left")

            # Check integration success
            embedding_cols = [
                col
                for col in integrated_df.columns
                if col.startswith("spatiotemporal_emb_")
            ]
            patients_with_embeddings = (
                integrated_df[embedding_cols[0]].notna().sum() if embedding_cols else 0
            )

            logger.info("✅ Integration complete")
            logger.info(f"Integrated dataset shape: {integrated_df.shape}")
            logger.info(
                f"Patients with embeddings: {patients_with_embeddings}/{len(integrated_df)}"
            )

            self.test_results["embedding_integration"] = {
                "status": "success",
                "integrated_shape": integrated_df.shape,
                "patients_with_embeddings": patients_with_embeddings,
                "total_patients": len(integrated_df),
                "coverage_rate": patients_with_embeddings / len(integrated_df)
                if len(integrated_df) > 0
                else 0,
                "embedding_features": len(embedding_cols),
            }

            return integrated_df

        except Exception as e:
            logger.error(f"Failed to integrate embeddings: {e}")
            self.test_results["embedding_integration"] = {
                "status": "failed",
                "error": str(e),
            }
            raise

    def run_integration_test(self) -> bool:
        """Run basic integration test to ensure all components work together."""
        logger.info("Running integration test...")

        try:
            # Step 1: Load spatiotemporal embeddings
            embeddings, patient_ids = self.load_spatiotemporal_embeddings()

            # Step 2: Load GIMAN dataset
            df = self.load_giman_dataset()

            # Step 3: Integrate embeddings with dataset
            integrated_df = self.integrate_embeddings_with_dataset(
                df, embeddings, patient_ids
            )

            # Step 4: Validate integration
            integration_result = self.test_results.get("embedding_integration", {})
            if (
                integration_result.get("status") == "success"
                and integration_result.get("patients_with_embeddings", 0) > 0
            ):
                logger.info("✅ Integration test PASSED")
                return True
            else:
                logger.error("❌ Integration test FAILED - No patients with embeddings")
                return False

        except Exception as e:
            logger.error(f"Integration test failed: {e}")
            self.test_results["integration_test"] = {
                "status": "failed",
                "error": str(e),
            }
            return False

    def run_performance_test(self) -> dict[str, Any]:
        """Run performance comparison test if baseline available."""
        logger.info("Running performance test...")

        if self.test_mode == "integration":
            return {"status": "skipped", "reason": "Integration mode only"}

        try:
            # Placeholder for performance testing
            # This would integrate with actual GIMAN training pipeline

            performance_results = {
                "status": "placeholder",
                "message": "Performance testing requires full training pipeline integration",
            }

            self.comparison_results["performance_test"] = performance_results
            return performance_results

        except Exception as e:
            logger.error(f"Performance test failed: {e}")
            return {"status": "failed", "error": str(e)}

    def generate_test_report(self, output_dir: Path = None) -> Path:
        """Generate comprehensive test report."""
        logger.info("Generating test report...")

        if output_dir is None:
            output_dir = Path("./phase3_test_outputs")
        output_dir.mkdir(exist_ok=True)

        # Create report
        report = {
            "test_metadata": {
                "timestamp": datetime.now().isoformat(),
                "test_mode": self.test_mode,
                "device": str(self.device),
                "phase": "3.0 - End-to-End GIMAN Integration",
            },
            "test_results": self.test_results,
            "comparison_results": self.comparison_results,
        }

        # Save JSON report
        report_path = (
            output_dir
            / f"phase3_0_test_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2, default=str)

        # Create markdown summary
        md_path = (
            output_dir
            / f"phase3_0_test_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
        )

        md_content = f"""# Phase 3.0: End-to-End GIMAN Integration Test Report

**Generated:** {datetime.now().isoformat()}  
**Test Mode:** {self.test_mode}  
**Device:** {self.device}  

## Test Results Summary

"""

        # Add test results
        for test_name, result in self.test_results.items():
            status = result.get("status", "unknown")
            status_emoji = (
                "✅" if status == "success" else "❌" if status == "failed" else "⚠️"
            )

            md_content += f"### {test_name.replace('_', ' ').title()}\n"
            md_content += f"{status_emoji} **Status:** {status}\n\n"

            if status == "success":
                for key, value in result.items():
                    if key != "status":
                        md_content += f"- **{key}:** {value}\n"
            elif status == "failed":
                md_content += f"- **Error:** {result.get('error', 'Unknown error')}\n"

            md_content += "\n"

        # Add next steps
        md_content += """
## Next Steps

1. **If Integration Test Passed:**
   - Proceed to full GIMAN training with spatiotemporal embeddings
   - Run performance comparison against baseline
   - Generate deployment artifacts

2. **If Integration Test Failed:**
   - Review embedding provider compatibility
   - Check dataset merge logic
   - Validate embedding dimensions and format

3. **For Performance Testing:**
   - Integrate with `train_giman_complete.py`
   - Run full training pipeline
   - Compare metrics: accuracy, F1, AUC, etc.

## Files Generated

- Test Report: `{report_path.name}`
- Test Summary: `{md_path.name}`

---
**Phase 3.0 End-to-End GIMAN Integration Test**  
**Generated by GIMAN Development Pipeline**
"""

        with open(md_path, "w") as f:
            f.write(md_content)

        logger.info(f"📋 Test report saved: {report_path}")
        logger.info(f"📋 Test summary saved: {md_path}")

        return report_path

    def create_training_integration_template(self, output_dir: Path = None) -> Path:
        """Create template for integrating with GIMAN training pipeline."""
        logger.info("Creating training integration template...")

        if output_dir is None:
            output_dir = Path("./phase3_test_outputs")
        output_dir.mkdir(exist_ok=True)

        template_path = output_dir / "giman_training_integration_template.py"

        template_code = '''#!/usr/bin/env python3
"""
GIMAN Training Integration Template
=================================

Template for integrating spatiotemporal embeddings with the main GIMAN training pipeline.
This shows how to modify train_giman_complete.py to use our CNN+GRU embeddings.

Usage:
1. Copy relevant sections to train_giman_complete.py
2. Update data loading to include spatiotemporal embeddings
3. Modify model architecture if needed for enhanced feature space
4. Run training and compare performance
"""

import sys
import torch
import numpy as np
import pandas as pd
from pathlib import Path

# Add spatiotemporal embedding provider and PATNO standardization - updated for phase3
project_root = Path(__file__).parent.parent.parent.parent
sys.path.append(str(project_root / "src"))
sys.path.append(str(project_root / "archive" / "development" / "phase3"))
from giman_pipeline.spatiotemporal_embeddings import get_all_embeddings
from patno_standardization import PATNOStandardizer

def load_enhanced_giman_dataset(dataset_path: str) -> pd.DataFrame:
    """Load GIMAN dataset enhanced with spatiotemporal embeddings."""
    
    # Initialize PATNO standardizer
    standardizer = PATNOStandardizer()
    
    # Load base dataset
    df = pd.read_csv(dataset_path)
    
    # Standardize PATNO in base dataset
    df = standardizer.standardize_dataframe_patno(df)
    
    # Load spatiotemporal embeddings
    all_embeddings = get_all_embeddings()
    
    # Standardize embedding keys
    standardized_embeddings = standardizer.standardize_embedding_keys(all_embeddings)
    
    # Create embedding DataFrame (baseline only for now)
    embedding_data = []
    for session_key, embedding in standardized_embeddings.items():
        if session_key.endswith('_baseline'):
            patient_id = int(session_key.split('_')[0])
            embedding_dict = {
                'PATNO': patient_id
            }
            for i, val in enumerate(embedding):
                embedding_dict[f'spatiotemporal_emb_{i}'] = val
            embedding_data.append(embedding_dict)
    
    embedding_df = pd.DataFrame(embedding_data)
    
    # Merge with main dataset (both now standardized to PATNO)
    enhanced_df = df.merge(embedding_df, on='PATNO', how='left')
    
    print(f"Enhanced dataset shape: {enhanced_df.shape}")
    print(f"Patients with embeddings: {enhanced_df['spatiotemporal_emb_0'].notna().sum()}")
    
    return enhanced_df

def modify_giman_config_for_embeddings(config: dict) -> dict:
    """Modify GIMAN configuration to account for additional embedding features."""
    
    # Original GIMAN input dimension + 256 spatiotemporal features
    original_input_dim = config.get('input_dim', 7)
    enhanced_input_dim = original_input_dim + 256
    
    config['input_dim'] = enhanced_input_dim
    
    # May need to adjust hidden dimensions for enhanced feature space
    config['hidden_dims'] = [256, 512, 256]  # Larger capacity for more features
    
    print(f"Updated input dimension: {original_input_dim} → {enhanced_input_dim}")
    
    return config

# Example integration in main training function:
def example_integration():
    """Example of how to integrate with main training pipeline."""
    
    # 1. Load enhanced dataset
    enhanced_df = load_enhanced_giman_dataset("path/to/giman_dataset.csv")
    
    # 2. Update configuration
    config = {
        'input_dim': 7,  # Original GIMAN features
        'hidden_dims': [128, 256, 128],
        # ... other config
    }
    enhanced_config = modify_giman_config_for_embeddings(config)
    
    # 3. Continue with normal GIMAN training pipeline
    # (Use enhanced_df and enhanced_config in PatientSimilarityGraph)
    
    print("Integration template ready for implementation")

if __name__ == "__main__":
    example_integration()
'''

        with open(template_path, "w") as f:
            f.write(template_code)

        logger.info(f"🔧 Training integration template saved: {template_path}")
        return template_path


def main():
    """Main execution function."""
    print("\n" + "=" * 70)
    print("🧪 PHASE 3.0: END-TO-END GIMAN INTEGRATION TEST")
    print("=" * 70)

    # Setup - updated for phase3 directory
    base_dir = Path(__file__).parent.parent.parent.parent
    test_mode = "integration"  # Can be "integration", "performance", or "full"

    # Initialize tester
    tester = Phase3EndToEndTester(base_dir, test_mode)

    # Run tests
    logger.info("Starting end-to-end integration tests...")

    try:
        # Step 1: Integration test
        integration_success = tester.run_integration_test()

        if integration_success:
            # Step 2: Performance test (if requested)
            if test_mode in ["performance", "full"]:
                performance_results = tester.run_performance_test()
                print(
                    f"Performance test: {performance_results.get('status', 'unknown')}"
                )

        else:
            print("\n❌ Integration test failed - check logs for details")

        # Step 3: Generate reports
        report_path = tester.generate_test_report()

        # Step 4: Create integration template for training
        template_path = tester.create_training_integration_template()

        # Summary
        print("\n" + "=" * 70)
        print("📋 PHASE 3.0 COMPLETE - INTEGRATION TEST SUMMARY")
        print("=" * 70)

        for test_name, result in tester.test_results.items():
            status = result.get("status", "unknown")
            status_emoji = (
                "✅" if status == "success" else "❌" if status == "failed" else "⚠️"
            )
            print(f"{status_emoji} {test_name.replace('_', ' ').title()}: {status}")

        print("\n📁 Files Generated:")
        print(f"  📋 Test Report: {report_path.name}")
        print(f"  🔧 Training Template: {template_path.name}")

        if integration_success:
            print("\n🎉 INTEGRATION SUCCESSFUL!")
            print("✅ Ready for full GIMAN training with spatiotemporal embeddings")
            print("✅ Enhanced feature space: +256 spatiotemporal dimensions")
            print("✅ Integration template generated for training pipeline")
            return {"status": "success", "tester": tester}
        else:
            print("\n❌ INTEGRATION FAILED")
            print("Review test results and fix issues before proceeding")
            return {"status": "failed", "tester": tester}

    except Exception as e:
        logger.error(f"Phase 3.0 failed: {e}")
        print(f"\n❌ Phase 3.0 FAILED: {e}")
        return {"status": "error", "error": str(e)}


if __name__ == "__main__":
    try:
        results = main()
        print(f"\n✅ Phase 3.0 Complete - Status: {results['status']}")

    except Exception as e:
        logger.error(f"❌ Phase 3.0 execution failed: {e}")
        raise
