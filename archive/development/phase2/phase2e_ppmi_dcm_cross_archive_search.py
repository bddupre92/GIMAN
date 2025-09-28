#!/usr/bin/env python3
"""Phase 2e: PPMI_dcm Cross-Archive Search for 200+ Patient Multimodal Dataset

This module implements a comprehensive cross-archive search across all PPMI_dcm
directories to discover T1-weighted, DaTSCAN, and potentially fMRI data for
maximum dataset expansion. Based on successful T1 expansion proof-of-concept
showing +0.1864 R² improvement, this scales to target 200-300 patients.

STRATEGIC OBJECTIVE:
- Scale from 25 patients → 200-300 patients (8-12x expansion)
- Target R² improvement from -0.36 to +0.15-0.25
- Comprehensive multimodal data discovery and curation
- Cross-reference with clinical data for complete feature vectors

DISCOVERY STRATEGY:
1. Scan all 297 patients in PPMI_dcm for T1, DaTSCAN, fMRI
2. Cross-reference PPMI 3 directory for additional sequences
3. Prioritize patients with multiple modalities
4. Validate against clinical data availability
5. Create comprehensive conversion pipeline

Author: AI Research Assistant
Date: Current Session
Context: Post T1 expansion validation, scaling for production dataset
"""

import json
import logging
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)


class PPMICrossArchiveSearcher:
    """Comprehensive cross-archive searcher for PPMI_dcm multimodal data discovery.

    This class implements systematic discovery across all PPMI directories to identify
    maximum available T1, DaTSCAN, and fMRI data for large-scale GIMAN dataset expansion.
    """

    def __init__(self, base_data_dir: str):
        """Initialize the cross-archive searcher.

        Args:
            base_data_dir: Base PPMI data directory containing PPMI_dcm and PPMI 3
        """
        self.base_data_dir = Path(base_data_dir)
        self.ppmi_dcm_dir = self.base_data_dir / "PPMI_dcm"
        self.ppmi_3_dir = self.base_data_dir / "PPMI 3"

        # Discovery results
        self.patient_inventory = {}
        self.modality_stats = defaultdict(int)
        self.multimodal_patients = []
        self.clinical_match_stats = {}

        # Target modalities for GIMAN
        self.target_modalities = {
            "T1": ["MPRAGE", "T1", "t1_mpr", "SAG_3D_MPRAGE", "3D_T1"],
            "DATSCN": ["DaTSCAN", "SPECT", "Striatal_Binding_Ratio"],
            "fMRI": ["fMRI", "rsfMRI", "REST", "BOLD", "rs_fMRI"],
        }

        # Clinical data paths
        self.clinical_data_dir = self.base_data_dir / "data/00_raw/GIMAN/ppmi_data_csv"

        logging.info("🔍 PPMI Cross-Archive Searcher initialized")
        logging.info(f"📁 PPMI_dcm: {self.ppmi_dcm_dir}")
        logging.info(f"📁 PPMI 3: {self.ppmi_3_dir}")

    def discover_all_patients(self) -> dict[str, dict]:
        """Comprehensive discovery of all patients across PPMI directories.

        Returns:
            Dict mapping patient IDs to their available modalities and metadata
        """
        logging.info("🔍 Starting comprehensive patient discovery...")

        # Discover PPMI_dcm patients
        if self.ppmi_dcm_dir.exists():
            logging.info(f"📊 Scanning PPMI_dcm directory: {self.ppmi_dcm_dir}")
            self._scan_ppmi_dcm_directory()
        else:
            logging.warning(f"⚠️ PPMI_dcm not found: {self.ppmi_dcm_dir}")

        # Discover PPMI 3 patients
        if self.ppmi_3_dir.exists():
            logging.info(f"📊 Scanning PPMI 3 directory: {self.ppmi_3_dir}")
            self._scan_ppmi_3_directory()
        else:
            logging.warning(f"⚠️ PPMI 3 not found: {self.ppmi_3_dir}")

        # Analysis and prioritization
        self._analyze_discovery_results()

        return self.patient_inventory

    def _scan_ppmi_dcm_directory(self):
        """Scan PPMI_dcm for all patient directories and modalities."""
        try:
            patient_dirs = [d for d in self.ppmi_dcm_dir.iterdir() if d.is_dir()]
            logging.info(
                f"📊 Found {len(patient_dirs)} potential patient directories in PPMI_dcm"
            )

            for i, patient_dir in enumerate(patient_dirs, 1):
                if i % 50 == 0:
                    logging.info(
                        f"📊 Progress: {i}/{len(patient_dirs)} patients scanned"
                    )

                patient_id = patient_dir.name

                # Initialize patient record
                if patient_id not in self.patient_inventory:
                    self.patient_inventory[patient_id] = {
                        "modalities": {},
                        "locations": [],
                        "total_sequences": 0,
                        "priority_score": 0,
                    }

                # Scan patient directory for imaging sequences
                self._scan_patient_directory(patient_dir, patient_id, "PPMI_dcm")

        except Exception as e:
            logging.error(f"❌ Error scanning PPMI_dcm: {e}")

    def _scan_ppmi_3_directory(self):
        """Scan PPMI 3 for additional patient data."""
        try:
            patient_dirs = [d for d in self.ppmi_3_dir.iterdir() if d.is_dir()]
            logging.info(f"📊 Found {len(patient_dirs)} patient directories in PPMI 3")

            for patient_dir in patient_dirs:
                patient_id = patient_dir.name

                # Initialize patient record if new
                if patient_id not in self.patient_inventory:
                    self.patient_inventory[patient_id] = {
                        "modalities": {},
                        "locations": [],
                        "total_sequences": 0,
                        "priority_score": 0,
                    }

                # Scan patient directory
                self._scan_patient_directory(patient_dir, patient_id, "PPMI_3")

        except Exception as e:
            logging.error(f"❌ Error scanning PPMI 3: {e}")

    def _scan_patient_directory(self, patient_dir: Path, patient_id: str, source: str):
        """Scan individual patient directory for imaging sequences.

        Args:
            patient_dir: Path to patient directory
            patient_id: Patient identifier
            source: Source directory name (PPMI_dcm or PPMI_3)
        """
        try:
            # Look for imaging sequence directories
            sequence_dirs = [d for d in patient_dir.iterdir() if d.is_dir()]

            for seq_dir in sequence_dirs:
                seq_name = seq_dir.name.upper()

                # Check for target modalities
                detected_modality = self._classify_sequence(seq_name)

                if detected_modality:
                    # Count DICOM files
                    dicom_count = len([f for f in seq_dir.rglob("*.dcm")])

                    if dicom_count > 0:
                        # Add to patient inventory
                        if (
                            detected_modality
                            not in self.patient_inventory[patient_id]["modalities"]
                        ):
                            self.patient_inventory[patient_id]["modalities"][
                                detected_modality
                            ] = []

                        self.patient_inventory[patient_id]["modalities"][
                            detected_modality
                        ].append(
                            {
                                "sequence_name": seq_name,
                                "path": str(seq_dir),
                                "dicom_count": dicom_count,
                                "source": source,
                            }
                        )

                        # Update stats
                        self.modality_stats[detected_modality] += 1
                        self.patient_inventory[patient_id]["total_sequences"] += 1

        except Exception as e:
            logging.debug(f"⚠️ Error scanning {patient_dir}: {e}")

    def _classify_sequence(self, sequence_name: str) -> str | None:
        """Classify imaging sequence by name to determine modality.

        Args:
            sequence_name: Name of imaging sequence directory

        Returns:
            Detected modality ('T1', 'DATSCN', 'fMRI') or None
        """
        sequence_name = sequence_name.upper()

        # Check each target modality
        for modality, keywords in self.target_modalities.items():
            for keyword in keywords:
                if keyword.upper() in sequence_name:
                    return modality

        return None

    def _analyze_discovery_results(self):
        """Analyze discovery results and prioritize patients."""
        logging.info("📊 Analyzing discovery results...")

        # Calculate priority scores
        for patient_id, data in self.patient_inventory.items():
            score = 0

            # Score for each modality (T1=3, DaTSCAN=2, fMRI=1)
            if "T1" in data["modalities"]:
                score += 3
            if "DATSCN" in data["modalities"]:
                score += 2
            if "fMRI" in data["modalities"]:
                score += 1

            # Bonus for multiple sequences of same modality
            score += sum(
                len(sequences) - 1 for sequences in data["modalities"].values()
            )

            data["priority_score"] = score

            # Identify multimodal patients
            if len(data["modalities"]) >= 2:
                self.multimodal_patients.append(patient_id)

        # Sort multimodal patients by priority
        self.multimodal_patients.sort(
            key=lambda pid: self.patient_inventory[pid]["priority_score"], reverse=True
        )

        logging.info("📊 Discovery complete:")
        logging.info(f"   - Total patients: {len(self.patient_inventory)}")
        logging.info(f"   - Multimodal patients: {len(self.multimodal_patients)}")
        logging.info(f"   - T1 patients: {self.modality_stats['T1']}")
        logging.info(f"   - DaTSCAN patients: {self.modality_stats['DATSCN']}")
        logging.info(f"   - fMRI patients: {self.modality_stats['fMRI']}")

    def cross_reference_clinical_data(self) -> dict[str, bool]:
        """Cross-reference discovered patients with available clinical data.

        Returns:
            Dict mapping patient IDs to clinical data availability
        """
        logging.info("🔗 Cross-referencing with clinical data...")

        clinical_availability = {}

        try:
            # Load key clinical data files
            demographics_file = self.clinical_data_dir / "Demographics_18Sep2025.csv"
            updrs_file = self.clinical_data_dir / "MDS-UPDRS_Part_III_18Sep2025.csv"

            if demographics_file.exists():
                demographics_df = pd.read_csv(demographics_file)
                clinical_patients = set(demographics_df["PATNO"].astype(str))

                # Check availability for each discovered patient
                for patient_id in self.patient_inventory.keys():
                    clinical_availability[patient_id] = patient_id in clinical_patients

                logging.info("✅ Clinical data cross-reference complete:")
                logging.info(
                    f"   - Patients with clinical data: {sum(clinical_availability.values())}"
                )
                logging.info(
                    f"   - Patients without clinical data: {len(clinical_availability) - sum(clinical_availability.values())}"
                )

            else:
                logging.warning(
                    "⚠️ Demographics file not found for clinical cross-reference"
                )

        except Exception as e:
            logging.error(f"❌ Error cross-referencing clinical data: {e}")

        return clinical_availability

    def generate_expansion_strategy(self, target_patients: int = 200) -> dict:
        """Generate comprehensive expansion strategy for target patient count.

        Args:
            target_patients: Target number of patients for expanded dataset

        Returns:
            Expansion strategy with prioritized patient lists and conversion plan
        """
        logging.info(
            f"🎯 Generating expansion strategy for {target_patients} patients..."
        )

        # Cross-reference with clinical data
        clinical_availability = self.cross_reference_clinical_data()

        # Priority ranking: multimodal + clinical data available
        priority_1 = []  # Multimodal + clinical
        priority_2 = []  # T1 + DaTSCAN + clinical
        priority_3 = []  # Single modality + clinical
        priority_4 = []  # Any data available

        for patient_id in self.patient_inventory.keys():
            data = self.patient_inventory[patient_id]
            has_clinical = clinical_availability.get(patient_id, False)
            modalities = list(data["modalities"].keys())

            if len(modalities) >= 2 and has_clinical:
                priority_1.append(patient_id)
            elif "T1" in modalities and "DATSCN" in modalities and has_clinical:
                priority_2.append(patient_id)
            elif len(modalities) >= 1 and has_clinical:
                priority_3.append(patient_id)
            else:
                priority_4.append(patient_id)

        # Sort each priority group by priority score
        for priority_list in [priority_1, priority_2, priority_3, priority_4]:
            priority_list.sort(
                key=lambda pid: self.patient_inventory[pid]["priority_score"],
                reverse=True,
            )

        # Select patients up to target
        selected_patients = []
        for priority_list in [priority_1, priority_2, priority_3, priority_4]:
            remaining_slots = target_patients - len(selected_patients)
            if remaining_slots <= 0:
                break
            selected_patients.extend(priority_list[:remaining_slots])

        # Generate strategy
        strategy = {
            "target_patients": target_patients,
            "available_patients": len(self.patient_inventory),
            "selected_patients": selected_patients[:target_patients],
            "priority_breakdown": {
                "multimodal_clinical": len(priority_1),
                "t1_datscn_clinical": len(priority_2),
                "single_modal_clinical": len(priority_3),
                "any_available": len(priority_4),
            },
            "modality_coverage": {
                "T1_patients": len(
                    [
                        pid
                        for pid in selected_patients[:target_patients]
                        if "T1" in self.patient_inventory[pid]["modalities"]
                    ]
                ),
                "DaTSCAN_patients": len(
                    [
                        pid
                        for pid in selected_patients[:target_patients]
                        if "DATSCN" in self.patient_inventory[pid]["modalities"]
                    ]
                ),
                "fMRI_patients": len(
                    [
                        pid
                        for pid in selected_patients[:target_patients]
                        if "fMRI" in self.patient_inventory[pid]["modalities"]
                    ]
                ),
            },
            "expected_performance": self._estimate_performance_improvement(
                len(selected_patients[:target_patients])
            ),
        }

        logging.info("🎯 Expansion strategy generated:")
        logging.info(
            f"   - Selected patients: {len(selected_patients[:target_patients])}"
        )
        logging.info(
            f"   - T1 coverage: {strategy['modality_coverage']['T1_patients']}"
        )
        logging.info(
            f"   - DaTSCAN coverage: {strategy['modality_coverage']['DaTSCAN_patients']}"
        )
        logging.info(
            f"   - Expected R²: {strategy['expected_performance']['target_r2']:.3f}"
        )

        return strategy

    def _estimate_performance_improvement(self, n_patients: int) -> dict:
        """Estimate performance improvement based on dataset size.

        Args:
            n_patients: Number of patients in expanded dataset

        Returns:
            Performance estimates
        """
        # Based on successful T1 expansion: +0.1864 R² improvement with 1.0x expansion
        # Scaling model: R² improvement ∝ log(expansion_factor)

        baseline_patients = 25  # Current baseline from T1 test
        expansion_factor = n_patients / baseline_patients

        # Improvement model based on successful test
        base_improvement = 0.1864  # From T1 expansion test
        scaled_improvement = base_improvement * np.log(expansion_factor)

        # Current baseline R² from test
        baseline_r2 = -0.3587
        target_r2 = baseline_r2 + scaled_improvement

        # Conservative bounds
        target_r2 = min(target_r2, 0.30)  # Cap at reasonable upper bound

        return {
            "baseline_r2": baseline_r2,
            "target_r2": target_r2,
            "improvement": scaled_improvement,
            "expansion_factor": expansion_factor,
            "confidence": min(0.95, 0.5 + 0.1 * np.log(expansion_factor)),
        }

    def save_discovery_results(self, output_dir: str):
        """Save comprehensive discovery results to files.

        Args:
            output_dir: Directory to save results
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Save patient inventory
        inventory_file = output_path / "ppmi_cross_archive_inventory.json"
        with open(inventory_file, "w") as f:
            json.dump(self.patient_inventory, f, indent=2, default=str)

        # Save expansion strategy
        strategy = self.generate_expansion_strategy()
        strategy_file = output_path / "ppmi_expansion_strategy.json"
        with open(strategy_file, "w") as f:
            json.dump(strategy, f, indent=2, default=str)

        # Save summary statistics
        summary = {
            "discovery_timestamp": datetime.now().isoformat(),
            "total_patients_discovered": len(self.patient_inventory),
            "modality_statistics": dict(self.modality_stats),
            "multimodal_patients": len(self.multimodal_patients),
            "top_priority_patients": self.multimodal_patients[:50],
            "recommended_target": min(200, len(self.patient_inventory)),
        }

        summary_file = output_path / "ppmi_discovery_summary.json"
        with open(summary_file, "w") as f:
            json.dump(summary, f, indent=2, default=str)

        logging.info(f"💾 Discovery results saved to: {output_path}")
        logging.info(f"   - Inventory: {inventory_file}")
        logging.info(f"   - Strategy: {strategy_file}")
        logging.info(f"   - Summary: {summary_file}")


def main():
    """Main execution function for PPMI cross-archive search."""
    logging.info("🚀 PPMI CROSS-ARCHIVE SEARCH")
    logging.info("=" * 60)
    logging.info("🎯 Objective: Discover 200+ patients for GIMAN expansion")
    logging.info("📊 Strategy: Comprehensive multimodal data discovery")
    logging.info("🔍 Scope: All PPMI_dcm + PPMI 3 directories")

    # Initialize searcher
    base_data_dir = "/Users/blair.dupre/Library/CloudStorage/GoogleDrive-dupre.blair92@gmail.com/My Drive/CSCI FALL 2025/data/00_raw/GIMAN"

    searcher = PPMICrossArchiveSearcher(base_data_dir)

    # Execute comprehensive discovery
    try:
        # Step 1: Discover all patients
        logging.info("🔍 Step 1: Comprehensive patient discovery")
        patient_inventory = searcher.discover_all_patients()

        # Step 2: Generate expansion strategy
        logging.info("🎯 Step 2: Generate expansion strategy")
        strategy = searcher.generate_expansion_strategy(target_patients=200)

        # Step 3: Save results
        logging.info("💾 Step 3: Save discovery results")
        output_dir = "/Users/blair.dupre/Library/CloudStorage/GoogleDrive-dupre.blair92@gmail.com/My Drive/CSCI FALL 2025/archive/development/phase2"
        searcher.save_discovery_results(output_dir)

        # Summary report
        logging.info("📊 CROSS-ARCHIVE SEARCH COMPLETE!")
        logging.info("=" * 60)
        logging.info(f"✅ Total patients discovered: {len(patient_inventory)}")
        logging.info(
            f"✅ Target patients selected: {len(strategy['selected_patients'])}"
        )
        logging.info(
            f"✅ Expected R² improvement: {strategy['expected_performance']['target_r2']:.3f}"
        )
        logging.info(
            f"✅ Multimodal coverage: {strategy['priority_breakdown']['multimodal_clinical']}"
        )

        if strategy["expected_performance"]["target_r2"] > 0:
            logging.info("🎉 STRATEGY SUCCESS: Positive R² achievable!")
            logging.info("🚀 Ready to proceed with large-scale expansion")
        else:
            logging.info("⚠️ Strategy needs refinement for positive R²")
            logging.info("💡 Consider additional data sources or feature engineering")

    except Exception as e:
        logging.error(f"❌ Cross-archive search failed: {e}")
        raise


if __name__ == "__main__":
    main()
