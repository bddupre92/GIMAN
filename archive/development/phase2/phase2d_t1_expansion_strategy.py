#!/usr/bin/env python3
"""Phase 2D: T1-Weighted Image Expansion Strategy

Revolutionary approach to dataset expansion by including ALL T1-weighted sequences
instead of just MPRAGE, potentially increasing dataset from 95 to 500+ patients.

Strategic Insight:
- Current: MPRAGE + DaTSCAN + genomics = 95 patients
- Expanded: Any T1 + DaTSCAN + genomics = 500+ patients
- T1 sequences share anatomical contrast properties
- FreeSurfer processes multiple T1 types effectively

Author: GIMAN Development Team
Date: September 27, 2025
Phase: 2D - T1 Dataset Expansion
"""

import logging
import warnings
from pathlib import Path

import pandas as pd

warnings.filterwarnings("ignore")

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


class T1ExpansionAnalyzer:
    """Analyze PPMI data for T1-weighted sequence expansion opportunities."""

    def __init__(self, data_dir: str):
        """Initialize T1 expansion analyzer."""
        self.data_dir = Path(data_dir)
        self.csv_files = {}
        logger.info("🔍 T1-Weighted Expansion Strategy Analyzer initialized")

    def load_ppmi_data(self):
        """Load all PPMI CSV files for analysis."""
        logger.info("📊 Loading PPMI data for T1 expansion analysis...")

        # Core data files
        csv_files = [
            "Demographics_18Sep2025.csv",
            "Participant_Status_18Sep2025.csv",
            "iu_genetic_consensus_20250515_18Sep2025.csv",
            "MDS-UPDRS_Part_I_18Sep2025.csv",
            "MDS-UPDRS_Part_III_18Sep2025.csv",
            "Montreal_Cognitive_Assessment__MoCA__18Sep2025.csv",
            "FS7_APARC_CTH_18Sep2025.csv",
            "Grey_Matter_Volume_18Sep2025.csv",
        ]

        for filename in csv_files:
            file_path = self.data_dir / filename
            if file_path.exists():
                try:
                    df = pd.read_csv(file_path)
                    self.csv_files[filename] = df
                    logger.info(f"✅ Loaded {filename}: {len(df)} rows")
                except Exception as e:
                    logger.warning(f"⚠️ Could not load {filename}: {e}")
            else:
                logger.warning(f"⚠️ File not found: {filename}")

    def analyze_current_cohort(self) -> dict:
        """Analyze current 95-patient cohort characteristics."""
        logger.info("🔍 Analyzing current 95-patient cohort...")

        # Get demographics data
        demographics = self.csv_files.get("Demographics_18Sep2025.csv")
        participant_status = self.csv_files.get("Participant_Status_18Sep2025.csv")
        genetics = self.csv_files.get("iu_genetic_consensus_20250515_18Sep2025.csv")

        analysis = {
            "total_demographics": len(demographics) if demographics is not None else 0,
            "total_participants": len(participant_status)
            if participant_status is not None
            else 0,
            "total_genetics": len(genetics) if genetics is not None else 0,
        }

        if demographics is not None:
            # Analyze cohort composition
            analysis["sex_distribution"] = demographics["SEX"].value_counts().to_dict()

            # Age analysis (approximate from birth dates)
            if "BIRTHDT" in demographics.columns:
                birth_years = pd.to_datetime(
                    demographics["BIRTHDT"], errors="coerce"
                ).dt.year
                current_year = 2025
                ages = current_year - birth_years
                analysis["age_range"] = {
                    "min": ages.min(),
                    "max": ages.max(),
                    "mean": ages.mean(),
                    "std": ages.std(),
                }

        if participant_status is not None:
            # Analyze cohort definitions
            if "COHORT_DEFINITION" in participant_status.columns:
                analysis["cohort_types"] = (
                    participant_status["COHORT_DEFINITION"].value_counts().to_dict()
                )

        logger.info(
            f"📈 Current cohort analysis complete: {analysis['total_demographics']} demographic records"
        )
        return analysis

    def identify_t1_expansion_opportunities(self) -> dict:
        """Identify opportunities for T1-weighted expansion."""
        logger.info("🎯 Identifying T1-weighted expansion opportunities...")

        # Analyze neuroimaging data availability
        fs_data = self.csv_files.get("FS7_APARC_CTH_18Sep2025.csv")
        gm_data = self.csv_files.get("Grey_Matter_Volume_18Sep2025.csv")

        opportunities = {
            "current_mprage_patients": 95,  # Known from previous analysis
            "freesurfer_available": len(fs_data) if fs_data is not None else 0,
            "grey_matter_available": len(gm_data) if gm_data is not None else 0,
        }

        if gm_data is not None:
            # Analyze unique patients with volumetric data
            unique_patients = gm_data["PATNO"].nunique()
            opportunities["unique_volumetric_patients"] = unique_patients

            # Analyze temporal coverage
            visit_coverage = gm_data["EVENT_ID"].value_counts().to_dict()
            opportunities["visit_coverage"] = visit_coverage

            logger.info(f"🧠 Found {unique_patients} patients with volumetric data")

        # Estimate expansion potential
        if fs_data is not None:
            # FreeSurfer data suggests T1-weighted processing occurred
            opportunities["potential_t1_patients"] = len(fs_data)
            expansion_factor = len(fs_data) / 95 if len(fs_data) > 95 else 1
            opportunities["expansion_factor"] = expansion_factor

            logger.info(
                f"🚀 Potential T1 expansion: {len(fs_data)} patients ({expansion_factor:.1f}x current)"
            )

        return opportunities

    def analyze_longitudinal_coverage(self) -> dict:
        """Analyze longitudinal visit coverage for expansion planning."""
        logger.info("⏰ Analyzing longitudinal visit coverage...")

        longitudinal_data = {}

        # Analyze MDS-UPDRS longitudinal coverage
        updrs_part1 = self.csv_files.get("MDS-UPDRS_Part_I_18Sep2025.csv")
        updrs_part3 = self.csv_files.get("MDS-UPDRS_Part_III_18Sep2025.csv")
        moca = self.csv_files.get("Montreal_Cognitive_Assessment__MoCA__18Sep2025.csv")

        for name, data in [
            ("UPDRS_I", updrs_part1),
            ("UPDRS_III", updrs_part3),
            ("MoCA", moca),
        ]:
            if data is not None:
                # Patients with longitudinal data
                patient_visits = data.groupby("PATNO")["EVENT_ID"].nunique()
                longitudinal_patients = (patient_visits > 1).sum()

                longitudinal_data[name] = {
                    "total_records": len(data),
                    "unique_patients": data["PATNO"].nunique(),
                    "longitudinal_patients": longitudinal_patients,
                    "avg_visits_per_patient": patient_visits.mean(),
                    "max_visits": patient_visits.max(),
                }

                logger.info(
                    f"📊 {name}: {longitudinal_patients} patients with multiple visits"
                )

        return longitudinal_data

    def create_expansion_strategy(self) -> dict:
        """Create comprehensive T1 expansion strategy."""
        logger.info("🎯 Creating T1-weighted expansion strategy...")

        current_analysis = self.analyze_current_cohort()
        expansion_opportunities = self.identify_t1_expansion_opportunities()
        longitudinal_coverage = self.analyze_longitudinal_coverage()

        strategy = {
            "current_state": current_analysis,
            "expansion_opportunities": expansion_opportunities,
            "longitudinal_analysis": longitudinal_coverage,
            "strategic_recommendations": self._generate_recommendations(
                current_analysis, expansion_opportunities, longitudinal_coverage
            ),
        }

        return strategy

    def _generate_recommendations(
        self, current: dict, expansion: dict, longitudinal: dict
    ) -> list[str]:
        """Generate strategic recommendations based on analysis."""
        recommendations = []

        # Data expansion recommendations
        if expansion.get("expansion_factor", 1) > 2:
            recommendations.append(
                f"🚀 PRIORITY 1: T1 expansion could increase dataset by {expansion['expansion_factor']:.1f}x "
                f"({expansion.get('potential_t1_patients', 0)} vs 95 current patients)"
            )

        # Technical implementation recommendations
        recommendations.extend(
            [
                "🔧 TECHNICAL: Modify Phase 2 encoders to accept any T1-weighted sequence (not just MPRAGE)",
                "📊 DATA QUALITY: Implement T1 sequence type detection and normalization",
                "🧠 PROCESSING: Extend FreeSurfer pipeline to handle multiple T1 types",
                "⚖️ VALIDATION: Ensure consistent anatomical feature extraction across T1 variants",
            ]
        )

        # Longitudinal recommendations
        longitudinal_patients = longitudinal.get("UPDRS_III", {}).get(
            "longitudinal_patients", 0
        )
        if longitudinal_patients > 100:
            recommendations.append(
                f"⏰ LONGITUDINAL: {longitudinal_patients} patients have multiple visits - "
                "leverage temporal modeling for stronger predictions"
            )

        # Model architecture recommendations
        recommendations.extend(
            [
                "🤖 ARCHITECTURE: Keep simplified ensemble as baseline until dataset expansion",
                "📈 VALIDATION: Re-test GAT architecture on expanded dataset (likely to improve significantly)",
                "🎯 TARGET: Aim for 300-500 patients to achieve positive R² consistently",
            ]
        )

        return recommendations

    def generate_implementation_plan(self) -> str:
        """Generate concrete implementation plan."""
        plan = """
# Phase 2D: T1-Weighted Expansion Implementation Plan

## 🎯 Objective: Expand from 95 to 300-500 patients via T1 sequence diversification

### Week 1: Data Discovery & Analysis
1. **DICOM Inventory**: Catalog all available T1-weighted sequences in PPMI
   - MPRAGE, SPGR, T1-FLAIR, etc.
   - Document sequence parameters and quality
   
2. **Eligibility Analysis**: Identify patients with:
   - Any T1-weighted image + DaTSCAN + genetic data
   - Clinical outcomes (MDS-UPDRS, MoCA)
   
3. **Quality Assessment**: 
   - FreeSurfer processing success rates across T1 types
   - Anatomical feature consistency validation

### Week 2: Pipeline Modification
1. **Phase 2 Encoder Updates**:
   - Modify `phase2_enhanced_encoders.py` for multi-T1 support
   - Implement T1 sequence normalization
   - Add quality control checks
   
2. **Data Integration Updates**:
   - Extend Phase 1 data loading for expanded cohort
   - Update patient matching logic
   - Validate multimodal alignment

### Week 3: Validation & Testing  
1. **Expanded Dataset Validation**:
   - Verify anatomical consistency across T1 types
   - Test feature extraction pipeline
   - Quality control for outlier detection
   
2. **Phase 5 Re-testing**:
   - Test simplified ensemble on expanded dataset
   - Re-evaluate GAT architecture performance
   - Compare 95-patient vs expanded cohort results

### Expected Outcomes:
- **Dataset**: 95 → 300-500 patients
- **R² Improvement**: -0.0189 → +0.15-0.30 (target)
- **Statistical Power**: Robust significance testing
- **Architecture**: GAT likely to become viable with larger dataset

### Success Metrics:
- ✅ Positive R² (>0.1) consistently achieved
- ✅ AUC > 0.65 for cognitive classification  
- ✅ Statistical significance (p < 0.05) in comparisons
- ✅ Robust performance across cross-validation folds
"""
        return plan


def run_t1_expansion_analysis():
    """Run comprehensive T1 expansion analysis."""
    logger.info("🚀 Starting T1-Weighted Expansion Analysis")
    logger.info("=" * 60)

    # Initialize analyzer
    data_dir = "/Users/blair.dupre/Library/CloudStorage/GoogleDrive-dupre.blair92@gmail.com/My Drive/CSCI FALL 2025/data/00_raw/GIMAN/ppmi_data_csv"
    analyzer = T1ExpansionAnalyzer(data_dir)

    # Load data
    analyzer.load_ppmi_data()

    # Create expansion strategy
    strategy = analyzer.create_expansion_strategy()

    # Display results
    logger.info("📊 T1 EXPANSION ANALYSIS RESULTS:")
    logger.info("=" * 40)

    # Current state
    current = strategy["current_state"]
    logger.info(
        f"📈 Current Demographics: {current.get('total_demographics', 0)} records"
    )
    logger.info(
        f"👥 Current Participants: {current.get('total_participants', 0)} records"
    )
    logger.info(f"🧬 Genetic Data: {current.get('total_genetics', 0)} records")

    # Expansion opportunities
    expansion = strategy["expansion_opportunities"]
    logger.info(
        f"🧠 FreeSurfer Data Available: {expansion.get('freesurfer_available', 0)} records"
    )
    logger.info(
        f"📊 Grey Matter Data: {expansion.get('grey_matter_available', 0)} records"
    )
    if expansion.get("expansion_factor", 1) > 1:
        logger.info(
            f"🚀 EXPANSION POTENTIAL: {expansion['expansion_factor']:.1f}x current dataset!"
        )

    # Strategic recommendations
    logger.info("\n🎯 STRATEGIC RECOMMENDATIONS:")
    for i, rec in enumerate(strategy["strategic_recommendations"], 1):
        logger.info(f"{i}. {rec}")

    # Implementation plan
    logger.info("\n📋 IMPLEMENTATION PLAN:")
    plan = analyzer.generate_implementation_plan()
    print(plan)

    return strategy


if __name__ == "__main__":
    strategy = run_t1_expansion_analysis()

    print("\n🎉 T1-Weighted Expansion Analysis Complete!")
    print("This strategy could be the breakthrough needed for positive R²!")
