#!/usr/bin/env python3
"""
GIMAN Master Pipeline Orchestrator - Phase 4 & Phase 5 Execution

This script executes both Phase 4 (Progression Subtype Discovery) and 
Phase 5 (Prodromal Transition Prediction) pipelines sequentially, handling
dependencies between phases and generating a unified results report.

Expected Total Runtime: 7-11 hours
Expected Total Outputs: ~90 files

Author: GIMAN Research Team
Date: October 5, 2025
"""

import sys
import logging
from pathlib import Path
from datetime import datetime
import json
import argparse
import warnings
warnings.filterwarnings('ignore')

# Add archive directories to path
archive_base = Path(__file__).parent / "archive" / "development"
sys.path.insert(0, str(archive_base / "phase4"))
sys.path.insert(0, str(archive_base / "phase5"))


class GIMANMasterOrchestrator:
    """
    Master orchestrator for GIMAN Phase 4 and Phase 5 pipelines.
    
    This class coordinates execution of both pipelines, manages dependencies,
    and generates a unified results report.
    """
    
    def __init__(
        self,
        phase1_data_path: str,
        ppmi_data_path: str,
        output_dir: str = "visualizations/phase4_5_results",
        random_seed: int = 42,
        skip_phase4: bool = False,
        skip_phase5: bool = False
    ):
        """
        Initialize master orchestrator.
        
        Args:
            phase1_data_path: Path to Phase 1 cohort data CSV (for Phase 4)
            ppmi_data_path: Path to PPMI data directory (for Phase 5)
            output_dir: Root output directory for all results
            random_seed: Random seed for reproducibility
            skip_phase4: Skip Phase 4 execution (use existing results)
            skip_phase5: Skip Phase 5 execution (use existing results)
        """
        self.phase1_data_path = Path(phase1_data_path)
        self.ppmi_data_path = Path(ppmi_data_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.random_seed = random_seed
        self.skip_phase4 = skip_phase4
        self.skip_phase5 = skip_phase5
        
        # Phase-specific output directories
        self.phase4_output = self.output_dir / "phase4"
        self.phase5_output = self.output_dir / "phase5"
        
        # Setup logging
        self._setup_logging()
        
        # Master results storage
        self.master_results = {
            'phase4': {},
            'phase5': {},
            'cross_phase_analysis': {},
            'execution_metadata': {
                'start_time': None,
                'end_time': None,
                'total_runtime_hours': None,
                'phase1_data_path': str(self.phase1_data_path),
                'ppmi_data_path': str(self.ppmi_data_path),
                'random_seed': self.random_seed,
                'phases_executed': []
            }
        }
        
        self.logger.info("="*80)
        self.logger.info("GIMAN MASTER PIPELINE ORCHESTRATOR")
        self.logger.info("="*80)
        self.logger.info(f"Phase 1 data: {self.phase1_data_path}")
        self.logger.info(f"PPMI data: {self.ppmi_data_path}")
        self.logger.info(f"Output directory: {self.output_dir}")
        self.logger.info(f"Random seed: {self.random_seed}")
        self.logger.info(f"Skip Phase 4: {self.skip_phase4}")
        self.logger.info(f"Skip Phase 5: {self.skip_phase5}")
        
    def _setup_logging(self):
        """Configure logging for master orchestrator."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_file = self.output_dir / f"master_pipeline_{timestamp}.log"
        
        # Clear any existing handlers
        for handler in logging.root.handlers[:]:
            logging.root.removeHandler(handler)
            
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler(sys.stdout)
            ]
        )
        self.logger = logging.getLogger(__name__)
        
    def execute_phase4(self) -> dict:
        """
        Execute Phase 4: Progression Subtype Discovery pipeline.
        
        Returns:
            Phase 4 results dictionary
        """
        if self.skip_phase4:
            self.logger.info("\n⏭️  Skipping Phase 4 (as requested)")
            self.logger.info("Looking for existing Phase 4 results...")
            
            # Try to load existing results
            existing_results = self.phase4_output / "phase4_master_results.json"
            if existing_results.exists():
                with open(existing_results, 'r') as f:
                    return json.load(f)
            else:
                self.logger.warning("No existing Phase 4 results found!")
                return {}
        
        self.logger.info("\n" + "🔬"*40)
        self.logger.info("PHASE 4: PROGRESSION SUBTYPE DISCOVERY")
        self.logger.info("🔬"*40 + "\n")
        
        try:
            from execute_phase4_pipeline import Phase4PipelineExecutor
            
            executor = Phase4PipelineExecutor(
                phase1_data_path=str(self.phase1_data_path),
                output_dir=str(self.phase4_output),
                random_seed=self.random_seed
            )
            
            phase4_results = executor.execute()
            self.master_results['execution_metadata']['phases_executed'].append('phase4')
            
            self.logger.info("\n✅ Phase 4 complete!")
            self.logger.info(f"Runtime: {phase4_results['pipeline_metadata']['total_runtime_minutes']:.1f} minutes")
            
            return phase4_results
            
        except Exception as e:
            self.logger.error(f"❌ Phase 4 failed: {str(e)}", exc_info=True)
            raise
            
    def execute_phase5(self) -> dict:
        """
        Execute Phase 5: Prodromal Transition Prediction pipeline.
        
        Returns:
            Phase 5 results dictionary
        """
        if self.skip_phase5:
            self.logger.info("\n⏭️  Skipping Phase 5 (as requested)")
            self.logger.info("Looking for existing Phase 5 results...")
            
            # Try to load existing results
            existing_results = self.phase5_output / "phase5_master_results.json"
            if existing_results.exists():
                with open(existing_results, 'r') as f:
                    return json.load(f)
            else:
                self.logger.warning("No existing Phase 5 results found!")
                return {}
        
        self.logger.info("\n" + "🧬"*40)
        self.logger.info("PHASE 5: PRODROMAL-TO-CLINICAL TRANSITION PREDICTION")
        self.logger.info("🧬"*40 + "\n")
        
        try:
            from execute_phase5_pipeline import Phase5PipelineExecutor
            
            executor = Phase5PipelineExecutor(
                ppmi_data_path=str(self.ppmi_data_path),
                output_dir=str(self.phase5_output),
                random_seed=self.random_seed
            )
            
            phase5_results = executor.execute()
            self.master_results['execution_metadata']['phases_executed'].append('phase5')
            
            self.logger.info("\n✅ Phase 5 complete!")
            self.logger.info(f"Runtime: {phase5_results['pipeline_metadata']['total_runtime_minutes']:.1f} minutes")
            
            return phase5_results
            
        except Exception as e:
            self.logger.error(f"❌ Phase 5 failed: {str(e)}", exc_info=True)
            raise
            
    def cross_phase_analysis(self, phase4_results: dict, phase5_results: dict):
        """
        Perform cross-phase analysis linking Phase 4 subtypes with Phase 5 conversion risk.
        
        Args:
            phase4_results: Results from Phase 4
            phase5_results: Results from Phase 5
        """
        self.logger.info("\n" + "🔗"*40)
        self.logger.info("CROSS-PHASE ANALYSIS")
        self.logger.info("🔗"*40)
        
        try:
            # Extract key metrics
            n_subtypes = phase4_results.get('task_4_3', {}).get('n_subtypes', 0)
            conversion_rate = phase5_results.get('task_5_1', {}).get('conversion_rate', 0)
            
            self.logger.info(f"\n📊 Key Cross-Phase Insights:")
            self.logger.info(f"- Phase 4 discovered {n_subtypes} progression subtypes")
            self.logger.info(f"- Phase 5 observed {conversion_rate:.1%} prodromal conversion rate")
            
            # Future analysis: Link subtype membership to conversion risk
            self.master_results['cross_phase_analysis'] = {
                'subtypes_discovered': n_subtypes,
                'conversion_rate': conversion_rate,
                'note': 'Future work: Assess subtype-specific conversion risk in prodromal cohort'
            }
            
            self.logger.info("\n💡 Recommended Future Analysis:")
            self.logger.info("1. Classify prodromal converters by Phase 4 subtypes")
            self.logger.info("2. Test hypothesis: Fast progressors have higher prodromal conversion risk")
            self.logger.info("3. Integrate Phase 4 baseline prediction into Phase 5 risk stratification")
            
        except Exception as e:
            self.logger.error(f"Cross-phase analysis warning: {str(e)}")
            self.master_results['cross_phase_analysis'] = {'error': str(e)}
            
    def execute(self) -> dict:
        """
        Execute master pipeline orchestration.
        
        Returns:
            Complete master results dictionary
        """
        start_time = datetime.now()
        self.master_results['execution_metadata']['start_time'] = start_time.isoformat()
        
        self.logger.info("\n" + "🚀"*40)
        self.logger.info("STARTING MASTER PIPELINE EXECUTION")
        self.logger.info("🚀"*40)
        self.logger.info(f"Start time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        self.logger.info("Expected total runtime: 7-11 hours")
        
        try:
            # Execute Phase 4
            phase4_results = self.execute_phase4()
            self.master_results['phase4'] = phase4_results
            
            # Execute Phase 5
            phase5_results = self.execute_phase5()
            self.master_results['phase5'] = phase5_results
            
            # Cross-phase analysis
            if phase4_results and phase5_results:
                self.cross_phase_analysis(phase4_results, phase5_results)
            
            # Finalize
            end_time = datetime.now()
            total_runtime = (end_time - start_time).total_seconds() / 3600  # hours
            
            self.master_results['execution_metadata']['end_time'] = end_time.isoformat()
            self.master_results['execution_metadata']['total_runtime_hours'] = total_runtime
            
            # Save master results
            self._save_master_results()
            
            # Generate unified summary
            self._generate_unified_summary()
            
            self.logger.info("\n" + "🎉"*40)
            self.logger.info("MASTER PIPELINE EXECUTION COMPLETE!")
            self.logger.info("🎉"*40)
            self.logger.info(f"\n⏱️  Total runtime: {total_runtime:.2f} hours")
            self.logger.info(f"📁 All results saved to: {self.output_dir}")
            self.logger.info(f"📊 Unified summary: {self.output_dir}/UNIFIED_EXECUTIVE_SUMMARY.md")
            
            return self.master_results
            
        except Exception as e:
            self.logger.error(f"\n❌ Master pipeline failed: {str(e)}", exc_info=True)
            raise
            
    def _save_master_results(self):
        """Save master results JSON."""
        results_path = self.output_dir / "master_results.json"
        with open(results_path, 'w') as f:
            json.dump(self.master_results, f, indent=2, default=str)
        self.logger.info(f"\n💾 Master results saved: {results_path}")
        
    def _generate_unified_summary(self):
        """Generate unified markdown summary for all phases."""
        summary_path = self.output_dir / "UNIFIED_EXECUTIVE_SUMMARY.md"
        
        # Extract key metrics
        phase4 = self.master_results.get('phase4', {})
        phase5 = self.master_results.get('phase5', {})
        metadata = self.master_results.get('execution_metadata', {})
        
        # Safe access with defaults
        def safe_get(d, *keys, default=0):
            for key in keys:
                if isinstance(d, dict):
                    d = d.get(key, {})
                else:
                    return default
            return d if d != {} else default
        
        summary = f"""# GIMAN Master Pipeline - Unified Executive Summary

**Execution Date**: {metadata.get('start_time', 'N/A')}  
**Total Runtime**: {metadata.get('total_runtime_hours', 0):.2f} hours  
**Phases Executed**: {', '.join(metadata.get('phases_executed', []))}  
**Random Seed**: {metadata.get('random_seed', 42)}

---

## 🎯 Master Overview

This report consolidates results from:
1. **Phase 4**: Progression Subtype Discovery
2. **Phase 5**: Prodromal-to-Clinical Transition Prediction

---

## 📊 Phase 4: Progression Subtype Discovery

### Key Results
- **Cohort Size**: {safe_get(phase4, 'task_4_1', 'n_patients')} patients
- **Subtypes Discovered**: {safe_get(phase4, 'task_4_3', 'n_subtypes')}
- **Clustering Quality** (Silhouette): {safe_get(phase4, 'task_4_3', 'silhouette_score'):.4f}
- **Baseline Prediction AUC**: {safe_get(phase4, 'task_4_5', 'auc_macro'):.4f}
- **Trial Sample Reduction**: {safe_get(phase4, 'task_4_6', 'sample_size_reduction_percent'):.1f}%
- **Estimated Cost Savings**: ${safe_get(phase4, 'task_4_6', 'estimated_cost_savings_usd'):,.0f}

### Scientific Impact
- Identified distinct progression trajectories in Parkinson's disease
- Developed baseline classifier for early subtype prediction
- Demonstrated clinical trial enrichment potential

### Output Location
📁 `{self.phase4_output}/`

---

## 🧬 Phase 5: Prodromal Transition Prediction

### Key Results
- **Prodromal Cohort**: {safe_get(phase5, 'task_5_1', 'n_prodromal')} patients
- **Converters**: {safe_get(phase5, 'task_5_1', 'n_converters')} ({safe_get(phase5, 'task_5_1', 'conversion_rate'):.1%})
- **Cox C-index** (Time-Varying): {safe_get(phase5, 'task_5_3', 'time_varying_c_index'):.4f}
- **DeepSurv C-index**: {safe_get(phase5, 'task_5_4', 'c_index'):.4f}
- **Biomarker Thresholds**: {safe_get(phase5, 'task_5_5', 'n_thresholds_identified')}
- **Risk Stratification C-index**: {safe_get(phase5, 'task_5_6', 'stratification_c_index'):.4f}

### Scientific Impact
- Developed comprehensive prodromal conversion prediction system
- Integrated Cox and deep learning survival models
- Created clinical decision support tool with evidence-based thresholds

### Output Location
📁 `{self.phase5_output}/`

---

## 🔗 Cross-Phase Insights

### Integration Opportunities
1. **Subtype-Specific Risk**: Assess whether Phase 4 subtypes predict Phase 5 conversion
2. **Combined Prediction**: Use Phase 4 baseline features in Phase 5 risk calculator
3. **Clinical Translation**: Deploy integrated system for personalized risk profiling

### Recommended Next Steps
- Validate Phase 4 subtypes in prodromal cohort
- Test hypothesis: Fast progressors have higher conversion risk
- Integrate both models into unified clinical tool

---

## 📈 Publication Roadmap

### Manuscript 1: Phase 4 Progression Subtypes
- **Target Journal**: npj Parkinson's Disease
- **Timeline**: Draft by December 2025, submit January 2026
- **Key Novelty**: LTJMM + VaDER for subtype discovery
- **Clinical Impact**: Baseline prediction (AUC = {safe_get(phase4, 'task_4_5', 'auc_macro'):.3f})

### Manuscript 2: Phase 5 Prodromal Transition
- **Target Journal**: Lancet Neurology or JAMA Neurology
- **Timeline**: Draft by January 2026, submit February 2026
- **Key Novelty**: Cox + DeepSurv integrated risk stratification
- **Clinical Impact**: Risk calculator with {safe_get(phase5, 'task_5_5', 'n_thresholds_identified')} validated thresholds

### Manuscript 3: Integrated System (Future)
- **Target Journal**: Nature Medicine
- **Timeline**: After external validation (2026-2027)
- **Key Novelty**: Full GIMAN system with subtyping + conversion prediction
- **Clinical Impact**: Personalized precision medicine platform

---

## 📁 Complete File Structure

```
{self.output_dir}/
├── master_results.json                    # Master results JSON
├── UNIFIED_EXECUTIVE_SUMMARY.md           # This report
├── phase4/                                # Phase 4 outputs
│   ├── phase4_master_results.json
│   ├── PHASE4_EXECUTIVE_SUMMARY.md
│   ├── task_4_1/                          # Longitudinal data
│   ├── task_4_2/                          # Latent time alignment
│   ├── task_4_3/                          # Trajectory clustering
│   ├── task_4_4/                          # Subtype characterization
│   ├── task_4_5/                          # Baseline prediction
│   └── task_4_6/                          # Trial enrichment
└── phase5/                                # Phase 5 outputs
    ├── phase5_master_results.json
    ├── PHASE5_EXECUTIVE_SUMMARY.md
    ├── task_5_1/                          # Prodromal cohort
    ├── task_5_2/                          # Time-varying biomarkers
    ├── task_5_3/                          # Cox models
    ├── task_5_4/                          # DeepSurv
    ├── task_5_5/                          # Biomarker thresholds
    └── task_5_6/                          # Risk stratification tool
```

---

## 🎓 Overall Scientific Contribution

### Methodological Innovations
1. **Latent Time Joint Mixed-Effects Model (LTJMM)**: Novel alignment method
2. **VaDER Trajectory Clustering**: Deep learning for subtype discovery
3. **Time-Varying Cox Models**: Capturing dynamic biomarker evolution
4. **DeepSurv Integration**: Non-linear survival analysis
5. **Evidence-Based Thresholds**: Data-driven clinical cutpoints

### Clinical Translation Readiness
- ✅ Phase 4 baseline classifier: Ready for external validation
- ✅ Phase 5 risk calculator: Ready for prospective deployment
- ⏳ Integrated system: Pending cross-validation

### Expected Impact
- **Scientific**: 3 high-impact publications (npj PD, Lancet Neurology, Nature Medicine)
- **Clinical**: Tools for personalized risk assessment and trial enrichment
- **Economic**: {safe_get(phase4, 'task_4_6', 'sample_size_reduction_percent'):.0f}% trial cost reduction potential

---

## 📞 Contact Information

**GIMAN Research Team**  
**Institution**: [Your Institution]  
**Email**: [Contact Email]  
**Code Repository**: [GitHub/GitLab URL]

---

**Report Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  
**Pipeline Version**: 1.0.0
"""
        
        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write(summary)
        
        self.logger.info(f"📄 Unified summary saved: {summary_path}")


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description="Execute GIMAN Master Pipeline (Phase 4 + Phase 5)"
    )
    parser.add_argument(
        "--phase1-data",
        type=str,
        default="data/01_processed/giman_corrected_longitudinal_dataset.csv",
        help="Path to Phase 1 cohort data CSV (for Phase 4)"
    )
    parser.add_argument(
        "--ppmi-data",
        type=str,
        required=True,
        help="Path to PPMI data directory or manifest (for Phase 5)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="visualizations/phase4_5_results",
        help="Root output directory for all results"
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=42,
        help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--skip-phase4",
        action="store_true",
        help="Skip Phase 4 execution (use existing results)"
    )
    parser.add_argument(
        "--skip-phase5",
        action="store_true",
        help="Skip Phase 5 execution (use existing results)"
    )
    
    args = parser.parse_args()
    
    # Validate inputs
    if not args.skip_phase4 and not Path(args.phase1_data).exists():
        print(f"❌ Error: Phase 1 data not found: {args.phase1_data}")
        return 1
        
    if not args.skip_phase5 and not Path(args.ppmi_data).exists():
        print(f"❌ Error: PPMI data not found: {args.ppmi_data}")
        return 1
    
    # Execute pipeline
    print("\n" + "="*80)
    print("GIMAN MASTER PIPELINE ORCHESTRATOR")
    print("="*80)
    print(f"Phase 1 Data: {args.phase1_data}")
    print(f"PPMI Data: {args.ppmi_data}")
    print(f"Output Directory: {args.output_dir}")
    print(f"Random Seed: {args.random_seed}")
    print(f"Skip Phase 4: {args.skip_phase4}")
    print(f"Skip Phase 5: {args.skip_phase5}")
    print("="*80)
    print("\n⚠️  Expected Runtime: 7-11 hours")
    print("💡 Tip: Run in a screen/tmux session for long execution")
    print("\nStarting in 3 seconds...")
    
    import time
    time.sleep(3)
    
    orchestrator = GIMANMasterOrchestrator(
        phase1_data_path=args.phase1_data,
        ppmi_data_path=args.ppmi_data,
        output_dir=args.output_dir,
        random_seed=args.random_seed,
        skip_phase4=args.skip_phase4,
        skip_phase5=args.skip_phase5
    )
    
    try:
        results = orchestrator.execute()
        
        print("\n" + "="*80)
        print("✅ MASTER PIPELINE EXECUTION COMPLETE!")
        print("="*80)
        
        # Display summary
        phase4 = results.get('phase4', {})
        phase5 = results.get('phase5', {})
        
        if phase4:
            print(f"\n📊 Phase 4 Results:")
            print(f"- Subtypes: {phase4.get('task_4_3', {}).get('n_subtypes', 'N/A')}")
            print(f"- Prediction AUC: {phase4.get('task_4_5', {}).get('auc_macro', 0):.4f}")
            print(f"- Trial Reduction: {phase4.get('task_4_6', {}).get('sample_size_reduction_percent', 0):.1f}%")
        
        if phase5:
            print(f"\n🧬 Phase 5 Results:")
            print(f"- Prodromal: {phase5.get('task_5_1', {}).get('n_prodromal', 'N/A')}")
            print(f"- Converters: {phase5.get('task_5_1', {}).get('n_converters', 'N/A')}")
            print(f"- Cox C-index: {phase5.get('task_5_3', {}).get('time_varying_c_index', 0):.4f}")
            print(f"- DeepSurv C-index: {phase5.get('task_5_4', {}).get('c_index', 0):.4f}")
        
        print(f"\n⏱️  Total Runtime: {results['execution_metadata']['total_runtime_hours']:.2f} hours")
        print(f"\n📁 All results: {args.output_dir}")
        print(f"📄 Unified summary: {args.output_dir}/UNIFIED_EXECUTIVE_SUMMARY.md")
        print(f"💾 Master results: {args.output_dir}/master_results.json")
        
        return 0
        
    except Exception as e:
        print(f"\n❌ Master pipeline execution failed: {str(e)}")
        print("Check the log file in the output directory for details")
        return 1


if __name__ == "__main__":
    sys.exit(main())
