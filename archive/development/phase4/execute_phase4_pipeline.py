#!/usr/bin/env python3
"""
Phase 4: Progression Subtype Discovery - Master Execution Pipeline

This script executes all 6 tasks in Phase 4 sequentially and generates
a comprehensive results report with all outputs.

Expected Runtime: 2-4 hours (depending on hardware)
Expected Outputs: ~50 files including models, visualizations, reports

Author: GIMAN Research Team
Date: October 5, 2025
"""

import sys
import logging
from pathlib import Path
from datetime import datetime
import json
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# Add archive directory to path
sys.path.insert(0, str(Path(__file__).parent / "archive" / "development" / "phase4"))


class Phase4PipelineExecutor:
    """
    Master orchestrator for Phase 4 Progression Subtype Discovery pipeline.
    
    This class runs all 6 tasks in sequence, handles data flow between tasks,
    and generates a comprehensive results report.
    """
    
    def __init__(
        self,
        phase1_data_path: str,
        output_dir: str = "results/phase4_execution",
        random_seed: int = 42
    ):
        """
        Initialize Phase 4 pipeline executor.
        
        Args:
            phase1_data_path: Path to Phase 1 cohort data CSV
            output_dir: Directory for all Phase 4 outputs
            random_seed: Random seed for reproducibility
        """
        self.phase1_data_path = Path(phase1_data_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.random_seed = random_seed
        
        # Setup logging
        self._setup_logging()
        
        # Results storage
        self.results = {
            'task_4_1': {},
            'task_4_2': {},
            'task_4_3': {},
            'task_4_4': {},
            'task_4_5': {},
            'task_4_6': {},
            'pipeline_metadata': {
                'start_time': None,
                'end_time': None,
                'total_runtime_minutes': None,
                'phase1_data_path': str(self.phase1_data_path),
                'random_seed': self.random_seed
            }
        }
        
        self.logger.info("="*80)
        self.logger.info("PHASE 4: PROGRESSION SUBTYPE DISCOVERY PIPELINE")
        self.logger.info("="*80)
        self.logger.info(f"Input data: {self.phase1_data_path}")
        self.logger.info(f"Output directory: {self.output_dir}")
        self.logger.info(f"Random seed: {self.random_seed}")
        
    def _setup_logging(self):
        """Configure logging for pipeline execution."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_file = self.output_dir / f"phase4_pipeline_{timestamp}.log"
        
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
        
    def run_task_4_1(self) -> Path:
        """
        Task 4.1: Longitudinal Data Preparation
        
        Returns:
            Path to longitudinal trajectories CSV
        """
        self.logger.info("\n" + "="*80)
        self.logger.info("TASK 4.1: LONGITUDINAL DATA PREPARATION")
        self.logger.info("="*80)
        
        try:
            from task_4_1_longitudinal_data_prep import LongitudinalDataPreparation
            
            prep = LongitudinalDataPreparation(
                data_path=str(self.phase1_data_path),
                output_dir=str(self.output_dir / "task_4_1"),
                min_timepoints=3,
                cohort_filter="Parkinson's Disease"
            )
            
            # Execute using current Task 4.1 API.
            self.logger.info("Executing Task 4.1 full pipeline...")
            trajectories_df, long_df = prep.run_full_pipeline()
            output_path = prep.output_dir / "patient_trajectory_features.csv"
            qc_metrics = {
                "n_patients": int(trajectories_df["PATNO"].nunique()) if "PATNO" in trajectories_df.columns else int(len(trajectories_df)),
                "n_total_visits": int(len(long_df)),
                "mean_visits": float(long_df.groupby("PATNO").size().mean()) if "PATNO" in long_df.columns and len(long_df) > 0 else 0.0,
            }
            
            # Store results
            self.results['task_4_1'] = {
                'n_patients': int(qc_metrics.get('n_patients', 0)),
                'n_total_visits': int(qc_metrics.get('n_total_visits', 0)),
                'mean_visits_per_patient': float(qc_metrics.get('mean_visits', 0)),
                'output_path': str(output_path),
                'qc_metrics': {k: float(v) if isinstance(v, (int, float, np.number)) else v 
                              for k, v in qc_metrics.items()}
            }
            
            self.logger.info(f"✅ Task 4.1 complete: {self.results['task_4_1']['n_patients']} patients")
            return output_path
            
        except Exception as e:
            self.logger.error(f"Task 4.1 failed: {str(e)}", exc_info=True)
            raise
        
    def run_task_4_2(self, longitudinal_data_path: Path) -> Path:
        """
        Task 4.2: Latent Time Joint Mixed-Effects Model
        
        Args:
            longitudinal_data_path: Path from Task 4.1
            
        Returns:
            Path to aligned data with latent disease time
        """
        self.logger.info("\n" + "="*80)
        self.logger.info("TASK 4.2: LATENT TIME ALIGNMENT (LTJMM)")
        self.logger.info("="*80)
        
        try:
            from task_4_2_latent_time_alignment import LatentTimeJointModel
            
            ltjmm = LatentTimeJointModel(
                longitudinal_data_path=str(longitudinal_data_path),
                output_dir=str(self.output_dir / "task_4_2")
            )
            
            self.logger.info("Fitting latent time model...")
            ltjmm.fit_model()
            
            self.logger.info("Aligning trajectories to disease time...")
            aligned_df = ltjmm.align_trajectories()
            
            self.logger.info("Evaluating alignment quality...")
            alignment_metrics = ltjmm.evaluate_alignment()
            
            self.logger.info("Generating visualizations...")
            ltjmm.visualize_alignment()
            
            self.logger.info("Saving results...")
            output_path = ltjmm.save_results()
            
            # Store results
            self.results['task_4_2'] = {
                'n_patients': int(len(aligned_df['PATNO'].unique())) if 'PATNO' in aligned_df.columns else 0,
                'alignment_quality': float(alignment_metrics.get('r2', 0)),
                'convergence_iterations': int(alignment_metrics.get('n_iterations', 0)),
                'output_path': str(output_path)
            }
            
            self.logger.info(
                f"✅ Task 4.2 complete: Alignment R² = "
                f"{self.results['task_4_2']['alignment_quality']:.4f}"
            )
            return output_path
            
        except Exception as e:
            self.logger.error(f"Task 4.2 failed: {str(e)}", exc_info=True)
            raise
        
    def run_task_4_3(self, aligned_data_path: Path) -> tuple[Path, dict]:
        """
        Task 4.3: VaDER Trajectory Clustering
        
        Args:
            aligned_data_path: Path from Task 4.2
            
        Returns:
            Tuple of (subtype labels path, clustering metrics)
        """
        self.logger.info("\n" + "="*80)
        self.logger.info("TASK 4.3: TRAJECTORY CLUSTERING (VaDER)")
        self.logger.info("="*80)
        
        try:
            from task_4_3_trajectory_clustering import TrajectoryClustering
            
            clustering = TrajectoryClustering(
                aligned_data_path=str(aligned_data_path),
                output_dir=str(self.output_dir / "task_4_3"),
                n_clusters=3,  # Fast, Moderate, Slow
                random_state=self.random_seed
            )
            
            self.logger.info("Training VaDER model...")
            clustering.train_vader_model()
            
            self.logger.info("Performing clustering...")
            subtype_labels = clustering.cluster_trajectories()
            
            self.logger.info("Evaluating clustering quality...")
            clustering_metrics = clustering.evaluate_clustering()
            
            self.logger.info("Comparing clustering methods...")
            method_comparison = clustering.compare_methods()
            
            self.logger.info("Generating visualizations...")
            clustering.visualize_clusters()
            
            self.logger.info("Saving results...")
            output_path = clustering.save_results()
            
            # Store results
            subtype_dist = subtype_labels['cluster'].value_counts().to_dict() if 'cluster' in subtype_labels.columns else {}
            
            self.results['task_4_3'] = {
                'n_subtypes': int(clustering_metrics.get('n_clusters', 3)),
                'silhouette_score': float(clustering_metrics.get('silhouette', 0)),
                'calinski_harabasz': float(clustering_metrics.get('calinski_harabasz', 0)),
                'davies_bouldin': float(clustering_metrics.get('davies_bouldin', 0)),
                'subtype_distribution': {str(k): int(v) for k, v in subtype_dist.items()},
                'best_method': str(method_comparison.get('best_method', 'VaDER')),
                'output_path': str(output_path)
            }
            
            self.logger.info(
                f"✅ Task 4.3 complete: {self.results['task_4_3']['n_subtypes']} subtypes, "
                f"silhouette = {self.results['task_4_3']['silhouette_score']:.4f}"
            )
            return output_path, clustering_metrics
            
        except Exception as e:
            self.logger.error(f"Task 4.3 failed: {str(e)}", exc_info=True)
            raise
        
    def run_task_4_4(
        self, 
        subtype_labels_path: Path,
        aligned_observations_path: Path
    ) -> Path:
        """
        Task 4.4: Subtype Characterization
        
        Args:
            subtype_labels_path: Path from Task 4.3
            aligned_observations_path: Path from Task 4.2
            
        Returns:
            Path to subtype characterization report
        """
        self.logger.info("\n" + "="*80)
        self.logger.info("TASK 4.4: SUBTYPE CHARACTERIZATION")
        self.logger.info("="*80)
        
        try:
            from task_4_4_subtype_characterization import SubtypeCharacterization
            
            char = SubtypeCharacterization(
                clustered_trajectories_path=str(subtype_labels_path),
                aligned_observations_path=str(aligned_observations_path),
                output_dir=str(self.output_dir / "task_4_4")
            )
            
            self.logger.info("Comparing baseline characteristics...")
            baseline_comparison = char.compare_baseline_characteristics()
            
            self.logger.info("Identifying distinguishing features...")
            distinguishing_features = char.identify_distinguishing_features()
            
            self.logger.info("Characterizing progression patterns...")
            progression_patterns = char.characterize_progression_patterns()
            
            self.logger.info("Analyzing milestone progression...")
            milestone_analysis = char.analyze_milestone_progression()
            
            self.logger.info("Generating visualizations...")
            char.visualize_subtype_profiles()
            
            self.logger.info("Creating characterization report...")
            report_path = char.generate_characterization_report()
            
            # Store results
            self.results['task_4_4'] = {
                'n_significant_features': int(len(distinguishing_features)) if distinguishing_features else 0,
                'top_distinguishing_features': list(distinguishing_features.keys())[:10] if distinguishing_features else [],
                'progression_patterns': {str(k): str(v) for k, v in progression_patterns.items()} if progression_patterns else {},
                'output_path': str(report_path)
            }
            
            self.logger.info(
                f"✅ Task 4.4 complete: "
                f"{self.results['task_4_4']['n_significant_features']} distinguishing features identified"
            )
            return report_path
            
        except Exception as e:
            self.logger.error(f"Task 4.4 failed: {str(e)}", exc_info=True)
            raise
        
    def run_task_4_5(
        self,
        subtype_labels_path: Path,
        longitudinal_data_path: Path
    ) -> tuple[Path, dict]:
        """
        Task 4.5: Baseline Subtype Prediction
        
        Args:
            subtype_labels_path: Path from Task 4.3
            longitudinal_data_path: Path from Task 4.1
            
        Returns:
            Tuple of (model path, performance metrics)
        """
        self.logger.info("\n" + "="*80)
        self.logger.info("TASK 4.5: BASELINE SUBTYPE PREDICTION")
        self.logger.info("="*80)
        
        try:
            from task_4_5_baseline_subtype_prediction import BaselineSubtypePrediction
            
            predictor = BaselineSubtypePrediction(
                subtype_labels_path=str(subtype_labels_path),
                longitudinal_data_path=str(longitudinal_data_path),
                output_dir=str(self.output_dir / "task_4_5"),
                random_state=self.random_seed
            )
            
            self.logger.info("Preparing baseline features (0-12 months)...")
            X_train, X_test, y_train, y_test = predictor.prepare_baseline_features()
            
            self.logger.info("Training classifiers...")
            models = predictor.train_multiple_classifiers(X_train, y_train)
            
            self.logger.info("Evaluating models...")
            performance_metrics = predictor.evaluate_models(X_test, y_test)
            
            self.logger.info("Analyzing feature importance...")
            feature_importance = predictor.analyze_feature_importance()
            
            self.logger.info("Calibration analysis...")
            calibration_results = predictor.calibration_analysis(X_test, y_test)
            
            self.logger.info("Generating visualizations...")
            predictor.visualize_performance()
            
            self.logger.info("Saving best model...")
            model_path = predictor.save_best_model()
            
            # Store results
            best_model_metrics = performance_metrics.get('best_model', {})
            
            self.results['task_4_5'] = {
                'best_model': str(best_model_metrics.get('model_name', 'Unknown')),
                'auc_macro': float(best_model_metrics.get('auc_macro', 0)),
                'accuracy': float(best_model_metrics.get('accuracy', 0)),
                'f1_macro': float(best_model_metrics.get('f1_macro', 0)),
                'balanced_accuracy': float(best_model_metrics.get('balanced_accuracy', 0)),
                'top_features': list(feature_importance.keys())[:10] if feature_importance else [],
                'model_path': str(model_path),
                'all_models_performance': {
                    str(k): {str(k2): float(v2) if isinstance(v2, (int, float, np.number)) else v2 
                            for k2, v2 in v.items()} 
                    for k, v in performance_metrics.items() if k != 'best_model'
                }
            }
            
            self.logger.info(
                f"✅ Task 4.5 complete: Best model ({self.results['task_4_5']['best_model']}) "
                f"AUC = {self.results['task_4_5']['auc_macro']:.4f}"
            )
            return model_path, best_model_metrics
            
        except Exception as e:
            self.logger.error(f"Task 4.5 failed: {str(e)}", exc_info=True)
            raise
        
    def run_task_4_6(
        self,
        subtype_labels_path: Path,
        prediction_metrics: dict
    ) -> Path:
        """
        Task 4.6: Clinical Trial Enrichment Simulation
        
        Args:
            subtype_labels_path: Path from Task 4.3
            prediction_metrics: Performance metrics from Task 4.5
            
        Returns:
            Path to trial enrichment report
        """
        self.logger.info("\n" + "="*80)
        self.logger.info("TASK 4.6: CLINICAL TRIAL ENRICHMENT SIMULATION")
        self.logger.info("="*80)
        
        try:
            from task_4_6_trial_enrichment_simulation import ClinicalTrialEnrichmentSimulation
            
            simulator = ClinicalTrialEnrichmentSimulation(
                labeled_trajectories_path=str(subtype_labels_path),
                output_dir=str(self.output_dir / "task_4_6"),
                n_simulations=1000,
                trial_duration_years=2.0,
                alpha=0.05,
                power_target=0.80
            )
            
            self.logger.info("Defining trial scenarios...")
            scenarios = simulator.define_trial_scenarios()
            
            self.logger.info("Running Monte Carlo simulations (1000 iterations)...")
            simulation_results = simulator.run_monte_carlo_simulations()
            
            self.logger.info("Performing power analysis...")
            power_analysis = simulator.power_analysis()
            
            self.logger.info("Calculating sample size reduction...")
            sample_size_analysis = simulator.calculate_sample_size_reduction()
            
            self.logger.info("Estimating cost savings...")
            cost_analysis = simulator.estimate_cost_savings()
            
            self.logger.info("Generating visualizations...")
            simulator.visualize_trial_simulations()
            
            self.logger.info("Creating enrichment report...")
            report_path = simulator.generate_enrichment_report()
            
            # Store results
            self.results['task_4_6'] = {
                'sample_size_reduction_percent': float(sample_size_analysis.get('reduction_percent', 0)),
                'power_fast_enriched': float(power_analysis.get('power_fast_enriched', 0)),
                'power_all_comers': float(power_analysis.get('power_all_comers', 0)),
                'power_improvement': float(power_analysis.get('power_improvement', 0)),
                'estimated_cost_savings_usd': float(cost_analysis.get('cost_savings', 0)),
                'n_simulations': 1000,
                'output_path': str(report_path)
            }
            
            self.logger.info(
                f"✅ Task 4.6 complete: "
                f"{self.results['task_4_6']['sample_size_reduction_percent']:.1f}% sample size reduction, "
                f"${self.results['task_4_6']['estimated_cost_savings_usd']:,.0f} savings"
            )
            return report_path
            
        except Exception as e:
            self.logger.error(f"Task 4.6 failed: {str(e)}", exc_info=True)
            raise
        
    def execute(self) -> dict:
        """
        Execute full Phase 4 pipeline.
        
        Returns:
            Complete results dictionary
        """
        start_time = datetime.now()
        self.results['pipeline_metadata']['start_time'] = start_time.isoformat()
        
        self.logger.info("\n" + "🚀"*40)
        self.logger.info("STARTING PHASE 4 PIPELINE EXECUTION")
        self.logger.info("🚀"*40 + "\n")
        
        try:
            # Task 4.1: Longitudinal Data Prep
            self.logger.info("📊 Executing Task 4.1...")
            longitudinal_path = self.run_task_4_1()
            
            # Task 4.2: Latent Time Alignment
            self.logger.info("\n⏱️  Executing Task 4.2...")
            aligned_path = self.run_task_4_2(longitudinal_path)
            
            # Task 4.3: Trajectory Clustering
            self.logger.info("\n🎯 Executing Task 4.3...")
            subtype_path, clustering_metrics = self.run_task_4_3(aligned_path)
            
            # Task 4.4: Subtype Characterization
            self.logger.info("\n📋 Executing Task 4.4...")
            char_report_path = self.run_task_4_4(subtype_path, aligned_path)
            
            # Task 4.5: Baseline Prediction
            self.logger.info("\n🤖 Executing Task 4.5...")
            model_path, pred_metrics = self.run_task_4_5(subtype_path, longitudinal_path)
            
            # Task 4.6: Trial Enrichment
            self.logger.info("\n💊 Executing Task 4.6...")
            enrichment_report_path = self.run_task_4_6(subtype_path, pred_metrics)
            
            # Finalize
            end_time = datetime.now()
            runtime = (end_time - start_time).total_seconds() / 60
            
            self.results['pipeline_metadata']['end_time'] = end_time.isoformat()
            self.results['pipeline_metadata']['total_runtime_minutes'] = runtime
            
            # Save master results
            self._save_master_results()
            
            # Generate executive summary
            self._generate_executive_summary()
            
            self.logger.info("\n" + "🎉"*40)
            self.logger.info("PHASE 4 PIPELINE COMPLETE!")
            self.logger.info("🎉"*40)
            self.logger.info(f"\n⏱️  Total runtime: {runtime:.1f} minutes ({runtime/60:.2f} hours)")
            self.logger.info(f"📁 Results saved to: {self.output_dir}")
            self.logger.info(f"📊 Executive summary: {self.output_dir}/PHASE4_EXECUTIVE_SUMMARY.md")
            
            return self.results
            
        except Exception as e:
            self.logger.error(f"\n❌ Pipeline failed: {str(e)}", exc_info=True)
            self.logger.error("Check the log file for detailed error information")
            raise
            
    def _save_master_results(self):
        """Save master results JSON."""
        results_path = self.output_dir / "phase4_master_results.json"
        with open(results_path, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        self.logger.info(f"\n💾 Master results saved: {results_path}")
        
    def _generate_executive_summary(self):
        """Generate markdown executive summary."""
        summary_path = self.output_dir / "PHASE4_EXECUTIVE_SUMMARY.md"
        
        summary = f"""# Phase 4: Progression Subtype Discovery - Executive Summary

**Pipeline Execution Date**: {self.results['pipeline_metadata']['start_time']}  
**Total Runtime**: {self.results['pipeline_metadata']['total_runtime_minutes']:.1f} minutes  
**Random Seed**: {self.results['pipeline_metadata']['random_seed']}

---

## 🎯 Key Findings

### Cohort Size
- **Total Patients**: {self.results['task_4_1']['n_patients']}
- **Total Visits**: {self.results['task_4_1']['n_total_visits']}
- **Mean Visits/Patient**: {self.results['task_4_1']['mean_visits_per_patient']:.1f}

### Discovered Subtypes
- **Number of Subtypes**: {self.results['task_4_3']['n_subtypes']}
- **Clustering Quality** (Silhouette Score): {self.results['task_4_3'].get('silhouette_score', 0):.4f}
- **Best Clustering Method**: {self.results['task_4_3'].get('best_method', 'N/A')}

#### Subtype Distribution
{self._format_subtype_distribution()}

### Latent Time Alignment
- **Alignment Quality (R²)**: {self.results['task_4_2'].get('alignment_quality', 0):.4f}
- **Convergence Iterations**: {self.results['task_4_2'].get('convergence_iterations', 0)}

### Baseline Subtype Prediction
- **Best Model**: {self.results['task_4_5'].get('best_model', 'N/A')}
- **AUC (Macro)**: {self.results['task_4_5'].get('auc_macro', 0):.4f}
- **Accuracy**: {self.results['task_4_5'].get('accuracy', 0):.4f}
- **F1 Score (Macro)**: {self.results['task_4_5'].get('f1_macro', 0):.4f}
- **Balanced Accuracy**: {self.results['task_4_5'].get('balanced_accuracy', 0):.4f}

#### Top Predictive Features
{self._format_top_features()}

### Clinical Trial Impact
- **Sample Size Reduction**: {self.results['task_4_6'].get('sample_size_reduction_percent', 0):.1f}%
- **Power (Fast-Enriched)**: {self.results['task_4_6'].get('power_fast_enriched', 0):.1%}
- **Power (All-Comers)**: {self.results['task_4_6'].get('power_all_comers', 0):.1%}
- **Power Improvement**: {self.results['task_4_6'].get('power_improvement', 0):.1%}
- **Estimated Cost Savings**: ${self.results['task_4_6'].get('estimated_cost_savings_usd', 0):,.0f}
- **Monte Carlo Simulations**: {self.results['task_4_6'].get('n_simulations', 0)}

---

## 📁 Output Files

All results are saved in: `{self.output_dir}`

### Task 4.1: Longitudinal Data
- 📄 `{Path(self.results['task_4_1']['output_path']).name}`
- Location: `{self.output_dir}/task_4_1/`

### Task 4.2: Latent Time Alignment
- 📄 `{Path(self.results['task_4_2']['output_path']).name}`
- Location: `{self.output_dir}/task_4_2/`

### Task 4.3: Trajectory Clustering
- 📄 `{Path(self.results['task_4_3']['output_path']).name}`
- Location: `{self.output_dir}/task_4_3/`

### Task 4.4: Subtype Characterization
- 📄 `{Path(self.results['task_4_4']['output_path']).name}`
- 🔬 {self.results['task_4_4']['n_significant_features']} significant distinguishing features identified
- Location: `{self.output_dir}/task_4_4/`

### Task 4.5: Baseline Prediction
- 🤖 Model: `{Path(self.results['task_4_5']['model_path']).name}`
- Location: `{self.output_dir}/task_4_5/`

### Task 4.6: Trial Enrichment
- 📊 Report: `{Path(self.results['task_4_6']['output_path']).name}`
- Location: `{self.output_dir}/task_4_6/`

---

## 🎓 Scientific Impact

### Clinical Utility
1. **Personalized Prognosis**: Patients can be classified into progression subtypes at baseline
2. **Treatment Stratification**: Different subtypes may respond differently to interventions
3. **Clinical Trial Efficiency**: {self.results['task_4_6'].get('sample_size_reduction_percent', 0):.0f}% sample size reduction through fast-progressor enrichment

### Publication Potential
- **Target Journal**: npj Parkinson's Disease
- **Key Novelty**: First comprehensive progression subtype discovery using latent time alignment and deep learning
- **Clinical Translation**: Baseline prediction model ready for external validation

---

## 📝 Next Steps

1. **External Validation**: Test subtype model on PDBP cohort
2. **Biomarker Integration**: Incorporate CSF and imaging biomarkers for subtype characterization
3. **Manuscript Preparation**: Draft methods and results sections (target: January 2026)
4. **Clinical Collaboration**: Partner with movement disorder centers for prospective validation
5. **Integration with Phase 5**: Assess subtype-specific prodromal conversion risk

---

## 🔗 Related Analyses

- **Phase 5**: Prodromal-to-Clinical Transition (assess conversion risk by subtype)
- **Phase 6**: GNN Explainability (understand subtype-specific attention patterns)
- **Research Plan Phase 2**: Prognostic model baseline (R² = 0.0346)

---

**Report Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  
**Pipeline Version**: 1.0.0  
**Contact**: GIMAN Research Team
"""
        
        with open(summary_path, 'w') as f:
            f.write(summary)
        
        self.logger.info(f"📄 Executive summary saved: {summary_path}")
        
    def _format_subtype_distribution(self) -> str:
        """Format subtype distribution for markdown."""
        dist = self.results['task_4_3'].get('subtype_distribution', {})
        total = sum(dist.values()) if dist else 1
        lines = []
        for subtype, count in sorted(dist.items()):
            pct = (count / total) * 100
            lines.append(f"- **Subtype {subtype}**: {count} patients ({pct:.1f}%)")
        return "\n".join(lines) if lines else "- No distribution data available"
        
    def _format_top_features(self) -> str:
        """Format top features for markdown."""
        features = self.results['task_4_5'].get('top_features', [])
        lines = []
        for i, feature in enumerate(features[:5], 1):
            lines.append(f"{i}. `{feature}`")
        return "\n".join(lines) if lines else "- No feature importance data available"


def main():
    """Main execution function."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Execute Phase 4: Progression Subtype Discovery Pipeline"
    )
    parser.add_argument(
        "--phase1-data",
        type=str,
        default="giman_expanded_cohort_final.csv",
        help="Path to Phase 1 cohort data CSV"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/phase4_execution",
        help="Output directory for all results"
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=42,
        help="Random seed for reproducibility"
    )
    
    args = parser.parse_args()
    
    # Execute pipeline
    print("\n" + "="*80)
    print("PHASE 4: PROGRESSION SUBTYPE DISCOVERY PIPELINE")
    print("="*80)
    print(f"Phase 1 Data: {args.phase1_data}")
    print(f"Output Directory: {args.output_dir}")
    print(f"Random Seed: {args.random_seed}")
    print("="*80 + "\n")
    
    executor = Phase4PipelineExecutor(
        phase1_data_path=args.phase1_data,
        output_dir=args.output_dir,
        random_seed=args.random_seed
    )
    
    try:
        results = executor.execute()
        
        print("\n" + "="*80)
        print("✅ PHASE 4 PIPELINE EXECUTION COMPLETE!")
        print("="*80)
        print(f"\n📊 Key Results:")
        print(f"- Subtypes Discovered: {results['task_4_3']['n_subtypes']}")
        print(f"- Baseline Prediction AUC: {results['task_4_5'].get('auc_macro', 0):.4f}")
        print(f"- Trial Sample Reduction: {results['task_4_6'].get('sample_size_reduction_percent', 0):.1f}%")
        print(f"- Cost Savings: ${results['task_4_6'].get('estimated_cost_savings_usd', 0):,.0f}")
        print(f"\n📁 All results saved to: {args.output_dir}")
        print(f"📄 Executive summary: {args.output_dir}/PHASE4_EXECUTIVE_SUMMARY.md")
        print(f"💾 Master results: {args.output_dir}/phase4_master_results.json")
        
        return 0
        
    except Exception as e:
        print(f"\n❌ Pipeline execution failed: {str(e)}")
        print("Check the log file in the output directory for details")
        return 1


if __name__ == "__main__":
    sys.exit(main())
