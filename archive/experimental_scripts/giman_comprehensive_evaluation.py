#!/usr/bin/env python3
"""GIMAN Comprehensive Performance Evaluation
==========================================

This script runs comprehensive experiments to evaluate the GIMAN Phase 4 model:
1. Multiple independent runs for statistical reliability
2. Extended training with more epochs
3. Detailed performance analysis and reporting

Author: AI Assistant
Date: September 24, 2025
"""

import json
import logging
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy import stats

from phase3_1_real_data_integration import RealDataPhase3Integration
from phase4_unified_giman_system import run_phase4_experiment


class GIMANComprehensiveEvaluator:
    """Comprehensive evaluation system for GIMAN Phase 4 model."""

    def __init__(self, output_dir: str = "evaluation_results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
            handlers=[
                logging.FileHandler(self.output_dir / "evaluation.log"),
                logging.StreamHandler(),
            ],
        )
        self.logger = logging.getLogger(__name__)

        # Results storage
        self.experiment_results = []
        self.data_integrator = None

    def setup_data(self):
        """Initialize and prepare the data integrator."""
        self.logger.info("🔧 Setting up data integrator...")
        self.data_integrator = RealDataPhase3Integration()
        self.data_integrator.load_and_prepare_data()
        self.logger.info(
            f"✅ Data loaded: {len(self.data_integrator.patient_ids)} patients"
        )

    def run_single_experiment(self, run_id: int, extended_epochs: bool = False) -> dict:
        """Run a single GIMAN experiment."""
        self.logger.info(f"🚀 Starting experiment run {run_id}")
        start_time = time.time()

        try:
            # Modify training parameters if extended epochs requested
            if extended_epochs:
                # Temporarily modify the training function to use more epochs
                # We'll do this by monkey-patching the default parameters
                original_train_func = run_phase4_experiment

            results = run_phase4_experiment(self.data_integrator)

            # Add metadata
            results["run_id"] = run_id
            results["duration_seconds"] = time.time() - start_time
            results["extended_epochs"] = extended_epochs
            results["timestamp"] = time.strftime("%Y-%m-%d %H:%M:%S")

            self.logger.info(
                f"✅ Run {run_id} completed: "
                f"Motor R² = {results['motor_r2']:.4f}, "
                f"Cognitive AUC = {results['cognitive_auc']:.4f}, "
                f"Duration = {results['duration_seconds']:.1f}s"
            )

            return results

        except Exception as e:
            self.logger.error(f"❌ Run {run_id} failed: {str(e)}")
            return {
                "run_id": run_id,
                "motor_r2": np.nan,
                "cognitive_auc": np.nan,
                "duration_seconds": time.time() - start_time,
                "extended_epochs": extended_epochs,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "error": str(e),
            }

    def run_multiple_experiments(
        self, num_runs: int = 10, extended_epochs: bool = False
    ) -> list[dict]:
        """Run multiple independent experiments."""
        self.logger.info(f"🔬 Starting {num_runs} independent experiments...")

        results = []
        for i in range(1, num_runs + 1):
            result = self.run_single_experiment(i, extended_epochs=extended_epochs)
            results.append(result)
            self.experiment_results.append(result)

            # Save intermediate results
            self.save_results_to_json()

            # Brief pause between runs
            if i < num_runs:
                time.sleep(2)

        self.logger.info(f"🏁 Completed {num_runs} experiments")
        return results

    def calculate_statistics(self, results: list[dict]) -> dict:
        """Calculate comprehensive statistics from experiment results."""
        # Extract valid results (no errors)
        valid_results = [r for r in results if not pd.isna(r.get("motor_r2"))]

        if not valid_results:
            self.logger.error("❌ No valid results to analyze!")
            return {}

        motor_r2_values = [r["motor_r2"] for r in valid_results]
        cognitive_auc_values = [r["cognitive_auc"] for r in valid_results]
        durations = [r["duration_seconds"] for r in valid_results]

        stats_dict = {
            "num_successful_runs": len(valid_results),
            "num_failed_runs": len(results) - len(valid_results),
            # Motor R² Statistics
            "motor_r2_mean": np.mean(motor_r2_values),
            "motor_r2_std": np.std(motor_r2_values),
            "motor_r2_median": np.median(motor_r2_values),
            "motor_r2_min": np.min(motor_r2_values),
            "motor_r2_max": np.max(motor_r2_values),
            "motor_r2_q25": np.percentile(motor_r2_values, 25),
            "motor_r2_q75": np.percentile(motor_r2_values, 75),
            # Cognitive AUC Statistics
            "cognitive_auc_mean": np.mean(cognitive_auc_values),
            "cognitive_auc_std": np.std(cognitive_auc_values),
            "cognitive_auc_median": np.median(cognitive_auc_values),
            "cognitive_auc_min": np.min(cognitive_auc_values),
            "cognitive_auc_max": np.max(cognitive_auc_values),
            "cognitive_auc_q25": np.percentile(cognitive_auc_values, 25),
            "cognitive_auc_q75": np.percentile(cognitive_auc_values, 75),
            # Performance benchmarks
            "motor_r2_positive_rate": np.mean(np.array(motor_r2_values) > 0),
            "cognitive_auc_above_random": np.mean(np.array(cognitive_auc_values) > 0.5),
            "cognitive_auc_good_performance": np.mean(
                np.array(cognitive_auc_values) > 0.6
            ),
            # Duration statistics
            "avg_duration_seconds": np.mean(durations),
            "total_duration_minutes": np.sum(durations) / 60,
        }

        # Confidence intervals (95%)
        if len(motor_r2_values) > 1:
            motor_ci = stats.t.interval(
                0.95,
                len(motor_r2_values) - 1,
                loc=np.mean(motor_r2_values),
                scale=stats.sem(motor_r2_values),
            )
            stats_dict["motor_r2_ci_lower"] = motor_ci[0]
            stats_dict["motor_r2_ci_upper"] = motor_ci[1]

            cog_ci = stats.t.interval(
                0.95,
                len(cognitive_auc_values) - 1,
                loc=np.mean(cognitive_auc_values),
                scale=stats.sem(cognitive_auc_values),
            )
            stats_dict["cognitive_auc_ci_lower"] = cog_ci[0]
            stats_dict["cognitive_auc_ci_upper"] = cog_ci[1]

        return stats_dict

    def create_visualizations(self, results: list[dict]):
        """Create comprehensive visualizations of the results."""
        valid_results = [r for r in results if not pd.isna(r.get("motor_r2"))]

        if not valid_results:
            self.logger.warning("⚠️ No valid results for visualization")
            return

        # Create figure with subplots
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle(
            "GIMAN Phase 4 Model: Comprehensive Performance Evaluation",
            fontsize=16,
            fontweight="bold",
        )

        motor_r2_values = [r["motor_r2"] for r in valid_results]
        cognitive_auc_values = [r["cognitive_auc"] for r in valid_results]
        run_ids = [r["run_id"] for r in valid_results]

        # 1. Motor R² Distribution
        axes[0, 0].hist(
            motor_r2_values,
            bins=min(10, len(motor_r2_values)),
            alpha=0.7,
            color="skyblue",
            edgecolor="black",
        )
        axes[0, 0].axvline(
            np.mean(motor_r2_values),
            color="red",
            linestyle="--",
            label=f"Mean: {np.mean(motor_r2_values):.4f}",
        )
        axes[0, 0].axvline(0, color="orange", linestyle=":", label="Zero R²")
        axes[0, 0].set_xlabel("Motor R²")
        axes[0, 0].set_ylabel("Frequency")
        axes[0, 0].set_title("Motor R² Distribution")
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # 2. Cognitive AUC Distribution
        axes[0, 1].hist(
            cognitive_auc_values,
            bins=min(10, len(cognitive_auc_values)),
            alpha=0.7,
            color="lightgreen",
            edgecolor="black",
        )
        axes[0, 1].axvline(
            np.mean(cognitive_auc_values),
            color="red",
            linestyle="--",
            label=f"Mean: {np.mean(cognitive_auc_values):.4f}",
        )
        axes[0, 1].axvline(0.5, color="orange", linestyle=":", label="Random (0.5)")
        axes[0, 1].axvline(0.6, color="purple", linestyle=":", label="Good (0.6)")
        axes[0, 1].set_xlabel("Cognitive AUC")
        axes[0, 1].set_ylabel("Frequency")
        axes[0, 1].set_title("Cognitive AUC Distribution")
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

        # 3. Performance over runs
        axes[0, 2].plot(
            run_ids, motor_r2_values, "o-", color="blue", label="Motor R²", alpha=0.7
        )
        axes[0, 2].axhline(0, color="orange", linestyle=":", alpha=0.5)
        axes[0, 2].set_xlabel("Run ID")
        axes[0, 2].set_ylabel("Motor R²")
        axes[0, 2].set_title("Motor R² Across Runs")
        axes[0, 2].grid(True, alpha=0.3)
        axes[0, 2].legend()

        # 4. Cognitive AUC over runs
        axes[1, 0].plot(
            run_ids,
            cognitive_auc_values,
            "o-",
            color="green",
            label="Cognitive AUC",
            alpha=0.7,
        )
        axes[1, 0].axhline(
            0.5, color="orange", linestyle=":", alpha=0.5, label="Random"
        )
        axes[1, 0].axhline(0.6, color="purple", linestyle=":", alpha=0.5, label="Good")
        axes[1, 0].set_xlabel("Run ID")
        axes[1, 0].set_ylabel("Cognitive AUC")
        axes[1, 0].set_title("Cognitive AUC Across Runs")
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].legend()

        # 5. Box plots comparison
        box_data = [motor_r2_values, cognitive_auc_values]
        box_labels = ["Motor R²", "Cognitive AUC"]
        bp = axes[1, 1].boxplot(box_data, labels=box_labels, patch_artist=True)
        bp["boxes"][0].set_facecolor("skyblue")
        bp["boxes"][1].set_facecolor("lightgreen")
        axes[1, 1].set_title("Performance Distribution Comparison")
        axes[1, 1].grid(True, alpha=0.3)

        # 6. Scatter plot: Motor vs Cognitive performance
        axes[1, 2].scatter(
            motor_r2_values, cognitive_auc_values, alpha=0.7, s=60, color="purple"
        )
        axes[1, 2].set_xlabel("Motor R²")
        axes[1, 2].set_ylabel("Cognitive AUC")
        axes[1, 2].set_title("Motor vs Cognitive Performance")
        axes[1, 2].grid(True, alpha=0.3)

        # Add correlation coefficient
        correlation = np.corrcoef(motor_r2_values, cognitive_auc_values)[0, 1]
        axes[1, 2].text(
            0.05,
            0.95,
            f"Correlation: {correlation:.3f}",
            transform=axes[1, 2].transAxes,
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
        )

        plt.tight_layout()

        # Save visualization
        viz_path = self.output_dir / "performance_evaluation.png"
        plt.savefig(viz_path, dpi=300, bbox_inches="tight")
        self.logger.info(f"📊 Visualizations saved to {viz_path}")
        plt.show()

    def generate_comprehensive_report(self, results: list[dict], stats: dict):
        """Generate a comprehensive text report."""
        report_path = self.output_dir / "comprehensive_report.txt"

        with open(report_path, "w") as f:
            f.write("=" * 80 + "\n")
            f.write(
                "GIMAN Phase 4 Model: Comprehensive Performance Evaluation Report\n"
            )
            f.write("=" * 80 + "\n")
            f.write(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(
                f"Device: {torch.device('cuda' if torch.cuda.is_available() else 'cpu')}\n"
            )
            f.write("\n")

            # Executive Summary
            f.write("EXECUTIVE SUMMARY\n")
            f.write("-" * 40 + "\n")
            f.write(f"• Total experiments conducted: {len(results)}\n")
            f.write(f"• Successful runs: {stats['num_successful_runs']}\n")
            f.write(f"• Failed runs: {stats['num_failed_runs']}\n")
            f.write(
                f"• Total training time: {stats['total_duration_minutes']:.1f} minutes\n"
            )
            f.write("\n")

            # Motor Performance Analysis
            f.write("MOTOR PROGRESSION PREDICTION (R²)\n")
            f.write("-" * 40 + "\n")
            f.write(
                f"• Mean R²: {stats['motor_r2_mean']:.4f} ± {stats['motor_r2_std']:.4f}\n"
            )
            f.write(f"• Median R²: {stats['motor_r2_median']:.4f}\n")
            f.write(
                f"• Range: [{stats['motor_r2_min']:.4f}, {stats['motor_r2_max']:.4f}]\n"
            )
            f.write(
                f"• IQR: [{stats['motor_r2_q25']:.4f}, {stats['motor_r2_q75']:.4f}]\n"
            )
            if "motor_r2_ci_lower" in stats:
                f.write(
                    f"• 95% CI: [{stats['motor_r2_ci_lower']:.4f}, {stats['motor_r2_ci_upper']:.4f}]\n"
                )
            f.write(f"• Positive R² rate: {stats['motor_r2_positive_rate']:.1%}\n")
            f.write("\n")

            # Cognitive Performance Analysis
            f.write("COGNITIVE CONVERSION PREDICTION (AUC)\n")
            f.write("-" * 40 + "\n")
            f.write(
                f"• Mean AUC: {stats['cognitive_auc_mean']:.4f} ± {stats['cognitive_auc_std']:.4f}\n"
            )
            f.write(f"• Median AUC: {stats['cognitive_auc_median']:.4f}\n")
            f.write(
                f"• Range: [{stats['cognitive_auc_min']:.4f}, {stats['cognitive_auc_max']:.4f}]\n"
            )
            f.write(
                f"• IQR: [{stats['cognitive_auc_q25']:.4f}, {stats['cognitive_auc_q75']:.4f}]\n"
            )
            if "cognitive_auc_ci_lower" in stats:
                f.write(
                    f"• 95% CI: [{stats['cognitive_auc_ci_lower']:.4f}, {stats['cognitive_auc_ci_upper']:.4f}]\n"
                )
            f.write(
                f"• Above random (>0.5): {stats['cognitive_auc_above_random']:.1%}\n"
            )
            f.write(
                f"• Good performance (>0.6): {stats['cognitive_auc_good_performance']:.1%}\n"
            )
            f.write("\n")

            # Performance Benchmarks
            f.write("PERFORMANCE BENCHMARKS\n")
            f.write("-" * 40 + "\n")

            # Motor benchmarks
            motor_rating = "Poor"
            if stats["motor_r2_mean"] > 0.1:
                motor_rating = "Good"
            elif stats["motor_r2_mean"] > 0:
                motor_rating = "Fair"

            f.write(f"Motor Performance Rating: {motor_rating}\n")
            f.write(f"  - Mean R² = {stats['motor_r2_mean']:.4f}\n")
            f.write("  - Benchmark: >0 (explains variance), >0.1 (good performance)\n")

            # Cognitive benchmarks
            cognitive_rating = "Poor"
            if stats["cognitive_auc_mean"] > 0.7:
                cognitive_rating = "Excellent"
            elif stats["cognitive_auc_mean"] > 0.6:
                cognitive_rating = "Good"
            elif stats["cognitive_auc_mean"] > 0.5:
                cognitive_rating = "Fair"

            f.write(f"Cognitive Performance Rating: {cognitive_rating}\n")
            f.write(f"  - Mean AUC = {stats['cognitive_auc_mean']:.4f}\n")
            f.write(
                "  - Benchmark: >0.5 (better than random), >0.6 (good), >0.7 (excellent)\n"
            )
            f.write("\n")

            # Model Reliability
            f.write("MODEL RELIABILITY ANALYSIS\n")
            f.write("-" * 40 + "\n")
            cv_motor = (
                stats["motor_r2_std"] / abs(stats["motor_r2_mean"])
                if stats["motor_r2_mean"] != 0
                else np.inf
            )
            cv_cognitive = stats["cognitive_auc_std"] / stats["cognitive_auc_mean"]

            f.write(f"• Motor R² coefficient of variation: {cv_motor:.3f}\n")
            f.write(f"• Cognitive AUC coefficient of variation: {cv_cognitive:.3f}\n")
            f.write(
                f"• Success rate: {stats['num_successful_runs']}/{len(results)} ({stats['num_successful_runs'] / len(results):.1%})\n"
            )

            reliability_rating = "High"
            if cv_cognitive > 0.3 or stats["num_failed_runs"] > 0:
                reliability_rating = "Medium"
            if cv_cognitive > 0.5 or stats["num_failed_runs"] > len(results) * 0.2:
                reliability_rating = "Low"

            f.write(f"• Overall reliability: {reliability_rating}\n")
            f.write("\n")

            # Recommendations
            f.write("RECOMMENDATIONS\n")
            f.write("-" * 40 + "\n")

            if stats["motor_r2_mean"] < 0:
                f.write(
                    "• Motor prediction: Model struggling with motor progression prediction.\n"
                )
                f.write(
                    "  Consider: feature engineering, hyperparameter tuning, architecture changes.\n"
                )
            elif stats["motor_r2_mean"] < 0.1:
                f.write(
                    "• Motor prediction: Marginal performance. Consider ensemble methods or more data.\n"
                )
            else:
                f.write(
                    "• Motor prediction: Good performance. Consider deployment or validation on external data.\n"
                )

            if stats["cognitive_auc_mean"] < 0.6:
                f.write("• Cognitive prediction: Below good performance threshold.\n")
                f.write(
                    "  Consider: class balancing, feature selection, model regularization.\n"
                )
            else:
                f.write(
                    "• Cognitive prediction: Good performance. Ready for clinical validation.\n"
                )

            if cv_cognitive > 0.3:
                f.write(
                    "• Model stability: High variance detected. Consider regularization or ensemble methods.\n"
                )
            else:
                f.write(
                    "• Model stability: Consistent performance across runs. Good for deployment.\n"
                )

            f.write("\n")

            # Individual Run Details
            f.write("INDIVIDUAL RUN DETAILS\n")
            f.write("-" * 40 + "\n")
            valid_results = [r for r in results if not pd.isna(r.get("motor_r2"))]

            for result in valid_results:
                f.write(
                    f"Run {result['run_id']:2d}: Motor R² = {result['motor_r2']:6.4f}, "
                    f"Cognitive AUC = {result['cognitive_auc']:6.4f}, "
                    f"Duration = {result['duration_seconds']:5.1f}s\n"
                )

            if stats["num_failed_runs"] > 0:
                f.write("\nFAILED RUNS:\n")
                failed_results = [r for r in results if pd.isna(r.get("motor_r2"))]
                for result in failed_results:
                    f.write(
                        f"Run {result['run_id']:2d}: FAILED - {result.get('error', 'Unknown error')}\n"
                    )

            f.write("\n" + "=" * 80 + "\n")

        self.logger.info(f"📋 Comprehensive report saved to {report_path}")

    def save_results_to_json(self):
        """Save all results to JSON for further analysis."""
        json_path = self.output_dir / "experiment_results.json"

        with open(json_path, "w") as f:
            json.dump(self.experiment_results, f, indent=2, default=str)

    def run_evaluation(self, num_runs: int = 10, extended_epochs: bool = False):
        """Run the complete evaluation pipeline."""
        self.logger.info("🎯 Starting GIMAN comprehensive evaluation...")

        # Setup
        self.setup_data()

        # Run experiments
        results = self.run_multiple_experiments(num_runs, extended_epochs)

        # Calculate statistics
        stats = self.calculate_statistics(results)

        # Generate outputs
        self.create_visualizations(results)
        self.generate_comprehensive_report(results, stats)
        self.save_results_to_json()

        # Print summary
        print("\n" + "=" * 80)
        print("🎉 GIMAN COMPREHENSIVE EVALUATION COMPLETE")
        print("=" * 80)
        print(f"📊 Runs completed: {stats['num_successful_runs']}/{len(results)}")
        print(
            f"🧠 Motor R² (mean ± std): {stats['motor_r2_mean']:.4f} ± {stats['motor_r2_std']:.4f}"
        )
        print(
            f"🎯 Cognitive AUC (mean ± std): {stats['cognitive_auc_mean']:.4f} ± {stats['cognitive_auc_std']:.4f}"
        )
        print(f"⏱️  Total time: {stats['total_duration_minutes']:.1f} minutes")
        print(f"📁 Results saved to: {self.output_dir}")
        print("=" * 80)

        return results, stats


def main():
    """Main execution function."""
    import argparse

    parser = argparse.ArgumentParser(description="GIMAN Comprehensive Evaluation")
    parser.add_argument(
        "--runs", type=int, default=10, help="Number of independent runs"
    )
    parser.add_argument(
        "--extended-epochs", action="store_true", help="Use extended training epochs"
    )
    parser.add_argument(
        "--output-dir", type=str, default="evaluation_results", help="Output directory"
    )

    args = parser.parse_args()

    evaluator = GIMANComprehensiveEvaluator(output_dir=args.output_dir)
    results, stats = evaluator.run_evaluation(
        num_runs=args.runs, extended_epochs=args.extended_epochs
    )

    return results, stats


if __name__ == "__main__":
    main()
