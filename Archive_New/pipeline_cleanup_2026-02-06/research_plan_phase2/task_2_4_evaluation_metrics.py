#!/usr/bin/env python3
"""
Research Plan Phase 2, Task 2.4: Comprehensive Evaluation Metrics

Implements detailed evaluation metrics for dual-task prognostic GIMAN:
- Motor Task: MAE, RMSE, R², Pearson/Spearman correlation
- Cognitive Task: AUC-ROC, F1, Precision, Recall, Specificity
- Per-cohort analysis (PD, HC, Prodromal)
- Visualization and error analysis

Author: GIMAN Development Team
Date: October 3, 2025
Research Plan Phase: 2 (Prognostic Model Architecture)
Task: 2.4 - Comprehensive evaluation metrics
"""

import sys
from pathlib import Path
from typing import Dict, Tuple, Optional
import logging

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
    classification_report
)

import torch

# Import our previous implementations
from task_2_1_giman_prognostic_model import GIMANPrognostic

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MotorTaskEvaluator:
    """
    Comprehensive evaluation for motor progression regression task.

    Metrics:
    - Mean Absolute Error (MAE)
    - Root Mean Squared Error (RMSE)
    - R² Score (coefficient of determination)
    - Pearson correlation
    - Spearman correlation
    """

    def __init__(self):
        self.predictions = None
        self.targets = None
        self.metrics = {}

    def evaluate(
        self,
        predictions: np.ndarray,
        targets: np.ndarray,
        cohort_labels: Optional[np.ndarray] = None
    ) -> Dict[str, float]:
        """
        Compute comprehensive motor task metrics.

        Args:
            predictions: Predicted motor slopes [n_samples]
            targets: True motor slopes [n_samples]
            cohort_labels: Cohort assignments (PD, HC, etc.) [n_samples]

        Returns:
            Dictionary of metrics
        """
        self.predictions = predictions
        self.targets = targets

        # Overall metrics
        mae = mean_absolute_error(targets, predictions)
        rmse = np.sqrt(mean_squared_error(targets, predictions))
        r2 = r2_score(targets, predictions)

        # Correlation metrics
        pearson_r, pearson_p = stats.pearsonr(predictions, targets)
        spearman_r, spearman_p = stats.spearmanr(predictions, targets)

        # Store metrics
        self.metrics = {
            'mae': mae,
            'rmse': rmse,
            'r2': r2,
            'pearson_r': pearson_r,
            'pearson_p': pearson_p,
            'spearman_r': spearman_r,
            'spearman_p': spearman_p,
            'mean_prediction': np.mean(predictions),
            'std_prediction': np.std(predictions),
            'mean_target': np.mean(targets),
            'std_target': np.std(targets)
        }

        # Per-cohort analysis if labels provided
        if cohort_labels is not None:
            self.metrics['per_cohort'] = self._compute_per_cohort_metrics(
                predictions, targets, cohort_labels
            )

        return self.metrics

    def _compute_per_cohort_metrics(
        self,
        predictions: np.ndarray,
        targets: np.ndarray,
        cohort_labels: np.ndarray
    ) -> Dict[str, Dict[str, float]]:
        """Compute metrics separately for each cohort."""
        cohort_metrics = {}

        for cohort in np.unique(cohort_labels):
            mask = cohort_labels == cohort
            if mask.sum() < 2:  # Need at least 2 samples
                continue

            cohort_pred = predictions[mask]
            cohort_target = targets[mask]

            cohort_metrics[str(cohort)] = {
                'n_samples': mask.sum(),
                'mae': mean_absolute_error(cohort_target, cohort_pred),
                'rmse': np.sqrt(mean_squared_error(cohort_target, cohort_pred)),
                'r2': r2_score(cohort_target, cohort_pred),
                'mean_prediction': np.mean(cohort_pred),
                'mean_target': np.mean(cohort_target)
            }

        return cohort_metrics

    def print_summary(self):
        """Print formatted metric summary."""
        print("\n" + "="*80)
        print("MOTOR TASK EVALUATION SUMMARY")
        print("="*80)

        print(f"\nOverall Performance:")
        print(f"  MAE:  {self.metrics['mae']:.4f} pts/year")
        print(f"  RMSE: {self.metrics['rmse']:.4f} pts/year")
        print(f"  R²:   {self.metrics['r2']:.4f}")

        print(f"\nCorrelation Analysis:")
        print(f"  Pearson r:  {self.metrics['pearson_r']:.4f} (p={self.metrics['pearson_p']:.4e})")
        print(f"  Spearman r: {self.metrics['spearman_r']:.4f} (p={self.metrics['spearman_p']:.4e})")

        print(f"\nDistribution Statistics:")
        print(f"  Predictions: {self.metrics['mean_prediction']:.3f} ± {self.metrics['std_prediction']:.3f}")
        print(f"  Targets:     {self.metrics['mean_target']:.3f} ± {self.metrics['std_target']:.3f}")

        if 'per_cohort' in self.metrics:
            print(f"\nPer-Cohort Performance:")
            for cohort, metrics in self.metrics['per_cohort'].items():
                print(f"  {cohort}:")
                print(f"    n={metrics['n_samples']}, R²={metrics['r2']:.4f}, MAE={metrics['mae']:.4f}")

        print("="*80)


class CognitiveTaskEvaluator:
    """
    Comprehensive evaluation for cognitive decline classification task.

    Metrics:
    - AUC-ROC (Area Under ROC Curve)
    - F1 Score
    - Precision
    - Recall (Sensitivity)
    - Specificity
    - Confusion Matrix
    """

    def __init__(self):
        self.predictions = None
        self.probabilities = None
        self.targets = None
        self.metrics = {}

    def evaluate(
        self,
        probabilities: np.ndarray,
        targets: np.ndarray,
        threshold: float = 0.5,
        cohort_labels: Optional[np.ndarray] = None
    ) -> Dict[str, float]:
        """
        Compute comprehensive cognitive task metrics.

        Args:
            probabilities: Predicted probabilities [n_samples] (for positive class)
            targets: True binary labels [n_samples] (0 or 1)
            threshold: Classification threshold
            cohort_labels: Cohort assignments

        Returns:
            Dictionary of metrics
        """
        self.probabilities = probabilities
        self.targets = targets
        self.predictions = (probabilities >= threshold).astype(int)

        # Check if we have both classes
        if len(np.unique(targets)) < 2:
            logger.warning("Only one class present in targets - some metrics will be invalid")

        # AUC-ROC
        try:
            auc_roc = roc_auc_score(targets, probabilities)
        except ValueError:
            auc_roc = 0.5  # Default if only one class

        # Classification metrics
        f1 = f1_score(targets, self.predictions, zero_division=0)
        precision = precision_score(targets, self.predictions, zero_division=0)
        recall = recall_score(targets, self.predictions, zero_division=0)

        # Confusion matrix
        tn, fp, fn, tp = confusion_matrix(targets, self.predictions).ravel()
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

        # Store metrics
        self.metrics = {
            'auc_roc': auc_roc,
            'f1': f1,
            'precision': precision,
            'recall': recall,
            'specificity': specificity,
            'true_positives': int(tp),
            'true_negatives': int(tn),
            'false_positives': int(fp),
            'false_negatives': int(fn),
            'positive_rate': np.mean(targets),
            'predicted_positive_rate': np.mean(self.predictions)
        }

        # Per-cohort analysis
        if cohort_labels is not None:
            self.metrics['per_cohort'] = self._compute_per_cohort_metrics(
                probabilities, targets, cohort_labels
            )

        return self.metrics

    def _compute_per_cohort_metrics(
        self,
        probabilities: np.ndarray,
        targets: np.ndarray,
        cohort_labels: np.ndarray
    ) -> Dict[str, Dict[str, float]]:
        """Compute metrics separately for each cohort."""
        cohort_metrics = {}

        for cohort in np.unique(cohort_labels):
            mask = cohort_labels == cohort
            if mask.sum() < 2:
                continue

            cohort_prob = probabilities[mask]
            cohort_target = targets[mask]

            # Only compute if both classes present
            if len(np.unique(cohort_target)) < 2:
                continue

            cohort_metrics[str(cohort)] = {
                'n_samples': mask.sum(),
                'auc_roc': roc_auc_score(cohort_target, cohort_prob),
                'positive_rate': np.mean(cohort_target),
                'mean_probability': np.mean(cohort_prob)
            }

        return cohort_metrics

    def print_summary(self):
        """Print formatted metric summary."""
        print("\n" + "="*80)
        print("COGNITIVE TASK EVALUATION SUMMARY")
        print("="*80)

        print(f"\nDiscriminative Performance:")
        print(f"  AUC-ROC: {self.metrics['auc_roc']:.4f}")

        print(f"\nClassification Metrics:")
        print(f"  F1 Score:    {self.metrics['f1']:.4f}")
        print(f"  Precision:   {self.metrics['precision']:.4f}")
        print(f"  Recall:      {self.metrics['recall']:.4f}")
        print(f"  Specificity: {self.metrics['specificity']:.4f}")

        print(f"\nConfusion Matrix:")
        print(f"  True Positives:  {self.metrics['true_positives']}")
        print(f"  True Negatives:  {self.metrics['true_negatives']}")
        print(f"  False Positives: {self.metrics['false_positives']}")
        print(f"  False Negatives: {self.metrics['false_negatives']}")

        print(f"\nClass Distribution:")
        print(f"  Actual positive rate:    {self.metrics['positive_rate']:.1%}")
        print(f"  Predicted positive rate: {self.metrics['predicted_positive_rate']:.1%}")

        if 'per_cohort' in self.metrics:
            print(f"\nPer-Cohort Performance:")
            for cohort, metrics in self.metrics['per_cohort'].items():
                print(f"  {cohort}:")
                print(f"    n={metrics['n_samples']}, AUC={metrics['auc_roc']:.4f}, Positive rate={metrics['positive_rate']:.1%}")

        print("="*80)


class ModelEvaluator:
    """
    Complete evaluation for dual-task GIMAN model.

    Combines motor and cognitive task evaluation with visualization.
    """

    def __init__(self, output_dir: Optional[Path] = None):
        """
        Initialize evaluator.

        Args:
            output_dir: Directory to save plots and reports
        """
        self.motor_evaluator = MotorTaskEvaluator()
        self.cognitive_evaluator = CognitiveTaskEvaluator()
        self.output_dir = output_dir

        if output_dir:
            output_dir.mkdir(parents=True, exist_ok=True)

    def evaluate_model(
        self,
        model: GIMANPrognostic,
        features: torch.Tensor,
        motor_targets: torch.Tensor,
        cognitive_targets: torch.Tensor,
        edge_index: torch.Tensor,
        cohort_labels: Optional[np.ndarray] = None,
        device: str = 'cpu'
    ) -> Dict:
        """
        Evaluate model on dataset.

        Args:
            model: Trained GIMANPrognostic model
            features: Node features
            motor_targets: Motor targets
            cognitive_targets: Cognitive targets
            edge_index: Graph edges
            cohort_labels: Optional cohort assignments
            device: Device to run on

        Returns:
            Dictionary with both task metrics
        """
        model.eval()
        model = model.to(device)
        features = features.to(device)
        edge_index = edge_index.to(device)

        # Get predictions
        with torch.no_grad():
            motor_pred, cognitive_pred = model(features, edge_index)

        # Convert to numpy
        motor_pred_np = motor_pred.squeeze().cpu().numpy()
        motor_targets_np = motor_targets.cpu().numpy()

        cognitive_probs = torch.softmax(cognitive_pred, dim=1)[:, 1].cpu().numpy()
        cognitive_targets_np = cognitive_targets.cpu().numpy()

        # Evaluate both tasks
        motor_metrics = self.motor_evaluator.evaluate(
            motor_pred_np,
            motor_targets_np,
            cohort_labels
        )

        cognitive_metrics = self.cognitive_evaluator.evaluate(
            cognitive_probs,
            cognitive_targets_np,
            cohort_labels=cohort_labels
        )

        return {
            'motor': motor_metrics,
            'cognitive': cognitive_metrics
        }

    def create_visualizations(self):
        """Create comprehensive visualization plots."""
        if not self.output_dir:
            logger.warning("No output directory specified - skipping visualizations")
            return

        fig = plt.figure(figsize=(16, 12))

        # 1. Motor: Prediction vs Actual scatter plot
        ax1 = plt.subplot(3, 3, 1)
        self._plot_motor_scatter(ax1)

        # 2. Motor: Residual plot
        ax2 = plt.subplot(3, 3, 2)
        self._plot_motor_residuals(ax2)

        # 3. Motor: Error distribution
        ax3 = plt.subplot(3, 3, 3)
        self._plot_motor_error_distribution(ax3)

        # 4. Cognitive: ROC curve
        ax4 = plt.subplot(3, 3, 4)
        self._plot_cognitive_roc(ax4)

        # 5. Cognitive: Precision-Recall curve
        ax5 = plt.subplot(3, 3, 5)
        self._plot_cognitive_pr_curve(ax5)

        # 6. Cognitive: Confusion matrix
        ax6 = plt.subplot(3, 3, 6)
        self._plot_cognitive_confusion_matrix(ax6)

        # 7. Cognitive: Probability distribution
        ax7 = plt.subplot(3, 3, 7)
        self._plot_cognitive_probability_distribution(ax7)

        # 8. Combined: Metrics summary
        ax8 = plt.subplot(3, 3, 8)
        self._plot_metrics_summary(ax8)

        plt.tight_layout()

        # Save
        output_file = self.output_dir / 'comprehensive_evaluation.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        logger.info(f"Visualization saved to: {output_file}")
        plt.close()

    def _plot_motor_scatter(self, ax):
        """Plot predicted vs actual motor slopes."""
        pred = self.motor_evaluator.predictions
        target = self.motor_evaluator.targets

        ax.scatter(target, pred, alpha=0.5, s=20)

        # Perfect prediction line
        min_val = min(target.min(), pred.min())
        max_val = max(target.max(), pred.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', label='Perfect prediction')

        # Best fit line
        z = np.polyfit(target, pred, 1)
        p = np.poly1d(z)
        ax.plot(target, p(target), 'b-', alpha=0.5, label='Best fit')

        ax.set_xlabel('Actual Motor Slope (pts/year)')
        ax.set_ylabel('Predicted Motor Slope (pts/year)')
        ax.set_title(f'Motor Prediction (R²={self.motor_evaluator.metrics["r2"]:.4f})')
        ax.legend()
        ax.grid(True, alpha=0.3)

    def _plot_motor_residuals(self, ax):
        """Plot residuals vs predictions."""
        pred = self.motor_evaluator.predictions
        target = self.motor_evaluator.targets
        residuals = target - pred

        ax.scatter(pred, residuals, alpha=0.5, s=20)
        ax.axhline(y=0, color='r', linestyle='--')
        ax.set_xlabel('Predicted Motor Slope')
        ax.set_ylabel('Residuals')
        ax.set_title('Motor Residual Plot')
        ax.grid(True, alpha=0.3)

    def _plot_motor_error_distribution(self, ax):
        """Plot distribution of prediction errors."""
        pred = self.motor_evaluator.predictions
        target = self.motor_evaluator.targets
        errors = target - pred

        ax.hist(errors, bins=50, edgecolor='black', alpha=0.7)
        ax.axvline(x=0, color='r', linestyle='--', label='Zero error')
        ax.axvline(x=np.mean(errors), color='g', linestyle='--', label=f'Mean: {np.mean(errors):.3f}')
        ax.set_xlabel('Prediction Error (pts/year)')
        ax.set_ylabel('Frequency')
        ax.set_title(f'Motor Error Distribution (MAE={self.motor_evaluator.metrics["mae"]:.4f})')
        ax.legend()
        ax.grid(True, alpha=0.3)

    def _plot_cognitive_roc(self, ax):
        """Plot ROC curve."""
        prob = self.cognitive_evaluator.probabilities
        target = self.cognitive_evaluator.targets

        fpr, tpr, _ = roc_curve(target, prob)
        auc = self.cognitive_evaluator.metrics['auc_roc']

        ax.plot(fpr, tpr, label=f'ROC (AUC={auc:.4f})')
        ax.plot([0, 1], [0, 1], 'r--', label='Random')
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('Cognitive ROC Curve')
        ax.legend()
        ax.grid(True, alpha=0.3)

    def _plot_cognitive_pr_curve(self, ax):
        """Plot Precision-Recall curve."""
        prob = self.cognitive_evaluator.probabilities
        target = self.cognitive_evaluator.targets

        precision, recall, _ = precision_recall_curve(target, prob)

        ax.plot(recall, precision)
        ax.set_xlabel('Recall')
        ax.set_ylabel('Precision')
        ax.set_title(f'Precision-Recall Curve (F1={self.cognitive_evaluator.metrics["f1"]:.4f})')
        ax.grid(True, alpha=0.3)

    def _plot_cognitive_confusion_matrix(self, ax):
        """Plot confusion matrix."""
        cm = confusion_matrix(
            self.cognitive_evaluator.targets,
            self.cognitive_evaluator.predictions
        )

        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
        ax.set_title('Confusion Matrix')

    def _plot_cognitive_probability_distribution(self, ax):
        """Plot probability distributions by class."""
        prob = self.cognitive_evaluator.probabilities
        target = self.cognitive_evaluator.targets

        ax.hist(prob[target == 0], bins=30, alpha=0.5, label='Stable', edgecolor='black')
        ax.hist(prob[target == 1], bins=30, alpha=0.5, label='Decline', edgecolor='black')
        ax.set_xlabel('Predicted Probability (Decline)')
        ax.set_ylabel('Frequency')
        ax.set_title('Probability Distribution by Class')
        ax.legend()
        ax.grid(True, alpha=0.3)

    def _plot_metrics_summary(self, ax):
        """Plot text summary of key metrics."""
        ax.axis('off')

        motor_m = self.motor_evaluator.metrics
        cognitive_m = self.cognitive_evaluator.metrics

        summary_text = f"""
EVALUATION SUMMARY

Motor Task (Regression):
  R² Score:    {motor_m['r2']:.4f}
  MAE:         {motor_m['mae']:.4f} pts/year
  RMSE:        {motor_m['rmse']:.4f} pts/year
  Pearson r:   {motor_m['pearson_r']:.4f}

Cognitive Task (Classification):
  AUC-ROC:     {cognitive_m['auc_roc']:.4f}
  F1 Score:    {cognitive_m['f1']:.4f}
  Precision:   {cognitive_m['precision']:.4f}
  Recall:      {cognitive_m['recall']:.4f}
  Specificity: {cognitive_m['specificity']:.4f}
        """

        ax.text(0.1, 0.5, summary_text, fontsize=10, family='monospace',
                verticalalignment='center')

    def print_full_report(self):
        """Print complete evaluation report."""
        self.motor_evaluator.print_summary()
        self.cognitive_evaluator.print_summary()

    def save_report(self, filename: str = 'evaluation_report.txt'):
        """Save evaluation report to file."""
        if not self.output_dir:
            return

        import sys
        from io import StringIO

        # Capture print output
        old_stdout = sys.stdout
        sys.stdout = buffer = StringIO()

        self.print_full_report()

        sys.stdout = old_stdout
        report_text = buffer.getvalue()

        # Save to file
        report_file = self.output_dir / filename
        with open(report_file, 'w') as f:
            f.write(report_text)

        logger.info(f"Report saved to: {report_file}")


def demo_evaluation():
    """Demonstration of evaluation metrics."""

    print("\n" + "="*80)
    print("RESEARCH PLAN PHASE 2, TASK 2.4: EVALUATION METRICS DEMO")
    print("="*80 + "\n")

    # Create synthetic data for demonstration
    np.random.seed(42)
    n_samples = 500

    # Motor task (regression)
    true_motor = np.random.randn(n_samples) * 2.8 + 0.97  # Match Phase 1 distribution
    pred_motor = true_motor + np.random.randn(n_samples) * 2.0  # Add noise

    # Cognitive task (classification)
    true_cognitive = np.random.binomial(1, 0.123, n_samples)  # 12.3% decline rate
    prob_cognitive = np.clip(true_cognitive + np.random.randn(n_samples) * 0.3, 0, 1)

    # Cohort labels
    cohorts = np.random.choice(['PD', 'HC', 'Prodromal'], n_samples)

    # Create evaluator
    output_dir = Path(__file__).parent / 'evaluation_output'
    evaluator = ModelEvaluator(output_dir=output_dir)

    # Evaluate motor task
    print("Evaluating Motor Task...")
    evaluator.motor_evaluator.evaluate(pred_motor, true_motor, cohorts)
    evaluator.motor_evaluator.print_summary()

    # Evaluate cognitive task
    print("\nEvaluating Cognitive Task...")
    evaluator.cognitive_evaluator.evaluate(prob_cognitive, true_cognitive, cohort_labels=cohorts)
    evaluator.cognitive_evaluator.print_summary()

    # Create visualizations
    print("\nCreating visualizations...")
    evaluator.create_visualizations()

    # Save report
    evaluator.save_report()

    print("\n" + "="*80)
    print("[OK] TASK 2.4 COMPLETE: Evaluation metrics successfully implemented")
    print("="*80)
    print("\nKey features:")
    print("  - Motor metrics: MAE, RMSE, R2, Pearson/Spearman correlation")
    print("  - Cognitive metrics: AUC, F1, Precision, Recall, Specificity")
    print("  - Per-cohort analysis (PD, HC, Prodromal)")
    print("  - Comprehensive 8-panel visualization")
    print("  - Detailed text report generation")
    print("\nNext steps:")
    print("  - Use these metrics to evaluate trained models from Task 2.3")
    print("  - Task 2.5: Hyperparameter tuning using these evaluation metrics")
    print("="*80 + "\n")


if __name__ == '__main__':
    demo_evaluation()
