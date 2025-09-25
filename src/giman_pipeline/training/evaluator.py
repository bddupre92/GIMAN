"""GIMAN Evaluation Framework - Phase 2.

This module provides comprehensive evaluation capabilities for GIMAN models,
including cross-validation, statistical analysis, and visualization tools
for Parkinson's Disease classification performance assessment.

Features:
- Cross-validation (K-fold, stratified)
- Statistical significance testing
- Performance visualization
- Detailed classification reports
- ROC curves and confusion matrices
- Model comparison utilities
"""

import logging
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from sklearn.metrics import (
    accuracy_score,
    auc,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import KFold, StratifiedKFold
from torch_geometric.data import DataLoader

from .models import GIMANClassifier
from .trainer import GIMANTrainer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GIMANEvaluator:
    """Comprehensive evaluation framework for GIMAN models.

    This class provides extensive evaluation capabilities including:
    - Cross-validation with multiple strategies
    - Statistical analysis and significance testing
    - Performance visualization
    - Model comparison and ablation studies
    - Detailed reporting with clinical interpretation

    Args:
        model: Trained GIMAN model for evaluation
        device: Computation device ('cpu' or 'cuda')
        results_dir: Directory to save evaluation results
    """

    def __init__(
        self,
        model: GIMANClassifier,
        device: str = "cpu",
        results_dir: Path | None = None,
    ):
        """Initialize GIMAN evaluator."""
        self.model = model.to(device)
        self.device = device
        self.results_dir = (
            Path(results_dir) if results_dir else Path("evaluation_results")
        )
        self.results_dir.mkdir(exist_ok=True)

        # Set up plotting style
        plt.style.use("seaborn-v0_8")
        sns.set_palette("husl")

        logger.info("🧪 GIMAN Evaluator initialized")
        logger.info(f"   - Model: {type(model).__name__}")
        logger.info(f"   - Device: {device}")
        logger.info(f"   - Results dir: {self.results_dir}")

    def evaluate_single(
        self, test_loader: DataLoader, split_name: str = "test"
    ) -> dict[str, Any]:
        """Evaluate model on a single dataset split."""
        logger.info(f"📊 Evaluating on {split_name} set")

        self.model.eval()
        all_preds = []
        all_targets = []
        all_probs = []

        with torch.no_grad():
            for batch in test_loader:
                batch = batch.to(self.device)
                output = self.model(batch)
                logits = output["logits"]

                probs = torch.nn.functional.softmax(logits, dim=1)
                pred = logits.argmax(dim=1)

                all_preds.extend(pred.cpu().numpy())
                all_targets.extend(batch.y.cpu().numpy())
                all_probs.extend(probs[:, 1].cpu().numpy())

        # Calculate comprehensive metrics
        results = self._calculate_metrics(all_targets, all_preds, all_probs)
        results["split_name"] = split_name
        results["n_samples"] = len(all_targets)

        # Log summary
        logger.info(f"✅ {split_name.capitalize()} evaluation complete:")
        logger.info(f"   - Samples: {results['n_samples']}")
        logger.info(f"   - Accuracy: {results['accuracy']:.4f}")
        logger.info(f"   - F1 Score: {results['f1']:.4f}")
        logger.info(f"   - AUC-ROC: {results['auc_roc']:.4f}")

        return results

    def cross_validate(
        self,
        dataset: list,  # List of graph data objects
        n_splits: int = 5,
        stratified: bool = True,
        random_state: int = 42,
        **trainer_kwargs,
    ) -> dict[str, Any]:
        """Perform k-fold cross-validation."""
        logger.info(f"🔄 Starting {n_splits}-fold cross-validation")

        # Extract labels for stratification
        labels = [data.y.item() for data in dataset]

        # Setup cross-validation strategy
        if stratified:
            cv = StratifiedKFold(
                n_splits=n_splits, shuffle=True, random_state=random_state
            )
            split_generator = cv.split(dataset, labels)
        else:
            cv = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
            split_generator = cv.split(dataset)

        cv_results = {
            "fold_results": [],
            "metrics_summary": {},
            "training_histories": [],
        }

        for fold, (train_idx, val_idx) in enumerate(split_generator):
            logger.info(f"🎯 Training fold {fold + 1}/{n_splits}")

            # Create data loaders for this fold
            train_data = [dataset[i] for i in train_idx]
            val_data = [dataset[i] for i in val_idx]

            train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
            val_loader = DataLoader(val_data, batch_size=32, shuffle=False)

            # Create fresh model for this fold
            model_config = self.model.get_model_info()
            fold_model = GIMANClassifier(
                input_dim=model_config["input_dim"],
                hidden_dims=model_config["hidden_dims"],
                output_dim=model_config["output_dim"],
                dropout_rate=model_config["dropout_rate"],
                pooling_method=model_config["pooling_method"],
            )

            # Train model for this fold
            trainer = GIMANTrainer(
                model=fold_model, device=self.device, **trainer_kwargs
            )

            training_history = trainer.train(
                train_loader=train_loader, val_loader=val_loader, verbose=False
            )

            # Evaluate this fold
            fold_results = self.evaluate_single(val_loader, f"fold_{fold + 1}")
            fold_results["fold"] = fold + 1
            fold_results["train_size"] = len(train_data)
            fold_results["val_size"] = len(val_data)

            cv_results["fold_results"].append(fold_results)
            cv_results["training_histories"].append(training_history)

            logger.info(
                f"   ✅ Fold {fold + 1} complete - Val Acc: {fold_results['accuracy']:.4f}"
            )

        # Calculate cross-validation summary statistics
        cv_results["metrics_summary"] = self._summarize_cv_results(
            cv_results["fold_results"]
        )

        logger.info("🏆 Cross-validation complete:")
        for metric, stats in cv_results["metrics_summary"].items():
            if isinstance(stats, dict) and "mean" in stats:
                logger.info(f"   - {metric}: {stats['mean']:.4f} ± {stats['std']:.4f}")

        return cv_results

    def _calculate_metrics(
        self, targets: list[int], predictions: list[int], probabilities: list[float]
    ) -> dict[str, Any]:
        """Calculate comprehensive evaluation metrics."""
        return {
            "accuracy": accuracy_score(targets, predictions),
            "precision": precision_score(targets, predictions, average="binary"),
            "recall": recall_score(targets, predictions, average="binary"),
            "f1": f1_score(targets, predictions, average="binary"),
            "auc_roc": roc_auc_score(targets, probabilities),
            "confusion_matrix": confusion_matrix(targets, predictions),
            "classification_report": classification_report(
                targets,
                predictions,
                target_names=["Healthy Control", "Parkinson's Disease"],
                output_dict=True,
            ),
            "targets": targets,
            "predictions": predictions,
            "probabilities": probabilities,
        }

    def _summarize_cv_results(self, fold_results: list[dict]) -> dict[str, dict]:
        """Summarize cross-validation results with statistics."""
        metrics = ["accuracy", "precision", "recall", "f1", "auc_roc"]
        summary = {}

        for metric in metrics:
            values = [result[metric] for result in fold_results]
            summary[metric] = {
                "mean": np.mean(values),
                "std": np.std(values),
                "min": np.min(values),
                "max": np.max(values),
                "values": values,
            }

        return summary

    def plot_roc_curve(
        self,
        results: dict[str, Any],
        title: str = "ROC Curve",
        save_path: Path | None = None,
    ):
        """Plot ROC curve for evaluation results."""
        targets = results["targets"]
        probabilities = results["probabilities"]

        fpr, tpr, thresholds = roc_curve(targets, probabilities)
        roc_auc = auc(fpr, tpr)

        plt.figure(figsize=(8, 6))
        plt.plot(
            fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (AUC = {roc_auc:.3f})"
        )
        plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(title)
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.show()

    def plot_precision_recall_curve(
        self,
        results: dict[str, Any],
        title: str = "Precision-Recall Curve",
        save_path: Path | None = None,
    ):
        """Plot Precision-Recall curve."""
        targets = results["targets"]
        probabilities = results["probabilities"]

        precision, recall, thresholds = precision_recall_curve(targets, probabilities)
        pr_auc = auc(recall, precision)

        plt.figure(figsize=(8, 6))
        plt.plot(
            recall,
            precision,
            color="blue",
            lw=2,
            label=f"PR curve (AUC = {pr_auc:.3f})",
        )
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title(title)
        plt.legend(loc="lower left")
        plt.grid(True, alpha=0.3)

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.show()

    def plot_confusion_matrix(
        self,
        results: dict[str, Any],
        title: str = "Confusion Matrix",
        save_path: Path | None = None,
    ):
        """Plot confusion matrix with clinical interpretation."""
        cm = results["confusion_matrix"]
        class_names = ["Healthy Control", "Parkinson's Disease"]

        plt.figure(figsize=(8, 6))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=class_names,
            yticklabels=class_names,
        )
        plt.title(title)
        plt.xlabel("Predicted")
        plt.ylabel("Actual")

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.show()

    def plot_cv_metrics(
        self,
        cv_results: dict[str, Any],
        title: str = "Cross-Validation Results",
        save_path: Path | None = None,
    ):
        """Plot cross-validation metrics distribution."""
        metrics_summary = cv_results["metrics_summary"]
        metrics = ["accuracy", "precision", "recall", "f1", "auc_roc"]

        fig, axes = plt.subplots(1, len(metrics), figsize=(20, 4))

        for i, metric in enumerate(metrics):
            values = metrics_summary[metric]["values"]
            mean_val = metrics_summary[metric]["mean"]
            std_val = metrics_summary[metric]["std"]

            axes[i].boxplot(values)
            axes[i].axhline(y=mean_val, color="red", linestyle="--", alpha=0.7)
            axes[i].set_title(f"{metric.upper()}\n{mean_val:.3f} ± {std_val:.3f}")
            axes[i].set_ylabel("Score")
            axes[i].grid(True, alpha=0.3)

        plt.suptitle(title)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.show()

    def generate_report(
        self, results: dict[str, Any], save_path: Path | None = None
    ) -> str:
        """Generate comprehensive evaluation report."""
        report_lines = [
            "=" * 80,
            "GIMAN MODEL EVALUATION REPORT",
            "=" * 80,
            "",
            f"Dataset: {results.get('split_name', 'Unknown')}",
            f"Samples: {results.get('n_samples', 'Unknown')}",
            "",
            "CLASSIFICATION METRICS:",
            "-" * 40,
            f"Accuracy:  {results['accuracy']:.4f}",
            f"Precision: {results['precision']:.4f}",
            f"Recall:    {results['recall']:.4f}",
            f"F1-Score:  {results['f1']:.4f}",
            f"AUC-ROC:   {results['auc_roc']:.4f}",
            "",
            "CONFUSION MATRIX:",
            "-" * 40,
        ]

        # Add confusion matrix
        cm = results["confusion_matrix"]
        report_lines.extend(
            [
                "                 Predicted",
                "              HC    PD",
                f"Actual HC   {cm[0, 0]:4d}  {cm[0, 1]:4d}",
                f"       PD   {cm[1, 0]:4d}  {cm[1, 1]:4d}",
                "",
            ]
        )

        # Add clinical interpretation
        tn, fp, fn, tp = cm.ravel()
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0

        report_lines.extend(
            [
                "CLINICAL INTERPRETATION:",
                "-" * 40,
                f"Sensitivity (True Positive Rate): {sensitivity:.4f}",
                f"Specificity (True Negative Rate): {specificity:.4f}",
                f"False Positive Rate: {fp / (fp + tn) if (fp + tn) > 0 else 0:.4f}",
                f"False Negative Rate: {fn / (fn + tp) if (fn + tp) > 0 else 0:.4f}",
                "",
                "DETAILED CLASSIFICATION REPORT:",
                "-" * 40,
            ]
        )

        # Add detailed classification report
        clf_report = results["classification_report"]
        for class_name, metrics in clf_report.items():
            if isinstance(metrics, dict) and "precision" in metrics:
                report_lines.append(
                    f"{class_name:20s}: "
                    f"Precision={metrics['precision']:.3f}, "
                    f"Recall={metrics['recall']:.3f}, "
                    f"F1={metrics['f1-score']:.3f}, "
                    f"Support={metrics['support']}"
                )

        report_lines.extend(["", "=" * 80, ""])

        report_text = "\n".join(report_lines)

        if save_path:
            with open(save_path, "w") as f:
                f.write(report_text)
            logger.info(f"📄 Report saved to {save_path}")

        return report_text

    def save_results(self, results: dict[str, Any], prefix: str = "evaluation"):
        """Save evaluation results with visualizations."""
        timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
        base_path = self.results_dir / f"{prefix}_{timestamp}"

        # Save plots
        self.plot_roc_curve(results, save_path=base_path.with_suffix("_roc.png"))
        self.plot_precision_recall_curve(
            results, save_path=base_path.with_suffix("_pr.png")
        )
        self.plot_confusion_matrix(results, save_path=base_path.with_suffix("_cm.png"))

        # Save report
        report_text = self.generate_report(results)
        with open(base_path.with_suffix("_report.txt"), "w") as f:
            f.write(report_text)

        # Save raw results
        results_clean = {
            k: v
            for k, v in results.items()
            if k not in ["targets", "predictions", "probabilities"]
        }

        import json

        with open(base_path.with_suffix("_results.json"), "w") as f:
            json.dump(results_clean, f, indent=2, default=str)

        logger.info(f"💾 Results saved with prefix: {base_path.name}")

        return base_path
