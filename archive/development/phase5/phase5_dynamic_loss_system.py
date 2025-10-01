#!/usr/bin/env python3
"""GIMAN Phase 5: Dynamic Loss Weighting System.

This system implements adaptive loss weighting to address task competition between
motor regression and cognitive classification. Building on the task-specific architecture,
it provides multiple loss weighting strategies to optimize dual-task performance.

Key Features:
- Adaptive loss weighting during training
- Multiple weighting strategies (fixed, adaptive, curriculum)
- Performance monitoring for both tasks
- Comparison framework with Phase 4 baseline

Weighting Strategies:
1. Fixed Weighting: Static 0.7 motor + 0.3 cognitive
2. Adaptive Weighting: Dynamic based on task performance
3. Curriculum Weighting: Progressive emphasis shift during training

Author: GIMAN Development Team
Date: September 2025
"""

import logging
import warnings

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import r2_score, roc_auc_score
from sklearn.model_selection import LeaveOneOut
from sklearn.preprocessing import StandardScaler

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# Import our dependencies
import os
import sys

# Add Phase 3 path
phase3_path = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "phase3"
)
sys.path.insert(0, phase3_path)
from phase3_1_real_data_integration import RealDataPhase3Integration

# Import from same directory
from phase5_task_specific_giman import TaskSpecificGIMANSystem

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


class DynamicLossWeighter:
    """Dynamic loss weighting strategies for dual-task learning."""

    def __init__(self, strategy: str = "fixed", initial_motor_weight: float = 0.7):
        """Initialize dynamic loss weighter.

        Args:
            strategy: Loss weighting strategy ("fixed", "adaptive", "curriculum")
            initial_motor_weight: Initial weight for motor task (cognitive = 1 - motor)
        """
        self.strategy = strategy
        self.motor_weight = initial_motor_weight
        self.cognitive_weight = 1.0 - initial_motor_weight

        # Adaptive weighting parameters
        self.motor_losses = []
        self.cognitive_losses = []
        self.adaptation_rate = 0.1

        # Curriculum parameters
        self.epoch_count = 0
        self.curriculum_epochs = 50

        logger.info(f"🎯 Dynamic Loss Weighter initialized with '{strategy}' strategy")
        logger.info(
            f"   Initial weights: Motor={self.motor_weight:.2f}, Cognitive={self.cognitive_weight:.2f}"
        )

    def get_weights(
        self, motor_loss: float, cognitive_loss: float, epoch: int = 0
    ) -> tuple[float, float]:
        """Get current loss weights based on strategy.

        Args:
            motor_loss: Current motor task loss
            cognitive_loss: Current cognitive task loss
            epoch: Current training epoch

        Returns:
            Tuple of (motor_weight, cognitive_weight)
        """
        if self.strategy == "fixed":
            return self._fixed_weighting()
        elif self.strategy == "adaptive":
            return self._adaptive_weighting(motor_loss, cognitive_loss)
        elif self.strategy == "curriculum":
            return self._curriculum_weighting(epoch)
        else:
            logger.warning(f"Unknown strategy '{self.strategy}', using fixed weighting")
            return self._fixed_weighting()

    def _fixed_weighting(self) -> tuple[float, float]:
        """Fixed weighting strategy."""
        return self.motor_weight, self.cognitive_weight

    def _adaptive_weighting(
        self, motor_loss: float, cognitive_loss: float
    ) -> tuple[float, float]:
        """Adaptive weighting based on relative task performance."""
        self.motor_losses.append(motor_loss)
        self.cognitive_losses.append(cognitive_loss)

        # Keep only recent losses for adaptation
        if len(self.motor_losses) > 10:
            self.motor_losses = self.motor_losses[-10:]
            self.cognitive_losses = self.cognitive_losses[-10:]

        if len(self.motor_losses) < 3:
            return self.motor_weight, self.cognitive_weight

        # Calculate relative performance (higher loss = worse performance = higher weight)
        avg_motor_loss = np.mean(self.motor_losses[-3:])
        avg_cognitive_loss = np.mean(self.cognitive_losses[-3:])

        # Normalize losses to [0, 1] range for comparison
        total_loss = avg_motor_loss + avg_cognitive_loss
        if total_loss > 0:
            motor_relative = avg_motor_loss / total_loss
            cognitive_relative = avg_cognitive_loss / total_loss

            # Adaptive update (focus more on worse-performing task)
            target_motor_weight = (
                0.5 + (motor_relative - 0.5) * 0.4
            )  # Range: [0.3, 0.7]
            target_cognitive_weight = 1.0 - target_motor_weight

            # Smooth adaptation
            self.motor_weight += self.adaptation_rate * (
                target_motor_weight - self.motor_weight
            )
            self.cognitive_weight = 1.0 - self.motor_weight

        return self.motor_weight, self.cognitive_weight

    def _curriculum_weighting(self, epoch: int) -> tuple[float, float]:
        """Curriculum weighting that shifts emphasis during training."""
        self.epoch_count = epoch

        if epoch < self.curriculum_epochs // 2:
            # Early training: Focus more on motor task (easier to optimize)
            progress = epoch / (self.curriculum_epochs // 2)
            motor_weight = 0.8 - 0.1 * progress  # 0.8 -> 0.7
            cognitive_weight = 1.0 - motor_weight
        else:
            # Later training: Gradually increase cognitive focus
            progress = (epoch - self.curriculum_epochs // 2) / (
                self.curriculum_epochs // 2
            )
            progress = min(progress, 1.0)
            motor_weight = 0.7 - 0.2 * progress  # 0.7 -> 0.5
            cognitive_weight = 1.0 - motor_weight

        return motor_weight, cognitive_weight

    def get_status(self) -> dict[str, float]:
        """Get current weighting status."""
        return {
            "strategy": self.strategy,
            "motor_weight": self.motor_weight,
            "cognitive_weight": self.cognitive_weight,
            "recent_motor_loss": self.motor_losses[-1] if self.motor_losses else 0.0,
            "recent_cognitive_loss": self.cognitive_losses[-1]
            if self.cognitive_losses
            else 0.0,
        }


class DynamicLossGIMANSystem(TaskSpecificGIMANSystem):
    """GIMAN system with dynamic loss weighting capabilities."""

    def __init__(self, loss_strategy: str = "fixed", **kwargs):
        """Initialize dynamic loss GIMAN system.

        Args:
            loss_strategy: Loss weighting strategy
            **kwargs: Arguments passed to TaskSpecificGIMANSystem
        """
        super().__init__(**kwargs)
        self.loss_weighter = DynamicLossWeighter(strategy=loss_strategy)
        logger.info(
            f"🔄 Dynamic Loss GIMAN initialized with '{loss_strategy}' weighting"
        )

    def compute_weighted_loss(
        self, motor_pred, motor_target, cognitive_pred, cognitive_target, epoch: int = 0
    ):
        """Compute dynamically weighted loss."""
        # Task-specific losses
        motor_criterion = nn.MSELoss()
        cognitive_criterion = nn.BCELoss()

        motor_loss = motor_criterion(motor_pred.squeeze(), motor_target)
        cognitive_loss = cognitive_criterion(cognitive_pred.squeeze(), cognitive_target)

        # Get dynamic weights
        motor_weight, cognitive_weight = self.loss_weighter.get_weights(
            motor_loss.item(), cognitive_loss.item(), epoch
        )

        # Weighted combination
        total_loss = motor_weight * motor_loss + cognitive_weight * cognitive_loss

        return total_loss, motor_loss, cognitive_loss, motor_weight, cognitive_weight


class DynamicLossLOOCVEvaluator:
    """LOOCV evaluator with dynamic loss weighting strategies."""

    def __init__(self, device="cpu"):
        """Initialize evaluator."""
        self.device = device
        self.scaler = StandardScaler()

    def evaluate_strategy(
        self,
        X_spatial,
        X_genomic,
        X_temporal,
        y_motor,
        y_cognitive,
        adj_matrix,
        loss_strategy: str = "fixed",
    ):
        """Evaluate a specific loss weighting strategy."""
        logger.info(f"🔄 Evaluating '{loss_strategy}' loss weighting strategy...")

        loo = LeaveOneOut()
        n_samples = X_spatial.shape[0]

        motor_predictions = []
        motor_actuals = []
        cognitive_predictions = []
        cognitive_actuals = []
        training_histories = []

        for fold, (train_idx, test_idx) in enumerate(loo.split(X_spatial)):
            logger.info(
                f"📊 Processing fold {fold + 1}/{n_samples} with {loss_strategy} weighting"
            )

            # Split data
            X_train_spatial, X_test_spatial = X_spatial[train_idx], X_spatial[test_idx]
            X_train_genomic, X_test_genomic = X_genomic[train_idx], X_genomic[test_idx]
            X_train_temporal, X_test_temporal = (
                X_temporal[train_idx],
                X_temporal[test_idx],
            )
            y_train_motor, y_test_motor = y_motor[train_idx], y_motor[test_idx]
            y_train_cognitive, y_test_cognitive = (
                y_cognitive[train_idx],
                y_cognitive[test_idx],
            )

            # Scale features
            scaler_spatial = StandardScaler()
            scaler_genomic = StandardScaler()
            scaler_temporal = StandardScaler()

            X_train_spatial = scaler_spatial.fit_transform(
                X_train_spatial.reshape(-1, X_train_spatial.shape[-1])
            ).reshape(X_train_spatial.shape)
            X_test_spatial = scaler_spatial.transform(
                X_test_spatial.reshape(-1, X_test_spatial.shape[-1])
            ).reshape(X_test_spatial.shape)

            X_train_genomic = scaler_genomic.fit_transform(X_train_genomic)
            X_test_genomic = scaler_genomic.transform(X_test_genomic)

            X_train_temporal = scaler_temporal.fit_transform(
                X_train_temporal.reshape(-1, X_train_temporal.shape[-1])
            ).reshape(X_train_temporal.shape)
            X_test_temporal = scaler_temporal.transform(
                X_test_temporal.reshape(-1, X_test_temporal.shape[-1])
            ).reshape(X_test_temporal.shape)

            # Train model with dynamic loss
            model = DynamicLossGIMANSystem(loss_strategy=loss_strategy).to(self.device)
            training_history = self._train_fold_dynamic(
                model,
                X_train_spatial,
                X_train_genomic,
                X_train_temporal,
                y_train_motor,
                y_train_cognitive,
                adj_matrix,
            )

            # Test model
            test_motor_pred, test_cognitive_pred = self._test_fold(
                model, X_test_spatial, X_test_genomic, X_test_temporal, adj_matrix
            )

            # Store results
            motor_predictions.append(test_motor_pred)
            motor_actuals.append(y_test_motor)
            cognitive_predictions.append(test_cognitive_pred)
            cognitive_actuals.append(y_test_cognitive)
            training_histories.append(training_history)

        # Calculate metrics
        motor_pred_array = np.array(motor_predictions).flatten()
        motor_actual_array = np.array(motor_actuals).flatten()
        cognitive_pred_array = np.array(cognitive_predictions).flatten()
        cognitive_actual_array = np.array(cognitive_actuals).flatten()

        # Motor regression metrics
        motor_r2 = r2_score(motor_actual_array, motor_pred_array)

        # Cognitive classification metrics
        try:
            cognitive_auc = roc_auc_score(cognitive_actual_array, cognitive_pred_array)
        except ValueError:
            cognitive_auc = 0.5  # Random baseline if all one class

        results = {
            "strategy": loss_strategy,
            "motor_r2": motor_r2,
            "cognitive_auc": cognitive_auc,
            "motor_predictions": motor_pred_array,
            "motor_actuals": motor_actual_array,
            "cognitive_predictions": cognitive_pred_array,
            "cognitive_actuals": cognitive_actual_array,
            "training_histories": training_histories,
            "n_samples": n_samples,
        }

        logger.info(f"✅ {loss_strategy} strategy completed!")
        logger.info(f"🎯 Motor R² = {motor_r2:.4f}")
        logger.info(f"🧠 Cognitive AUC = {cognitive_auc:.4f}")

        return results

    def _train_fold_dynamic(
        self, model, X_spatial, X_genomic, X_temporal, y_motor, y_cognitive, adj_matrix
    ):
        """Train a single fold with dynamic loss weighting."""
        # Convert to tensors
        X_spatial = torch.FloatTensor(X_spatial).to(self.device)
        X_genomic = torch.FloatTensor(X_genomic).to(self.device)
        X_temporal = torch.FloatTensor(X_temporal).to(self.device)
        y_motor = torch.FloatTensor(y_motor).to(self.device)
        y_cognitive = torch.FloatTensor(y_cognitive).to(self.device)
        adj_matrix = torch.FloatTensor(adj_matrix).to(self.device)

        # Optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-2)

        # Training loop with dynamic loss
        training_history = {
            "total_losses": [],
            "motor_losses": [],
            "cognitive_losses": [],
            "motor_weights": [],
            "cognitive_weights": [],
        }

        best_loss = float("inf")
        patience = 10
        patience_counter = 0

        model.train()
        for epoch in range(100):  # Max 100 epochs
            optimizer.zero_grad()

            # Forward pass
            motor_pred, cognitive_pred = model(
                X_spatial, X_genomic, X_temporal, adj_matrix
            )

            # Dynamic weighted loss
            total_loss, motor_loss, cognitive_loss, motor_weight, cognitive_weight = (
                model.compute_weighted_loss(
                    motor_pred, y_motor, cognitive_pred, y_cognitive, epoch
                )
            )

            # Backward pass
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            # Record training history
            training_history["total_losses"].append(total_loss.item())
            training_history["motor_losses"].append(motor_loss.item())
            training_history["cognitive_losses"].append(cognitive_loss.item())
            training_history["motor_weights"].append(motor_weight)
            training_history["cognitive_weights"].append(cognitive_weight)

            # Early stopping check
            if total_loss.item() < best_loss:
                best_loss = total_loss.item()
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    break

        return training_history

    def _test_fold(self, model, X_spatial, X_genomic, X_temporal, adj_matrix):
        """Test a single fold."""
        model.eval()

        with torch.no_grad():
            # Convert to tensors
            X_spatial = torch.FloatTensor(X_spatial).to(self.device)
            X_genomic = torch.FloatTensor(X_genomic).to(self.device)
            X_temporal = torch.FloatTensor(X_temporal).to(self.device)
            adj_matrix = torch.FloatTensor(adj_matrix).to(self.device)

            # Handle single sample case
            if X_spatial.dim() == 2:
                X_spatial = X_spatial.unsqueeze(0)
                X_genomic = X_genomic.unsqueeze(0)
                X_temporal = X_temporal.unsqueeze(0)

            # Forward pass
            motor_pred, cognitive_pred = model(
                X_spatial, X_genomic, X_temporal, adj_matrix
            )

            return motor_pred.cpu().numpy(), cognitive_pred.cpu().numpy()


def run_dynamic_loss_experiments():
    """Run comprehensive dynamic loss weighting experiments."""
    logger.info("🚀 Starting Phase 5 Dynamic Loss Weighting Experiments")
    logger.info("=" * 70)

    # Load data
    logger.info("📊 Loading Phase 3 real data integration...")
    integrator = RealDataPhase3Integration()
    integrator.load_and_validate_embeddings()
    integrator.create_adjacency_matrices()
    integrator.generate_prognostic_targets()

    # Extract data
    X_spatial = integrator.spatiotemporal_embeddings
    X_genomic = integrator.genomic_embeddings
    X_temporal = integrator.temporal_embeddings
    y_motor = integrator.prognostic_targets[:, 0]
    y_cognitive = integrator.prognostic_targets[:, 1]
    adj_matrix = integrator.adjacency_matrices[0]

    logger.info(f"📈 Dataset: {X_spatial.shape[0]} patients")

    # Test different loss weighting strategies
    strategies = ["fixed", "adaptive", "curriculum"]
    all_results = {}

    evaluator = DynamicLossLOOCVEvaluator(device="cpu")

    for strategy in strategies:
        logger.info(f"\n🔄 Testing '{strategy}' loss weighting strategy...")

        results = evaluator.evaluate_strategy(
            X_spatial,
            X_genomic,
            X_temporal,
            y_motor,
            y_cognitive,
            adj_matrix,
            loss_strategy=strategy,
        )

        all_results[strategy] = results

        logger.info(f"📊 {strategy.upper()} Results:")
        logger.info(f"   Motor R² = {results['motor_r2']:.4f}")
        logger.info(f"   Cognitive AUC = {results['cognitive_auc']:.4f}")

    # Compare strategies
    logger.info("\n🏆 STRATEGY COMPARISON:")
    logger.info("=" * 50)

    best_motor_strategy = max(
        all_results.keys(), key=lambda k: all_results[k]["motor_r2"]
    )
    best_cognitive_strategy = max(
        all_results.keys(), key=lambda k: all_results[k]["cognitive_auc"]
    )

    for strategy in strategies:
        results = all_results[strategy]
        motor_flag = "🥇" if strategy == best_motor_strategy else "  "
        cognitive_flag = "🥇" if strategy == best_cognitive_strategy else "  "

        logger.info(
            f"{motor_flag} {cognitive_flag} {strategy.upper():>10}: Motor R² = {results['motor_r2']:>7.4f} | Cognitive AUC = {results['cognitive_auc']:>6.4f}"
        )

    return all_results


if __name__ == "__main__":
    # Run dynamic loss experiments
    results = run_dynamic_loss_experiments()

    print("\n🎉 Phase 5 Dynamic Loss Weighting Experiments Completed!")
    print("\nBest performing strategies:")

    best_motor = max(results.keys(), key=lambda k: results[k]["motor_r2"])
    best_cognitive = max(results.keys(), key=lambda k: results[k]["cognitive_auc"])

    print(f"🎯 Best Motor R²: {best_motor} ({results[best_motor]['motor_r2']:.4f})")
    print(
        f"🧠 Best Cognitive AUC: {best_cognitive} ({results[best_cognitive]['cognitive_auc']:.4f})"
    )
    print("\n📈 Dynamic loss weighting analysis complete!")
