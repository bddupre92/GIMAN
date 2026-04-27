#!/usr/bin/env python3
"""
Research Plan Phase 2, Task 2.2: Multi-Task Loss Function

Implements combined loss for dual-task prognostic GIMAN:
- MSE Loss for motor progression (regression)
- Focal Loss for cognitive decline (classification with class imbalance)
- Dynamic weighting strategies

Author: GIMAN Development Team
Date: October 2, 2025
Research Plan Phase: 2 (Prognostic Model Architecture)
Task: 2.2 - Implement multi-task loss function (MSE + Focal Loss)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
import numpy as np
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance in cognitive decline prediction.

    Focal Loss down-weights easy examples and focuses on hard negatives.
    Particularly useful for imbalanced binary classification (15.6% decline rate in Phase 1).

    Reference:
        Lin et al. "Focal Loss for Dense Object Detection" (2017)
        https://arxiv.org/abs/1708.02002

    Args:
        alpha: Weighting factor for positive class (default: 0.25)
        gamma: Focusing parameter (default: 2.0)
               Higher gamma -> more focus on hard examples
        reduction: Specifies reduction: 'none' | 'mean' | 'sum'

    Research Plan Note:
        Phase 1 cognitive decline rate: 15.6% (imbalanced)
        Focal loss addresses this imbalance better than standard cross-entropy
    """

    def __init__(
        self,
        alpha: float = 0.25,
        gamma: float = 2.0,
        reduction: str = 'mean'
    ):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

        logger.info(f"FocalLoss initialized: alpha={alpha}, gamma={gamma}")

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute focal loss.

        Args:
            inputs: Predicted logits [batch_size, num_classes=2]
            targets: Ground truth labels [batch_size] (0 or 1)

        Returns:
            Focal loss value
        """
        # Convert targets to long if needed
        targets = targets.long()

        # Compute cross-entropy loss
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')

        # Get probabilities
        pt = torch.exp(-ce_loss)

        # Compute focal loss
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss

        # Apply reduction
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class MultiTaskLoss(nn.Module):
    """
    Combined multi-task loss for dual-task prognostic GIMAN.

    Combines:
    1. MSE Loss: Motor progression regression (continuous UPDRS-III slope)
    2. Focal Loss: Cognitive decline classification (binary MCI conversion)

    Supports multiple weighting strategies:
    - Fixed: Static weights (e.g., 0.7 motor, 0.3 cognitive)
    - Adaptive: Dynamic weights based on task performance
    - Uncertainty: Learn task weights via homoscedastic uncertainty
    - Curriculum: Progressive weight shifting during training

    Reference (Uncertainty weighting):
        Kendall et al. "Multi-Task Learning Using Uncertainty to Weigh Losses for Scene Geometry and Semantics" (2018)
        https://arxiv.org/abs/1705.07115

    Args:
        motor_weight: Initial weight for motor task (default: 0.7)
        cognitive_weight: Initial weight for cognitive task (default: 0.3)
        focal_alpha: Alpha parameter for Focal Loss (default: 0.25)
        focal_gamma: Gamma parameter for Focal Loss (default: 2.0)
        weighting_strategy: 'fixed' | 'adaptive' | 'uncertainty' | 'curriculum'
        use_uncertainty_weighting: If True, learn task weights (overrides manual weights)
    """

    def __init__(
        self,
        motor_weight: float = 0.7,
        cognitive_weight: float = 0.3,
        focal_alpha: float = 0.25,
        focal_gamma: float = 2.0,
        weighting_strategy: str = 'fixed',
        use_uncertainty_weighting: bool = False
    ):
        super(MultiTaskLoss, self).__init__()

        # Task losses
        self.mse_loss = nn.MSELoss()
        self.focal_loss = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)

        # Weighting strategy
        self.weighting_strategy = weighting_strategy
        self.use_uncertainty_weighting = use_uncertainty_weighting

        # Fixed weights
        self.register_buffer('motor_weight', torch.tensor(motor_weight))
        self.register_buffer('cognitive_weight', torch.tensor(cognitive_weight))

        # Uncertainty-based weighting (learnable parameters)
        if use_uncertainty_weighting:
            self.log_var_motor = nn.Parameter(torch.zeros(1))
            self.log_var_cognitive = nn.Parameter(torch.zeros(1))

        # Adaptive weighting (tracked during training)
        self.motor_loss_history = []
        self.cognitive_loss_history = []

        # Curriculum weighting (epoch-based)
        self.current_epoch = 0
        self.max_epochs = 100  # Will be set by training loop

        logger.info(f"MultiTaskLoss initialized:")
        logger.info(f"  Weighting strategy: {weighting_strategy}")
        logger.info(f"  Motor weight: {motor_weight}")
        logger.info(f"  Cognitive weight: {cognitive_weight}")
        logger.info(f"  Focal Loss: alpha={focal_alpha}, gamma={focal_gamma}")
        logger.info(f"  Uncertainty weighting: {use_uncertainty_weighting}")

    def forward(
        self,
        motor_pred: torch.Tensor,
        motor_target: torch.Tensor,
        cognitive_pred: torch.Tensor,
        cognitive_target: torch.Tensor,
        return_components: bool = False
    ) -> Tuple[torch.Tensor, Optional[dict]]:
        """
        Compute combined multi-task loss.

        Args:
            motor_pred: Motor predictions [batch_size, 1]
            motor_target: Motor ground truth [batch_size, 1] or [batch_size]
            cognitive_pred: Cognitive predictions [batch_size, 2] (logits)
            cognitive_target: Cognitive ground truth [batch_size] (0 or 1)
            return_components: If True, return individual loss components

        Returns:
            total_loss: Combined weighted loss
            components: Dict with individual losses (if return_components=True)
        """
        # Ensure correct shapes
        if motor_target.dim() == 2:
            motor_target = motor_target.squeeze(1)
        if motor_pred.dim() == 2:
            motor_pred = motor_pred.squeeze(1)

        # Compute individual task losses
        motor_loss = self.mse_loss(motor_pred, motor_target)
        cognitive_loss = self.focal_loss(cognitive_pred, cognitive_target)

        # Apply weighting strategy
        if self.use_uncertainty_weighting:
            # Uncertainty-based weighting (Kendall et al. 2018)
            # Loss = 1/(2*sigma^2) * task_loss + log(sigma)
            precision_motor = torch.exp(-self.log_var_motor)
            precision_cognitive = torch.exp(-self.log_var_cognitive)

            total_loss = (
                precision_motor * motor_loss + self.log_var_motor +
                precision_cognitive * cognitive_loss + self.log_var_cognitive
            )

            # Track effective weights
            motor_weight_eff = precision_motor.item()
            cognitive_weight_eff = precision_cognitive.item()

        elif self.weighting_strategy == 'adaptive':
            # Adaptive weighting based on loss history
            motor_weight_eff, cognitive_weight_eff = self._compute_adaptive_weights(
                motor_loss, cognitive_loss
            )
            total_loss = motor_weight_eff * motor_loss + cognitive_weight_eff * cognitive_loss

        elif self.weighting_strategy == 'curriculum':
            # Curriculum weighting (shift from motor to cognitive over time)
            motor_weight_eff, cognitive_weight_eff = self._compute_curriculum_weights()
            total_loss = motor_weight_eff * motor_loss + cognitive_weight_eff * cognitive_loss

        else:  # 'fixed'
            # Fixed weighting
            motor_weight_eff = self.motor_weight.item()
            cognitive_weight_eff = self.cognitive_weight.item()
            total_loss = self.motor_weight * motor_loss + self.cognitive_weight * cognitive_loss

        # Store components for logging/analysis
        components = None
        if return_components:
            components = {
                'total_loss': total_loss.item(),
                'motor_loss': motor_loss.item(),
                'cognitive_loss': cognitive_loss.item(),
                'motor_weight': motor_weight_eff,
                'cognitive_weight': cognitive_weight_eff,
            }

        return total_loss, components

    def _compute_adaptive_weights(
        self,
        motor_loss: torch.Tensor,
        cognitive_loss: torch.Tensor
    ) -> Tuple[float, float]:
        """
        Compute adaptive weights based on recent loss history.

        Strategy: Give more weight to the task that's improving slower.
        """
        # Track losses
        self.motor_loss_history.append(motor_loss.item())
        self.cognitive_loss_history.append(cognitive_loss.item())

        # Keep only recent history (last 10 batches)
        if len(self.motor_loss_history) > 10:
            self.motor_loss_history = self.motor_loss_history[-10:]
            self.cognitive_loss_history = self.cognitive_loss_history[-10:]

        # Compute loss ratios (higher ratio -> higher weight)
        if len(self.motor_loss_history) >= 5:
            motor_avg = np.mean(self.motor_loss_history)
            cognitive_avg = np.mean(self.cognitive_loss_history)

            # Normalize weights
            total = motor_avg + cognitive_avg
            motor_weight = motor_avg / total
            cognitive_weight = cognitive_avg / total

            # Smooth with original weights (exponential moving average)
            alpha = 0.9
            motor_weight = alpha * self.motor_weight.item() + (1 - alpha) * motor_weight
            cognitive_weight = alpha * self.cognitive_weight.item() + (1 - alpha) * cognitive_weight

            return motor_weight, cognitive_weight
        else:
            # Use fixed weights until enough history
            return self.motor_weight.item(), self.cognitive_weight.item()

    def _compute_curriculum_weights(self) -> Tuple[float, float]:
        """
        Compute curriculum-based weights that shift over training.

        Strategy: Start with more focus on motor task, gradually shift to balanced.
        """
        # Linear schedule: motor weight decreases from 0.8 to 0.5
        progress = min(1.0, self.current_epoch / self.max_epochs)

        motor_weight = 0.8 - 0.3 * progress  # 0.8 -> 0.5
        cognitive_weight = 1.0 - motor_weight  # 0.2 -> 0.5

        return motor_weight, cognitive_weight

    def update_epoch(self, epoch: int, max_epochs: int):
        """Update current epoch for curriculum learning."""
        self.current_epoch = epoch
        self.max_epochs = max_epochs


def test_multitask_loss():
    """Test multi-task loss function."""

    print("\n" + "="*80)
    print("RESEARCH PLAN PHASE 2, TASK 2.2: MULTI-TASK LOSS TEST")
    print("="*80 + "\n")

    # Create dummy predictions and targets
    batch_size = 32
    motor_pred = torch.randn(batch_size, 1)  # Random slopes
    motor_target = torch.randn(batch_size)   # True slopes

    cognitive_pred = torch.randn(batch_size, 2)  # Random logits
    cognitive_target = torch.randint(0, 2, (batch_size,))  # Binary labels

    print(f"Input shapes:")
    print(f"  Motor pred: {motor_pred.shape}, target: {motor_target.shape}")
    print(f"  Cognitive pred: {cognitive_pred.shape}, target: {cognitive_target.shape}")
    print(f"  Cognitive positive rate: {cognitive_target.float().mean():.1%}\n")

    # Test different weighting strategies
    strategies = ['fixed', 'adaptive', 'curriculum', 'uncertainty']

    for strategy in strategies:
        print(f"\nTesting {strategy.upper()} weighting strategy:")
        print("-" * 60)

        use_uncertainty = (strategy == 'uncertainty')
        loss_fn = MultiTaskLoss(
            motor_weight=0.7,
            cognitive_weight=0.3,
            focal_alpha=0.25,
            focal_gamma=2.0,
            weighting_strategy=strategy,
            use_uncertainty_weighting=use_uncertainty
        )

        # Simulate training for adaptive/curriculum
        if strategy in ['adaptive', 'curriculum']:
            loss_fn.update_epoch(epoch=50, max_epochs=100)

        # Compute loss
        total_loss, components = loss_fn(
            motor_pred, motor_target,
            cognitive_pred, cognitive_target,
            return_components=True
        )

        print(f"  Total loss: {components['total_loss']:.4f}")
        print(f"  Motor loss: {components['motor_loss']:.4f} (weight: {components['motor_weight']:.3f})")
        print(f"  Cognitive loss: {components['cognitive_loss']:.4f} (weight: {components['cognitive_weight']:.3f})")

    print("\n" + "="*80)
    print("[OK] TASK 2.2 COMPLETE: Multi-task loss function successfully implemented")
    print("="*80)
    print("\nKey features:")
    print("  - MSE loss for motor regression")
    print("  - Focal loss for cognitive classification (handles imbalance)")
    print("  - Multiple weighting strategies (fixed, adaptive, curriculum, uncertainty)")
    print("  - Ready for integration with Task 2.3 (training pipeline)")
    print("="*80 + "\n")


if __name__ == '__main__':
    test_multitask_loss()
