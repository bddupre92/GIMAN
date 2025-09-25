"""GIMAN Training Engine - Phase 2.

This module implements the core training engine for the Graph-Informed
Multimodal Attention Network (GIMAN). It provides comprehensive training,
validation, and evaluation capabilities for Parkinson's Disease classification.

Key Features:
- Binary classification (PD vs Healthy Control)
- Advanced optimization with learning rate scheduling
- Comprehensive metrics tracking (accuracy, F1, AUC-ROC)
- Early stopping and checkpointing
- Integration with MLflow for experiment tracking
- Cross-validation support

Architecture Integration:
- Uses Phase 1 GNN backbone (GIMANBackbone, GIMANClassifier)
- Supports both node-level and graph-level tasks
- Handles PyTorch Geometric data format
- Real PPMI biomarker and graph data
"""

import logging
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.utils.class_weight import compute_class_weight
from torch.optim import SGD, Adam, AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR

from .models import GIMANBackbone, GIMANClassifier

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FocalLoss(nn.Module):
    """Focal Loss implementation for handling class imbalance.
    
    This loss focuses on hard-to-classify examples and reduces the relative loss
    for well-classified examples, addressing class imbalance issues.
    
    Args:
        alpha (float): Weighting factor for rare class (default: 1.0)
        gamma (float): Focusing parameter (default: 2.0)
        class_weights (torch.Tensor): Optional class weights for additional balancing
    """
    
    def __init__(self, alpha=1.0, gamma=2.0, class_weights=None):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.class_weights = class_weights
        
    def forward(self, inputs, targets):
        ce_loss = nn.functional.cross_entropy(inputs, targets, weight=self.class_weights, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()


class LabelSmoothingCrossEntropy(nn.Module):
    """CrossEntropyLoss with label smoothing and class weighting.
    
    Label smoothing helps prevent overconfident predictions and can improve
    generalization, especially with imbalanced datasets.
    """
    
    def __init__(self, smoothing=0.1, class_weights=None):
        super(LabelSmoothingCrossEntropy, self).__init__()
        self.smoothing = smoothing
        self.class_weights = class_weights
        
    def forward(self, inputs, targets):
        log_prob = nn.functional.log_softmax(inputs, dim=-1)
        
        # Apply label smoothing
        n_classes = inputs.size(-1)
        one_hot = torch.zeros_like(log_prob).scatter(1, targets.unsqueeze(1), 1)
        one_hot = one_hot * (1 - self.smoothing) + self.smoothing / n_classes
        
        # Compute loss
        loss = -(one_hot * log_prob).sum(dim=-1)
        
        # Apply class weights if provided
        if self.class_weights is not None:
            weight = self.class_weights[targets]
            loss = loss * weight
            
        return loss.mean()


class GIMANTrainer:
    """Advanced training engine for GIMAN models.

    This class provides comprehensive training, validation, and evaluation
    capabilities for GIMAN models on Parkinson's Disease classification tasks.

    Features:
    - Multiple optimizer support (Adam, AdamW, SGD)
    - Learning rate scheduling
    - Early stopping with patience
    - Comprehensive metrics tracking
    - Model checkpointing
    - Cross-validation support
    - MLflow experiment tracking integration

    Args:
        model: GIMAN model to train (GIMANBackbone or GIMANClassifier)
        device: Training device ('cpu' or 'cuda')
        optimizer_name: Optimizer type ('adam', 'adamw', 'sgd')
        learning_rate: Initial learning rate
        weight_decay: L2 regularization strength
        scheduler_type: Learning rate scheduler ('plateau', 'step', None)
        early_stopping_patience: Early stopping patience epochs
        checkpoint_dir: Directory for model checkpoints
        experiment_name: Name for MLflow experiment tracking
    """

    def __init__(
        self,
        model: GIMANBackbone | GIMANClassifier,
        device: str = "cpu",
        optimizer_name: str = "adam",
        learning_rate: float = 0.001,
        weight_decay: float = 1e-5,
        scheduler_type: str | None = "plateau",
        early_stopping_patience: int = 10,
        checkpoint_dir: Path | None = None,
        experiment_name: str = "giman_training",
    ):
        """Initialize GIMAN trainer."""
        self.model = model.to(device)
        self.device = device
        self.experiment_name = experiment_name

        # Training configuration
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.early_stopping_patience = early_stopping_patience

        # Setup optimizer
        self.optimizer = self._create_optimizer(
            optimizer_name, learning_rate, weight_decay
        )

        # Setup learning rate scheduler
        self.scheduler = self._create_scheduler(scheduler_type)

        # Setup loss function with class balancing support
        self.class_weights = None
        self.criterion = nn.CrossEntropyLoss()

        # Training state
        self.epoch = 0
        self.best_val_loss = float("inf")
        self.early_stopping_counter = 0
        self.training_history = {
            "train_loss": [],
            "train_acc": [],
            "val_loss": [],
            "val_acc": [],
            "val_f1": [],
            "val_auc": [],
            "learning_rate": [],
        }

        # Checkpointing
        self.checkpoint_dir = (
            Path(checkpoint_dir) if checkpoint_dir else Path("checkpoints")
        )
        self.checkpoint_dir.mkdir(exist_ok=True)

        logger.info("🚀 GIMAN Trainer initialized")
        logger.info(f"   - Model: {type(model).__name__}")
        logger.info(f"   - Parameters: {sum(p.numel() for p in model.parameters()):,}")
        logger.info(f"   - Device: {device}")
        logger.info(f"   - Optimizer: {optimizer_name}")
        logger.info(f"   - Learning Rate: {learning_rate}")
        logger.info(f"   - Scheduler: {scheduler_type}")

    def _create_optimizer(
        self, name: str, lr: float, wd: float
    ) -> torch.optim.Optimizer:
        """Create optimizer based on name."""
        if name.lower() == "adam":
            return Adam(self.model.parameters(), lr=lr, weight_decay=wd)
        elif name.lower() == "adamw":
            return AdamW(self.model.parameters(), lr=lr, weight_decay=wd)
        elif name.lower() == "sgd":
            return SGD(self.model.parameters(), lr=lr, weight_decay=wd, momentum=0.9)
        else:
            raise ValueError(f"Unsupported optimizer: {name}")

    def _create_scheduler(self, scheduler_type: str | None):
        """Create learning rate scheduler."""
        if scheduler_type is None:
            return None
        elif scheduler_type.lower() == "plateau":
            return ReduceLROnPlateau(self.optimizer, mode="min", patience=5, factor=0.5)
        elif scheduler_type.lower() == "step":
            return StepLR(self.optimizer, step_size=10, gamma=0.9)
        else:
            raise ValueError(f"Unsupported scheduler: {scheduler_type}")

    def train_epoch(self, train_loader) -> dict[str, float]:
        """Train model for one epoch."""
        self.model.train()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        for batch_idx, batch in enumerate(train_loader):
            batch = batch.to(self.device)

            # Forward pass
            self.optimizer.zero_grad()

            if isinstance(self.model, GIMANBackbone):
                # For backbone, we need to add a classification head
                embeddings = self.model(batch)
                # This would require additional classification layers
                raise NotImplementedError(
                    "Training GIMANBackbone directly not supported. Use GIMANClassifier."
                )
            else:
                # GIMANClassifier
                output = self.model(batch)
                logits = output["logits"]

            # Calculate loss - for node-level classification
            loss = self.criterion(logits, batch.y)

            # Backward pass
            loss.backward()
            self.optimizer.step()

            # Track metrics
            total_loss += loss.item()
            pred = logits.argmax(dim=1)
            total_correct += (pred == batch.y).sum().item()
            total_samples += batch.y.size(0)

            if batch_idx % 10 == 0:
                logger.debug(f"Batch {batch_idx}: Loss = {loss.item():.4f}")

        metrics = {
            "loss": total_loss / len(train_loader),
            "accuracy": total_correct / total_samples,
        }

        return metrics

    def validate_epoch(self, val_loader) -> dict[str, float]:
        """Validate model for one epoch."""
        self.model.eval()
        total_loss = 0.0
        all_preds = []
        all_targets = []
        all_probs = []

        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(self.device)

                # Forward pass
                output = self.model(batch)
                logits = output["logits"]

                # Calculate loss
                loss = self.criterion(logits, batch.y)
                total_loss += loss.item()

                # Collect predictions and targets
                probs = torch.nn.functional.softmax(logits, dim=1)
                pred = logits.argmax(dim=1)

                all_preds.extend(pred.cpu().numpy())
                all_targets.extend(batch.y.cpu().numpy())

                # Store probabilities for all classes (multiclass support)
                if probs.shape[1] == 2:
                    # Binary classification: store probability of positive class
                    all_probs.extend(probs[:, 1].cpu().numpy())
                else:
                    # Multiclass: store all class probabilities
                    all_probs.extend(probs.cpu().numpy())

        # Calculate comprehensive metrics (multiclass support)
        num_classes = len(set(all_targets))

        if num_classes == 2:
            # Binary classification
            metrics = {
                "loss": total_loss / len(val_loader),
                "accuracy": accuracy_score(all_targets, all_preds),
                "precision": precision_score(all_targets, all_preds, average="binary"),
                "recall": recall_score(all_targets, all_preds, average="binary"),
                "f1": f1_score(all_targets, all_preds, average="binary"),
                "auc_roc": roc_auc_score(all_targets, all_probs),
            }
        else:
            # Multiclass classification
            import numpy as np

            all_probs_array = np.array(all_probs)

            metrics = {
                "loss": total_loss / len(val_loader),
                "accuracy": accuracy_score(all_targets, all_preds),
                "precision": precision_score(all_targets, all_preds, average="macro"),
                "recall": recall_score(all_targets, all_preds, average="macro"),
                "f1": f1_score(all_targets, all_preds, average="macro"),
                "auc_roc": roc_auc_score(
                    all_targets, all_probs_array, average="macro", multi_class="ovr"
                ),
            }

        return metrics

    def train(
        self, train_loader, val_loader, num_epochs: int = 100, verbose: bool = True
    ) -> dict[str, list[float]]:
        """Train the model with validation."""
        logger.info(f"🏃 Starting GIMAN training for {num_epochs} epochs")
        start_time = time.time()

        for epoch in range(num_epochs):
            self.epoch = epoch

            # Training phase
            train_metrics = self.train_epoch(train_loader)

            # Validation phase
            val_metrics = self.validate_epoch(val_loader)

            # Update learning rate scheduler
            if self.scheduler:
                if isinstance(self.scheduler, ReduceLROnPlateau):
                    self.scheduler.step(val_metrics["loss"])
                else:
                    self.scheduler.step()

            # Record metrics
            self.training_history["train_loss"].append(train_metrics["loss"])
            self.training_history["train_acc"].append(train_metrics["accuracy"])
            self.training_history["val_loss"].append(val_metrics["loss"])
            self.training_history["val_acc"].append(val_metrics["accuracy"])
            self.training_history["val_f1"].append(val_metrics["f1"])
            self.training_history["val_auc"].append(val_metrics["auc_roc"])
            self.training_history["learning_rate"].append(
                self.optimizer.param_groups[0]["lr"]
            )

            # Check for best model
            if val_metrics["loss"] < self.best_val_loss:
                self.best_val_loss = val_metrics["loss"]
                self.early_stopping_counter = 0
                self.save_checkpoint(is_best=True)
            else:
                self.early_stopping_counter += 1

            # Verbose logging
            if verbose and (epoch + 1) % 5 == 0:
                logger.info(
                    f"Epoch {epoch + 1:3d}/{num_epochs} | "
                    f"Train Loss: {train_metrics['loss']:.4f} | "
                    f"Train Acc: {train_metrics['accuracy']:.4f} | "
                    f"Val Loss: {val_metrics['loss']:.4f} | "
                    f"Val Acc: {val_metrics['accuracy']:.4f} | "
                    f"Val F1: {val_metrics['f1']:.4f} | "
                    f"Val AUC: {val_metrics['auc_roc']:.4f} | "
                    f"LR: {self.optimizer.param_groups[0]['lr']:.2e}"
                )

            # Early stopping
            if self.early_stopping_counter >= self.early_stopping_patience:
                logger.info(f"Early stopping triggered after {epoch + 1} epochs")
                break

        total_time = time.time() - start_time
        logger.info(f"✅ Training completed in {total_time:.2f} seconds")
        logger.info(f"   - Best validation loss: {self.best_val_loss:.4f}")
        logger.info(f"   - Final validation accuracy: {val_metrics['accuracy']:.4f}")
        logger.info(f"   - Final validation F1: {val_metrics['f1']:.4f}")
        logger.info(f"   - Final validation AUC: {val_metrics['auc_roc']:.4f}")

        return self.training_history

    def save_checkpoint(self, is_best: bool = False):
        """Save model checkpoint."""
        checkpoint = {
            "epoch": self.epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "best_val_loss": self.best_val_loss,
            "training_history": self.training_history,
        }

        if self.scheduler:
            checkpoint["scheduler_state_dict"] = self.scheduler.state_dict()

        # Save regular checkpoint
        checkpoint_path = self.checkpoint_dir / f"checkpoint_epoch_{self.epoch}.pt"
        torch.save(checkpoint, checkpoint_path)

        # Save best model
        if is_best:
            best_path = self.checkpoint_dir / "best_model.pt"
            torch.save(checkpoint, best_path)
            logger.debug(f"💾 Saved best model checkpoint to {best_path}")

    def load_checkpoint(self, checkpoint_path: Path):
        """Load model from checkpoint."""
        checkpoint = torch.load(
            checkpoint_path, map_location=self.device, weights_only=False
        )

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.epoch = checkpoint["epoch"]
        self.best_val_loss = checkpoint["best_val_loss"]
        self.training_history = checkpoint["training_history"]

        if self.scheduler and "scheduler_state_dict" in checkpoint:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        logger.info(f"📂 Loaded checkpoint from epoch {self.epoch}")

    def evaluate(self, test_loader) -> dict[str, Any]:
        """Comprehensive model evaluation."""
        logger.info("🧪 Running comprehensive model evaluation")

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

                # Store probabilities for all classes (multiclass support)
                if probs.shape[1] == 2:
                    # Binary classification: store probability of positive class
                    all_probs.extend(probs[:, 1].cpu().numpy())
                else:
                    # Multiclass: store all class probabilities
                    all_probs.extend(probs.cpu().numpy())

        # Comprehensive evaluation metrics with multiclass support
        num_classes = len(set(all_targets))

        if num_classes == 2:
            # Binary classification
            results = {
                "accuracy": accuracy_score(all_targets, all_preds),
                "precision": precision_score(all_targets, all_preds, average="binary"),
                "recall": recall_score(all_targets, all_preds, average="binary"),
                "f1": f1_score(all_targets, all_preds, average="binary"),
                "auc_roc": roc_auc_score(all_targets, all_probs),
                "confusion_matrix": confusion_matrix(all_targets, all_preds),
                "classification_report": classification_report(
                    all_targets,
                    all_preds,
                    target_names=["Healthy Control", "Parkinson's Disease"],
                    output_dict=True,
                ),
            }
        else:
            # Multiclass classification
            import numpy as np

            all_probs_array = np.array(all_probs)

            results = {
                "accuracy": accuracy_score(all_targets, all_preds),
                "precision": precision_score(all_targets, all_preds, average="macro"),
                "recall": recall_score(all_targets, all_preds, average="macro"),
                "f1": f1_score(all_targets, all_preds, average="macro"),
                "auc_roc": roc_auc_score(
                    all_targets, all_probs_array, average="macro", multi_class="ovr"
                ),
                "confusion_matrix": confusion_matrix(all_targets, all_preds),
                "classification_report": classification_report(
                    all_targets,
                    all_preds,
                    target_names=[
                        "Healthy Control",
                        "Parkinson's Disease",
                        "Prodromal",
                        "SWEDD",
                    ],
                    output_dict=True,
                ),
            }

        # Log results
        logger.info("📊 Evaluation Results:")
        logger.info(f"   - Accuracy: {results['accuracy']:.4f}")
        logger.info(f"   - Precision: {results['precision']:.4f}")
        logger.info(f"   - Recall: {results['recall']:.4f}")
        logger.info(f"   - F1 Score: {results['f1']:.4f}")
        logger.info(f"   - AUC-ROC: {results['auc_roc']:.4f}")
        logger.info(f"   - Confusion Matrix:\n{results['confusion_matrix']}")

        return results

    def get_training_summary(self) -> dict[str, Any]:
        """Get comprehensive training summary."""
        return {
            "model_info": {
                "type": type(self.model).__name__,
                "parameters": sum(p.numel() for p in self.model.parameters()),
                "device": self.device,
            },
            "training_config": {
                "optimizer": type(self.optimizer).__name__,
                "learning_rate": self.learning_rate,
                "weight_decay": self.weight_decay,
                "scheduler": type(self.scheduler).__name__ if self.scheduler else None,
                "early_stopping_patience": self.early_stopping_patience,
            },
            "training_results": {
                "total_epochs": len(self.training_history["train_loss"]),
                "best_val_loss": self.best_val_loss,
                "final_metrics": {
                    "train_loss": self.training_history["train_loss"][-1]
                    if self.training_history["train_loss"]
                    else None,
                    "val_loss": self.training_history["val_loss"][-1]
                    if self.training_history["val_loss"]
                    else None,
                    "val_accuracy": self.training_history["val_acc"][-1]
                    if self.training_history["val_acc"]
                    else None,
                    "val_f1": self.training_history["val_f1"][-1]
                    if self.training_history["val_f1"]
                    else None,
                    "val_auc": self.training_history["val_auc"][-1]
                    if self.training_history["val_auc"]
                    else None,
                },
            },
            "history": self.training_history,
        }

    def compute_class_weights(self, train_loader):
        """Compute class weights for imbalanced dataset using sklearn.
        
        Args:
            train_loader: Training data loader
            
        Returns:
            torch.Tensor: Class weights for loss function
        """
        # Collect all labels from training data
        all_labels = []
        for batch in train_loader:
            if hasattr(batch, 'y'):
                all_labels.extend(batch.y.cpu().numpy())
            elif isinstance(batch, (list, tuple)) and len(batch) >= 2:
                all_labels.extend(batch[1].cpu().numpy())
        
        # Compute balanced class weights
        unique_classes = np.unique(all_labels)
        class_weights_array = compute_class_weight(
            'balanced', 
            classes=unique_classes, 
            y=all_labels
        )
        
        # Convert to torch tensor
        class_weights = torch.FloatTensor(class_weights_array).to(self.device)
        
        logger.info(f"Computed class weights: {class_weights}")
        logger.info(f"Class distribution: {np.bincount(all_labels)}")
        
        return class_weights
    
    def setup_focal_loss(self, train_loader, alpha=1.0, gamma=2.0):
        """Setup Focal Loss with computed class weights.
        
        Args:
            train_loader: Training data loader for class weight computation
            alpha (float): Weighting factor for rare class
            gamma (float): Focusing parameter
        """
        # Compute class weights
        class_weights = self.compute_class_weights(train_loader)
        
        # Setup focal loss
        self.criterion = FocalLoss(
            alpha=alpha, 
            gamma=gamma, 
            class_weights=class_weights
        )
        self.class_weights = class_weights
        
        logger.info(f"Setup Focal Loss with alpha={alpha}, gamma={gamma}")
        
    def setup_weighted_loss(self, train_loader):
        """Setup weighted CrossEntropyLoss with computed class weights.
        
        Args:
            train_loader: Training data loader for class weight computation
        """
        # Compute class weights
        class_weights = self.compute_class_weights(train_loader)
        
        # Setup weighted cross entropy loss
        self.criterion = nn.CrossEntropyLoss(weight=class_weights)
        self.class_weights = class_weights
        
        logger.info(f"Setup Weighted CrossEntropyLoss")
        
    def setup_label_smoothing_loss(self, train_loader, smoothing=0.1):
        """Setup Label Smoothing CrossEntropyLoss with computed class weights.
        
        Args:
            train_loader: Training data loader for class weight computation
            smoothing (float): Label smoothing factor (0.1 = 10% smoothing)
        """
        # Compute class weights
        class_weights = self.compute_class_weights(train_loader)
        
        # Setup label smoothing loss
        self.criterion = LabelSmoothingCrossEntropy(
            smoothing=smoothing,
            class_weights=class_weights
        )
        self.class_weights = class_weights
        
        logger.info(f"Setup Label Smoothing CrossEntropyLoss with smoothing={smoothing}")
