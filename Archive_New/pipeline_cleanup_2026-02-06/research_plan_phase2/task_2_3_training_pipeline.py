#!/usr/bin/env python3
"""
Research Plan Phase 2, Task 2.3: Training Pipeline for Dual-Task Learning

Implements complete training pipeline for GIMANPrognostic model:
- Loads Phase 1 prognostic dataset (2,046 patients)
- Creates patient similarity graph
- Trains dual-task model (motor + cognitive)
- 5-fold cross-validation
- Model checkpointing and evaluation

Author: GIMAN Development Team
Date: October 2, 2025
Research Plan Phase: 2 (Prognostic Model Architecture)
Task: 2.3 - Training pipeline for dual-task learning
"""

import sys
from pathlib import Path
import json
from datetime import datetime
from typing import Dict, Tuple, Optional
import logging

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, roc_auc_score, mean_absolute_error
from sklearn.metrics.pairwise import cosine_similarity

# Import our Task 2.1 and 2.2 implementations
from task_2_1_giman_prognostic_model import GIMANPrognostic, create_giman_prognostic
from task_2_2_multitask_loss import MultiTaskLoss

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Phase1DataLoader:
    """
    Load and prepare Phase 1 prognostic dataset for training.

    Loads the 2,046-patient longitudinal cohort with validated prognostic endpoints.
    """

    def __init__(self, phase1_dir: Path):
        """
        Initialize data loader.

        Args:
            phase1_dir: Path to phase1 directory containing prognostic dataset
        """
        self.phase1_dir = Path(phase1_dir)
        self.df = None
        self.features = None
        self.motor_targets = None
        self.cognitive_targets = None
        self.patient_ids = None

        logger.info(f"Phase1DataLoader initialized with directory: {phase1_dir}")

    def load_data(self) -> pd.DataFrame:
        """
        Load Phase 1 prognostic dataset.

        Returns:
            DataFrame with prognostic data
        """
        # Find the most recent prognostic dataset
        pattern = 'prognostic_dataset_complete_*.csv'
        files = list(self.phase1_dir.glob(pattern))

        if not files:
            raise FileNotFoundError(
                f"No prognostic dataset found in {self.phase1_dir}\n"
                f"Looking for: {pattern}"
            )

        # Use most recent file
        data_file = max(files, key=lambda p: p.stat().st_mtime)
        logger.info(f"Loading: {data_file.name}")

        self.df = pd.read_csv(data_file)
        logger.info(f"Loaded {len(self.df)} patients")

        return self.df

    def prepare_features(self, feature_cols: Optional[list] = None) -> np.ndarray:
        """
        Extract and prepare baseline features.

        Args:
            feature_cols: List of feature column names (default: auto-select)

        Returns:
            Feature matrix [num_patients, num_features]
        """
        if feature_cols is None:
            # Default baseline feature set
            feature_cols = [
                'UPDRS_III_BL',    # Motor score at baseline
                'MOCA_BL',         # Cognitive score at baseline
                'AGE_APPROX',      # Age
                'SEX',             # Gender (0/1)
                'HANDED',          # Handedness
                'HISPLAT',         # Hispanic/Latino ethnicity
                'BIRTH_YEAR',      # Birth year (alternative age proxy)
            ]

        # Check which features are available
        available_cols = [col for col in feature_cols if col in self.df.columns]
        missing_cols = [col for col in feature_cols if col not in self.df.columns]

        if missing_cols:
            logger.warning(f"Missing columns: {missing_cols}")

        logger.info(f"Using features: {available_cols}")

        # Extract features
        X = self.df[available_cols].values

        # Handle missing values (simple imputation with median)
        for i in range(X.shape[1]):
            col_data = X[:, i]
            mask = ~np.isnan(col_data)
            if mask.sum() > 0:
                median_val = np.median(col_data[mask])
                X[~mask, i] = median_val

        self.features = X
        logger.info(f"Feature matrix shape: {X.shape}")

        return X

    def prepare_targets(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract prognostic targets.

        Returns:
            motor_targets: UPDRS-III slopes [num_patients]
            cognitive_targets: MCI conversion labels [num_patients]
        """
        # Motor progression target (continuous)
        motor_col = 'motor_slope_per_year'
        if motor_col not in self.df.columns:
            # Try alternative column names
            if 'MOTOR_PROGRESSION_SLOPE' in self.df.columns:
                motor_col = 'MOTOR_PROGRESSION_SLOPE'
            else:
                raise ValueError(f"Motor target column not found. Available: {self.df.columns.tolist()}")

        self.motor_targets = self.df[motor_col].values

        # Cognitive decline target (binary)
        cognitive_col = 'cognitive_decline'
        if cognitive_col not in self.df.columns:
            # Try alternative column names
            if 'COGNITIVE_DECLINE_LABEL' in self.df.columns:
                cognitive_col = 'COGNITIVE_DECLINE_LABEL'
            else:
                raise ValueError(f"Cognitive target column not found")

        self.cognitive_targets = self.df[cognitive_col].values

        # Handle missing targets
        motor_valid = ~np.isnan(self.motor_targets)
        cognitive_valid = ~np.isnan(self.cognitive_targets)
        both_valid = motor_valid & cognitive_valid

        logger.info(f"Motor targets: {motor_valid.sum()} valid / {len(self.motor_targets)} total")
        logger.info(f"  Mean slope: {np.nanmean(self.motor_targets):.3f} ± {np.nanstd(self.motor_targets):.3f} pts/year")
        logger.info(f"Cognitive targets: {cognitive_valid.sum()} valid / {len(self.cognitive_targets)} total")
        logger.info(f"  Decline rate: {np.nanmean(self.cognitive_targets):.1%}")
        logger.info(f"Both valid: {both_valid.sum()} patients")

        return self.motor_targets, self.cognitive_targets

    def get_patient_ids(self) -> np.ndarray:
        """Get patient IDs."""
        self.patient_ids = self.df['PATNO'].values
        return self.patient_ids


def create_patient_similarity_graph(
    features: np.ndarray,
    k: int = 6,
    similarity_metric: str = 'cosine'
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Create patient similarity graph using k-nearest neighbors.

    Args:
        features: Feature matrix [num_patients, num_features]
        k: Number of nearest neighbors
        similarity_metric: 'cosine' or 'euclidean'

    Returns:
        edge_index: Edge connectivity [2, num_edges]
        edge_weights: Edge weights [num_edges]
    """
    logger.info(f"Creating patient similarity graph (k={k}, metric={similarity_metric})")

    num_patients = features.shape[0]

    # Compute similarity matrix
    if similarity_metric == 'cosine':
        similarity_matrix = cosine_similarity(features)
    else:
        # Euclidean distance converted to similarity
        from sklearn.metrics.pairwise import euclidean_distances
        dist_matrix = euclidean_distances(features)
        similarity_matrix = 1 / (1 + dist_matrix)

    # For each patient, find k nearest neighbors
    edges = []
    weights = []

    for i in range(num_patients):
        # Get similarities for patient i
        similarities = similarity_matrix[i]

        # Exclude self (diagonal)
        similarities[i] = -np.inf

        # Get indices of k nearest neighbors
        neighbor_indices = np.argsort(similarities)[-k:]

        # Add edges
        for j in neighbor_indices:
            edges.append([i, j])
            weights.append(similarity_matrix[i, j])

    # Convert to tensors
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    edge_weights = torch.tensor(weights, dtype=torch.float32)

    logger.info(f"Graph created: {num_patients} nodes, {len(edges)} edges")
    logger.info(f"  Average degree: {len(edges) / num_patients:.1f}")
    logger.info(f"  Average edge weight: {edge_weights.mean():.3f}")

    return edge_index, edge_weights


class GIMANTrainer:
    """
    Trainer for GIMANPrognostic model.

    Handles training loop, validation, checkpointing, and logging.
    """

    def __init__(
        self,
        model: GIMANPrognostic,
        loss_fn: MultiTaskLoss,
        device: str = 'cpu',
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        patience: int = 10,
        min_delta: float = 1e-4
    ):
        """
        Initialize trainer.

        Args:
            model: GIMANPrognostic model
            loss_fn: MultiTaskLoss function
            device: 'cpu' or 'cuda'
            learning_rate: Initial learning rate
            weight_decay: L2 regularization
            patience: Early stopping patience
            min_delta: Minimum improvement for early stopping
        """
        self.model = model.to(device)
        self.loss_fn = loss_fn
        self.device = device

        # Optimizer and scheduler
        self.optimizer = AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        self.scheduler = ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=5,
            verbose=True,
            min_lr=1e-6
        )

        # Early stopping
        self.patience = patience
        self.min_delta = min_delta
        self.best_val_loss = np.inf
        self.patience_counter = 0

        # History
        self.history = {
            'train_loss': [],
            'train_motor_loss': [],
            'train_cognitive_loss': [],
            'val_loss': [],
            'val_motor_loss': [],
            'val_cognitive_loss': [],
            'val_motor_r2': [],
            'val_cognitive_auc': [],
            'learning_rate': []
        }

        logger.info(f"GIMANTrainer initialized:")
        logger.info(f"  Device: {device}")
        logger.info(f"  Learning rate: {learning_rate}")
        logger.info(f"  Weight decay: {weight_decay}")
        logger.info(f"  Patience: {patience}")

    def train_epoch(
        self,
        features: torch.Tensor,
        motor_targets: torch.Tensor,
        cognitive_targets: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weights: Optional[torch.Tensor] = None
    ) -> Dict[str, float]:
        """
        Train for one epoch.

        Args:
            features: Node features [num_nodes, num_features]
            motor_targets: Motor targets [num_nodes]
            cognitive_targets: Cognitive targets [num_nodes]
            edge_index: Edge connectivity [2, num_edges]
            edge_weights: Edge weights [num_edges]

        Returns:
            Dictionary of training metrics
        """
        self.model.train()

        # Forward pass
        motor_pred, cognitive_pred = self.model(features, edge_index, edge_weights)

        # Compute loss
        loss, components = self.loss_fn(
            motor_pred, motor_targets,
            cognitive_pred, cognitive_targets,
            return_components=True
        )

        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

        self.optimizer.step()

        return components

    @torch.no_grad()
    def validate(
        self,
        features: torch.Tensor,
        motor_targets: torch.Tensor,
        cognitive_targets: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weights: Optional[torch.Tensor] = None
    ) -> Dict[str, float]:
        """
        Validate model.

        Args:
            features: Node features
            motor_targets: Motor targets
            cognitive_targets: Cognitive targets
            edge_index: Edge connectivity
            edge_weights: Edge weights

        Returns:
            Dictionary of validation metrics
        """
        self.model.eval()

        # Forward pass
        motor_pred, cognitive_pred = self.model(features, edge_index, edge_weights)

        # Compute loss
        loss, components = self.loss_fn(
            motor_pred, motor_targets,
            cognitive_pred, cognitive_targets,
            return_components=True
        )

        # Compute task-specific metrics
        # Motor: R² score
        motor_pred_np = motor_pred.squeeze().cpu().numpy()
        motor_targets_np = motor_targets.cpu().numpy()
        motor_r2 = r2_score(motor_targets_np, motor_pred_np)

        # Cognitive: AUC-ROC
        cognitive_probs = torch.softmax(cognitive_pred, dim=1)[:, 1].cpu().numpy()
        cognitive_targets_np = cognitive_targets.cpu().numpy()

        # Check if we have both classes in validation set
        if len(np.unique(cognitive_targets_np)) > 1:
            cognitive_auc = roc_auc_score(cognitive_targets_np, cognitive_probs)
        else:
            cognitive_auc = 0.5  # Default if only one class present

        # Add metrics to components
        components['motor_r2'] = motor_r2
        components['cognitive_auc'] = cognitive_auc

        return components

    def fit(
        self,
        train_features: torch.Tensor,
        train_motor: torch.Tensor,
        train_cognitive: torch.Tensor,
        train_edge_index: torch.Tensor,
        val_features: torch.Tensor,
        val_motor: torch.Tensor,
        val_cognitive: torch.Tensor,
        val_edge_index: torch.Tensor,
        num_epochs: int = 100,
        verbose: bool = True
    ) -> Dict:
        """
        Train model with validation.

        Args:
            train_* : Training data
            val_*: Validation data
            num_epochs: Maximum number of epochs
            verbose: Print progress

        Returns:
            Training history
        """
        logger.info(f"Starting training for {num_epochs} epochs...")

        for epoch in range(num_epochs):
            # Train
            train_metrics = self.train_epoch(
                train_features, train_motor, train_cognitive, train_edge_index
            )

            # Validate
            val_metrics = self.validate(
                val_features, val_motor, val_cognitive, val_edge_index
            )

            # Update history
            self.history['train_loss'].append(train_metrics['total_loss'])
            self.history['train_motor_loss'].append(train_metrics['motor_loss'])
            self.history['train_cognitive_loss'].append(train_metrics['cognitive_loss'])
            self.history['val_loss'].append(val_metrics['total_loss'])
            self.history['val_motor_loss'].append(val_metrics['motor_loss'])
            self.history['val_cognitive_loss'].append(val_metrics['cognitive_loss'])
            self.history['val_motor_r2'].append(val_metrics['motor_r2'])
            self.history['val_cognitive_auc'].append(val_metrics['cognitive_auc'])
            self.history['learning_rate'].append(self.optimizer.param_groups[0]['lr'])

            # Learning rate scheduling
            self.scheduler.step(val_metrics['total_loss'])

            # Print progress
            if verbose and (epoch + 1) % 10 == 0:
                logger.info(
                    f"Epoch {epoch+1}/{num_epochs} | "
                    f"Train Loss: {train_metrics['total_loss']:.4f} | "
                    f"Val Loss: {val_metrics['total_loss']:.4f} | "
                    f"Motor R2: {val_metrics['motor_r2']:.4f} | "
                    f"Cognitive AUC: {val_metrics['cognitive_auc']:.4f}"
                )

            # Early stopping
            if val_metrics['total_loss'] < self.best_val_loss - self.min_delta:
                self.best_val_loss = val_metrics['total_loss']
                self.patience_counter = 0
            else:
                self.patience_counter += 1

            if self.patience_counter >= self.patience:
                logger.info(f"Early stopping at epoch {epoch+1}")
                break

        return self.history

    def save_model(self, filepath: Path):
        """Save model checkpoint."""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'history': self.history,
            'best_val_loss': self.best_val_loss
        }, filepath)
        logger.info(f"Model saved to: {filepath}")

    def load_model(self, filepath: Path):
        """Load model checkpoint."""
        checkpoint = torch.load(filepath)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.history = checkpoint['history']
        self.best_val_loss = checkpoint['best_val_loss']
        logger.info(f"Model loaded from: {filepath}")


def run_cross_validation(
    features: np.ndarray,
    motor_targets: np.ndarray,
    cognitive_targets: np.ndarray,
    n_folds: int = 5,
    num_epochs: int = 100,
    output_dir: Optional[Path] = None
) -> Dict:
    """
    Run k-fold cross-validation.

    Args:
        features: Feature matrix
        motor_targets: Motor targets
        cognitive_targets: Cognitive targets
        n_folds: Number of CV folds
        num_epochs: Epochs per fold
        output_dir: Directory to save results

    Returns:
        Cross-validation results
    """
    logger.info(f"\n{'='*80}")
    logger.info(f"STARTING {n_folds}-FOLD CROSS-VALIDATION")
    logger.info(f"{'='*80}\n")

    kfold = KFold(n_splits=n_folds, shuffle=True, random_state=42)

    cv_results = {
        'motor_r2_scores': [],
        'cognitive_auc_scores': [],
        'fold_histories': []
    }

    for fold, (train_idx, val_idx) in enumerate(kfold.split(features)):
        logger.info(f"\n{'='*60}")
        logger.info(f"FOLD {fold + 1}/{n_folds}")
        logger.info(f"{'='*60}")

        # Split data
        X_train, X_val = features[train_idx], features[val_idx]
        motor_train, motor_val = motor_targets[train_idx], motor_targets[val_idx]
        cognitive_train, cognitive_val = cognitive_targets[train_idx], cognitive_targets[val_idx]

        # Standardize features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)

        # Create separate graphs for train and validation sets
        train_edge_index, _ = create_patient_similarity_graph(X_train_scaled, k=6)
        val_edge_index, _ = create_patient_similarity_graph(X_val_scaled, k=6)

        # Convert to tensors
        X_train_t = torch.tensor(X_train_scaled, dtype=torch.float32)
        X_val_t = torch.tensor(X_val_scaled, dtype=torch.float32)
        motor_train_t = torch.tensor(motor_train, dtype=torch.float32)
        motor_val_t = torch.tensor(motor_val, dtype=torch.float32)
        cognitive_train_t = torch.tensor(cognitive_train, dtype=torch.long)
        cognitive_val_t = torch.tensor(cognitive_val, dtype=torch.long)

        # Create model and trainer
        model = create_giman_prognostic(input_dim=features.shape[1])
        loss_fn = MultiTaskLoss(weighting_strategy='fixed')
        trainer = GIMANTrainer(model, loss_fn, learning_rate=1e-3)

        # Train
        history = trainer.fit(
            X_train_t, motor_train_t, cognitive_train_t, train_edge_index,
            X_val_t, motor_val_t, cognitive_val_t, val_edge_index,
            num_epochs=num_epochs,
            verbose=True
        )

        # Get final metrics
        final_motor_r2 = history['val_motor_r2'][-1]
        final_cognitive_auc = history['val_cognitive_auc'][-1]

        cv_results['motor_r2_scores'].append(final_motor_r2)
        cv_results['cognitive_auc_scores'].append(final_cognitive_auc)
        cv_results['fold_histories'].append(history)

        logger.info(f"\nFold {fold + 1} Results:")
        logger.info(f"  Motor R2: {final_motor_r2:.4f}")
        logger.info(f"  Cognitive AUC: {final_cognitive_auc:.4f}")

        # Save fold model
        if output_dir:
            output_dir.mkdir(parents=True, exist_ok=True)
            model_path = output_dir / f'model_fold_{fold+1}.pth'
            trainer.save_model(model_path)

    # Summary statistics
    motor_r2_mean = np.mean(cv_results['motor_r2_scores'])
    motor_r2_std = np.std(cv_results['motor_r2_scores'])
    cognitive_auc_mean = np.mean(cv_results['cognitive_auc_scores'])
    cognitive_auc_std = np.std(cv_results['cognitive_auc_scores'])

    logger.info(f"\n{'='*80}")
    logger.info(f"CROSS-VALIDATION SUMMARY")
    logger.info(f"{'='*80}")
    logger.info(f"Motor R2: {motor_r2_mean:.4f} ± {motor_r2_std:.4f}")
    logger.info(f"Cognitive AUC: {cognitive_auc_mean:.4f} ± {cognitive_auc_std:.4f}")
    logger.info(f"{'='*80}\n")

    cv_results['summary'] = {
        'motor_r2_mean': motor_r2_mean,
        'motor_r2_std': motor_r2_std,
        'cognitive_auc_mean': cognitive_auc_mean,
        'cognitive_auc_std': cognitive_auc_std
    }

    return cv_results


def main():
    """Main training pipeline."""

    print("\n" + "="*80)
    print("RESEARCH PLAN PHASE 2, TASK 2.3: TRAINING PIPELINE")
    print("="*80 + "\n")

    # Paths
    base_dir = Path(__file__).parent.parent
    phase1_dir = base_dir / 'phase1'
    output_dir = Path(__file__).parent / 'training_output'
    output_dir.mkdir(exist_ok=True)

    # Load Phase 1 data
    print("Step 1: Loading Phase 1 Data")
    print("-" * 60)
    loader = Phase1DataLoader(phase1_dir)
    df = loader.load_data()
    X = loader.prepare_features()
    motor_y, cognitive_y = loader.prepare_targets()
    patient_ids = loader.get_patient_ids()

    # Run cross-validation (graphs created per fold)
    print("\nStep 2: Running 5-Fold Cross-Validation")
    print("-" * 60)
    print("Note: Patient similarity graphs will be created for each fold")
    print("Training with 100 epochs per fold (this will take ~10-15 minutes)")
    cv_results = run_cross_validation(
        features=X,
        motor_targets=motor_y,
        cognitive_targets=cognitive_y,
        n_folds=5,
        num_epochs=100,  # Full training
        output_dir=output_dir
    )

    # Save results
    print("\nStep 4: Saving Results")
    print("-" * 60)
    results_file = output_dir / f'cv_results_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'

    # Convert numpy types to Python types for JSON serialization
    cv_results_json = {
        'summary': cv_results['summary'],
        'motor_r2_scores': [float(x) for x in cv_results['motor_r2_scores']],
        'cognitive_auc_scores': [float(x) for x in cv_results['cognitive_auc_scores']],
    }

    with open(results_file, 'w') as f:
        json.dump(cv_results_json, f, indent=2)

    logger.info(f"Results saved to: {results_file}")

    print("\n" + "="*80)
    print("[OK] TASK 2.3 COMPLETE: Training pipeline successfully executed")
    print("="*80)
    print("\nNext steps:")
    print("  - Task 2.4: Implement comprehensive evaluation metrics")
    print("  - Task 2.5: Hyperparameter tuning")
    print("  - Full training run with optimized hyperparameters")
    print("="*80 + "\n")


if __name__ == '__main__':
    main()
