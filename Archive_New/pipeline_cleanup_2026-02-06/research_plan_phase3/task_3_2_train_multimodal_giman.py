"""
RESEARCH PLAN PHASE 3 - TASK 3.2: TRAIN MULTIMODAL GIMAN
==========================================================

Trains GIMANPrognostic model with multimodal features:
- Clinical baseline: 5 features (UPDRS, MoCA, demographics)
- DAT-SPECT SBR: 20 features (striatal binding ratios)
- Grey matter volume: 1 feature
- Prognostic targets: Motor slopes + cognitive decline

Uses optimized hyperparameters from Phase 2 Task 2.5.
Implements smart imputation for missing imaging data.

Expected improvement: Motor R² from 0.0346 -> 0.10-0.15

Author: Research Plan Phase 3 Implementation
Date: October 2025
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from pathlib import Path
import json
from datetime import datetime
from typing import Dict, Tuple
import logging
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import r2_score, roc_auc_score, mean_absolute_error

# Import Phase 2 components
import sys
sys.path.append(str(Path(__file__).parent.parent / "research_plan_phase2"))

from task_2_1_giman_prognostic_model import GIMANPrognostic
from task_2_2_multitask_loss import MultiTaskLoss
from task_2_3_training_pipeline import create_patient_similarity_graph

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MultimodalDataLoader:
    """Load and prepare multimodal dataset for training"""

    def __init__(self, multimodal_dataset_path: Path):
        """
        Initialize data loader.

        Args:
            multimodal_dataset_path: Path to multimodal CSV from Task 3.1
        """
        self.dataset_path = Path(multimodal_dataset_path)
        self.df = None
        self.features = None
        self.motor_targets = None
        self.cognitive_targets = None
        self.feature_names = None

        logger.info(f"MultimodalDataLoader initialized with: {multimodal_dataset_path}")

    def load_data(self) -> pd.DataFrame:
        """Load multimodal dataset"""
        self.df = pd.read_csv(self.dataset_path)
        logger.info(f"Loaded {len(self.df)} patients with {self.df.shape[1]} columns")
        return self.df

    def prepare_multimodal_features(self, imputation_strategy: str = 'median') -> np.ndarray:
        """
        Prepare multimodal feature matrix with intelligent imputation.

        Args:
            imputation_strategy: 'median', 'mean', or 'indicator'

        Returns:
            Feature matrix [n_patients, n_features]
        """
        logger.info("\n" + "="*80)
        logger.info("PREPARING MULTIMODAL FEATURES")
        logger.info("="*80)

        # Define feature groups
        clinical_features = [
            'UPDRS_III_BL', 'MOCA_BL', 'AGE_APPROX', 'SEX', 'HANDED'
        ]

        # DAT-SPECT SBR features
        sbr_features = [col for col in self.df.columns
                       if any(x in col.upper() for x in
                              ['STRIATUM', 'CAUDATE', 'PUTAMEN', 'ASYMMETRY', '_RATIO'])]

        # Grey matter features
        gm_features = ['GM_VOLUME'] if 'GM_VOLUME' in self.df.columns else []

        # Combine all features
        all_features = clinical_features + sbr_features + gm_features

        # Check availability
        available_features = [f for f in all_features if f in self.df.columns]
        missing_features = [f for f in all_features if f not in self.df.columns]

        if missing_features:
            logger.warning(f"Missing {len(missing_features)} features: {missing_features[:5]}")

        logger.info(f"\nFeature groups:")
        logger.info(f"  Clinical: {len([f for f in clinical_features if f in available_features])}")
        logger.info(f"  DAT-SPECT SBR: {len(sbr_features)}")
        logger.info(f"  Grey Matter: {len(gm_features)}")
        logger.info(f"  Total: {len(available_features)} features")

        # Extract feature matrix
        X = self.df[available_features].values

        # Analyze missingness
        missing_pct = np.isnan(X).mean(axis=0) * 100
        logger.info(f"\nMissingness analysis:")
        logger.info(f"  Clinical features: {missing_pct[:len(clinical_features)].mean():.1f}% missing")
        if sbr_features:
            sbr_start = len(clinical_features)
            sbr_end = sbr_start + len(sbr_features)
            logger.info(f"  DAT-SPECT features: {missing_pct[sbr_start:sbr_end].mean():.1f}% missing")
        if gm_features:
            logger.info(f"  Grey matter: {missing_pct[-len(gm_features):].mean():.1f}% missing")

        # Smart imputation strategy
        if imputation_strategy == 'indicator':
            # Add missingness indicators for imaging features
            logger.info(f"\nUsing 'indicator' imputation strategy")

            # Impute with median
            imputer = SimpleImputer(strategy='median')
            X_imputed = imputer.fit_transform(X)

            # Add binary indicators for imaging missingness
            imaging_start = len(clinical_features)
            imaging_features = X[:, imaging_start:]
            missing_indicators = np.isnan(imaging_features).astype(float)

            # Combine: original features + missingness indicators
            X = np.hstack([X_imputed, missing_indicators])

            logger.info(f"Added {missing_indicators.shape[1]} missingness indicators")
            logger.info(f"Final feature count: {X.shape[1]}")

        else:
            # Simple imputation
            logger.info(f"\nUsing '{imputation_strategy}' imputation strategy")
            imputer = SimpleImputer(strategy=imputation_strategy)
            X = imputer.fit_transform(X)

        self.features = X
        self.feature_names = available_features

        logger.info(f"\nFeature matrix shape: {X.shape}")
        logger.info(f"No missing values: {not np.isnan(X).any()}")

        return X

    def prepare_targets(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract prognostic targets.

        Returns:
            (motor_targets, cognitive_targets)
        """
        logger.info("\nPreparing prognostic targets...")

        # Motor progression target
        motor_col = 'motor_slope_per_year'
        if motor_col in self.df.columns:
            motor_targets = self.df[motor_col].values
        else:
            raise ValueError(f"Motor target '{motor_col}' not found")

        # Cognitive decline target
        cog_col = 'cognitive_decline'
        if cog_col in self.df.columns:
            cognitive_targets = self.df[cog_col].values
        else:
            raise ValueError(f"Cognitive target '{cog_col}' not found")

        # Remove NaNs
        valid_mask = ~(np.isnan(motor_targets) | np.isnan(cognitive_targets))
        n_invalid = (~valid_mask).sum()

        if n_invalid > 0:
            logger.warning(f"Removing {n_invalid} patients with missing targets")
            motor_targets = motor_targets[valid_mask]
            cognitive_targets = cognitive_targets[valid_mask]
            self.features = self.features[valid_mask]

        self.motor_targets = motor_targets
        self.cognitive_targets = cognitive_targets.astype(int)

        logger.info(f"Motor targets: {len(motor_targets)} patients")
        logger.info(f"  Mean: {motor_targets.mean():.2f} +/- {motor_targets.std():.2f} pts/year")
        logger.info(f"Cognitive targets: {len(cognitive_targets)} patients")
        logger.info(f"  Decline rate: {cognitive_targets.mean()*100:.1f}%")

        return motor_targets, cognitive_targets


class MultimodalGIMANTrainer:
    """Train GIMANPrognostic with multimodal features"""

    def __init__(self, best_hyperparams: Dict, device: str = 'cpu'):
        """
        Initialize trainer with optimized hyperparameters from Phase 2.

        Args:
            best_hyperparams: Hyperparameters from Phase 2 Task 2.5
            device: 'cpu' or 'cuda'
        """
        self.hyperparams = best_hyperparams
        self.device = torch.device(device)

        logger.info("\nMultimodalGIMANTrainer initialized")
        logger.info(f"Device: {self.device}")
        logger.info(f"Hyperparameters:")
        for key, value in best_hyperparams.items():
            logger.info(f"  {key}: {value}")

    def train_cross_validation(self, features: np.ndarray,
                              motor_targets: np.ndarray,
                              cognitive_targets: np.ndarray,
                              n_folds: int = 5,
                              num_epochs: int = 100) -> Dict:
        """
        Run 5-fold cross-validation training.

        Args:
            features: Feature matrix [n_patients, n_features]
            motor_targets: Motor slope targets
            cognitive_targets: Cognitive decline labels
            n_folds: Number of CV folds
            num_epochs: Training epochs per fold

        Returns:
            Dictionary with CV results
        """
        logger.info("\n" + "="*80)
        logger.info("MULTIMODAL GIMAN TRAINING - 5-FOLD CROSS-VALIDATION")
        logger.info("="*80)
        logger.info(f"Dataset: {len(features)} patients, {features.shape[1]} features")
        logger.info(f"Epochs: {num_epochs}, Folds: {n_folds}")
        logger.info("="*80 + "\n")

        kfold = KFold(n_splits=n_folds, shuffle=True, random_state=42)
        results = []

        for fold, (train_idx, val_idx) in enumerate(kfold.split(features), 1):
            logger.info(f"Fold {fold}/{n_folds}:")

            # Split data
            X_train = features[train_idx]
            X_val = features[val_idx]
            y_motor_train = motor_targets[train_idx]
            y_motor_val = motor_targets[val_idx]
            y_cog_train = cognitive_targets[train_idx]
            y_cog_val = cognitive_targets[val_idx]

            # Standardize features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_val_scaled = scaler.transform(X_val)

            # Create patient similarity graphs
            train_edge_index, _ = create_patient_similarity_graph(
                X_train_scaled, k=self.hyperparams['k_neighbors']
            )
            val_edge_index, _ = create_patient_similarity_graph(
                X_val_scaled, k=self.hyperparams['k_neighbors']
            )

            # Convert to tensors
            X_train_t = torch.FloatTensor(X_train_scaled).to(self.device)
            X_val_t = torch.FloatTensor(X_val_scaled).to(self.device)
            y_motor_train_t = torch.FloatTensor(y_motor_train).unsqueeze(1).to(self.device)
            y_motor_val_t = torch.FloatTensor(y_motor_val).unsqueeze(1).to(self.device)
            y_cog_train_t = torch.LongTensor(y_cog_train).to(self.device)
            y_cog_val_t = torch.LongTensor(y_cog_val).to(self.device)
            train_edge_index_t = torch.LongTensor(train_edge_index).to(self.device)
            val_edge_index_t = torch.LongTensor(val_edge_index).to(self.device)

            # Initialize model with multimodal input dimension
            model = GIMANPrognostic(
                input_dim=features.shape[1],  # Multimodal feature count
                hidden_dim=self.hyperparams['hidden_dim'],
                num_gat_layers=self.hyperparams['num_gat_layers'],
                num_attention_heads=self.hyperparams['num_attention_heads'],
                dropout=self.hyperparams['dropout']
            ).to(self.device)

            # Loss and optimizer
            criterion = MultiTaskLoss(
                motor_weight=self.hyperparams['motor_weight'],
                cognitive_weight=1.0 - self.hyperparams['motor_weight'],
                weighting_strategy='fixed',
                focal_alpha=self.hyperparams['focal_alpha'],
                focal_gamma=self.hyperparams['focal_gamma']
            )

            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=self.hyperparams['learning_rate'],
                weight_decay=self.hyperparams['weight_decay']
            )

            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='min', factor=0.5, patience=10, verbose=False
            )

            # Training loop
            best_val_loss = np.inf
            patience_counter = 0
            patience = 15

            for epoch in range(num_epochs):
                # Train
                model.train()
                optimizer.zero_grad()
                motor_pred, cognitive_logits = model(X_train_t, train_edge_index_t)
                loss, _ = criterion(motor_pred, y_motor_train_t, cognitive_logits, y_cog_train_t)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

                # Validate
                model.eval()
                with torch.no_grad():
                    motor_pred_val, cognitive_logits_val = model(X_val_t, val_edge_index_t)
                    val_loss, _ = criterion(motor_pred_val, y_motor_val_t,
                                           cognitive_logits_val, y_cog_val_t)

                scheduler.step(val_loss)

                # Early stopping
                if val_loss.item() < best_val_loss:
                    best_val_loss = val_loss.item()
                    best_model_state = model.state_dict()
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        logger.info(f"  Early stopping at epoch {epoch+1}")
                        break

            # Load best model and evaluate
            model.load_state_dict(best_model_state)
            model.eval()
            with torch.no_grad():
                motor_pred_val, cognitive_logits_val = model(X_val_t, val_edge_index_t)

                # Motor metrics
                motor_pred_np = motor_pred_val.cpu().numpy().flatten()
                motor_r2 = r2_score(y_motor_val, motor_pred_np)
                motor_mae = mean_absolute_error(y_motor_val, motor_pred_np)

                # Cognitive metrics
                cognitive_probs = torch.softmax(cognitive_logits_val, dim=1)[:, 1].cpu().numpy()
                cognitive_auc = roc_auc_score(y_cog_val, cognitive_probs)

            logger.info(f"  Motor R2: {motor_r2:.4f}, MAE: {motor_mae:.2f}")
            logger.info(f"  Cognitive AUC: {cognitive_auc:.4f}")

            results.append({
                'fold': fold,
                'motor_r2': motor_r2,
                'motor_mae': motor_mae,
                'cognitive_auc': cognitive_auc,
                'epochs_trained': epoch + 1
            })

        # Summary statistics
        motor_r2_mean = np.mean([r['motor_r2'] for r in results])
        motor_r2_std = np.std([r['motor_r2'] for r in results])
        motor_mae_mean = np.mean([r['motor_mae'] for r in results])
        cognitive_auc_mean = np.mean([r['cognitive_auc'] for r in results])
        cognitive_auc_std = np.std([r['cognitive_auc'] for r in results])

        logger.info("\n" + "="*80)
        logger.info("MULTIMODAL GIMAN - FINAL RESULTS")
        logger.info("="*80)
        logger.info(f"\nMotor Progression:")
        logger.info(f"  R2:  {motor_r2_mean:.4f} +/- {motor_r2_std:.4f}")
        logger.info(f"  MAE: {motor_mae_mean:.2f} pts/year")
        logger.info(f"\nCognitive Decline:")
        logger.info(f"  AUC: {cognitive_auc_mean:.4f} +/- {cognitive_auc_std:.4f}")
        logger.info("="*80 + "\n")

        return {
            'motor_r2_mean': motor_r2_mean,
            'motor_r2_std': motor_r2_std,
            'motor_mae_mean': motor_mae_mean,
            'cognitive_auc_mean': cognitive_auc_mean,
            'cognitive_auc_std': cognitive_auc_std,
            'fold_results': results
        }


def compare_with_phase2_baseline(multimodal_results: Dict,
                                 phase2_results: Dict,
                                 output_dir: Path):
    """
    Compare multimodal results with Phase 2 baseline.

    Args:
        multimodal_results: Results from multimodal GIMAN
        phase2_results: Baseline results from Phase 2
        output_dir: Output directory
    """
    logger.info("\n" + "="*80)
    logger.info("PERFORMANCE COMPARISON: PHASE 3 vs PHASE 2 BASELINE")
    logger.info("="*80)

    # Calculate improvements
    motor_r2_improvement = multimodal_results['motor_r2_mean'] - phase2_results['motor_r2']
    motor_r2_pct = (motor_r2_improvement / abs(phase2_results['motor_r2'])) * 100

    cognitive_auc_improvement = multimodal_results['cognitive_auc_mean'] - phase2_results['cognitive_auc']
    cognitive_auc_pct = (cognitive_auc_improvement / phase2_results['cognitive_auc']) * 100

    logger.info(f"\nMotor R2:")
    logger.info(f"  Phase 2 (7 features):    {phase2_results['motor_r2']:.4f}")
    logger.info(f"  Phase 3 (multimodal):    {multimodal_results['motor_r2_mean']:.4f}")
    logger.info(f"  Improvement:             {motor_r2_improvement:+.4f} ({motor_r2_pct:+.1f}%)")

    logger.info(f"\nCognitive AUC:")
    logger.info(f"  Phase 2 (7 features):    {phase2_results['cognitive_auc']:.4f}")
    logger.info(f"  Phase 3 (multimodal):    {multimodal_results['cognitive_auc_mean']:.4f}")
    logger.info(f"  Improvement:             {cognitive_auc_improvement:+.4f} ({cognitive_auc_pct:+.1f}%)")

    logger.info("\n" + "="*80)

    # Save comparison report
    report_path = output_dir / "phase3_vs_phase2_comparison.txt"
    with open(report_path, 'w') as f:
        f.write("PHASE 3 MULTIMODAL vs PHASE 2 BASELINE COMPARISON\n")
        f.write("="*80 + "\n\n")
        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        f.write("MOTOR PROGRESSION (R2 Score):\n")
        f.write(f"  Phase 2 Baseline (7 features):   {phase2_results['motor_r2']:.4f}\n")
        f.write(f"  Phase 3 Multimodal (~62 features): {multimodal_results['motor_r2_mean']:.4f} +/- {multimodal_results['motor_r2_std']:.4f}\n")
        f.write(f"  Absolute Improvement:            {motor_r2_improvement:+.4f}\n")
        f.write(f"  Relative Improvement:            {motor_r2_pct:+.1f}%\n\n")

        f.write("COGNITIVE DECLINE (AUC-ROC):\n")
        f.write(f"  Phase 2 Baseline (7 features):   {phase2_results['cognitive_auc']:.4f}\n")
        f.write(f"  Phase 3 Multimodal (~62 features): {multimodal_results['cognitive_auc_mean']:.4f} +/- {multimodal_results['cognitive_auc_std']:.4f}\n")
        f.write(f"  Absolute Improvement:            {cognitive_auc_improvement:+.4f}\n")
        f.write(f"  Relative Improvement:            {cognitive_auc_pct:+.1f}%\n\n")

        f.write("FEATURE COMPARISON:\n")
        f.write(f"  Phase 2: 7 clinical features\n")
        f.write(f"  Phase 3: ~62 multimodal features\n")
        f.write(f"    - Clinical: 5 features\n")
        f.write(f"    - DAT-SPECT SBR: 20 features (41.7% coverage)\n")
        f.write(f"    - Grey matter: 1 feature (6.2% coverage)\n")
        f.write(f"    - Other: ~36 features\n\n")

        f.write("CONCLUSION:\n")
        if motor_r2_improvement > 0:
            f.write(f"  Multimodal integration IMPROVED motor prediction by {motor_r2_pct:.1f}%\n")
        else:
            f.write(f"  WARNING: Multimodal model did not improve motor prediction\n")

        if cognitive_auc_improvement > 0:
            f.write(f"  Multimodal integration IMPROVED cognitive prediction by {cognitive_auc_pct:.1f}%\n")
        else:
            f.write(f"  WARNING: Multimodal model did not improve cognitive prediction\n")

    logger.info(f"Saved comparison report: {report_path}")


if __name__ == "__main__":
    # Set up paths
    base_dir = Path(__file__).parent
    multimodal_output = base_dir / "multimodal_output"
    phase2_tuning = base_dir.parent / "research_plan_phase2" / "tuning_output"
    output_dir = base_dir / "training_output"
    output_dir.mkdir(exist_ok=True)

    # Load multimodal dataset (most recent)
    multimodal_files = list(multimodal_output.glob("multimodal_dataset_*.csv"))
    if not multimodal_files:
        raise FileNotFoundError(f"No multimodal dataset found in {multimodal_output}")

    multimodal_dataset = max(multimodal_files, key=lambda p: p.stat().st_mtime)
    logger.info(f"Multimodal dataset: {multimodal_dataset.name}")

    # Load Phase 2 best hyperparameters
    hyperparams_file = phase2_tuning / "best_hyperparameters.json"
    if not hyperparams_file.exists():
        raise FileNotFoundError(f"Hyperparameters not found: {hyperparams_file}")

    with open(hyperparams_file, 'r') as f:
        best_hyperparams = json.load(f)

    logger.info("Loaded Phase 2 optimized hyperparameters")

    # Load and prepare data
    loader = MultimodalDataLoader(multimodal_dataset)
    loader.load_data()
    features = loader.prepare_multimodal_features(imputation_strategy='median')
    motor_targets, cognitive_targets = loader.prepare_targets()

    # Train multimodal GIMAN
    trainer = MultimodalGIMANTrainer(best_hyperparams, device='cpu')
    results = trainer.train_cross_validation(
        features=features,
        motor_targets=motor_targets,
        cognitive_targets=cognitive_targets,
        n_folds=5,
        num_epochs=100
    )

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = output_dir / f"multimodal_giman_results_{timestamp}.json"

    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)

    logger.info(f"Saved results: {results_file}")

    # Compare with Phase 2 baseline
    phase2_baseline = {
        'motor_r2': 0.0346,
        'cognitive_auc': 0.6646
    }

    compare_with_phase2_baseline(results, phase2_baseline, output_dir)

    logger.info("\n" + "="*80)
    logger.info("TASK 3.2 COMPLETE")
    logger.info("="*80)
