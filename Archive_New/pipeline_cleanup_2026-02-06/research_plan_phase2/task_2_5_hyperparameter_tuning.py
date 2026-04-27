"""
RESEARCH PLAN PHASE 2 - TASK 2.5: HYPERPARAMETER TUNING
========================================================

Systematically optimize hyperparameters for dual-task GIMAN model:
- Model architecture: hidden_dim, GAT layers, attention heads, dropout
- Training: learning rate, weight decay, batch processing
- Loss function: task weights, focal loss parameters
- Graph construction: k-nearest neighbors

Uses Optuna for Bayesian optimization with early stopping.
Evaluates on both motor (R2) and cognitive (AUC-ROC) tasks.

Author: Research Plan Phase 2 Implementation
Date: October 2025
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from pathlib import Path
import json
from datetime import datetime
from typing import Dict, List, Tuple
import optuna
from optuna.visualization import plot_optimization_history, plot_param_importances
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, roc_auc_score

# Import Task 2.1-2.4 components
import sys
sys.path.append(str(Path(__file__).parent))

from task_2_1_giman_prognostic_model import GIMANPrognostic
from task_2_2_multitask_loss import MultiTaskLoss
from task_2_3_training_pipeline import Phase1DataLoader, create_patient_similarity_graph


class HyperparameterOptimizer:
    """Bayesian hyperparameter optimization using Optuna"""

    def __init__(self, features, motor_targets, cognitive_targets,
                 n_trials=50, n_folds=3, max_epochs=30):
        """
        Args:
            features: Patient feature matrix [n_patients, n_features]
            motor_targets: Motor slope targets [n_patients]
            cognitive_targets: Cognitive decline targets [n_patients]
            n_trials: Number of optimization trials
            n_folds: Number of CV folds for each trial
            max_epochs: Max epochs per trial (reduced for efficiency)
        """
        self.features = features
        self.motor_targets = motor_targets
        self.cognitive_targets = cognitive_targets
        self.n_trials = n_trials
        self.n_folds = n_folds
        self.max_epochs = max_epochs

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.best_params = None
        self.best_score = -np.inf
        self.optimization_history = []

    def objective(self, trial: optuna.Trial) -> float:
        """Optuna objective function - returns validation score to maximize"""

        # Sample hyperparameters
        params = {
            # Model architecture
            'hidden_dim': trial.suggest_categorical('hidden_dim', [64, 128, 256]),
            'num_gat_layers': trial.suggest_int('num_gat_layers', 2, 4),
            'num_attention_heads': trial.suggest_categorical('num_attention_heads', [2, 4, 8]),
            'dropout': trial.suggest_float('dropout', 0.1, 0.5),

            # Training
            'learning_rate': trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True),
            'weight_decay': trial.suggest_float('weight_decay', 1e-5, 1e-3, log=True),

            # Loss function
            'motor_weight': trial.suggest_float('motor_weight', 0.5, 0.9),
            'focal_alpha': trial.suggest_float('focal_alpha', 0.1, 0.5),
            'focal_gamma': trial.suggest_float('focal_gamma', 1.0, 3.0),

            # Graph construction
            'k_neighbors': trial.suggest_int('k_neighbors', 4, 10)
        }

        # Run cross-validation with these parameters
        cv_scores = self._cross_validate(params)

        # Combined metric: 0.7 * motor_R2 + 0.3 * cognitive_AUC
        # (scale AUC to similar range as R2)
        combined_score = 0.7 * cv_scores['motor_r2'] + 0.3 * (cv_scores['cognitive_auc'] - 0.5) * 2

        # Store for analysis
        self.optimization_history.append({
            'trial': trial.number,
            'params': params,
            'motor_r2': cv_scores['motor_r2'],
            'cognitive_auc': cv_scores['cognitive_auc'],
            'combined_score': combined_score
        })

        return combined_score

    def _cross_validate(self, params: Dict) -> Dict[str, float]:
        """Run k-fold CV with given hyperparameters"""

        kfold = KFold(n_splits=self.n_folds, shuffle=True, random_state=42)
        motor_r2_scores = []
        cognitive_auc_scores = []

        for fold, (train_idx, val_idx) in enumerate(kfold.split(self.features)):
            # Split data
            X_train = self.features[train_idx]
            X_val = self.features[val_idx]
            y_motor_train = self.motor_targets[train_idx]
            y_motor_val = self.motor_targets[val_idx]
            y_cog_train = self.cognitive_targets[train_idx]
            y_cog_val = self.cognitive_targets[val_idx]

            # Standardize features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_val_scaled = scaler.transform(X_val)

            # Create patient similarity graphs
            train_edge_index, train_edge_weights = create_patient_similarity_graph(
                X_train_scaled, k=params['k_neighbors']
            )
            val_edge_index, val_edge_weights = create_patient_similarity_graph(
                X_val_scaled, k=params['k_neighbors']
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

            # Initialize model
            model = GIMANPrognostic(
                input_dim=self.features.shape[1],
                hidden_dim=params['hidden_dim'],
                num_gat_layers=params['num_gat_layers'],
                num_attention_heads=params['num_attention_heads'],
                dropout=params['dropout']
            ).to(self.device)

            # Initialize loss and optimizer
            criterion = MultiTaskLoss(
                motor_weight=params['motor_weight'],
                cognitive_weight=1.0 - params['motor_weight'],
                weighting_strategy='fixed',
                focal_alpha=params['focal_alpha'],
                focal_gamma=params['focal_gamma']
            )

            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=params['learning_rate'],
                weight_decay=params['weight_decay']
            )

            # Training loop (reduced epochs for efficiency)
            best_val_loss = np.inf
            patience_counter = 0
            patience = 5

            for epoch in range(self.max_epochs):
                # Training
                model.train()
                optimizer.zero_grad()
                motor_pred, cognitive_logits = model(X_train_t, train_edge_index_t)
                loss, _ = criterion(motor_pred, y_motor_train_t, cognitive_logits, y_cog_train_t)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

                # Validation
                model.eval()
                with torch.no_grad():
                    motor_pred_val, cognitive_logits_val = model(X_val_t, val_edge_index_t)
                    val_loss, _ = criterion(motor_pred_val, y_motor_val_t,
                                           cognitive_logits_val, y_cog_val_t)

                # Early stopping
                if val_loss.item() < best_val_loss:
                    best_val_loss = val_loss.item()
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        break

            # Final evaluation
            model.eval()
            with torch.no_grad():
                motor_pred_val, cognitive_logits_val = model(X_val_t, val_edge_index_t)

                # Motor R2
                motor_pred_np = motor_pred_val.cpu().numpy().flatten()
                motor_r2 = r2_score(y_motor_val, motor_pred_np)
                motor_r2_scores.append(motor_r2)

                # Cognitive AUC
                cognitive_probs = torch.softmax(cognitive_logits_val, dim=1)[:, 1].cpu().numpy()
                cognitive_auc = roc_auc_score(y_cog_val, cognitive_probs)
                cognitive_auc_scores.append(cognitive_auc)

        return {
            'motor_r2': np.mean(motor_r2_scores),
            'cognitive_auc': np.mean(cognitive_auc_scores)
        }

    def optimize(self, output_dir: Path = None) -> Dict:
        """Run hyperparameter optimization"""

        print("\n" + "="*80)
        print("HYPERPARAMETER OPTIMIZATION")
        print("="*80)
        print(f"Trials: {self.n_trials}")
        print(f"CV Folds: {self.n_folds}")
        print(f"Max Epochs/Trial: {self.max_epochs}")
        print(f"Device: {self.device}")
        print("="*80 + "\n")

        # Create Optuna study
        study = optuna.create_study(
            direction='maximize',
            sampler=optuna.samplers.TPESampler(seed=42)
        )

        # Run optimization
        study.optimize(self.objective, n_trials=self.n_trials, show_progress_bar=True)

        # Store results
        self.best_params = study.best_params
        self.best_score = study.best_value

        # Print results
        print("\n" + "="*80)
        print("OPTIMIZATION COMPLETE")
        print("="*80)
        print(f"\nBest Combined Score: {self.best_score:.4f}")
        print("\nBest Hyperparameters:")
        for param, value in self.best_params.items():
            print(f"  {param}: {value}")

        # Find best trial metrics
        best_trial_idx = np.argmax([h['combined_score'] for h in self.optimization_history])
        best_trial = self.optimization_history[best_trial_idx]
        print(f"\nBest Trial Performance:")
        print(f"  Motor R2: {best_trial['motor_r2']:.4f}")
        print(f"  Cognitive AUC: {best_trial['cognitive_auc']:.4f}")
        print("="*80 + "\n")

        # Save results if output directory provided
        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)

            # Save best parameters
            with open(output_dir / 'best_hyperparameters.json', 'w') as f:
                json.dump(self.best_params, f, indent=2)

            # Save optimization history
            history_df = pd.DataFrame(self.optimization_history)
            history_df.to_csv(output_dir / 'optimization_history.csv', index=False)

            # Create visualizations
            self._create_visualizations(study, output_dir)

            # Create detailed report
            self._create_report(study, output_dir)

        return {
            'best_params': self.best_params,
            'best_score': self.best_score,
            'study': study,
            'history': self.optimization_history
        }

    def _create_visualizations(self, study: optuna.Study, output_dir: Path):
        """Create optimization visualizations"""

        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        # 1. Optimization history
        trials = [t.number for t in study.trials]
        values = [t.value for t in study.trials]
        best_values = np.maximum.accumulate(values)

        axes[0, 0].plot(trials, values, 'o-', alpha=0.6, label='Trial Score')
        axes[0, 0].plot(trials, best_values, 'r-', linewidth=2, label='Best Score')
        axes[0, 0].set_xlabel('Trial')
        axes[0, 0].set_ylabel('Combined Score')
        axes[0, 0].set_title('Optimization Progress')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # 2. Parameter importance
        if len(study.trials) >= 10:
            importances = optuna.importance.get_param_importances(study)
            params = list(importances.keys())
            values = list(importances.values())

            axes[0, 1].barh(params, values)
            axes[0, 1].set_xlabel('Importance')
            axes[0, 1].set_title('Hyperparameter Importance')
            axes[0, 1].grid(True, alpha=0.3, axis='x')

        # 3. Motor R2 vs Cognitive AUC tradeoff
        history_df = pd.DataFrame(self.optimization_history)
        scatter = axes[1, 0].scatter(
            history_df['motor_r2'],
            history_df['cognitive_auc'],
            c=history_df['combined_score'],
            cmap='viridis',
            s=100,
            alpha=0.6
        )
        axes[1, 0].set_xlabel('Motor R2')
        axes[1, 0].set_ylabel('Cognitive AUC')
        axes[1, 0].set_title('Task Performance Tradeoff')
        plt.colorbar(scatter, ax=axes[1, 0], label='Combined Score')
        axes[1, 0].grid(True, alpha=0.3)

        # 4. Best parameters visualization
        best_params_df = pd.DataFrame([self.best_params]).T
        best_params_df.columns = ['Value']

        # Separate numeric and categorical
        numeric_params = {}
        categorical_params = {}
        for param, value in self.best_params.items():
            if isinstance(value, (int, float)):
                numeric_params[param] = value
            else:
                categorical_params[param] = value

        # Plot numeric parameters
        if numeric_params:
            params = list(numeric_params.keys())
            values = list(numeric_params.values())
            axes[1, 1].barh(params, values, color='steelblue')
            axes[1, 1].set_xlabel('Value')
            axes[1, 1].set_title('Best Hyperparameters (Numeric)')
            axes[1, 1].grid(True, alpha=0.3, axis='x')

            # Add value labels
            for i, (param, value) in enumerate(numeric_params.items()):
                axes[1, 1].text(value, i, f' {value:.4f}', va='center')

        plt.tight_layout()
        plt.savefig(output_dir / 'hyperparameter_optimization.png', dpi=300, bbox_inches='tight')
        plt.close()

        print(f"Saved visualization: {output_dir / 'hyperparameter_optimization.png'}")

    def _create_report(self, study: optuna.Study, output_dir: Path):
        """Create detailed text report"""

        report_path = output_dir / 'hyperparameter_tuning_report.txt'

        with open(report_path, 'w') as f:
            f.write("="*80 + "\n")
            f.write("HYPERPARAMETER OPTIMIZATION REPORT\n")
            f.write("="*80 + "\n\n")

            f.write(f"Optimization Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Total Trials: {len(study.trials)}\n")
            f.write(f"CV Folds: {self.n_folds}\n")
            f.write(f"Max Epochs/Trial: {self.max_epochs}\n\n")

            f.write("="*80 + "\n")
            f.write("BEST CONFIGURATION\n")
            f.write("="*80 + "\n\n")

            f.write(f"Combined Score: {self.best_score:.4f}\n\n")

            # Find best trial metrics
            best_trial_idx = np.argmax([h['combined_score'] for h in self.optimization_history])
            best_trial = self.optimization_history[best_trial_idx]

            f.write("Performance:\n")
            f.write(f"  Motor R2:      {best_trial['motor_r2']:.4f}\n")
            f.write(f"  Cognitive AUC: {best_trial['cognitive_auc']:.4f}\n\n")

            f.write("Hyperparameters:\n")
            f.write("\nModel Architecture:\n")
            f.write(f"  hidden_dim:           {self.best_params['hidden_dim']}\n")
            f.write(f"  num_gat_layers:       {self.best_params['num_gat_layers']}\n")
            f.write(f"  num_attention_heads:  {self.best_params['num_attention_heads']}\n")
            f.write(f"  dropout:              {self.best_params['dropout']:.4f}\n")

            f.write("\nTraining:\n")
            f.write(f"  learning_rate:        {self.best_params['learning_rate']:.6f}\n")
            f.write(f"  weight_decay:         {self.best_params['weight_decay']:.6f}\n")

            f.write("\nLoss Function:\n")
            f.write(f"  motor_weight:         {self.best_params['motor_weight']:.4f}\n")
            f.write(f"  focal_alpha:          {self.best_params['focal_alpha']:.4f}\n")
            f.write(f"  focal_gamma:          {self.best_params['focal_gamma']:.4f}\n")

            f.write("\nGraph Construction:\n")
            f.write(f"  k_neighbors:          {self.best_params['k_neighbors']}\n\n")

            f.write("="*80 + "\n")
            f.write("TOP 5 TRIALS\n")
            f.write("="*80 + "\n\n")

            # Sort by combined score
            sorted_history = sorted(self.optimization_history,
                                  key=lambda x: x['combined_score'],
                                  reverse=True)

            for i, trial in enumerate(sorted_history[:5], 1):
                f.write(f"\n{i}. Trial {trial['trial']}:\n")
                f.write(f"   Combined Score: {trial['combined_score']:.4f}\n")
                f.write(f"   Motor R2:       {trial['motor_r2']:.4f}\n")
                f.write(f"   Cognitive AUC:  {trial['cognitive_auc']:.4f}\n")
                f.write(f"   Key params: hidden_dim={trial['params']['hidden_dim']}, ")
                f.write(f"k={trial['params']['k_neighbors']}, ")
                f.write(f"lr={trial['params']['learning_rate']:.6f}\n")

            f.write("\n" + "="*80 + "\n")
            f.write("PARAMETER STATISTICS\n")
            f.write("="*80 + "\n\n")

            # Analyze parameter distributions
            history_df = pd.DataFrame([h['params'] for h in self.optimization_history])

            for param in history_df.columns:
                if history_df[param].dtype in [np.float64, np.int64]:
                    f.write(f"\n{param}:\n")
                    f.write(f"  Mean:   {history_df[param].mean():.4f}\n")
                    f.write(f"  Std:    {history_df[param].std():.4f}\n")
                    f.write(f"  Min:    {history_df[param].min():.4f}\n")
                    f.write(f"  Max:    {history_df[param].max():.4f}\n")
                    f.write(f"  Median: {history_df[param].median():.4f}\n")

            f.write("\n" + "="*80 + "\n")
            f.write("RECOMMENDATIONS\n")
            f.write("="*80 + "\n\n")

            # Calculate improvement over baseline
            baseline_r2 = 0.0132  # From Task 2.3 results
            baseline_auc = 0.6218
            improvement_r2 = best_trial['motor_r2'] - baseline_r2
            improvement_auc = best_trial['cognitive_auc'] - baseline_auc

            f.write(f"Improvement over Task 2.3 baseline:\n")
            f.write(f"  Motor R2:      {improvement_r2:+.4f} ({improvement_r2/baseline_r2*100:+.1f}%)\n")
            f.write(f"  Cognitive AUC: {improvement_auc:+.4f} ({improvement_auc/baseline_auc*100:+.1f}%)\n\n")

            f.write("Next Steps:\n")
            f.write("  1. Run full 100-epoch training with best hyperparameters\n")
            f.write("  2. Validate on held-out test set\n")
            f.write("  3. Proceed to Research Plan Phase 3 (Multimodal Integration)\n\n")

            f.write("Expected Performance with Full Training:\n")
            f.write(f"  Motor R2:      {best_trial['motor_r2'] * 1.1:.4f} - {best_trial['motor_r2'] * 1.2:.4f}\n")
            f.write(f"  Cognitive AUC: {best_trial['cognitive_auc'] * 1.02:.4f} - {best_trial['cognitive_auc'] * 1.05:.4f}\n")

        print(f"Saved report: {report_path}")


def run_full_validation(best_params: Dict, features, motor_targets, cognitive_targets,
                        output_dir: Path, num_epochs: int = 100):
    """Run full 100-epoch training with best hyperparameters"""

    print("\n" + "="*80)
    print("FULL VALIDATION WITH BEST HYPERPARAMETERS")
    print("="*80)
    print(f"Epochs: {num_epochs}")
    print("="*80 + "\n")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    kfold = KFold(n_splits=5, shuffle=True, random_state=42)

    all_results = []

    for fold, (train_idx, val_idx) in enumerate(kfold.split(features), 1):
        print(f"\nFold {fold}/5:")

        # Split and scale data
        X_train = features[train_idx]
        X_val = features[val_idx]
        y_motor_train = motor_targets[train_idx]
        y_motor_val = motor_targets[val_idx]
        y_cog_train = cognitive_targets[train_idx]
        y_cog_val = cognitive_targets[val_idx]

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)

        # Create graphs
        train_edge_index, _ = create_patient_similarity_graph(
            X_train_scaled, k=best_params['k_neighbors']
        )
        val_edge_index, _ = create_patient_similarity_graph(
            X_val_scaled, k=best_params['k_neighbors']
        )

        # Convert to tensors
        X_train_t = torch.FloatTensor(X_train_scaled).to(device)
        X_val_t = torch.FloatTensor(X_val_scaled).to(device)
        y_motor_train_t = torch.FloatTensor(y_motor_train).unsqueeze(1).to(device)
        y_motor_val_t = torch.FloatTensor(y_motor_val).unsqueeze(1).to(device)
        y_cog_train_t = torch.LongTensor(y_cog_train).to(device)
        y_cog_val_t = torch.LongTensor(y_cog_val).to(device)
        train_edge_index_t = torch.LongTensor(train_edge_index).to(device)
        val_edge_index_t = torch.LongTensor(val_edge_index).to(device)

        # Initialize model with best params
        model = GIMANPrognostic(
            input_dim=features.shape[1],
            hidden_dim=best_params['hidden_dim'],
            num_gat_layers=best_params['num_gat_layers'],
            num_attention_heads=best_params['num_attention_heads'],
            dropout=best_params['dropout']
        ).to(device)

        criterion = MultiTaskLoss(
            motor_weight=best_params['motor_weight'],
            cognitive_weight=1.0 - best_params['motor_weight'],
            weighting_strategy='fixed',
            focal_alpha=best_params['focal_alpha'],
            focal_gamma=best_params['focal_gamma']
        )

        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=best_params['learning_rate'],
            weight_decay=best_params['weight_decay']
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
                patience_counter = 0
                # Save best model
                best_model_state = model.state_dict()
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"  Early stopping at epoch {epoch+1}")
                    break

        # Load best model and evaluate
        model.load_state_dict(best_model_state)
        model.eval()
        with torch.no_grad():
            motor_pred_val, cognitive_logits_val = model(X_val_t, val_edge_index_t)

            motor_pred_np = motor_pred_val.cpu().numpy().flatten()
            motor_r2 = r2_score(y_motor_val, motor_pred_np)

            cognitive_probs = torch.softmax(cognitive_logits_val, dim=1)[:, 1].cpu().numpy()
            cognitive_auc = roc_auc_score(y_cog_val, cognitive_probs)

        print(f"  Motor R2: {motor_r2:.4f}, Cognitive AUC: {cognitive_auc:.4f}")

        all_results.append({
            'fold': fold,
            'motor_r2': motor_r2,
            'cognitive_auc': cognitive_auc,
            'epochs_trained': epoch + 1
        })

    # Calculate summary statistics
    motor_r2_mean = np.mean([r['motor_r2'] for r in all_results])
    motor_r2_std = np.std([r['motor_r2'] for r in all_results])
    cognitive_auc_mean = np.mean([r['cognitive_auc'] for r in all_results])
    cognitive_auc_std = np.std([r['cognitive_auc'] for r in all_results])

    print("\n" + "="*80)
    print("FULL VALIDATION RESULTS")
    print("="*80)
    print(f"\nMotor R2:      {motor_r2_mean:.4f} +/- {motor_r2_std:.4f}")
    print(f"Cognitive AUC: {cognitive_auc_mean:.4f} +/- {cognitive_auc_std:.4f}")
    print("="*80 + "\n")

    # Save results
    results_df = pd.DataFrame(all_results)
    results_df.to_csv(output_dir / 'full_validation_results.csv', index=False)

    # Save summary
    summary = {
        'motor_r2_mean': float(motor_r2_mean),
        'motor_r2_std': float(motor_r2_std),
        'cognitive_auc_mean': float(cognitive_auc_mean),
        'cognitive_auc_std': float(cognitive_auc_std),
        'best_params': best_params,
        'timestamp': datetime.now().isoformat()
    }

    with open(output_dir / 'full_validation_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)

    return summary


if __name__ == "__main__":
    # Set up paths
    base_dir = Path(__file__).parent
    output_dir = base_dir / "tuning_output"
    output_dir.mkdir(exist_ok=True)

    # Phase 1 directory (same level as research_plan_phase2)
    phase1_dir = base_dir.parent / "phase1"

    # Load Phase 1 data
    print("Loading Phase 1 prognostic dataset...")
    loader = Phase1DataLoader(phase1_dir=phase1_dir)
    data_df = loader.load_data()
    features = loader.prepare_features()
    motor_targets, cognitive_targets = loader.prepare_targets()

    print(f"Dataset: {len(features)} patients, {features.shape[1]} features")
    print(f"Motor targets: {motor_targets.mean():.2f} +/- {motor_targets.std():.2f} pts/year")
    print(f"Cognitive decline rate: {cognitive_targets.mean()*100:.1f}%")

    # Run hyperparameter optimization
    optimizer = HyperparameterOptimizer(
        features=features,
        motor_targets=motor_targets,
        cognitive_targets=cognitive_targets,
        n_trials=30,  # 30 trials for reasonable exploration
        n_folds=3,    # 3-fold CV for speed
        max_epochs=30 # 30 epochs per trial
    )

    results = optimizer.optimize(output_dir=output_dir)

    # Run full validation with best params
    print("\n" + "="*80)
    print("PHASE 2.5 COMPLETE: Running Full Validation")
    print("="*80)

    validation_summary = run_full_validation(
        best_params=results['best_params'],
        features=features,
        motor_targets=motor_targets,
        cognitive_targets=cognitive_targets,
        output_dir=output_dir,
        num_epochs=100
    )

    print("\n" + "="*80)
    print("TASK 2.5 COMPLETE")
    print("="*80)
    print("\nOutput files:")
    print(f"  - {output_dir / 'best_hyperparameters.json'}")
    print(f"  - {output_dir / 'optimization_history.csv'}")
    print(f"  - {output_dir / 'hyperparameter_optimization.png'}")
    print(f"  - {output_dir / 'hyperparameter_tuning_report.txt'}")
    print(f"  - {output_dir / 'full_validation_results.csv'}")
    print(f"  - {output_dir / 'full_validation_summary.json'}")
    print("\nNext: Research Plan Phase 3 (Multimodal Feature Integration)")
    print("="*80)
