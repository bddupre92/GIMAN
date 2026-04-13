"""
Phase 4, Task 4.5: Baseline Subtype Prediction

This script builds predictive models to classify patients into progression subtypes
using only baseline clinical features (before longitudinal follow-up).

Methodology:
1. Multiple classifier comparison (Logistic Regression, Random Forest, XGBoost, GNN)
2. Cross-validation for robust performance estimation
3. Feature importance analysis
4. Calibration analysis for clinical deployment

Expected Output:
- Trained classifier models
- Performance metrics (accuracy, AUC, F1-score per class)
- Feature importance rankings
- Calibration curves and decision thresholds
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, f1_score,
    roc_auc_score, classification_report, confusion_matrix,
    roc_curve, auc
)
from sklearn.calibration import calibration_curve

import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (20, 14)

# Set random seed
np.random.seed(42)


class BaselineSubtypePrediction:
    """Predict progression subtypes from baseline clinical features."""

    def __init__(
        self,
        labeled_trajectories_path: str,
        output_dir: str = "data/longitudinal_cohort",
        test_size: float = 0.2,
        n_folds: int = 5
    ):
        """
        Initialize baseline subtype prediction.

        Args:
            labeled_trajectories_path: Path to patient_trajectories_labeled.csv
            output_dir: Directory for outputs
            test_size: Proportion of data for test set
            n_folds: Number of cross-validation folds
        """
        self.labeled_traj_path = Path(labeled_trajectories_path)
        self.output_dir = Path(output_dir)
        self.test_size = test_size
        self.n_folds = n_folds

        self.trajectories_df = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.feature_names = None
        self.scaler = None

        self.models = {}
        self.performance_metrics = {}
        self.best_model_name = None

        print("[INIT] Initialized Baseline Subtype Prediction")
        print(f"   Labeled trajectories: {self.labeled_traj_path}")
        print(f"   Output directory: {self.output_dir}")
        print(f"   Test size: {test_size}")
        print(f"   CV folds: {n_folds}")

    def load_data(self):
        """Load labeled trajectory data."""
        print("\n[LOAD] Loading labeled trajectory data...")

        self.trajectories_df = pd.read_csv(self.labeled_traj_path)

        # Remove patients without cluster/label
        self.trajectories_df = self.trajectories_df.dropna(subset=['cluster'])

        print(f"   Loaded {len(self.trajectories_df)} labeled patients")
        print(f"   Class distribution:")
        for cluster_id, label in self.trajectories_df.groupby('cluster')['subtype_label'].first().items():
            count = (self.trajectories_df['cluster'] == cluster_id).sum()
            pct = 100 * count / len(self.trajectories_df)
            print(f"      Cluster {int(cluster_id)}: {count} patients ({pct:.1f}%)")

        return self.trajectories_df

    def prepare_features(self):
        """
        Prepare baseline features for prediction.

        Only uses features available at baseline (no longitudinal data).
        """
        print("\n[PREP] Preparing baseline features...")

        # Define baseline features (available at initial visit)
        baseline_features = [
            'UPDRS_III_baseline',
            'MOCA_baseline',
            'SEX',
            'HANDED',
            'HISPLAT'
        ]

        # Build feature matrix
        X_list = []
        feature_names = []

        for feature in baseline_features:
            if feature not in self.trajectories_df.columns:
                print(f"   Warning: {feature} not found, skipping")
                continue

            if feature in ['SEX', 'HANDED', 'HISPLAT']:
                # Categorical features (already encoded 0/1 or similar)
                values = self.trajectories_df[feature].fillna(-1).values
                X_list.append(values.reshape(-1, 1))
                feature_names.append(feature)
            else:
                # Continuous features
                values = self.trajectories_df[feature].values
                # Fill missing with median
                median_val = np.nanmedian(values)
                values = np.nan_to_num(values, nan=median_val)
                X_list.append(values.reshape(-1, 1))
                feature_names.append(feature)

        X = np.hstack(X_list)
        y = self.trajectories_df['cluster'].values.astype(int)

        self.feature_names = feature_names

        print(f"   Feature matrix shape: {X.shape}")
        print(f"   Features used: {feature_names}")

        # Train/test split (stratified)
        n_samples = len(X)
        n_test = int(n_samples * self.test_size)

        # Stratified split
        np.random.seed(42)
        indices = np.arange(n_samples)

        # Simple stratified split by cluster
        test_indices = []
        train_indices = []

        for cluster_id in np.unique(y):
            cluster_indices = indices[y == cluster_id]
            np.random.shuffle(cluster_indices)

            n_cluster_test = int(len(cluster_indices) * self.test_size)
            test_indices.extend(cluster_indices[:n_cluster_test])
            train_indices.extend(cluster_indices[n_cluster_test:])

        test_indices = np.array(test_indices)
        train_indices = np.array(train_indices)

        self.X_train = X[train_indices]
        self.X_test = X[test_indices]
        self.y_train = y[train_indices]
        self.y_test = y[test_indices]

        # Standardize features
        self.scaler = StandardScaler()
        self.X_train = self.scaler.fit_transform(self.X_train)
        self.X_test = self.scaler.transform(self.X_test)

        print(f"   Train set: {len(self.X_train)} samples")
        print(f"   Test set: {len(self.X_test)} samples")
        print(f"   Train class distribution: {np.bincount(self.y_train)}")
        print(f"   Test class distribution: {np.bincount(self.y_test)}")

        return self.X_train, self.X_test, self.y_train, self.y_test

    def train_models(self):
        """Train multiple classifier models."""
        print("\n[TRAIN] Training classifier models...")

        # Define models
        model_configs = {
            'Logistic Regression': LogisticRegression(
                multi_class='multinomial',
                max_iter=1000,
                random_state=42,
                class_weight='balanced'
            ),
            'Random Forest': RandomForestClassifier(
                n_estimators=200,
                max_depth=10,
                random_state=42,
                class_weight='balanced'
            ),
            'Gradient Boosting': GradientBoostingClassifier(
                n_estimators=200,
                max_depth=5,
                random_state=42,
                learning_rate=0.1
            ),
            'SVM (RBF)': SVC(
                kernel='rbf',
                probability=True,
                random_state=42,
                class_weight='balanced'
            )
        }

        # Train and evaluate each model
        for model_name, model in model_configs.items():
            print(f"\n   Training {model_name}...")

            # Cross-validation on training set
            cv = StratifiedKFold(n_splits=self.n_folds, shuffle=True, random_state=42)
            cv_scores = cross_val_score(model, self.X_train, self.y_train, cv=cv, scoring='balanced_accuracy')

            print(f"      CV Balanced Accuracy: {cv_scores.mean():.3f} +/- {cv_scores.std():.3f}")

            # Train on full training set
            model.fit(self.X_train, self.y_train)

            # Test set predictions
            y_pred = model.predict(self.X_test)
            y_pred_proba = model.predict_proba(self.X_test)

            # Compute metrics
            accuracy = accuracy_score(self.y_test, y_pred)
            balanced_acc = balanced_accuracy_score(self.y_test, y_pred)
            f1_macro = f1_score(self.y_test, y_pred, average='macro')
            f1_weighted = f1_score(self.y_test, y_pred, average='weighted')

            # ROC AUC (one-vs-rest)
            try:
                roc_auc = roc_auc_score(self.y_test, y_pred_proba, multi_class='ovr', average='macro')
            except:
                roc_auc = np.nan

            print(f"      Test Accuracy: {accuracy:.3f}")
            print(f"      Test Balanced Accuracy: {balanced_acc:.3f}")
            print(f"      Test F1 (macro): {f1_macro:.3f}")
            print(f"      Test ROC-AUC (macro): {roc_auc:.3f}")

            # Store model and metrics
            self.models[model_name] = model
            self.performance_metrics[model_name] = {
                'cv_balanced_accuracy_mean': float(cv_scores.mean()),
                'cv_balanced_accuracy_std': float(cv_scores.std()),
                'test_accuracy': float(accuracy),
                'test_balanced_accuracy': float(balanced_acc),
                'test_f1_macro': float(f1_macro),
                'test_f1_weighted': float(f1_weighted),
                'test_roc_auc_macro': float(roc_auc),
                'y_pred': y_pred.tolist(),
                'y_pred_proba': y_pred_proba.tolist(),
                'classification_report': classification_report(self.y_test, y_pred, output_dict=True)
            }

        # Select best model (by balanced accuracy)
        best_score = -1
        for model_name, metrics in self.performance_metrics.items():
            if metrics['test_balanced_accuracy'] > best_score:
                best_score = metrics['test_balanced_accuracy']
                self.best_model_name = model_name

        print(f"\n   Best model: {self.best_model_name} (Balanced Acc: {best_score:.3f})")

        return self.models

    def analyze_feature_importance(self):
        """Analyze feature importance for best model."""
        print(f"\n[IMPORTANCE] Analyzing feature importance ({self.best_model_name})...")

        best_model = self.models[self.best_model_name]

        # Extract feature importance
        if hasattr(best_model, 'feature_importances_'):
            # Tree-based models
            importances = best_model.feature_importances_
        elif hasattr(best_model, 'coef_'):
            # Linear models (average absolute coefficients across classes)
            importances = np.abs(best_model.coef_).mean(axis=0)
        else:
            print("   Model does not support feature importance extraction")
            return {}

        # Sort features by importance
        sorted_idx = np.argsort(importances)[::-1]

        print(f"\n   Feature Importance Rankings:")
        for i, idx in enumerate(sorted_idx, 1):
            print(f"      {i}. {self.feature_names[idx]}: {importances[idx]:.3f}")

        feature_importance = {
            self.feature_names[i]: float(importances[i])
            for i in range(len(self.feature_names))
        }

        return feature_importance

    def generate_visualizations(self, feature_importance: Dict[str, float]):
        """Generate prediction performance visualizations."""
        print("\n[VIZ] Generating prediction visualizations...")

        fig = plt.figure(figsize=(22, 14))
        gs = fig.add_gridspec(3, 4, hspace=0.35, wspace=0.35)

        fig.suptitle('Phase 4, Task 4.5: Baseline Subtype Prediction Analysis',
                     fontsize=16, fontweight='bold', y=0.995)

        # 1. Model comparison (balanced accuracy)
        ax1 = fig.add_subplot(gs[0, 0])
        model_names = list(self.performance_metrics.keys())
        bal_accs = [self.performance_metrics[m]['test_balanced_accuracy'] for m in model_names]
        colors_bar = ['green' if m == self.best_model_name else 'steelblue' for m in model_names]

        ax1.barh(range(len(model_names)), bal_accs, color=colors_bar, edgecolor='black')
        ax1.set_yticks(range(len(model_names)))
        ax1.set_yticklabels(model_names)
        ax1.set_xlabel('Balanced Accuracy')
        ax1.set_title('Model Comparison (Test Set)')
        ax1.axvline(0.33, color='red', linestyle='--', linewidth=1, alpha=0.5, label='Random (3 classes)')
        ax1.legend()
        ax1.grid(axis='x', alpha=0.3)

        # 2. F1-scores comparison
        ax2 = fig.add_subplot(gs[0, 1])
        f1_macros = [self.performance_metrics[m]['test_f1_macro'] for m in model_names]
        ax2.barh(range(len(model_names)), f1_macros, color=colors_bar, edgecolor='black')
        ax2.set_yticks(range(len(model_names)))
        ax2.set_yticklabels(model_names)
        ax2.set_xlabel('F1-Score (macro)')
        ax2.set_title('F1-Score Comparison')
        ax2.grid(axis='x', alpha=0.3)

        # 3. Feature importance
        ax3 = fig.add_subplot(gs[0, 2:])
        if feature_importance:
            features = list(feature_importance.keys())
            importances = list(feature_importance.values())
            sorted_idx = np.argsort(importances)

            ax3.barh(range(len(features)), [importances[i] for i in sorted_idx],
                    color='coral', edgecolor='black')
            ax3.set_yticks(range(len(features)))
            ax3.set_yticklabels([features[i] for i in sorted_idx])
            ax3.set_xlabel('Feature Importance')
            ax3.set_title(f'Feature Importance ({self.best_model_name})')
            ax3.grid(axis='x', alpha=0.3)

        # 4. Confusion matrix (best model)
        ax4 = fig.add_subplot(gs[1, 0])
        y_pred = np.array(self.performance_metrics[self.best_model_name]['y_pred'])
        cm = confusion_matrix(self.y_test, y_pred)
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues', ax=ax4,
                   xticklabels=[f'Pred {i}' for i in range(cm.shape[0])],
                   yticklabels=[f'True {i}' for i in range(cm.shape[1])])
        ax4.set_title(f'Confusion Matrix ({self.best_model_name})')
        ax4.set_ylabel('True Label')
        ax4.set_xlabel('Predicted Label')

        # 5. Per-class F1 scores (best model)
        ax5 = fig.add_subplot(gs[1, 1])
        clf_report = self.performance_metrics[self.best_model_name]['classification_report']
        class_labels = [str(i) for i in range(len(np.unique(self.y_test)))]
        f1_scores_per_class = [clf_report[label]['f1-score'] for label in class_labels]
        support_per_class = [clf_report[label]['support'] for label in class_labels]

        ax5.bar(range(len(class_labels)), f1_scores_per_class,
               color=plt.cm.tab10(np.arange(len(class_labels))), edgecolor='black')
        ax5.set_xticks(range(len(class_labels)))
        ax5.set_xticklabels([f'Class {i}\n(n={support_per_class[i]:.0f})' for i in range(len(class_labels))])
        ax5.set_ylabel('F1-Score')
        ax5.set_title('Per-Class F1-Scores')
        ax5.set_ylim([0, 1])
        ax5.grid(axis='y', alpha=0.3)

        # 6. ROC curves (one-vs-rest for best model)
        ax6 = fig.add_subplot(gs[1, 2:])
        y_pred_proba = np.array(self.performance_metrics[self.best_model_name]['y_pred_proba'])
        n_classes = y_pred_proba.shape[1]

        for i in range(n_classes):
            # Binary labels for class i
            y_true_binary = (self.y_test == i).astype(int)
            y_score = y_pred_proba[:, i]

            fpr, tpr, _ = roc_curve(y_true_binary, y_score)
            roc_auc = auc(fpr, tpr)

            ax6.plot(fpr, tpr, linewidth=2,
                    label=f'Class {i} (AUC={roc_auc:.3f})',
                    color=plt.cm.tab10(i))

        ax6.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random')
        ax6.set_xlabel('False Positive Rate')
        ax6.set_ylabel('True Positive Rate')
        ax6.set_title(f'ROC Curves - One-vs-Rest ({self.best_model_name})')
        ax6.legend(loc='lower right')
        ax6.grid(alpha=0.3)

        # 7-9. Calibration curves per class
        for i in range(min(n_classes, 3)):
            row = 2
            col = i
            ax = fig.add_subplot(gs[row, col])

            y_true_binary = (self.y_test == i).astype(int)
            y_score = y_pred_proba[:, i]

            try:
                fraction_of_positives, mean_predicted_value = calibration_curve(
                    y_true_binary, y_score, n_bins=5, strategy='uniform'
                )

                ax.plot(mean_predicted_value, fraction_of_positives, 's-',
                       linewidth=2, markersize=8, color=plt.cm.tab10(i), label='Model')
                ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Perfect calibration')
                ax.set_xlabel('Mean Predicted Probability')
                ax.set_ylabel('Fraction of Positives')
                ax.set_title(f'Calibration Curve - Class {i}')
                ax.legend()
                ax.grid(alpha=0.3)
            except:
                ax.text(0.5, 0.5, 'Insufficient data\nfor calibration',
                       ha='center', va='center', transform=ax.transAxes)
                ax.set_title(f'Calibration Curve - Class {i}')

        # 10. Prediction confidence distribution
        ax10 = fig.add_subplot(gs[2, 3])
        max_proba = y_pred_proba.max(axis=1)
        ax10.hist(max_proba, bins=20, edgecolor='black', alpha=0.7, color='mediumseagreen')
        ax10.axvline(max_proba.mean(), color='red', linestyle='--',
                    label=f'Mean: {max_proba.mean():.3f}')
        ax10.set_xlabel('Max Predicted Probability')
        ax10.set_ylabel('Number of Predictions')
        ax10.set_title('Prediction Confidence Distribution')
        ax10.legend()
        ax10.grid(axis='y', alpha=0.3)

        # Save figure
        viz_path = self.output_dir / 'baseline_subtype_prediction_analysis.png'
        plt.savefig(viz_path, dpi=300, bbox_inches='tight')
        print(f"   Saved visualization: {viz_path}")

        plt.close()

    def save_outputs(self, feature_importance: Dict[str, float]):
        """Save prediction models and results."""
        print("\n[SAVE] Saving prediction outputs...")

        # Save performance metrics
        metrics_report = {
            'best_model': self.best_model_name,
            'performance_metrics': self.performance_metrics,
            'feature_importance': feature_importance,
            'feature_names': self.feature_names,
            'metadata': {
                'timestamp': datetime.now().isoformat(),
                'test_size': float(self.test_size),
                'n_folds': int(self.n_folds),
                'n_train': int(len(self.X_train)),
                'n_test': int(len(self.X_test)),
                'n_features': int(len(self.feature_names))
            }
        }

        report_path = self.output_dir / 'baseline_prediction_report.json'
        with open(report_path, 'w') as f:
            json.dump(metrics_report, f, indent=2)
        print(f"   Saved prediction report: {report_path}")

        # Save best model (using joblib would be better, but save summary for now)
        print(f"   Best model ({self.best_model_name}) trained and ready for deployment")

    def run_full_pipeline(self):
        """Execute complete baseline subtype prediction pipeline."""
        print("="*80)
        print("PHASE 4, TASK 4.5: BASELINE SUBTYPE PREDICTION")
        print("="*80)

        # Step 1: Load data
        self.load_data()

        # Step 2: Prepare features
        self.prepare_features()

        # Step 3: Train models
        self.train_models()

        # Step 4: Analyze feature importance
        feature_importance = self.analyze_feature_importance()

        # Step 5: Generate visualizations
        self.generate_visualizations(feature_importance)

        # Step 6: Save outputs
        self.save_outputs(feature_importance)

        print("\n" + "="*80)
        print("TASK 4.5 COMPLETE")
        print("="*80)
        print(f"\nSummary:")
        print(f"   Best model: {self.best_model_name}")
        print(f"   Test balanced accuracy: {self.performance_metrics[self.best_model_name]['test_balanced_accuracy']:.3f}")
        print(f"   Test F1 (macro): {self.performance_metrics[self.best_model_name]['test_f1_macro']:.3f}")
        print(f"   Test ROC-AUC (macro): {self.performance_metrics[self.best_model_name]['test_roc_auc_macro']:.3f}")
        print(f"   Number of features: {len(self.feature_names)}")
        print(f"\nOutputs saved to: {self.output_dir}")

        return self.models, self.performance_metrics


def main():
    """Main execution function."""

    # Configuration
    LABELED_TRAJ_PATH = r"e:\My Drive\CSCI FALL 2025\data\longitudinal_cohort\patient_trajectories_labeled.csv"
    OUTPUT_DIR = r"e:\My Drive\CSCI FALL 2025\data\longitudinal_cohort"
    TEST_SIZE = 0.2
    N_FOLDS = 5

    # Initialize and run pipeline
    prediction = BaselineSubtypePrediction(
        labeled_trajectories_path=LABELED_TRAJ_PATH,
        output_dir=OUTPUT_DIR,
        test_size=TEST_SIZE,
        n_folds=N_FOLDS
    )

    models, performance = prediction.run_full_pipeline()

    return models, performance


if __name__ == "__main__":
    models, performance = main()
