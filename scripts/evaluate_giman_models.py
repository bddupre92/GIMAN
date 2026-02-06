"""
GIMAN Dual-Model Test Set Evaluation Script

This script evaluates both GIMAN-Progression and GIMAN-Conversion models
on the held-out test set (20 patients) with comprehensive metrics and
confidence intervals.

Week 4, Task 5: Model Evaluation
Author: GIMAN Development Team
Date: October 12, 2025
"""

import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
from lifelines.utils import concordance_index
from scipy import stats
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    auc,
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from torch_geometric.data import Data

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from models.giman_progression import GIMANProgression
from models.giman_conversion import GIMANConversion


# ============================================================================
# Configuration
# ============================================================================

RESULTS_DIR = Path("results/week4")
PROGRESSION_DIR = RESULTS_DIR / "progression"
CONVERSION_DIR = RESULTS_DIR / "conversion"
EVALUATION_DIR = RESULTS_DIR / "evaluation"
FIGURES_DIR = RESULTS_DIR / "figures"

# Create output directories
EVALUATION_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(EVALUATION_DIR / "evaluation.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Random seed for reproducibility
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

# Bootstrap parameters
N_BOOTSTRAP = 1000
CONFIDENCE_LEVEL = 0.95


# ============================================================================
# Data Loading
# ============================================================================

def load_test_data() -> Tuple[Data, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Load test set data and labels."""
    logger.info("Loading test set data...")
    
    # Load graph data
    data_dir = Path("data/02_processed/training_ready")
    test_data = torch.load(data_dir / "test_data.pt", weights_only=False)
    
    # Load hybrid survival endpoints
    survival_file = Path("data/02_processed/progression_survival_data_hybrid.csv")
    survival_df = pd.read_csv(survival_file)
    
    # Load hybrid conversion labels
    conversion_file = Path("data/02_processed/conversion_labels_hybrid.csv")
    conversion_df = pd.read_csv(conversion_file)
    
    # Load patient split info
    split_file = data_dir / "split_info.json"
    with open(split_file, 'r') as f:
        split_info = json.load(f)
    test_patnos = split_info['test_patnos']
    
    # Filter for test set
    test_survival = survival_df[survival_df['PATNO'].isin(test_patnos)]
    test_conversion = conversion_df[conversion_df['PATNO'].isin(test_patnos)]
    
    # Convert to tensors
    event_times = torch.tensor(test_survival['event_time'].values, dtype=torch.float32)
    event_observed = torch.tensor(test_survival['event_observed'].values, dtype=torch.float32)
    conversion_labels = torch.tensor(test_conversion['converted'].values, dtype=torch.float32)
    
    logger.info(f"   Test set: {len(test_patnos)} patients")
    logger.info(f"   Survival events: {event_observed.sum():.0f}/{len(event_observed)} ({event_observed.mean():.1%})")
    logger.info(f"   Conversions: {conversion_labels.sum():.0f}/{len(conversion_labels)} ({conversion_labels.mean():.1%})")
    
    return test_data, event_times, event_observed, conversion_labels


def load_models() -> Tuple[GIMANProgression, GIMANConversion]:
    """Load trained models from checkpoints."""
    logger.info("Loading trained models...")
    
    # Load GIMAN-Progression
    progression_checkpoint_path = PROGRESSION_DIR / "checkpoints" / "best_checkpoint.pt"
    checkpoint = torch.load(progression_checkpoint_path, weights_only=False)
    
    # Get model config from checkpoint
    config = checkpoint['config']
    prog_config = config['giman_progression']['model']
    
    # Initialize model with correct parameters
    progression_model = GIMANProgression(
        num_features=prog_config['num_features'],
        hidden_dim=prog_config['hidden_dim'],
        num_gat_layers=prog_config['num_gat_layers'],
        num_heads=prog_config['num_heads'],
        survival_hidden_dims=prog_config['survival_hidden_dims'],
        dropout=prog_config['dropout']
    )
    progression_model.load_state_dict(checkpoint['model_state_dict'])
    progression_model.eval()
    logger.info(f"   ✓ GIMAN-Progression loaded (Epoch {checkpoint['epoch']}, Val C-index: {checkpoint['best_val_cindex']:.4f})")
    
    # Load GIMAN-Conversion
    conversion_checkpoint_path = CONVERSION_DIR / "checkpoints" / "best_checkpoint.pt"
    checkpoint = torch.load(conversion_checkpoint_path, weights_only=False)
    
    # Get model config from checkpoint
    config = checkpoint['config']
    conv_config = config['giman_conversion']['model']
    
    # Initialize model with correct parameters
    conversion_model = GIMANConversion(
        num_features=conv_config['num_features'],
        hidden_dim=conv_config['hidden_dim'],
        num_gat_layers=conv_config['num_gat_layers'],
        num_heads=conv_config['num_heads'],
        conversion_hidden_dims=conv_config['classifier_hidden_dims'],
        dropout=conv_config['dropout']
    )
    conversion_model.load_state_dict(checkpoint['model_state_dict'])
    conversion_model.eval()
    logger.info(f"   ✓ GIMAN-Conversion loaded (Epoch {checkpoint['epoch']})")
    
    return progression_model, conversion_model


# ============================================================================
# Bootstrap Confidence Intervals
# ============================================================================

def bootstrap_metric(y_true: np.ndarray, y_pred: np.ndarray,
                     metric_fn, n_bootstrap: int = N_BOOTSTRAP,
                     confidence: float = CONFIDENCE_LEVEL) -> Tuple[float, float, float]:
    """
    Compute metric with bootstrap confidence interval.
    
    Args:
        y_true: True labels
        y_pred: Predicted values
        metric_fn: Function to compute metric (takes y_true, y_pred)
        n_bootstrap: Number of bootstrap samples
        confidence: Confidence level (0.95 = 95% CI)
    
    Returns:
        (point_estimate, lower_ci, upper_ci)
    """
    n = len(y_true)
    bootstrap_scores = []
    
    for _ in range(n_bootstrap):
        # Resample with replacement
        indices = np.random.choice(n, size=n, replace=True)
        y_true_boot = y_true[indices]
        y_pred_boot = y_pred[indices]
        
        # Compute metric
        try:
            score = metric_fn(y_true_boot, y_pred_boot)
            bootstrap_scores.append(score)
        except:
            continue
    
    bootstrap_scores = np.array(bootstrap_scores)
    
    # Compute point estimate and CI
    point_estimate = metric_fn(y_true, y_pred)
    alpha = (1 - confidence) / 2
    lower_ci = np.percentile(bootstrap_scores, alpha * 100)
    upper_ci = np.percentile(bootstrap_scores, (1 - alpha) * 100)
    
    return point_estimate, lower_ci, upper_ci


# ============================================================================
# Survival Analysis Evaluation
# ============================================================================

def evaluate_progression(model: GIMANProgression, test_data: Data,
                         event_times: torch.Tensor, event_observed: torch.Tensor) -> Dict:
    """Evaluate GIMAN-Progression model."""
    logger.info("\n" + "="*70)
    logger.info("GIMAN-PROGRESSION EVALUATION")
    logger.info("="*70)
    
    results = {}
    
    # Get predictions
    with torch.no_grad():
        risk_scores = model(test_data.x, test_data.edge_index).squeeze().numpy()
    
    event_times_np = event_times.numpy()
    event_observed_np = event_observed.numpy()
    
    # 1. Concordance Index (C-index)
    logger.info("\n1. Concordance Index (C-index)")
    
    def cindex_fn(times, scores):
        return concordance_index(times, -scores, event_observed_np)
    
    cindex, cindex_lower, cindex_upper = bootstrap_metric(
        event_times_np, risk_scores,
        lambda t, s: cindex_fn(t, s)
    )
    
    results['cindex'] = {
        'value': float(cindex),
        'ci_lower': float(cindex_lower),
        'ci_upper': float(cindex_upper),
        'ci_level': CONFIDENCE_LEVEL
    }
    
    logger.info(f"   C-index: {cindex:.4f} (95% CI: [{cindex_lower:.4f}, {cindex_upper:.4f}])")
    
    # 2. Risk Stratification
    logger.info("\n2. Risk Stratification by Quartiles")
    
    # Divide into risk quartiles
    risk_quartiles = pd.qcut(risk_scores, q=4, labels=['Q1-Low', 'Q2-Medium', 'Q3-High', 'Q4-Very High'])
    
    stratification = []
    for quartile in ['Q1-Low', 'Q2-Medium', 'Q3-High', 'Q4-Very High']:
        mask = risk_quartiles == quartile
        n_patients = mask.sum()
        n_events = event_observed_np[mask].sum()
        event_rate = event_observed_np[mask].mean()
        median_time = np.median(event_times_np[mask])
        
        stratification.append({
            'quartile': quartile,
            'n_patients': int(n_patients),
            'n_events': int(n_events),
            'event_rate': float(event_rate),
            'median_time': float(median_time)
        })
        
        logger.info(f"   {quartile}: {n_patients} patients, {n_events} events ({event_rate:.1%}), median time {median_time:.2f} years")
    
    results['risk_stratification'] = stratification
    
    # 3. Event Statistics
    logger.info("\n3. Event Statistics")
    
    n_events = event_observed_np.sum()
    n_censored = len(event_observed_np) - n_events
    event_rate = event_observed_np.mean()
    
    event_times_events = event_times_np[event_observed_np == 1]
    if len(event_times_events) > 0:
        mean_event_time = event_times_events.mean()
        median_event_time = np.median(event_times_events)
    else:
        mean_event_time = median_event_time = np.nan
    
    results['event_statistics'] = {
        'total_patients': int(len(event_observed_np)),
        'n_events': int(n_events),
        'n_censored': int(n_censored),
        'event_rate': float(event_rate),
        'mean_event_time': float(mean_event_time),
        'median_event_time': float(median_event_time)
    }
    
    logger.info(f"   Total patients: {len(event_observed_np)}")
    logger.info(f"   Events: {n_events} ({event_rate:.1%})")
    logger.info(f"   Censored: {n_censored} ({1-event_rate:.1%})")
    logger.info(f"   Mean event time: {mean_event_time:.2f} years")
    logger.info(f"   Median event time: {median_event_time:.2f} years")
    
    return results


# ============================================================================
# Binary Classification Evaluation
# ============================================================================

def evaluate_conversion(model: GIMANConversion, test_data: Data,
                        labels: torch.Tensor) -> Dict:
    """Evaluate GIMAN-Conversion model."""
    logger.info("\n" + "="*70)
    logger.info("GIMAN-CONVERSION EVALUATION")
    logger.info("="*70)
    
    results = {}
    
    # Get predictions (logits from model)
    with torch.no_grad():
        logits = model(test_data.x, test_data.edge_index).squeeze()
        probs = torch.sigmoid(logits).numpy()  # Convert logits to probabilities
    
    labels_np = labels.numpy()
    
    # 1. AUC-ROC
    logger.info("\n1. AUC-ROC (Area Under ROC Curve)")
    
    auc_roc, auc_lower, auc_upper = bootstrap_metric(
        labels_np, probs,
        lambda y, p: roc_auc_score(y, p)
    )
    
    results['auc_roc'] = {
        'value': float(auc_roc),
        'ci_lower': float(auc_lower),
        'ci_upper': float(auc_upper),
        'ci_level': CONFIDENCE_LEVEL
    }
    
    logger.info(f"   AUC-ROC: {auc_roc:.4f} (95% CI: [{auc_lower:.4f}, {auc_upper:.4f}])")
    
    # 2. AUC-PR (Precision-Recall)
    logger.info("\n2. AUC-PR (Area Under Precision-Recall Curve)")
    
    precision, recall, _ = precision_recall_curve(labels_np, probs)
    auc_pr = auc(recall, precision)
    
    # Bootstrap AUC-PR
    def compute_auc_pr(y, p):
        prec, rec, _ = precision_recall_curve(y, p)
        return auc(rec, prec)
    
    auc_pr, auc_pr_lower, auc_pr_upper = bootstrap_metric(
        labels_np, probs, compute_auc_pr
    )
    
    results['auc_pr'] = {
        'value': float(auc_pr),
        'ci_lower': float(auc_pr_lower),
        'ci_upper': float(auc_pr_upper),
        'ci_level': CONFIDENCE_LEVEL
    }
    
    logger.info(f"   AUC-PR: {auc_pr:.4f} (95% CI: [{auc_pr_lower:.4f}, {auc_pr_upper:.4f}])")
    
    # 3. Optimal Threshold (Youden's J statistic)
    logger.info("\n3. Optimal Classification Threshold")
    
    fpr, tpr, thresholds = roc_curve(labels_np, probs)
    j_scores = tpr - fpr
    optimal_idx = np.argmax(j_scores)
    optimal_threshold = thresholds[optimal_idx]
    
    results['optimal_threshold'] = {
        'value': float(optimal_threshold),
        'tpr': float(tpr[optimal_idx]),
        'fpr': float(fpr[optimal_idx]),
        'youdens_j': float(j_scores[optimal_idx])
    }
    
    logger.info(f"   Optimal threshold: {optimal_threshold:.4f}")
    logger.info(f"   TPR (Sensitivity): {tpr[optimal_idx]:.4f}")
    logger.info(f"   FPR (1 - Specificity): {fpr[optimal_idx]:.4f}")
    logger.info(f"   Youden's J: {j_scores[optimal_idx]:.4f}")
    
    # 4. Classification Metrics at Optimal Threshold
    logger.info("\n4. Classification Metrics (at optimal threshold)")
    
    preds_binary = (probs >= optimal_threshold).astype(int)
    
    # Confusion matrix
    tn, fp, fn, tp = confusion_matrix(labels_np, preds_binary).ravel()
    
    accuracy = accuracy_score(labels_np, preds_binary)
    balanced_acc = balanced_accuracy_score(labels_np, preds_binary)
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    ppv = tp / (tp + fp) if (tp + fp) > 0 else 0  # Precision
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0
    f1 = f1_score(labels_np, preds_binary)
    
    results['classification_metrics'] = {
        'accuracy': float(accuracy),
        'balanced_accuracy': float(balanced_acc),
        'sensitivity': float(sensitivity),
        'specificity': float(specificity),
        'ppv': float(ppv),
        'npv': float(npv),
        'f1_score': float(f1)
    }
    
    results['confusion_matrix'] = {
        'tn': int(tn),
        'fp': int(fp),
        'fn': int(fn),
        'tp': int(tp)
    }
    
    logger.info(f"   Accuracy: {accuracy:.4f}")
    logger.info(f"   Balanced Accuracy: {balanced_acc:.4f}")
    logger.info(f"   Sensitivity (Recall): {sensitivity:.4f}")
    logger.info(f"   Specificity: {specificity:.4f}")
    logger.info(f"   PPV (Precision): {ppv:.4f}")
    logger.info(f"   NPV: {npv:.4f}")
    logger.info(f"   F1-Score: {f1:.4f}")
    
    logger.info(f"\n   Confusion Matrix:")
    logger.info(f"   TN: {tn}, FP: {fp}")
    logger.info(f"   FN: {fn}, TP: {tp}")
    
    # 5. Calibration
    logger.info("\n5. Calibration Analysis")
    
    prob_true, prob_pred = calibration_curve(labels_np, probs, n_bins=5, strategy='quantile')
    
    # Compute calibration error (mean absolute difference)
    calibration_error = np.mean(np.abs(prob_true - prob_pred))
    
    results['calibration'] = {
        'prob_true': prob_true.tolist(),
        'prob_pred': prob_pred.tolist(),
        'calibration_error': float(calibration_error)
    }
    
    logger.info(f"   Mean calibration error: {calibration_error:.4f}")
    logger.info(f"   Predicted probs: {prob_pred}")
    logger.info(f"   Actual frequencies: {prob_true}")
    
    return results


# ============================================================================
# Main Evaluation
# ============================================================================

def main():
    """Main evaluation pipeline."""
    logger.info("\n" + "="*70)
    logger.info("GIMAN DUAL-MODEL TEST SET EVALUATION")
    logger.info("="*70)
    logger.info(f"Date: October 12, 2025")
    logger.info(f"Test set: 20 held-out patients")
    logger.info(f"Bootstrap samples: {N_BOOTSTRAP}")
    logger.info(f"Confidence level: {CONFIDENCE_LEVEL:.0%}")
    logger.info("="*70)
    
    # Load data and models
    test_data, event_times, event_observed, conversion_labels = load_test_data()
    progression_model, conversion_model = load_models()
    
    # Evaluate GIMAN-Progression
    progression_results = evaluate_progression(
        progression_model, test_data, event_times, event_observed
    )
    
    # Evaluate GIMAN-Conversion
    conversion_results = evaluate_conversion(
        conversion_model, test_data, conversion_labels
    )
    
    # Combine results
    evaluation_report = {
        'metadata': {
            'date': '2025-10-12',
            'test_set_size': 20,
            'bootstrap_samples': N_BOOTSTRAP,
            'confidence_level': CONFIDENCE_LEVEL,
            'random_seed': RANDOM_SEED
        },
        'giman_progression': progression_results,
        'giman_conversion': conversion_results
    }
    
    # Save results
    output_file = EVALUATION_DIR / "evaluation_report.json"
    with open(output_file, 'w') as f:
        json.dump(evaluation_report, f, indent=2)
    
    logger.info("\n" + "="*70)
    logger.info("EVALUATION COMPLETE")
    logger.info("="*70)
    logger.info(f"Results saved to: {output_file}")
    logger.info(f"Logs saved to: {EVALUATION_DIR / 'evaluation.log'}")
    logger.info("\nNext steps:")
    logger.info("  1. Generate visualizations (KM curves, ROC curves)")
    logger.info("  2. Create patient-level reports")
    logger.info("  3. Write Week 4 completion documentation")


if __name__ == "__main__":
    main()
