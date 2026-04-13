"""
Phase 8.3: Train GIMAN-SAA Model

Train Graph-Informed Multimodal Attention Network for SAA prediction.

Input:
    - data/04_saa/saa_training_data.csv (608 observations, 59 features)
    
Output:
    - Trained GIMAN-SAA model
    - Training metrics and evaluation results
    - Feature importance analysis

Author: GIMAN Research Team
Date: October 13, 2025
"""

import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path("e:/My Drive/CSCI FALL 2025")
sys.path.insert(0, str(project_root))

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, DataLoader
from torch_geometric.utils import k_hop_subgraph
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import (
    roc_auc_score, average_precision_score, 
    accuracy_score, balanced_accuracy_score,
    precision_recall_curve, roc_curve,
    confusion_matrix, classification_report
)
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
import seaborn as sns
import json
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Import GIMAN-SAA model
sys.path.insert(0, str(project_root / "archive/development/phase8/subphase8_3_saa_integration"))
from models.giman_saa import GIMAN_SAA
from configs.saa_config import SAAConfig


class SAATrainer:
    """Train and evaluate GIMAN-SAA model."""
    
    def __init__(self, config: SAAConfig):
        """Initialize trainer with configuration."""
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Create output directories
        self.config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        self.config.MODEL_DIR.mkdir(parents=True, exist_ok=True)
        self.config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        self.config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
        
        # Data storage
        self.df = None
        self.train_data = None
        self.val_data = None
        self.test_data = None
        self.model = None
        self.optimizer = None
        self.scheduler = None
        
        # Results storage
        self.train_history = {
            'loss': [], 'auc': [], 'acc': []
        }
        self.val_history = {
            'loss': [], 'auc': [], 'acc': []
        }
        self.best_val_auc = 0.0
        self.best_epoch = 0
    
    def load_data(self):
        """Load SAA training dataset."""
        print("\n" + "=" * 80)
        print("LOADING SAA TRAINING DATA")
        print("=" * 80)
        
        data_path = self.config.SAA_TRAINING_DATA
        if not data_path.exists():
            raise FileNotFoundError(f"Training data not found: {data_path}")
        
        self.df = pd.read_csv(data_path)
        print(f"Loaded {len(self.df)} observations from {self.df['PATNO'].nunique()} patients")
        print(f"Features: {len(self.df.columns)}")
        print(f"SAA+ rate: {self.df['SAA_POSITIVE'].mean():.1%}")
        
        # Check for missing values
        missing = self.df.isnull().sum().sum()
        if missing > 0:
            print(f"Warning: {missing} missing values detected")
        else:
            print(f"OK - No missing values")
        
        return self.df
    
    def prepare_features(self):
        """Prepare feature matrix and labels with standardization."""
        from sklearn.preprocessing import StandardScaler
        
        print("\n" + "=" * 80)
        print("PREPARING FEATURES")
        print("=" * 80)
        
        # Exclude non-feature columns
        exclude_cols = [
            'PATNO', 'EVENT_ID', 'SAA_POSITIVE', 'ALPHA_SYN_VALUE',
            'time_to_event', 'phenoconverted', 'landmark_month',
            'original_time', 'original_event', 'cohort'
        ]
        
        feature_cols = [col for col in self.df.columns if col not in exclude_cols]
        print(f"Feature columns: {len(feature_cols)}")
        
        # Extract features and labels
        X = self.df[feature_cols].values.astype(np.float32)
        y = self.df['SAA_POSITIVE'].values.astype(np.int64)
        patient_ids = self.df['PATNO'].values
        
        print(f"Feature matrix shape: {X.shape}")
        print(f"Label distribution: SAA- = {(y==0).sum()}, SAA+ = {(y==1).sum()}")
        
        # CRITICAL: Standardize features (mean=0, std=1)
        # This is essential for graph neural networks with different feature scales
        print("\nApplying StandardScaler...")
        self.scaler = StandardScaler()
        X = self.scaler.fit_transform(X)
        
        print(f"Standardized features: mean={X.mean():.6f}, std={X.std():.6f}")
        print(f"Feature ranges: [{X.min():.2f}, {X.max():.2f}]")
        
        return X, y, patient_ids, feature_cols
    
    def build_graph(self, X, k=10):
        """
        Build k-NN graph from feature matrix.
        
        Args:
            X: Feature matrix (n_samples, n_features)
            k: Number of nearest neighbors
            
        Returns:
            edge_index: Graph edges (2, num_edges)
        """
        print(f"\nBuilding k-NN graph (k={k})...")
        
        # Find k nearest neighbors
        nbrs = NearestNeighbors(n_neighbors=k+1, metric='cosine', n_jobs=-1)
        nbrs.fit(X)
        distances, indices = nbrs.kneighbors(X)
        
        # Build edge list (exclude self-loops)
        edge_list = []
        for i in range(len(X)):
            for j in indices[i][1:]:  # Skip first neighbor (self)
                edge_list.append([i, j])
        
        edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
        
        print(f"Graph: {len(X)} nodes, {edge_index.shape[1]} edges")
        print(f"Average degree: {edge_index.shape[1] / len(X):.1f}")
        
        return edge_index
    
    def create_pyg_data(self, X, y, edge_index):
        """
        Create PyTorch Geometric Data object.
        
        Args:
            X: Feature matrix
            y: Labels
            edge_index: Graph edges
            
        Returns:
            PyG Data object
        """
        data = Data(
            x=torch.tensor(X, dtype=torch.float32),
            y=torch.tensor(y, dtype=torch.long),
            edge_index=edge_index
        )
        return data
    
    def split_data(self, X, y, patient_ids):
        """
        Split data into train/val/test sets (patient-level split).
        
        Args:
            X: Feature matrix
            y: Labels
            patient_ids: Patient identifiers
            
        Returns:
            Train, validation, and test PyG Data objects
        """
        print("\n" + "=" * 80)
        print("SPLITTING DATA")
        print("=" * 80)
        
        # Get unique patients
        unique_patients = np.unique(patient_ids)
        patient_labels = np.array([y[patient_ids == p][0] for p in unique_patients])
        
        # Split patients (stratified)
        train_patients, test_patients = train_test_split(
            unique_patients, 
            test_size=self.config.TEST_SPLIT,
            stratify=patient_labels,
            random_state=self.config.RANDOM_SEED
        )
        
        train_val_labels = np.array([patient_labels[np.where(unique_patients == p)[0][0]] 
                                     for p in train_patients])
        train_patients, val_patients = train_test_split(
            train_patients,
            test_size=self.config.VAL_SPLIT / (1 - self.config.TEST_SPLIT),
            stratify=train_val_labels,
            random_state=self.config.RANDOM_SEED
        )
        
        # Create indices
        train_idx = np.where(np.isin(patient_ids, train_patients))[0]
        val_idx = np.where(np.isin(patient_ids, val_patients))[0]
        test_idx = np.where(np.isin(patient_ids, test_patients))[0]
        
        print(f"Train: {len(train_idx)} observations ({len(train_patients)} patients)")
        print(f"Val:   {len(val_idx)} observations ({len(val_patients)} patients)")
        print(f"Test:  {len(test_idx)} observations ({len(test_patients)} patients)")
        
        # Build separate graphs for each split (use config's KNN_K)
        train_edge_index = self.build_graph(X[train_idx], k=self.config.KNN_K)
        val_edge_index = self.build_graph(X[val_idx], k=self.config.KNN_K)
        test_edge_index = self.build_graph(X[test_idx], k=self.config.KNN_K)
        
        # Create PyG Data objects
        train_data = self.create_pyg_data(X[train_idx], y[train_idx], train_edge_index)
        val_data = self.create_pyg_data(X[val_idx], y[val_idx], val_edge_index)
        test_data = self.create_pyg_data(X[test_idx], y[test_idx], test_edge_index)
        
        return train_data, val_data, test_data
    
    def initialize_model(self, num_features):
        """Initialize GIMAN-SAA model."""
        print("\n" + "=" * 80)
        print("INITIALIZING MODEL")
        print("=" * 80)
        
        self.model = GIMAN_SAA(
            num_features=num_features,
            hidden_dim=self.config.HIDDEN_DIM,
            num_gat_layers=self.config.NUM_GAT_LAYERS,
            num_heads=self.config.NUM_HEADS,
            dropout=self.config.DROPOUT,
            use_batch_norm=self.config.USE_BATCH_NORM
        ).to(self.device)
        
        # Count parameters
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        print(f"Model: GIMAN-SAA")
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        print(f"\nArchitecture:")
        print(f"  Input features: {num_features}")
        print(f"  Hidden dimension: {self.config.HIDDEN_DIM}")
        print(f"  GAT layers: {self.config.NUM_GAT_LAYERS}")
        print(f"  Attention heads: {self.config.NUM_HEADS}")
        print(f"  Dropout: {self.config.DROPOUT}")
        
        return self.model
    
    def initialize_optimizer(self):
        """Initialize optimizer and scheduler."""
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.LEARNING_RATE,
            weight_decay=self.config.WEIGHT_DECAY
        )
        
        if self.config.USE_SCHEDULER:
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                **self.config.SCHEDULER_PARAMS
            )
        
        print(f"\nOptimizer: AdamW (lr={self.config.LEARNING_RATE}, wd={self.config.WEIGHT_DECAY})")
        if self.config.USE_SCHEDULER:
            print(f"Scheduler: ReduceLROnPlateau")
    
    def compute_loss(self, logits, labels, use_focal=True):
        """
        Compute loss function for imbalanced classification.
        
        Args:
            logits: Model output logits
            labels: True labels
            use_focal: If True, use Focal Loss; else use Weighted BCE
            
        Returns:
            Loss value
        """
        if use_focal:
            # Focal Loss: Better for extreme class imbalance
            # FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
            # Use alpha and gamma from config (can be tuned)
            alpha = getattr(self.config, 'FOCAL_ALPHA', 0.75)
            gamma = getattr(self.config, 'FOCAL_GAMMA', 2.0)
            
            probs = torch.sigmoid(logits.squeeze())
            labels_float = labels.float()
            
            # Compute focal loss
            ce_loss = nn.functional.binary_cross_entropy_with_logits(
                logits.squeeze(), labels_float, reduction='none'
            )
            p_t = probs * labels_float + (1 - probs) * (1 - labels_float)
            focal_weight = (1 - p_t) ** gamma
            
            # Apply alpha weighting
            alpha_t = alpha * labels_float + (1 - alpha) * (1 - labels_float)
            loss = (alpha_t * focal_weight * ce_loss).mean()
        else:
            # Standard weighted BCE
            pos_weight = torch.tensor([self.config.POS_WEIGHT], device=self.device)
            criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
            loss = criterion(logits.squeeze(), labels.float())
        
        return loss
    
    def train_epoch(self, data, epoch=0):
        """Train for one epoch with debugging."""
        self.model.train()
        data = data.to(self.device)
        
        self.optimizer.zero_grad()
        logits = self.model(data.x, data.edge_index)
        loss = self.compute_loss(logits, data.y, use_focal=True)
        loss.backward()
        self.optimizer.step()
        
        # Compute metrics
        with torch.no_grad():
            probs = torch.sigmoid(logits).cpu().numpy()
            labels = data.y.cpu().numpy()
            preds = (probs > 0.5).astype(int)
            
            auc = roc_auc_score(labels, probs)
            acc = accuracy_score(labels, preds)
            
            # Debug output for first few epochs
            if epoch <= 3:
                print(f"    DEBUG epoch {epoch}:")
                print(f"      Logits: min={logits.min():.4f}, max={logits.max():.4f}, mean={logits.mean():.4f}")
                print(f"      Probs:  min={probs.min():.4f}, max={probs.max():.4f}, mean={probs.mean():.4f}")
                print(f"      Preds:  SAA-={( preds==0).sum()}, SAA+={(preds==1).sum()}")
                print(f"      Labels: SAA-={(labels==0).sum()}, SAA+={(labels==1).sum()}")
        
        return loss.item(), auc, acc
    
    def validate(self, data):
        """Validate model."""
        self.model.eval()
        data = data.to(self.device)
        
        with torch.no_grad():
            logits = self.model(data.x, data.edge_index)
            loss = self.compute_loss(logits, data.y)
            
            probs = torch.sigmoid(logits).cpu().numpy()
            labels = data.y.cpu().numpy()
            preds = (probs > 0.5).astype(int)
            
            auc = roc_auc_score(labels, probs)
            acc = accuracy_score(labels, preds)
        
        return loss.item(), auc, acc, probs, preds
    
    def train(self):
        """Main training loop."""
        print("\n" + "=" * 80)
        print("TRAINING GIMAN-SAA MODEL")
        print("=" * 80)
        
        best_val_auc = 0.0
        patience_counter = 0
        
        for epoch in range(self.config.MAX_EPOCHS):
            # Train
            train_loss, train_auc, train_acc = self.train_epoch(self.train_data, epoch=epoch)
            self.train_history['loss'].append(train_loss)
            self.train_history['auc'].append(train_auc)
            self.train_history['acc'].append(train_acc)
            
            # Validate
            val_loss, val_auc, val_acc, _, _ = self.validate(self.val_data)
            self.val_history['loss'].append(val_loss)
            self.val_history['auc'].append(val_auc)
            self.val_history['acc'].append(val_acc)
            
            # Learning rate scheduling
            if self.scheduler is not None:
                self.scheduler.step(val_auc)
            
            # Print progress
            if (epoch + 1) % 5 == 0:
                lr = self.optimizer.param_groups[0]['lr']
                print(f"Epoch {epoch+1:3d}/{self.config.MAX_EPOCHS} | "
                      f"Train Loss: {train_loss:.4f}, AUC: {train_auc:.4f}, Acc: {train_acc:.4f} | "
                      f"Val Loss: {val_loss:.4f}, AUC: {val_auc:.4f}, Acc: {val_acc:.4f} | "
                      f"LR: {lr:.6f}")
            
            # Save best model
            if val_auc > best_val_auc:
                best_val_auc = val_auc
                self.best_val_auc = val_auc
                self.best_epoch = epoch + 1
                patience_counter = 0
                
                # Save model
                model_path = self.config.MODEL_DIR / "best_giman_saa_model.pt"
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_auc': val_auc,
                    'config': self.config
                }, model_path)
            else:
                patience_counter += 1
            
            # Early stopping
            if patience_counter >= self.config.PATIENCE:
                print(f"\nEarly stopping at epoch {epoch+1}")
                print(f"Best validation AUC: {best_val_auc:.4f} at epoch {self.best_epoch}")
                break
        
        print(f"\nTraining complete!")
        print(f"Best validation AUC: {best_val_auc:.4f} at epoch {self.best_epoch}")
    
    def evaluate(self, data, split_name="Test"):
        """Evaluate model on test set."""
        print("\n" + "=" * 80)
        print(f"EVALUATING ON {split_name.upper()} SET")
        print("=" * 80)
        
        # Load best model
        model_path = self.config.MODEL_DIR / "best_giman_saa_model.pt"
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        # Evaluate
        loss, auc, acc, probs, preds = self.validate(data)
        labels = data.y.cpu().numpy()
        
        # Compute additional metrics
        balanced_acc = balanced_accuracy_score(labels, preds)
        
        # Compute metrics at different thresholds
        precision, recall, thresholds = precision_recall_curve(labels, probs)
        f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)
        best_f1_idx = np.argmax(f1_scores)
        best_threshold = thresholds[best_f1_idx] if best_f1_idx < len(thresholds) else 0.5
        
        # Apply best threshold
        preds_best = (probs > best_threshold).astype(int)
        
        # Classification report
        print(f"\nMetrics (threshold=0.5):")
        print(f"  Loss: {loss:.4f}")
        print(f"  AUC: {auc:.4f}")
        print(f"  Accuracy: {acc:.4f}")
        print(f"  Balanced Accuracy: {balanced_acc:.4f}")
        
        print(f"\nClassification Report:")
        print(classification_report(labels, preds, target_names=['SAA-', 'SAA+'], digits=4))
        
        print(f"\nConfusion Matrix:")
        cm = confusion_matrix(labels, preds)
        print(cm)
        
        # Calculate per-class recalls
        tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
        saa_neg_recall = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        saa_pos_recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        
        # Save results
        results = {
            'split': split_name,
            'loss': float(loss),
            'auc': float(auc),
            'accuracy': float(acc),
            'balanced_accuracy': float(balanced_acc),
            'saa_neg_recall': float(saa_neg_recall),
            'saa_pos_recall': float(saa_pos_recall),
            'best_threshold': float(best_threshold),
            'best_epoch': self.best_epoch,
            'confusion_matrix': cm.tolist(),
            'classification_report': classification_report(labels, preds, target_names=['SAA-', 'SAA+'], output_dict=True)
        }
        
        results_path = self.config.RESULTS_DIR / f"{split_name.lower()}_results.json"
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {results_path}")
        
        return results, probs, labels
    
    def plot_training_history(self):
        """Plot training curves."""
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        
        metrics = ['loss', 'auc', 'acc']
        titles = ['Loss', 'AUC-ROC', 'Accuracy']
        
        for ax, metric, title in zip(axes, metrics, titles):
            ax.plot(self.train_history[metric], label='Train', linewidth=2)
            ax.plot(self.val_history[metric], label='Validation', linewidth=2)
            ax.axvline(self.best_epoch - 1, color='red', linestyle='--', alpha=0.5, label='Best')
            ax.set_xlabel('Epoch', fontweight='bold')
            ax.set_ylabel(title, fontweight='bold')
            ax.set_title(f'Training {title}', fontweight='bold')
            ax.legend()
            ax.grid(alpha=0.3)
        
        plt.tight_layout()
        fig_path = self.config.FIGURES_DIR / "training_history.png"
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        print(f"Training history saved to: {fig_path}")
        plt.close()
    
    def plot_roc_curve(self, probs, labels, split_name="Test"):
        """Plot ROC curve."""
        fpr, tpr, _ = roc_curve(labels, probs)
        auc = roc_auc_score(labels, probs)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, linewidth=2, label=f'ROC (AUC = {auc:.3f})')
        plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random')
        plt.xlabel('False Positive Rate', fontweight='bold')
        plt.ylabel('True Positive Rate', fontweight='bold')
        plt.title(f'ROC Curve - {split_name} Set', fontweight='bold')
        plt.legend()
        plt.grid(alpha=0.3)
        
        fig_path = self.config.FIGURES_DIR / f"roc_curve_{split_name.lower()}.png"
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        print(f"ROC curve saved to: {fig_path}")
        plt.close()
    
    def run(self):
        """Execute complete training pipeline."""
        print("\n" + "=" * 80)
        print("PHASE 8.3: GIMAN-SAA TRAINING PIPELINE")
        print("=" * 80)
        print(f"Device: {self.device}")
        print(f"Random seed: {self.config.RANDOM_SEED}")
        
        # Set random seeds
        torch.manual_seed(self.config.RANDOM_SEED)
        np.random.seed(self.config.RANDOM_SEED)
        
        # 1. Load data
        self.load_data()
        
        # 2. Prepare features
        X, y, patient_ids, feature_cols = self.prepare_features()
        
        # 3. Split data
        self.train_data, self.val_data, self.test_data = self.split_data(X, y, patient_ids)
        
        # 4. Initialize model
        self.initialize_model(num_features=X.shape[1])
        self.initialize_optimizer()
        
        # 5. Train
        self.train()
        
        # 6. Plot training history
        self.plot_training_history()
        
        # 7. Evaluate on test set
        test_results, test_probs, test_labels = self.evaluate(self.test_data, "Test")
        
        # 8. Plot ROC curve
        self.plot_roc_curve(test_probs, test_labels, "Test")
        
        print("\n" + "=" * 80)
        print("TRAINING COMPLETE!")
        print("=" * 80)
        print(f"Best validation AUC: {self.best_val_auc:.4f}")
        print(f"Test AUC: {test_results['auc']:.4f}")
        print(f"Test Accuracy: {test_results['accuracy']:.4f}")
        print(f"\nModel saved to: {self.config.MODEL_DIR}")
        print(f"Results saved to: {self.config.RESULTS_DIR}")
        
        return test_results


def main():
    """Main execution function."""
    # Initialize configuration
    config = SAAConfig()
    
    # Create trainer
    trainer = SAATrainer(config)
    
    # Run training pipeline
    results = trainer.run()
    
    return results


if __name__ == "__main__":
    results = main()
