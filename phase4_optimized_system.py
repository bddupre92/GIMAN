#!/usr/bin/env python3
"""
Phase 4 Optimized GIMAN System
==============================

Optimized version based on analysis of the enhanced system.
Key improvements:
- Simpler architecture to prevent overfitting
- Stronger regularization
- Better training stability
- Focus on generalization over complexity

Author: GIMAN Development Team
Date: September 24, 2025
"""

import os
import sys
import logging
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, TensorDataset
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import r2_score, roc_auc_score, mean_squared_error
from sklearn.preprocessing import StandardScaler, RobustScaler

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class OptimizedConfig:
    """Optimized configuration focusing on generalization."""
    # Model architecture - Simplified
    embed_dim: int = 128
    num_heads: int = 4
    dropout_rate: float = 0.5
    
    # Training parameters - Conservative
    learning_rate: float = 0.0001
    weight_decay: float = 1e-3
    gradient_clip_value: float = 0.5
    
    # Advanced training - More aggressive regularization
    lr_scheduler_patience: int = 5
    lr_scheduler_factor: float = 0.3
    early_stopping_patience: int = 15
    warmup_epochs: int = 3
    
    # Batch and regularization
    batch_size: int = 32
    label_smoothing: float = 0.0
    
    # Cross-validation
    n_folds: int = 5
    
    # Hardware
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'


class SimpleAttentionModule(nn.Module):
    """Simplified attention module focusing on interpretability."""
    
    def __init__(self, input_dim: int, embed_dim: int, dropout: float = 0.5):
        super().__init__()
        self.embed_dim = embed_dim
        
        # Project input features to desired embedding dimension
        self.input_projection = nn.Linear(input_dim, embed_dim)
        
        # Simple feature processing
        self.feature_norm = nn.LayerNorm(embed_dim)
        
        # Attention weights for each modality
        self.attention_weights = nn.Parameter(torch.ones(3) / 3.0)
        
        # Simple fusion
        self.fusion = nn.Sequential(
            nn.Linear(embed_dim * 3, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
    def forward(self, spatial_features, genomic_features, temporal_features):
        """Simple attention-based fusion."""
        # Project to desired embedding dimension
        spatial_proj = self.input_projection(spatial_features)
        genomic_proj = self.input_projection(genomic_features)
        temporal_proj = self.input_projection(temporal_features)
        
        # Normalize features
        spatial_norm = self.feature_norm(spatial_proj)
        genomic_norm = self.feature_norm(genomic_proj)
        temporal_norm = self.feature_norm(temporal_proj)
        
        # Apply learned attention weights
        attention_weights = F.softmax(self.attention_weights, dim=0)
        
        weighted_spatial = spatial_norm * attention_weights[0]
        weighted_genomic = genomic_norm * attention_weights[1]
        weighted_temporal = temporal_norm * attention_weights[2]
        
        # Concatenate and fuse
        combined = torch.cat([weighted_spatial, weighted_genomic, weighted_temporal], dim=1)
        fused_features = self.fusion(combined)
        
        return fused_features, attention_weights


class RobustPredictor(nn.Module):
    """Robust predictor with strong regularization."""
    
    def __init__(self, input_dim: int, dropout: float = 0.5):
        super().__init__()
        
        # Simplified architecture
        self.predictor = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.LayerNorm(input_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(input_dim // 2, input_dim // 4),
            nn.LayerNorm(input_dim // 4),
            nn.ReLU(),
            nn.Dropout(dropout * 0.7),
            nn.Linear(input_dim // 4, 1)
        )
        
    def forward(self, features):
        return self.predictor(features)


class OptimizedGIMANSystem(nn.Module):
    """Optimized GIMAN system focusing on generalization."""
    
    def __init__(self, config: OptimizedConfig):
        super().__init__()
        
        self.config = config
        self.embed_dim = config.embed_dim
        
        # Simple attention module (input_dim=256, embed_dim=128)
        self.attention_module = SimpleAttentionModule(256, config.embed_dim, config.dropout_rate)
        
        # Separate predictors for each task
        self.motor_predictor = RobustPredictor(config.embed_dim, config.dropout_rate)
        self.cognitive_predictor = RobustPredictor(config.embed_dim, config.dropout_rate)
        
        # Add sigmoid for cognitive predictions
        self.sigmoid = nn.Sigmoid()
        
        # Initialize weights
        self._initialize_weights()
        
    def _initialize_weights(self):
        """Conservative weight initialization."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight, gain=0.5)  # Reduced gain
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
        
    def forward(self, spatial_emb: torch.Tensor, genomic_emb: torch.Tensor, 
                temporal_emb: torch.Tensor, return_attention: bool = False):
        """Forward pass with optional attention weights."""
        
        # Attention-based fusion
        fused_features, attention_weights = self.attention_module(
            spatial_emb, genomic_emb, temporal_emb
        )
        
        # Make predictions
        motor_pred = self.motor_predictor(fused_features)
        cognitive_logits = self.cognitive_predictor(fused_features)
        cognitive_pred = self.sigmoid(cognitive_logits)
        
        if return_attention:
            return motor_pred, cognitive_pred, fused_features, attention_weights
        
        return motor_pred, cognitive_pred


class OptimizedTrainer:
    """Optimized trainer with robust training strategies."""
    
    def __init__(self, config: OptimizedConfig):
        self.config = config
        self.device = torch.device(config.device)
        
        logger.info(f"🚀 Optimized GIMAN System initialized on {self.device}")
        
    def prepare_data_robust(self, spatial_embeddings: np.ndarray, genomic_embeddings: np.ndarray,
                           temporal_embeddings: np.ndarray, motor_scores: np.ndarray, 
                           cognitive_labels: np.ndarray):
        """Robust data preparation with proper scaling."""
        
        # Robust scaling for better generalization
        spatial_scaler = RobustScaler()
        genomic_scaler = RobustScaler()
        temporal_scaler = RobustScaler()
        motor_scaler = RobustScaler()
        
        spatial_scaled = spatial_scaler.fit_transform(spatial_embeddings)
        genomic_scaled = genomic_scaler.fit_transform(genomic_embeddings)
        temporal_scaled = temporal_scaler.fit_transform(temporal_embeddings)
        motor_scaled = motor_scaler.fit_transform(motor_scores.reshape(-1, 1)).flatten()
        
        logger.info("✅ Applied robust scaling to all features")
        logger.info(f"   Spatial: mean={spatial_scaled.mean():.4f}, std={spatial_scaled.std():.4f}")
        logger.info(f"   Genomic: mean={genomic_scaled.mean():.4f}, std={genomic_scaled.std():.4f}")
        logger.info(f"   Temporal: mean={temporal_scaled.mean():.4f}, std={temporal_scaled.std():.4f}")
        logger.info(f"   Motor: mean={motor_scaled.mean():.4f}, std={motor_scaled.std():.4f}")
        
        # Class balance analysis
        positive_ratio = cognitive_labels.mean()
        logger.info(f"   Cognitive class balance: {positive_ratio:.3f} positive, {1-positive_ratio:.3f} negative")
        
        return (spatial_scaled, genomic_scaled, temporal_scaled, 
                motor_scaled, cognitive_labels, motor_scaler)
    
    def cross_validate(self, spatial_embeddings, genomic_embeddings, temporal_embeddings,
                      motor_scores, cognitive_labels):
        """Cross-validation for robust evaluation."""
        
        logger.info(f"🔄 Starting {self.config.n_folds}-fold cross-validation...")
        
        # Prepare data
        spatial_scaled, genomic_scaled, temporal_scaled, motor_scaled, cognitive_labels, motor_scaler = \
            self.prepare_data_robust(spatial_embeddings, genomic_embeddings, temporal_embeddings,
                                   motor_scores, cognitive_labels)
        
        # Stratified K-fold for balanced splits
        skf = StratifiedKFold(n_splits=self.config.n_folds, shuffle=True, random_state=42)
        
        cv_results = {
            'motor_r2_scores': [],
            'cognitive_auc_scores': [],
            'fold_histories': [],
            'attention_weights': []
        }
        
        for fold, (train_idx, val_idx) in enumerate(skf.split(spatial_scaled, cognitive_labels)):
            logger.info(f"\n📊 Fold {fold + 1}/{self.config.n_folds}")
            
            # Split data
            X_train = (spatial_scaled[train_idx], genomic_scaled[train_idx], temporal_scaled[train_idx])
            X_val = (spatial_scaled[val_idx], genomic_scaled[val_idx], temporal_scaled[val_idx])
            y_train = (motor_scaled[train_idx], cognitive_labels[train_idx])
            y_val = (motor_scaled[val_idx], cognitive_labels[val_idx])
            
            # Create data loaders
            train_loader = self._create_dataloader(X_train, y_train, shuffle=True)
            val_loader = self._create_dataloader(X_val, y_val, shuffle=False)
            
            # Initialize model for this fold
            model = OptimizedGIMANSystem(self.config).to(self.device)
            
            # Train model
            history = self._train_fold(model, train_loader, val_loader, fold)
            
            # Evaluate model
            motor_r2, cognitive_auc, attention_weights = self._evaluate_fold(model, val_loader, motor_scaler)
            
            # Store results
            cv_results['motor_r2_scores'].append(motor_r2)
            cv_results['cognitive_auc_scores'].append(cognitive_auc)
            cv_results['fold_histories'].append(history)
            cv_results['attention_weights'].append(attention_weights.cpu().numpy())
            
            logger.info(f"   Fold {fold + 1} Results: Motor R² = {motor_r2:.4f}, Cognitive AUC = {cognitive_auc:.4f}")
        
        # Compute cross-validation statistics
        motor_mean, motor_std = np.mean(cv_results['motor_r2_scores']), np.std(cv_results['motor_r2_scores'])
        cognitive_mean, cognitive_std = np.mean(cv_results['cognitive_auc_scores']), np.std(cv_results['cognitive_auc_scores'])
        
        logger.info(f"\n🎯 Cross-Validation Results:")
        logger.info(f"   Motor R²: {motor_mean:.4f} ± {motor_std:.4f}")
        logger.info(f"   Cognitive AUC: {cognitive_mean:.4f} ± {cognitive_std:.4f}")
        
        # Analyze attention weights consistency
        attention_weights_array = np.array(cv_results['attention_weights'])
        attention_mean = attention_weights_array.mean(axis=0)
        attention_std = attention_weights_array.std(axis=0)
        
        logger.info(f"\n🎯 Attention Weights Consistency:")
        modalities = ['Spatial', 'Genomic', 'Temporal']
        for i, modality in enumerate(modalities):
            logger.info(f"   {modality}: {attention_mean[i]:.3f} ± {attention_std[i]:.3f}")
        
        cv_results['motor_mean'] = motor_mean
        cv_results['motor_std'] = motor_std
        cv_results['cognitive_mean'] = cognitive_mean
        cv_results['cognitive_std'] = cognitive_std
        cv_results['attention_mean'] = attention_mean
        cv_results['attention_std'] = attention_std
        
        return cv_results
    
    def _create_dataloader(self, X, y, shuffle=True):
        """Create dataloader from data."""
        dataset = TensorDataset(
            torch.FloatTensor(X[0]),  # spatial
            torch.FloatTensor(X[1]),  # genomic
            torch.FloatTensor(X[2]),  # temporal
            torch.FloatTensor(y[0]),  # motor
            torch.FloatTensor(y[1])   # cognitive
        )
        return DataLoader(dataset, batch_size=self.config.batch_size, shuffle=shuffle)
    
    def _train_fold(self, model, train_loader, val_loader, fold):
        """Train model for one fold."""
        
        # Initialize optimizer and scheduler
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay
        )
        
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=self.config.lr_scheduler_factor,
            patience=self.config.lr_scheduler_patience,
            verbose=False
        )
        
        # Loss functions
        mse_loss = nn.MSELoss()
        bce_loss = nn.BCELoss()
        
        # Training history
        history = {'train_losses': [], 'val_losses': []}
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(100):  # Max epochs
            # Training
            model.train()
            train_loss = 0.0
            
            for spatial, genomic, temporal, motor_scores, cognitive_labels in train_loader:
                spatial, genomic, temporal = spatial.to(self.device), genomic.to(self.device), temporal.to(self.device)
                motor_scores, cognitive_labels = motor_scores.to(self.device), cognitive_labels.to(self.device)
                
                optimizer.zero_grad()
                
                motor_pred, cognitive_pred = model(spatial, genomic, temporal)
                
                motor_loss = mse_loss(motor_pred.squeeze(), motor_scores)
                cognitive_loss = bce_loss(cognitive_pred.squeeze(), cognitive_labels)
                
                # Balanced loss weighting
                total_loss = 0.5 * motor_loss + 0.5 * cognitive_loss
                
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), self.config.gradient_clip_value)
                optimizer.step()
                
                train_loss += total_loss.item()
            
            # Validation
            model.eval()
            val_loss = 0.0
            
            with torch.no_grad():
                for spatial, genomic, temporal, motor_scores, cognitive_labels in val_loader:
                    spatial, genomic, temporal = spatial.to(self.device), genomic.to(self.device), temporal.to(self.device)
                    motor_scores, cognitive_labels = motor_scores.to(self.device), cognitive_labels.to(self.device)
                    
                    motor_pred, cognitive_pred = model(spatial, genomic, temporal)
                    
                    motor_loss = mse_loss(motor_pred.squeeze(), motor_scores)
                    cognitive_loss = bce_loss(cognitive_pred.squeeze(), cognitive_labels)
                    total_loss = 0.5 * motor_loss + 0.5 * cognitive_loss
                    
                    val_loss += total_loss.item()
            
            # Record history
            avg_train_loss = train_loss / len(train_loader)
            avg_val_loss = val_loss / len(val_loader)
            history['train_losses'].append(avg_train_loss)
            history['val_losses'].append(avg_val_loss)
            
            # Scheduler step
            scheduler.step(avg_val_loss)
            
            # Early stopping
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
            else:
                patience_counter += 1
            
            if patience_counter >= self.config.early_stopping_patience:
                break
        
        return history
    
    def _evaluate_fold(self, model, val_loader, motor_scaler):
        """Evaluate model for one fold."""
        model.eval()
        
        all_motor_preds = []
        all_motor_true = []
        all_cognitive_preds = []
        all_cognitive_true = []
        attention_weights_sum = None
        n_batches = 0
        
        with torch.no_grad():
            for spatial, genomic, temporal, motor_scores, cognitive_labels in val_loader:
                spatial, genomic, temporal = spatial.to(self.device), genomic.to(self.device), temporal.to(self.device)
                
                motor_pred, cognitive_pred, _, attention_weights = model(
                    spatial, genomic, temporal, return_attention=True
                )
                
                all_motor_preds.extend(motor_pred.cpu().numpy().flatten())
                all_motor_true.extend(motor_scores.numpy().flatten())
                all_cognitive_preds.extend(cognitive_pred.cpu().numpy().flatten())
                all_cognitive_true.extend(cognitive_labels.numpy().flatten())
                
                # Accumulate attention weights
                if attention_weights_sum is None:
                    attention_weights_sum = attention_weights
                else:
                    attention_weights_sum += attention_weights
                n_batches += 1
        
        # Inverse transform motor predictions and targets
        all_motor_preds = motor_scaler.inverse_transform(np.array(all_motor_preds).reshape(-1, 1)).flatten()
        all_motor_true = motor_scaler.inverse_transform(np.array(all_motor_true).reshape(-1, 1)).flatten()
        
        # Calculate metrics
        motor_r2 = r2_score(all_motor_true, all_motor_preds)
        
        # Handle potential issues with AUC calculation
        try:
            cognitive_auc = roc_auc_score(all_cognitive_true, all_cognitive_preds)
        except ValueError:
            cognitive_auc = 0.5  # Random performance if calculation fails
        
        # Average attention weights
        avg_attention_weights = attention_weights_sum / n_batches
        
        return motor_r2, cognitive_auc, avg_attention_weights


def main():
    """Main function for optimized Phase 4 system."""
    
    logger.info("🌟 Optimized Phase 4 GIMAN System")
    logger.info("=" * 50)
    
    # Initialize optimized configuration
    config = OptimizedConfig()
    logger.info(f"📋 Optimized Configuration:")
    for key, value in config.__dict__.items():
        logger.info(f"   {key}: {value}")
    
    # Load data
    from phase3_1_real_data_integration import RealDataPhase3Integration
    
    integrator = RealDataPhase3Integration()
    integrator.load_real_ppmi_data()
    integrator.generate_spatiotemporal_embeddings()
    integrator.generate_genomic_embeddings()
    integrator.load_prognostic_targets()
    
    logger.info(f"📊 Loaded data for {len(integrator.patient_ids)} patients")
    
    # Prepare synthetic temporal embeddings (match spatial/genomic dimensions)
    np.random.seed(42)
    temporal_embeddings = np.random.randn(len(integrator.patient_ids), 256).astype(np.float32)
    temporal_embeddings = (temporal_embeddings - temporal_embeddings.mean()) / temporal_embeddings.std()
    
    # Initialize trainer
    trainer = OptimizedTrainer(config)
    
    # Run cross-validation
    cv_results = trainer.cross_validate(
        integrator.spatiotemporal_embeddings,
        integrator.genomic_embeddings,
        temporal_embeddings,
        integrator.prognostic_targets[:, 0],  # Motor scores
        integrator.prognostic_targets[:, 1]   # Cognitive labels
    )
    
    # Save results
    results_summary = {
        'config': config.__dict__,
        'cv_results': cv_results,
        'data_info': {
            'n_patients': len(integrator.patient_ids),
            'spatial_shape': integrator.spatiotemporal_embeddings.shape,
            'genomic_shape': integrator.genomic_embeddings.shape,
            'temporal_shape': temporal_embeddings.shape
        }
    }
    
    torch.save(results_summary, 'optimized_phase4_results.pth', _use_new_zipfile_serialization=False)
    logger.info("💾 Results saved to 'optimized_phase4_results.pth'")
    
    logger.info("✅ Optimized Phase 4 system completed successfully!")
    
    return trainer, cv_results


if __name__ == "__main__":
    trainer, results = main()