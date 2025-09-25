#!/usr/bin/env python3
"""
Phase 4 Unified System Debug Runner

This script debugs the unified GIMAN system to identify where NaN values are coming from
and fix the issues properly.

Author: GIMAN Development Team
Date: September 24, 2025
"""

import sys
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import logging
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('phase4_debug.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

def check_tensor_health(tensor, name):
    """Check tensor for NaN/inf values and print diagnostics."""
    if torch.isnan(tensor).any():
        logger.error(f"❌ {name} contains NaN values!")
        logger.error(f"   NaN count: {torch.isnan(tensor).sum().item()}")
    if torch.isinf(tensor).any():
        logger.error(f"❌ {name} contains Inf values!")
        logger.error(f"   Inf count: {torch.isinf(tensor).sum().item()}")
    
    logger.info(f"✅ {name}: shape={tensor.shape}, mean={tensor.mean().item():.4f}, std={tensor.std().item():.4f}, min={tensor.min().item():.4f}, max={tensor.max().item():.4f}")
    return torch.isnan(tensor).any() or torch.isinf(tensor).any()

def debug_forward_pass(model, spatial, genomic, temporal, name="Forward"):
    """Debug a forward pass through the model."""
    logger.info(f"🔍 Debugging {name} pass...")
    
    # Check inputs
    has_nan = False
    has_nan |= check_tensor_health(spatial, f"{name}_spatial_input")
    has_nan |= check_tensor_health(genomic, f"{name}_genomic_input") 
    has_nan |= check_tensor_health(temporal, f"{name}_temporal_input")
    
    if has_nan:
        logger.error(f"❌ {name} inputs contain NaN/Inf!")
        return None
    
    model.eval()
    with torch.no_grad():
        # Step through the forward pass
        try:
            # Unified attention
            attention_output = model.unified_attention(spatial, genomic, temporal)
            unified_features = attention_output['unified_features']
            check_tensor_health(unified_features, f"{name}_unified_features")
            
            # Feature processing
            processed_features = model.feature_processor(unified_features)
            check_tensor_health(processed_features, f"{name}_processed_features")
            
            # Feature importance
            importance_weights = model.feature_importance(processed_features)
            check_tensor_health(importance_weights, f"{name}_importance_weights")
            
            weighted_features = processed_features * importance_weights
            check_tensor_health(weighted_features, f"{name}_weighted_features")
            
            # Motor prediction
            motor_output = model.motor_predictor(weighted_features)
            motor_pred = motor_output['ensemble_prediction']
            check_tensor_health(motor_pred, f"{name}_motor_prediction")
            
            # Cognitive prediction
            cognitive_output = model.cognitive_predictor(weighted_features)
            cognitive_pred = model.cognitive_sigmoid(cognitive_output['ensemble_prediction'])
            check_tensor_health(cognitive_pred, f"{name}_cognitive_prediction")
            
            return {
                'motor_prediction': motor_pred,
                'cognitive_prediction': cognitive_pred,
                'unified_features': unified_features,
                'processed_features': processed_features
            }
            
        except Exception as e:
            logger.error(f"❌ Error in {name} forward pass: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return None

def main():
    """Run Phase 4 unified system debug."""
    
    logger.info("🔍 Starting Phase 4 Unified System Debug")
    
    try:
        # Import necessary modules
        from phase3_1_real_data_integration import RealDataPhase3Integration
        from phase4_unified_giman_system import UnifiedGIMANSystem, GIMANResearchAnalyzer
        
        # === PHASE 4 SYSTEM SETUP ===
        logger.info("📊 Loading PPMI data...")
        
        # Setup device first
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"🔧 Using device: {device}")
        
        # Load data using Phase 3.1 integration class
        phase3_integration = RealDataPhase3Integration(device=device)
        prognostic_data = phase3_integration.load_real_ppmi_data()
        phase3_integration.generate_spatiotemporal_embeddings()
        phase3_integration.generate_genomic_embeddings()
        
        # Get embeddings from the integration class
        spatial_emb = phase3_integration.spatiotemporal_embeddings
        genomic_emb = phase3_integration.genomic_embeddings
        
        # Check if embeddings were created successfully
        if spatial_emb is None or genomic_emb is None:
            logger.error("Failed to generate embeddings. Cannot proceed.")
            return None
        
        # Create temporal embeddings with proper initialization
        n_patients = spatial_emb.shape[0]
        temporal_emb = np.random.randn(n_patients, 256).astype(np.float32)
        
        logger.info(f"✅ Loaded data for {n_patients} patients")
        logger.info(f"   Spatial embeddings: {spatial_emb.shape}")
        logger.info(f"   Genomic embeddings: {genomic_emb.shape}")
        logger.info(f"   Temporal embeddings: {temporal_emb.shape}")
        
        # Check for NaN/inf in embeddings
        logger.info("🔍 Checking embeddings for NaN/inf...")
        if np.isnan(spatial_emb).any():
            logger.error(f"❌ Spatial embeddings contain NaN! Count: {np.isnan(spatial_emb).sum()}")
        if np.isinf(spatial_emb).any():
            logger.error(f"❌ Spatial embeddings contain Inf! Count: {np.isinf(spatial_emb).sum()}")
        if np.isnan(genomic_emb).any():
            logger.error(f"❌ Genomic embeddings contain NaN! Count: {np.isnan(genomic_emb).sum()}")
        if np.isinf(genomic_emb).any():
            logger.error(f"❌ Genomic embeddings contain Inf! Count: {np.isinf(genomic_emb).sum()}")
        
        logger.info(f"✅ Spatial: mean={np.mean(spatial_emb):.4f}, std={np.std(spatial_emb):.4f}")
        logger.info(f"✅ Genomic: mean={np.mean(genomic_emb):.4f}, std={np.std(genomic_emb):.4f}")
        logger.info(f"✅ Temporal: mean={np.mean(temporal_emb):.4f}, std={np.std(temporal_emb):.4f}")
        
        # Setup unified system
        embed_dim = spatial_emb.shape[1]  # Should be 256
        
        unified_system = UnifiedGIMANSystem(
            embed_dim=embed_dim,
            num_heads=4,
            dropout=0.3  # Reduce dropout for debugging
        ).to(device)
        
        logger.info("✅ Unified system initialized")
        
        # Initialize weights properly
        def init_weights(m):
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias)
        
        unified_system.apply(init_weights)
        logger.info("✅ Weights initialized")
        
        # === PREPARE DATA ===
        logger.info("🎯 Preparing data...")
        
        # Create realistic synthetic targets
        np.random.seed(42)
        motor_scores = np.random.randn(n_patients).astype(np.float32) * 2.0
        cognitive_conversion = np.random.binomial(1, 0.3, n_patients).astype(np.float32)
        
        # Normalize motor scores
        motor_mean = np.mean(motor_scores)
        motor_std = np.std(motor_scores)
        motor_scores_norm = ((motor_scores - motor_mean) / motor_std).astype(np.float32)
        
        logger.info(f"✅ Motor scores: mean={np.mean(motor_scores_norm):.4f}, std={np.std(motor_scores_norm):.4f}")
        logger.info(f"✅ Cognitive conversion rate: {np.mean(cognitive_conversion):.2f}")
        
        # Train/validation/test split (60/20/20)
        n_train = int(0.6 * n_patients)
        n_val = int(0.2 * n_patients)
        
        indices = np.random.permutation(n_patients)
        train_idx = indices[:n_train]
        val_idx = indices[n_train:n_train+n_val]
        test_idx = indices[n_train+n_val:]
        
        logger.info(f"✅ Data split: {n_train} train, {len(val_idx)} val, {len(test_idx)} test")
        
        # === SINGLE FORWARD PASS DEBUG ===
        logger.info("🔍 Testing single forward pass...")
        
        # Convert to tensors
        spatial_tensor = torch.tensor(spatial_emb[:5], dtype=torch.float32).to(device)  # Small batch
        genomic_tensor = torch.tensor(genomic_emb[:5], dtype=torch.float32).to(device)
        temporal_tensor = torch.tensor(temporal_emb[:5], dtype=torch.float32).to(device)
        
        # Debug forward pass
        debug_result = debug_forward_pass(unified_system, spatial_tensor, genomic_tensor, temporal_tensor, "Debug")
        
        if debug_result is None:
            logger.error("❌ Forward pass failed!")
            return None
        
        logger.info("✅ Single forward pass successful!")
        
        # === TRAINING LOOP DEBUG ===
        logger.info("🎯 Starting training with careful monitoring...")
        
        # Convert all data to tensors
        spatial_tensor_full = torch.tensor(spatial_emb, dtype=torch.float32).to(device)
        genomic_tensor_full = torch.tensor(genomic_emb, dtype=torch.float32).to(device)
        temporal_tensor_full = torch.tensor(temporal_emb, dtype=torch.float32).to(device)
        motor_tensor = torch.tensor(motor_scores_norm, dtype=torch.float32).to(device)
        cognitive_tensor = torch.tensor(cognitive_conversion, dtype=torch.float32).to(device)
        
        # Training setup
        optimizer = torch.optim.AdamW(unified_system.parameters(), lr=0.0001, weight_decay=0.001)  # Lower LR
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)
        
        # Training loop with extensive monitoring
        best_val_loss = float('inf')
        patience = 10
        patience_counter = 0
        
        for epoch in range(50):  # Reduced epochs for debugging
            unified_system.train()
            
            # Training step
            optimizer.zero_grad()
            
            try:
                train_outputs = unified_system(
                    spatial_tensor_full[train_idx],
                    genomic_tensor_full[train_idx],
                    temporal_tensor_full[train_idx]
                )
                
                # Check outputs for NaN
                motor_pred = train_outputs['motor_prediction'].squeeze()
                cognitive_pred = train_outputs['cognitive_prediction'].squeeze()
                
                if torch.isnan(motor_pred).any() or torch.isnan(cognitive_pred).any():
                    logger.error(f"❌ Epoch {epoch}: NaN in predictions!")
                    logger.error(f"   Motor NaN: {torch.isnan(motor_pred).sum().item()}")
                    logger.error(f"   Cognitive NaN: {torch.isnan(cognitive_pred).sum().item()}")
                    break
                
                # Calculate losses with proper handling
                motor_targets = motor_tensor[train_idx]
                cognitive_targets = cognitive_tensor[train_idx]
                
                # Use MSE instead of Huber for debugging
                motor_loss = F.mse_loss(motor_pred, motor_targets)
                cognitive_loss = F.binary_cross_entropy(cognitive_pred, cognitive_targets)
                
                if torch.isnan(motor_loss) or torch.isnan(cognitive_loss):
                    logger.error(f"❌ Epoch {epoch}: NaN in losses!")
                    logger.error(f"   Motor loss: {motor_loss.item()}")
                    logger.error(f"   Cognitive loss: {cognitive_loss.item()}")
                    break
                
                total_loss = motor_loss + cognitive_loss
                
                if torch.isnan(total_loss):
                    logger.error(f"❌ Epoch {epoch}: NaN in total loss!")
                    break
                
                total_loss.backward()
                
                # Check gradients
                grad_norm = torch.nn.utils.clip_grad_norm_(unified_system.parameters(), max_norm=1.0)
                if torch.isnan(grad_norm):
                    logger.error(f"❌ Epoch {epoch}: NaN in gradients!")
                    break
                
                optimizer.step()
                scheduler.step()
                
                # Validation every 5 epochs
                if epoch % 5 == 0:
                    unified_system.eval()
                    with torch.no_grad():
                        val_outputs = unified_system(
                            spatial_tensor_full[val_idx],
                            genomic_tensor_full[val_idx],
                            temporal_tensor_full[val_idx]
                        )
                        
                        val_motor_pred = val_outputs['motor_prediction'].squeeze()
                        val_cognitive_pred = val_outputs['cognitive_prediction'].squeeze()
                        
                        val_motor_loss = F.mse_loss(val_motor_pred, motor_tensor[val_idx])
                        val_cognitive_loss = F.binary_cross_entropy(val_cognitive_pred, cognitive_tensor[val_idx])
                        val_total_loss = val_motor_loss + val_cognitive_loss
                        
                        logger.info(f"Epoch {epoch:3d}: Train Loss = {total_loss:.4f}, Val Loss = {val_total_loss:.4f}")
                        logger.info(f"           Motor: {motor_loss:.4f} -> {val_motor_loss:.4f}, Cognitive: {cognitive_loss:.4f} -> {val_cognitive_loss:.4f}")
                        logger.info(f"           Grad norm: {grad_norm:.4f}")
                        
                        # Early stopping
                        if val_total_loss < best_val_loss:
                            best_val_loss = val_total_loss
                            patience_counter = 0
                        else:
                            patience_counter += 1
                            
                        if patience_counter >= patience:
                            logger.info(f"Early stopping at epoch {epoch}")
                            break
                
            except Exception as e:
                logger.error(f"❌ Error at epoch {epoch}: {str(e)}")
                import traceback
                logger.error(traceback.format_exc())
                break
        
        # === FINAL TESTING ===
        logger.info("🧪 Testing Phase 4 Unified System...")
        
        unified_system.eval()
        with torch.no_grad():
            test_outputs = unified_system(
                spatial_tensor_full[test_idx],
                genomic_tensor_full[test_idx],
                temporal_tensor_full[test_idx]
            )
            
            motor_pred = test_outputs['motor_prediction'].squeeze().cpu().numpy()
            cognitive_pred = test_outputs['cognitive_prediction'].squeeze().cpu().numpy()
            
            motor_true = motor_tensor[test_idx].cpu().numpy()
            cognitive_true = cognitive_tensor[test_idx].cpu().numpy()
        
        # Check for NaN in final predictions
        if np.isnan(motor_pred).any() or np.isnan(cognitive_pred).any():
            logger.error("❌ Final predictions contain NaN!")
            logger.error(f"   Motor NaN: {np.isnan(motor_pred).sum()}")
            logger.error(f"   Cognitive NaN: {np.isnan(cognitive_pred).sum()}")
            return None
        
        # Calculate metrics
        from sklearn.metrics import r2_score, roc_auc_score
        
        motor_r2 = r2_score(motor_true, motor_pred)
        cognitive_auc = roc_auc_score(cognitive_true, cognitive_pred)
        
        logger.info(f"🎯 Phase 4 Debug Results:")
        logger.info(f"   Motor R²: {motor_r2:.4f}")
        logger.info(f"   Cognitive AUC: {cognitive_auc:.4f}")
        
        # Compare to all previous phases
        phase_comparison = {
            'Phase 3.1': {'motor_r2': -0.6481, 'cognitive_auc': 0.4417},
            'Phase 3.2 (Original)': {'motor_r2': -1.4432, 'cognitive_auc': 0.5333},
            'Phase 3.2 (Improved)': {'motor_r2': -0.0760, 'cognitive_auc': 0.7647},
            'Phase 4 (Unified)': {'motor_r2': motor_r2, 'cognitive_auc': cognitive_auc}
        }
        
        logger.info("\n📊 Complete Phase Comparison:")
        for phase, metrics in phase_comparison.items():
            logger.info(f"   {phase}: Motor R² = {metrics['motor_r2']:+.4f}, Cognitive AUC = {metrics['cognitive_auc']:.4f}")
        
        # Determine best performing phase
        best_motor_phase = max(phase_comparison.items(), key=lambda x: x[1]['motor_r2'])
        best_cognitive_phase = max(phase_comparison.items(), key=lambda x: x[1]['cognitive_auc'])
        
        logger.info(f"\n🏆 Best Performance:")
        logger.info(f"   Motor Prediction: {best_motor_phase[0]} (R² = {best_motor_phase[1]['motor_r2']:.4f})")
        logger.info(f"   Cognitive Prediction: {best_cognitive_phase[0]} (AUC = {best_cognitive_phase[1]['cognitive_auc']:.4f})")
        
        training_results = {
            'test_indices': test_idx,
            'test_metrics': {
                'motor_r2': motor_r2,
                'cognitive_auc': cognitive_auc
            },
            'test_predictions': {
                'motor': motor_pred,
                'motor_true': motor_true,
                'cognitive': cognitive_pred,
                'cognitive_true': cognitive_true
            }
        }
        
        logger.info("✅ Phase 4 debugging complete!")
        return training_results
        
    except Exception as e:
        logger.error(f"❌ Error in Phase 4 debug: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        raise


if __name__ == "__main__":
    results = main()