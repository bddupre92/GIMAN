#!/usr/bin/env python3
"""
Enhanced GIMAN v1.1.0 Training Script
=====================================

This script trains GIMAN v1.1.0 with the enhanced 12-feature dataset,
building upon the production v1.0.0 model (98.93% AUC-ROC) architecture.

Enhanced Features (12 total):
- Current 7: LRRK2, GBA, APOE_RISK, PTAU, TTAU, UPSIT_TOTAL, ALPHA_SYN
- Enhanced +5: AGE_COMPUTED, NHY, SEX, NP3TOT, HAS_DATSCAN

Architecture:
- Uses optimal configuration from v1.0.0 (98.93% AUC-ROC)
- Updates input dimension from 7 to 12 features
- Preserves proven architecture: [96, 256, 64] hidden layers
- Maintains k=6 cosine similarity graph structure
- Uses Focal Loss with optimal parameters (α=1.0, γ=2.09)

Author: GIMAN Enhancement Team
Date: September 24, 2025
Version: 1.1.0
"""

import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Any

import torch
import torch.nn.functional as F
from torch_geometric.data import Data

# Import from parent package - now inside training folder
sys.path.append(str(Path(__file__).parent.parent.parent))  # Add project root

from configs.optimal_binary_config import get_optimal_config
from .models import GIMANClassifier
from .trainer import GIMANTrainer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(
            f"giman_enhanced_v1.1.0_training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        ),
    ],
)
logger = logging.getLogger(__name__)


def load_enhanced_graph_data() -> Data:
    """Load the enhanced 12-feature graph data."""
    logger.info("📥 Loading enhanced 12-feature graph data...")
    
    graph_path = Path("data/enhanced/enhanced_graph_data_latest.pth")
    
    if not graph_path.exists():
        raise FileNotFoundError(
            f"Enhanced graph data not found at {graph_path}. "
            f"Please run scripts/create_enhanced_dataset_v2.py first."
        )
    
    # Load with proper torch_geometric imports
    torch.serialization.add_safe_globals([Data])
    graph_data = torch.load(graph_path, weights_only=False)
    
    logger.info(f"✅ Enhanced graph data loaded:")
    logger.info(f"   📊 Nodes: {graph_data.num_nodes}")
    logger.info(f"   📊 Edges: {graph_data.num_edges}")
    logger.info(f"   📊 Features: {graph_data.x.shape[1]} (enhanced from 7 to 12)")
    logger.info(f"   📊 Feature names: {graph_data.feature_names}")
    
    return graph_data


def create_enhanced_config() -> Dict[str, Any]:
    """Create enhanced configuration based on optimal v1.0.0 config."""
    # Start with optimal configuration from v1.0.0
    base_config = get_optimal_config()
    
    # Create enhanced configuration
    enhanced_config = base_config.copy()
    
    # Update for enhanced features
    enhanced_config["model_params"]["input_dim"] = 12  # 7 → 12 features
    enhanced_config["model_params"]["feature_names"] = [
        "LRRK2", "GBA", "APOE_RISK", "PTAU", "TTAU", "UPSIT_TOTAL", "ALPHA_SYN",  # Current 7
        "AGE_COMPUTED", "NHY", "SEX", "NP3TOT", "HAS_DATSCAN"  # Enhanced +5
    ]
    
    # Update version info
    enhanced_config["version"] = "1.1.0"
    enhanced_config["description"] = "Enhanced 12-feature GIMAN model"
    enhanced_config["base_model"] = "v1.0.0 (98.93% AUC-ROC)"
    
    # Keep optimal hyperparameters but allow for potential improvement
    enhanced_config["training_params"]["max_epochs"] = 150  # More epochs for new features
    enhanced_config["training_params"]["patience"] = 15     # Slightly more patience
    
    logger.info("🔧 Enhanced Configuration Created:")
    logger.info(f"   📊 Input Features: 7 → 12 ({enhanced_config['model_params']['input_dim']})")
    logger.info(f"   🏗️  Architecture: {enhanced_config['model_params']['hidden_dims']}")
    logger.info(f"   🔗 Graph: k={enhanced_config['graph_params']['top_k_connections']} cosine similarity")
    logger.info(f"   🎯 Loss: Focal(α={enhanced_config['loss_params']['focal_alpha']}, γ={enhanced_config['loss_params']['focal_gamma']})")
    logger.info(f"   ⚡ Max Epochs: {enhanced_config['training_params']['max_epochs']}")
    
    return enhanced_config


def create_data_splits(graph_data: Data, config: Dict[str, Any]) -> tuple[Data, Data, Data]:
    """Create train/val/test splits from enhanced graph data."""
    logger.info("🔄 Creating train/validation/test splits...")
    
    num_nodes = graph_data.num_nodes
    train_ratio = config["data_params"]["train_ratio"]
    val_ratio = config["data_params"]["val_ratio"] 
    test_ratio = config["data_params"]["test_ratio"]
    
    # Create random split indices
    torch.manual_seed(config["data_params"]["random_state"])
    indices = torch.randperm(num_nodes)
    
    train_size = int(train_ratio * num_nodes)
    val_size = int(val_ratio * num_nodes)
    
    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)
    
    train_mask[indices[:train_size]] = True
    val_mask[indices[train_size:train_size + val_size]] = True
    test_mask[indices[train_size + val_size:]] = True
    
    # Create split data objects
    train_data = graph_data.clone()
    train_data.train_mask = train_mask
    train_data.val_mask = val_mask
    train_data.test_mask = test_mask
    
    val_data = train_data.clone()
    test_data = train_data.clone()
    
    logger.info(f"✅ Data splits created:")
    logger.info(f"   📊 Train: {train_mask.sum()} nodes ({train_ratio:.1%})")
    logger.info(f"   📊 Val: {val_mask.sum()} nodes ({val_ratio:.1%})")
    logger.info(f"   📊 Test: {test_mask.sum()} nodes ({test_ratio:.1%})")
    
    return train_data, val_data, test_data


def train_enhanced_model() -> Dict[str, Any]:
    """Train the enhanced GIMAN v1.1.0 model."""
    logger.info("🚀 GIMAN Enhanced v1.1.0 Training Started")
    logger.info("=" * 80)
    
    try:
        # Load enhanced configuration
        config = create_enhanced_config()
        
        # Load enhanced graph data
        graph_data = load_enhanced_graph_data()
        
        # Create data splits
        train_data, val_data, test_data = create_data_splits(graph_data, config)
        
        # Initialize model with enhanced input dimension
        logger.info("🧠 Initializing Enhanced GIMAN model...")
        model = GIMANClassifier(
            input_dim=config["model_params"]["input_dim"],  # 12 features
            hidden_dims=config["model_params"]["hidden_dims"],
            output_dim=config["model_params"]["num_classes"],
            dropout_rate=config["model_params"]["dropout_rate"],
        )
        
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        logger.info(f"✅ Model initialized:")
        logger.info(f"   📊 Input Dimension: {config['model_params']['input_dim']} features")
        logger.info(f"   📊 Hidden Dimensions: {config['model_params']['hidden_dims']}")
        logger.info(f"   📊 Total Parameters: {total_params:,}")
        logger.info(f"   📊 Trainable Parameters: {trainable_params:,}")
        
        # Initialize trainer
        logger.info("🏋️  Initializing Enhanced GIMAN trainer...")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        trainer = GIMANTrainer(
            model=model,
            device=str(device),
            optimizer_name=config["training_params"]["optimizer"],
            learning_rate=config["training_params"]["learning_rate"],
            weight_decay=config["training_params"]["weight_decay"],
            scheduler_type=config["training_params"]["scheduler"],
            early_stopping_patience=config["training_params"]["patience"],
            experiment_name="giman_enhanced_v1.1.0"
        )
        
        # Note: Experiment tracking handled by trainer's built-in logging
        
        # Set up Focal Loss (as used in optimal config)
        from torch_geometric.loader import DataLoader
        
        # Create data loaders - for node-level tasks, we use single graph in loader
        train_loader = [train_data]  # Single graph for node classification
        val_loader = [val_data]     # Single graph for node classification
        
        # Setup focal loss with optimal parameters
        trainer.setup_focal_loss(
            train_loader=train_loader,
            alpha=config["loss_params"]["focal_alpha"],
            gamma=config["loss_params"]["focal_gamma"]
        )
        
        # Start training
        logger.info("🎯 Starting enhanced model training...")
        train_results = trainer.train(
            train_loader=train_loader,
            val_loader=val_loader,
            num_epochs=config["training_params"]["max_epochs"],
            verbose=True
        )
        
        # Evaluate on test set
        logger.info("📊 Evaluating enhanced model on test set...")
        test_loader = [test_data]  # Single graph for node classification
        test_results = trainer.evaluate(test_loader)
        
        # Save enhanced model
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_dir = Path(f"models/registry/giman_enhanced_v1.1.0_{timestamp}")
        save_dir.mkdir(parents=True, exist_ok=True)
        
        model_path = save_dir / "model.pth"
        config_path = save_dir / "config.json"
        results_path = save_dir / "results.json"
        graph_path = save_dir / "graph_data.pth"
        
        # Save model and components
        torch.save({
            'model_state_dict': trainer.model.state_dict(),
            'model_config': config["model_params"],
            'training_config': config["training_params"],
            'feature_names': config["model_params"]["feature_names"],
            'version': '1.1.0',
            'base_model': 'v1.0.0 (98.93% AUC-ROC)',
            'enhancement': '12-feature enhanced'
        }, model_path)
        
        # Save enhanced graph data for inference
        torch.save(graph_data, graph_path)
        
        # Save configuration
        import json
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2, default=str)
            
        # Compile results
        final_results = {
            'version': '1.1.0',
            'base_model': 'v1.0.0 (98.93% AUC-ROC)',
            'enhancement': '12-feature dataset',
            'train_results': train_results,
            'test_results': test_results,
            'model_params': {
                'total_parameters': total_params,
                'trainable_parameters': trainable_params,
                'input_features': config["model_params"]["input_dim"],
                'feature_names': config["model_params"]["feature_names"]
            },
            'save_path': str(save_dir),
            'timestamp': timestamp
        }
        
        # Save results
        with open(results_path, 'w') as f:
            json.dump(final_results, f, indent=2, default=str)
        
        # Create latest symlink
        latest_dir = Path("models/registry/giman_enhanced_latest")
        if latest_dir.exists():
            latest_dir.unlink()
        latest_dir.symlink_to(save_dir.name)
        
        logger.info("🎉 Enhanced GIMAN v1.1.0 Training Completed!")
        logger.info("=" * 80)
        logger.info(f"📊 Test AUC-ROC: {test_results['auc_roc']:.4f}")
        logger.info(f"📊 Test Accuracy: {test_results['accuracy']:.4f}")
        logger.info(f"📊 Test F1 Score: {test_results['f1']:.4f}")
        logger.info(f"📊 Improvement from v1.0.0: {test_results['auc_roc'] - 0.9893:+.4f} AUC-ROC")
        logger.info(f"💾 Model saved to: {save_dir}")
        logger.info("=" * 80)
        
        return final_results
        
    except Exception as e:
        logger.error(f"❌ Enhanced training failed: {e}")
        raise


def main():
    """Main execution function."""
    try:
        results = train_enhanced_model()
        
        print("\n" + "="*60)
        print("🏆 GIMAN ENHANCED v1.1.0 TRAINING COMPLETED!")
        print("="*60)
        print(f"📊 Test AUC-ROC: {results['test_results']['auc_roc']:.2%}")
        print(f"📊 Test Accuracy: {results['test_results']['accuracy']:.2%}")
        print(f"📊 Test F1 Score: {results['test_results']['f1']:.2%}")
        print(f"📊 Features: 7 → 12 (enhanced)")
        print(f"💾 Model Location: {results['save_path']}")
        print("="*60)
        
        return results
        
    except Exception as e:
        print(f"\n❌ Enhanced training failed: {e}")
        return None


if __name__ == "__main__":
    main()