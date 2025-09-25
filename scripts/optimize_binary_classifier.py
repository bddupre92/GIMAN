#!/usr/bin/env python3
"""Hyperparameter Optimization for GIMAN Binary Classifier.

This script uses Optuna to optimize the binary classification performance
(Healthy vs Disease) to achieve >90% AUC-ROC.

Key hyperparameters to optimize:
- Graph structure: k for k-NN graph
- Model architecture: hidden dimensions, dropout
- Training: learning rate, weight decay, focal loss parameters
"""

import logging
import os
import sys
from pathlib import Path
from typing import Dict, Any

import optuna
import torch
from torch_geometric.loader import DataLoader

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent / "src"))

from giman_pipeline.modeling.patient_similarity import PatientSimilarityGraph
from giman_pipeline.training.models import GIMANClassifier
from giman_pipeline.training.trainer import GIMANTrainer
from giman_pipeline.training.experiment_tracker import GIMANExperimentTracker

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger = logging.getLogger(__name__)


def objective(trial):
    """Optuna optimization objective function."""
    try:
        # Sample hyperparameters
        params = {
            "top_k_connections": trial.suggest_int("top_k_connections", 5, 25),
            "similarity_metric": trial.suggest_categorical("similarity_metric", ["euclidean", "cosine"]),
            "hidden_dim_1": trial.suggest_int("hidden_dim_1", 64, 256, step=32),
            "hidden_dim_2": trial.suggest_int("hidden_dim_2", 128, 512, step=64),
            "hidden_dim_3": trial.suggest_int("hidden_dim_3", 32, 128, step=16),
            "dropout_rate": trial.suggest_float("dropout_rate", 0.2, 0.6),
            "learning_rate": trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True),
            "weight_decay": trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True),
            "focal_gamma": trial.suggest_float("focal_gamma", 1.0, 2.5),
            "batch_size": trial.suggest_categorical("batch_size", [16, 32, 64]),
        }
        
        # Create checkpoint directory for this trial
        checkpoint_dir = f"temp_checkpoints/trial_{trial.number}"
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Create similarity graph with trial parameters
        similarity_graph = PatientSimilarityGraph(
            similarity_threshold=None,
            top_k_connections=params["top_k_connections"],
            similarity_metric=params["similarity_metric"],
            random_state=42,
            binary_classification=True,
        )
        
        similarity_graph.load_enhanced_cohort()
        similarity_graph.calculate_patient_similarity(feature_scaling=True)
        similarity_graph.create_similarity_graph()
        
        train_data, val_data, test_data = similarity_graph.split_for_training(
            test_size=0.15,
            val_size=0.15,
            random_state=42,
        )
        
        train_loader = [train_data]
        val_loader = [val_data]
        
        # Create model
        model = GIMANClassifier(
            input_dim=7,
            hidden_dims=[params["hidden_dim_1"], params["hidden_dim_2"], params["hidden_dim_3"]],
            output_dim=2,
            dropout_rate=params["dropout_rate"],
            pooling_method="concat",
            classification_level="node",
        )
        
        # Create trainer
        trainer = GIMANTrainer(
            model=model,
            device="cuda" if torch.cuda.is_available() else "cpu",
            optimizer_name="adamw",
            learning_rate=params["learning_rate"],
            weight_decay=params["weight_decay"],
            scheduler_type="plateau",
            early_stopping_patience=15,
            checkpoint_dir=Path(checkpoint_dir),
            experiment_name=f"trial_{trial.number}",
        )
        
        trainer.setup_focal_loss(
            train_loader,
            alpha=1.0,
            gamma=params["focal_gamma"]
        )
        
        # Train model with limited epochs for optimization
        training_history = trainer.train(
            train_loader=train_loader,
            val_loader=val_loader,
            num_epochs=50,  # Limited epochs for faster optimization
            verbose=False,
        )
        
        # Return validation AUC as objective to maximize
        if 'epochs' in training_history and len(training_history['epochs']) > 0:
            best_val_auc = max([epoch['val_auc_roc'] for epoch in training_history['epochs']])
        else:
            # Fallback to final metrics if epochs not available
            val_results = trainer.evaluate(val_loader)
            best_val_auc = val_results.get('auc_roc', 0.0)
        
        return best_val_auc
        
    except Exception as e:
        logger.error(f"Trial {trial.number} failed: {e}")
        return 0.0  # Return worst possible score on failure


def run_optimization(n_trials: int = 100) -> Dict[str, Any]:
    """Run hyperparameter optimization study.
    
    Args:
        n_trials: Number of optimization trials to run.
        
    Returns:
        Dictionary containing optimization results and best parameters.
    """
    logger.info(f"🚀 Starting hyperparameter optimization with {n_trials} trials")
    
    # Create Optuna study
    study = optuna.create_study(
        direction="maximize",
        study_name="giman_binary_optimization",
        storage=None,  # In-memory storage
    )
    
    # Run optimization
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    
    # Log results
    logger.info("🎉 Optimization completed!")
    logger.info(f"Best AUC: {study.best_value:.4f}")
    logger.info("Best parameters:")
    for key, value in study.best_params.items():
        logger.info(f"  {key}: {value}")
    
    return {
        "best_params": study.best_params,
        "best_value": study.best_value,
        "study": study,
    }


def train_final_model(best_params: Dict[str, Any]) -> Dict[str, Any]:
    """Train final model with optimized parameters.
    
    Args:
        best_params: Best hyperparameters from optimization.
        
    Returns:
        Training results and model performance.
    """
    logger.info("🏆 Training final model with optimized parameters")
    
    # Build configuration with best parameters
    config = {
        # Graph structure
        "top_k_connections": best_params["top_k_connections"],
        "similarity_metric": best_params["similarity_metric"],
        "similarity_threshold": None,
        
        # Model architecture
        "input_dim": 7,
        "hidden_dims": [
            best_params["hidden_dim_1"],
            best_params["hidden_dim_2"], 
            best_params["hidden_dim_3"]
        ],
        "output_dim": 2,
        "dropout_rate": best_params["dropout_rate"],
        "pooling_method": "concat",
        "classification_level": "node",
        
        # Training parameters
        "learning_rate": best_params["learning_rate"],
        "weight_decay": best_params["weight_decay"],
        "focal_alpha": 1.0,
        "focal_gamma": best_params["focal_gamma"],
        "batch_size": best_params["batch_size"],
        "num_epochs": 150,  # Full training
        "early_stopping_patience": 25,
        
        # Fixed parameters
        "test_size": 0.15,
        "val_size": 0.15,
        "random_state": 42,
        "binary_classification": True,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
    }
    
    # Train final model (similar to original training script)
    similarity_graph = PatientSimilarityGraph(
        similarity_threshold=config["similarity_threshold"],
        top_k_connections=config["top_k_connections"],
        similarity_metric=config["similarity_metric"],
        random_state=config["random_state"],
        binary_classification=config["binary_classification"],
    )
    
    similarity_graph.load_enhanced_cohort()
    similarity_graph.calculate_patient_similarity(feature_scaling=True)
    similarity_graph.create_similarity_graph()
    
    train_data, val_data, test_data = similarity_graph.split_for_training(
        test_size=config["test_size"],
        val_size=config["val_size"],
        random_state=config["random_state"],
    )
    
    train_loader = [train_data]
    val_loader = [val_data]
    test_loader = [test_data]
    
    model = GIMANClassifier(
        input_dim=config["input_dim"],
        hidden_dims=config["hidden_dims"],
        output_dim=config["output_dim"],
        dropout_rate=config["dropout_rate"],
        pooling_method=config["pooling_method"],
        classification_level=config["classification_level"],
    )
    
    trainer = GIMANTrainer(
        model=model,
        device=config["device"],
        optimizer_name="adamw",
        learning_rate=config["learning_rate"],
        weight_decay=config["weight_decay"],
        scheduler_type="plateau",
        early_stopping_patience=config["early_stopping_patience"],
        checkpoint_dir=Path("checkpoints/optimized_binary_model"),
        experiment_name="optimized_binary_giman",
    )
    
    trainer.setup_focal_loss(
        train_loader,
        alpha=config["focal_alpha"],
        gamma=config["focal_gamma"]
    )
    
    training_history = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=config["num_epochs"],
        verbose=True,
    )
    
    # Evaluate final model
    test_results = trainer.evaluate(test_loader)
    
    logger.info("🎯 Final Model Results:")
    logger.info(f"  Test AUC: {test_results['auc_roc']:.4f}")
    logger.info(f"  Test Accuracy: {test_results['accuracy']:.4f}")
    logger.info(f"  Test F1: {test_results['f1']:.4f}")
    
    return {
        "config": config,
        "training_history": training_history,
        "test_results": test_results,
    }


if __name__ == "__main__":
    # Run hyperparameter optimization
    optimization_results = run_optimization(n_trials=50)
    
    # Train final optimized model
    final_results = train_final_model(optimization_results["best_params"])
    
    logger.info("✅ Binary classifier optimization completed!")
    logger.info(f"Final AUC: {final_results['test_results']['auc_roc']:.4f}")