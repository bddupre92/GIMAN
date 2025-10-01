#!/usr/bin/env python3
"""Complete GIMAN Training Script with Consolidated Data Pipeline.

This script runs the complete GIMAN training pipeline using:
1. PatientSimilarityGraph for graph construction and data loading
2. GIMANClassifier for the model architecture
3. GIMANTrainer for training management
4. Comprehensive evaluation and validation

The script trains on the complete 557-patient dataset with 100% biomarker completeness.
"""

import logging
import sys
from datetime import datetime
from pathlib import Path

import torch

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent / "src"))

from giman_pipeline.modeling.patient_similarity import PatientSimilarityGraph
from giman_pipeline.training.models import GIMANClassifier
from giman_pipeline.training.trainer import GIMANTrainer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(
            f"giman_training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        ),
    ],
)
logger = logging.getLogger(__name__)


def main():
    """Main training pipeline execution."""
    logger.info("🚀 Starting GIMAN Complete Training Pipeline")
    logger.info("=" * 80)

    # Configuration
    config = {
        # Data parameters
        "similarity_threshold": 0.3,
        "top_k_connections": 10,
        "similarity_metric": "cosine",
        "test_size": 0.15,
        "val_size": 0.15,
        "random_state": 42,
        # Model parameters
        "input_dim": 7,
        "hidden_dims": [128, 256, 128],  # Increased capacity for better learning
        "output_dim": 2,  # Binary: Healthy vs Disease (for >90% performance)
        "dropout_rate": 0.4,  # Slightly higher dropout for regularization
        "pooling_method": "concat",
        "classification_level": "node",  # Node-level classification for patient predictions
        # Training parameters
        "batch_size": 32,
        "num_epochs": 100,
        "learning_rate": 0.001,
        "weight_decay": 1e-5,
        "optimizer": "adamw",
        "scheduler_type": "plateau",
        "early_stopping_patience": 15,
        # Class balancing parameters
        "loss_function": "focal",  # Use Focal Loss for severe class imbalance
        "focal_alpha": 1.0,  # Focal loss alpha parameter
        "focal_gamma": 1.8,  # Moderate increase for harder examples
        "label_smoothing": 0.1,  # Label smoothing factor (10% smoothing)
        # Training adjustments for imbalanced data
        "learning_rate": 0.0005,  # Slightly higher learning rate for faster convergence
        "early_stopping_patience": 25,  # Reduce patience for faster training
        "num_epochs": 150,  # More epochs to allow learning minority classes
        # Device
        "device": "cuda" if torch.cuda.is_available() else "cpu",
    }

    logger.info("Training Configuration:")
    for key, value in config.items():
        logger.info(f"  {key}: {value}")
    logger.info("=" * 80)

    try:
        # Step 1: Data Loading and Graph Construction
        logger.info("📊 Step 1: Loading data and constructing similarity graph...")

        similarity_graph = PatientSimilarityGraph(
            similarity_threshold=config["similarity_threshold"],
            top_k_connections=config["top_k_connections"],
            similarity_metric=config["similarity_metric"],
            random_state=config["random_state"],
            binary_classification=True,  # Enable binary classification for >90% performance
        )

        # Load data and build graph
        similarity_graph.load_enhanced_cohort()
        similarity_graph.calculate_patient_similarity(feature_scaling=True)
        similarity_graph.create_similarity_graph()

        # Split data for training
        train_data, val_data, test_data = similarity_graph.split_for_training(
            test_size=config["test_size"],
            val_size=config["val_size"],
            random_state=config["random_state"],
        )

        # Create data loaders - Single graph, no batching needed
        train_loader = [train_data]  # Just wrap in list for iteration
        val_loader = [val_data]
        test_loader = [test_data]

        logger.info("✅ Data loading and graph construction completed")
        logger.info(f"   Training samples: {train_data.x.shape[0]}")
        logger.info(f"   Validation samples: {val_data.x.shape[0]}")
        logger.info(f"   Test samples: {test_data.x.shape[0]}")
        logger.info(
            f"   Graph density: {train_data.edge_index.shape[1] / (train_data.x.shape[0] * (train_data.x.shape[0] - 1)):.4f}"
        )

        # Step 2: Model Initialization
        logger.info("🧠 Step 2: Initializing GIMAN model...")

        model = GIMANClassifier(
            input_dim=config["input_dim"],
            hidden_dims=config["hidden_dims"],
            output_dim=config["output_dim"],
            dropout_rate=config["dropout_rate"],
            pooling_method=config["pooling_method"],
            classification_level=config["classification_level"],
        )

        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        logger.info(f"✅ Model initialized: {type(model).__name__}")
        logger.info(f"   Total parameters: {total_params:,}")
        logger.info(f"   Trainable parameters: {trainable_params:,}")
        logger.info(f"   Device: {config['device']}")

        # Step 3: Trainer Setup
        logger.info("🏃 Step 3: Setting up trainer...")

        trainer = GIMANTrainer(
            model=model,
            device=config["device"],
            optimizer_name=config["optimizer"],
            learning_rate=config["learning_rate"],
            weight_decay=config["weight_decay"],
            scheduler_type=config["scheduler_type"],
            early_stopping_patience=config["early_stopping_patience"],
            checkpoint_dir=Path("checkpoints")
            / f"giman_run_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            experiment_name="giman_complete_training",
        )

        logger.info("✅ Trainer setup completed")

        # Step 3.5: Setup Class Balancing for Imbalanced Data
        logger.info("⚖️ Setting up class balancing to address data imbalance...")

        loss_function = config["loss_function"]

        if loss_function == "focal":
            # Use Focal Loss for severe imbalance
            trainer.setup_focal_loss(
                train_loader, alpha=config["focal_alpha"], gamma=config["focal_gamma"]
            )
            logger.info(
                f"   Using Focal Loss with alpha={config['focal_alpha']}, gamma={config['focal_gamma']}"
            )

        elif loss_function == "label_smoothing":
            # Use Label Smoothing CrossEntropyLoss for better generalization
            trainer.setup_label_smoothing_loss(
                train_loader, smoothing=config["label_smoothing"]
            )
            logger.info(
                f"   Using Label Smoothing CrossEntropyLoss with smoothing={config['label_smoothing']}"
            )

        else:  # "weighted"
            # Use Weighted CrossEntropyLoss (most stable option)
            trainer.setup_weighted_loss(train_loader)
            logger.info("   Using Weighted CrossEntropyLoss")

        logger.info("✅ Class balancing setup completed")

        # Step 4: Model Training
        logger.info("🎯 Step 4: Starting model training...")

        training_history = trainer.train(
            train_loader=train_loader,
            val_loader=val_loader,
            num_epochs=config["num_epochs"],
            verbose=True,
        )

        logger.info("✅ Training completed successfully!")

        # Step 5: Model Evaluation
        logger.info("📈 Step 5: Evaluating trained model...")

        # Load best model for evaluation
        best_model_path = trainer.checkpoint_dir / "best_model.pt"
        if best_model_path.exists():
            trainer.load_checkpoint(best_model_path)
            logger.info("📂 Loaded best model checkpoint for evaluation")

        # Comprehensive evaluation
        test_results = trainer.evaluate(test_loader)

        logger.info("✅ Evaluation completed!")

        # Step 6: Results Summary
        logger.info("📋 Step 6: Training and evaluation summary...")
        logger.info("=" * 80)
        logger.info("🎉 GIMAN TRAINING PIPELINE COMPLETED SUCCESSFULLY!")
        logger.info("=" * 80)

        # Training summary
        training_summary = trainer.get_training_summary()
        logger.info("📊 Training Summary:")
        logger.info(
            f"   Total epochs: {training_summary['training_results']['total_epochs']}"
        )
        logger.info(
            f"   Best validation loss: {training_summary['training_results']['best_val_loss']:.4f}"
        )

        final_metrics = training_summary["training_results"]["final_metrics"]
        if final_metrics["val_accuracy"]:
            logger.info(
                f"   Final validation accuracy: {final_metrics['val_accuracy']:.4f}"
            )
            logger.info(f"   Final validation F1: {final_metrics['val_f1']:.4f}")
            logger.info(f"   Final validation AUC: {final_metrics['val_auc']:.4f}")

        # Test evaluation summary
        logger.info("🧪 Test Evaluation Results:")
        logger.info(f"   Test accuracy: {test_results['accuracy']:.4f}")
        logger.info(f"   Test precision: {test_results['precision']:.4f}")
        logger.info(f"   Test recall: {test_results['recall']:.4f}")
        logger.info(f"   Test F1 score: {test_results['f1']:.4f}")
        logger.info(f"   Test AUC-ROC: {test_results['auc_roc']:.4f}")

        # Cohort distribution summary
        logger.info("👥 Dataset Cohort Distributions:")
        cohort_mapping = train_data.cohort_mapping
        import numpy as np

        for split_name, split_data in [
            ("Train", train_data),
            ("Val", val_data),
            ("Test", test_data),
        ]:
            y_labels = split_data.y.cpu().numpy()
            unique, counts = np.unique(y_labels, return_counts=True)
            distribution = {
                cohort_mapping[int(label)]: int(count)
                for label, count in zip(unique, counts, strict=False)
            }
            logger.info(f"   {split_name}: {distribution}")

        # Save results
        results_dir = (
            Path("outputs")
            / f"giman_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
        results_dir.mkdir(parents=True, exist_ok=True)

        # Save training history
        import json

        with open(results_dir / "training_history.json", "w") as f:
            # Convert numpy types to Python types for JSON serialization
            history_json = {}
            for key, values in training_history.items():
                history_json[key] = [float(v) for v in values]
            json.dump(history_json, f, indent=2)

        # Save test results
        test_results_serializable = {}
        for key, value in test_results.items():
            if key == "confusion_matrix":
                test_results_serializable[key] = value.tolist()
            elif key == "classification_report":
                test_results_serializable[key] = value
            else:
                test_results_serializable[key] = float(value)

        with open(results_dir / "test_results.json", "w") as f:
            json.dump(test_results_serializable, f, indent=2)

        # Save configuration
        with open(results_dir / "config.json", "w") as f:
            json.dump(config, f, indent=2)

        logger.info(f"💾 Results saved to: {results_dir}")
        logger.info("=" * 80)

        return {
            "training_history": training_history,
            "test_results": test_results,
            "training_summary": training_summary,
            "config": config,
            "results_dir": results_dir,
        }

    except Exception as e:
        logger.error(f"❌ Training pipeline failed: {str(e)}")
        import traceback

        logger.error(f"Traceback:\n{traceback.format_exc()}")
        raise


def get_cohort_distribution(data):
    """Get cohort distribution for a PyTorch Geometric Data object."""
    y_counts = torch.bincount(data.y)
    distribution = {}

    cohort_mapping = data.cohort_mapping
    for encoded_label, count in enumerate(y_counts):
        if encoded_label in cohort_mapping:
            cohort_name = cohort_mapping[encoded_label]
            distribution[cohort_name] = count.item()

    return distribution


if __name__ == "__main__":
    results = main()
    print("\n🎉 GIMAN training pipeline completed successfully!")
    print(f"📁 Results saved to: {results['results_dir']}")
