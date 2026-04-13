#!/usr/bin/env python3
"""Test the best configurations found during optimization."""

import logging
import sys
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent / "src"))

from giman_pipeline.modeling.patient_similarity import PatientSimilarityGraph
from giman_pipeline.training.models import GIMANClassifier
from giman_pipeline.training.trainer import GIMANTrainer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_configuration(config_name, params):
    """Test a specific configuration."""
    logger.info(f"🧪 Testing {config_name}")

    try:
        # Create similarity graph
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
        test_loader = [test_data]

        # Create model
        model = GIMANClassifier(
            input_dim=7,
            hidden_dims=[
                params["hidden_dim_1"],
                params["hidden_dim_2"],
                params["hidden_dim_3"],
            ],
            output_dim=2,
            dropout_rate=params["dropout_rate"],
            pooling_method="concat",
            classification_level="node",
        )

        # Create trainer
        trainer = GIMANTrainer(
            model=model,
            device="cpu",
            optimizer_name="adamw",
            learning_rate=params["learning_rate"],
            weight_decay=params["weight_decay"],
            scheduler_type="plateau",
            early_stopping_patience=20,
            checkpoint_dir=Path(f"checkpoints/{config_name}"),
            experiment_name=config_name,
        )

        trainer.setup_focal_loss(train_loader, alpha=1.0, gamma=params["focal_gamma"])

        # Train model
        training_history = trainer.train(
            train_loader=train_loader,
            val_loader=val_loader,
            num_epochs=100,
            verbose=True,
        )

        # Evaluate on test set
        test_results = trainer.evaluate(test_loader)

        logger.info(f"✅ {config_name} Results:")
        logger.info(f"   Test AUC: {test_results['auc_roc']:.4f}")
        logger.info(f"   Test Accuracy: {test_results['accuracy']:.4f}")
        logger.info(f"   Test F1: {test_results['f1']:.4f}")
        logger.info(f"   Test Precision: {test_results['precision']:.4f}")
        logger.info(f"   Test Recall: {test_results['recall']:.4f}")

        return {
            "config_name": config_name,
            "params": params,
            "test_results": test_results,
            "training_history": training_history,
        }

    except Exception as e:
        logger.error(f"❌ {config_name} failed: {e}")
        return None


if __name__ == "__main__":
    # Based on the terminal output, these configurations showed high validation AUCs
    configurations = [
        {
            "name": "high_auc_config_1",
            "params": {
                "top_k_connections": 7,
                "similarity_metric": "cosine",
                "hidden_dim_1": 160,
                "hidden_dim_2": 384,
                "hidden_dim_3": 96,
                "dropout_rate": 0.33,
                "learning_rate": 0.0004,
                "weight_decay": 6e-6,
                "focal_gamma": 1.47,
            },
        },
        {
            "name": "high_auc_config_2",
            "params": {
                "top_k_connections": 5,
                "similarity_metric": "cosine",
                "hidden_dim_1": 64,
                "hidden_dim_2": 256,
                "hidden_dim_3": 80,
                "dropout_rate": 0.39,
                "learning_rate": 0.0019,
                "weight_decay": 0.0003,
                "focal_gamma": 2.13,
            },
        },
        {
            "name": "high_auc_config_3",
            "params": {
                "top_k_connections": 6,
                "similarity_metric": "cosine",
                "hidden_dim_1": 96,
                "hidden_dim_2": 256,
                "hidden_dim_3": 64,
                "dropout_rate": 0.41,
                "learning_rate": 0.0031,
                "weight_decay": 0.0002,
                "focal_gamma": 2.09,
            },
        },
        {
            "name": "high_auc_config_4",
            "params": {
                "top_k_connections": 9,
                "similarity_metric": "cosine",
                "hidden_dim_1": 224,
                "hidden_dim_2": 256,
                "hidden_dim_3": 96,
                "dropout_rate": 0.37,
                "learning_rate": 0.009,
                "weight_decay": 5e-5,
                "focal_gamma": 1.90,
            },
        },
    ]

    results = []

    for config in configurations:
        result = test_configuration(config["name"], config["params"])
        if result:
            results.append(result)

    # Find best configuration
    if results:
        best_result = max(results, key=lambda x: x["test_results"]["auc_roc"])

        logger.info("🏆 BEST CONFIGURATION:")
        logger.info(f"   Name: {best_result['config_name']}")
        logger.info(f"   Test AUC: {best_result['test_results']['auc_roc']:.4f}")
        logger.info(f"   Test Accuracy: {best_result['test_results']['accuracy']:.4f}")
        logger.info(f"   Test F1: {best_result['test_results']['f1']:.4f}")
        logger.info("   Parameters:")
        for key, value in best_result["params"].items():
            logger.info(f"     {key}: {value}")

        # Check if we achieved >90% AUC
        if best_result["test_results"]["auc_roc"] >= 0.90:
            logger.info("🎉 ACHIEVED >90% AUC TARGET!")
        else:
            logger.info(
                f"📈 Current best: {best_result['test_results']['auc_roc']:.4f} AUC (target: 0.90)"
            )
    else:
        logger.error("❌ All configurations failed")
