"""Final Binary GIMAN Model Creation and Persistence
================================================

This script creates the final optimized binary classification model
and saves it for future use.

Author: GIMAN Team
Date: September 23, 2    print("="*60)
    print("🏆 FINAL BINARY GIMAN MODEL CREATED SUCCESSFULLY!")
    print("="*60)
    print(f"📊 Test AUC-ROC: {results['test_results']['auc_roc']:.2%}")
    print(f"📊 Test Accuracy: {results['test_results']['accuracy']:.2%}")
    print(f"📊 Test F1 Score: {results['test_results']['f1']:.2%}")
    print(f"💾 Model Location: {results['save_path']}")
    print("="*60)ormance: 98.93% AUC-ROC
"""

import logging
import os
import sys
from datetime import datetime
from pathlib import Path

import torch

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from configs.optimal_binary_config import get_optimal_config
from giman_pipeline.modeling.patient_similarity import PatientSimilarityGraph

from .models import GIMANClassifier
from .trainer import GIMANTrainer

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def create_final_binary_model(save_path: str = None) -> dict:
    """Create and save the final optimized binary GIMAN model.

    Args:
        save_path: Optional custom save path for the model

    Returns:
        dict: Training results and model information
    """
    # Get optimal configuration
    config = get_optimal_config()
    logger.info("🏆 Creating final binary GIMAN model with optimal configuration")

    # Create save directory if not specified
    if save_path is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = f"models/final_binary_giman_{timestamp}"

    os.makedirs(save_path, exist_ok=True)

    try:
        # 1. Initialize Patient Similarity Graph with optimal parameters
        logger.info("📊 Initializing PatientSimilarityGraph with optimal parameters")
        graph_params = config["graph_params"]

        psg = PatientSimilarityGraph(
            similarity_threshold=None,  # Use top_k instead of threshold
            similarity_metric=graph_params["similarity_metric"],
            top_k_connections=graph_params["top_k_connections"],
            binary_classification=True,
        )

        # 2. Load and prepare data
        logger.info("🔄 Loading and preparing PPMI data")
        psg.load_enhanced_cohort()
        psg.calculate_patient_similarity()
        psg.create_similarity_graph()

        # Convert to PyTorch Geometric format
        data = psg.to_pytorch_geometric()

        # 3. Split data
        data_params = config["data_params"]
        train_data, val_data, test_data = psg.split_for_training(
            test_size=data_params["test_ratio"],
            val_size=data_params["val_ratio"],
            random_state=data_params["random_state"],
        )

        # Prepare data loaders (as lists for the trainer)
        train_loader = [train_data]
        val_loader = [val_data]
        test_loader = [test_data]

        # 4. Create model with optimal architecture
        logger.info("🏗️ Creating GIMAN model with optimal architecture")
        model_params = config["model_params"]

        model = GIMANClassifier(
            input_dim=data.x.size(1),
            hidden_dims=model_params["hidden_dims"],
            output_dim=model_params["num_classes"],
            dropout_rate=model_params["dropout_rate"],
        )

        logger.info(
            f"📈 Model created with {sum(p.numel() for p in model.parameters()):,} parameters"
        )

        # 5. Initialize trainer with optimal parameters
        logger.info("🚀 Initializing GIMAN trainer with optimal parameters")
        training_params = config["training_params"]
        loss_params = config["loss_params"]

        trainer = GIMANTrainer(
            model=model,
            optimizer_name=training_params["optimizer"],
            learning_rate=training_params["learning_rate"],
            weight_decay=training_params["weight_decay"],
            scheduler_type=training_params["scheduler"],
            early_stopping_patience=training_params["patience"],
        )

        # Setup focal loss with optimal parameters
        logger.info("🎯 Setting up Focal Loss with optimal parameters")
        trainer.setup_focal_loss(
            train_loader,
            alpha=loss_params["focal_alpha"],
            gamma=loss_params["focal_gamma"],
        )

        # 6. Train the final model
        logger.info("🏃 Training final optimized binary GIMAN model")
        training_results = trainer.train(
            train_loader=train_loader,
            val_loader=val_loader,
            num_epochs=training_params["max_epochs"],
            verbose=True,
        )

        # 7. Evaluate on test set
        logger.info("🧪 Evaluating final model on test set")
        test_results = trainer.evaluate(test_loader)

        # 8. Save the complete model package
        logger.info(f"💾 Saving final model to {save_path}")

        # Save model state
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "model_config": model_params,
                "training_config": config,
                "training_results": training_results,
                "test_results": test_results,
                "data_info": {
                    "num_nodes": data.x.size(0),
                    "num_features": data.x.size(1),
                    "num_edges": data.edge_index.size(1),
                    "cohort_mapping": getattr(
                        data, "cohort_mapping", {"0": "Disease", "1": "Healthy"}
                    ),
                },
            },
            f"{save_path}/final_binary_giman.pth",
        )

        # Save configuration
        import json

        with open(f"{save_path}/optimal_config.json", "w") as f:
            # Convert any non-serializable objects to strings
            serializable_config = {}
            for key, value in config.items():
                if isinstance(value, dict):
                    serializable_config[key] = {
                        k: str(v)
                        if not isinstance(v, (int, float, str, bool, list))
                        else v
                        for k, v in value.items()
                    }
                else:
                    serializable_config[key] = (
                        str(value)
                        if not isinstance(value, (int, float, str, bool, list))
                        else value
                    )
            json.dump(serializable_config, f, indent=2)

        # Save graph data
        torch.save(data, f"{save_path}/graph_data.pth")

        # Create model summary
        summary = {
            "model_name": "Final Binary GIMAN",
            "creation_date": datetime.now().isoformat(),
            "performance": test_results,
            "architecture": model_params["hidden_dims"],
            "total_parameters": sum(p.numel() for p in model.parameters()),
            "graph_structure": {
                "nodes": data.x.size(0),
                "edges": data.edge_index.size(1),
                "k_connections": graph_params["top_k_connections"],
                "similarity_metric": graph_params["similarity_metric"],
            },
            "save_path": save_path,
        }

        with open(f"{save_path}/model_summary.json", "w") as f:
            json.dump(summary, f, indent=2, default=str)

        logger.info("✅ Final binary GIMAN model created and saved successfully!")
        logger.info(f"📊 Test AUC-ROC: {test_results['auc_roc']:.4f}")
        logger.info(f"📊 Test Accuracy: {test_results['accuracy']:.4f}")
        logger.info(f"📊 Test F1: {test_results['f1']:.4f}")
        logger.info(f"💾 Model saved to: {save_path}")

        return {
            "save_path": save_path,
            "test_results": test_results,
            "training_results": training_results,
            "model_summary": summary,
        }

    except Exception as e:
        logger.error(f"❌ Error creating final model: {str(e)}")
        raise


def load_final_binary_model(model_path: str):
    """Load the final binary GIMAN model.

    Args:
        model_path: Path to the saved model directory

    Returns:
        tuple: (model, config, data, results)
    """
    # Load model checkpoint
    checkpoint = torch.load(f"{model_path}/final_binary_giman.pth", map_location="cpu")

    # Reconstruct model
    model_config = checkpoint["model_config"]
    input_dim = checkpoint["data_info"]["num_features"]

    model = GIMANClassifier(
        input_dim=input_dim,
        hidden_dims=model_config["hidden_dims"],
        num_classes=model_config["num_classes"],
        dropout_rate=model_config["dropout_rate"],
    )

    model.load_state_dict(checkpoint["model_state_dict"])

    # Load graph data
    graph_data = torch.load(f"{model_path}/graph_data.pth", map_location="cpu")

    return model, checkpoint["training_config"], graph_data, checkpoint["test_results"]


if __name__ == "__main__":
    # Create the final optimized binary model
    results = create_final_binary_model()

    print("\n" + "=" * 60)
    print("🏆 FINAL BINARY GIMAN MODEL CREATED SUCCESSFULLY!")
    print("=" * 60)
    print(f"📊 Test AUC-ROC: {results['test_results']['auc_roc']:.2%}")
    print(f"📊 Test Accuracy: {results['test_results']['accuracy']:.2%}")
    print(f"📊 Test F1 Score: {results['test_results']['f1']:.2%}")
    print(f"💾 Model Location: {results['save_path']}")
    print("=" * 60)
