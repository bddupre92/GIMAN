#!/usr/bin/env python3
"""Complete GIMAN Training Script.

This script runs the complete GIMAN training pipeline:
1. Generate similarity graphs (if needed)
2. Load corrected longitudinal dataset
3. Create PyTorch Geometric data format
4. Train GIMAN GNN model
5. Evaluate and save results

Usage:
    python scripts/train_giman.py --data-dir data/01_processed --epochs 50
"""

import argparse
import logging
import sys
from pathlib import Path

# Import from parent package - now inside training folder
sys.path.append(str(Path(__file__).parent.parent.parent))  # Add project root

import torch
from torch_geometric.loader import DataLoader

from giman_pipeline.modeling.patient_similarity import create_patient_similarity_graph
from .data_loaders import GIMANDataLoader
from .models import GIMANClassifier
from .trainer import GIMANTrainer

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description="Train GIMAN model")
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/01_processed",
        help="Directory containing processed data",
    )
    parser.add_argument(
        "--epochs", type=int, default=50, help="Number of training epochs"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Batch size (usually 1 for graph-level tasks)",
    )
    parser.add_argument(
        "--learning-rate", type=float, default=0.001, help="Learning rate"
    )
    parser.add_argument(
        "--device", type=str, default="auto", help="Device to use (cpu, cuda, auto)"
    )
    parser.add_argument(
        "--similarity-threshold",
        type=float,
        default=0.3,
        help="Similarity threshold for graph edges",
    )

    args = parser.parse_args()

    # Set device
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    logger.info("🚀 Starting GIMAN training pipeline")
    logger.info(f"   - Data directory: {args.data_dir}")
    logger.info(f"   - Training epochs: {args.epochs}")
    logger.info(f"   - Device: {device}")
    logger.info(f"   - Similarity threshold: {args.similarity_threshold}")

    # Step 1: Generate similarity graphs (if needed)
    logger.info("📊 Step 1: Checking/generating similarity graphs...")

    try:
        graph, adjacency_matrix, metadata = create_patient_similarity_graph(
            data_path=args.data_dir,
            similarity_threshold=args.similarity_threshold,
            save_results=True,
        )
        logger.info(
            f"✅ Similarity graph ready: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges"
        )
        logger.info(f"   - Density: {metadata.get('graph_density', 'N/A'):.4f}")
        logger.info(f"   - Communities: {metadata.get('n_communities', 'N/A')}")
    except Exception as e:
        logger.error(f"❌ Error generating similarity graph: {e}")
        return 1

    # Step 2: Load data using GIMAN DataLoader
    logger.info("📁 Step 2: Loading data for training...")

    try:
        data_loader = GIMANDataLoader(
            data_dir=args.data_dir, similarity_threshold=args.similarity_threshold
        )

        data_loader.load_preprocessed_data()
        pyg_data = data_loader.create_pyg_data()

        logger.info("✅ Data loaded successfully")
        logger.info(f"   - Nodes: {pyg_data.x.shape[0]}")
        logger.info(f"   - Features: {pyg_data.x.shape[1]}")
        logger.info(f"   - Edges: {pyg_data.edge_index.shape[1]}")
        logger.info(
            f"   - Labels: PD={pyg_data.y.sum().item()}, HC={len(pyg_data.y) - pyg_data.y.sum().item()}"
        )

    except Exception as e:
        logger.error(f"❌ Error loading data: {e}")
        return 1

    # Step 3: Create train/val/test splits
    logger.info("🔀 Step 3: Creating train/validation/test splits...")

    try:
        train_data, val_data, test_data = data_loader.create_train_val_test_split(
            test_size=0.2, val_size=0.2, random_state=42
        )

        # Create data loaders
        train_loader = DataLoader(
            [train_data], batch_size=args.batch_size, shuffle=True
        )
        val_loader = DataLoader([val_data], batch_size=args.batch_size, shuffle=False)
        test_loader = DataLoader([test_data], batch_size=args.batch_size, shuffle=False)

        logger.info("✅ Data splits created:")
        logger.info(f"   - Training: {train_data.x.shape[0]} nodes")
        logger.info(f"   - Validation: {val_data.x.shape[0]} nodes")
        logger.info(f"   - Test: {test_data.x.shape[0]} nodes")

    except Exception as e:
        logger.error(f"❌ Error creating data splits: {e}")
        return 1

    # Step 4: Initialize GIMAN model
    logger.info("🧠 Step 4: Initializing GIMAN model...")

    try:
        model = GIMANClassifier(
            input_dim=pyg_data.x.shape[1],  # Number of biomarker features
            hidden_dims=[64, 128, 64],
            output_dim=2,  # Binary classification
            dropout_rate=0.3,
        )

        logger.info("✅ GIMAN model initialized:")
        logger.info(f"   - Parameters: {sum(p.numel() for p in model.parameters()):,}")
        logger.info(f"   - Input features: {pyg_data.x.shape[1]}")
        logger.info("   - Hidden dimensions: [64, 128, 64]")

    except Exception as e:
        logger.error(f"❌ Error initializing model: {e}")
        return 1

    # Step 5: Initialize trainer
    logger.info("🎓 Step 5: Initializing trainer...")

    try:
        trainer = GIMANTrainer(
            model=model,
            device=device,
            learning_rate=args.learning_rate,
            early_stopping_patience=15,
            checkpoint_dir=Path("checkpoints/giman_training"),
        )

        logger.info("✅ Trainer initialized with early stopping patience=15")

    except Exception as e:
        logger.error(f"❌ Error initializing trainer: {e}")
        return 1

    # Step 6: Train the model
    logger.info(f"🏋️ Step 6: Training GIMAN model for {args.epochs} epochs...")

    try:
        training_history = trainer.train(
            train_loader=train_loader,
            val_loader=val_loader,
            num_epochs=args.epochs,
            verbose=True,
        )

        logger.info("✅ Training completed successfully!")

    except Exception as e:
        logger.error(f"❌ Error during training: {e}")
        return 1

    # Step 7: Final evaluation
    logger.info("🧪 Step 7: Final model evaluation...")

    try:
        evaluation_results = trainer.evaluate(test_loader)

        logger.info("🎯 FINAL GIMAN PERFORMANCE:")
        logger.info(f"   - Test Accuracy: {evaluation_results['accuracy']:.4f}")
        logger.info(f"   - Test F1 Score: {evaluation_results['f1']:.4f}")
        logger.info(f"   - Test AUC-ROC: {evaluation_results['auc_roc']:.4f}")
        logger.info(f"   - Test Precision: {evaluation_results['precision']:.4f}")
        logger.info(f"   - Test Recall: {evaluation_results['recall']:.4f}")

    except Exception as e:
        logger.error(f"❌ Error during evaluation: {e}")
        return 1

    # Step 8: Save training summary
    logger.info("💾 Step 8: Saving training results...")

    try:
        summary = trainer.get_training_summary()

        # Save training history and results
        results_dir = Path("results/giman_training")
        results_dir.mkdir(parents=True, exist_ok=True)

        import json

        with open(results_dir / "training_summary.json", "w") as f:
            # Convert tensors to lists for JSON serialization
            json_summary = {}
            for key, value in summary.items():
                if key == "history":
                    json_summary[key] = {
                        k: [float(v) if torch.is_tensor(v) else v for v in vals]
                        for k, vals in value.items()
                    }
                else:
                    json_summary[key] = value
            json.dump(json_summary, f, indent=2, default=str)

        logger.info(
            f"✅ Training summary saved to {results_dir / 'training_summary.json'}"
        )

    except Exception as e:
        logger.error(f"❌ Error saving results: {e}")
        return 1

    # Final success message
    logger.info("🎉 GIMAN TRAINING PIPELINE COMPLETED SUCCESSFULLY!")
    logger.info("=" * 60)
    logger.info("📊 FINAL SUMMARY:")
    logger.info("   ✅ Model: GIMAN Graph Neural Network")
    logger.info(
        f"   ✅ Data: {pyg_data.x.shape[0]} patients, {pyg_data.x.shape[1]} biomarkers"
    )
    logger.info(
        f"   ✅ Graph: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges"
    )
    logger.info(
        f"   ✅ Performance: Acc={evaluation_results['accuracy']:.4f}, F1={evaluation_results['f1']:.4f}, AUC={evaluation_results['auc_roc']:.4f}"
    )
    logger.info("   ✅ Checkpoints: checkpoints/giman_training/")
    logger.info("   ✅ Results: results/giman_training/")
    logger.info("=" * 60)

    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
