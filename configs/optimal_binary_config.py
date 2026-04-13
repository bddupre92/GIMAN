"""Optimal Binary Classification Configuration for GIMAN
=====================================================

This configuration achieved 98.93% AUC-ROC on PPMI binary classification task.
Results from hyperparameter optimization on September 23, 2025.

Performance Metrics:
- AUC-ROC: 98.93%
- Accuracy: 69.05%
- F1 Score: 31.58%
- Precision: 18.75%
- Recall: 100.00% (perfect minority class detection)
- Confusion Matrix: [[52, 26], [0, 6]]
"""

from typing import Any

OPTIMAL_BINARY_CONFIG: dict[str, Any] = {
    # Graph Construction Parameters
    "graph_params": {
        "top_k_connections": 6,
        "similarity_metric": "cosine",
        "threshold": None,
    },
    # Model Architecture Parameters
    "model_params": {
        "hidden_dims": [96, 256, 64],  # 92,866 total parameters
        "dropout_rate": 0.41,
        "num_classes": 2,
        "activation": "relu",
    },
    # Training Parameters
    "training_params": {
        "learning_rate": 0.0031,
        "weight_decay": 0.0002,
        "optimizer": "adamw",
        "scheduler": "plateau",
        "max_epochs": 100,
        "patience": 10,
        "batch_size": None,  # Full batch for GNN
    },
    # Loss Function Parameters
    "loss_params": {
        "loss_type": "focal",
        "focal_alpha": 1.0,
        "focal_gamma": 2.09,
        "use_class_weights": True,
    },
    # Data Parameters
    "data_params": {
        "classification_type": "binary",
        "train_ratio": 0.7,
        "val_ratio": 0.15,
        "test_ratio": 0.15,
        "random_state": 42,
    },
    # Performance Achieved
    "performance_metrics": {
        "auc_roc": 0.9893,
        "accuracy": 0.6905,
        "f1_score": 0.3158,
        "precision": 0.1875,
        "recall": 1.0000,
        "training_time_seconds": 1.45,
        "total_parameters": 92866,
    },
}


def get_optimal_config() -> dict[str, Any]:
    """Get the optimal binary classification configuration."""
    return OPTIMAL_BINARY_CONFIG.copy()


def print_config_summary():
    """Print a summary of the optimal configuration."""
    config = OPTIMAL_BINARY_CONFIG

    print("🏆 OPTIMAL BINARY GIMAN CONFIGURATION")
    print("=" * 50)
    print(f"📊 Performance: {config['performance_metrics']['auc_roc']:.1%} AUC-ROC")
    print(f"🏗️  Architecture: {config['model_params']['hidden_dims']}")
    print(f"📈 Parameters: {config['performance_metrics']['total_parameters']:,}")
    print(
        f"🔗 Graph: k={config['graph_params']['top_k_connections']} {config['graph_params']['similarity_metric']} similarity"
    )
    print(
        f"🎯 Loss: Focal(α={config['loss_params']['focal_alpha']}, γ={config['loss_params']['focal_gamma']})"
    )
    print(f"⚡ Training: {config['performance_metrics']['training_time_seconds']:.2f}s")
    print("=" * 50)


if __name__ == "__main__":
    print_config_summary()
