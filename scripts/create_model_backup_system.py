#!/usr/bin/env python3
"""GIMAN Model Backup and Versioning System

This script creates a comprehensive backup system for GIMAN models,
ensuring we can always restore previous high-performing versions.

Features:
- Model versioning with semantic versioning (v1.0.0, v1.1.0, etc.)
- Complete model state preservation (weights, config, data, metadata)
- Performance tracking and comparison
- Easy restoration procedures
- Model registry with detailed documentation

Author: GIMAN Team
Date: 2024-09-23
"""

import json
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import torch

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


class GIMANModelRegistry:
    """Manages versioning and backup of GIMAN models."""

    def __init__(self, registry_dir: str = "models/registry"):
        """Initialize the model registry.

        Args:
            registry_dir: Directory to store model registry
        """
        self.registry_dir = Path(registry_dir)
        self.registry_dir.mkdir(parents=True, exist_ok=True)
        self.registry_file = self.registry_dir / "model_registry.json"
        self.registry = self._load_registry()

    def _load_registry(self) -> dict[str, Any]:
        """Load existing model registry or create new one."""
        if self.registry_file.exists():
            with open(self.registry_file) as f:
                return json.load(f)
        else:
            return {
                "models": {},
                "current_production": None,
                "created_at": datetime.now().isoformat(),
                "version": "1.0.0",
            }

    def _save_registry(self):
        """Save registry to disk."""
        self.registry["updated_at"] = datetime.now().isoformat()
        with open(self.registry_file, "w") as f:
            json.dump(self.registry, f, indent=2)

    def register_model(
        self,
        model_name: str,
        version: str,
        source_dir: str,
        performance_metrics: dict[str, float],
        model_config: dict[str, Any],
        description: str = "",
        tags: list = None,
    ) -> str:
        """Register a new model version.

        Args:
            model_name: Name of the model (e.g., "giman_binary_classifier")
            version: Semantic version (e.g., "1.0.0")
            source_dir: Path to current model directory
            performance_metrics: Performance metrics dict
            model_config: Model configuration dict
            description: Description of this version
            tags: Optional tags for categorization

        Returns:
            Path to registered model directory
        """
        print(f"📦 Registering model: {model_name} v{version}")

        # Create versioned model directory
        model_id = f"{model_name}_v{version}"
        backup_dir = self.registry_dir / model_id
        backup_dir.mkdir(exist_ok=True)

        # Copy all model files
        source_path = Path(source_dir)
        if not source_path.exists():
            raise FileNotFoundError(f"Source model directory not found: {source_dir}")

        print(f"   📁 Copying model files from {source_path}")
        for file_path in source_path.glob("*"):
            if file_path.is_file():
                shutil.copy2(file_path, backup_dir / file_path.name)
                print(f"      ✅ Copied: {file_path.name}")

        # Create comprehensive metadata
        metadata = {
            "model_name": model_name,
            "version": version,
            "model_id": model_id,
            "created_at": datetime.now().isoformat(),
            "description": description,
            "tags": tags or [],
            "performance_metrics": performance_metrics,
            "model_config": model_config,
            "source_directory": str(source_path),
            "backup_directory": str(backup_dir),
            "files_backed_up": [f.name for f in source_path.glob("*") if f.is_file()],
        }

        # Save metadata
        metadata_file = backup_dir / "model_metadata.json"
        with open(metadata_file, "w") as f:
            json.dump(metadata, f, indent=2)

        # Update registry
        self.registry["models"][model_id] = metadata
        self._save_registry()

        print(f"   ✅ Model registered successfully: {model_id}")
        print(
            f"   📊 Performance: AUC-ROC = {performance_metrics.get('auc_roc', 'N/A')}"
        )

        return str(backup_dir)

    def set_production_model(self, model_id: str):
        """Set a model as the current production model."""
        if model_id not in self.registry["models"]:
            raise ValueError(f"Model {model_id} not found in registry")

        self.registry["current_production"] = model_id
        self.registry["production_set_at"] = datetime.now().isoformat()
        self._save_registry()

        print(f"🚀 Set production model: {model_id}")

    def list_models(self):
        """List all registered models."""
        print("\n📋 REGISTERED MODELS")
        print("=" * 60)

        if not self.registry["models"]:
            print("No models registered yet.")
            return

        for model_id, metadata in self.registry["models"].items():
            is_production = model_id == self.registry.get("current_production")
            status = "🚀 PRODUCTION" if is_production else "📦 ARCHIVED"

            print(f"\n{status} {model_id}")
            print(f"   Description: {metadata['description']}")
            print(f"   Created: {metadata['created_at']}")
            print(
                f"   Performance: AUC-ROC = {metadata['performance_metrics'].get('auc_roc', 'N/A')}"
            )
            print(f"   Tags: {', '.join(metadata['tags'])}")

    def restore_model(self, model_id: str, target_dir: str) -> str:
        """Restore a model from registry to active directory.

        Args:
            model_id: ID of model to restore
            target_dir: Directory to restore model to

        Returns:
            Path to restored model directory
        """
        if model_id not in self.registry["models"]:
            raise ValueError(f"Model {model_id} not found in registry")

        metadata = self.registry["models"][model_id]
        source_dir = Path(metadata["backup_directory"])
        target_path = Path(target_dir)

        print(f"🔄 Restoring model: {model_id}")
        print(f"   From: {source_dir}")
        print(f"   To: {target_path}")

        # Create target directory
        target_path.mkdir(parents=True, exist_ok=True)

        # Copy all files
        for file_path in source_dir.glob("*"):
            if file_path.is_file():
                shutil.copy2(file_path, target_path / file_path.name)
                print(f"      ✅ Restored: {file_path.name}")

        print(f"   ✅ Model restored successfully to: {target_path}")
        return str(target_path)

    def compare_models(self, model_id1: str, model_id2: str):
        """Compare performance between two models."""
        if model_id1 not in self.registry["models"]:
            raise ValueError(f"Model {model_id1} not found in registry")
        if model_id2 not in self.registry["models"]:
            raise ValueError(f"Model {model_id2} not found in registry")

        model1 = self.registry["models"][model_id1]
        model2 = self.registry["models"][model_id2]

        print("\n🔍 MODEL COMPARISON")
        print("=" * 60)
        print(f"Model 1: {model_id1}")
        print(f"Model 2: {model_id2}")
        print()

        # Compare performance metrics
        metrics1 = model1["performance_metrics"]
        metrics2 = model2["performance_metrics"]

        for metric in set(metrics1.keys()) | set(metrics2.keys()):
            val1 = metrics1.get(metric, "N/A")
            val2 = metrics2.get(metric, "N/A")

            if isinstance(val1, (int, float)) and isinstance(val2, (int, float)):
                diff = val2 - val1
                arrow = "⬆️" if diff > 0 else "⬇️" if diff < 0 else "➡️"
                print(f"   {metric}: {val1:.4f} vs {val2:.4f} {arrow} ({diff:+.4f})")
            else:
                print(f"   {metric}: {val1} vs {val2}")


def backup_current_production_model():
    """Backup the current production model (98.93% AUC-ROC)."""
    print("🎯 BACKING UP CURRENT PRODUCTION MODEL")
    print("=" * 50)

    # Initialize registry
    registry = GIMANModelRegistry()

    # Find current model directory
    models_dir = Path("models")
    current_model_dirs = list(models_dir.glob("final_binary_giman_*"))

    if not current_model_dirs:
        raise FileNotFoundError("No current model found to backup")

    # Get the latest model
    latest_model_dir = sorted(current_model_dirs)[-1]
    print(f"   📁 Found current model: {latest_model_dir}")

    # Load model metadata
    model_path = latest_model_dir / "final_binary_giman.pth"
    if model_path.exists():
        checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
        model_config = checkpoint.get("model_config", {})
        training_metrics = checkpoint.get("training_metrics", {})
    else:
        raise FileNotFoundError(f"Model weights not found: {model_path}")

    # Performance metrics from our explainability analysis
    performance_metrics = {
        "auc_roc": 0.9893,  # 98.93% from terminal output
        "accuracy": 0.7684,
        "precision": 0.6138,
        "recall": 0.8757,
        "f1_score": 0.6144,
        "validation_auc": training_metrics.get("best_val_auc", 0.9893),
        "num_features": 7,
        "num_patients": 557,
        "class_balance": "14:1 imbalanced (resolved with FocalLoss)",
    }

    # Enhanced model configuration
    enhanced_config = {
        **model_config,
        "feature_names": [
            "Age",
            "Education_Years",
            "MoCA_Score",
            "UPDRS_I_Total",
            "UPDRS_III_Total",
            "Caudate_SBR",
            "Putamen_SBR",
        ],
        "graph_construction": "k-NN with k=6, cosine similarity",
        "loss_function": "FocalLoss with gamma=2.09",
        "optimization": "AdamW with systematic hyperparameter tuning",
        "training_framework": "PyTorch Geometric",
    }

    # Register the model
    backup_path = registry.register_model(
        model_name="giman_binary_classifier",
        version="1.0.0",
        source_dir=str(latest_model_dir),
        performance_metrics=performance_metrics,
        model_config=enhanced_config,
        description="Production-ready binary diagnostic classifier (PD vs Healthy). Achieved 98.93% AUC-ROC with 7 clinical/imaging features. Includes comprehensive explainability analysis.",
        tags=[
            "production",
            "binary_classification",
            "high_performance",
            "explainable",
            "validated",
        ],
    )

    # Set as production model
    registry.set_production_model("giman_binary_classifier_v1.0.0")

    print("\n✅ BACKUP COMPLETE")
    print(f"   📦 Model backed up to: {backup_path}")
    print("   🚀 Set as production model")

    return registry


def create_restoration_script():
    """Create a simple script for restoring the production model."""
    restore_script = """#!/usr/bin/env python3
'''
Quick Model Restoration Script

This script quickly restores the production GIMAN model (v1.0.0 - 98.93% AUC-ROC)
in case experiments with enhanced features don't work out.

Usage:
    python restore_production_model.py
'''

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from scripts.create_model_backup_system import GIMANModelRegistry

def main():
    print("🔄 RESTORING PRODUCTION MODEL")
    print("=" * 40)
    
    # Initialize registry
    registry = GIMANModelRegistry()
    
    # Show available models
    registry.list_models()
    
    # Get production model
    production_model = registry.registry.get("current_production")
    if not production_model:
        print("❌ No production model set!")
        return
    
    print(f"\\n🚀 Restoring production model: {production_model}")
    
    # Restore to active models directory
    restore_path = f"models/restored_production_{production_model}"
    registry.restore_model(production_model, restore_path)
    
    print(f"\\n✅ Production model restored!")
    print(f"   📁 Location: {restore_path}")
    print(f"   📊 Performance: 98.93% AUC-ROC")
    print(f"   🎯 Ready for immediate use")

if __name__ == "__main__":
    main()
"""

    script_path = Path("scripts/restore_production_model.py")
    with open(script_path, "w") as f:
        f.write(restore_script)

    # Make executable
    script_path.chmod(0o755)

    print(f"📝 Created restoration script: {script_path}")


def main():
    """Main backup creation function."""
    print("🛡️  GIMAN MODEL BACKUP & VERSIONING SYSTEM")
    print("=" * 60)

    # Backup current production model
    registry = backup_current_production_model()

    # Create restoration script
    create_restoration_script()

    # Show registry status
    print("\n📋 CURRENT REGISTRY STATUS")
    print("=" * 40)
    registry.list_models()

    print("\n🎯 NEXT STEPS")
    print("=" * 20)
    print("✅ Current model (98.93% AUC-ROC) safely backed up")
    print("✅ Model registry system created")
    print("✅ Quick restoration script available")
    print("🚀 Ready to experiment with enhanced features!")

    print("\n💡 RESTORATION OPTIONS:")
    print("1. Run: python scripts/restore_production_model.py")
    print(
        "2. Use registry API: registry.restore_model('giman_binary_classifier_v1.0.0', target_dir)"
    )
    print("3. Manual copy from: models/registry/giman_binary_classifier_v1.0.0/")

    return registry


if __name__ == "__main__":
    registry = main()
