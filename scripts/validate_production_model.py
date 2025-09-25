#!/usr/bin/env python3
"""
Production Model Validation & Testing Script

This script validates that a restored production model works exactly 
as expected and produces the same results as the original.

Usage:
    python validate_production_model.py [model_path]

Author: GIMAN Team
Date: 2024-09-23
"""

import sys
import torch
import numpy as np
from pathlib import Path
from typing import Dict, Any, Tuple

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.giman_pipeline.training.models import GIMANClassifier
import warnings
warnings.filterwarnings('ignore')


def validate_model_integrity(model_path: str) -> Dict[str, Any]:
    """Validate that a model loads and functions correctly.
    
    Args:
        model_path: Path to model directory
        
    Returns:
        Validation results dictionary
    """
    model_dir = Path(model_path)
    print(f"🔍 VALIDATING MODEL: {model_dir}")
    print("=" * 50)
    
    results = {
        "model_path": str(model_dir),
        "validation_passed": False,
        "errors": [],
        "warnings": [],
        "metrics": {}
    }
    
    try:
        # Check required files exist
        required_files = [
            "final_binary_giman.pth",
            "graph_data.pth", 
            "model_summary.json",
            "optimal_config.json"
        ]
        
        print("📋 Checking required files...")
        for file_name in required_files:
            file_path = model_dir / file_name
            if file_path.exists():
                print(f"   ✅ {file_name}")
            else:
                error_msg = f"Missing required file: {file_name}"
                print(f"   ❌ {error_msg}")
                results["errors"].append(error_msg)
        
        if results["errors"]:
            return results
        
        # Load graph data
        print(f"\n📊 Loading graph data...")
        graph_data = torch.load(model_dir / "graph_data.pth", weights_only=False)
        print(f"   ✅ Graph: {graph_data.x.shape[0]} nodes, {graph_data.x.shape[1]} features")
        
        # Verify expected dimensions
        expected_nodes = 557
        expected_features = 7
        
        if graph_data.x.shape[0] != expected_nodes:
            error_msg = f"Unexpected number of nodes: {graph_data.x.shape[0]} (expected {expected_nodes})"
            results["errors"].append(error_msg)
            print(f"   ❌ {error_msg}")
        
        if graph_data.x.shape[1] != expected_features:
            error_msg = f"Unexpected number of features: {graph_data.x.shape[1]} (expected {expected_features})"
            results["errors"].append(error_msg)
            print(f"   ❌ {error_msg}")
        
        # Load model
        print(f"\n🧠 Loading model...")
        checkpoint = torch.load(model_dir / "final_binary_giman.pth", map_location='cpu', weights_only=False)
        model_config = checkpoint['model_config']
        
        # Initialize model
        model = GIMANClassifier(
            input_dim=graph_data.x.shape[1],
            hidden_dims=model_config['hidden_dims'],
            output_dim=model_config['num_classes'],
            dropout_rate=model_config['dropout_rate'],
            classification_level="node"
        )
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        param_count = sum(p.numel() for p in model.parameters())
        expected_params = 92866
        
        print(f"   ✅ Model loaded: {param_count} parameters")
        
        if param_count != expected_params:
            warning_msg = f"Parameter count differs: {param_count} (expected {expected_params})"
            results["warnings"].append(warning_msg)
            print(f"   ⚠️  {warning_msg}")
        
        # Test forward pass
        print(f"\n🔬 Testing forward pass...")
        with torch.no_grad():
            output = model(graph_data)
            logits = output['logits']
            
            if logits.shape[0] != expected_nodes:
                error_msg = f"Output shape mismatch: {logits.shape[0]} (expected {expected_nodes})"
                results["errors"].append(error_msg)
                print(f"   ❌ {error_msg}")
            else:
                print(f"   ✅ Forward pass successful: {logits.shape}")
        
        # Test predictions
        print(f"\n📈 Testing predictions...")
        with torch.no_grad():
            if logits.shape[1] == 1:  # Binary classification
                probs = torch.sigmoid(logits)
                preds = (probs > 0.5).int()
            else:  # Multi-class classification
                probs = torch.softmax(logits, dim=1)
                preds = torch.argmax(probs, dim=1)
            
            class_0_count = (preds == 0).sum().item()
            class_1_count = (preds == 1).sum().item()
            
            print(f"   ✅ Predictions: {class_0_count} healthy, {class_1_count} diseased")
            
            results["metrics"] = {
                "predictions_healthy": class_0_count,
                "predictions_diseased": class_1_count,
                "total_predictions": class_0_count + class_1_count,
                "mean_probability": probs.mean().item() if logits.shape[1] == 1 else probs[:, 1].mean().item(),
                "probability_std": probs.std().item() if logits.shape[1] == 1 else probs[:, 1].std().item()
            }
        
        # Performance validation if labels available
        if hasattr(graph_data, 'y') and graph_data.y is not None:
            print(f"\n🎯 Validating performance...")
            from sklearn.metrics import accuracy_score, roc_auc_score
            
            y_true = graph_data.y.cpu().numpy()
            y_pred = preds.cpu().numpy().squeeze()
            
            # Handle probability extraction based on output format
            if logits.shape[1] == 1:  # Binary with single output
                y_probs = probs.cpu().numpy().squeeze()
            else:  # Multi-class, use class 1 probabilities
                y_probs = probs[:, 1].cpu().numpy()
            
            accuracy = accuracy_score(y_true, y_pred)
            auc_roc = roc_auc_score(y_true, y_probs)
            
            print(f"   📊 Accuracy: {accuracy:.4f}")
            print(f"   📊 AUC-ROC: {auc_roc:.4f}")
            
            # Check if performance matches expected
            expected_auc = 0.9893
            if abs(auc_roc - expected_auc) > 0.01:  # Allow 1% tolerance
                warning_msg = f"AUC-ROC differs from expected: {auc_roc:.4f} vs {expected_auc:.4f}"
                results["warnings"].append(warning_msg)
                print(f"   ⚠️  {warning_msg}")
            else:
                print(f"   ✅ Performance matches expected values")
            
            results["metrics"].update({
                "accuracy": accuracy,
                "auc_roc": auc_roc,
                "expected_auc": expected_auc,
                "performance_validated": abs(auc_roc - expected_auc) <= 0.01
            })
        
        # If we get here without errors, validation passed
        if not results["errors"]:
            results["validation_passed"] = True
            print(f"\n✅ MODEL VALIDATION PASSED")
        
    except Exception as e:
        error_msg = f"Validation failed with exception: {str(e)}"
        results["errors"].append(error_msg)
        print(f"\n❌ {error_msg}")
        import traceback
        traceback.print_exc()
    
    return results


def compare_model_outputs(model_path1: str, model_path2: str) -> Dict[str, Any]:
    """Compare outputs between two models to ensure they're identical.
    
    Args:
        model_path1: Path to first model
        model_path2: Path to second model
        
    Returns:
        Comparison results
    """
    print(f"\n🔍 COMPARING MODEL OUTPUTS")
    print("=" * 40)
    print(f"Model 1: {model_path1}")
    print(f"Model 2: {model_path2}")
    
    # Load both models
    def load_model(path):
        model_dir = Path(path)
        graph_data = torch.load(model_dir / "graph_data.pth", weights_only=False)
        checkpoint = torch.load(model_dir / "final_binary_giman.pth", map_location='cpu', weights_only=False)
        
        model = GIMANClassifier(
            input_dim=graph_data.x.shape[1],
            hidden_dims=checkpoint['model_config']['hidden_dims'],
            output_dim=checkpoint['model_config']['num_classes'],
            dropout_rate=checkpoint['model_config']['dropout_rate'],
            classification_level="node"
        )
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        return model, graph_data
    
    try:
        model1, graph1 = load_model(model_path1)
        model2, graph2 = load_model(model_path2)
        
        # Compare graph data
        graph_identical = torch.allclose(graph1.x, graph2.x, atol=1e-6)
        print(f"   Graph data identical: {graph_identical}")
        
        # Compare model outputs
        with torch.no_grad():
            out1 = model1(graph1)['logits']
            out2 = model2(graph2)['logits']
            
            outputs_identical = torch.allclose(out1, out2, atol=1e-6)
            max_diff = torch.max(torch.abs(out1 - out2)).item()
            
            print(f"   Model outputs identical: {outputs_identical}")
            print(f"   Maximum difference: {max_diff:.2e}")
            
            return {
                "comparison_successful": True,
                "graph_data_identical": graph_identical,
                "outputs_identical": outputs_identical,
                "max_output_difference": max_diff,
                "models_equivalent": graph_identical and outputs_identical
            }
    
    except Exception as e:
        print(f"   ❌ Comparison failed: {str(e)}")
        return {
            "comparison_successful": False,
            "error": str(e)
        }


def main():
    """Main validation function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Validate GIMAN production model")
    parser.add_argument("model_path", nargs="?", 
                       default="models/registry/giman_binary_classifier_v1.0.0",
                       help="Path to model directory to validate")
    parser.add_argument("--compare", 
                       help="Path to second model for comparison")
    
    args = parser.parse_args()
    
    print("🛡️  GIMAN PRODUCTION MODEL VALIDATION")
    print("=" * 50)
    
    # Validate primary model
    results = validate_model_integrity(args.model_path)
    
    # Show results
    print(f"\n📋 VALIDATION SUMMARY")
    print("=" * 30)
    
    if results["validation_passed"]:
        print("✅ Overall Status: PASSED")
    else:
        print("❌ Overall Status: FAILED")
    
    if results["errors"]:
        print(f"\n❌ Errors ({len(results['errors'])}):")
        for error in results["errors"]:
            print(f"   • {error}")
    
    if results["warnings"]:
        print(f"\n⚠️  Warnings ({len(results['warnings'])}):")
        for warning in results["warnings"]:
            print(f"   • {warning}")
    
    if results["metrics"]:
        print(f"\n📊 Performance Metrics:")
        for metric, value in results["metrics"].items():
            if isinstance(value, float):
                print(f"   • {metric}: {value:.4f}")
            else:
                print(f"   • {metric}: {value}")
    
    # Optional comparison
    if args.compare:
        comparison = compare_model_outputs(args.model_path, args.compare)
        
        print(f"\n🔍 Model Comparison:")
        if comparison.get("comparison_successful"):
            equiv = comparison.get("models_equivalent", False)
            status = "✅ EQUIVALENT" if equiv else "❌ DIFFERENT"
            print(f"   Status: {status}")
            print(f"   Max difference: {comparison.get('max_output_difference', 'N/A'):.2e}")
        else:
            print(f"   ❌ Comparison failed: {comparison.get('error', 'Unknown error')}")
    
    return results


if __name__ == "__main__":
    results = main()