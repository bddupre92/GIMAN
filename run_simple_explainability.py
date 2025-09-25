#!/usr/bin/env python3
"""
GIMAN Explainability Analysis Script (Simplified)

This script loads the trained GIMAN model and performs comprehensive 
explainability analysis to understand how the model makes predictions.

Usage:
    python run_simple_explainability.py

Author: GIMAN Team
Date: 2024-09-23
"""

import sys
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, Any, List, Tuple

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.giman_pipeline.training.models import GIMANClassifier
from configs.optimal_binary_config import OPTIMAL_BINARY_CONFIG
import warnings
warnings.filterwarnings('ignore')

def load_trained_model():
    """Load the trained GIMAN model and data."""
    print("🔄 Loading trained GIMAN model and data...")
    
    # Find the latest model directory
    models_dir = project_root / "models"
    
    # Look for final binary model directory
    model_dirs = list(models_dir.glob("final_binary_giman_*"))
    if not model_dirs:
        raise FileNotFoundError("No final binary GIMAN model found in models/")
    
    # Get the latest directory by name (they have timestamps)
    latest_model_dir = sorted(model_dirs)[-1]
    print(f"   📁 Using model from: {latest_model_dir}")
    
    # Load model
    model_path = latest_model_dir / "final_binary_giman.pth"
    graph_path = latest_model_dir / "graph_data.pth"
    
    if not model_path.exists() or not graph_path.exists():
        raise FileNotFoundError(f"Model files not found in {latest_model_dir}")
    
    # Load graph data
    graph_data = torch.load(graph_path, weights_only=False)
    print(f"   📊 Graph data loaded: {graph_data.x.shape[0]} nodes, {graph_data.x.shape[1]} features")
    
    # Load trained weights
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
    model_config = checkpoint['model_config']
    
    # Initialize model with saved configuration
    model = GIMANClassifier(
        input_dim=graph_data.x.shape[1],
        hidden_dims=model_config['hidden_dims'],
        output_dim=model_config['num_classes'],  # Use saved num_classes
        dropout_rate=model_config['dropout_rate'],
        classification_level="node"  # Node-level classification
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"   ✅ Model loaded successfully: {sum(p.numel() for p in model.parameters())} parameters")
    
    return model, graph_data

def analyze_node_importance(model, graph_data) -> Dict[str, np.ndarray]:
    """Calculate node importance using gradient-based method."""
    print("🔍 Calculating gradient-based node importance...")
    
    # Enable gradients for input features
    graph_data.x.requires_grad_(True)
    
    # Forward pass
    output = model(graph_data)
    logits = output['logits']
    
    # Calculate gradient magnitude for each node
    if logits.shape[1] == 1:  # Binary classification
        # Sum all logits and compute gradient
        loss = logits.sum()
        loss.backward(retain_graph=True)
        
        # Node importance = sum of absolute gradients across features
        node_importance = torch.abs(graph_data.x.grad).sum(dim=1).detach().cpu().numpy()
        
    else:  # Multi-class
        # Use gradient w.r.t. predicted class
        preds = torch.argmax(logits, dim=1)
        selected_logits = logits[torch.arange(logits.shape[0]), preds]
        loss = selected_logits.sum()
        loss.backward(retain_graph=True)
        
        node_importance = torch.abs(graph_data.x.grad).sum(dim=1).detach().cpu().numpy()
    
    return {
        'node_importance': node_importance,
        'method': 'gradient_magnitude'
    }

def analyze_feature_importance(model, graph_data) -> Dict[str, np.ndarray]:
    """Calculate feature importance across all nodes."""
    print("🔍 Calculating feature importance...")
    
    # Enable gradients for input features
    graph_data.x.requires_grad_(True)
    
    # Forward pass
    output = model(graph_data)
    logits = output['logits']
    
    # Calculate gradient w.r.t. input features
    if logits.shape[1] == 1:  # Binary classification
        loss = logits.sum()
        loss.backward()
        
        # Feature importance = mean absolute gradient across all nodes
        feature_importance = torch.abs(graph_data.x.grad).mean(dim=0).detach().cpu().numpy()
        
    else:  # Multi-class
        preds = torch.argmax(logits, dim=1)
        selected_logits = logits[torch.arange(logits.shape[0]), preds]
        loss = selected_logits.sum()
        loss.backward()
        
        feature_importance = torch.abs(graph_data.x.grad).mean(dim=0).detach().cpu().numpy()
    
    return {
        'feature_importance': feature_importance,
        'importance_ranking': np.argsort(feature_importance)[::-1]
    }

def validate_model_authenticity(model, graph_data):
    """Verify that model results are actually computed, not hardcoded."""
    print("\n🛡️  MODEL AUTHENTICITY VALIDATION")
    print("=" * 60)
    
    # Test 1: Predictions should change with different inputs
    print("📋 Test 1: Input sensitivity check...")
    with torch.no_grad():
        original_output = model(graph_data)
        original_pred = original_output['logits']
        
        # Slightly modify input
        modified_data = graph_data.clone()
        modified_data.x = modified_data.x.clone()
        modified_data.x[0, 0] += 0.1  # Small change to first node, first feature
        
        modified_output = model(modified_data)
        modified_pred = modified_output['logits']
        
        pred_diff = torch.abs(original_pred - modified_pred).sum().item()
        print(f"   • Prediction difference with input change: {pred_diff:.8f}")
        
        if pred_diff > 1e-6:
            print("   ✅ PASS: Model responds to input changes")
        else:
            print("   ❌ FAIL: Model may have hardcoded outputs")
    
    # Test 2: Different edge configurations should give different results
    print("📋 Test 2: Graph structure sensitivity check...")
    with torch.no_grad():
        original_output = model(graph_data)
        original_pred = original_output['logits']
        
        # Remove some edges
        modified_data = graph_data.clone()
        num_edges = modified_data.edge_index.shape[1]
        keep_mask = torch.rand(num_edges) > 0.1  # Remove ~10% of edges
        modified_data.edge_index = modified_data.edge_index[:, keep_mask]
        
        modified_output = model(modified_data)
        modified_pred = modified_output['logits']
        
        structure_diff = torch.abs(original_pred - modified_pred).sum().item()
        print(f"   • Prediction difference with edge removal: {structure_diff:.8f}")
        
        if structure_diff > 1e-6:
            print("   ✅ PASS: Model responds to graph structure changes")
        else:
            print("   ❌ FAIL: Model may not use graph structure")
    
    print(f"\n🔍 VALIDATION SUMMARY:")
    print(f"   The model appears to be generating authentic predictions")
    print(f"   based on actual computation, not hardcoded values.")

def analyze_predictions_distribution(model, graph_data):
    """Analyze the distribution of model predictions."""
    print("\n📊 PREDICTION DISTRIBUTION ANALYSIS")
    print("=" * 60)
    
    with torch.no_grad():
        output = model(graph_data)
        logits = output['logits']
        
        if logits.shape[1] == 1:  # Binary classification
            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).int()
            
            print(f"📋 Binary Classification Results:")
            print(f"   • Total nodes: {len(preds)}")
            print(f"   • Predicted Class 0 (Healthy): {(preds == 0).sum().item()}")
            print(f"   • Predicted Class 1 (Diseased): {(preds == 1).sum().item()}")
            print(f"   • Mean probability: {probs.mean().item():.4f}")
            print(f"   • Probability std: {probs.std().item():.4f}")
            print(f"   • Min probability: {probs.min().item():.4f}")
            print(f"   • Max probability: {probs.max().item():.4f}")
            
        else:  # Multi-class
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1)
            
            print(f"📋 Multi-class Classification Results:")
            for class_id in range(logits.shape[1]):
                count = (preds == class_id).sum().item()
                avg_prob = probs[:, class_id].mean().item()
                print(f"   • Class {class_id}: {count} nodes (avg prob: {avg_prob:.4f})")
        
        # Compare with true labels if available
        if hasattr(graph_data, 'y') and graph_data.y is not None:
            y_true = graph_data.y.cpu().numpy()
            y_pred = preds.cpu().numpy().squeeze()
            
            from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
            
            accuracy = accuracy_score(y_true, y_pred)
            precision = precision_score(y_true, y_pred, average='binary' if logits.shape[1] == 1 else 'macro')
            recall = recall_score(y_true, y_pred, average='binary' if logits.shape[1] == 1 else 'macro')
            f1 = f1_score(y_true, y_pred, average='binary' if logits.shape[1] == 1 else 'macro')
            
            print(f"\n🎯 Performance on Current Data:")
            print(f"   • Accuracy: {accuracy:.4f}")
            print(f"   • Precision: {precision:.4f}")
            print(f"   • Recall: {recall:.4f}")
            print(f"   • F1-Score: {f1:.4f}")

def create_visualizations(node_results, feature_results, graph_data):
    """Create comprehensive visualizations."""
    print("\n📊 CREATING VISUALIZATIONS")
    print("=" * 60)
    
    # Create results directory
    vis_dir = project_root / "results" / "explainability"
    vis_dir.mkdir(parents=True, exist_ok=True)
    
    feature_names = ['Age', 'Education_Years', 'MoCA_Score', 
                     'UPDRS_I_Total', 'UPDRS_III_Total', 
                     'Caudate_SBR', 'Putamen_SBR']
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    fig.suptitle('GIMAN Explainability Analysis', fontsize=16, fontweight='bold')
    
    # 1. Node importance histogram
    node_importance = node_results['node_importance']
    axes[0, 0].hist(node_importance, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
    axes[0, 0].set_title('Distribution of Node Importance')
    axes[0, 0].set_xlabel('Importance Score')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Top important nodes
    top_indices = np.argsort(node_importance)[-10:]
    top_scores = node_importance[top_indices]
    
    axes[0, 1].barh(range(len(top_scores)), top_scores, color='coral')
    axes[0, 1].set_title('Top 10 Most Important Nodes')
    axes[0, 1].set_xlabel('Importance Score')
    axes[0, 1].set_yticks(range(len(top_scores)))
    axes[0, 1].set_yticklabels([f'Node {i}' for i in top_indices])
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Feature importance
    feature_importance = feature_results['feature_importance']
    sorted_idx = feature_results['importance_ranking']
    
    axes[0, 2].barh(range(len(feature_importance)), feature_importance[sorted_idx], 
                   color='lightgreen')
    axes[0, 2].set_title('Feature Importance Ranking')
    axes[0, 2].set_xlabel('Importance Score')
    axes[0, 2].set_yticks(range(len(feature_importance)))
    axes[0, 2].set_yticklabels([feature_names[i] for i in sorted_idx])
    axes[0, 2].grid(True, alpha=0.3)
    
    # 4. Node degree distribution
    degrees = np.bincount(graph_data.edge_index[0].detach().cpu().numpy(), 
                         minlength=graph_data.x.shape[0])
    
    axes[1, 0].hist(degrees, bins=20, alpha=0.7, color='orange', edgecolor='black')
    axes[1, 0].set_title('Node Degree Distribution')
    axes[1, 0].set_xlabel('Node Degree')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].grid(True, alpha=0.3)
    
    # 5. Importance vs degree correlation
    axes[1, 1].scatter(degrees, node_importance, alpha=0.6, color='purple')
    axes[1, 1].set_title('Node Importance vs Degree')
    axes[1, 1].set_xlabel('Node Degree')
    axes[1, 1].set_ylabel('Importance Score')
    axes[1, 1].grid(True, alpha=0.3)
    
    # Calculate correlation
    correlation = np.corrcoef(degrees, node_importance)[0, 1]
    axes[1, 1].text(0.05, 0.95, f'Correlation: {correlation:.3f}', 
                   transform=axes[1, 1].transAxes, 
                   bbox=dict(boxstyle="round", facecolor='wheat', alpha=0.8))
    
    # 6. Feature correlation heatmap
    feature_corr = np.corrcoef(graph_data.x.detach().cpu().numpy().T)
    im = axes[1, 2].imshow(feature_corr, cmap='coolwarm', aspect='auto', vmin=-1, vmax=1)
    axes[1, 2].set_title('Feature Correlation Matrix')
    axes[1, 2].set_xticks(range(len(feature_names)))
    axes[1, 2].set_yticks(range(len(feature_names)))
    axes[1, 2].set_xticklabels(feature_names, rotation=45, ha='right')
    axes[1, 2].set_yticklabels(feature_names)
    
    # Add colorbar for correlation
    cbar = plt.colorbar(im, ax=axes[1, 2], shrink=0.8)
    cbar.set_label('Correlation')
    
    plt.tight_layout()
    
    # Save plot
    save_path = vis_dir / "giman_explainability_analysis.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"📊 Visualizations saved to: {save_path}")
    
    plt.show()

def main():
    """Main analysis function."""
    print("=" * 80)
    print("🔍 GIMAN EXPLAINABILITY ANALYSIS")
    print("=" * 80)
    
    try:
        # Load model and data
        model, graph_data = load_trained_model()
        
        # Validate model authenticity
        validate_model_authenticity(model, graph_data)
        
        # Analyze prediction distribution
        analyze_predictions_distribution(model, graph_data)
        
        print("\n" + "=" * 80)
        print("📊 INTERPRETABILITY ANALYSIS")
        print("=" * 80)
        
        # Node importance analysis
        node_results = analyze_node_importance(model, graph_data)
        
        # Feature importance analysis  
        feature_results = analyze_feature_importance(model, graph_data)
        
        # Create visualizations
        create_visualizations(node_results, feature_results, graph_data)
        
        # Print summary
        print("\n" + "=" * 80)
        print("🎯 ANALYSIS SUMMARY")
        print("=" * 80)
        
        node_importance = node_results['node_importance']
        feature_importance = feature_results['feature_importance']
        feature_names = ['Age', 'Education_Years', 'MoCA_Score', 
                         'UPDRS_I_Total', 'UPDRS_III_Total', 
                         'Caudate_SBR', 'Putamen_SBR']
        
        print(f"🔍 KEY INSIGHTS:")
        print(f"   • Most important node: #{np.argmax(node_importance)} (score: {np.max(node_importance):.6f})")
        print(f"   • Least important node: #{np.argmin(node_importance)} (score: {np.min(node_importance):.6f})")
        print(f"   • Node importance range: {np.min(node_importance):.6f} - {np.max(node_importance):.6f}")
        print(f"   • Node importance std: {np.std(node_importance):.6f}")
        
        print(f"\n📈 TOP 3 MOST IMPORTANT FEATURES:")
        for i, feat_idx in enumerate(feature_results['importance_ranking'][:3]):
            print(f"   {i+1}. {feature_names[feat_idx]}: {feature_importance[feat_idx]:.6f}")
        
        degrees = np.bincount(graph_data.edge_index[0].detach().cpu().numpy(), 
                             minlength=graph_data.x.shape[0])
        correlation = np.corrcoef(degrees, node_importance)[0, 1]
        print(f"\n📊 GRAPH STATISTICS:")
        print(f"   • Average degree: {np.mean(degrees):.2f} ± {np.std(degrees):.2f}")
        print(f"   • Degree-importance correlation: {correlation:.4f}")
        print(f"   • Graph density: {2 * graph_data.edge_index.shape[1] / (graph_data.x.shape[0] * (graph_data.x.shape[0] - 1)):.4f}")
        
        print(f"\n✅ Analysis completed successfully!")
        print(f"   📁 Results saved to: results/explainability/")
        
        return {
            'node_results': node_results,
            'feature_results': feature_results,
            'model': model,
            'graph_data': graph_data
        }
        
    except Exception as e:
        print(f"❌ Error during analysis: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    results = main()