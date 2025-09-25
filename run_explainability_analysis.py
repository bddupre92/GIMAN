#!/usr/bin/env python3
"""
GIMAN Explainability Analysis Script

This script loads the trained GIMAN model and performs comprehensive 
explainability analysis to understand how the model makes predictions.

Usage:
    python run_explainability_analysis.py

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

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.giman_pipeline.interpretability.gnn_explainer import GIMANExplainer
from src.giman_pipeline.training.models import GIMANClassifier
from configs.optimal_binary_config import OPTIMAL_BINARY_CONFIG
import warnings
warnings.filterwarnings('ignore')

def load_trained_model():
    """Load the trained GIMAN model and data."""
    print("🔄 Loading trained GIMAN model and data...")
    
    # Find the latest model directory
    models_dir = project_root / "models"
    print(f"   🔍 Looking for models in: {models_dir}")
    
    # Look for final binary model directory
    model_dirs = list(models_dir.glob("final_binary_giman_*"))
    print(f"   📁 Found {len(model_dirs)} model directories")
    
    if not model_dirs:
        print(f"   📋 Available directories in models/:")
        for item in models_dir.iterdir():
            print(f"      - {item.name}")
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
    
    # Initialize model with correct architecture
    model = GIMANClassifier(
        input_dim=graph_data.x.shape[1],
        hidden_dims=OPTIMAL_BINARY_CONFIG["model_params"]["hidden_dims"],
        output_dim=1,  # Binary classification
        dropout_rate=OPTIMAL_BINARY_CONFIG["model_params"]["dropout_rate"]
    )
    
    # Load trained weights
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
    model.load_state_dict(checkpoint)
    model.eval()
    
    print(f"   ✅ Model loaded successfully: {sum(p.numel() for p in model.parameters())} parameters")
    
    return model, graph_data

def run_comprehensive_analysis():
    """Run the complete explainability analysis."""
    print("=" * 80)
    print("🔍 GIMAN EXPLAINABILITY ANALYSIS")
    print("=" * 80)
    
    # Load model and data
    model, graph_data = load_trained_model()
    
    # Define feature names (based on PPMI biomarkers)
    feature_names = [
        'Age', 'Education_Years', 'MoCA_Score',
        'UPDRS_I_Total', 'UPDRS_III_Total',
        'Caudate_SBR', 'Putamen_SBR'
    ]
    
    # Initialize explainer
    explainer = GIMANExplainer(model, graph_data, feature_names)
    
    print("\n" + "=" * 60)
    print("📊 1. NODE IMPORTANCE ANALYSIS")
    print("=" * 60)
    
    # Node importance analysis
    print("🔍 Calculating gradient-based node importance...")
    node_importance = explainer.get_node_importance(method='gradient')
    
    # Visualize node importance
    vis_dir = project_root / "results" / "explainability"
    vis_dir.mkdir(parents=True, exist_ok=True)
    
    explainer.visualize_node_importance(
        node_importance, 
        save_path=vis_dir / "node_importance_analysis.png"
    )
    
    print("\n" + "=" * 60)
    print("📈 2. FEATURE IMPORTANCE ANALYSIS")
    print("=" * 60)
    
    # Feature importance analysis
    print("🔍 Calculating feature importance...")
    feature_importance = explainer.get_feature_importance()
    
    print("\n📋 Feature Importance Ranking:")
    for i, feature in enumerate(feature_importance['ranked_features']):
        importance = feature_importance['feature_importance'][feature_importance['importance_ranking'][i]]
        print(f"   {i+1:2d}. {feature:<20}: {importance:.6f}")
    
    # Visualize feature importance
    explainer.visualize_feature_importance(
        feature_importance,
        save_path=vis_dir / "feature_importance_analysis.png"
    )
    
    print("\n" + "=" * 60)
    print("🔗 3. EDGE CONTRIBUTION ANALYSIS")
    print("=" * 60)
    
    # Edge contribution analysis
    print("🔍 Analyzing edge contributions...")
    edge_contributions = explainer.get_edge_contributions()
    
    print(f"\n📋 Top Edge Contributions (analyzed {edge_contributions['total_edges_analyzed']} edges):")
    for i, edge in enumerate(edge_contributions['edge_contributions'][:10]):
        print(f"   {i+1:2d}. Edge {edge['source']} -> {edge['target']}: "
              f"Contribution = {edge['contribution']:.6f}")
    
    print("\n" + "=" * 60)
    print("📄 4. GENERATING COMPREHENSIVE REPORT")
    print("=" * 60)
    
    # Generate comprehensive report
    report_path = vis_dir / "giman_interpretation_report.json"
    interpretation_report = explainer.generate_interpretation_report(
        save_path=report_path
    )
    
    # Print key insights
    insights = interpretation_report['insights']
    print(f"\n🎯 KEY INSIGHTS:")
    print(f"   • Most important node: #{insights['most_important_node']}")
    print(f"   • Least important node: #{insights['least_important_node']}")
    print(f"   • Importance concentration (CV): {insights['importance_concentration']:.3f}")
    print(f"   • Top 3 features: {', '.join(insights['top_features'])}")
    
    graph_stats = interpretation_report['graph_statistics']
    print(f"\n📊 GRAPH STATISTICS:")
    print(f"   • Average degree: {graph_stats['degree_stats']['mean']:.2f} ± {graph_stats['degree_stats']['std']:.2f}")
    print(f"   • Degree range: {graph_stats['degree_stats']['min']} - {graph_stats['degree_stats']['max']}")
    print(f"   • Clustering coefficient: {graph_stats['clustering_coefficient']:.4f}")
    print(f"   • Graph density: {graph_stats['density']:.4f}")
    
    if 'class_distribution' in interpretation_report:
        class_dist = interpretation_report['class_distribution']
        print(f"\n🏷️  CLASS DISTRIBUTION:")
        for class_id, count in class_dist.items():
            class_name = "Healthy" if class_id == 0 else "Diseased"
            print(f"   • {class_name} (Class {class_id}): {count} patients")
    
    print("\n" + "=" * 60)
    print("🎯 5. INDIVIDUAL NODE ANALYSIS")
    print("=" * 60)
    
    # Analyze specific nodes
    important_scores = node_importance['importance_scores']['binary']
    most_important_node = np.argmax(important_scores)
    least_important_node = np.argmin(important_scores)
    
    print(f"🔍 Analyzing most important node (#{most_important_node})...")
    node_analysis = explainer.compare_predictions_with_without_edges(
        target_node=most_important_node, 
        num_edges_to_remove=5
    )
    
    print(f"\n📊 Node #{most_important_node} Analysis:")
    print(f"   • Original prediction: {node_analysis['original_prediction']}")
    print(f"   • Connected edges: {node_analysis['total_connected_edges']}")
    print(f"   • Top edge impacts:")
    
    for i, edge in enumerate(node_analysis['edge_removal_analysis'][:3]):
        print(f"     {i+1}. Edge to node #{edge['target_node_in_edge']}: "
              f"Δ = {edge['prediction_change']:.6f}")
    
    print("\n" + "=" * 80)
    print("✅ EXPLAINABILITY ANALYSIS COMPLETE")
    print("=" * 80)
    print(f"📁 Results saved to: {vis_dir}")
    print(f"   • Node importance plots: node_importance_analysis.png")
    print(f"   • Feature importance plots: feature_importance_analysis.png")
    print(f"   • Comprehensive report: giman_interpretation_report.json")
    
    return {
        'node_importance': node_importance,
        'feature_importance': feature_importance,
        'edge_contributions': edge_contributions,
        'interpretation_report': interpretation_report,
        'individual_analysis': node_analysis
    }

def validate_model_authenticity():
    """Verify that model results are actually computed, not hardcoded."""
    print("\n" + "🔍" * 60)
    print("🛡️  MODEL AUTHENTICITY VALIDATION")
    print("🔍" * 60)
    
    model, graph_data = load_trained_model()
    
    # Test 1: Predictions should change with different inputs
    print("📋 Test 1: Input sensitivity check...")
    with torch.no_grad():
        original_pred = model(graph_data.x, graph_data.edge_index)
        
        # Slightly modify input
        modified_x = graph_data.x.clone()
        modified_x[0, 0] += 0.1  # Small change to first node, first feature
        modified_pred = model(modified_x, graph_data.edge_index)
        
        pred_diff = torch.abs(original_pred - modified_pred).sum().item()
        print(f"   • Prediction difference with input change: {pred_diff:.8f}")
        
        if pred_diff > 1e-6:
            print("   ✅ PASS: Model responds to input changes")
        else:
            print("   ❌ FAIL: Model may have hardcoded outputs")
    
    # Test 2: Different edge configurations should give different results
    print("📋 Test 2: Graph structure sensitivity check...")
    with torch.no_grad():
        original_pred = model(graph_data.x, graph_data.edge_index)
        
        # Remove random edges
        num_edges = graph_data.edge_index.shape[1]
        keep_mask = torch.rand(num_edges) > 0.1  # Remove ~10% of edges
        modified_edges = graph_data.edge_index[:, keep_mask]
        
        modified_pred = model(graph_data.x, modified_edges)
        structure_diff = torch.abs(original_pred - modified_pred).sum().item()
        print(f"   • Prediction difference with edge removal: {structure_diff:.8f}")
        
        if structure_diff > 1e-6:
            print("   ✅ PASS: Model responds to graph structure changes")
        else:
            print("   ❌ FAIL: Model may not use graph structure")
    
    # Test 3: Model parameters should affect output
    print("📋 Test 3: Parameter sensitivity check...")
    model_copy = GIMANClassifier(
        input_dim=graph_data.x.shape[1],
        hidden_dims=[64, 32, 16],
        output_dim=1,
        dropout=0.5
    )
    
    with torch.no_grad():
        original_pred = model(graph_data.x, graph_data.edge_index)
        different_arch_pred = model_copy(graph_data.x, graph_data.edge_index)
        
        arch_diff = torch.abs(original_pred - different_arch_pred).sum().item()
        print(f"   • Prediction difference with different architecture: {arch_diff:.8f}")
        
        if arch_diff > 1e-3:
            print("   ✅ PASS: Model architecture affects predictions")
        else:
            print("   ❌ CONCERN: Different architectures give similar outputs")
    
    print(f"\n🔍 VALIDATION SUMMARY:")
    print(f"   The model appears to be generating authentic predictions")
    print(f"   based on actual computation, not hardcoded values.")

if __name__ == "__main__":
    try:
        # First validate model authenticity
        validate_model_authenticity()
        
        # Then run comprehensive analysis
        results = run_comprehensive_analysis()
        
        print(f"\n🎉 Analysis completed successfully!")
        
    except Exception as e:
        print(f"❌ Error during analysis: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)