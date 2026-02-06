import sys
from pathlib import Path
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from lifelines import KaplanMeierFitter

# Add project root
project_root = Path(__file__).resolve().parents[3]
sys.path.append(str(project_root))

sys.path.append(str(project_root / "archive/development/phase8/subphase8_2_dynamic_endpoints"))
from train_final_giman_survival import GIMANSurvivalGAT
from archive.development.phase9.neuro_fuzzy import NeuroFuzzyGIMAN

# Setup output
interactive_output = project_root / "visualizations/interactive"
interactive_output.mkdir(parents=True, exist_ok=True)

def create_interactive_survival_curves(dataset, device):
    """Interactive Kaplan-Meier with Plotly"""
    print("📊 Creating Interactive Survival Curves...")
    
    # Load model
    in_features = dataset.x.shape[1]
    model = GIMANSurvivalGAT(in_features=in_features, hidden_dim=128).to(device)
    checkpoint = project_root / "outputs/phase8_2_final_training/giman_survival_final.pth"
    
    if not checkpoint.exists():
        print("Phase 8 checkpoint not found.")
        return
    
    model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=False)['model_state_dict'])
    model.eval()
    
    with torch.no_grad():
        dataset = dataset.to(device)
        risk_scores = model(dataset).cpu().numpy()
        times = dataset.time.cpu().numpy()
        events = dataset.event.cpu().numpy()
    
    # Stratify
    p33, p66 = np.percentile(risk_scores, [33, 66])
    groups = np.zeros_like(risk_scores, dtype=int)
    groups[risk_scores > p33] = 1
    groups[risk_scores > p66] = 2
    
    # Create KM curves
    kmf = KaplanMeierFitter()
    fig = go.Figure()
    
    colors = ['#2ecc71', '#f1c40f', '#e74c3c']
    labels = ['Low Risk', 'Medium Risk', 'High Risk']
    
    for i in range(3):
        mask = (groups == i)
        if mask.sum() == 0:
            continue
            
        kmf.fit(times[mask], events[mask])
        
        # Get KM estimates
        time_points = kmf.survival_function_.index
        survival_prob = kmf.survival_function_.values.flatten()
        ci_lower = kmf.confidence_interval_.iloc[:, 0].values
        ci_upper = kmf.confidence_interval_.iloc[:, 1].values
        
        # Add main curve
        fig.add_trace(go.Scatter(
            x=time_points,
            y=survival_prob,
            name=f'{labels[i]} (n={mask.sum()})',
            line=dict(color=colors[i], width=3),
            mode='lines',
            hovertemplate='<b>%{fullData.name}</b><br>Time: %{x:.1f} months<br>Survival: %{y:.3f}<extra></extra>'
        ))
        
        # Add confidence interval
        fig.add_trace(go.Scatter(
            x=list(time_points) + list(time_points[::-1]),
            y=list(ci_upper) + list(ci_lower[::-1]),
            fill='toself',
            fillcolor=colors[i],
            opacity=0.2,
            line=dict(width=0),
            showlegend=False,
            hoverinfo='skip'
        ))
    
    fig.update_layout(
        title='Interactive Survival Analysis: Risk Stratification',
        xaxis_title='Time (Months)',
        yaxis_title='Survival Probability',
        template='plotly_white',
        hovermode='x unified',
        width=1000,
        height=600
    )
    
    output_path = interactive_output / "interactive_survival.html"
    fig.write_html(str(output_path))
    print(f"✅ Saved: {output_path}")

def create_fuzzy_rule_explorer(dataset, device):
    """Interactive 3D exploration of fuzzy rules"""
    print("📊 Creating Fuzzy Rule Explorer...")
    
    in_features = dataset.x.shape[1]
    gat_encoder = GIMANSurvivalGAT(in_features=in_features, hidden_dim=128)
    model = NeuroFuzzyGIMAN(gat_encoder, num_classes=2, num_rules=32).to(device)
    checkpoint = project_root / "outputs/phase9_neuro_fuzzy/neuro_fuzzy_best.pth"
    
    if not checkpoint.exists():
        print("Phase 9 checkpoint not found.")
        return
    
    model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=False))
    model.eval()
    
    with torch.no_grad():
        dataset = dataset.to(device)
        logits, weights = model(dataset)
        probs = F.softmax(logits, dim=1)[:, 1].cpu().numpy()
        
    # Get top 3 rules by variance
    weights_np = weights.cpu().numpy()
    rule_var = np.var(weights_np, axis=0)
    top_rules = np.argsort(rule_var)[::-1][:3]
    
    # Create 3D scatter
    targets = dataset.event.cpu().numpy()
    
    rule_0, rule_1, rule_2 = top_rules[0], top_rules[1], top_rules[2]
    
    fig = go.Figure(data=[go.Scatter3d(
        x=weights_np[:, rule_0],
        y=weights_np[:, rule_1],
        z=weights_np[:, rule_2],
        mode='markers',
        marker=dict(
            size=5,
            color=targets,
            colorscale='RdYlGn',
            showscale=True,
            colorbar=dict(title="SAA Status"),
            opacity=0.8
        ),
        text=[f'Patient {i}<br>Prob: {p:.3f}' for i, p in enumerate(probs)],
        hovertemplate=f'<b>%{{text}}</b><br>Rule {rule_0}: %{{x:.3f}}<br>Rule {rule_1}: %{{y:.3f}}<br>Rule {rule_2}: %{{z:.3f}}<extra></extra>'
    )])
    
    fig.update_layout(
        title=f'Fuzzy Rule Activation Space (Top 3 Rules: {rule_0}, {rule_1}, {rule_2})',
        scene=dict(
            xaxis_title=f'Rule {rule_0} Activation',
            yaxis_title=f'Rule {rule_1} Activation',
            zaxis_title=f'Rule {rule_2} Activation'
        ),
        width=1000,
        height=800
    )
    
    output_path = interactive_output / "fuzzy_rule_3d.html"
    fig.write_html(str(output_path))
    print(f"✅ Saved: {output_path}")

def create_feature_importance_dashboard():
    """Interactive feature importance from Phase 9"""
    print("📊 Creating Feature Importance Dashboard...")
    
    # Load saved SHAP values if available, otherwise use rule weights
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data_path = project_root / "data/03_prodromal/final_pyg_data/train_data.pt"
    dataset = torch.load(data_path, weights_only=False)
    
    in_features = dataset.x.shape[1]
    gat_encoder = GIMANSurvivalGAT(in_features=in_features, hidden_dim=128)
    model = NeuroFuzzyGIMAN(gat_encoder, num_classes=2, num_rules=32).to(device)
    checkpoint = project_root / "outputs/phase9_neuro_fuzzy/neuro_fuzzy_best.pth"
    
    if not checkpoint.exists():
        print("Phase 9 checkpoint not found.")
        return
    
    model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=False))
    
    # Calculate rule importance
    importance = (model.output_layer.weight[1] - model.output_layer.weight[0]).detach().cpu().numpy()
    rule_ids = np.arange(len(importance))
    
    # Create interactive bar chart
    fig = go.Figure()
    
    colors = ['red' if imp > 0 else 'blue' for imp in importance]
    
    fig.add_trace(go.Bar(
        x=rule_ids,
        y=importance,
        marker_color=colors,
        text=[f'{imp:.3f}' for imp in importance],
        textposition='outside',
        hovertemplate='<b>Rule %{x}</b><br>Importance: %{y:.4f}<extra></extra>'
    ))
    
    fig.update_layout(
        title='Fuzzy Rule Importance (Interactive)',
        xaxis_title='Rule ID',
        yaxis_title='Importance Score (Positive = Predicts SAA+)',
        template='plotly_white',
        width=1200,
        height=600,
        hovermode='x'
    )
    
    fig.add_hline(y=0, line_dash="dash", line_color="black")
    
    output_path = interactive_output / "rule_importance_interactive.html"
    fig.write_html(str(output_path))
    print(f"✅ Saved: {output_path}")

def main():
    print("="*80)
    print("PHASE 11: INTERACTIVE VISUALIZATIONS (PLOTLY)")
    print("="*80)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data_path = project_root / "data/03_prodromal/final_pyg_data/train_data.pt"
    
    if not data_path.exists():
        print(f"Data not found at {data_path}")
        return
    
    dataset = torch.load(data_path, weights_only=False)
    
    create_interactive_survival_curves(dataset, device)
    create_fuzzy_rule_explorer(dataset, device)
    create_feature_importance_dashboard()
    
    print("\n" + "="*80)
    print("INTERACTIVE VISUALIZATIONS COMPLETE")
    print("="*80)
    print(f"\nAll interactive HTML files saved to: {interactive_output}")

if __name__ == "__main__":
    main()
